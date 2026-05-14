"""Threaded frame grabber and FPS meter.

The grabber runs ``cap.read()`` in a daemon thread and atomically swaps in
the latest frame. The main loop pulls the freshest frame at any time without
blocking, which avoids the lag pile-up that hits cv2.VideoCapture on macOS
(where CAP_PROP_BUFFERSIZE is silently ignored by the AVFoundation backend).
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, fields, replace
from typing import Protocol

import cv2
import numpy as np


class _Capture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


# ---------------------------------------------------------------------------
# CaptureConfig — declarative camera setup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureConfig:
    """Declarative camera setup. Every field is optional; None = leave default."""
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    exposure: float | None = None
    focus: float | None = None
    wb_temp: float | None = None
    auto_exposure: bool | None = None
    auto_focus: bool | None = None
    auto_wb: bool | None = None


# Manual exposure prop value differs by backend:
# - macOS AVFoundation: 0.25 (manual) / 0.75 (auto)
# - V4L2 (Linux): 1 (manual) / 3 (auto)
# Pick the right one at runtime.
_AUTO_EXPOSURE_MANUAL = 0.25 if sys.platform == "darwin" else 1.0
_AUTO_EXPOSURE_AUTO = 0.75 if sys.platform == "darwin" else 3.0


CAPTURE_PRESETS: dict[str, CaptureConfig] = {
    "low": CaptureConfig(width=640, height=480, fps=60),
    "balanced": CaptureConfig(width=1280, height=720, fps=60),
    "high": CaptureConfig(width=1920, height=1080, fps=30),
}


def merge_capture_config(base: CaptureConfig, override: CaptureConfig) -> CaptureConfig:
    """Overlay any non-None field from ``override`` onto ``base``."""
    overrides = {f.name: getattr(override, f.name)
                 for f in fields(CaptureConfig)
                 if getattr(override, f.name) is not None}
    return replace(base, **overrides) if overrides else base


def apply_capture_config(cap, config: CaptureConfig) -> dict[str, bool]:
    """Apply each non-None field to ``cap``; return a map of which ones took.

    Returns ``{field_name: success_bool}`` so the caller can log which
    properties the camera actually accepted (they're all best-effort).
    """
    result: dict[str, bool] = {}

    if config.width is not None:
        result["width"] = bool(cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width))
    if config.height is not None:
        result["height"] = bool(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height))
    if config.fps is not None:
        result["fps"] = bool(cap.set(cv2.CAP_PROP_FPS, config.fps))

    if config.auto_exposure is not None:
        value = _AUTO_EXPOSURE_AUTO if config.auto_exposure else _AUTO_EXPOSURE_MANUAL
        result["auto_exposure"] = bool(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value))
    if config.exposure is not None:
        result["exposure"] = bool(cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure))

    if config.auto_focus is not None:
        result["auto_focus"] = bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0 if config.auto_focus else 0.0))
    if config.focus is not None:
        result["focus"] = bool(cap.set(cv2.CAP_PROP_FOCUS, config.focus))

    if config.auto_wb is not None:
        result["auto_wb"] = bool(cap.set(cv2.CAP_PROP_AUTO_WB, 1.0 if config.auto_wb else 0.0))
    if config.wb_temp is not None:
        result["wb_temp"] = bool(cap.set(cv2.CAP_PROP_WB_TEMPERATURE, config.wb_temp))

    return result


# ---------------------------------------------------------------------------
# GrabberStats — drop-rate diagnostics
# ---------------------------------------------------------------------------


class GrabberStats:
    """Tracks how many camera frames were captured vs actually consumed."""

    def __init__(self):
        self.captured: int = 0
        self.unique_reads: int = 0
        self._last_read_id: int = 0

    def on_capture(self) -> None:
        self.captured += 1

    def on_read(self) -> None:
        if self.captured > self._last_read_id:
            self.unique_reads += 1
            self._last_read_id = self.captured

    @property
    def drop_rate(self) -> float:
        if self.captured == 0:
            return 0.0
        return (self.captured - self.unique_reads) / self.captured


class ThreadedGrabber:
    """Wrap a VideoCapture-like object and continuously hold its latest frame."""

    def __init__(self, cap: _Capture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.stats = GrabberStats()
        self._capture_fps = FpsMeter()

    def start(self) -> "ThreadedGrabber":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Camera dropped a frame — back off briefly, don't busy-loop.
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
                self.stats.on_capture()
                self._capture_fps.tick()

    def read(self) -> np.ndarray | None:
        with self._lock:
            self.stats.on_read()
            return self._frame

    @property
    def capture_fps(self) -> float:
        return self._capture_fps.value

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._cap.release()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "ThreadedGrabber":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


class FpsMeter:
    """Rolling-window frames-per-second meter.

    ``tick()`` records the current time and returns the FPS computed over the
    most recent samples. Returns 0.0 until at least two ticks have happened.
    """

    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)
        self._value = 0.0

    def tick(self) -> float:
        self._times.append(time.monotonic())
        if len(self._times) < 2:
            self._value = 0.0
        else:
            elapsed = self._times[-1] - self._times[0]
            self._value = (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
        return self._value

    @property
    def value(self) -> float:
        return self._value
