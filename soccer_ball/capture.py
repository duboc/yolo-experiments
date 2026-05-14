"""Threaded frame grabber and FPS meter.

The grabber runs ``cap.read()`` in a daemon thread and atomically swaps in
the latest frame. The main loop pulls the freshest frame at any time without
blocking, which avoids the lag pile-up that hits cv2.VideoCapture on macOS
(where CAP_PROP_BUFFERSIZE is silently ignored by the AVFoundation backend).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Protocol

import numpy as np


class _Capture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...


class ThreadedGrabber:
    """Wrap a VideoCapture-like object and continuously hold its latest frame."""

    def __init__(self, cap: _Capture):
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

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

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

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
