"""Camera enumeration and interactive selection.

The probe and picker are split so the picker can be unit-tested without ever
touching real OpenCV captures, and so callers can show a probed list once and
re-use it (e.g. CLI listing without prompting).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import cv2


class _Capture(Protocol):
    def isOpened(self) -> bool: ...
    def get(self, prop: int) -> float: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[int], _Capture]


@dataclass(frozen=True)
class CameraInfo:
    index: int
    width: int
    height: int
    fps: float


def _default_factory(index: int) -> _Capture:
    """Open the camera with the macOS-native AVFoundation backend when present."""
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return cv2.VideoCapture(index, backend)


def probe_cameras(
    max_index: int = 5,
    capture_factory: CaptureFactory = _default_factory,
) -> list[CameraInfo]:
    """Open camera indices ``[0, max_index)`` and return info for those that opened."""
    cams: list[CameraInfo] = []
    for i in range(max_index):
        cap = capture_factory(i)
        try:
            if not cap.isOpened():
                continue
            cams.append(
                CameraInfo(
                    index=i,
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=float(cap.get(cv2.CAP_PROP_FPS)),
                )
            )
        finally:
            cap.release()
    return cams


def format_camera_menu(cams: list[CameraInfo]) -> str:
    lines = ["Available cameras:"]
    for menu_pos, c in enumerate(cams):
        lines.append(
            f"  [{menu_pos}] index={c.index}  {c.width}x{c.height} @ {c.fps:.0f}fps"
        )
    return "\n".join(lines)


def pick_camera_interactive(
    cams: list[CameraInfo],
    input_fn: Callable[[str], str] = input,
) -> int:
    """Prompt the user to pick from ``cams`` and return the chosen camera index.

    Re-prompts on invalid input. Translates EOF to KeyboardInterrupt so the
    caller can treat both ctrl+C and ctrl+D as a clean cancel.
    """
    if not cams:
        raise RuntimeError("No cameras available to pick from.")

    print(format_camera_menu(cams))
    while True:
        try:
            raw = input_fn(f"Choose camera [0-{len(cams) - 1}]: ").strip()
        except EOFError as exc:
            raise KeyboardInterrupt from exc

        try:
            choice = int(raw)
        except ValueError:
            print(f"  '{raw}' is not a number, try again.")
            continue

        if not 0 <= choice < len(cams):
            print(f"  {choice} is out of range, try again.")
            continue

        return cams[choice].index
