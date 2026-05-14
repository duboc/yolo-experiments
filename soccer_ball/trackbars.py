"""OpenCV trackbar panel for live-tunable runtime settings.

Pure encode/decode helpers translate between ``RuntimeSettings`` and the
integer values cv2 trackbars work with. The ``TrackbarPanel`` class is a
thin wrapper around cv2.createTrackbar / cv2.getTrackbarPos so the loop can
read the latest user-tweaked settings each frame.
"""

from __future__ import annotations

import cv2

from soccer_ball.settings import RuntimeSettings

# Trackbar names (also used as ids when reading positions).
_NAMES = ["conf", "iou", "max_det", "imgsz", "ball_class", "agnostic_nms", "show_fps", "show_label", "show_settings", "min_area_pct"]

_IMGSZ_STEP = 32
_IMGSZ_MIN = 320  # = 10 * 32
_IMGSZ_MAX = 1280  # = 40 * 32

_TRACKBAR_MAX = {
    "conf": 100,
    "iou": 100,
    "max_det": 300,
    "imgsz": 40,
    "ball_class": 79,  # COCO has 80 classes (0-79)
    "agnostic_nms": 1,
    "show_fps": 1,
    "show_label": 1,
    "show_settings": 1,
    "min_area_pct": 100,  # 0..100 → 0.0..10.0 percent (×10 step = 0.1%)
}


def decode_conf(pos: int) -> float:
    return pos / 100.0


def decode_iou(pos: int) -> float:
    return pos / 100.0


def decode_imgsz(pos: int) -> int:
    pos = max(_IMGSZ_MIN // _IMGSZ_STEP, min(_IMGSZ_MAX // _IMGSZ_STEP, pos))
    return pos * _IMGSZ_STEP


def decode_bool(pos: int) -> bool:
    return bool(pos)


def decode_min_area_pct(pos: int) -> float:
    return pos / 10.0


def encode_settings(s: RuntimeSettings) -> dict[str, int]:
    return {
        "conf": int(round(s.conf * 100)),
        "iou": int(round(s.iou * 100)),
        "max_det": int(s.max_det),
        "imgsz": int(s.imgsz // _IMGSZ_STEP),
        "ball_class": int(s.ball_class),
        "agnostic_nms": int(s.agnostic_nms),
        "show_fps": int(s.show_fps),
        "show_label": int(s.show_label),
        "show_settings": int(s.show_settings),
        "min_area_pct": int(round(s.min_area_pct * 10)),
    }


def decode_trackbars(positions: dict[str, int]) -> RuntimeSettings:
    return RuntimeSettings(
        conf=decode_conf(positions["conf"]),
        iou=decode_iou(positions["iou"]),
        max_det=int(positions["max_det"]),
        imgsz=decode_imgsz(positions["imgsz"]),
        ball_class=int(positions["ball_class"]),
        agnostic_nms=decode_bool(positions["agnostic_nms"]),
        show_fps=decode_bool(positions["show_fps"]),
        show_label=decode_bool(positions["show_label"]),
        show_settings=decode_bool(positions["show_settings"]),
        min_area_pct=decode_min_area_pct(positions["min_area_pct"]),
    )


class TrackbarPanel:
    """cv2 trackbar panel attached to a named window."""

    def __init__(self, window_name: str = "Settings"):
        self.window_name = window_name
        self._created = False

    def create(self, initial: RuntimeSettings) -> "TrackbarPanel":
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 320)
        positions = encode_settings(initial)
        for name in _NAMES:
            # The 4th arg is a no-op callback; we poll positions instead.
            cv2.createTrackbar(name, self.window_name, positions[name], _TRACKBAR_MAX[name], lambda _v: None)
        self._created = True
        return self

    def current(self) -> RuntimeSettings:
        if not self._created:
            raise RuntimeError("TrackbarPanel.create() must be called before current()")
        positions = {name: cv2.getTrackbarPos(name, self.window_name) for name in _NAMES}
        return decode_trackbars(positions)

    def close(self) -> None:
        if self._created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass
            self._created = False
