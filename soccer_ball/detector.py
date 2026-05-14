"""Pure helpers for the soccer-ball detector.

Kept free of OpenCV window / capture I/O so the test suite can import this
module without a webcam, display, or model weights present.
"""

from __future__ import annotations

import cv2
import numpy as np

# COCO class index for "sports ball". Soccer balls are labelled under this
# umbrella class along with basketballs, baseballs, etc.
COCO_SPORTS_BALL_ID = 32


def filter_sports_ball(
    cls: np.ndarray,
    conf: np.ndarray,
    xyxy: np.ndarray,
    ball_class_id: int = COCO_SPORTS_BALL_ID,
    conf_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only detections whose class matches and whose confidence clears the threshold.

    Returns (xyxy_kept, conf_kept). Empty inputs return correctly-shaped
    empty arrays so downstream code can treat the result uniformly.
    """
    if cls.size == 0:
        return np.zeros((0, 4), dtype=xyxy.dtype), np.zeros((0,), dtype=conf.dtype)

    mask = (cls == ball_class_id) & (conf >= conf_threshold)
    return xyxy[mask], conf[mask]


def format_label(conf: float) -> str:
    return f"soccer ball {conf:.2f}"


_BOX_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 0, 0)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1


def annotate_frame(
    frame: np.ndarray,
    xyxy: np.ndarray,
    confs: np.ndarray,
    show_label: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` with green boxes (and optional labels) per detection."""
    out = frame.copy()
    for (x1, y1, x2, y2), conf in zip(xyxy, confs):
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, _BOX_COLOR, _BOX_THICKNESS)

        if not show_label:
            continue

        label = format_label(float(conf))
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        bg_top_left = (p1[0], max(0, p1[1] - th - baseline - 2))
        bg_bottom_right = (p1[0] + tw + 2, p1[1])
        cv2.rectangle(out, bg_top_left, bg_bottom_right, _BOX_COLOR, cv2.FILLED)
        cv2.putText(
            out,
            label,
            (p1[0] + 1, p1[1] - baseline),
            _FONT,
            _FONT_SCALE,
            _TEXT_COLOR,
            _FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return out


_FPS_COLOR = (0, 255, 255)
_FPS_BG = (0, 0, 0)


def overlay_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    out = frame.copy()
    text = f"FPS: {fps:5.1f}"
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.6, 2)
    cv2.rectangle(out, (5, 5), (5 + tw + 6, 5 + th + baseline + 4), _FPS_BG, cv2.FILLED)
    cv2.putText(out, text, (8, 5 + th + 2), _FONT, 0.6, _FPS_COLOR, 2, cv2.LINE_AA)
    return out
