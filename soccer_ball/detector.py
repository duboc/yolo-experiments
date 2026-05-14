"""Pure helpers for the soccer-ball detector.

Kept free of OpenCV window / capture I/O so the test suite can import this
module without a webcam, display, or model weights present.
"""

from __future__ import annotations

import cv2
import numpy as np

from soccer_ball.pose import BodyPart

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


def filter_by_area(
    xyxy: np.ndarray,
    conf: np.ndarray,
    min_area_pct: float,
    frame_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Drop boxes whose pixel area is below ``min_area_pct`` of total frame area."""
    if xyxy.size == 0 or min_area_pct <= 0:
        return xyxy, conf
    h, w = frame_shape[:2]
    min_pixels = (min_area_pct / 100.0) * h * w
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    areas = widths * heights
    mask = areas >= min_pixels
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


def overlay_fps(
    frame: np.ndarray,
    fps: float,
    cam_fps: float | None = None,
    drop_rate: float | None = None,
) -> np.ndarray:
    out = frame.copy()
    text = f"FPS: {fps:5.1f}"
    if cam_fps is not None:
        text = f"{text}  (cam {cam_fps:4.0f}"
        if drop_rate is not None:
            text = f"{text}  drop {drop_rate * 100:3.0f}%"
        text = f"{text})"
    (tw, th), baseline = cv2.getTextSize(text, _FONT, 0.6, 2)
    cv2.rectangle(out, (5, 5), (5 + tw + 6, 5 + th + baseline + 4), _FPS_BG, cv2.FILLED)
    cv2.putText(out, text, (8, 5 + th + 2), _FONT, 0.6, _FPS_COLOR, 2, cv2.LINE_AA)
    return out


_SETTINGS_FONT_SCALE = 0.5
_SETTINGS_FONT_THICKNESS = 1
_SETTINGS_TEXT_COLOR = (255, 255, 255)
_SETTINGS_LINE_HEIGHT = 18
_SETTINGS_PAD = 8


def overlay_settings(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Render ``lines`` in a semi-transparent panel anchored to the top-right corner."""
    out = frame.copy()
    if not lines:
        return out

    sizes = [cv2.getTextSize(line, _FONT, _SETTINGS_FONT_SCALE, _SETTINGS_FONT_THICKNESS) for line in lines]
    max_w = max(w for (w, _h), _b in sizes)
    block_w = max_w + 2 * _SETTINGS_PAD
    block_h = _SETTINGS_LINE_HEIGHT * len(lines) + _SETTINGS_PAD

    h, w = out.shape[:2]
    x2 = w - 5
    x1 = max(0, x2 - block_w)
    y1 = 5
    y2 = min(h - 1, y1 + block_h)

    bg = out.copy()
    cv2.rectangle(bg, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(bg, 0.6, out, 0.4, 0, out)

    for i, line in enumerate(lines):
        baseline_y = y1 + _SETTINGS_PAD + _SETTINGS_LINE_HEIGHT * i + 12
        cv2.putText(
            out,
            line,
            (x1 + _SETTINGS_PAD, baseline_y),
            _FONT,
            _SETTINGS_FONT_SCALE,
            _SETTINGS_TEXT_COLOR,
            _SETTINGS_FONT_THICKNESS,
            cv2.LINE_AA,
        )
    return out


_TRAIL_COLOR = (0, 200, 255)


def overlay_trail(frame: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    """Draw a fading dot trail over ``frame`` from oldest to newest point."""
    out = frame.copy()
    if not points:
        return out
    n = len(points)
    for i, (x, y) in enumerate(points):
        # Older points smaller and dimmer; newest point brightest.
        fade = (i + 1) / n
        radius = max(2, int(2 + 4 * fade))
        color = tuple(int(c * fade) for c in _TRAIL_COLOR)
        cv2.circle(out, (int(x), int(y)), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return out


_KICKUP_COLOR = (255, 255, 255)
_KICKUP_FLASH_COLOR = (0, 255, 0)
_KICKUP_BG = (0, 0, 0)


def overlay_kickup(
    frame: np.ndarray,
    count: int,
    just_kicked: bool,
    part: BodyPart | None = None,
) -> np.ndarray:
    """Render a center-top KICKUPS counter; flashes green on the bounce frame.

    When ``part`` is provided and ``just_kicked`` is True, the flash text
    appends the body part, e.g. "KICKUPS: 7 (foot)".
    """
    out = frame.copy()
    text = f"KICKUPS: {count}"
    if just_kicked and part is not None:
        text = f"{text} ({part.value})"
    color = _KICKUP_FLASH_COLOR if just_kicked else _KICKUP_COLOR
    scale = 1.2 if just_kicked else 1.0
    thickness = 3
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)

    h, w = out.shape[:2]
    x = (w - tw) // 2
    y = th + 20

    pad = 10
    bg = out.copy()
    cv2.rectangle(bg, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), _KICKUP_BG, cv2.FILLED)
    cv2.addWeighted(bg, 0.55, out, 0.45, 0, out)
    cv2.putText(out, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)
    return out


_POSE_COLORS = {
    "foot": (255, 255, 0),    # cyan
    "knee": (0, 255, 255),    # yellow
    "head": (255, 0, 255),    # magenta
}


_DEBUG_BG = (0, 0, 0)
_DEBUG_TEXT = (200, 255, 200)
_DEBUG_FONT_SCALE = 0.5
_DEBUG_LINE_HEIGHT = 18
_DEBUG_PAD = 8


def overlay_debug(frame: np.ndarray, info: dict[str, str]) -> np.ndarray:
    """Render a small bottom-left debug panel showing kickup state-machine internals."""
    out = frame.copy()
    if not info:
        return out
    lines = [f"{k:<8} {v}" for k, v in info.items()]
    sizes = [cv2.getTextSize(l, _FONT, _DEBUG_FONT_SCALE, 1) for l in lines]
    max_w = max(w for (w, _h), _b in sizes)
    block_w = max_w + 2 * _DEBUG_PAD
    block_h = _DEBUG_LINE_HEIGHT * len(lines) + _DEBUG_PAD

    h = out.shape[0]
    x1 = 5
    y2 = h - 5
    y1 = max(0, y2 - block_h)
    x2 = x1 + block_w

    bg = out.copy()
    cv2.rectangle(bg, (x1, y1), (x2, y2), _DEBUG_BG, cv2.FILLED)
    cv2.addWeighted(bg, 0.6, out, 0.4, 0, out)

    for i, line in enumerate(lines):
        baseline_y = y1 + _DEBUG_PAD + _DEBUG_LINE_HEIGHT * i + 12
        cv2.putText(out, line, (x1 + _DEBUG_PAD, baseline_y),
                    _FONT, _DEBUG_FONT_SCALE, _DEBUG_TEXT, 1, cv2.LINE_AA)
    return out


def overlay_pose(
    frame: np.ndarray,
    parts: dict[BodyPart, list[tuple[int, int]]],
    proximity_px: int,
) -> np.ndarray:
    """Draw colored dots per body part with an outlined proximity radius."""
    out = frame.copy()
    if not any(parts.values()):
        return out
    for part, points in parts.items():
        color = _POSE_COLORS.get(part.value, (200, 200, 200))
        for x, y in points:
            cv2.circle(out, (int(x), int(y)), 5, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), int(proximity_px), color, thickness=1, lineType=cv2.LINE_AA)
    return out
