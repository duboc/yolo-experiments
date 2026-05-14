"""Pose helpers for kickup attribution.

Bridges Ultralytics keypoint output (COCO 17-keypoint format) to the
``BodyPart`` enum used by the kickup counter. Pure functions over numpy
arrays so tests don't need to construct fake YOLO result objects.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class BodyPart(str, Enum):
    FOOT = "foot"
    KNEE = "knee"
    HEAD = "head"


# COCO 17-keypoint indices (https://docs.ultralytics.com/datasets/pose/coco/):
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
# 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow, 9 l_wrist, 10 r_wrist,
# 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
KEYPOINT_GROUPS: dict[BodyPart, tuple[int, ...]] = {
    BodyPart.FOOT: (15, 16),
    BodyPart.KNEE: (13, 14),
    BodyPart.HEAD: (0,),  # nose only — eyes/ears are noisy and not impact-points
}


def extract_body_keypoints(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    conf_threshold: float = 0.3,
) -> dict[BodyPart, list[tuple[int, int]]]:
    """Group COCO keypoints into FOOT / KNEE / HEAD lists, filtered by confidence.

    Args:
        keypoints_xy:   shape (N_persons, 17, 2)
        keypoints_conf: shape (N_persons, 17)
    """
    out: dict[BodyPart, list[tuple[int, int]]] = {bp: [] for bp in BodyPart}
    if keypoints_xy.shape[0] == 0:
        return out
    for person_xy, person_conf in zip(keypoints_xy, keypoints_conf):
        for body_part, indices in KEYPOINT_GROUPS.items():
            for idx in indices:
                if person_conf[idx] >= conf_threshold:
                    x, y = person_xy[idx]
                    out[body_part].append((int(x), int(y)))
    return out


def nearest_body_part(
    centroid: tuple[int, int],
    parts: dict[BodyPart, list[tuple[int, int]]],
    max_distance_px: float,
) -> BodyPart | None:
    """Return the body part whose closest keypoint is nearest to ``centroid``,
    provided that distance is within ``max_distance_px``.
    """
    cx, cy = centroid
    best_part: BodyPart | None = None
    best_dist = float("inf")
    for body_part, points in parts.items():
        for px, py in points:
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_part = body_part
    if best_part is None or best_dist > max_distance_px:
        return None
    return best_part


def decide_bounce_credit(
    parts: dict[BodyPart, list[tuple[int, int]]],
    centroid: tuple[int, int],
    proximity_px: float,
    fallback: BodyPart = BodyPart.FOOT,
) -> BodyPart | None:
    """Soft pose-gate: decide which body part (if any) gets credit for a bounce.

    - If pose detected NO body parts at all this frame, credit ``fallback``
      anyway. Pose-model failure shouldn't penalize a real bounce.
    - If pose detected someone, return the nearest part within proximity.
    - Otherwise (someone visible but too far from the ball), return None
      so the caller can reject the bounce.
    """
    if not any(pts for pts in parts.values()):
        return fallback
    return nearest_body_part(centroid, parts, proximity_px)
