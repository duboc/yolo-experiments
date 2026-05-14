"""Tests for pose helpers — pure functions over keypoint arrays."""

import numpy as np
import pytest

from soccer_ball.pose import (
    KEYPOINT_GROUPS,
    BodyPart,
    decide_bounce_credit,
    extract_body_keypoints,
    nearest_body_part,
)


# COCO keypoint indices used inline below for clarity:
# 0=nose, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle


def _zeros_kp(num_persons: int = 1) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((num_persons, 17, 2), dtype=np.float32),
        np.zeros((num_persons, 17), dtype=np.float32),
    )


class TestExtractBodyKeypoints:
    def test_returns_empty_groups_for_no_persons(self):
        xy = np.zeros((0, 17, 2), dtype=np.float32)
        conf = np.zeros((0, 17), dtype=np.float32)
        out = extract_body_keypoints(xy, conf)
        assert out == {BodyPart.FOOT: [], BodyPart.KNEE: [], BodyPart.HEAD: []}

    def test_filters_by_confidence(self):
        xy, conf = _zeros_kp(1)
        # Set a high-conf left ankle (15) and a low-conf right ankle (16)
        xy[0, 15] = (100, 200)
        xy[0, 16] = (110, 210)
        conf[0, 15] = 0.9
        conf[0, 16] = 0.1
        out = extract_body_keypoints(xy, conf, conf_threshold=0.3)
        assert out[BodyPart.FOOT] == [(100, 200)]

    def test_groups_left_and_right(self):
        xy, conf = _zeros_kp(1)
        xy[0, 13] = (10, 20)
        xy[0, 14] = (30, 40)
        conf[0, 13] = 0.8
        conf[0, 14] = 0.7
        out = extract_body_keypoints(xy, conf)
        assert sorted(out[BodyPart.KNEE]) == [(10, 20), (30, 40)]

    def test_extracts_head(self):
        xy, conf = _zeros_kp(1)
        xy[0, 0] = (500, 50)
        conf[0, 0] = 0.95
        out = extract_body_keypoints(xy, conf)
        assert out[BodyPart.HEAD] == [(500, 50)]

    def test_handles_multiple_persons(self):
        xy, conf = _zeros_kp(2)
        # Person 1: left ankle (15)
        xy[0, 15] = (100, 200)
        conf[0, 15] = 0.9
        # Person 2: right ankle (16)
        xy[1, 16] = (300, 400)
        conf[1, 16] = 0.85
        out = extract_body_keypoints(xy, conf)
        assert sorted(out[BodyPart.FOOT]) == [(100, 200), (300, 400)]


class TestNearestBodyPart:
    def _parts(self) -> dict[BodyPart, list[tuple[int, int]]]:
        return {
            BodyPart.FOOT: [(100, 500)],
            BodyPart.KNEE: [(100, 400)],
            BodyPart.HEAD: [(100, 100)],
        }

    def test_returns_closest_within_radius(self):
        # Ball is at (110, 510); nearest is FOOT at distance ~14
        assert nearest_body_part((110, 510), self._parts(), max_distance_px=50) == BodyPart.FOOT

    def test_returns_none_when_all_outside_radius(self):
        # Tiny radius — nothing close.
        assert nearest_body_part((110, 510), self._parts(), max_distance_px=5) is None

    def test_picks_closer_when_two_in_range(self):
        # Ball halfway between knee (100, 400) and foot (100, 500), but slightly
        # closer to foot.
        assert nearest_body_part((100, 460), self._parts(), max_distance_px=200) == BodyPart.FOOT
        assert nearest_body_part((100, 440), self._parts(), max_distance_px=200) == BodyPart.KNEE

    def test_empty_parts_returns_none(self):
        assert nearest_body_part((100, 100), {BodyPart.FOOT: [], BodyPart.KNEE: [], BodyPart.HEAD: []}, 50) is None


class TestKeypointGroups:
    def test_includes_expected_indices(self):
        assert KEYPOINT_GROUPS[BodyPart.FOOT] == (15, 16)
        assert KEYPOINT_GROUPS[BodyPart.KNEE] == (13, 14)
        assert KEYPOINT_GROUPS[BodyPart.HEAD] == (0,)


class TestDecideBounceCredit:
    """The soft pose-gate logic: don't penalize bounces when pose finds nothing."""

    def test_empty_parts_returns_fallback(self):
        empty = {BodyPart.FOOT: [], BodyPart.KNEE: [], BodyPart.HEAD: []}
        assert decide_bounce_credit(empty, (100, 100), 80) == BodyPart.FOOT

    def test_empty_parts_honors_custom_fallback(self):
        empty = {BodyPart.FOOT: [], BodyPart.KNEE: [], BodyPart.HEAD: []}
        assert decide_bounce_credit(empty, (100, 100), 80, fallback=BodyPart.HEAD) == BodyPart.HEAD

    def test_ball_near_part_returns_that_part(self):
        parts = {
            BodyPart.FOOT: [(100, 500)],
            BodyPart.KNEE: [],
            BodyPart.HEAD: [],
        }
        assert decide_bounce_credit(parts, (110, 510), 80) == BodyPart.FOOT

    def test_ball_far_from_all_returns_none(self):
        parts = {
            BodyPart.FOOT: [(100, 500)],
            BodyPart.KNEE: [(100, 400)],
            BodyPart.HEAD: [(100, 100)],
        }
        # Ball at (1000, 1000) — far from every keypoint with radius 80
        assert decide_bounce_credit(parts, (1000, 1000), 80) is None

    def test_picks_closest_part(self):
        parts = {
            BodyPart.FOOT: [(100, 500)],
            BodyPart.KNEE: [(100, 400)],
            BodyPart.HEAD: [(100, 100)],
        }
        # Ball closer to knee (100, 400) than to anything else
        assert decide_bounce_credit(parts, (100, 410), 200) == BodyPart.KNEE
