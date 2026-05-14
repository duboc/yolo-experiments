"""Unit tests for soccer_ball.detector pure helpers.

These tests must run without a webcam, without GPU, and without downloading
any model weights — they only exercise NumPy/OpenCV transformations.
"""

import numpy as np
import pytest

from soccer_ball.detector import (
    annotate_frame,
    filter_by_area,
    filter_sports_ball,
    format_label,
    overlay_fps,
    overlay_kickup,
    overlay_settings,
    overlay_trail,
)


class TestFilterSportsBall:
    def test_returns_only_matching_class_above_threshold(self):
        cls = np.array([32, 0, 32, 32])
        conf = np.array([0.9, 0.95, 0.5, 0.1])
        xyxy = np.array(
            [
                [10, 10, 50, 50],
                [60, 60, 100, 100],
                [110, 110, 150, 150],
                [160, 160, 200, 200],
            ],
            dtype=np.float32,
        )

        kept_xyxy, kept_conf = filter_sports_ball(
            cls, conf, xyxy, ball_class_id=32, conf_threshold=0.25
        )

        np.testing.assert_array_equal(
            kept_xyxy,
            np.array([[10, 10, 50, 50], [110, 110, 150, 150]], dtype=np.float32),
        )
        np.testing.assert_allclose(kept_conf, np.array([0.9, 0.5]))

    def test_no_matches_returns_empty(self):
        cls = np.array([0, 1, 2])
        conf = np.array([0.9, 0.9, 0.9])
        xyxy = np.zeros((3, 4), dtype=np.float32)

        kept_xyxy, kept_conf = filter_sports_ball(cls, conf, xyxy)

        assert kept_xyxy.shape == (0, 4)
        assert kept_conf.shape == (0,)

    def test_empty_input_returns_empty(self):
        cls = np.array([], dtype=np.int64)
        conf = np.array([], dtype=np.float32)
        xyxy = np.zeros((0, 4), dtype=np.float32)

        kept_xyxy, kept_conf = filter_sports_ball(cls, conf, xyxy)

        assert kept_xyxy.shape == (0, 4)
        assert kept_conf.shape == (0,)

    def test_sub_threshold_filtered_out(self):
        cls = np.array([32, 32])
        conf = np.array([0.24, 0.26])
        xyxy = np.array(
            [[0, 0, 1, 1], [2, 2, 3, 3]], dtype=np.float32
        )

        kept_xyxy, kept_conf = filter_sports_ball(
            cls, conf, xyxy, conf_threshold=0.25
        )

        np.testing.assert_array_equal(kept_xyxy, np.array([[2, 2, 3, 3]], dtype=np.float32))
        np.testing.assert_allclose(kept_conf, np.array([0.26]))


class TestFormatLabel:
    def test_rounds_to_two_decimals(self):
        assert format_label(0.8732) == "soccer ball 0.87"

    def test_handles_one(self):
        assert format_label(1.0) == "soccer ball 1.00"

    def test_handles_low(self):
        assert format_label(0.2549) == "soccer ball 0.25"


class TestAnnotateFrame:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_preserves_shape_and_dtype(self):
        frame = self._blank()
        xyxy = np.array([[100, 100, 200, 200]], dtype=np.float32)
        confs = np.array([0.9])

        out = annotate_frame(frame, xyxy, confs)

        assert out.shape == frame.shape
        assert out.dtype == frame.dtype

    def test_does_not_mutate_input(self):
        frame = self._blank()
        original = frame.copy()
        xyxy = np.array([[100, 100, 200, 200]], dtype=np.float32)
        confs = np.array([0.9])

        annotate_frame(frame, xyxy, confs)

        np.testing.assert_array_equal(frame, original)

    def test_actually_draws_when_detection_present(self):
        frame = self._blank()
        xyxy = np.array([[100, 100, 200, 200]], dtype=np.float32)
        confs = np.array([0.9])

        out = annotate_frame(frame, xyxy, confs)

        # Output must differ from blank input — something was drawn.
        assert not np.array_equal(out, frame)

    def test_empty_detections_returns_unchanged_copy(self):
        frame = self._blank()
        xyxy = np.zeros((0, 4), dtype=np.float32)
        confs = np.zeros((0,), dtype=np.float32)

        out = annotate_frame(frame, xyxy, confs)

        np.testing.assert_array_equal(out, frame)
        assert out is not frame  # still a copy

    def test_show_label_false_skips_text(self):
        """With show_label=False, the box-only output must differ from the labelled one."""
        frame = self._blank()
        xyxy = np.array([[100, 100, 200, 200]], dtype=np.float32)
        confs = np.array([0.9])

        with_label = annotate_frame(frame, xyxy, confs, show_label=True)
        without_label = annotate_frame(frame, xyxy, confs, show_label=False)

        assert not np.array_equal(with_label, without_label)
        # Box still drawn either way → both differ from the blank input.
        assert not np.array_equal(without_label, frame)


class TestOverlayFps:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_does_not_mutate_input(self):
        frame = self._blank()
        original = frame.copy()
        overlay_fps(frame, 42.0)
        np.testing.assert_array_equal(frame, original)

    def test_draws_something(self):
        frame = self._blank()
        out = overlay_fps(frame, 42.0)
        assert not np.array_equal(out, frame)
        assert out.shape == frame.shape and out.dtype == frame.dtype

    def test_zero_fps_still_renders(self):
        frame = self._blank()
        out = overlay_fps(frame, 0.0)
        assert not np.array_equal(out, frame)


class TestOverlaySettings:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_does_not_mutate_input(self):
        frame = self._blank()
        original = frame.copy()
        overlay_settings(frame, ["model: yolo26l.pt", "device: mps"])
        np.testing.assert_array_equal(frame, original)

    def test_draws_lines(self):
        frame = self._blank()
        out = overlay_settings(frame, ["conf: 0.25", "iou: 0.70"])
        assert out.shape == frame.shape and out.dtype == frame.dtype
        assert not np.array_equal(out, frame)

    def test_empty_lines_returns_unchanged_copy(self):
        frame = self._blank()
        out = overlay_settings(frame, [])
        np.testing.assert_array_equal(out, frame)
        assert out is not frame

    def test_renders_in_top_right_quadrant(self):
        frame = self._blank()
        out = overlay_settings(frame, ["x"])
        # Top-left quadrant must be untouched (FPS overlay's territory).
        np.testing.assert_array_equal(
            out[: frame.shape[0] // 2, : frame.shape[1] // 2],
            frame[: frame.shape[0] // 2, : frame.shape[1] // 2],
        )


class TestFilterByArea:
    # 1080p frame for predictable percentages.
    _FRAME_SHAPE = (1080, 1920, 3)
    _FRAME_AREA = 1080 * 1920

    def test_drops_below_threshold(self):
        # 100x100 box is ~0.48% of 1080p; threshold 1% should drop it.
        xyxy = np.array([[0, 0, 100, 100]], dtype=np.float32)
        conf = np.array([0.9])
        out_xyxy, out_conf = filter_by_area(xyxy, conf, 1.0, self._FRAME_SHAPE)
        assert out_xyxy.shape == (0, 4)
        assert out_conf.shape == (0,)

    def test_keeps_above_threshold(self):
        # 200x200 box is ~1.93% of 1080p; threshold 1% keeps it.
        xyxy = np.array([[0, 0, 200, 200]], dtype=np.float32)
        conf = np.array([0.9])
        out_xyxy, out_conf = filter_by_area(xyxy, conf, 1.0, self._FRAME_SHAPE)
        assert out_xyxy.shape == (1, 4)
        np.testing.assert_allclose(out_conf, [0.9])

    def test_zero_threshold_passes_everything(self):
        xyxy = np.array([[0, 0, 5, 5], [0, 0, 50, 50]], dtype=np.float32)
        conf = np.array([0.5, 0.6])
        out_xyxy, out_conf = filter_by_area(xyxy, conf, 0.0, self._FRAME_SHAPE)
        assert out_xyxy.shape == (2, 4)

    def test_empty_input(self):
        out_xyxy, out_conf = filter_by_area(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            1.0, self._FRAME_SHAPE,
        )
        assert out_xyxy.shape == (0, 4)
        assert out_conf.shape == (0,)

    def test_mixed_results(self):
        xyxy = np.array(
            [[0, 0, 100, 100], [0, 0, 300, 300], [0, 0, 50, 50]],
            dtype=np.float32,
        )
        conf = np.array([0.9, 0.8, 0.7])
        out_xyxy, out_conf = filter_by_area(xyxy, conf, 1.0, self._FRAME_SHAPE)
        # Only the 300x300 box (~4.3%) survives a 1% threshold.
        assert out_xyxy.shape == (1, 4)
        np.testing.assert_allclose(out_conf, [0.8])


class TestOverlayTrail:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_does_not_mutate(self):
        frame = self._blank()
        original = frame.copy()
        overlay_trail(frame, [(100, 200), (110, 210)])
        np.testing.assert_array_equal(frame, original)

    def test_empty_returns_unchanged_copy(self):
        frame = self._blank()
        out = overlay_trail(frame, [])
        np.testing.assert_array_equal(out, frame)
        assert out is not frame

    def test_draws_when_points(self):
        frame = self._blank()
        out = overlay_trail(frame, [(100, 200), (200, 300), (300, 400)])
        assert not np.array_equal(out, frame)
        assert out.shape == frame.shape and out.dtype == frame.dtype


class TestOverlayKickup:
    def _blank(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def test_does_not_mutate(self):
        frame = self._blank()
        original = frame.copy()
        overlay_kickup(frame, count=3, just_kicked=False)
        np.testing.assert_array_equal(frame, original)

    def test_draws_count(self):
        frame = self._blank()
        out = overlay_kickup(frame, count=7, just_kicked=False)
        assert not np.array_equal(out, frame)

    def test_flash_color_differs_when_just_kicked(self):
        frame = self._blank()
        normal = overlay_kickup(frame, count=5, just_kicked=False)
        flashed = overlay_kickup(frame, count=5, just_kicked=True)
        # The two outputs must visibly differ.
        assert not np.array_equal(normal, flashed)
