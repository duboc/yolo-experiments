"""Tests for the pure trackbar encode/decode helpers.

The TrackbarPanel cv2 wrapper itself is not unit-tested — its only logic is
calling cv2.createTrackbar / cv2.getTrackbarPos, which require a real window.
The pure helpers cover all the value translation logic.
"""

import pytest

from soccer_ball.settings import RuntimeSettings
from soccer_ball.trackbars import (
    decode_trackbars,
    encode_settings,
)


class TestDecodeIndividualValues:
    def test_decode_conf(self):
        from soccer_ball.trackbars import decode_conf
        assert decode_conf(0) == 0.0
        assert decode_conf(50) == 0.5
        assert decode_conf(100) == 1.0

    def test_decode_iou(self):
        from soccer_ball.trackbars import decode_iou
        assert decode_iou(70) == 0.7

    def test_decode_imgsz_aligns_to_32(self):
        from soccer_ball.trackbars import decode_imgsz
        assert decode_imgsz(20) == 640
        assert decode_imgsz(10) == 320  # min
        assert decode_imgsz(40) == 1280  # max

    def test_decode_bool_trackbar(self):
        from soccer_ball.trackbars import decode_bool
        assert decode_bool(0) is False
        assert decode_bool(1) is True


class TestEncodeSettings:
    def test_encodes_defaults(self):
        s = RuntimeSettings()
        out = encode_settings(s)
        assert out["conf"] == 25
        assert out["iou"] == 70
        assert out["max_det"] == 300
        assert out["imgsz"] == 20  # 640 / 32
        assert out["ball_class"] == 32
        assert out["agnostic_nms"] == 0
        assert out["show_fps"] == 1
        assert out["show_label"] == 1
        assert out["show_settings"] == 1


class TestRoundTrip:
    def test_encode_then_decode_preserves_settings(self):
        original = RuntimeSettings(
            conf=0.4, iou=0.55, max_det=150, imgsz=960,
            ball_class=0, agnostic_nms=True, show_fps=False, show_label=False,
            show_settings=False,
        )
        restored = decode_trackbars(encode_settings(original))
        assert restored == original

    def test_decode_clamps_imgsz(self):
        # Trackbar position below 10 should clamp to min imgsz
        encoded = encode_settings(RuntimeSettings())
        encoded["imgsz"] = 5
        s = decode_trackbars(encoded)
        assert s.imgsz == 320
