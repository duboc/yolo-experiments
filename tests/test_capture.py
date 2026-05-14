"""Tests for ThreadedGrabber and FpsMeter."""

import threading
import time

import numpy as np
import pytest

from soccer_ball.capture import (
    CAPTURE_PRESETS,
    CaptureConfig,
    FpsMeter,
    GrabberStats,
    ThreadedGrabber,
    apply_capture_config,
    merge_capture_config,
)


class _SequenceCapture:
    """Fake VideoCapture: returns a new monotonically-counted frame each read()."""

    def __init__(self, *, fail_after: int | None = None):
        self._counter = 0
        self._fail_after = fail_after
        self._lock = threading.Lock()
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        with self._lock:
            if self._fail_after is not None and self._counter >= self._fail_after:
                return False, None
            self._counter += 1
            # int32 so the daemon thread can spin past 255 without overflowing
            # the test fake — OpenCV-shape compatibility doesn't matter here.
            frame = np.full((4, 4, 3), self._counter, dtype=np.int32)
            return True, frame

    def release(self) -> None:
        self.released = True


def _wait_until(predicate, timeout: float = 1.0, interval: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class TestThreadedGrabber:
    def test_reads_latest_frame(self):
        cap = _SequenceCapture()
        with ThreadedGrabber(cap) as g:
            assert _wait_until(lambda: g.read() is not None), "grabber never produced a frame"
            first = g.read()
            assert _wait_until(lambda: g.read() is not None and g.read()[0, 0, 0] > first[0, 0, 0])
            second = g.read()
            assert second[0, 0, 0] >= first[0, 0, 0]

    def test_release_on_context_exit(self):
        cap = _SequenceCapture()
        with ThreadedGrabber(cap) as g:
            _wait_until(lambda: g.read() is not None)
        assert cap.released
        assert not g.is_alive()

    def test_stop_joins_thread(self):
        cap = _SequenceCapture()
        g = ThreadedGrabber(cap)
        g.start()
        _wait_until(lambda: g.read() is not None)
        g.stop()
        assert not g.is_alive()
        assert cap.released

    def test_handles_failed_reads_without_crashing(self):
        cap = _SequenceCapture(fail_after=3)
        with ThreadedGrabber(cap) as g:
            _wait_until(lambda: g.read() is not None)
            time.sleep(0.05)  # let it try a few failing reads
        # Just survives without raising; no further assertion needed.


class TestFpsMeter:
    def test_returns_zero_before_enough_samples(self):
        m = FpsMeter()
        assert m.tick() == 0.0  # first tick has no delta

    def test_returns_positive_after_ticks(self):
        m = FpsMeter()
        m.tick()
        time.sleep(0.02)
        m.tick()
        time.sleep(0.02)
        fps = m.tick()
        assert 5 < fps < 200  # very loose bounds, just sanity

    def test_value_property_matches_last_tick(self):
        m = FpsMeter()
        m.tick()
        time.sleep(0.01)
        last = m.tick()
        assert m.value == last


# ---------------------------------------------------------------------------
# CaptureConfig + apply_capture_config + presets
# ---------------------------------------------------------------------------


import cv2


class _FakeCap:
    """Minimal cv2.VideoCapture stand-in for capture-config tests."""

    def __init__(self, *, refuse_props: set[int] | None = None):
        self.calls: list[tuple[int, float]] = []
        self._refuse = refuse_props or set()

    def set(self, prop: int, value) -> bool:
        self.calls.append((prop, float(value)))
        return prop not in self._refuse


class TestCaptureConfigDefaults:
    def test_all_fields_default_to_none(self):
        # None means "don't touch the camera property" — the safest default.
        c = CaptureConfig()
        assert c.width is None and c.height is None and c.fps is None
        assert c.exposure is None and c.focus is None and c.wb_temp is None
        assert c.auto_exposure is None
        assert c.auto_focus is None
        assert c.auto_wb is None


class TestApplyCaptureConfig:
    def test_no_changes_when_all_defaults(self):
        cap = _FakeCap()
        result = apply_capture_config(cap, CaptureConfig())
        assert cap.calls == []
        assert result == {}

    def test_sets_resolution_and_fps(self):
        cap = _FakeCap()
        result = apply_capture_config(cap, CaptureConfig(width=1280, height=720, fps=60))
        # Three set() calls — width, height, fps. Order doesn't matter for the test.
        prop_to_value = {prop: value for prop, value in cap.calls}
        assert prop_to_value[cv2.CAP_PROP_FRAME_WIDTH] == 1280
        assert prop_to_value[cv2.CAP_PROP_FRAME_HEIGHT] == 720
        assert prop_to_value[cv2.CAP_PROP_FPS] == 60
        assert result["width"] is True
        assert result["height"] is True
        assert result["fps"] is True

    def test_failure_recorded_in_result(self):
        cap = _FakeCap(refuse_props={cv2.CAP_PROP_EXPOSURE})
        result = apply_capture_config(
            cap, CaptureConfig(auto_exposure=False, exposure=-7.0)
        )
        assert result["exposure"] is False
        assert result["auto_exposure"] is True  # the auto-mode set() succeeded

    def test_disabling_auto_exposure_sets_manual_mode(self):
        cap = _FakeCap()
        apply_capture_config(cap, CaptureConfig(auto_exposure=False, exposure=-7.0))
        # Manual mode value on macOS AVFoundation is 0.25; on V4L2 it's typically 1.
        # We assert SOMETHING was set on AUTO_EXPOSURE; concrete value is platform-dep.
        prop_to_value = {prop: value for prop, value in cap.calls}
        assert cv2.CAP_PROP_AUTO_EXPOSURE in prop_to_value
        assert prop_to_value[cv2.CAP_PROP_EXPOSURE] == -7.0

    def test_disabling_auto_focus_and_setting_focus(self):
        cap = _FakeCap()
        apply_capture_config(cap, CaptureConfig(auto_focus=False, focus=120.0))
        prop_to_value = {prop: value for prop, value in cap.calls}
        assert cv2.CAP_PROP_AUTOFOCUS in prop_to_value
        assert prop_to_value[cv2.CAP_PROP_AUTOFOCUS] == 0.0
        assert prop_to_value[cv2.CAP_PROP_FOCUS] == 120.0

    def test_disabling_auto_wb_and_setting_temp(self):
        cap = _FakeCap()
        apply_capture_config(cap, CaptureConfig(auto_wb=False, wb_temp=4500.0))
        prop_to_value = {prop: value for prop, value in cap.calls}
        assert cv2.CAP_PROP_AUTO_WB in prop_to_value
        assert prop_to_value[cv2.CAP_PROP_AUTO_WB] == 0.0
        assert prop_to_value[cv2.CAP_PROP_WB_TEMPERATURE] == 4500.0


class TestPresets:
    def test_low_balanced_high_exist(self):
        assert "low" in CAPTURE_PRESETS
        assert "balanced" in CAPTURE_PRESETS
        assert "high" in CAPTURE_PRESETS

    def test_low_is_640x480_60fps(self):
        cfg = CAPTURE_PRESETS["low"]
        assert cfg.width == 640 and cfg.height == 480 and cfg.fps == 60

    def test_balanced_is_720p_60fps(self):
        cfg = CAPTURE_PRESETS["balanced"]
        assert cfg.width == 1280 and cfg.height == 720 and cfg.fps == 60

    def test_high_is_1080p_30fps(self):
        cfg = CAPTURE_PRESETS["high"]
        assert cfg.width == 1920 and cfg.height == 1080 and cfg.fps == 30


class TestMergeCaptureConfig:
    def test_overlays_non_none_fields(self):
        base = CaptureConfig(width=1280, height=720, fps=60)
        override = CaptureConfig(width=1920)
        merged = merge_capture_config(base, override)
        assert merged.width == 1920
        assert merged.height == 720
        assert merged.fps == 60

    def test_bool_fields_overlay_when_explicitly_set(self):
        # auto_exposure defaults to None; merging another CaptureConfig with
        # auto_exposure=False should set it.
        base = CaptureConfig()
        override = CaptureConfig(auto_exposure=False, exposure=-7.0)
        merged = merge_capture_config(base, override)
        assert merged.auto_exposure is False
        assert merged.exposure == -7.0


class TestGrabberStats:
    def test_starts_empty(self):
        s = GrabberStats()
        assert s.captured == 0
        assert s.unique_reads == 0
        assert s.drop_rate == 0.0

    def test_capture_increments(self):
        s = GrabberStats()
        s.on_capture()
        s.on_capture()
        assert s.captured == 2

    def test_unique_read_only_when_new_frame(self):
        s = GrabberStats()
        s.on_capture()         # captured=1
        s.on_read()             # unique_reads=1
        s.on_read()             # same frame, no increment
        assert s.unique_reads == 1
        s.on_capture()         # captured=2
        s.on_read()             # unique_reads=2
        assert s.unique_reads == 2

    def test_drop_rate_when_loop_keeps_up(self):
        s = GrabberStats()
        for _ in range(10):
            s.on_capture()
            s.on_read()
        assert s.drop_rate == 0.0

    def test_drop_rate_when_loop_falls_behind(self):
        s = GrabberStats()
        # 10 captures, only 1 read — 9/10 dropped.
        for _ in range(10):
            s.on_capture()
        s.on_read()
        assert s.drop_rate == pytest.approx(0.9)

    def test_drop_rate_zero_when_no_captures(self):
        s = GrabberStats()
        s.on_read()
        assert s.drop_rate == 0.0


class TestThreadedGrabberStats:
    def test_grabber_exposes_stats(self):
        cap = _SequenceCapture()
        with ThreadedGrabber(cap) as g:
            assert _wait_until(lambda: g.read() is not None)
            time.sleep(0.05)
            for _ in range(3):
                g.read()
                time.sleep(0.005)
        assert g.stats.captured > 0
        assert g.stats.unique_reads > 0
        assert 0.0 <= g.stats.drop_rate <= 1.0

    def test_grabber_exposes_capture_fps(self):
        cap = _SequenceCapture()
        with ThreadedGrabber(cap) as g:
            assert _wait_until(lambda: g.capture_fps > 0, timeout=2.0)
