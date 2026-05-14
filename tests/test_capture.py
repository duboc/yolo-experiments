"""Tests for ThreadedGrabber and FpsMeter."""

import threading
import time

import numpy as np
import pytest

from soccer_ball.capture import FpsMeter, ThreadedGrabber


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
