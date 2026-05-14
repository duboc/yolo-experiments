"""Tests for KickupCounter and MotionTrail.

The counter watches centroid Y values frame-by-frame; one full down→up
reversal increments the count by 1.
"""

import pytest

from soccer_ball.kickup import KickupCounter, MotionTrail


class TestKickupCounter:
    def test_constant_y_no_count(self):
        c = KickupCounter()
        for _ in range(50):
            c.update(y=300.0)
        assert c.count == 0

    def test_only_falling_no_count(self):
        c = KickupCounter()
        for y in range(100, 500, 5):
            c.update(y=float(y))
        assert c.count == 0

    def test_only_rising_no_count(self):
        c = KickupCounter()
        for y in range(500, 100, -5):
            c.update(y=float(y))
        assert c.count == 0

    def test_one_full_kickup(self):
        c = KickupCounter()
        # Fall (y increasing) then rise (y decreasing) — one bounce.
        for y in range(100, 500, 10):
            c.update(y=float(y))
        for y in range(500, 100, -10):
            c.update(y=float(y))
        assert c.count == 1

    def test_two_kickups(self):
        c = KickupCounter()
        for cycle in range(2):
            for y in range(100, 500, 10):
                c.update(y=float(y))
            for y in range(500, 100, -10):
                c.update(y=float(y))
        assert c.count == 2

    def test_micro_jitter_does_not_count(self):
        c = KickupCounter()
        # Tiny back-and-forth around 300; sub-threshold velocity.
        for i in range(200):
            c.update(y=300.0 + (0.1 if i % 2 == 0 else -0.1))
        assert c.count == 0

    def test_long_absence_resets(self):
        c = KickupCounter(absence_reset_frames=10)
        # One bounce
        for y in range(100, 500, 10):
            c.update(y=float(y))
        for y in range(500, 100, -10):
            c.update(y=float(y))
        assert c.count == 1

        # 11 frames of nothing
        for _ in range(11):
            c.update(y=None)

        assert c.count == 0

    def test_just_kicked_clears_after_one_frame(self):
        c = KickupCounter()
        # Establish FALLING state.
        for y in range(100, 500, 10):
            c.update(y=float(y))
        # Drive smoothed velocity strongly negative to fire the bounce edge.
        for y in [490, 470, 440, 400]:
            c.update(y=float(y))
            if c.just_kicked:
                break
        assert c.just_kicked is True
        c.update(y=350.0)  # next rising frame
        assert c.just_kicked is False

    def test_manual_reset(self):
        c = KickupCounter()
        for y in range(100, 500, 10):
            c.update(y=float(y))
        for y in range(500, 100, -10):
            c.update(y=float(y))
        assert c.count == 1
        c.reset()
        assert c.count == 0


class TestMotionTrail:
    def test_starts_empty(self):
        t = MotionTrail(max_len=10)
        assert list(t) == []

    def test_append_within_capacity(self):
        t = MotionTrail(max_len=5)
        t.append((10, 20))
        t.append((15, 25))
        assert list(t) == [(10, 20), (15, 25)]

    def test_evicts_oldest_when_full(self):
        t = MotionTrail(max_len=3)
        for i in range(5):
            t.append((i, i * 2))
        assert list(t) == [(2, 4), (3, 6), (4, 8)]

    def test_clear(self):
        t = MotionTrail(max_len=5)
        t.append((1, 1))
        t.append((2, 2))
        t.clear()
        assert list(t) == []

    def test_append_none_is_skipped(self):
        t = MotionTrail(max_len=5)
        t.append((10, 20))
        t.append(None)
        t.append((30, 40))
        assert list(t) == [(10, 20), (30, 40)]
