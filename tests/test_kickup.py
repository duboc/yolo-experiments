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


class TestKickupAccelerationGate:
    def test_slow_rollover_does_not_count(self):
        # Very gentle bounce: small velocity, tiny acceleration. Must NOT count.
        c = KickupCounter(min_velocity=0.5, acceleration_threshold=20.0)
        # Slow fall then slow rise — small acceleration.
        ys = list(range(100, 200, 1)) + list(range(200, 100, -1))
        for y in ys:
            c.update(y=float(y))
        assert c.count == 0

    def test_sharp_impact_counts(self):
        # Ball is falling fast, then sharply reverses — large acceleration spike.
        c = KickupCounter(min_velocity=0.5, acceleration_threshold=2.0)
        # 30 frames of fast fall (v ≈ +20)
        for y in range(100, 700, 20):
            c.update(y=float(y))
        # Sharp reversal: now rising fast (v ≈ -20). Acceleration ≈ -40.
        for y in range(700, 100, -20):
            c.update(y=float(y))
        assert c.count == 1

    def test_acceleration_default_does_not_break_existing(self):
        # Without supplying acceleration_threshold, behavior matches the
        # original test_one_full_kickup: classic counter still works.
        c = KickupCounter()
        for y in range(100, 500, 10):
            c.update(y=float(y))
        for y in range(500, 100, -10):
            c.update(y=float(y))
        assert c.count == 1


class TestResolutionAwareVelocity:
    def test_frame_height_scales_threshold(self):
        # min_velocity_pct=0.01 → threshold = 0.01 * 1080 = 10.8 px/frame on 1080p.
        # A "kickup" with 5px/frame motion should NOT count on 1080p.
        c = KickupCounter(min_velocity=0.5, min_velocity_pct=0.01)
        for y in range(100, 200, 5):
            c.update(y=float(y), frame_height=1080)
        for y in range(200, 100, -5):
            c.update(y=float(y), frame_height=1080)
        assert c.count == 0

    def test_same_motion_counts_on_smaller_frame(self):
        # Same 5px/frame motion on 240p: threshold = 0.01 * 240 = 2.4 → counts.
        c = KickupCounter(min_velocity=0.5, min_velocity_pct=0.01)
        for y in range(100, 200, 5):
            c.update(y=float(y), frame_height=240)
        for y in range(200, 100, -5):
            c.update(y=float(y), frame_height=240)
        assert c.count == 1

    def test_frame_height_none_keeps_literal_threshold(self):
        # frame_height=None → use min_velocity literal as before.
        c = KickupCounter(min_velocity=1.0, min_velocity_pct=0.01)
        for y in range(100, 200, 5):
            c.update(y=float(y))  # no frame_height
        for y in range(200, 100, -5):
            c.update(y=float(y))
        assert c.count == 1


class TestKickupBodyPartCredit:
    def _trigger_one_kickup(self, c: KickupCounter) -> None:
        for y in range(100, 500, 10):
            c.update(y=float(y))
        for y in range(500, 100, -10):
            c.update(y=float(y))
            if c.just_kicked:
                return

    def test_starts_with_empty_per_part_counts(self):
        from soccer_ball.pose import BodyPart
        c = KickupCounter()
        assert c.counts_by_part == {BodyPart.FOOT: 0, BodyPart.KNEE: 0, BodyPart.HEAD: 0}
        assert c.last_part is None

    def test_credit_increments_per_part(self):
        from soccer_ball.pose import BodyPart
        c = KickupCounter()
        self._trigger_one_kickup(c)
        c.credit_last(BodyPart.FOOT)
        assert c.counts_by_part[BodyPart.FOOT] == 1
        assert c.counts_by_part[BodyPart.KNEE] == 0
        assert c.last_part == BodyPart.FOOT
        # Aggregate count untouched by credit.
        assert c.count == 1

    def test_credit_none_rejects_the_bounce(self):
        c = KickupCounter()
        self._trigger_one_kickup(c)
        assert c.count == 1
        c.credit_last(None)
        # Aggregate count and just_kicked are rolled back.
        assert c.count == 0
        assert c.just_kicked is False
        assert c.last_part is None

    def test_credit_resets_with_full_reset(self):
        from soccer_ball.pose import BodyPart
        c = KickupCounter()
        self._trigger_one_kickup(c)
        c.credit_last(BodyPart.KNEE)
        c.reset()
        assert c.counts_by_part == {BodyPart.FOOT: 0, BodyPart.KNEE: 0, BodyPart.HEAD: 0}
        assert c.last_part is None


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
