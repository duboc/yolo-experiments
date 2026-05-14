"""Tests for SingleBallTracker.

The tracker consumes per-frame (ids, xyxy) from `boxes.id` / `boxes.xyxy`,
locks onto a single track ID, and follows it across frames so the kickup
counter sees one continuous trajectory instead of "biggest each frame".
"""

import numpy as np
import pytest

from soccer_ball.tracking import SingleBallTracker


def _xyxy(*boxes):
    return np.array(boxes, dtype=np.float32)


class TestNoIdsFallback:
    def test_returns_largest_when_ids_none(self):
        t = SingleBallTracker()
        boxes = _xyxy([0, 0, 10, 10], [0, 0, 100, 100], [0, 0, 50, 50])
        locked_id, locked_box = t.update(ids=None, xyxy=boxes)
        assert locked_id is None
        np.testing.assert_array_equal(locked_box, boxes[1])

    def test_returns_none_when_empty(self):
        t = SingleBallTracker()
        locked_id, locked_box = t.update(ids=None, xyxy=np.zeros((0, 4), dtype=np.float32))
        assert locked_id is None
        assert locked_box is None


class TestLockOnLargest:
    def test_initial_lock_picks_largest_with_id(self):
        t = SingleBallTracker()
        ids = np.array([7, 3, 9])
        boxes = _xyxy([0, 0, 10, 10], [0, 0, 200, 200], [0, 0, 50, 50])
        locked_id, locked_box = t.update(ids=ids, xyxy=boxes)
        assert locked_id == 3
        np.testing.assert_array_equal(locked_box, boxes[1])


class TestFollowLockedId:
    def test_follows_same_id_even_when_smaller(self):
        t = SingleBallTracker()
        # Frame 1: lock onto id=3 (largest)
        ids1 = np.array([3, 7])
        boxes1 = _xyxy([0, 0, 100, 100], [0, 0, 50, 50])
        t.update(ids=ids1, xyxy=boxes1)
        # Frame 2: id=3 is now smaller than id=7 — must still follow id=3
        ids2 = np.array([3, 7])
        boxes2 = _xyxy([0, 0, 30, 30], [0, 0, 200, 200])
        locked_id, locked_box = t.update(ids=ids2, xyxy=boxes2)
        assert locked_id == 3
        np.testing.assert_array_equal(locked_box, boxes2[0])

    def test_falls_back_to_largest_when_locked_id_missing(self):
        """When the locked ID disappears (ByteTrack reassigned it) but other balls
        are still in frame, return the largest available box rather than None.
        The lock is a hint, not a hard requirement — we never starve the kickup
        state machine just because the tracker lost an ID."""
        t = SingleBallTracker(lose_after_frames=5)
        t.update(ids=np.array([3]), xyxy=_xyxy([0, 0, 100, 100]))
        # Frame 2: id=3 missing, only id=7 present (much bigger).
        locked_id, locked_box = t.update(
            ids=np.array([7]), xyxy=_xyxy([0, 0, 200, 200])
        )
        # Returned the largest available box, not None.
        np.testing.assert_array_equal(locked_box, _xyxy([0, 0, 200, 200])[0])
        # Lock NOT silently changed — we still want to re-lock on id=3 if it returns.
        assert t.locked_id == 3


class TestDropAfterMisses:
    def test_drops_lock_after_n_consecutive_misses(self):
        t = SingleBallTracker(lose_after_frames=3)
        t.update(ids=np.array([3]), xyxy=_xyxy([0, 0, 100, 100]))
        # 3 frames with id=3 missing
        for _ in range(3):
            t.update(ids=np.array([7]), xyxy=_xyxy([0, 0, 50, 50]))
        # Next frame: id=3 still missing; lock should have dropped, re-lock on largest available
        locked_id, locked_box = t.update(
            ids=np.array([7, 9]), xyxy=_xyxy([0, 0, 50, 50], [0, 0, 200, 200])
        )
        assert locked_id == 9  # newly locked onto largest
        np.testing.assert_array_equal(locked_box, _xyxy([0, 0, 200, 200])[0])

    def test_miss_counter_resets_when_id_returns(self):
        t = SingleBallTracker(lose_after_frames=3)
        t.update(ids=np.array([3]), xyxy=_xyxy([0, 0, 100, 100]))
        # 2 misses
        t.update(ids=np.array([7]), xyxy=_xyxy([0, 0, 50, 50]))
        t.update(ids=np.array([7]), xyxy=_xyxy([0, 0, 50, 50]))
        # id=3 returns — miss counter must reset
        locked_id, locked_box = t.update(
            ids=np.array([3, 7]), xyxy=_xyxy([0, 0, 80, 80], [0, 0, 50, 50])
        )
        assert locked_id == 3
        # 2 more misses should NOT yet drop the lock (counter was reset to 0)
        t.update(ids=np.array([7]), xyxy=_xyxy([0, 0, 50, 50]))
        t.update(ids=np.array([7]), xyxy=_xyxy([0, 0, 50, 50]))
        assert t.locked_id == 3


class TestEmptyFrameAdvancesMisses:
    def test_no_detections_counts_as_miss(self):
        t = SingleBallTracker(lose_after_frames=2)
        t.update(ids=np.array([3]), xyxy=_xyxy([0, 0, 100, 100]))
        t.update(ids=None, xyxy=np.zeros((0, 4), dtype=np.float32))
        t.update(ids=None, xyxy=np.zeros((0, 4), dtype=np.float32))
        # 2 misses reached → next frame drops lock
        locked_id, _ = t.update(
            ids=np.array([9]), xyxy=_xyxy([0, 0, 100, 100])
        )
        assert locked_id == 9
