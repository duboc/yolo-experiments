"""Single-ball tracker.

Locks onto one tracker ID (from `model.track()` output) and follows it across
frames so the kickup counter sees a continuous trajectory rather than the
"largest box per frame" jitter that hops between balls.

When the upstream tracker has not assigned IDs yet (`ids is None`, common on
the very first frames), the tracker falls back to "biggest box this frame"
so the loop keeps producing useful centroids.
"""

from __future__ import annotations

import numpy as np


class SingleBallTracker:
    def __init__(self, lose_after_frames: int = 5):
        self._lose_after = lose_after_frames
        self.locked_id: int | None = None
        self._misses: int = 0

    def update(
        self,
        ids: np.ndarray | None,
        xyxy: np.ndarray,
    ) -> tuple[int | None, np.ndarray | None]:
        """Return (locked_id, locked_box) for this frame.

        ``locked_id`` is None when the tracker is using the no-IDs fallback or
        when the locked ID is temporarily missing from this frame.
        """
        if xyxy.shape[0] == 0:
            self._register_miss()
            return None, None

        if ids is None:
            # Tracker hasn't assigned IDs yet — biggest-box fallback, no lock.
            return None, _largest_box(xyxy)

        if self.locked_id is None:
            return self._lock_on_largest(ids, xyxy)

        # Locked: try to find the same ID in this frame.
        match_idx = _index_of(self.locked_id, ids)
        if match_idx is not None:
            self._misses = 0
            return int(self.locked_id), xyxy[match_idx]

        # Locked ID missing this frame. Don't starve the kickup state machine —
        # fall back to the largest available box so the centroid keeps flowing.
        # Still register the miss so the lock eventually drops if persistent.
        self._register_miss()
        return None, _largest_box(xyxy)

    def _lock_on_largest(
        self, ids: np.ndarray, xyxy: np.ndarray
    ) -> tuple[int, np.ndarray]:
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        idx = int(areas.argmax())
        self.locked_id = int(ids[idx])
        self._misses = 0
        return self.locked_id, xyxy[idx]

    def _register_miss(self) -> None:
        if self.locked_id is None:
            return
        self._misses += 1
        if self._misses >= self._lose_after:
            self.locked_id = None
            self._misses = 0

    def reset(self) -> None:
        self.locked_id = None
        self._misses = 0


def _index_of(target_id: int, ids: np.ndarray) -> int | None:
    matches = np.where(ids == target_id)[0]
    return int(matches[0]) if matches.size else None


def _largest_box(xyxy: np.ndarray) -> np.ndarray:
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return xyxy[int(areas.argmax())]
