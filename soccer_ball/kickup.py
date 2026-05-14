"""Kickup counting and motion-trail helpers.

The counter tracks the vertical position of the largest detected ball and
counts each time the EMA-smoothed velocity flips from positive (falling, y
increasing in image coords) to negative (rising). Sub-threshold velocity
changes are filtered out so micro-jitter doesn't inflate the count.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator


class KickupCounter:
    def __init__(
        self,
        ema_alpha: float = 0.3,
        min_velocity: float = 1.0,
        absence_reset_frames: int = 30,
    ):
        self._alpha = ema_alpha
        self._min_velocity = min_velocity
        self._absence_reset = absence_reset_frames

        self.count: int = 0
        self.just_kicked: bool = False

        self._last_y: float | None = None
        self._smoothed_v: float = 0.0
        self._absence: int = 0
        self._state: str = "NEUTRAL"

    def update(self, y: float | None) -> None:
        # Clear the one-frame "just kicked" pulse at the start of every update.
        self.just_kicked = False

        if y is None:
            self._absence += 1
            if self._absence >= self._absence_reset:
                self.reset()
            return

        self._absence = 0

        if self._last_y is None:
            self._last_y = y
            self._smoothed_v = 0.0
            return

        instant_v = y - self._last_y  # positive = ball moving down (image coords)
        self._smoothed_v = self._alpha * instant_v + (1 - self._alpha) * self._smoothed_v
        self._last_y = y

        # State machine: NEUTRAL → FALLING → RISING (counts a kickup) → FALLING → ...
        # An immediate first-frame rise from NEUTRAL never counts.
        if self._smoothed_v > self._min_velocity:
            self._state = "FALLING"
        elif self._smoothed_v < -self._min_velocity:
            if self._state == "FALLING":
                self.count += 1
                self.just_kicked = True
            self._state = "RISING"

    def reset(self) -> None:
        self.count = 0
        self.just_kicked = False
        self._last_y = None
        self._smoothed_v = 0.0
        self._absence = 0
        self._state = "NEUTRAL"


class MotionTrail:
    """Bounded deque of (x, y) centroids; ``append(None)`` is a no-op."""

    def __init__(self, max_len: int = 30):
        self._buf: deque[tuple[int, int]] = deque(maxlen=max_len)

    def append(self, point: tuple[int, int] | None) -> None:
        if point is None:
            return
        self._buf.append(point)

    def clear(self) -> None:
        self._buf.clear()

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
