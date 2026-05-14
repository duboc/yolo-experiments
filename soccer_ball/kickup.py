"""Kickup counting and motion-trail helpers.

The counter tracks the vertical position of the largest detected ball and
counts each time the EMA-smoothed velocity flips from positive (falling, y
increasing in image coords) to negative (rising). Sub-threshold velocity
changes are filtered out so micro-jitter doesn't inflate the count.
"""

from __future__ import annotations

from collections import deque
from typing import Iterator

from soccer_ball.pose import BodyPart


class KickupCounter:
    def __init__(
        self,
        ema_alpha: float = 0.3,
        min_velocity: float = 1.0,
        absence_reset_frames: int = 30,
        min_velocity_pct: float = 0.0,
        acceleration_threshold: float = 0.0,
    ):
        self._alpha = ema_alpha
        self._min_velocity_floor = min_velocity
        self._min_velocity_pct = min_velocity_pct
        self._absence_reset = absence_reset_frames
        self._accel_threshold = acceleration_threshold

        self.count: int = 0
        self.just_kicked: bool = False
        self.counts_by_part: dict[BodyPart, int] = {bp: 0 for bp in BodyPart}
        self.last_part: BodyPart | None = None
        self.rejected_count: int = 0

        self._last_y: float | None = None
        self._smoothed_v: float = 0.0
        self._prev_smoothed_v: float = 0.0
        self._absence: int = 0
        self._state: str = "NEUTRAL"

    @property
    def state(self) -> str:
        return self._state

    @property
    def smoothed_velocity(self) -> float:
        return self._smoothed_v

    @property
    def last_acceleration(self) -> float:
        return self._smoothed_v - self._prev_smoothed_v

    def update(self, y: float | None, frame_height: int | None = None) -> None:
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
            self._prev_smoothed_v = 0.0
            return

        instant_v = y - self._last_y  # positive = ball moving down (image coords)
        self._prev_smoothed_v = self._smoothed_v
        self._smoothed_v = self._alpha * instant_v + (1 - self._alpha) * self._smoothed_v
        self._last_y = y

        threshold = self._effective_min_velocity(frame_height)
        acceleration = self._smoothed_v - self._prev_smoothed_v

        # State machine: NEUTRAL → FALLING → RISING (counts a kickup) → ...
        if self._smoothed_v > threshold:
            self._state = "FALLING"
        elif self._smoothed_v < -threshold:
            if self._state == "FALLING" and abs(acceleration) >= self._accel_threshold:
                self.count += 1
                self.just_kicked = True
            self._state = "RISING"

    def _effective_min_velocity(self, frame_height: int | None) -> float:
        if frame_height is None or self._min_velocity_pct <= 0:
            return self._min_velocity_floor
        return max(self._min_velocity_floor, frame_height * self._min_velocity_pct)

    def credit_last(self, part: BodyPart | None) -> None:
        """Attribute the most recent bounce to a body part — or reject it.

        Called by the main loop right after an update where ``just_kicked`` was
        True. Passing None rolls back the bounce (count -= 1, just_kicked = False)
        so pose-gated rejections don't inflate the counter; ``rejected_count``
        is bumped so the user can see how many bounces died at the gate.
        """
        if part is None:
            if self.just_kicked and self.count > 0:
                self.count -= 1
                self.just_kicked = False
                self.rejected_count += 1
            self.last_part = None
            return
        self.last_part = part
        self.counts_by_part[part] = self.counts_by_part.get(part, 0) + 1

    def reset(self) -> None:
        self.count = 0
        self.just_kicked = False
        self.counts_by_part = {bp: 0 for bp in BodyPart}
        self.last_part = None
        self.rejected_count = 0
        self._last_y = None
        self._smoothed_v = 0.0
        self._prev_smoothed_v = 0.0
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
