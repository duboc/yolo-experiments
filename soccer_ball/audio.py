"""Fire-and-forget sound effects for kickup events.

Uses the OS's built-in audio CLI (``afplay`` on macOS, ``aplay`` on Linux)
to keep the dependency surface zero. Calls are non-blocking (Popen without
.wait()) so the inference loop never stalls on audio.

The player auto-disables on the first failure (missing binary, bad path,
permissions) so a misconfigured sound never crashes the loop.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Callable

DEFAULT_DARWIN_SOUND = "/System/Library/Sounds/Pop.aiff"

PopenFn = Callable[..., object]


def _platform_command(platform_name: str, sound_path: str) -> list[str] | None:
    if not sound_path:
        return None
    if platform_name == "darwin":
        return ["afplay", sound_path]
    if platform_name.startswith("linux"):
        return ["aplay", "-q", sound_path]
    return None


class SoundPlayer:
    def __init__(
        self,
        sound_path: str | None = None,
        *,
        enabled: bool = True,
        platform: str = sys.platform,
        popen_fn: PopenFn = subprocess.Popen,
    ):
        if sound_path is None:
            sound_path = DEFAULT_DARWIN_SOUND if platform == "darwin" else ""
        self.sound_path = sound_path
        self.enabled = enabled
        self._cmd = _platform_command(platform, sound_path)
        self._popen = popen_fn

    def play(self) -> None:
        if not self.enabled or self._cmd is None:
            return
        try:
            self._popen(
                self._cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            self.enabled = False
