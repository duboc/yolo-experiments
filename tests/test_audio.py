"""Tests for the SoundPlayer.

The player is intentionally tiny: pick a platform-appropriate command, fire
it via Popen, swallow errors. We test it with an injectable popen_fn so we
never actually launch a subprocess in CI.
"""

from unittest.mock import MagicMock

import pytest

from soccer_ball.audio import DEFAULT_DARWIN_SOUND, SoundPlayer


class TestDisabled:
    def test_no_op_when_disabled(self):
        popen = MagicMock()
        p = SoundPlayer(sound_path="/anything.aiff", platform="darwin", enabled=False, popen_fn=popen)
        p.play()
        popen.assert_not_called()

    def test_no_op_when_no_command(self):
        popen = MagicMock()
        p = SoundPlayer(sound_path="/some.aiff", platform="windows", popen_fn=popen)
        p.play()
        # No command on Windows → silently no-op.
        popen.assert_not_called()

    def test_no_op_when_sound_path_empty(self):
        popen = MagicMock()
        p = SoundPlayer(sound_path="", platform="darwin", popen_fn=popen)
        p.play()
        popen.assert_not_called()


class TestPlatformCommand:
    def test_darwin_uses_afplay(self):
        popen = MagicMock()
        p = SoundPlayer(sound_path="/foo.aiff", platform="darwin", popen_fn=popen)
        p.play()
        cmd = popen.call_args[0][0]
        assert cmd[0] == "afplay"
        assert cmd[-1] == "/foo.aiff"

    def test_linux_uses_aplay(self):
        popen = MagicMock()
        p = SoundPlayer(sound_path="/foo.wav", platform="linux", popen_fn=popen)
        p.play()
        cmd = popen.call_args[0][0]
        assert cmd[0] == "aplay"
        assert cmd[-1] == "/foo.wav"


class TestErrorHandling:
    def test_disables_on_first_failure(self):
        popen = MagicMock(side_effect=FileNotFoundError("afplay missing"))
        p = SoundPlayer(sound_path="/foo.aiff", platform="darwin", popen_fn=popen)
        p.play()
        # First attempt fired (and failed), second should be a no-op.
        assert popen.call_count == 1
        p.play()
        assert popen.call_count == 1
        assert p.enabled is False

    def test_oserror_also_disables(self):
        popen = MagicMock(side_effect=OSError("permission denied"))
        p = SoundPlayer(sound_path="/foo.aiff", platform="darwin", popen_fn=popen)
        p.play()
        assert p.enabled is False


class TestDefaults:
    def test_default_darwin_sound_path_is_pop(self):
        # We expect a real macOS system sound; the constant value matters
        # because the module-level fallback uses it.
        assert "Sounds" in DEFAULT_DARWIN_SOUND
        assert DEFAULT_DARWIN_SOUND.endswith(".aiff")

    def test_omitting_sound_path_uses_darwin_default(self):
        popen = MagicMock()
        p = SoundPlayer(platform="darwin", popen_fn=popen)
        p.play()
        cmd = popen.call_args[0][0]
        assert cmd[-1] == DEFAULT_DARWIN_SOUND
