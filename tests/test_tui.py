"""Tests for the pre-loop TUI prompt."""

import pytest

from soccer_ball.settings import LaunchConfig
from soccer_ball.tui import prompt_launch_config


def _scripted(answers: list[str]):
    """Build an input_fn that returns each scripted answer in turn."""
    it = iter(answers)
    return lambda _prompt: next(it)


class TestPromptLaunchConfig:
    def test_blank_input_keeps_defaults(self):
        defaults = LaunchConfig()
        # 7 prompts: model, device, source, width, height, half, preset
        cfg = prompt_launch_config(defaults, input_fn=_scripted([""] * 7))
        assert cfg == defaults

    def test_overrides_model_and_device(self):
        defaults = LaunchConfig()
        cfg = prompt_launch_config(
            defaults,
            input_fn=_scripted(["yolo26x.pt", "mps", "", "", "", "", ""]),
        )
        assert cfg.model == "yolo26x.pt"
        assert cfg.device == "mps"

    def test_invalid_device_reprompts(self):
        defaults = LaunchConfig()
        cfg = prompt_launch_config(
            defaults,
            input_fn=_scripted(["", "tpu", "cpu", "", "", "", "", ""]),
        )
        assert cfg.device == "cpu"

    def test_half_yes_no(self):
        defaults = LaunchConfig()
        cfg = prompt_launch_config(
            defaults,
            input_fn=_scripted(["", "", "", "", "", "y", ""]),
        )
        assert cfg.half is True

    def test_half_invalid_reprompts(self):
        defaults = LaunchConfig()
        cfg = prompt_launch_config(
            defaults,
            input_fn=_scripted(["", "", "", "", "", "maybe", "n", ""]),
        )
        assert cfg.half is False

    def test_width_height_parses_int(self):
        defaults = LaunchConfig()
        cfg = prompt_launch_config(
            defaults,
            input_fn=_scripted(["", "", "", "1920", "1080", "", ""]),
        )
        assert cfg.width == 1920
        assert cfg.height == 1080

    def test_width_invalid_reprompts(self):
        defaults = LaunchConfig()
        cfg = prompt_launch_config(
            defaults,
            input_fn=_scripted(["", "", "", "abc", "1280", "", "", ""]),
        )
        assert cfg.width == 1280

    def test_eof_propagates_keyboard_interrupt(self):
        defaults = LaunchConfig()

        def raise_eof(_prompt):
            raise EOFError

        with pytest.raises(KeyboardInterrupt):
            prompt_launch_config(defaults, input_fn=raise_eof)

    def test_source_preserved_when_blank(self):
        defaults = LaunchConfig(source="0")
        cfg = prompt_launch_config(defaults, input_fn=_scripted([""] * 7))
        assert cfg.source == "0"

    def test_preset_override(self):
        defaults = LaunchConfig()
        cfg = prompt_launch_config(
            defaults,
            input_fn=_scripted(["", "", "", "", "", "", "indoor"]),
        )
        assert cfg.preset == "indoor"
