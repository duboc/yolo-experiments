"""Pre-loop TUI for collecting init-time launch config.

Each prompt shows the current default in brackets; blank input keeps it.
EOF (ctrl+D) is translated to KeyboardInterrupt so the caller can treat both
ctrl+C and ctrl+D as a clean cancel.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from soccer_ball.settings import LaunchConfig

_VALID_DEVICES = {"auto", "mps", "cuda", "cpu"}
_VALID_BOOLS = {"y": True, "yes": True, "n": False, "no": False}

InputFn = Callable[[str], str]


def _ask(prompt: str, default: str, input_fn: InputFn) -> str:
    try:
        raw = input_fn(f"{prompt} [{default}]: ").strip()
    except EOFError as exc:
        raise KeyboardInterrupt from exc
    return raw or default


def _ask_until(prompt: str, default: str, validator: Callable[[str], bool], input_fn: InputFn) -> str:
    while True:
        value = _ask(prompt, default, input_fn)
        if validator(value):
            return value
        print(f"  '{value}' is not a valid choice, try again.")


def _ask_optional_int(prompt: str, default: int | None, input_fn: InputFn) -> int | None:
    default_str = "" if default is None else str(default)
    while True:
        try:
            raw = input_fn(f"{prompt} [{default_str}]: ").strip()
        except EOFError as exc:
            raise KeyboardInterrupt from exc
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            print(f"  '{raw}' is not an integer, try again.")


def _ask_bool(prompt: str, default: bool, input_fn: InputFn) -> bool:
    default_str = "y" if default else "n"
    while True:
        raw = _ask(prompt, default_str, input_fn).lower()
        if raw in _VALID_BOOLS:
            return _VALID_BOOLS[raw]
        print(f"  '{raw}' is not y/n, try again.")


def prompt_launch_config(
    defaults: LaunchConfig, input_fn: InputFn = input
) -> LaunchConfig:
    """Walk the user through every launch param, returning the final config."""
    print("\n--- Launch configuration (Enter to accept default) ---")

    model = _ask("Model", defaults.model, input_fn)
    device = _ask_until(
        "Device (auto/mps/cuda/cpu)",
        defaults.device,
        lambda v: v in _VALID_DEVICES,
        input_fn,
    )
    source_default = "" if defaults.source is None else defaults.source
    source_raw = _ask("Source (blank = camera picker)", source_default, input_fn)
    source = source_raw or None
    width = _ask_optional_int("Camera width (blank = camera default)", defaults.width, input_fn)
    height = _ask_optional_int("Camera height (blank = camera default)", defaults.height, input_fn)
    half = _ask_bool("Use FP16 half-precision?", defaults.half, input_fn)
    preset = _ask("Preset name", defaults.preset, input_fn)

    return replace(
        defaults,
        model=model,
        device=device,
        source=source,
        width=width,
        height=height,
        half=half,
        preset=preset,
    )
