"""Device auto-detection for Ultralytics inference.

Importing torch is expensive and not always desirable at import time, so the
public API accepts an injectable ``torch_module`` (mainly for tests) and
imports torch lazily otherwise.
"""

from __future__ import annotations

from typing import Any


def auto_device(torch_module: Any | None = None) -> str:
    """Return the best available device string: ``mps`` > ``cuda`` > ``cpu``."""
    if torch_module is None:
        import torch as torch_module  # local import keeps test cost low

    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"
