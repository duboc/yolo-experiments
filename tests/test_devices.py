"""Tests for device auto-detection.

Uses a SimpleNamespace as a stand-in for the torch module so the tests run
on any platform regardless of whether MPS or CUDA is actually available.
"""

from types import SimpleNamespace

import pytest

from soccer_ball.devices import auto_device


def _fake_torch(*, mps_available: bool, cuda_available: bool) -> SimpleNamespace:
    return SimpleNamespace(
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available),
        ),
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )


def test_prefers_mps_when_available():
    torch = _fake_torch(mps_available=True, cuda_available=True)
    assert auto_device(torch) == "mps"


def test_falls_back_to_cuda_when_no_mps():
    torch = _fake_torch(mps_available=False, cuda_available=True)
    assert auto_device(torch) == "cuda"


def test_falls_back_to_cpu_when_neither():
    torch = _fake_torch(mps_available=False, cuda_available=False)
    assert auto_device(torch) == "cpu"


def test_handles_missing_mps_backend():
    """Older torch builds may not expose torch.backends.mps at all."""
    torch = SimpleNamespace(
        backends=SimpleNamespace(),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    assert auto_device(torch) == "cuda"
