"""Tests for the settings dataclasses and JSON preset I/O."""

import json
from pathlib import Path

import pytest

from soccer_ball.settings import (
    LaunchConfig,
    RuntimeSettings,
    list_presets,
    load_preset,
    merge,
    save_preset,
)


class TestLaunchConfigDefaults:
    def test_has_expected_defaults(self):
        cfg = LaunchConfig()
        assert cfg.model == "yolo26l.pt"
        assert cfg.device == "auto"
        assert cfg.source is None
        assert cfg.width is None
        assert cfg.height is None
        assert cfg.half is True  # M-series friendly default; CPU users get auto-override
        assert cfg.preset == "default"


class TestRuntimeSettingsDefaults:
    def test_has_expected_defaults(self):
        s = RuntimeSettings()
        assert s.conf == 0.25
        assert s.iou == 0.7
        assert s.max_det == 300
        assert s.imgsz == 640
        assert s.ball_class == 32
        assert s.agnostic_nms is False
        assert s.show_fps is True
        assert s.show_label is True
        assert s.show_settings is True
        assert s.min_area_pct == 1.0
        assert s.proximity_px == 150
        assert s.show_pose is True
        assert s.show_debug is False


class TestRuntimeSettingsRoundTrip:
    def test_to_dict_then_from_dict(self):
        original = RuntimeSettings(conf=0.5, iou=0.6, max_det=100, imgsz=960,
                                   ball_class=0, agnostic_nms=True,
                                   show_fps=False, show_label=False)
        restored = RuntimeSettings.from_dict(original.to_dict())
        assert restored == original


class TestMerge:
    def test_overlays_non_none_fields(self):
        base = RuntimeSettings(conf=0.25, iou=0.7)
        override = {"conf": 0.4, "iou": None}  # iou=None must be ignored
        out = merge(base, override)
        assert out.conf == 0.4
        assert out.iou == 0.7

    def test_unknown_keys_ignored(self):
        base = RuntimeSettings()
        out = merge(base, {"this_key_does_not_exist": 999})
        assert out == base

    def test_returns_new_instance(self):
        base = RuntimeSettings()
        out = merge(base, {"conf": 0.5})
        assert out is not base
        assert base.conf == 0.25  # base unchanged


class TestPresetIO:
    def test_save_then_load_round_trip(self, tmp_path: Path):
        s = RuntimeSettings(conf=0.42, iou=0.55, ball_class=0)
        save_preset("foo", s, presets_dir=tmp_path)
        loaded = load_preset("foo", presets_dir=tmp_path)
        assert loaded == s

    def test_save_creates_directory(self, tmp_path: Path):
        nested = tmp_path / "nested" / "deeper"
        save_preset("x", RuntimeSettings(), presets_dir=nested)
        assert (nested / "x.json").exists()

    def test_load_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_preset("nope", presets_dir=tmp_path)

    def test_save_writes_json(self, tmp_path: Path):
        save_preset("p", RuntimeSettings(conf=0.33), presets_dir=tmp_path)
        data = json.loads((tmp_path / "p.json").read_text())
        assert data["conf"] == 0.33

    def test_list_presets_sorted(self, tmp_path: Path):
        for name in ["zebra", "alpha", "mike"]:
            save_preset(name, RuntimeSettings(), presets_dir=tmp_path)
        assert list_presets(presets_dir=tmp_path) == ["alpha", "mike", "zebra"]

    def test_list_presets_empty(self, tmp_path: Path):
        assert list_presets(presets_dir=tmp_path) == []

    def test_list_presets_missing_dir(self, tmp_path: Path):
        assert list_presets(presets_dir=tmp_path / "does-not-exist") == []
