"""Configuration dataclasses and JSON preset persistence.

``LaunchConfig`` holds params fixed for a single run (model, device, source).
``RuntimeSettings`` holds params live-tunable during the loop (conf, iou, ...).
Presets persist only ``RuntimeSettings`` — launch config is process-scoped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

DEFAULT_PRESETS_DIR = Path("presets")


@dataclass(frozen=True)
class LaunchConfig:
    model: str = "yolo26l.pt"
    device: str = "auto"
    source: str | None = None
    width: int | None = None
    height: int | None = None
    half: bool = True  # MPS/CUDA-friendly default; resolved to False on CPU at runtime
    preset: str = "default"


@dataclass(frozen=True)
class RuntimeSettings:
    conf: float = 0.25
    iou: float = 0.7
    max_det: int = 300
    imgsz: int = 640
    ball_class: int = 32
    agnostic_nms: bool = False
    show_fps: bool = True
    show_label: bool = True
    show_settings: bool = True
    min_area_pct: float = 1.0  # Drop boxes smaller than this % of frame area
    proximity_px: int = 80     # Foot/knee/head proximity radius for kickup gating
    show_pose: bool = True     # Render pose-keypoint overlay

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeSettings":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


def merge(base: RuntimeSettings, override: dict[str, Any]) -> RuntimeSettings:
    """Return a new RuntimeSettings with non-None override values applied."""
    valid = {f.name for f in fields(RuntimeSettings)}
    applied = {k: v for k, v in override.items() if k in valid and v is not None}
    return replace(base, **applied)


def _preset_path(name: str, presets_dir: Path) -> Path:
    return Path(presets_dir) / f"{name}.json"


def save_preset(
    name: str, settings: RuntimeSettings, presets_dir: Path = DEFAULT_PRESETS_DIR
) -> Path:
    presets_dir = Path(presets_dir)
    presets_dir.mkdir(parents=True, exist_ok=True)
    path = _preset_path(name, presets_dir)
    path.write_text(json.dumps(settings.to_dict(), indent=2, sort_keys=True))
    return path


def load_preset(
    name: str, presets_dir: Path = DEFAULT_PRESETS_DIR
) -> RuntimeSettings:
    path = _preset_path(name, Path(presets_dir))
    if not path.exists():
        raise FileNotFoundError(path)
    return RuntimeSettings.from_dict(json.loads(path.read_text()))


def list_presets(presets_dir: Path = DEFAULT_PRESETS_DIR) -> list[str]:
    presets_dir = Path(presets_dir)
    if not presets_dir.exists():
        return []
    return sorted(p.stem for p in presets_dir.glob("*.json"))
