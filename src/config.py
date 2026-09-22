"""Configuration loading: settings.yaml + one zone JSON per camera."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Zone:
    id: str
    type: str          # entrance | counter | queue | table | beverage | exit | other
    label: str
    points: np.ndarray
    seats: int = 0
    occupancy: bool = True   # does this zone contribute to store-wide occupancy?


@dataclass
class Line:
    id: str
    label: str
    points: np.ndarray       # shape (2, 2): A -> B
    in_side: str = "right"   # which side of A->B is "into the store"


@dataclass
class Mask:
    id: str
    label: str
    points: np.ndarray


@dataclass
class CameraConfig:
    name: str
    label: str
    video: Path
    role: str
    start_time: str | None = None   # wall-clock of video t=0, for readable reports
    zones: list[Zone] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    masks: list[Mask] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None

    def zone(self, zone_id: str) -> Zone | None:
        return next((z for z in self.zones if z.id == zone_id), None)


@dataclass
class Settings:
    model: dict[str, Any]
    processing: dict[str, Any]
    analytics: dict[str, Any]
    output: dict[str, Any]
    cameras: list[CameraConfig]


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def _load_zone_file(path: Path) -> tuple[list[Zone], list[Line], list[Mask], tuple[int, int] | None]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    zones = [
        Zone(
            id=z["id"],
            type=z.get("type", "other"),
            label=z.get("label", z["id"]),
            points=np.asarray(z["points"], dtype=np.int32),
            seats=int(z.get("seats", 0)),
            occupancy=bool(z.get("occupancy", True)),
        )
        for z in raw.get("polygons", [])
    ]
    lines = [
        Line(
            id=l["id"],
            label=l.get("label", l["id"]),
            points=np.asarray(l["points"], dtype=np.float32),
            in_side=l.get("in_side", "right"),
        )
        for l in raw.get("lines", [])
    ]
    masks = [
        Mask(
            id=m["id"],
            label=m.get("label", m["id"]),
            points=np.asarray(m["points"], dtype=np.int32),
        )
        for m in raw.get("masks", [])
    ]
    fs = raw.get("frame_size")
    return zones, lines, masks, (tuple(fs) if fs else None)


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    cfg_path = _resolve(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    cameras: list[CameraConfig] = []
    for c in raw.get("cameras", []):
        if not c.get("enabled", True):
            continue
        zones, lines, masks, frame_size = _load_zone_file(_resolve(c["zones"]))
        cameras.append(
            CameraConfig(
                name=c["name"],
                label=c.get("label", c["name"]),
                video=_resolve(c["video"]),
                role=c.get("role", "secondary"),
                start_time=c.get("start_time"),
                zones=zones,
                lines=lines,
                masks=masks,
                frame_size=frame_size,
            )
        )

    return Settings(
        model=raw.get("model", {}),
        processing=raw.get("processing", {}),
        analytics=raw.get("analytics", {}),
        output=raw.get("output", {}),
        cameras=cameras,
    )
