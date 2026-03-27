"""Zone configuration dataclasses and JSON persistence helpers."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SlotZone:
    """A named parking zone defined by a polygon.

    Attributes:
        name: Human-readable zone identifier, e.g. "zone_A".
        polygon: Polygon vertices as an (N, 2) int32 array of (x, y) pixel coords.
    """

    name: str
    polygon: np.ndarray  # shape (N, 2), dtype int32


@dataclass
class ZoneConfig:
    """Complete zone configuration for a single video source.

    Attributes:
        source: Filename or path of the video this config was annotated on.
        frame_width: Width in pixels of the reference frame.
        frame_height: Height in pixels of the reference frame.
        zones: List of SlotZone definitions.
    """

    source: str
    frame_width: int
    frame_height: int
    zones: list[SlotZone] = field(default_factory=list)


def _default_config_path(video_path: str | Path) -> Path:
    """Derive the default zone config path for a given video file.

    Convention: data/zones/<video_stem>_zones.json, relative to the
    repository root (two levels up from this file's directory).

    Args:
        video_path: Path to the video file.

    Returns:
        Resolved Path for the zone config JSON.
    """
    stem = Path(video_path).stem
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "processing"/ "parking_slots" / f"{stem}_zones.json"


def load_zone_config(config_path: str | Path) -> ZoneConfig:
    """Load zone definitions from a JSON file.

    Args:
        config_path: Absolute or relative path to the JSON config file.

    Returns:
        A ZoneConfig populated with SlotZone entries.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If the JSON structure is invalid or a polygon has fewer than 3 vertices.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Zone config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    required_keys = {"source", "frame_width", "frame_height", "zones"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Zone config missing keys: {missing}")

    zones: list[SlotZone] = []
    for entry in data["zones"]:
        if "name" not in entry or "polygon" not in entry:
            raise ValueError(f"Zone entry missing 'name' or 'polygon': {entry}")
        polygon = np.array(entry["polygon"], dtype=np.int32)
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError(
                f"Polygon for zone '{entry['name']}' must be shape (N, 2), got {polygon.shape}"
            )
        if len(polygon) < 3:
            raise ValueError(
                f"Polygon for zone '{entry['name']}' needs at least 3 vertices, got {len(polygon)}"
            )
        zones.append(SlotZone(name=entry["name"], polygon=polygon))

    return ZoneConfig(
        source=data["source"],
        frame_width=int(data["frame_width"]),
        frame_height=int(data["frame_height"]),
        zones=zones,
    )


def save_zone_config(config: ZoneConfig, config_path: str | Path) -> None:
    """Persist a ZoneConfig to disk as JSON.

    Parent directories are created if absent.

    Args:
        config: The zone configuration to serialize.
        config_path: Destination path for the JSON file.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source": config.source,
        "frame_width": config.frame_width,
        "frame_height": config.frame_height,
        "zones": [
            {"name": z.name, "polygon": z.polygon.tolist()}
            for z in config.zones
        ],
    }

    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
