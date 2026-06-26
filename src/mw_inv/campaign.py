"""Deposit campaigns — versioned mine-block models as first-class inputs (backlog G3).

A campaign bundles ore profiles, measured dielectric library references, and
evaluation defaults so pipeline runs and envelope studies are reproducible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Campaign:
    """Versioned deposit evaluation bundle."""

    campaign_id: str
    description: str
    ore_globs: tuple[str, ...]
    measured_dielectrics: str | None = None
    manifest_path: str | None = None
    target_T_K: float = 298.15
    gangue_T_K: float = 298.15
    freq_hz: float = 2.45e9
    created: str = ""

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "description": self.description,
            "ore_globs": list(self.ore_globs),
            "measured_dielectrics": self.measured_dielectrics,
            "manifest_path": self.manifest_path,
            "target_T_K": self.target_T_K,
            "gangue_T_K": self.gangue_T_K,
            "freq_hz": self.freq_hz,
            "created": self.created,
        }


def _data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def discover_campaign_files(root: Path | None = None) -> list[Path]:
    """Find ``campaign.json`` files under ``data/campaigns/``."""
    base = root or _data_root()
    camp_dir = base / "campaigns"
    if not camp_dir.is_dir():
        return []
    return sorted(camp_dir.rglob("campaign.json"))


def load_campaign(path: Path | str) -> Campaign:
    p = Path(path)
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{p}: campaign must be a JSON object")
    cid = data.get("campaign_id") or p.parent.name
    globs = data.get("ore_globs") or data.get("ore_paths") or []
    if isinstance(globs, str):
        globs = [globs]
    if not globs:
        raise ValueError(f"{p}: ore_globs (or ore_paths) is required")
    return Campaign(
        campaign_id=str(cid),
        description=str(data.get("description", "")),
        ore_globs=tuple(str(g) for g in globs),
        measured_dielectrics=data.get("measured_dielectrics"),
        manifest_path=data.get("manifest_path"),
        target_T_K=float(data.get("target_T_K", 298.15)),
        gangue_T_K=float(data.get("gangue_T_K", 298.15)),
        freq_hz=float(data.get("freq_hz", 2.45e9)),
        created=str(data.get("created", "")),
    )


def resolve_ore_paths(campaign: Campaign, data_root: Path | None = None) -> list[Path]:
    """Expand campaign ore globs relative to *data_root* (default repo ``data/``)."""
    root = data_root or _data_root()
    paths: list[Path] = []
    seen: set[str] = set()
    for pattern in campaign.ore_globs:
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            if p.name.startswith("_"):
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
    return paths


def campaign_summary(campaign: Campaign, data_root: Path | None = None) -> dict:
    """Lightweight discovery payload for manifests and real-data eval."""
    root = data_root or _data_root()
    ore_paths = resolve_ore_paths(campaign, root)
    return {
        **campaign.to_dict(),
        "n_ores": len(ore_paths),
        "ore_paths_sample": [str(p.relative_to(root)) for p in ore_paths[:5]],
    }
