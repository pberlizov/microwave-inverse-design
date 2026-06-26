"""Machine-readable catalog of open / online datasets and ingest status."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetCatalogEntry:
    id: str
    title: str
    url: str
    license: str
    tags: tuple[str, ...]
    priority: int
    ingest_adapter: str | None
    ingest_output: str | None
    ingest_auto: bool
    ingest_notes: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DatasetCatalogEntry":
        ingest = row.get("ingest") or {}
        return cls(
            id=str(row["id"]),
            title=str(row.get("title", row["id"])),
            url=str(row.get("url", "")),
            license=str(row.get("license", "")),
            tags=tuple(str(t) for t in row.get("tags", [])),
            priority=int(row.get("priority", 99)),
            ingest_adapter=(None if ingest.get("adapter") is None else str(ingest["adapter"])),
            ingest_output=(None if not ingest.get("output") else str(ingest["output"])),
            ingest_auto=bool(ingest.get("auto", False)),
            ingest_notes=str(ingest.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "license": self.license,
            "tags": list(self.tags),
            "priority": self.priority,
            "ingest": {
                "adapter": self.ingest_adapter,
                "output": self.ingest_output,
                "auto": self.ingest_auto,
                "notes": self.ingest_notes,
            },
        }


@dataclass(frozen=True)
class DatasetsCatalog:
    version: str
    description: str
    entries: tuple[DatasetCatalogEntry, ...]

    def auto_ingest_entries(self) -> tuple[DatasetCatalogEntry, ...]:
        return tuple(e for e in self.entries if e.ingest_auto and e.ingest_adapter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "n_entries": len(self.entries),
            "datasets": [e.to_dict() for e in self.entries],
        }


def _default_catalog_path(data_root: Path) -> Path:
    return data_root / "datasets_catalog.json"


def load_datasets_catalog(data_root: Path | None = None) -> DatasetsCatalog:
    root = data_root or Path(__file__).resolve().parents[2] / "data"
    path = _default_catalog_path(root)
    if not path.is_file():
        return DatasetsCatalog(version="", description="", entries=())
    data = json.loads(path.read_text())
    entries = tuple(
        DatasetCatalogEntry.from_dict(row)
        for row in sorted(data.get("datasets", []), key=lambda r: int(r.get("priority", 99)))
    )
    return DatasetsCatalog(
        version=str(data.get("version", "")),
        description=str(data.get("description", "")),
        entries=entries,
    )


def ingest_status(data_root: Path | None = None) -> list[dict[str, Any]]:
    """Report which catalog outputs exist on disk."""
    root = data_root or Path(__file__).resolve().parents[2] / "data"
    catalog = load_datasets_catalog(root)
    rows: list[dict[str, Any]] = []
    for entry in catalog.entries:
        if not entry.ingest_output:
            rows.append({
                "id": entry.id,
                "status": "manual",
                "path": None,
                "adapter": entry.ingest_adapter,
            })
            continue
        out = root / entry.ingest_output
        rows.append({
            "id": entry.id,
            "status": "ingested" if out.is_file() else "pending",
            "path": str(out.resolve()) if out.is_file() else str(out),
            "adapter": entry.ingest_adapter,
            "auto": entry.ingest_auto,
        })
    return rows
