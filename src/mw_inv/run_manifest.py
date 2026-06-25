"""Run manifest — single JSON artifact per pipeline execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mw_inv.promotion import PromotionAssessment, assess_promotion


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class RunManifest:
    run_id: str
    materials: str
    created_at: str = field(default_factory=_utc_now)
    preset: str = "em"
    search_path: str | None = None
    search_summary: dict[str, Any] = field(default_factory=dict)
    benchmarks_path: str | None = None
    benchmarks_passed: bool | None = None
    gate_path: str | None = None
    gate: dict[str, Any] = field(default_factory=dict)
    triangulation_path: str | None = None
    triangulation: dict[str, Any] = field(default_factory=dict)
    export_dir: str | None = None
    export_summary: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    bench: dict[str, Any] = field(default_factory=dict)
    ore: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "materials": self.materials,
            "preset": self.preset,
            "search": {
                "path": self.search_path,
                "summary": self.search_summary,
            },
            "benchmarks": {
                "path": self.benchmarks_path,
                "passed": self.benchmarks_passed,
            },
            "gate": {
                "path": self.gate_path,
                **self.gate,
            },
            "triangulation": {
                "path": self.triangulation_path,
                **self.triangulation,
            },
            "export": {
                "dir": self.export_dir,
                **self.export_summary,
            },
            "evaluation": self.evaluation,
            "bench": self.bench,
            "ore": self.ore,
            "promotion": self.promotion,
            "notes": self.notes,
        }

    def write(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> RunManifest:
        data = json.loads(Path(path).read_text())
        m = cls(
            run_id=data["run_id"],
            materials=data["materials"],
            created_at=data.get("created_at", _utc_now()),
            preset=data.get("preset", "em"),
        )
        search = data.get("search", {})
        m.search_path = search.get("path")
        m.search_summary = search.get("summary", {})
        bench = data.get("benchmarks", {})
        m.benchmarks_path = bench.get("path")
        m.benchmarks_passed = bench.get("passed")
        gate = data.get("gate", {})
        m.gate_path = gate.get("path")
        m.gate = {k: v for k, v in gate.items() if k != "path"}
        tri = data.get("triangulation", {})
        m.triangulation_path = tri.get("path")
        m.triangulation = {k: v for k, v in tri.items() if k != "path"}
        export = data.get("export", {})
        m.export_dir = export.get("dir")
        m.export_summary = {k: v for k, v in export.items() if k != "dir"}
        m.evaluation = data.get("evaluation", {})
        m.bench = data.get("bench", {})
        m.ore = data.get("ore", {})
        m.promotion = data.get("promotion", {})
        m.notes = list(data.get("notes", []))
        return m


def finalize_promotion(manifest: RunManifest) -> PromotionAssessment:
    """Compute promotion tier from manifest fields and attach to manifest."""
    from mw_inv.promotion import _gate_from_dict, _rows_from_dict
    from mw_inv.validation_gate import ValidationGateReport

    gate_obj: ValidationGateReport | None = None
    if manifest.gate.get("passed") is not None:
        gate_obj = _gate_from_dict(manifest.gate)

    rows = _rows_from_dict(manifest.triangulation) if manifest.triangulation else None

    assessment = assess_promotion(
        benchmarks_passed=manifest.benchmarks_passed,
        gate=gate_obj,
        triangulation_rows=rows,
        phantom_label=manifest.bench.get("phantom_label"),
        measured_eps_path=manifest.bench.get("measured_eps_path"),
    )
    manifest.promotion = assessment.to_dict()
    return assessment


def default_run_dir(base: Path | str = "data/runs") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(base) / stamp
