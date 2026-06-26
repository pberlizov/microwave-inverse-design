"""Shared triangulation + gate + promotion refresh for pipeline and openEMS ingest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.fdfd import Grid
from mw_inv.geometry import Materials
from mw_inv.promotion import PromotionAssessment
from mw_inv.run_manifest import RunManifest, finalize_promotion
from mw_inv.solver_triangulation import SolverRow, triangulate_from_search, write_triangulation_report
from mw_inv.validation_gate import GateThresholds, ValidationGateReport, evaluate_gate


@dataclass(frozen=True)
class TriangulationRefreshResult:
    rows: list[SolverRow]
    gate: ValidationGateReport
    assessment: PromotionAssessment


def apply_triangulation_refresh(
    manifest: RunManifest,
    run_dir: Path,
    *,
    search_path: Path,
    grid: Grid,
    materials: Materials,
    materials_label: str,
    openems_dump_dir: Path | None = None,
    Lz: float = 0.36,
    top_k: int | None = None,
    gate_thresholds: GateThresholds | None = None,
    triangulation_meta: dict | None = None,
) -> TriangulationRefreshResult:
    """Recompute triangulation, gate, and promotion tier on a run manifest."""
    rows = triangulate_from_search(
        search_path,
        grid,
        materials,
        Lz=Lz,
        openems_dump_dir=openems_dump_dir,
        top_k=top_k,
    )
    tri_path = run_dir / "solver_triangulation.json"
    meta = {
        "search_source": str(search_path),
        "openems_dump_dir": str(openems_dump_dir) if openems_dump_dir else None,
        **(triangulation_meta or {}),
    }
    write_triangulation_report(
        tri_path,
        rows,
        materials_label=materials_label,
        meta=meta,
    )

    gate = evaluate_gate(rows, gate_thresholds)
    gate_path = run_dir / "validation_gate_report.json"
    gate_payload = {
        "materials": materials_label,
        "search_source": str(search_path),
        "openems_dump_dir": str(openems_dump_dir) if openems_dump_dir else None,
        "gate": gate.to_dict(),
        "triangulation": [r.to_dict() for r in rows],
    }
    gate_path.write_text(json.dumps(gate_payload, indent=2))

    manifest.triangulation_path = str(tri_path)
    manifest.triangulation = {
        "rows": [r.to_dict() for r in rows],
        "rank_agreement": gate.rank_agreement,
        "openems_dump_dir": str(openems_dump_dir) if openems_dump_dir else None,
    }
    manifest.gate_path = str(gate_path)
    manifest.gate = gate.to_dict()
    manifest.search_path = str(search_path)

    assessment = finalize_promotion(manifest)
    return TriangulationRefreshResult(rows=rows, gate=gate, assessment=assessment)
