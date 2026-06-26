"""Cross-solver triangulation: FDFD vs MEEP vs openEMS dumps.

Used to verify that geometry-driven selectivity trends survive when moving from the
cheap 2D FDFD core to FDTD reference solvers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mw_inv.design_export import DesignCase, load_search_cases
from mw_inv.fdfd import Grid, solve_scene
from mw_inv.fom import evaluate
from mw_inv.geometry import Materials, build_scene
from mw_inv.meep_3d import compare_fdfd_meep_3d
from mw_inv.meep_compare import compare_fdfd_meep, meep_available
from mw_inv.openems_postprocess import h5py_available, ingest_openems_case


@dataclass
class SolverRow:
    label: str
    fdfd_selectivity: float
    fdfd_coupling_eff: float | None = None
    fdfd_pec_loss_fraction: float | None = None
    meep_2d_selectivity: float | None = None
    meep_3d_primitive_selectivity: float | None = None
    openems_selectivity: float | None = None
    openems_s11_mag: float | None = None
    openems_coupling_eff: float | None = None
    rel_err_meep_2d: float | None = None
    rel_err_meep_3d: float | None = None
    rel_err_openems: float | None = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "fdfd_selectivity": self.fdfd_selectivity,
            "fdfd_coupling_eff": self.fdfd_coupling_eff,
            "fdfd_pec_loss_fraction": self.fdfd_pec_loss_fraction,
            "meep_2d_selectivity": self.meep_2d_selectivity,
            "meep_3d_primitive_selectivity": self.meep_3d_primitive_selectivity,
            "openems_selectivity": self.openems_selectivity,
            "openems_s11_mag": self.openems_s11_mag,
            "openems_coupling_eff": self.openems_coupling_eff,
            "rel_err_meep_2d": self.rel_err_meep_2d,
            "rel_err_meep_3d": self.rel_err_meep_3d,
            "rel_err_openems": self.rel_err_openems,
            "params": self.params,
        }


def _rel_err(ref: float, val: float | None) -> float | None:
    if val is None:
        return None
    return abs(val - ref) / max(ref, 1e-6)


def triangulate_case(
    case: DesignCase,
    grid: Grid,
    materials: Materials,
    *,
    Lz: float = 0.36,
    openems_case_dir: Path | None = None,
) -> SolverRow:
    """FDFD reference + optional MEEP / openEMS for one design."""
    scene = build_scene(grid, case.params, materials)
    res = solve_scene(grid, scene)
    fom = evaluate(res, scene)
    fdfd_sel = float(fom.selectivity)

    row = SolverRow(
        label=case.label,
        fdfd_selectivity=fdfd_sel,
        fdfd_coupling_eff=float(fom.coupling_eff),
        fdfd_pec_loss_fraction=float(fom.pec_loss_fraction),
        params={
            k: getattr(case.params, k)
            for k in case.params.__dataclass_fields__
            if k != "tuner_field"
        },
    )

    if meep_available():
        cmp2 = compare_fdfd_meep(scene, grid, fdfd_sel)
        if not cmp2.get("skipped"):
            row.meep_2d_selectivity = float(cmp2["meep_2d_selectivity"])
            row.rel_err_meep_2d = _rel_err(fdfd_sel, row.meep_2d_selectivity)
        cmp3 = compare_fdfd_meep_3d(scene, grid, fdfd_sel, materials=materials, Lz=Lz)
        if not cmp3.get("skipped"):
            row.meep_3d_primitive_selectivity = float(cmp3["meep_3d_primitive_selectivity"])
            row.rel_err_meep_3d = _rel_err(fdfd_sel, row.meep_3d_primitive_selectivity)

    if openems_case_dir is not None and openems_case_dir.is_dir():
        metrics = ingest_openems_case(openems_case_dir, case.params, materials, Lz=Lz)
        row.openems_selectivity = metrics.selectivity
        row.openems_s11_mag = metrics.s11_mag
        row.openems_coupling_eff = metrics.coupling_eff
        if metrics.selectivity is not None:
            row.rel_err_openems = _rel_err(fdfd_sel, metrics.selectivity)

    return row


def triangulate_cases(
    cases: list[DesignCase],
    grid: Grid,
    materials: Materials,
    *,
    Lz: float = 0.36,
    openems_dump_dir: Path | None = None,
) -> list[SolverRow]:
    rows: list[SolverRow] = []
    for case in cases:
        case_dir: Path | None = None
        if openems_dump_dir is not None:
            candidate = openems_dump_dir / case.label
            if candidate.is_dir():
                case_dir = candidate
        rows.append(triangulate_case(case, grid, materials, Lz=Lz, openems_case_dir=case_dir))
    return rows


def rank_agreement(rows: list[SolverRow]) -> dict:
    """Check whether solver rankings match FDFD (untuned vs optimised)."""
    if len(rows) < 2:
        return {"n_cases": len(rows)}

    fdfd_order = sorted(range(len(rows)), key=lambda i: rows[i].fdfd_selectivity, reverse=True)
    out: dict = {"n_cases": len(rows), "fdfd_rank_order": fdfd_order}
    for attr in ("meep_2d_selectivity", "meep_3d_primitive_selectivity", "openems_selectivity"):
        if any(getattr(r, attr) is None for r in rows):
            out[f"{attr}_rank_order"] = None
            out[f"{attr}_rankings_match_fdfd"] = None
        else:
            order = sorted(
                range(len(rows)),
                key=lambda i, a=attr: getattr(rows[i], a),
                reverse=True,
            )
            out[f"{attr}_rank_order"] = order
            out[f"{attr}_rankings_match_fdfd"] = order == fdfd_order
    return out


def write_triangulation_report(
    path: Path | str,
    rows: list[SolverRow],
    *,
    materials_label: str,
    meta: dict | None = None,
) -> Path:
    path = Path(path)
    payload = {
        "materials": materials_label,
        "meep_available": meep_available(),
        "h5py_available": h5py_available(),
        "rows": [r.to_dict() for r in rows],
        "rank_agreement": rank_agreement(rows),
        **(meta or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def triangulate_from_search(
    search_path: Path | str,
    grid: Grid,
    materials: Materials,
    *,
    Lz: float = 0.36,
    openems_dump_dir: Path | None = None,
    top_k: int | None = None,
) -> list[SolverRow]:
    return triangulate_cases(
        load_search_cases(search_path, top_k=top_k),
        grid,
        materials,
        Lz=Lz,
        openems_dump_dir=openems_dump_dir,
    )
