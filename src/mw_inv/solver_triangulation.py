"""Cross-solver triangulation: FDFD vs MEEP vs openEMS dumps.

Used to verify that geometry-driven selectivity trends survive when moving from the
cheap 2D FDFD core to FDTD reference solvers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mw_inv.design_export import DesignCase, fdfd_selectivity, load_search_cases
from mw_inv.fdfd import Grid
from mw_inv.geometry import Materials, build_scene
from mw_inv.meep_3d import compare_fdfd_meep_3d
from mw_inv.meep_compare import compare_fdfd_meep, meep_available
from mw_inv.openems_postprocess import h5py_available, selectivity_from_openems_dump


@dataclass
class SolverRow:
    label: str
    fdfd_selectivity: float
    meep_2d_selectivity: float | None = None
    meep_3d_primitive_selectivity: float | None = None
    openems_selectivity: float | None = None
    rel_err_meep_2d: float | None = None
    rel_err_meep_3d: float | None = None
    rel_err_openems: float | None = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "fdfd_selectivity": self.fdfd_selectivity,
            "meep_2d_selectivity": self.meep_2d_selectivity,
            "meep_3d_primitive_selectivity": self.meep_3d_primitive_selectivity,
            "openems_selectivity": self.openems_selectivity,
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
    openems_dump: Path | None = None,
) -> SolverRow:
    """FDFD reference + optional MEEP / openEMS for one design."""
    scene = build_scene(grid, case.params, materials)
    fdfd_sel = fdfd_selectivity(grid, case.params, materials)

    row = SolverRow(
        label=case.label,
        fdfd_selectivity=fdfd_sel,
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

    if openems_dump is not None and openems_dump.is_file():
        row.openems_selectivity = selectivity_from_openems_dump(
            openems_dump, case.params, materials, Lz=Lz,
        )
        row.rel_err_openems = _rel_err(fdfd_sel, row.openems_selectivity)

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
        dump = None
        if openems_dump_dir is not None:
            for name in (f"{case.label}_Et_0000.h5", f"Et_{case.label}_0000.h5", "Et_0000.h5", "Et/Et_0000.h5"):
                p = openems_dump_dir / case.label / name
                if p.is_file():
                    dump = p
                    break
                p2 = openems_dump_dir / name
                if p2.is_file() and len(cases) == 1:
                    dump = p2
                    break
        rows.append(triangulate_case(case, grid, materials, Lz=Lz, openems_dump=dump))
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
