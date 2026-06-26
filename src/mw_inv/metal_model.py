"""FDFD Dirichlet metal vs openEMS AddMetal alignment (backlog B0/B2).

Compares energy-consistent FDFD ``coupling_eff`` / ``pec_loss_fraction`` with
openEMS matched-port ``coupling_eff`` when a tuning plate is present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.fdfd import Grid, solve_scene
from mw_inv.fom import evaluate as evaluate_fom
from mw_inv.geometry import CavityParams, Materials, build_scene
from mw_inv.openems_postprocess import ingest_openems_case
from mw_inv.validation_gate import GateThresholds


@dataclass(frozen=True)
class MetalModelAlignmentReport:
    """Structural metal model sanity for one cavity design."""

    structure_model: str
    plate_len_frac: float
    fdfd_coupling_eff: float
    fdfd_pec_loss_fraction: float
    openems_coupling_eff: float | None
    coupling_ratio: float | None
    gate_passed: bool | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "structure_model": self.structure_model,
            "plate_len_frac": self.plate_len_frac,
            "fdfd_coupling_eff": self.fdfd_coupling_eff,
            "fdfd_pec_loss_fraction": self.fdfd_pec_loss_fraction,
            "openems_coupling_eff": self.openems_coupling_eff,
            "coupling_ratio": self.coupling_ratio,
            "gate_passed": self.gate_passed,
            "detail": self.detail,
        }


def evaluate_metal_model_alignment(
    grid: Grid,
    params: CavityParams,
    materials: Materials,
    *,
    openems_case_dir: Path | str | None = None,
    gate_thresholds: GateThresholds | None = None,
) -> MetalModelAlignmentReport:
    """FDFD field metrics ± openEMS port coupling for a single design."""
    scene = build_scene(grid, params, materials)
    fom = evaluate_fom(solve_scene(grid, scene), scene)

    oems_c: float | None = None
    ratio: float | None = None
    gate_ok: bool | None = None
    detail = "FDFD only"

    if openems_case_dir is not None:
        metrics = ingest_openems_case(openems_case_dir, params, materials)
        oems_c = metrics.coupling_eff
        if oems_c is not None and fom.coupling_eff > 0:
            ratio = float(oems_c) / float(fom.coupling_eff)
            th = gate_thresholds or GateThresholds()
            gate_ok = (
                th.openems_fdfd_coupling_ratio_min
                <= ratio
                <= th.openems_fdfd_coupling_ratio_max
            )
            detail = (
                f"coupling ratio={ratio:.3f} "
                f"(allowed [{th.openems_fdfd_coupling_ratio_min}, "
                f"{th.openems_fdfd_coupling_ratio_max}])"
            )
        else:
            detail = "openEMS port metrics missing"

    return MetalModelAlignmentReport(
        structure_model=params.structure_model,
        plate_len_frac=params.plate_len_frac,
        fdfd_coupling_eff=float(fom.coupling_eff),
        fdfd_pec_loss_fraction=float(fom.pec_loss_fraction),
        openems_coupling_eff=oems_c,
        coupling_ratio=ratio,
        gate_passed=gate_ok,
        detail=detail,
    )


def write_metal_model_report(path: Path | str, report: MetalModelAlignmentReport) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2))
    return p
