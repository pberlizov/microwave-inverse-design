"""FDFD Dirichlet plate vs openEMS AddMetal triangulation regression."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_export import DesignCase, export_design_bundle  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.metal_model import evaluate_metal_model_alignment  # noqa: E402
from mw_inv.openems_runner import synthesize_port_dumps  # noqa: E402

GRID = Grid(nx=61, ny=61, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")


def test_dirichlet_plate_low_pec_loss():
    params = replace(CavityParams(), plate_len_frac=0.28, structure_model="dirichlet")
    rep = evaluate_metal_model_alignment(GRID, params, MATS)
    assert rep.fdfd_pec_loss_fraction < 0.05
    assert rep.fdfd_coupling_eff > 0.5


def test_plate_openems_coupling_ratio_with_synthetic_dumps(tmp_path: Path):
    params = replace(
        CavityParams(),
        plate_len_frac=0.30,
        plate_angle_deg=45.0,
        structure_model="dirichlet",
    )
    export_dir = tmp_path / "export"
    bundle = export_design_bundle(
        export_dir,
        DesignCase("plate_check", params, "test"),
        MATS,
        grid_n=GRID.nx,
    )
    dump_root = synthesize_port_dumps(export_dir, [bundle])
    case_dir = dump_root / bundle.label

    rep = evaluate_metal_model_alignment(GRID, params, MATS, openems_case_dir=case_dir)
    assert rep.openems_coupling_eff is not None
    assert rep.coupling_ratio is not None
    assert rep.gate_passed is True
    assert 0.35 <= rep.coupling_ratio <= 2.50


def test_triangulate_plate_case_passes_metal_ratio_gate(tmp_path: Path):
    params = replace(CavityParams(), plate_len_frac=0.25, structure_model="dirichlet")
    export_dir = tmp_path / "export"
    bundle = export_design_bundle(
        export_dir,
        DesignCase("untuned", params, "test"),
        MATS,
        grid_n=41,
    )
    dump_root = synthesize_port_dumps(export_dir, [bundle])
    grid = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
    rep = evaluate_metal_model_alignment(
        grid, params, MATS, openems_case_dir=dump_root / bundle.label,
    )
    assert rep.gate_passed is True


def test_lossy_imag_plate_high_structural_fraction_point_feed():
    """Legacy lossy metal absorbs; Dirichlet should not (regression guard)."""
    from mw_inv.fdfd import solve
    from mw_inv.fom import evaluate
    from mw_inv.geometry import build_scene

    lossy = replace(CavityParams(), plate_len_frac=0.25, structure_model="lossy_imag")
    dirichlet = replace(CavityParams(), plate_len_frac=0.25, structure_model="dirichlet")
    sc_l = build_scene(GRID, lossy, MATS)
    sc_d = build_scene(GRID, dirichlet, MATS)
    f_l = evaluate(
        solve(GRID, sc_l.eps_r, sc_l.freq_hz, source_xy=sc_l.source_xy, mu_r=sc_l.mu_r),
        sc_l,
    )
    f_d = evaluate(
        solve(
            GRID, sc_d.eps_r, sc_d.freq_hz, source_xy=sc_d.source_xy, mu_r=sc_d.mu_r,
            dirichlet_mask=sc_d.pec_mask,
        ),
        sc_d,
    )
    assert f_l.pec_loss_fraction > f_d.pec_loss_fraction


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
