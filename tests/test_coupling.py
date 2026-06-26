"""Coupling efficiency metrics (backlog A0).

Selectivity is not actionable on its own: a design can post a high target *fraction*
while coupling almost no power into the charge. These tests pin the energy-consistent
coupling metric and the floor penalty that stops the search exploiting it.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_evaluator import EvaluationConfig, evaluate_design  # noqa: E402
from mw_inv.fdfd import Grid, solve, solve_scene  # noqa: E402
from mw_inv.fom import evaluate  # noqa: E402
from mw_inv.geometry import CavityParams, Materials, build_scene  # noqa: E402

GRID = Grid(nx=61, ny=61, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")


def _fom(params: CavityParams):
    sc = build_scene(GRID, params, MATS)
    return evaluate(solve_scene(GRID, sc), sc)


def _fom_point(params: CavityParams):
    """Legacy point feed — still used to regression-test fake-PEC coupling pathology."""
    sc = build_scene(GRID, params, MATS)
    return evaluate(
        solve(GRID, sc.eps_r, sc.freq_hz, source_xy=sc.source_xy, mu_r=sc.mu_r),
        sc,
    )


def test_coupling_metrics_invariants():
    f = _fom(CavityParams())
    assert 0.0 <= f.coupling_eff <= 1.0
    assert 0.0 <= f.pec_loss_fraction <= 1.0
    # Energy split is consistent: total = charge + structural.
    assert f.p_abs_total == pytest.approx(f.p_total_charge + f.p_structural, rel=1e-9)


def test_no_pec_design_couples_fully():
    """With no internal PEC structure, all absorbed power lands in the charge."""
    f = _fom(CavityParams())
    assert f.coupling_eff > 0.99
    assert f.pec_loss_fraction < 0.01


def test_baffle_exposes_coupling_pathology():
    """The Im(eps)=1e6 'PEC' baffle is a strong *absorber*, not a lossless reflector:
    it can raise selectivity while dumping nearly all power into structure. Coupling
    efficiency surfaces this where selectivity alone hides it.

    Line-port excitation at the stub mouth does not re-create this trap; the pathology
    is pinned with the legacy point feed that previously drove the search stack.
    """
    base = _fom_point(CavityParams())
    baffled = _fom_point(replace(CavityParams(), baffle_len_frac=0.4, structure_model="lossy_imag"))
    assert baffled.coupling_eff < 0.1            # almost no power reaches the charge
    assert baffled.pec_loss_fraction > 0.9       # ... it is dumped in the "PEC" baffle
    assert baffled.p_total_charge < base.p_total_charge * 1e-3
    # And the trap: selectivity does NOT warn you.
    assert baffled.selectivity >= base.selectivity - 0.05


def test_coupling_floor_penalises_low_coupling_designs():
    cfg = EvaluationConfig(materials=MATS, mode="em", objective="em_selectivity",
                           coupling_floor=0.5)
    good = evaluate_design(GRID, CavityParams(), cfg)
    assert not good.coupling_floor_applied

    # Production line-port path: verify the fake-PEC trap still collapses coupling
    # under point excitation (metric regression guard).
    bad_params = replace(CavityParams(), baffle_len_frac=0.4, structure_model="lossy_imag")
    assert _fom_point(bad_params).coupling_eff < cfg.coupling_floor

    # Search stack uses line-port; lossy-imag baffle still reduces coupling below unity.
    weak_coupling = replace(
        CavityParams(), baffle_len_frac=0.4, structure_model="lossy_imag",
    )
    strict = EvaluationConfig(
        materials=MATS, mode="em", objective="em_selectivity", coupling_floor=0.999,
        coupling_score_factor=cfg.coupling_score_factor,
    )
    bad = evaluate_design(GRID, weak_coupling, strict)
    assert bad.coupling_eff < strict.coupling_floor
    assert bad.coupling_floor_applied
    # Penalised score is a fraction of the raw objective.
    assert bad.score == pytest.approx(bad.em_selectivity * cfg.coupling_score_factor)
    assert bad.score < evaluate_design(GRID, CavityParams(), cfg).score


def test_dirichlet_plate_low_structural_absorption():
    """True PEC (Ez=0) should not dump power into Im(eps) structural loss."""
    base = _fom(CavityParams())
    plated = _fom(replace(CavityParams(), plate_len_frac=0.25, structure_model="dirichlet"))
    assert plated.pec_loss_fraction < 0.05
    assert plated.coupling_eff > 0.3
    # Plate may redirect field but should not create fake selectivity via absorber.
    assert plated.p_structural < base.p_abs_total * 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
