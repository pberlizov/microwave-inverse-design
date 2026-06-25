"""Tests for unified DesignEvaluator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_evaluator import (  # noqa: E402
    DesignEvaluator,
    evaluate_design,
    preset_config,
)
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.search import evaluate_params, evaluate_stress_params, evaluate_thermal_params  # noqa: E402

GRID = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
PAIR = "pyrite_in_calcite"
MATS = Materials.from_pair(PAIR)


def test_em_preset_matches_legacy_evaluate_params():
    params = CavityParams()
    legacy = evaluate_params(GRID, params, MATS)
    cfg = preset_config("em", materials=MATS)
    rep = evaluate_design(GRID, params, cfg)
    assert abs(rep.em_selectivity - legacy.selectivity) < 1e-9
    assert abs(rep.em_contrast - legacy.contrast) < 1e-9


def test_thermal_preset_matches_legacy():
    params = CavityParams()
    legacy = evaluate_thermal_params(GRID, params, PAIR, objective="delta_T")
    cfg = preset_config("thermal:delta_T", pair_label=PAIR)
    rep = evaluate_design(GRID, params, cfg)
    assert abs(rep.score - legacy.score) < 1e-6
    assert rep.delta_T_K is not None


def test_stress_preset_matches_legacy():
    params = CavityParams()
    legacy = evaluate_stress_params(GRID, params, PAIR, objective="stress_score")
    cfg = preset_config("stress:stress_score", pair_label=PAIR)
    rep = evaluate_design(GRID, params, cfg)
    assert abs(rep.score - legacy.score) < 1.0
    assert rep.stress_score is not None


def test_composite_liberation_runs_all_foms():
    ev = DesignEvaluator.from_preset(GRID, "composite:liberation", pair_label=PAIR)
    rep = ev.evaluate(CavityParams())
    assert rep.delta_T_K is not None
    assert rep.stress_score is not None
    assert rep.objective_key == "composite"
    assert 0.0 < rep.score <= 1.5


def test_arcing_penalty_reduces_score():
    cfg = preset_config("em", materials=MATS, check_arcing=True)
    rep = evaluate_design(GRID, CavityParams(), cfg)
    if rep.arcing_risk:
        assert rep.arcing_penalty_applied
        assert rep.score <= rep.em_selectivity


def test_report_to_dict_roundtrip():
    rep = DesignEvaluator.from_preset(GRID, "em", materials=MATS).evaluate(CavityParams())
    d = rep.to_dict()
    assert d["em_selectivity"] == rep.em_selectivity
    assert "params" in d
