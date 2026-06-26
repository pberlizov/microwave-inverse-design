"""Tests for phase transitions, uncertainty gate, tuning procedure, multi industrial."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.dielectric_data import mineral_eps  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, build_scene_at_T  # noqa: E402
from mw_inv.phase_transitions import mineral_key_at_T  # noqa: E402
from mw_inv.search import MultiTrial, pareto_recommend  # noqa: E402
from mw_inv.tuning_procedure import build_tuning_procedure  # noqa: E402
from mw_inv.uncertainty_gate import evaluate_uncertainty_gate  # noqa: E402


def test_pyrite_to_pyrrhotite_at_high_T():
    assert mineral_key_at_T("pyrite", 298.0) == "pyrite"
    assert mineral_key_at_T("pyrite", 700.0) == "pyrrhotite"


def test_build_scene_at_T_phase_transition_changes_eps():
    grid = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
    params = CavityParams()
    T_cold = np.full((grid.ny, grid.nx), 298.0)
    T_hot = np.full((grid.ny, grid.nx), 750.0)
    sc_cold = build_scene_at_T(grid, params, "pyrite_in_calcite", T_cold)
    sc_hot = build_scene_at_T(grid, params, "pyrite_in_calcite", T_hot)
    eps_c = sc_cold.eps_r[sc_cold.target_mask].mean()
    eps_h = sc_hot.eps_r[sc_hot.target_mask].mean()
    ref_pyrr = mineral_eps("pyrrhotite", 750.0, params.freq_hz)
    assert abs(eps_h - ref_pyrr) < abs(eps_c - ref_pyrr)


def test_uncertainty_gate_material_p05():
    block = {
        "mode": "material",
        "best_material": {"p05_selectivity": 0.42, "p05_coupling_eff": 0.8},
        "untuned_material": {"p05_selectivity": 0.38},
    }
    ok = evaluate_uncertainty_gate(block, min_p05_selectivity=0.35)
    assert ok.passed
    bad = evaluate_uncertainty_gate(block, min_p05_selectivity=0.45)
    assert not bad.passed


def test_tuning_procedure_steps():
    proc = build_tuning_procedure(CavityParams(), label="test")
    assert len(proc.steps) >= 4
    assert "2.45" in proc.to_markdown()


def test_pareto_recommend_industrial_weights():
    from types import SimpleNamespace

    class FakeStudy:
        best_trials = [SimpleNamespace(number=0), SimpleNamespace(number=1)]

    trials = [
        MultiTrial({}, 0.9, 0.5, 1.0, 1.0, gangue_power_fraction=0.6, min_particle_fraction=0.05),
        MultiTrial({}, 0.7, 0.9, 1.0, 1.0, gangue_power_fraction=0.2, min_particle_fraction=0.15),
    ]
    pick = pareto_recommend(
        trials,
        FakeStudy(),
        weight_selectivity=0.2,
        weight_coupling=0.2,
        weight_gangue_budget=0.3,
        weight_particle_floor=0.3,
    )
    assert pick.gangue_power_fraction == 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
