"""Tests for spatial EM–thermal coupling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, build_scene_at_T  # noqa: E402
from mw_inv.thermal import (  # noqa: E402
    ThermalConfig,
    TransientConfig,
    coupled_steady_state,
    isothermal_baseline,
    simulate_transient,
    solve_steady_heat,
)
from mw_inv.search import (  # noqa: E402
    best_thermal,
    evaluate_thermal_params,
    random_thermal_search,
)


GRID = Grid(nx=51, ny=51, Lx=0.36, Ly=0.36)
PAIR = "pyrite_in_calcite"


def test_steady_heat_raises_temperature_with_source():
    """Uniform volumetric heating + cooling should produce T > T_amb in the interior."""
    k = np.full((GRID.ny, GRID.nx), 3.0)
    Q = np.zeros((GRID.ny, GRID.nx))
    Q[15:35, 15:35] = 1e6
    T = solve_steady_heat(GRID, k, Q, T_amb_K=298.0, bulk_cooling=500.0)
    assert T[25, 25] > 298.0 + 1.0
    assert np.allclose(T[0, :], 298.0)
    assert np.allclose(T[-1, :], 298.0)


def test_build_scene_at_T_spatial_eps():
    """Hotter target pixels should get higher ε″ than ambient."""
    T = np.full((GRID.ny, GRID.nx), 298.0)
    T[20:30, 20:30] = 773.0
    scene = build_scene_at_T(GRID, CavityParams(), PAIR, T)
    t_hot = scene.eps_r[scene.target_mask].imag
    assert t_hot.max() > 0.35


def test_coupled_run_converges():
    cfg = ThermalConfig(drive=10.0, max_iters=30, tol_K=3.0)
    res = coupled_steady_state(GRID, PAIR, config=cfg)
    assert res.thermal.n_iters <= 30
    assert len(res.history_max_delta) >= 1
    assert res.history_max_delta[-1] < 10.0 or res.thermal.converged


def test_coupled_selective_heating():
    """Target should run hotter than gangue under microwave drive."""
    cfg = ThermalConfig(drive=8.0, max_iters=25, tol_K=3.0)
    res = coupled_steady_state(GRID, PAIR, config=cfg)
    assert res.thermal.delta_T_K > 3.0
    assert res.thermal.T_mean_target_K > res.thermal.T_mean_gangue_K
    assert res.thermal.T_mean_target_K < 1200.0  # not saturated everywhere


def test_coupled_em_feedback_changes_selectivity():
    """ε(T) feedback should move EM selectivity vs isothermal baseline."""
    drive = 8.0
    iso, _, _ = isothermal_baseline(GRID, PAIR, drive=drive)
    cfg = ThermalConfig(drive=drive, max_iters=25, tol_K=3.0)
    res = coupled_steady_state(GRID, PAIR, config=cfg)
    assert abs(res.thermal.em_selectivity - iso.selectivity) > 0.003


def test_heat_selectivity_matches_em_in_charge():
    cfg = ThermalConfig(drive=10.0)
    res = coupled_steady_state(GRID, PAIR, config=cfg)
    assert 0.0 < res.thermal.heat_selectivity <= 1.0
    assert abs(res.thermal.heat_selectivity - res.thermal.em_selectivity) < 0.05


def test_transient_target_heats_faster_than_gangue():
    """Target should reach threshold before gangue under microwave drive."""
    cfg = TransientConfig(
        drive=12.0,
        dt_s=0.5,
        t_end_s=60.0,
        em_refresh_s=5.0,
        T_threshold_K=500.0,
    )
    res = simulate_transient(GRID, PAIR, config=cfg)
    assert res.report.target_runaway_first
    assert res.report.t_target_s < res.report.t_gangue_s
    assert res.mean_T_target[-1] > res.mean_T_gangue[-1]


def test_thermal_search_runs():
    """Thermal-coupled search should complete a few trials and return scores."""
    cfg = ThermalConfig(drive=8.0, max_iters=10, tol_K=5.0)
    trials = random_thermal_search(GRID, PAIR, n_trials=2, seed=42, thermal_cfg=cfg)
    assert len(trials) == 2
    best = best_thermal(trials)
    assert best.delta_T_K >= 0.0
    t0 = evaluate_thermal_params(GRID, CavityParams(), PAIR, cfg, "delta_T")
    assert t0.score == t0.delta_T_K
