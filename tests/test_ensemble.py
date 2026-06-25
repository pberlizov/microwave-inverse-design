"""Tests for ensemble ore layouts and multi-objective search (steps 5–6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.ensemble import evaluate_ensemble  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, sample_inclusion_offsets  # noqa: E402
from mw_inv.materials import Materials  # noqa: E402
from mw_inv.search import (  # noqa: E402
    evaluate_params,
    evaluate_robust_params,
    optuna_multi_search,
    optuna_robust_search,
)


GRID = Grid(nx=51, ny=51, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")


def test_sample_inclusion_offsets_non_overlapping():
    p = CavityParams()
    rng = np.random.default_rng(0)
    offsets = sample_inclusion_offsets(p, 5, rng)
    assert len(offsets) >= 3
    r = p.inclusion_radius_frac
    min_sep = 2.2 * r
    for i, (x0, y0) in enumerate(offsets):
        for x1, y1 in offsets[i + 1:]:
            d = (x0 - x1) ** 2 + (y0 - y1) ** 2
            assert d >= min_sep ** 2 - 1e-12


def test_ensemble_spreads_selectivity():
    rep = evaluate_ensemble(GRID, CavityParams(), MATS, n_realizations=6, n_grains=5, seed=1)
    assert rep.n_realizations == 6
    assert rep.std_selectivity >= 0.0
    assert rep.max_selectivity >= rep.min_selectivity


def test_robust_search_runs():
    t = evaluate_robust_params(GRID, CavityParams(), MATS, n_realizations=3, n_grains=4, seed=0)
    assert 0.0 <= t.mean_selectivity <= 1.0
    trials = optuna_robust_search(GRID, n_trials=2, seed=0, materials=MATS, n_realizations=2, n_grains=3)
    assert len(trials) == 2


def test_multi_search_returns_pareto():
    trials, study = optuna_multi_search(GRID, n_trials=6, seed=0, materials=MATS)
    assert len(trials) == 6
    assert len(study.directions) == 2
    base = evaluate_params(GRID, CavityParams(), MATS)
    assert base.p_total > 0.0


def test_frequency_robust_spreads_with_freq():
    from mw_inv.ensemble import evaluate_frequency_robust

    rep = evaluate_frequency_robust(GRID, CavityParams(), MATS, pair_label="pyrite_in_calcite", n_freqs=5)
    assert rep.n_freqs == 5
    assert rep.min_selectivity <= rep.mean_selectivity + 1e-9


def test_freq_robust_search_runs():
    from mw_inv.search import evaluate_freq_robust_params, optuna_freq_robust_search

    t = evaluate_freq_robust_params(GRID, CavityParams(), MATS, pair_label="pyrite_in_calcite", n_freqs=3)
    assert t.min_selectivity <= t.mean_selectivity + 1e-9
    trials = optuna_freq_robust_search(GRID, 2, 0, MATS, pair_label="pyrite_in_calcite", n_freqs=3)
    assert len(trials) == 2
    assert "freq_hz" not in trials[0].params


def test_thermal_ensemble_runs():
    from mw_inv.search import evaluate_thermal_ensemble_params, optuna_thermal_ensemble_search
    from mw_inv.thermal import ThermalConfig, thermal_props_for_pair

    cfg = ThermalConfig(drive=8.0, thermal_props=thermal_props_for_pair("pyrite_in_calcite"), max_iters=8, tol_K=5.0)
    t = evaluate_thermal_ensemble_params(
        GRID, CavityParams(), "pyrite_in_calcite",
        n_realizations=2, n_grains=3, seed=0, thermal_cfg=cfg,
    )
    assert t.mean_delta_T_K >= 0.0
    trials = optuna_thermal_ensemble_search(
        GRID, "pyrite_in_calcite", 2, 0,
        n_realizations=2, n_grains=3, thermal_cfg=cfg,
    )
    assert len(trials) == 2
