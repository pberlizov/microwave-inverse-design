"""Tests for manufacturable applicator geometry (step 4)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.fom import evaluate  # noqa: E402
from mw_inv.geometry import CavityParams, build_scene, resolve_feed  # noqa: E402
from mw_inv.materials import Materials  # noqa: E402
from mw_inv.search import (  # noqa: E402
    evaluate_params,
    optuna_search,
    random_search,
)


GRID = Grid(nx=61, ny=61, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")


def _selectivity(params: CavityParams) -> float:
    sc = build_scene(GRID, params, MATS)
    from mw_inv.fdfd import solve

    rep = evaluate(solve(GRID, sc.eps_r, sc.freq_hz, sc.source_xy, mu_r=sc.mu_r), sc)
    return rep.selectivity


def test_default_feed_matches_legacy_position():
    p = CavityParams()
    x, y = resolve_feed(p, GRID)
    assert abs(x - 0.5 * GRID.Lx) < 1e-9
    assert abs(y - 0.08 * GRID.Ly) < 1e-9


def test_feed_wall_left_places_source_on_wall():
    p = replace(CavityParams(), feed_wall="left", feed_along_frac=0.5, stub_depth_frac=0.06)
    x, y = resolve_feed(p, GRID)
    assert abs(x - 0.06 * GRID.Lx) < 1e-9
    assert abs(y - 0.5 * GRID.Ly) < 1e-9


def test_movable_plate_rasterizes():
    p = replace(CavityParams(), plate_len_frac=0.25, plate_angle_deg=90.0)
    scene = build_scene(GRID, p, MATS)
    assert scene.pec_mask.any()


def test_bed_position_moves_inclusions():
    p0 = CavityParams()
    p1 = replace(CavityParams(), charge_cy_frac=0.50)
    s0 = build_scene(GRID, p0, MATS)
    s1 = build_scene(GRID, p1, MATS)
    y0 = np.where(s0.target_mask)[0].mean()
    y1 = np.where(s1.target_mask)[0].mean()
    assert y1 < y0


def test_plate_changes_selectivity():
    base = _selectivity(CavityParams())
    tuned = _selectivity(replace(CavityParams(), plate_len_frac=0.35, plate_cx_frac=0.35))
    assert abs(tuned - base) > 1e-4


def test_manufacturable_search_runs():
    rnd = random_search(GRID, n_trials=3, seed=0, materials=MATS)
    assert len(rnd) == 3
    assert "feed_wall" in rnd[0].params
    tpe = optuna_search(GRID, n_trials=3, seed=0, materials=MATS)
    assert len(tpe) == 3


def test_legacy_search_still_works():
    base = evaluate_params(GRID, CavityParams(), MATS, legacy=True)
    assert 0.0 < base.selectivity <= 1.0
    trials = random_search(GRID, n_trials=2, seed=1, materials=MATS, legacy=True)
    assert "feed_x_frac" in trials[0].params
    assert "feed_wall" not in trials[0].params
