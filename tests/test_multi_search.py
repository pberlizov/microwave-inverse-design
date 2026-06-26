"""C0 multi-objective search: Pareto pick and hotspot safety filter."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.search import (  # noqa: E402
    MultiTrial,
    optuna_multi_search,
    pareto_recommend,
)


def _trial(
    sel: float,
    coupling: float,
    *,
    arcing: bool = False,
    hotspot_violation: bool = False,
) -> MultiTrial:
    return MultiTrial(
        params={"stub_depth_frac": 0.1},
        selectivity=sel,
        coupling_eff=coupling,
        p_total=1.0,
        contrast=1.0,
        arcing_risk=arcing,
        hotspot_delta_T_K=500.0 if hotspot_violation else 200.0,
        hotspot_violation=hotspot_violation,
    )


def test_pareto_recommend_excludes_hotspot_violations() -> None:
    trials = [
        _trial(0.90, 0.80, hotspot_violation=True),
        _trial(0.70, 0.70, hotspot_violation=False),
    ]
    study = MagicMock()
    study.best_trials = [MagicMock(number=0), MagicMock(number=1)]

    pick = pareto_recommend(
        trials,
        study,
        weight_selectivity=0.5,
        weight_coupling=0.5,
        exclude_arcing=False,
        exclude_hotspot=True,
    )
    assert pick.selectivity == 0.70
    assert not pick.hotspot_violation


def test_optuna_multi_search_records_hotspot_when_enabled() -> None:
    grid = Grid(nx=31, ny=31, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair("pyrite_in_calcite")
    trials, _study = optuna_multi_search(
        grid,
        2,
        seed=11,
        materials=materials,
        check_hotspot=True,
        max_hotspot_delta_T_K=10_000.0,
    )
    assert len(trials) == 2
    assert all(t.hotspot_delta_T_K is not None for t in trials)
    assert all(t.hotspot_delta_T_K >= 0.0 for t in trials)
