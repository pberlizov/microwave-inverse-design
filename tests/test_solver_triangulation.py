"""Tests for openEMS post-process and solver triangulation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.openems_postprocess import selectivity_from_e2  # noqa: E402
from mw_inv.solver_triangulation import SolverRow, rank_agreement, triangulate_case  # noqa: E402
from mw_inv.design_export import DesignCase  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402


def test_selectivity_from_e2_synthetic():
    params = CavityParams()
    mats = Materials.from_pair("pyrite_in_calcite")
    shape = (21, 21, 11)
    e2 = np.zeros(shape)
    gangue, target = __import__(
        "mw_inv.openems_postprocess", fromlist=["_charge_masks"]
    )._charge_masks(params, shape)
    e2[target] = 10.0
    e2[gangue & ~target] = 1.0
    sel = selectivity_from_e2(e2, params, mats, params.freq_hz)
    assert sel > 0.85


def test_triangulate_case_fdfd_only():
    grid = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
    mats = Materials.from_pair("pyrite_in_calcite")
    case = DesignCase("untuned", CavityParams(), "test")
    row = triangulate_case(case, grid, mats)
    assert 0.0 < row.fdfd_selectivity <= 1.0


def test_rank_agreement():
    rows = [
        SolverRow("a", 0.6),
        SolverRow("b", 0.8),
        SolverRow("c", 0.7),
    ]
    rows[0].meep_2d_selectivity = 0.55
    rows[1].meep_2d_selectivity = 0.82
    rows[2].meep_2d_selectivity = 0.71
    agg = rank_agreement(rows)
    assert agg["fdfd_rank_order"] == [1, 2, 0]
    assert agg["meep_2d_selectivity_rankings_match_fdfd"] is True
