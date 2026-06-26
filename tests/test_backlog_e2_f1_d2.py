"""Tests for E2 hotspot gate, F1 openEMS schedule, D2 particle tail."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_export import DesignCase  # noqa: E402
from mw_inv.ensemble import evaluate_particle_power  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.hotspot_gate import evaluate_hotspot_gate  # noqa: E402
from mw_inv.openems_schedule import schedule_openems_cases  # noqa: E402
from mw_inv.particle_tail_gate import evaluate_particle_tail_gate  # noqa: E402

GRID = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")


def test_hotspot_gate_reports_evolved_and_frozen():
    rep = evaluate_hotspot_gate(
        GRID,
        replace(CavityParams(), plate_len_frac=0.3),
        "pyrite_in_calcite",
        max_hotspot_delta_T_K=2000.0,
    )
    assert rep.evolved_delta_T_K >= 0.0
    assert rep.frozen_delta_T_K >= 0.0
    assert rep.uses_evolved_for_gate


def test_openems_schedule_skips_winners_when_gate_fails():
    cases = [
        DesignCase("untuned", CavityParams(), "base"),
        DesignCase("tpe_best", CavityParams(), "tpe"),
        DesignCase("tpe_k1", CavityParams(), "k1"),
    ]
    scheduled, meta = schedule_openems_cases(cases, gate_passed=False, budget=3)
    assert [c.label for c in scheduled] == ["untuned"]
    assert meta["reason"] == "fdfd_gate_failed_diagnostic_only"


def test_openems_schedule_winners_only_when_gate_passes():
    cases = [
        DesignCase("untuned", CavityParams(), "base"),
        DesignCase("tpe_best", CavityParams(), "tpe"),
        DesignCase("tpe_k1", CavityParams(), "k1"),
    ]
    scheduled, meta = schedule_openems_cases(cases, gate_passed=True, budget=2)
    assert "untuned" not in [c.label for c in scheduled]
    assert len(scheduled) == 2
    assert scheduled[0].label == "tpe_best"


def test_particle_power_tail_percentiles():
    pp = evaluate_particle_power(GRID, CavityParams(), MATS)
    assert pp.p05_particle_fraction <= pp.p95_particle_fraction + 1e-12
    assert pp.n_particles >= 1


def test_particle_tail_gate_floor():
    ok = evaluate_particle_tail_gate(
        {"p05_particle_fraction": 0.08, "gangue_power_fraction": 0.3},
        min_p05_particle_fraction=0.05,
        max_gangue_power_fraction=0.85,
    )
    assert ok.passed
    bad = evaluate_particle_tail_gate(
        {"p05_particle_fraction": 0.02, "gangue_power_fraction": 0.3},
        min_p05_particle_fraction=0.05,
    )
    assert not bad.passed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
