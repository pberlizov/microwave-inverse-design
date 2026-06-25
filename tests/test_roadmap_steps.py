"""Tests for validation gate, stress FOM, ore profiles."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402
from mw_inv.ore_profiles import ORE_PROFILES, bruggeman_effective_eps  # noqa: E402
from mw_inv.solver_triangulation import SolverRow  # noqa: E402
from mw_inv.stress import evaluate_stress, grain_size_penalty_factor  # noqa: E402
from mw_inv.validation_gate import GateThresholds, evaluate_gate  # noqa: E402
from mw_inv.openems_export import generate_calibration_script, generate_openems_script  # noqa: E402


GRID = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)


def test_gate_passes_when_fdfd_improves():
    rows = [
        SolverRow("untuned", 0.54),
        SolverRow("tpe_best", 0.67),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.05))
    assert gate.passed
    assert any(c.name == "fdfd_optimised_beats_untuned" and c.passed for c in gate.checks)


def test_stress_evaluation_runs():
    report, _ = evaluate_stress(GRID, CavityParams(), "pyrite_in_calcite")
    assert report.mean_interface_stress_Pa >= 0.0
    assert 0.0 <= report.stress_selectivity <= 1.0


def test_grain_size_penalty_peak():
    assert grain_size_penalty_factor(2.5e-3) > grain_size_penalty_factor(0.05e-3)


def test_bruggeman_between_constituents():
    eps = bruggeman_effective_eps([4.0 + 0.01j, 16.0 + 3.0j], [0.7, 0.3])
    assert 4.0 < eps.real < 16.0


def test_ore_heating_class():
    ore = ORE_PROFILES["disseminated_pyrite_porphyry"]
    assert ore.hmap_wt_percent == 6.0
    assert "III" in ore.heating_class() or "II" in ore.heating_class()


def test_calibration_script_has_calcport():
    text = generate_calibration_script()
    assert "calcPort" in text
    assert "AddLumpedPort" in text


def test_coax_gap_port_in_export():
    text = generate_openems_script(port_mode="coax_gap")
    assert "calcPort" in text
    assert "pin" in text.lower() or "Coax gap" in text
