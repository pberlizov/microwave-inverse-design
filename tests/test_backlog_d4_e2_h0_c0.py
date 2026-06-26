"""Tests for D4 changelog, E2 transient evolve, H0 manufacturing, C0 industrial Pareto."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.deposit_calibration import (  # noqa: E402
    DepositCalibrationReport,
    PhaseCalibrationDiff,
    calibrate_ore_profile,
    diff_calibration_reports,
)
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402
from mw_inv.manufacturing_tolerance import (  # noqa: E402
    evaluate_manufacturing_robust,
    jitter_cavity_params,
)
from mw_inv.promotion import PromotionTier, assess_promotion  # noqa: E402
from mw_inv.search import optuna_multi_search, pareto_recommend  # noqa: E402
from mw_inv.thermal import TransientConfig, simulate_transient  # noqa: E402
from mw_inv.validation_gate import GateCheck, ValidationGateReport  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
GRID = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)


def _sample_calibration_report(label: str = "test") -> DepositCalibrationReport:
    phase = PhaseCalibrationDiff(
        phase="target",
        mineral="pyrite",
        predicted_eps_real=10.0,
        predicted_eps_imag=0.5,
        measured_eps_real=9.5,
        measured_eps_imag=0.48,
        delta_eps_real=0.5,
        delta_eps_imag=0.02,
        rel_error_real=0.05,
        rel_error_imag=0.04,
    )
    return DepositCalibrationReport(
        ore_label=label,
        ore_path="/tmp/ore.json",
        measured_library="/tmp/lib.json",
        target_T_K=298.0,
        gangue_T_K=298.0,
        freq_hz=2.45e9,
        phases=(phase,),
        max_rel_error_real=0.05,
        max_rel_error_imag=0.04,
    )


def test_calibration_changelog_detects_shift():
    base = _sample_calibration_report("v1")
    current = replace(
        _sample_calibration_report("v2"),
        max_rel_error_real=0.18,
        phases=(
            PhaseCalibrationDiff(
                phase="target",
                mineral="pyrite",
                predicted_eps_real=10.0,
                predicted_eps_imag=0.5,
                measured_eps_real=9.5,
                measured_eps_imag=0.48,
                delta_eps_real=0.5,
                delta_eps_imag=0.02,
                rel_error_real=0.18,
                rel_error_imag=0.04,
            ),
        ),
    )
    diff = diff_calibration_reports(base, current)
    assert diff["max_rel_error_real_shift"] == pytest.approx(0.13, abs=0.01)


def test_promotion_blocks_failed_bruggeman_calibration():
    gate = ValidationGateReport(passed=True, checks=[GateCheck("x", True, "")])
    cal_fail = {"passes_calibration": False, "passes_default_tolerance": True}
    blocked = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        ore_block={"materials_mode": "measured", "measured_dielectrics": {"path": "x", "dataset": {"dataset_id": "d"}}},
        deposit_calibration=cal_fail,
    )
    assert blocked.tier != PromotionTier.DEPOSIT_CALIBRATED


def test_jitter_stays_in_bounds():
    rng = np.random.default_rng(42)
    p0 = CavityParams()
    p1 = jitter_cavity_params(p0, rng, 0.05)
    assert 0.02 <= p1.charge_cx_frac <= 0.98
    assert 0.02 <= p1.feed_along_frac <= 0.98


def test_manufacturing_robust_min_selectivity():
    rep = evaluate_manufacturing_robust(
        GRID, CavityParams(), None, n_samples=4, placement_tol_frac=0.02, seed=1,
        legacy=False,
    )
    assert rep.min_selectivity <= rep.mean_selectivity
    assert rep.n_samples == 4


def test_transient_evolved_differs_from_frozen():
    """High drive: ε(T)+phase rules should diverge from frozen RT ε."""
    cfg_evolved = TransientConfig(
        dt_s=1.0,
        t_end_s=40.0,
        em_refresh_s=2.0,
        drive=25.0,
        bulk_cooling=2.0e4,
        evolve_properties=True,
    )
    cfg_frozen = replace(cfg_evolved, evolve_properties=False)
    params = replace(CavityParams(), plate_len_frac=0.25)
    evolved = simulate_transient(GRID, "pyrite_in_calcite", config=cfg_evolved, params=params)
    frozen = simulate_transient(GRID, "pyrite_in_calcite", config=cfg_frozen, params=params)
    assert evolved.report.T_final_mean_target_K != pytest.approx(
        frozen.report.T_final_mean_target_K, abs=0.5
    )


def test_industrial_multi_objective_runs():
    trials, study = optuna_multi_search(
        GRID,
        n_trials=6,
        seed=99,
        industrial_objectives=True,
    )
    assert len(trials) == 6
    assert len(study.directions) == 4
    pick = pareto_recommend(
        trials,
        study,
        weight_selectivity=0.25,
        weight_coupling=0.25,
        weight_gangue_budget=0.25,
        weight_particle_floor=0.25,
    )
    assert pick.selectivity >= 0.0


def test_forster_calibration_report_structure():
    ore_path = DATA / "ores" / "forster" / "forster_good_pyrite_calcite.json"
    if not ore_path.is_file():
        pytest.skip("forster ore fixture missing")
    rep = calibrate_ore_profile(ore_path)
    assert len(rep.phases) == 2
    assert rep.max_rel_error_real >= 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
