"""M4 pilot_ready gate and promotion tier tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.pilot_gate import evaluate_pilot_gate  # noqa: E402
from mw_inv.promotion import PromotionTier, assess_promotion, meets_tier  # noqa: E402
from mw_inv.solver_triangulation import SolverRow  # noqa: E402
from mw_inv.validation_gate import GateThresholds, evaluate_gate  # noqa: E402


def _bench_rows() -> list[SolverRow]:
    return [
        SolverRow("untuned", 0.50, openems_selectivity=0.48, rel_err_openems=0.04),
        SolverRow("tpe_best", 0.62, openems_selectivity=0.60, rel_err_openems=0.03),
    ]


def test_pilot_gate_passes_with_robust_multi_safety() -> None:
    evaluation = {
        "robust_gate": {"passed": True, "detail": "ok"},
        "tpe_best": {"coupling_eff": 0.42},
    }
    search_summary = {
        "search_mode": "multi_objective",
        "multi_search": {
            "check_arcing": True,
            "check_hotspot": True,
            "recommended": {"arcing_risk": False, "hotspot_violation": False},
        },
    }
    report = evaluate_pilot_gate(evaluation, search_summary)
    assert report.passed
    assert all(c.passed for c in report.checks)


def test_pilot_gate_fails_without_robust_or_safety() -> None:
    evaluation = {
        "robust_gate": {"passed": False},
        "tpe_best": {"coupling_eff": 0.05},
    }
    report = evaluate_pilot_gate(evaluation, {"search_mode": "legacy"})
    assert not report.passed


def test_assess_pilot_ready_above_bench() -> None:
    rows = _bench_rows()
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.0))
    pilot_gate = {"passed": True, "checks": []}
    assessment = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
        phantom_label="saline_2_vs_0.5",
        measured_eps_path="data/measured_eps.example.json",
        pilot_gate=pilot_gate,
    )
    assert assessment.tier == PromotionTier.PILOT_READY
    assert assessment.requirements["pilot_safety_repeatability"] is True
    assert meets_tier(PromotionTier.PILOT_READY, PromotionTier.BENCH_CALIBRATED)
