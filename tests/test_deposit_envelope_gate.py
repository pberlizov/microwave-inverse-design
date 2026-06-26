"""Deposit envelope promotion gate and openEMS/FDFD metal coupling alignment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.deposit_envelope import (  # noqa: E402
    DepositEnvelopeReport,
    DepositEnvelopeThresholds,
    evaluate_deposit_envelope_gate,
)
from mw_inv.promotion import PromotionTier, assess_promotion  # noqa: E402
from mw_inv.solver_triangulation import SolverRow  # noqa: E402
from mw_inv.validation_gate import GateThresholds, evaluate_gate  # noqa: E402


def _sample_envelope(**kwargs) -> DepositEnvelopeReport:
    defaults = dict(
        n_ores=3,
        n_ok=3,
        min_selectivity=0.55,
        mean_selectivity=0.70,
        min_coupling_eff=0.80,
        max_gangue_power_fraction=0.45,
        max_pec_loss_fraction=0.02,
        results=(),
    )
    defaults.update(kwargs)
    return DepositEnvelopeReport(**defaults)


def test_deposit_envelope_gate_pass():
    rep = _sample_envelope()
    gate = evaluate_deposit_envelope_gate(
        rep,
        DepositEnvelopeThresholds(min_selectivity=0.50, min_coupling_eff=0.50),
    )
    assert gate.passed
    assert all(c.passed for c in gate.checks)


def test_deposit_envelope_gate_fails_min_selectivity():
    rep = _sample_envelope(min_selectivity=0.20)
    gate = evaluate_deposit_envelope_gate(
        rep,
        DepositEnvelopeThresholds(min_selectivity=0.50),
    )
    assert not gate.passed
    failed = [c.name for c in gate.checks if not c.passed]
    assert "envelope_min_selectivity" in failed


def test_promotion_deposit_requires_envelope_gate():
    rows = [
        SolverRow("untuned", 0.54, fdfd_coupling_eff=0.95),
        SolverRow("tpe_best", 0.62, fdfd_coupling_eff=0.93),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.01))
    campaign = {
        "campaign_id": "forster_literature_v1",
        "measured_dielectrics": "measured_dielectrics/forster_hmap_minerals_v1.json",
    }
    env_gate_pass = {"passed": True, "checks": []}
    env_gate_fail = {"passed": False, "checks": [{"name": "envelope_min_selectivity", "passed": False}]}

    ok = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
        campaign_block=campaign,
        deposit_envelope_gate=env_gate_pass,
    )
    assert ok.tier == PromotionTier.DEPOSIT_CALIBRATED
    assert ok.requirements["deposit_envelope_gate"] is True

    blocked = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
        campaign_block=campaign,
        deposit_envelope_gate=env_gate_fail,
    )
    assert blocked.tier == PromotionTier.FDFD_OPTIMISED
    assert blocked.requirements["deposit_envelope_gate"] is False


def test_openems_fdfd_coupling_ratio_gate():
    rows = [
        SolverRow(
            "untuned",
            0.54,
            fdfd_coupling_eff=0.90,
            openems_coupling_eff=0.85,
        ),
        SolverRow(
            "tpe_best",
            0.62,
            fdfd_coupling_eff=0.88,
            openems_coupling_eff=0.80,
        ),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.01))
    names = [c.name for c in gate.checks]
    assert "openems_fdfd_coupling_ratio" in names
    ratio_check = next(c for c in gate.checks if c.name == "openems_fdfd_coupling_ratio")
    assert ratio_check.passed

    bad_rows = [
        SolverRow(
            "untuned",
            0.54,
            fdfd_coupling_eff=0.90,
            openems_coupling_eff=0.85,
        ),
        SolverRow(
            "tpe_best",
            0.62,
            fdfd_coupling_eff=0.10,
            openems_coupling_eff=0.85,
        ),
    ]
    bad_gate = evaluate_gate(bad_rows, GateThresholds(min_fdfd_improvement=0.01))
    ratio_bad = next(c for c in bad_gate.checks if c.name == "openems_fdfd_coupling_ratio")
    assert not ratio_bad.passed


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
