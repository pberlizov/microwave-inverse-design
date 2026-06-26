"""Tests for promotion tiers, run manifest, and pipeline smoke."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.benchmarks import run_benchmarks  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.promotion import (  # noqa: E402
    PromotionError,
    PromotionTier,
    assert_tier_at_least,
    assess_promotion,
    meets_tier,
)
from mw_inv.run_manifest import RunManifest, finalize_promotion  # noqa: E402
from mw_inv.search import best, evaluate_params, optuna_search  # noqa: E402
from mw_inv.solver_triangulation import SolverRow, triangulate_from_search  # noqa: E402
from mw_inv.validation_gate import GateThresholds, evaluate_gate  # noqa: E402


GRID = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
PAIR = "pyrite_in_calcite"


def test_tier_ordering():
    assert meets_tier(PromotionTier.FDFD_OPTIMISED, PromotionTier.LITERATURE_GROUNDED)
    assert meets_tier(PromotionTier.DEPOSIT_CALIBRATED, PromotionTier.FDFD_OPTIMISED)
    assert meets_tier(PromotionTier.SOLVER_TRIANGULATED, PromotionTier.DEPOSIT_CALIBRATED)
    assert not meets_tier(PromotionTier.LITERATURE_GROUNDED, PromotionTier.FDFD_OPTIMISED)
    assert not meets_tier(PromotionTier.FDFD_OPTIMISED, PromotionTier.DEPOSIT_CALIBRATED)
    assert not meets_tier(PromotionTier.UNRANKED, PromotionTier.LITERATURE_GROUNDED)


def test_assess_fdfd_optimised_from_gate():
    rows = [
        SolverRow("untuned", 0.54),
        SolverRow("tpe_best", 0.62),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.01))
    assessment = assess_promotion(benchmarks_passed=True, gate=gate, triangulation_rows=rows)
    assert assessment.tier == PromotionTier.FDFD_OPTIMISED
    assert assessment.requirements["fdfd_gate"]


def test_export_tier_guard_raises():
    with __import__("pytest").raises(PromotionError):
        assert_tier_at_least(PromotionTier.LITERATURE_GROUNDED, PromotionTier.FDFD_OPTIMISED, action="export")


def test_manifest_roundtrip(tmp_path):
    m = RunManifest(run_id="test_run", materials=PAIR, benchmarks_passed=True)
    m.gate = {"passed": True, "checks": []}
    m.triangulation = {"rows": [{"label": "untuned", "fdfd_selectivity": 0.5}]}
    path = m.write(tmp_path / "manifest.json")
    loaded = RunManifest.load(path)
    assert loaded.run_id == "test_run"
    finalize_promotion(loaded)
    assert "tier" in loaded.promotion


def test_manifest_can_store_provenance(tmp_path: Path) -> None:
    m = RunManifest(run_id="prov", materials=PAIR)
    m.provenance = {"runtime": {"python": "x"}, "packages": {"mw-inv": "y"}, "git": {"commit": "z"}}
    m.cli = {"args": {"grid": 41}}
    path = m.write(tmp_path / "manifest.json")
    loaded = RunManifest.load(path)
    assert loaded.provenance["runtime"]["python"] == "x"
    assert loaded.cli["args"]["grid"] == 41


def test_pipeline_smoke(tmp_path):
    """Tier-1 CI smoke: benchmarks + tiny search + gate + manifest."""
    bench = run_benchmarks()
    assert bench.passed

    materials = Materials.from_pair(PAIR)
    base = evaluate_params(GRID, CavityParams(), materials)
    trials = optuna_search(GRID, 4, seed=42, materials=materials)
    tpe_best = best(trials)

    search_summary = {
        "materials": PAIR,
        "grid": GRID.nx,
        "trials": 4,
        "baseline_untuned": {"selectivity": base.selectivity},
        "tpe_search": {
            "best_selectivity": tpe_best.selectivity,
            "best_params": tpe_best.params,
        },
    }
    search_path = tmp_path / "search_summary.json"
    search_path.write_text(json.dumps(search_summary))

    rows = triangulate_from_search(search_path, GRID, materials)
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.0))

    manifest = RunManifest(
        run_id="smoke",
        materials=PAIR,
        benchmarks_passed=True,
        search_path=str(search_path),
        search_summary=search_summary,
        gate=gate.to_dict(),
        triangulation={"rows": [r.to_dict() for r in rows]},
    )
    assessment = finalize_promotion(manifest)
    manifest.write(tmp_path / "manifest.json")

    assert assessment.tier in (
        PromotionTier.FDFD_OPTIMISED,
        PromotionTier.LITERATURE_GROUNDED,
        PromotionTier.SOLVER_TRIANGULATED,
    )
    if tpe_best.selectivity > base.selectivity:
        assert assessment.tier == PromotionTier.FDFD_OPTIMISED


def test_bench_calibration_requires_rank_if_lab_measurements_present(tmp_path: Path) -> None:
    """Bench tier check: measured eps drift + measured ΔT rank agreement when lab JSON is provided."""
    # Measured eps matches the phantom anchors exactly -> zero drift.
    measured = {
        "batches": [
            {"label": "salt_2pct", "salt_wt_percent": 2.0, "eps_real": 16.0, "eps_imag": 3.5, "freq_hz": 2.45e9},
            {"label": "salt_0.5pct", "salt_wt_percent": 0.5, "eps_real": 7.0, "eps_imag": 0.35, "freq_hz": 2.45e9},
        ]
    }
    measured_path = tmp_path / "measured_eps.json"
    measured_path.write_text(json.dumps(measured))

    lab_good = [
        {"phantom": "saline_2_vs_0.5", "measured_delta_T_K": 12.0, "untuned_measured_delta_T_K": 8.0}
    ]
    lab_path = tmp_path / "lab.json"
    lab_path.write_text(json.dumps(lab_good))

    rows = [
        SolverRow("untuned", 0.54, openems_selectivity=0.50),
        SolverRow("tpe_best", 0.62, openems_selectivity=0.58),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.0))

    assessment = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
        phantom_label="saline_2_vs_0.5",
        measured_eps_path=str(measured_path),
        lab_measurements_path=str(lab_path),
    )
    assert assessment.requirements["bench_phantom_calibration"] is True

    # Now provide a lab file where optimized does not beat untuned -> should fail bench check.
    lab_bad = [
        {"phantom": "saline_2_vs_0.5", "measured_delta_T_K": 7.0, "untuned_measured_delta_T_K": 8.0}
    ]
    lab_bad_path = tmp_path / "lab_bad.json"
    lab_bad_path.write_text(json.dumps(lab_bad))
    assessment2 = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
        phantom_label="saline_2_vs_0.5",
        measured_eps_path=str(measured_path),
        lab_measurements_path=str(lab_bad_path),
    )
    assert assessment2.requirements["bench_phantom_calibration"] is False


def test_assess_solver_triangulated_when_openems_present() -> None:
    """Promotion reaches solver_triangulated when external solver data exists and gate checks pass."""
    rows = [
        SolverRow("untuned", 0.50, openems_selectivity=0.48, rel_err_openems=0.04),
        SolverRow("random_best", 0.60, openems_selectivity=0.58, rel_err_openems=0.03),
        SolverRow("tpe_best", 0.62, openems_selectivity=0.60, rel_err_openems=0.03),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.0))
    assessment = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
    )
    assert assessment.tier in (PromotionTier.SOLVER_TRIANGULATED, PromotionTier.BENCH_CALIBRATED)
    assert assessment.requirements["external_solver_validation"] is True


def test_assess_deposit_calibrated_with_measured_ore() -> None:
    """Deposit tier sits above fdfd_optimised when validated measured ore ε is on the manifest."""
    rows = [
        SolverRow("untuned", 0.54),
        SolverRow("tpe_best", 0.62),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.01))
    ore_block = {
        "materials_mode": "measured",
        "measured_dielectrics": {
            "path": "data/measured_dielectrics/example.json",
            "dataset": {"dataset_id": "lab_v1", "phases": {"target": {}, "gangue": {}}},
        },
    }
    assessment = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
        ore_block=ore_block,
    )
    assert assessment.tier == PromotionTier.DEPOSIT_CALIBRATED
    assert assessment.requirements["deposit_measured_eps"] is True
    assert not assessment.requirements["external_solver_validation"]

    bruggeman_ore = {"materials_mode": "bruggeman", "measured_dielectrics": {}}
    assessment2 = assess_promotion(
        benchmarks_passed=True,
        gate=gate,
        triangulation_rows=rows,
        ore_block=bruggeman_ore,
    )
    assert assessment2.tier == PromotionTier.FDFD_OPTIMISED
    assert assessment2.requirements["deposit_measured_eps"] is False
