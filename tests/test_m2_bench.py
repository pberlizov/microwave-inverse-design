"""M2 bench-calibrated promotion path tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.pipeline import main as pipeline_main  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.phantom_calibration import evaluate_bench_gate  # noqa: E402
from mw_inv.promotion import PromotionTier  # noqa: E402
from mw_inv.run_manifest import RunManifest  # noqa: E402
from mw_inv.search import best, evaluate_params, optuna_search  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEASURED_EPS = ROOT / "data" / "measured_eps.example.json"
LAB_JSON = ROOT / "data" / "lab_measurements.example.json"


def test_bench_gate_passes_on_example_fixtures() -> None:
    report = evaluate_bench_gate(
        "saline_2_vs_0.5",
        MEASURED_EPS,
        LAB_JSON,
    )
    assert report.passed
    assert report.probe_calibration is not None


def test_pipeline_reaches_bench_calibrated(tmp_path: Path) -> None:
    """M2: solver_triangulated + phantom probe ε + lab rank → bench_calibrated."""
    if not MEASURED_EPS.is_file() or not LAB_JSON.is_file():
        pytest.skip("bench fixture JSON missing")

    grid = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair("pyrite_in_calcite")
    base = evaluate_params(grid, CavityParams(), materials)
    tpe_best = best(optuna_search(grid, 16, seed=7, materials=materials))
    if tpe_best.selectivity <= base.selectivity:
        pytest.skip("search did not improve on this seed")

    run_dir = tmp_path / "m2_run"
    run_dir.mkdir()
    search_summary = {
        "materials": "pyrite_in_calcite",
        "grid": 41,
        "tpe_search": {
            "best_selectivity": tpe_best.selectivity,
            "best_params": tpe_best.params,
        },
    }
    search_path = run_dir / "search_summary.json"
    search_path.write_text(json.dumps(search_summary))

    pipeline_main([
        "--search",
        str(search_path),
        "--materials",
        "pyrite_in_calcite",
        "--grid",
        "41",
        "--run-dir",
        str(run_dir),
        "--gate-min-improvement",
        "0",
        "--synthesize-openems-dumps",
        "--phantom",
        "saline_2_vs_0.5",
        "--measured-eps",
        str(MEASURED_EPS),
        "--lab-measurements",
        str(LAB_JSON),
        "--bench-study",
    ])

    manifest = RunManifest.load(run_dir / "manifest.json")
    assert manifest.bench.get("gate", {}).get("passed") is True
    tier = PromotionTier(manifest.promotion["tier"])
    assert tier == PromotionTier.BENCH_CALIBRATED
    assert manifest.promotion["requirements"]["bench_phantom_calibration"] is True
