"""M1 solver-triangulated pipeline integration tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.pipeline import main as pipeline_main  # noqa: E402
from mw_inv.design_export import ExportBundle  # noqa: E402
from mw_inv.openems_runner import (  # noqa: E402
    openems_dump_dir,
    port_metrics_ready,
    synthesize_port_dumps,
)
from mw_inv.promotion import PromotionTier  # noqa: E402
from mw_inv.run_manifest import RunManifest  # noqa: E402
from mw_inv.run_refresh import apply_triangulation_refresh  # noqa: E402
from mw_inv.validation_gate import GateThresholds  # noqa: E402


def test_synthesize_port_dumps_creates_metrics(tmp_path: Path) -> None:
    export_dir = tmp_path / "design_exports"
    export_dir.mkdir()
    bundles = [
        ExportBundle(
            label="untuned",
            openems_path=export_dir / "untuned_cavity.m",
            scene_npz_path=export_dir / "untuned_scene.npz",
            manifest_path=export_dir / "untuned_manifest.json",
            fdfd_selectivity=0.52,
        ),
        ExportBundle(
            label="tpe_best",
            openems_path=export_dir / "tpe_best_cavity.m",
            scene_npz_path=export_dir / "tpe_best_scene.npz",
            manifest_path=export_dir / "tpe_best_manifest.json",
            fdfd_selectivity=0.61,
        ),
    ]
    dump = synthesize_port_dumps(export_dir, bundles)
    assert port_metrics_ready(dump, ["untuned", "tpe_best"])
    assert openems_dump_dir(export_dir) == dump


def test_pipeline_solver_triangulated_with_synthetic_openems(tmp_path: Path) -> None:
    """One-command M1: export + synthetic openEMS dumps → solver_triangulated tier."""
    from mw_inv.fdfd import Grid
    from mw_inv.geometry import CavityParams, Materials
    from mw_inv.search import best, evaluate_params, optuna_search

    grid = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair("pyrite_in_calcite")
    base = evaluate_params(grid, CavityParams(), materials)
    tpe_best = best(optuna_search(grid, 16, seed=42, materials=materials))
    if tpe_best.selectivity <= base.selectivity:
        pytest.skip("search did not improve over untuned on this seed")

    run_dir = tmp_path / "m1_run"
    run_dir.mkdir()
    search_summary = {
        "materials": "pyrite_in_calcite",
        "grid": 41,
        "trials": 16,
        "seed": 42,
        "baseline_untuned": {"selectivity": base.selectivity, "contrast": base.contrast},
        "tpe_search": {
            "best_selectivity": tpe_best.selectivity,
            "best_contrast": tpe_best.contrast,
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
    ])

    manifest = RunManifest.load(run_dir / "manifest.json")
    assert manifest.export_dir is not None
    dump = openems_dump_dir(manifest.export_dir)
    labels = [b["label"] for b in manifest.export_summary.get("bundles", [])]
    assert labels
    assert port_metrics_ready(dump, labels)

    tier = PromotionTier(manifest.promotion["tier"])
    assert tier in (PromotionTier.SOLVER_TRIANGULATED, PromotionTier.BENCH_CALIBRATED)
    assert manifest.promotion["requirements"]["external_solver_validation"] is True

    rows = manifest.triangulation.get("rows", [])
    assert any(r.get("openems_selectivity") is not None for r in rows)


def test_refresh_run_with_openems_fixture(tmp_path: Path) -> None:
    """update_run_with_openems path via shared refresh helper."""
    from mw_inv.fdfd import Grid
    from mw_inv.geometry import CavityParams, Materials
    from mw_inv.search import best, evaluate_params, optuna_search

    grid = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair("pyrite_in_calcite")
    base = evaluate_params(grid, CavityParams(), materials)
    tpe_best = best(optuna_search(grid, 4, seed=1, materials=materials))

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    search_summary = {
        "materials": "pyrite_in_calcite",
        "grid": 41,
        "baseline_untuned": {"selectivity": base.selectivity},
        "tpe_search": {
            "best_selectivity": tpe_best.selectivity,
            "best_params": tpe_best.params,
        },
    }
    search_path = run_dir / "search_summary.json"
    search_path.write_text(json.dumps(search_summary))

    export_dir = run_dir / "design_exports"
    bundles = [
        ExportBundle(
            label="untuned",
            openems_path=export_dir / "untuned_cavity.m",
            scene_npz_path=export_dir / "untuned_scene.npz",
            manifest_path=export_dir / "untuned_manifest.json",
            fdfd_selectivity=base.selectivity,
        ),
        ExportBundle(
            label="tpe_best",
            openems_path=export_dir / "tpe_best_cavity.m",
            scene_npz_path=export_dir / "tpe_best_scene.npz",
            manifest_path=export_dir / "tpe_best_manifest.json",
            fdfd_selectivity=tpe_best.selectivity,
        ),
    ]
    dump = synthesize_port_dumps(export_dir, bundles)

    manifest = RunManifest(
        run_id="fixture",
        materials="pyrite_in_calcite",
        benchmarks_passed=True,
        search_path=str(search_path),
        search_summary=search_summary,
    )
    refresh = apply_triangulation_refresh(
        manifest,
        run_dir,
        search_path=search_path,
        grid=grid,
        materials=materials,
        materials_label="pyrite_in_calcite",
        openems_dump_dir=dump,
        gate_thresholds=GateThresholds(min_fdfd_improvement=0.0),
    )
    manifest.write(run_dir / "manifest.json")

    assert refresh.assessment.tier in (
        PromotionTier.SOLVER_TRIANGULATED,
        PromotionTier.BENCH_CALIBRATED,
    )
    assert refresh.gate.passed or any(
        c.name.startswith("openems") for c in refresh.gate.checks
    )
