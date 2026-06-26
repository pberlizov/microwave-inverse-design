"""Tests for promotion-aware openEMS gating (F1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.pipeline import main as pipeline_main  # noqa: E402
from mw_inv.run_manifest import RunManifest  # noqa: E402


def test_openems_skipped_when_fdfd_gate_fails(tmp_path: Path) -> None:
    """F1: do not burn openEMS budget when FDFD gate fails."""
    run_dir = tmp_path / "bad_run"
    run_dir.mkdir()
    # TPE worse than untuned → gate fails with default min improvement.
    search_summary = {
        "materials": "pyrite_in_calcite",
        "grid": 41,
        "baseline_untuned": {"selectivity": 0.80},
        "tpe_search": {
            "best_selectivity": 0.70,
            "best_params": {"plate_len_frac": 0.2, "plate_angle_deg": 45.0},
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
        "--skip-benchmarks",
        "--synthesize-openems-dumps",
    ])

    manifest = RunManifest.load(run_dir / "manifest.json")
    assert manifest.export_dir is None or "openEMS skipped" in " ".join(manifest.notes)
    assert manifest.triangulation.get("openems_dump_dir") in (None, "None")
