"""Pipeline --target-tier deposit_calibrated enforcement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.pipeline import main as pipeline_main  # noqa: E402
from mw_inv.promotion import PromotionTier  # noqa: E402
from mw_inv.run_manifest import RunManifest  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
CAMPAIGN = DATA / "campaigns" / "forster_literature_v1" / "campaign.json"


def test_target_tier_deposit_requires_campaign_or_envelope():
    with pytest.raises(SystemExit):
        pipeline_main([
            "--target-tier", "deposit_calibrated",
            "--materials", "pyrite_in_calcite",
            "--trials", "2", "--grid", "41",
            "--run-dir", "/tmp/should-not-run",
            "--skip-benchmarks", "--skip-export",
        ])


def test_target_tier_deposit_campaign_smoke(tmp_path: Path):
    if not CAMPAIGN.is_file():
        pytest.skip("forster campaign fixture missing")

    run_dir = tmp_path / "deposit_run"
    # Pre-bake search so we only test tier/envelope wiring (fast).
    search = {
        "materials": "custom",
        "grid": 41,
        "trials": 2,
        "baseline_untuned": {"selectivity": 0.50},
        "tpe_search": {
            "best_selectivity": 0.55,
            "best_params": {"plate_len_frac": 0.15, "plate_angle_deg": 60.0},
        },
    }
    search_path = run_dir / "search_summary.json"
    run_dir.mkdir(parents=True)
    search_path.write_text(json.dumps(search))

    with pytest.raises(SystemExit) as exc:
        pipeline_main([
            "--search", str(search_path),
            "--campaign", str(CAMPAIGN),
            "--target-tier", "deposit_calibrated",
            "--trials", "2", "--grid", "41",
            "--run-dir", str(run_dir),
            "--skip-benchmarks", "--skip-export",
            "--gate-min-improvement", "0",
            "--envelope-min-selectivity", "0.0",
        ])
    # Exit 7 = tier not reached (benchmarks skipped → unranked), or 6 = envelope fail
    assert exc.value.code in (6, 7)

    manifest = RunManifest.load(run_dir / "manifest.json")
    assert manifest.evaluation.get("deposit_envelope") is not None
    assert manifest.evaluation.get("deposit_envelope_gate") is not None
    assert manifest.evaluation.get("campaign") is not None


def test_target_tier_fdfd_smoke_no_tier_enforce_fail(tmp_path: Path):
    """Default target tier must not trigger exit 7 (tier_enforce only above fdfd_optimised)."""
    run_dir = tmp_path / "fdfd_run"
    search = {
        "materials": "pyrite_in_calcite",
        "grid": 41,
        "baseline_untuned": {"selectivity": 0.50},
        "tpe_search": {
            "best_selectivity": 0.58,
            "best_params": {"plate_len_frac": 0.12},
        },
        "random_search": {
            "best_selectivity": 0.52,
            "best_params": {"plate_len_frac": 0.08},
        },
    }
    run_dir.mkdir()
    search_path = run_dir / "search_summary.json"
    search_path.write_text(json.dumps(search))

    pipeline_main([
        "--search", str(search_path),
        "--materials", "pyrite_in_calcite",
        "--target-tier", PromotionTier.FDFD_OPTIMISED.value,
        "--run-dir", str(run_dir),
        "--skip-benchmarks", "--skip-export",
        "--gate-min-improvement", "0",
    ])
    manifest = RunManifest.load(run_dir / "manifest.json")
    assert (run_dir / "manifest.json").is_file()
    assert manifest.promotion.get("tier") is not None
