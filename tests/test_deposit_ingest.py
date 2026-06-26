"""D0 deposit-calibrated ore pipeline and measured ε ingest tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.pipeline import main as pipeline_main  # noqa: E402
from mw_inv.measured_dielectrics import load_measured_dielectrics  # noqa: E402
from mw_inv.ore_profiles import (  # noqa: E402
    load_ore_profile,
    materials_from_ore,
    ore_summary,
    resolve_measured_dielectrics_path,
)
from mw_inv.promotion import PromotionTier  # noqa: E402
from mw_inv.run_manifest import RunManifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORE_EXAMPLE = ROOT / "data" / "ores" / "disseminated_pyrite_porphyry_measured_example.json"
DEPOSIT_EPS = ROOT / "data" / "measured_dielectrics" / "disseminated_pyrite_porphyry_deposit.json"


def test_deposit_dataset_loads_multi_dimensional() -> None:
    lib = load_measured_dielectrics(DEPOSIT_EPS)
    summary = lib.summary()
    assert summary["dataset_id"] == "disseminated_pyrite_porphyry_lab_v1"
    target = summary["phases"]["target"]
    assert target["n_points"] >= 5
    assert target["temp_K_range"][1] > target["temp_K_range"][0]
    assert len(target["moisture_wt_percent_levels"]) >= 2


def test_relative_measured_path_from_ore_json() -> None:
    ore = load_ore_profile(ORE_EXAMPLE)
    mp = resolve_measured_dielectrics_path(ORE_EXAMPLE, ore.measured_dielectrics or {})
    assert mp.resolve() == DEPOSIT_EPS.resolve()


def test_materials_from_deposit_interp_temp_and_moisture() -> None:
    ore = load_ore_profile(ORE_EXAMPLE)
    mats_cold = materials_from_ore(
        ore, ore_profile_path=ORE_EXAMPLE, target_T_K=298.0, freq_hz=2.45e9, moisture_wt_percent=0.0,
    )
    mats_hot = materials_from_ore(
        ore, ore_profile_path=ORE_EXAMPLE, target_T_K=373.0, freq_hz=2.45e9, moisture_wt_percent=0.0,
    )
    assert mats_hot.target.imag > mats_cold.target.imag

    mats_dry = materials_from_ore(
        ore, ore_profile_path=ORE_EXAMPLE, target_T_K=298.0, freq_hz=2.45e9, moisture_wt_percent=0.0,
    )
    mats_wet = materials_from_ore(
        ore, ore_profile_path=ORE_EXAMPLE, target_T_K=298.0, freq_hz=2.45e9, moisture_wt_percent=3.0,
    )
    mats_mid = materials_from_ore(
        ore, ore_profile_path=ORE_EXAMPLE, target_T_K=298.0, freq_hz=2.45e9, moisture_wt_percent=1.5,
    )
    assert mats_dry.target.imag < mats_wet.target.imag
    assert mats_mid.target.imag > mats_dry.target.imag
    assert mats_mid.target.imag < mats_wet.target.imag


def test_pipeline_ore_records_measured_provenance(tmp_path: Path) -> None:
    if not ORE_EXAMPLE.is_file():
        pytest.skip("deposit example ore JSON missing")

    run_dir = tmp_path / "d0_run"
    pipeline_main([
        "--ore",
        str(ORE_EXAMPLE),
        "--ore-target-t",
        "373",
        "--ore-moisture",
        "1.0",
        "--trials",
        "4",
        "--grid",
        "41",
        "--run-dir",
        str(run_dir),
        "--skip-export",
        "--skip-benchmarks",
        "--gate-min-improvement",
        "0",
    ])

    manifest = RunManifest.load(run_dir / "manifest.json")
    ore = manifest.ore
    assert ore.get("materials_mode") == "measured"
    md = ore.get("measured_dielectrics") or {}
    assert md.get("dataset", {}).get("dataset_id") == "disseminated_pyrite_porphyry_lab_v1"
    assert manifest.search_summary.get("ore", {}).get("materials_mode") == "measured"
    assert manifest.ore.get("eval_conditions", {}).get("target_T_K") == 373.0
    if manifest.gate.get("passed") and manifest.benchmarks_passed:
        assert manifest.promotion.get("tier") == PromotionTier.DEPOSIT_CALIBRATED.value
    else:
        assert manifest.promotion.get("requirements", {}).get("deposit_measured_eps") is False


def test_pipeline_multi_objective_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "multi_run"
    pipeline_main([
        "--materials",
        "pyrite_in_calcite",
        "--multi-objective",
        "--trials",
        "4",
        "--grid",
        "41",
        "--run-dir",
        str(run_dir),
        "--skip-export",
        "--skip-benchmarks",
        "--gate-min-improvement",
        "0",
    ])
    summary = json.loads((run_dir / "search_summary.json").read_text())
    assert summary.get("search_mode") == "multi_objective"
    assert "multi_search" in summary
    assert summary["tpe_search"].get("source") == "pareto_recommend"
    assert summary["multi_search"]["recommended"]["params"]
    manifest = RunManifest.load(run_dir / "manifest.json")
    assert manifest.search_summary.get("search_mode") == "multi_objective"
