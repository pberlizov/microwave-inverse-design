"""Literature dataset ingest and catalog wiring tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.external_datasets import ingest_status, load_datasets_catalog  # noqa: E402
from mw_inv.literature_ingest import (  # noqa: E402
    build_hartlieb_bedrock_library,
    build_usbm_coax_library,
    ingest_all_auto,
    list_adapters,
)
from mw_inv.ore_profiles import load_ore_profile, materials_from_ore  # noqa: E402
from mw_inv.real_data_eval import discover_real_data_catalog, evaluate_real_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data"


def test_all_adapters_registered_in_catalog() -> None:
    cat = load_datasets_catalog(ROOT)
    catalog_adapters = {e.ingest_adapter for e in cat.auto_ingest_entries()}
    assert catalog_adapters == set(list_adapters())


def test_hartlieb_library_structure() -> None:
    payload = build_hartlieb_bedrock_library()
    assert payload["dataset_id"] == "hartlieb_bedrock_v1"
    labels = {p["label"] for p in payload["phases"]}
    assert labels == {"basalt", "granite", "sandstone"}


def test_usbm_coax_has_hmap_minerals() -> None:
    payload = build_usbm_coax_library(ROOT)
    labels = {p["label"] for p in payload["phases"]}
    assert "pyrite" in labels
    assert "magnetite" in labels
    assert len(payload["phases"][0]["points"]) == 8  # 4 temps × 2 freqs


def test_ingest_all_auto_writes_everything() -> None:
    paths = ingest_all_auto(ROOT)
    assert len(paths) >= 50  # 9 dielectric libs + 42 forster ores + manifest
    status = ingest_status(ROOT)
    auto = [r for r in status if r.get("auto")]
    assert len(auto) == len(list_adapters())
    assert all(r["status"] == "ingested" for r in auto)


def test_forster_ore_loads_measured_materials() -> None:
    ingest_all_auto(ROOT)
    ore_path = ROOT / "ores" / "forster" / "forster_good_porphyry_a.json"
    if not ore_path.is_file():
        pytest.skip("forster ores not ingested")
    ore = load_ore_profile(ore_path)
    mats = materials_from_ore(
        ore, ore_profile_path=ore_path, target_T_K=298.15, gangue_T_K=298.15, freq_hz=2.45e9,
    )
    assert mats.target.imag > mats.gangue.imag


def test_catalog_discovers_all_literature_sources() -> None:
    ingest_all_auto(ROOT)
    cat = discover_real_data_catalog(ROOT)
    kinds = {s.kind for s in cat.sources}
    assert "literature_ingest" in kinds
    ingested_ids = {s.label for s in cat.sources if s.kind == "literature_ingest"}
    assert "forster_hmap_minerals_v1" in ingested_ids
    assert "usbm_coax_minerals_v1" in ingested_ids
    assert "gabriel_saline_phantoms_v1" in ingested_ids


def test_quick_eval_covers_literature_deposits() -> None:
    ingest_all_auto(ROOT)
    report = evaluate_real_data(ROOT, quick=True)
    ids = {r.get("dataset_id") for r in report["deposit_libraries"]}
    assert "hartlieb_bedrock_v1" in ids
    assert "forster_hmap_minerals_v1" in ids
    assert "europeg_pegmatite_v1" in ids
    assert report["summary"]["n_ores"] >= 45
    assert report["summary"]["n_deposit_points"] >= 100
    assert all("error" not in r for r in report["ore_profiles"])
