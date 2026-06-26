"""Tests for Tier-2 ore JSON ingest, pair selection, and texture mapping."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.ore_profiles import (  # noqa: E402
    ORE_PROFILES,
    cavity_params_from_ore,
    dominant_hmap,
    infer_gangue_mineral,
    load_ore_profile,
    materials_from_ore,
    ore_summary,
    suggest_material_pair,
)


DATA_ORES = Path(__file__).resolve().parents[1] / "data" / "ores"


def test_load_ore_profile_fixture():
    ore = load_ore_profile(DATA_ORES / "disseminated_pyrite_porphyry.json")
    assert ore.label == "disseminated_pyrite_porphyry"
    assert ore.source == "QEMSCAN"
    assert ore.texture is not None
    assert ore.texture.texture_class == "disseminated"
    assert ore.texture.mean_grain_radius_m == 0.0025


def test_suggest_pair_porphyry():
    ore = load_ore_profile(DATA_ORES / "disseminated_pyrite_porphyry.json")
    pair = suggest_material_pair(ore)
    assert pair in ("pyrite_in_calcite", "chalcopyrite_in_calcite")
    assert dominant_hmap(ore) == "pyrite"
    assert infer_gangue_mineral(ore) == "quartz"


def test_suggest_pair_massive_pyrite():
    ore = load_ore_profile(DATA_ORES / "massive_pyrite.json")
    assert suggest_material_pair(ore) == "pyrite_in_calcite"
    assert ore.heating_class() == "IV_excellent"


def test_materials_from_ore_auto_pair():
    ore = ORE_PROFILES["disseminated_pyrite_porphyry"]
    mats = materials_from_ore(ore)
    assert mats.pair_label is not None
    assert mats.target.imag > mats.gangue.imag


def test_texture_maps_to_inclusion_radius():
    ore = load_ore_profile(DATA_ORES / "disseminated_pyrite_porphyry.json")
    p = cavity_params_from_ore(ore, cavity_span_m=0.36)
    assert 0.01 <= p.inclusion_radius_frac <= 0.12
    assert len(p.inclusion_offsets_frac) >= 3


def test_massive_texture_single_grain():
    ore = load_ore_profile(DATA_ORES / "massive_pyrite.json")
    p = cavity_params_from_ore(ore, cavity_span_m=0.36)
    assert len(p.inclusion_offsets_frac) == 1
    assert abs(p.inclusion_radius_frac - 0.008 / 0.36) < 0.01


def test_barren_ore_summary():
    ore = load_ore_profile(DATA_ORES / "barren_quartz.json")
    summary = ore_summary(ore)
    assert summary["hmap_wt_percent"] == 0.0
    assert summary["heating_class"] == "I_poor"
    mats = materials_from_ore(ore)
    assert mats.target.imag == mats.gangue.imag


def test_ore_summary_roundtrip_keys():
    ore = load_ore_profile(DATA_ORES / "massive_pyrite.json")
    s = ore_summary(ore)
    for key in (
        "label", "hmap_wt_percent", "suggested_pair", "heating_class",
        "loss_contrast", "inferred_gangue",
    ):
        assert key in s


def test_materials_from_ore_uses_measured_dielectrics(tmp_path: Path) -> None:
    measured_path = tmp_path / "measured_eps.json"
    measured_path.write_text(json.dumps({
        "description": "synthetic deposit dataset",
        "phases": {
            "target": [
                {"temp_K": 298, "freq_hz": 2.45e9, "eps_real": 9.0, "eps_imag": 0.9},
            ],
            "gangue": [
                {"temp_K": 298, "freq_hz": 2.45e9, "eps_real": 5.0, "eps_imag": 0.05},
            ],
        },
    }))
    ore_path = tmp_path / "ore.json"
    ore_path.write_text(json.dumps({
        "label": "ore_measured",
        "source": "TEST",
        "fractions": {"pyrite": 0.1},
        "gangue_mineral": "quartz",
        "measured_dielectrics": {
            "path": str(measured_path),
            "target_phase": "target",
            "gangue_phase": "gangue",
            "moisture_wt_percent": 0.0,
        },
    }))
    ore = load_ore_profile(ore_path)
    mats = materials_from_ore(ore, target_T_K=298, gangue_T_K=298, freq_hz=2.45e9)
    assert abs(mats.target.real - 9.0) < 1e-12
    assert abs(mats.target.imag - 0.9) < 1e-12
    assert abs(mats.gangue.real - 5.0) < 1e-12
    assert abs(mats.gangue.imag - 0.05) < 1e-12
