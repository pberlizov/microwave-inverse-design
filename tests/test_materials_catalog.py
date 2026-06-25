"""Tests for expanded HMAP mineral catalog."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.dielectric_data import MINERAL_MODELS, mineral_eps  # noqa: E402
from mw_inv.materials import PAIRS, Materials  # noqa: E402
from mw_inv.mineral_catalog import CATALOG, MicrowaveClass, hmap_minerals, loss_contrast  # noqa: E402
from mw_inv.ore_profiles import ORE_PROFILES, materials_from_ore  # noqa: E402


def test_all_hmap_minerals_in_models():
    for name in hmap_minerals():
        assert name in MINERAL_MODELS
        assert name in CATALOG
        assert CATALOG[name].mw_class == MicrowaveClass.HMAP


def test_new_material_pairs_load():
    for label in (
        "chalcopyrite_in_calcite",
        "pyrrhotite_in_quartz",
        "galena_in_calcite",
        "molybdenite_in_quartz",
    ):
        mats = Materials.from_pair(label)
        assert mats.target.imag > mats.gangue.imag * 3


def test_materials_from_ore_bruggeman():
    ore = ORE_PROFILES["disseminated_pyrite_porphyry"]
    mats = materials_from_ore(ore)
    assert mats.target.imag > 0.25
    assert mats.gangue.imag < 0.1
    assert abs(mats.target.real - mats.gangue.real) < 3.0


def test_loss_hierarchy_molybdenite_top():
    im_mo = mineral_eps("molybdenite").imag
    im_py = mineral_eps("pyrite").imag
    im_ca = mineral_eps("calcite").imag
    assert im_mo > im_py > im_ca


def test_chalcopyrite_eps_rises_with_T():
    cold = mineral_eps("chalcopyrite", 298.0)
    hot = mineral_eps("chalcopyrite", 773.0)
    assert hot.imag > cold.imag


def test_materials_from_ore_auto_pair_label():
    ore = ORE_PROFILES["magnetite_skarn"]
    mats = materials_from_ore(ore)
    assert mats.pair_label == "magnetite_in_quartz"
    assert mats.target.imag > mats.gangue.imag
