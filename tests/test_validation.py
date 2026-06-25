"""Tests for validation suite and upgraded materials."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.dielectric_data import PYRITE, polyakova_bulk_eps  # noqa: E402
from mw_inv.materials import Materials  # noqa: E402
from mw_inv.validation import run_all  # noqa: E402


def test_validation_suite_passes():
    report = run_all(include_meep=False)
    assert report.passed, [c for c in report.checks if not c.passed]


def test_materials_temperature_increases_pyrite_loss():
    cold = Materials.from_pair("pyrite_in_calcite", target_T_K=298.0)
    hot = Materials.from_pair("pyrite_in_calcite", target_T_K=773.0)
    assert hot.target.imag > cold.target.imag
    assert hot.target.real >= cold.target.real


def test_magnetite_has_magnetic_permeability():
    mats = Materials.from_pair("magnetite_in_quartz")
    assert mats.target_mu.imag > 0.3


def test_polyakova_bulk_differs_from_disseminated():
    bulk = abs(polyakova_bulk_eps("pyrite", 2.45e9))
    scene = abs(Materials.from_pair("pyrite_in_calcite").target)
    assert bulk / scene > 5.0


def test_pyrite_table_monotone_loss_with_T():
    losses = [PYRITE.eps(T).imag for T in (298.0, 573.0, 773.0, 973.0)]
    assert losses == sorted(losses)
