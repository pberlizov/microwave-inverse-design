"""Tests for MEEP cross-check helpers and lab phantom predictions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, build_scene  # noqa: E402
from mw_inv.materials import Materials  # noqa: E402
from mw_inv.meep_compare import compare_fdfd_meep, meep_available  # noqa: E402
from mw_inv.phantom import PHANTOMS, materials_from_phantom, predict_lab_outcome  # noqa: E402


GRID = Grid(nx=51, ny=51, Lx=0.36, Ly=0.36)


def test_phantom_materials_load():
    for label in PHANTOMS:
        m = materials_from_phantom(label)
        assert m.target.imag > m.gangue.imag


def test_phantom_prediction_runs():
    pred = predict_lab_outcome("saline_2_vs_0.5", GRID, n_opt_trials=4, seed=0)
    assert 0.0 < pred.untuned_selectivity <= 1.0
    assert len(pred.measurement_protocol) >= 3
    assert pred.optimized_selectivity >= pred.untuned_selectivity - 0.05
    assert pred.untuned_delta_T_K > 0.0


def test_openems_export_generates_runnable_script():
    from mw_inv.openems_export import generate_openems_script

    text = generate_openems_script()
    assert "InitCSX" in text
    assert "AddLumpedPort" in text
    assert "RunOpenEMS" in text
    assert "ReadHDF5Dump" in text


def test_openems_stub_alias():
    from mw_inv.openems_stub import generate_openems_stub

    text = generate_openems_stub()
    assert "InitCSX" in text
    assert "freq" in text


def test_saline_eps_monotonic():
    from mw_inv.phantom_data import saline_eps

    e0 = saline_eps(0.0)
    e3 = saline_eps(3.0)
    assert e3.real > e0.real
    assert e3.imag > e0.imag


def test_meep_compare_skips_without_meep():
    mats = Materials.from_pair("pyrite_in_calcite")
    scene = build_scene(GRID, CavityParams(), mats)
    cmp = compare_fdfd_meep(scene, GRID, 0.75)
    if not meep_available():
        assert cmp.get("skipped") is True
    else:
        assert "meep_2d_selectivity" in cmp
