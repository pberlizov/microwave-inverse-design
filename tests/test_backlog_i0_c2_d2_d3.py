"""Backlog I0/C2/D2/D3 implementation smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.deposit_envelope import discover_ore_json_paths, evaluate_deposit_envelope  # noqa: E402
from mw_inv.design_evaluator import DesignEvaluator, preset_config  # noqa: E402
from mw_inv.ensemble import (  # noqa: E402
    evaluate_material_robust,
    evaluate_particle_power,
    evaluate_frequency_robust,
)
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402
from mw_inv.ism_band import IsmBandConfig, IsmBandMode  # noqa: E402
from mw_inv.materials import Materials  # noqa: E402
from mw_inv.ore_profiles import load_ore_profile  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data"
GRID = Grid(nx=51, ny=51, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")


def test_industrial_metrics_in_design_report():
    rep = DesignEvaluator(GRID, preset_config("em", materials=MATS)).evaluate(CavityParams())
    assert "gangue_power_fraction" in rep.foms
    assert "target_power_fraction" in rep.foms
    assert rep.foms["gangue_power_fraction"] + rep.foms["target_power_fraction"] == pytest.approx(1.0, rel=0.02)
    assert rep.foms["charge_tonnes_proxy"] is not None


def test_ism_band_tunable_subset():
    band = IsmBandConfig(mode=IsmBandMode.TUNABLE, tolerance_mhz=25.0, n_samples=3)
    freqs = band.freqs_hz()
    assert len(freqs) == 3
    assert float(freqs.min()) >= 2.45e9 - 26e6
    assert float(freqs.max()) <= 2.45e9 + 26e6


def test_frequency_robust_respects_band():
    full = evaluate_frequency_robust(
        GRID, CavityParams(), MATS,
        band=IsmBandConfig(mode=IsmBandMode.FULL, n_samples=5),
    )
    tunable = evaluate_frequency_robust(
        GRID, CavityParams(), MATS,
        band=IsmBandConfig(mode=IsmBandMode.TUNABLE, tolerance_mhz=10.0, n_samples=3),
    )
    assert full.n_freqs == 5
    assert tunable.n_freqs == 3


def test_particle_power_fractions_sum_on_charge():
    p = evaluate_particle_power(GRID, CavityParams(), MATS)
    assert p.n_particles == len(CavityParams().inclusion_offsets_frac)
    total = sum(p.particle_power_fractions) + p.gangue_power_fraction
    assert total == pytest.approx(1.0, rel=0.05)


def test_deposit_envelope_forster_subset():
    forster = ROOT / "ores" / "forster"
    if not forster.is_dir():
        pytest.skip("forster ores not ingested")
    paths = discover_ore_json_paths(forster)[:6]
    rep = evaluate_deposit_envelope(paths, GRID, CavityParams())
    assert rep.n_ok == len(paths)
    assert rep.min_selectivity <= rep.mean_selectivity + 1e-9


def test_material_robust_percentiles_and_coupling():
    ore_path = ROOT / "ores" / "disseminated_pyrite_porphyry_measured_example.json"
    if not ore_path.is_file():
        pytest.skip("measured ore example missing")
    ore = load_ore_profile(ore_path)
    rep = evaluate_material_robust(
        GRID, CavityParams(), ore, ore_profile_path=str(ore_path), n_scenarios=5, seed=1,
    )
    assert rep.p05_selectivity <= rep.mean_selectivity + 1e-9
    assert rep.p95_selectivity >= rep.mean_selectivity - 1e-9
    assert rep.min_coupling_eff <= rep.mean_coupling_eff + 1e-9
