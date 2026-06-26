"""E0 bench ingest, D1 PSD layouts, C1 material robustness tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.bench_ingest import validate_lab_measurements  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402
from mw_inv.ensemble import evaluate_ensemble, evaluate_material_robust  # noqa: E402
from mw_inv.ore_profiles import (  # noqa: E402
    load_ore_profile,
    psd_radii_frac,
    sample_psd_radii_m,
)
from mw_inv.phantom_calibration import evaluate_bench_gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "lab_measurements.example.json"
EPS = ROOT / "data" / "measured_eps.example.json"
ORE = ROOT / "data" / "ores" / "disseminated_pyrite_porphyry.json"


def test_validate_lab_measurements_example() -> None:
    issues = validate_lab_measurements(LAB)
    assert not issues


def test_bench_gate_includes_model_checks() -> None:
    report = evaluate_bench_gate(
        "saline_2_vs_0.5",
        EPS,
        LAB,
        bench_grid=31,
        bench_trials=4,
    )
    names = {c.name for c in report.checks}
    assert "model_rank_optimized_beats_untuned" in names
    assert "model_delta_t_tolerance" in names


def test_psd_samples_multiple_radii() -> None:
    import numpy as np

    ore = load_ore_profile(ORE)
    assert ore.texture is not None
    rng = np.random.default_rng(0)
    radii = sample_psd_radii_m(ore.texture, 5, rng)
    assert len(radii) == 5
    assert max(radii) > min(radii)


def test_ensemble_psd_layout_runs() -> None:
    ore = load_ore_profile(ORE)
    grid = Grid(nx=31, ny=31, Lx=0.36, Ly=0.36)
    rep = evaluate_ensemble(
        grid,
        CavityParams(),
        n_realizations=2,
        n_grains=4,
        seed=3,
        ore=ore,
    )
    assert rep.min_selectivity >= 0.0


def test_material_robust_spreads_with_moisture() -> None:
    ore_path = ROOT / "data" / "ores" / "disseminated_pyrite_porphyry_measured_example.json"
    if not ore_path.is_file():
        pytest.skip("measured ore example missing")
    ore = load_ore_profile(ore_path)
    grid = Grid(nx=31, ny=31, Lx=0.36, Ly=0.36)
    rep = evaluate_material_robust(
        grid,
        CavityParams(),
        ore,
        ore_profile_path=str(ore_path),
        n_scenarios=3,
        seed=9,
    )
    assert rep.n_scenarios == 3
    assert rep.min_selectivity <= rep.mean_selectivity + 1e-9
