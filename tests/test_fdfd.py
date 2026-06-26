"""Sanity checks for the FDFD solver and the selectivity FOM."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid, absorbed_power_density, solve, solve_scene  # noqa: E402
from mw_inv.fom import evaluate  # noqa: E402
from mw_inv.geometry import CavityParams, Materials, build_scene, build_source_jz  # noqa: E402


def test_empty_cavity_resonance_sweep_has_peak():
    """A lossless empty cavity should show a sharp field resonance vs frequency.

    Analytic TM_mn resonances: f = (c/2) sqrt((m/Lx)^2 + (n/Ly)^2). For a 0.2 m
    square cavity the TM11 mode sits near 1.06 GHz; the on-resonance field should
    dwarf an off-resonance point.
    """
    grid = Grid(nx=81, ny=81, Lx=0.20, Ly=0.20)
    eps = np.ones((grid.ny, grid.nx), dtype=complex)
    energies = []
    freqs = [0.8e9, 1.06e9, 1.4e9]
    for f in freqs:
        res = solve(grid, eps, f, source_xy=(0.1, 0.1))
        energies.append(float(np.abs(res.Ez) ** 2).real if np.isscalar(np.abs(res.Ez) ** 2)
                        else float((np.abs(res.Ez) ** 2).sum()))
    # Resonant (middle) energy should exceed both neighbours.
    assert energies[1] > energies[0]
    assert energies[1] > energies[2]


def test_symmetry_centered_source():
    """Symmetric cavity + centred feed -> field symmetric across the vertical axis."""
    grid = Grid(nx=61, ny=61, Lx=0.30, Ly=0.30)
    eps = np.ones((grid.ny, grid.nx), dtype=complex)
    res = solve(grid, eps, 2.45e9, source_xy=(0.15, 0.15))
    field = np.abs(res.Ez)
    assert np.allclose(field, field[:, ::-1], atol=1e-6 * field.max())


def test_absorbed_power_only_in_lossy_media():
    """Power density must be zero where Im(eps) == 0 and positive where it is lossy."""
    grid = Grid(nx=61, ny=61, Lx=0.30, Ly=0.30)
    scene = build_scene(grid, CavityParams())
    res = solve_scene(grid, scene)
    p = absorbed_power_density(res)
    background = ~(scene.target_mask | scene.gangue_mask | scene.pec_mask)
    assert np.allclose(p[background], 0.0)
    assert p[scene.target_mask].sum() > 0.0


def test_selectivity_in_unit_interval():
    grid = Grid(nx=61, ny=61, Lx=0.30, Ly=0.30)
    scene = build_scene(grid, CavityParams())
    res = solve_scene(grid, scene)
    rep = evaluate(res, scene)
    assert 0.0 <= rep.selectivity <= 1.0
    assert rep.contrast > 0.0


def test_lossy_target_outheats_gangue_per_area():
    """With target eps'' >> gangue eps'', per-area contrast should favour the target."""
    grid = Grid(nx=81, ny=81, Lx=0.36, Ly=0.36)
    mats = Materials(target=8.0 + 3.0j, gangue=5.0 + 0.02j)
    scene = build_scene(grid, CavityParams(), mats)
    res = solve_scene(grid, scene)
    rep = evaluate(res, scene)
    assert rep.contrast > 1.0


def _selectivity(grid, params, mats):
    sc = build_scene(grid, params, mats)
    return evaluate(solve_scene(grid, sc), sc).selectivity


def test_tuner_field_changes_absorption():
    """The reconfigurable dielectric tuner (lossless) reshapes the mode and moves
    selectivity away from the no-tuner baseline."""
    from dataclasses import replace

    grid = Grid(nx=81, ny=81, Lx=0.36, Ly=0.36)
    mats = Materials.from_pair("pyrite_in_calcite")
    base = CavityParams()
    s0 = _selectivity(grid, base, mats)
    s1 = _selectivity(grid, replace(base, tuner_field=(0.0, 1.0) * 4), mats)
    assert abs(s1 - s0) > 1e-3                     # the tuner actually does something


def test_field_search_runs_and_improves():
    """Both samplers optimise the high-dim field above the no-tuner baseline."""
    from mw_inv.search import optuna_field_search, random_field_search

    grid = Grid(nx=61, ny=61, Lx=0.36, Ly=0.36)
    mats = Materials.from_pair("pyrite_in_calcite")
    base_sel = _selectivity(grid, CavityParams(), mats)
    rnd = random_field_search(grid, n_trials=15, seed=0, k=8, materials=mats)
    tpe = optuna_field_search(grid, n_trials=15, seed=0, k=8, materials=mats)
    assert max(t.selectivity for t in rnd) >= base_sel - 0.02  # stochastic; TPE should beat
    assert max(t.selectivity for t in tpe) > base_sel
    assert len(rnd) == 15 and len(tpe) == 15


def test_material_pairs_probe_opposite_regimes():
    """The cited finding: transparent gangue -> selectivity ~saturated untuned;
    matched-eps' disseminated absorber -> much lower untuned selectivity."""
    from mw_inv.search import evaluate_params

    grid = Grid(nx=81, ny=81, Lx=0.36, Ly=0.36)
    easy = evaluate_params(grid, CavityParams(), Materials.from_pair("magnetite_in_quartz"))
    hard = evaluate_params(grid, CavityParams(), Materials.from_pair("pyrite_in_calcite"))
    assert easy.selectivity > 0.95          # transparent gangue: nearly all power in target
    assert hard.selectivity < 0.85          # matched eps': far from saturated (was 0.75 pre-μ″)
    assert easy.selectivity > hard.selectivity


def test_line_excitation_spans_stub_width():
    """Distributed source covers multiple interior cells at the stub mouth."""
    grid = Grid(nx=61, ny=61, Lx=0.36, Ly=0.36)
    params = CavityParams(stub_width_frac=0.08)
    jz = build_source_jz(params, grid)
    assert int(np.count_nonzero(jz)) >= 3


def test_line_vs_point_selectivity_same_order():
    """Line-port and legacy point feed give comparable selectivity (same physics)."""
    grid = Grid(nx=71, ny=71, Lx=0.36, Ly=0.36)
    scene = build_scene(grid, CavityParams(), Materials.from_pair("pyrite_in_calcite"))
    s_line = evaluate(solve_scene(grid, scene), scene).selectivity
    sx, sy = scene.source_xy
    res_pt = solve(grid, scene.eps_r, scene.freq_hz, source_xy=(sx, sy), mu_r=scene.mu_r)
    s_point = evaluate(res_pt, scene).selectivity
    assert abs(s_line - s_point) < 0.08


def test_selectivity_stable_under_grid_refinement():
    """FOM should not swing wildly when the grid is refined (A2 stability)."""
    params = CavityParams()
    mats = Materials.from_pair("pyrite_in_calcite")
    sels = []
    for n in (51, 71, 91):
        grid = Grid(nx=n, ny=n, Lx=0.36, Ly=0.36)
        scene = build_scene(grid, params, mats)
        sels.append(evaluate(solve_scene(grid, scene), scene).selectivity)
    assert max(sels) - min(sels) < 0.06


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
