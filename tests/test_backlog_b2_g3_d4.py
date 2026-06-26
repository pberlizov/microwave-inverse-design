"""Backlog B2 (Dirichlet PEC), G3 (campaigns), D4 (deposit calibration)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.campaign import discover_campaign_files, load_campaign, resolve_ore_paths  # noqa: E402
from mw_inv.deposit_calibration import calibrate_ore_profile  # noqa: E402
from mw_inv.design_evaluator import preset_config, evaluate_design  # noqa: E402
from mw_inv.fdfd import Grid, solve, solve_scene  # noqa: E402
from mw_inv.fom import evaluate  # noqa: E402
from mw_inv.geometry import CavityParams, Materials, build_scene  # noqa: E402
from mw_inv.industrial_metrics import IndustrialMetrics  # noqa: E402


GRID = Grid(nx=61, ny=61, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")
DATA = Path(__file__).resolve().parents[1] / "data"


def _fom_point(params: CavityParams):
    sc = build_scene(GRID, params, MATS)
    return evaluate(
        solve(GRID, sc.eps_r, sc.freq_hz, source_xy=sc.source_xy, mu_r=sc.mu_r,
              dirichlet_mask=sc.pec_mask if params.structure_model == "dirichlet" else None),
        sc,
    )


def test_dirichlet_vs_lossy_baffle_pec_loss():
    """Point-feed regression: lossy Im(eps) baffle absorbs; Dirichlet metal does not."""
    f_lossy = _fom_point(
        replace(CavityParams(), baffle_len_frac=0.4, structure_model="lossy_imag"),
    )
    f_dir = _fom_point(
        replace(CavityParams(), baffle_len_frac=0.4, structure_model="dirichlet"),
    )
    assert f_lossy.pec_loss_fraction > 0.5
    assert f_dir.pec_loss_fraction < 0.05


def test_forster_campaign_discovery():
    files = discover_campaign_files(DATA)
    assert any("forster_literature_v1" in str(p) for p in files)
    camp = load_campaign(DATA / "campaigns" / "forster_literature_v1" / "campaign.json")
    paths = resolve_ore_paths(camp, DATA)
    assert len(paths) >= 40
    assert all(p.name.startswith("forster_") for p in paths)


def test_deposit_calibration_forster_ore():
    ore_path = DATA / "ores" / "forster" / "forster_good_pyrite_calcite.json"
    if not ore_path.is_file():
        pytest.skip("forster ore fixture missing")
    rep = calibrate_ore_profile(ore_path)
    assert len(rep.phases) == 2
    assert rep.max_rel_error_real >= 0.0


def test_composite_industrial_preset():
    cfg = preset_config("composite:industrial", pair_label="pyrite_in_calcite")
    rep = evaluate_design(GRID, CavityParams(), cfg)
    assert rep.objective_key == "composite"
    assert "gangue_budget" in rep.foms
    assert rep.score > 0.0


def test_industrial_throughput_proxy():
    scene = build_scene(GRID, CavityParams(), MATS)
    fom = evaluate(solve_scene(GRID, scene), scene)
    from mw_inv.ore_profiles import charge_volume_m3

    vol = charge_volume_m3(CavityParams(), Lx=GRID.Lx, Ly=GRID.Ly)
    ind = IndustrialMetrics.from_fom(fom, charge_volume_m3=vol, residence_time_s=60.0)
    assert ind.throughput_proxy_t_per_h is not None
    assert ind.throughput_proxy_t_per_h > 0.0
    assert ind.delivered_kw_proxy is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
