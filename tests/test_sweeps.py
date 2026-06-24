"""Tests for the frequency / loss-factor / thermal sweeps."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "src")

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import Materials  # noqa: E402
from mw_inv.sweeps import (  # noqa: E402
    EpsTModel,
    frequency_sweep,
    grain_size_sweep,
    loss_response,
    runaway_curve,
    skin_depth_m,
)

GRID = Grid(nx=71, ny=71, Lx=0.36, Ly=0.36)
MATS = Materials.from_pair("pyrite_in_calcite")


def test_frequency_sweep_varies_selectivity():
    """Real EM: selectivity should actually move across the band (frequency is a knob)."""
    freqs = np.linspace(2.35e9, 2.55e9, 11)
    pts = frequency_sweep(GRID, freqs, materials=MATS)
    sel = np.array([p.selectivity for p in pts])
    assert len(pts) == 11
    assert sel.max() - sel.min() > 0.02          # non-trivial spread
    assert np.all((sel >= 0.0) & (sel <= 1.0))


def test_loss_response_is_non_monotonic_with_a_peak():
    """Absorbed power vs eps'' peaks at an interior optimum (skin-depth/impedance match),
    not at the largest loss -- the self-shielding result."""
    epps = np.linspace(0.02, 8.0, 25)
    pts = loss_response(GRID, epps, eps_real=8.0, base_materials=MATS)
    p = np.array([q.p_target for q in pts])
    sel = np.array([q.selectivity for q in pts])
    i_peak = int(np.argmax(p))
    assert 0 < i_peak < len(p) - 1                # interior peak, not an endpoint
    assert p[i_peak] > p[-1]                      # high-loss end is self-shielded
    assert sel[-1] > sel[0]                       # selectivity still rises monotonically-ish


def test_eps_t_model_ramps_and_caps():
    m = EpsTModel(eps_real=8.0, eps_imag_ref=0.3, activation_K=1000.0, max_loss_tangent=0.6)
    assert m.eps_imag(298) == pytest.approx(0.3, rel=1e-3)
    assert m.eps_imag(1000) > m.eps_imag(298)     # rises with temperature
    assert m.eps_imag(5000) <= 0.6 * 8.0 + 1e-9   # capped at max loss tangent
    inert = EpsTModel(eps_real=8.0, eps_imag_ref=0.3, ramps_with_T=False)
    assert inert.eps_imag(1000) == pytest.approx(0.3)


def test_thermal_response_is_self_limiting():
    """Steady-state temperature rises smoothly with drive (no downward jump) -- bounded,
    self-limiting heating rather than unbounded runaway for grains > skin depth."""
    eps_t = EpsTModel(eps_real=8.0, eps_imag_ref=0.3, activation_K=1000.0, max_loss_tangent=0.6)
    temps = np.linspace(298.0, 1300.0, 40)
    run = runaway_curve(GRID, eps_t, cooling_coeff=5e-3, temps_K=temps, base_materials=MATS)
    assert run.T_steady[0] == pytest.approx(298.0, abs=1.0)
    assert np.all(np.diff(run.T_steady) >= -1.0)  # monotone up in drive
    assert run.T_steady[-1] > run.T_steady[0]


def test_skin_depth_decreases_with_loss():
    f = 2.45e9
    d_low = skin_depth_m(f, 8.0, 0.3)
    d_high = skin_depth_m(f, 8.0, 8.0)
    assert d_low > d_high > 0                      # more loss -> shallower penetration
    assert d_high < 0.05 and d_low > 0.05          # ~mm vs ~cm-dm scale at 2.45 GHz
    assert np.isinf(skin_depth_m(f, 8.0, 0.0))     # lossless -> infinite penetration


def test_grain_crossover_larger_grains_self_limit_sooner():
    """Larger grains hit the absorption turnover at a *lower* loss factor; the smallest
    grain may not turn over at all (runaway-prone). Turnover ~ grain size ~ skin depth."""
    epps = np.geomspace(0.1, 8.0, 10)
    rows = grain_size_sweep(GRID, np.array([0.04, 0.11]), epps, eps_real=8.0,
                            base_materials=MATS)
    small, large = rows
    assert large.diameter_m > small.diameter_m
    # large grain must turn over within range
    assert not large.monotonic
    # and at a lower loss factor than the small grain (which may be monotonic)
    small_star = np.inf if small.monotonic else small.turnover_eps_imag
    assert large.turnover_eps_imag < small_star
    # turnover lands at grain ~ skin depth (order unity, generous bounds for 2D/discrete)
    assert 0.3 < large.ratio_d_over_delta < 6.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
