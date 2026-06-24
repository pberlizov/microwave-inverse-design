"""Frequency and temperature sweeps over the forward model.

Two analyses:

1. ``frequency_sweep`` -- REAL electromagnetics. Re-solves the FDFD problem across a
   band of drive frequencies and reports how selective heating varies. Nothing
   parametric here beyond the cited material permittivities.

2. ``loss_response`` + ``runaway_curve`` -- temperature / loss-factor analysis.
   Microwave loss in good absorbers RISES with temperature, which *could* create
   positive feedback (hotter -> more absorption -> hotter -> runaway). But absorbed
   power vs eps'' is NON-monotonic: it peaks at an impedance/skin-depth-matched optimum
   eps''* and then FALLS as the grain expels the field (self-shielding). For grains
   larger than the skin depth -- as here -- the operating point quickly passes eps''*,
   so the feedback turns negative and heating is SELF-LIMITING rather than runaway.
   ``runaway_curve`` integrates the lumped energy balance and shows this bounded,
   smooth temperature response. (True unbounded runaway needs grains << skin depth so
   absorption keeps climbing with eps''; that regime is flagged, not modelled here.)

Provenance of the temperature behaviour (qualitative, from primary literature -- the
exact eps''(T) curve here is a MODEL, not a digitised measurement):
  - Pyrite / chalcopyrite / chalcocite: eps' and eps'' vary *significantly* with
    temperature, measured ambient->650 C at 615/1410/2210 MHz. Cumbane et al. (2008).
  - Galena, sphalerite: little variation up to ~500 C (so not all sulphides ramp).
  - Some ores reach loss tangent ~1 by ~1000 C at 2.45 GHz -- a magnitude ceiling.
  - Carbonate/silicate gangue (calcite, quartz) stays low-loss ("inactive"), so the
    ramp is *asymmetric*: the target runs away while the gangue does not. That
    asymmetry is the selective-heating / thermally-assisted-liberation mechanism
    (Salsman et al. 1996: pyrite-in-calcite stress concentration).
See docs/MATERIALS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mw_inv.fdfd import Grid, solve
from mw_inv.fom import evaluate
from mw_inv.geometry import CavityParams, Materials, build_scene


# ---------------------------------------------------------------------------
# 1. Frequency sweep (real EM)
# ---------------------------------------------------------------------------

@dataclass
class FreqPoint:
    freq_hz: float
    selectivity: float
    contrast: float
    p_target: float
    p_charge: float


def frequency_sweep(
    grid: Grid,
    freqs_hz: np.ndarray,
    params: CavityParams | None = None,
    materials: Materials | None = None,
) -> list[FreqPoint]:
    params = params or CavityParams()
    out: list[FreqPoint] = []
    for f in freqs_hz:
        scene = build_scene(grid, replace(params, freq_hz=float(f)), materials)
        result = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy)
        r = evaluate(result, scene)
        out.append(
            FreqPoint(float(f), r.selectivity, r.contrast, r.p_target, r.p_total_charge)
        )
    return out


# ---------------------------------------------------------------------------
# 2. Temperature dependence + thermal runaway (parametric eps''(T))
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpsTModel:
    """Arrhenius-like ramp of the loss factor: eps''(T) grows with temperature,
    capped at a maximum loss tangent. eps' held fixed (a simplification; eps' also
    rises in reality). Defaults loosely anchored to "loss tangent approaches ~1 at
    high T" for a strongly microwave-active sulphide; NOT a fit to a specific ore."""

    eps_real: float
    eps_imag_ref: float          # eps'' at reference temperature
    T_ref_K: float = 298.0
    activation_K: float = 1000.0  # Ea/k; larger = steeper ramp (gradual across 300-1300K)
    max_loss_tangent: float = 0.6
    ramps_with_T: bool = True     # False -> inert phase (e.g. calcite gangue)

    def eps_imag(self, T_K: float) -> float:
        if not self.ramps_with_T:
            return self.eps_imag_ref
        factor = np.exp(self.activation_K * (1.0 / self.T_ref_K - 1.0 / T_K))
        e_imag = self.eps_imag_ref * factor
        return float(min(e_imag, self.max_loss_tangent * self.eps_real))

    def eps(self, T_K: float) -> complex:
        return self.eps_real + 1j * self.eps_imag(T_K)


@dataclass
class LossPoint:
    eps_imag: float
    p_target: float
    selectivity: float


def loss_response(
    grid: Grid,
    eps_imag_values: np.ndarray,
    eps_real: float,
    params: CavityParams | None = None,
    base_materials: Materials | None = None,
) -> list[LossPoint]:
    """Real-EM sweep of the target's loss factor eps''. Absorbed power in the target is
    NON-monotonic: it peaks at an impedance/skin-depth-matched optimum eps''* and then
    falls as the grain expels the field (self-shielding). 'More loss' is not 'more heat'.
    Selectivity, by contrast, keeps rising. eps' is held fixed."""
    params = params or CavityParams()
    base = base_materials or Materials()
    out: list[LossPoint] = []
    for epp in eps_imag_values:
        mats = replace(base, target=complex(eps_real, float(epp)))
        scene = build_scene(grid, params, mats)
        r = evaluate(solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy), scene)
        out.append(LossPoint(float(epp), r.p_target, r.selectivity))
    return out


@dataclass
class RunawayResult:
    temps_K: np.ndarray          # temperature grid for G(T)
    p_gen: np.ndarray            # absorbed power in target at unit drive vs T (G(T))
    drives: np.ndarray           # swept drive levels
    T_steady: np.ndarray         # steady-state target temperature vs drive
    critical_drive: float        # smallest drive whose steady state exceeds T_runaway
    cooling_coeff: float
    T_runaway_K: float


def power_vs_temperature(
    grid: Grid,
    temps_K: np.ndarray,
    eps_t: EpsTModel,
    params: CavityParams | None = None,
    base_materials: Materials | None = None,
) -> np.ndarray:
    """Absorbed power in the target region at unit drive, re-solving the field with the
    target loss factor set by eps_t at each temperature. The generation curve G(T)."""
    params = params or CavityParams()
    base = base_materials or Materials()
    p_gen = np.empty(temps_K.shape)
    for i, T in enumerate(temps_K):
        mats = replace(base, target=eps_t.eps(float(T)))
        scene = build_scene(grid, params, mats)
        result = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy)
        p_gen[i] = evaluate(result, scene).p_target
    return p_gen


def _steady_state_T(
    drive: float,
    T_grid: np.ndarray,
    g_grid: np.ndarray,
    cooling_coeff: float,
    T_amb_K: float,
    T_max_K: float,
) -> float:
    """Integrate dT/dt = drive*G(T) - cooling_coeff*(T - T_amb) to steady state.

    G(T) is interpolated from the precomputed grid (clamped at the ends -- beyond the
    grid the loss factor has saturated at its cap, so G is flat). Returns the steady
    temperature, capped at T_max_K (reaching the cap == thermal runaway / phase jump)."""
    T = T_amb_K
    dt = 0.5
    for _ in range(20000):
        G = float(np.interp(T, T_grid, g_grid))
        dT = dt * (drive * G - cooling_coeff * (T - T_amb_K))
        T += dT
        if T >= T_max_K:
            return T_max_K
        if abs(dT) < 1e-6 and T > T_amb_K:
            break
        if T < T_amb_K:
            T = T_amb_K
    return T


def runaway_curve(
    grid: Grid,
    eps_t: EpsTModel,
    cooling_coeff: float,
    temps_K: np.ndarray | None = None,
    params: CavityParams | None = None,
    base_materials: Materials | None = None,
    T_amb_K: float = 298.0,
    drives: np.ndarray | None = None,
) -> RunawayResult:
    """Sweep drive level; for each, integrate the lumped energy balance to steady state.

    When the target loss factor ramps with temperature, generation G(T) accelerates and
    the steady-state temperature jumps almost vertically past a critical drive -- thermal
    runaway. An inert phase (``ramps_with_T=False``) gives a gentle, monotone T(drive)
    with no jump. The contrast between the two critical drives is the selective-heating
    margin."""
    if temps_K is None:
        temps_K = np.linspace(T_amb_K, 1300.0, 60)
    p_gen = power_vs_temperature(grid, temps_K, eps_t, params, base_materials)
    T_runaway = float(temps_K[-1])

    if drives is None:
        # Scale so the interesting jump lands inside the window.
        d0 = cooling_coeff * (T_runaway - T_amb_K) / max(p_gen.max(), 1e-30)
        drives = np.linspace(0.0, 4.0 * d0, 80)

    T_steady = np.array([
        _steady_state_T(float(s), temps_K, p_gen, cooling_coeff, T_amb_K, T_runaway)
        for s in drives
    ])
    over = np.where(T_steady >= 0.9 * T_runaway)[0]
    critical = float(drives[over[0]]) if over.size else float("inf")

    return RunawayResult(
        temps_K=temps_K,
        p_gen=p_gen,
        drives=drives,
        T_steady=T_steady,
        critical_drive=critical,
        cooling_coeff=cooling_coeff,
        T_runaway_K=T_runaway,
    )
