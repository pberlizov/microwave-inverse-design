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

_C0 = 2.99792458e8  # speed of light, m/s


def skin_depth_m(freq_hz: float, eps_real: float, eps_imag: float) -> float:
    """Power penetration depth (1/e of absorbed power) of a plane wave in a medium with
    relative permittivity eps_real + i*eps_imag, in metres.

    Exact: alpha = (omega/c) * |eps|^1/2 * sin(theta/2) is the field attenuation
    constant (theta = arg(eps')), and the power 1/e depth is 1/(2 alpha). This is the
    length scale a grain is large or small *relative to* -- the quantity that decides
    whether the field penetrates the grain (absorption rises with eps'') or is expelled
    from it (self-shielding, absorption falls)."""
    mag = float(np.hypot(eps_real, eps_imag))
    theta = float(np.arctan2(eps_imag, eps_real))
    alpha = (2.0 * np.pi * freq_hz / _C0) * np.sqrt(mag) * np.sin(theta / 2.0)
    return float("inf") if alpha <= 0 else 1.0 / (2.0 * alpha)


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
        result = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r)
        r = evaluate(result, scene)
        out.append(
            FreqPoint(float(f), r.selectivity, r.contrast, r.p_target, r.p_total_charge)
        )
    return out


# ---------------------------------------------------------------------------
# 2. Temperature dependence + thermal runaway (parametric eps''(T))
# ---------------------------------------------------------------------------

from mw_inv.dielectric_data import EpsTModel  # noqa: E402  (re-export for scripts/tests)


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
    base = base_materials or Materials.from_pair("pyrite_in_calcite")
    base = replace(base, target_mu=1.0 + 0.0j)
    out: list[LossPoint] = []
    for epp in eps_imag_values:
        mats = replace(base, target=complex(eps_real, float(epp)))
        scene = build_scene(grid, params, mats)
        r = evaluate(solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r), scene)
        out.append(LossPoint(float(epp), r.p_target, r.selectivity))
    return out


# ---------------------------------------------------------------------------
# 2c. Grain size vs skin depth: where does the absorption turnover happen?
# ---------------------------------------------------------------------------

@dataclass
class GrainRow:
    diameter_m: float            # 2 * inclusion radius
    turnover_eps_imag: float     # eps''* at peak mean absorbed power density (nan if none)
    skin_depth_at_turnover_m: float
    ratio_d_over_delta: float    # grain diameter / skin depth at the turnover
    monotonic: bool              # True -> no interior peak in range (runaway-prone)
    eps_imag: np.ndarray
    mean_power_density: np.ndarray


def grain_size_sweep(
    grid: Grid,
    radius_fracs: np.ndarray,
    eps_imag_values: np.ndarray,
    eps_real: float,
    base_materials: Materials | None = None,
    base_params: CavityParams | None = None,
) -> list[GrainRow]:
    """For each inclusion size, sweep the loss factor and find where the *mean absorbed
    power density* in the grain turns over.

    The turnover is the self-shielding onset. Because skin depth shrinks as eps'' grows,
    sweeping eps'' at fixed grain size traverses the grain/skin-depth ratio, and the
    turnover lands where the grain diameter ~ skin depth. Small grains (diameter < skin
    depth even at the largest eps'') never turn over -> absorption keeps rising with
    eps'' -> positive feedback -> runaway-prone. Large grains turn over early ->
    self-limiting. A single centred inclusion is used so size varies without grains
    merging."""
    base = base_materials or Materials()
    bp = base_params or CavityParams()
    cell = grid.dx * grid.dy
    rows: list[GrainRow] = []
    for rf in radius_fracs:
        params = replace(bp, inclusion_offsets_frac=((0.0, 0.0),),
                         inclusion_radius_frac=float(rf))
        mean_p = np.empty(eps_imag_values.shape)
        for i, epp in enumerate(eps_imag_values):
            mats = replace(base, target=complex(eps_real, float(epp)))
            scene = build_scene(grid, params, mats)
            r = evaluate(solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r), scene)
            n = int(scene.target_mask.sum())
            mean_p[i] = r.p_target / (n * cell) if n else 0.0
        i_peak = int(np.argmax(mean_p))
        monotonic = i_peak >= len(mean_p) - 1
        eps_star = float(eps_imag_values[i_peak])
        delta = skin_depth_m(bp.freq_hz, eps_real, eps_star)
        diameter = 2.0 * rf * min(grid.Lx, grid.Ly)
        rows.append(GrainRow(
            diameter_m=diameter,
            turnover_eps_imag=float("nan") if monotonic else eps_star,
            skin_depth_at_turnover_m=delta,
            ratio_d_over_delta=diameter / delta if np.isfinite(delta) else 0.0,
            monotonic=monotonic,
            eps_imag=np.asarray(eps_imag_values, dtype=float),
            mean_power_density=mean_p,
        ))
    return rows


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
        result = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r)
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
