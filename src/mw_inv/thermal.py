"""Spatial steady-state EM–thermal coupling on the FDFD grid.

Coupling loop (quasi-steady, 2D, per unit depth):

    1. Build ε(T), μ(T) from local temperature via ``dielectric_data`` tables.
    2. Solve FDFD → absorbed power density q(x, y) [W/m³].
    3. Solve k∇²T − h(T−T_amb) + q = 0 with cooled cavity walls (Dirichlet T = T_amb).
    4. Repeat until T converges.

Transient mode (``simulate_transient``): explicit dT/dt with the same q and ε(T)
re-solved periodically — reports time-to-threshold for target vs gangue (runaway timing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mw_inv.fdfd import Grid, SolveResult, absorbed_power_density, solve, solve_scene
from mw_inv.fom import FomReport, evaluate
from mw_inv.geometry import CavityParams, Materials, Scene, build_scene, build_scene_at_T


@dataclass(frozen=True)
class PhaseThermalProps:
    """Representative thermal properties per phase (not measured ore-specific)."""

    target_k: float = 15.0       # W/m/K — pyrite / magnetite
    gangue_k: float = 3.0        # W/m/K — calcite / quartz
    background_k: float = 0.026  # air
    target_rho_cp: float = 3.0e6   # J/m³/K — pyrite ~5000 kg/m³ × ~600 J/kg/K
    gangue_rho_cp: float = 2.4e6   # calcite ~2700 × ~900
    background_rho_cp: float = 1200.0  # air — unused where Q ≈ 0


@dataclass
class ThermalConfig:
    T_amb_K: float = 298.0
    T_max_K: float = 1273.0      # clip T and ε lookup to anchor table span
    drive: float = 1.0           # scales microwave source / absorbed power
    bulk_cooling: float = 4.0e4  # W/m³/K volumetric loss h*(T−T_amb); balances Q at steady state
    max_iters: int = 25
    tol_K: float = 2.0           # max |ΔT| in charge for convergence
    relax: float = 0.45          # under-relaxation for stable ε(T) feedback
    thermal_props: PhaseThermalProps = field(default_factory=PhaseThermalProps)


@dataclass
class ThermalReport:
    """Spatial thermal figures of merit at coupled steady state."""

    T_mean_target_K: float
    T_mean_gangue_K: float
    delta_T_K: float             # target minus gangue mean (selective heating)
    T_max_target_K: float
    heat_selectivity: float        # target heat / (target + gangue) from q field
    em_selectivity: float          # same ratio from final EM solve
    em_contrast: float
    converged: bool
    n_iters: int
    max_delta_K: float

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "T_mean_target_K": self.T_mean_target_K,
            "T_mean_gangue_K": self.T_mean_gangue_K,
            "delta_T_K": self.delta_T_K,
            "T_max_target_K": self.T_max_target_K,
            "heat_selectivity": self.heat_selectivity,
            "em_selectivity": self.em_selectivity,
            "em_contrast": self.em_contrast,
            "converged": self.converged,
            "n_iters": self.n_iters,
            "max_delta_K": self.max_delta_K,
        }


@dataclass
class CoupledResult:
    temperature_K: np.ndarray      # (ny, nx)
    heat_generation: np.ndarray    # q''' [W/m³] at convergence
    em: SolveResult
    scene: Scene
    em_report: FomReport
    thermal: ThermalReport
    history_max_delta: list[float] = field(default_factory=list)


def _build_k_map(scene: Scene, props: PhaseThermalProps) -> np.ndarray:
    k = np.full((scene.grid.ny, scene.grid.nx), props.background_k, dtype=float)
    k[scene.gangue_mask] = props.gangue_k
    k[scene.target_mask] = props.target_k
    k[scene.pec_mask] = props.gangue_k
    return k


def _build_rho_cp_map(scene: Scene, props: PhaseThermalProps) -> np.ndarray:
    rho = np.full((scene.grid.ny, scene.grid.nx), props.background_rho_cp, dtype=float)
    rho[scene.gangue_mask] = props.gangue_rho_cp
    rho[scene.target_mask] = props.target_rho_cp
    rho[scene.pec_mask] = props.gangue_rho_cp
    return rho


def _laplacian_dirichlet(grid: Grid, T: np.ndarray, T_amb_K: float) -> np.ndarray:
    """Five-point Laplacian; wall neighbours replaced by T_amb (Dirichlet)."""
    ny, nx = grid.ny, grid.nx
    dx2, dy2 = grid.dx ** 2, grid.dy ** 2
    lap = np.zeros_like(T)
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            t_c = T[iy, ix]
            lap[iy, ix] = (
                (T[iy, ix + 1] + T[iy, ix - 1] - 2.0 * t_c) / dx2
                + (T[iy + 1, ix] + T[iy - 1, ix] - 2.0 * t_c) / dy2
            )
    return lap


def solve_steady_heat(
    grid: Grid,
    k: np.ndarray,
    Q: np.ndarray,
    T_amb_K: float,
    bulk_cooling: float = 0.0,
) -> np.ndarray:
    """Steady 2D: k∇²T − h(T−T_amb) + Q = 0  (h = bulk_cooling), Dirichlet walls at T_amb."""
    nx, ny = grid.nx, grid.ny
    dx2, dy2 = grid.dx ** 2, grid.dy ** 2
    n = nx * ny
    rows, cols, vals = [], [], []

    def add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for iy in range(ny):
        for ix in range(nx):
            p = iy * nx + ix
            if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
                add(p, p, 1.0)
                continue
            kc = k[iy, ix]
            add(p, p, -2.0 * kc / dx2 - 2.0 * kc / dy2 - bulk_cooling)
            add(p, p + 1, kc / dx2)
            add(p, p - 1, kc / dx2)
            add(p, p + nx, kc / dy2)
            add(p, p - nx, kc / dy2)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=float)
    b = (-Q.reshape(-1) - bulk_cooling * T_amb_K).astype(float)
    for iy in range(ny):
        for ix in range(nx):
            if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
                p = iy * nx + ix
                b[p] = T_amb_K
    T = spla.spsolve(A, b).reshape(ny, nx)
    return np.asarray(T, dtype=float)


def _heat_report(
    T: np.ndarray,
    Q: np.ndarray,
    scene: Scene,
    em_report: FomReport,
    converged: bool,
    n_iters: int,
    max_delta: float,
) -> ThermalReport:
    cell = scene.grid.dx * scene.grid.dy
    q_t = float(Q[scene.target_mask].sum() * cell)
    q_g = float(Q[scene.gangue_mask].sum() * cell)
    q_tot = q_t + q_g
    heat_sel = q_t / q_tot if q_tot > 0 else 0.0

    t_t = T[scene.target_mask]
    t_g = T[scene.gangue_mask]
    mean_t = float(t_t.mean()) if t_t.size else 0.0
    mean_g = float(t_g.mean()) if t_g.size else 0.0

    return ThermalReport(
        T_mean_target_K=mean_t,
        T_mean_gangue_K=mean_g,
        delta_T_K=mean_t - mean_g,
        T_max_target_K=float(t_t.max()) if t_t.size else 0.0,
        heat_selectivity=heat_sel,
        em_selectivity=em_report.selectivity,
        em_contrast=em_report.contrast,
        converged=converged,
        n_iters=n_iters,
        max_delta_K=max_delta,
    )


def coupled_steady_state(
    grid: Grid,
    pair_label: str,
    config: ThermalConfig | None = None,
    params: CavityParams | None = None,
    materials: Materials | None = None,
) -> CoupledResult:
    """Iterate EM ↔ heat until T stabilises in the ore charge.

    If ``materials`` is given, use fixed ε (gel phantoms / static tests) without ε(T) tables.
    """
    cfg = config or ThermalConfig()
    params = params or CavityParams()
    freq = params.freq_hz

    T = np.full((grid.ny, grid.nx), cfg.T_amb_K, dtype=float)
    charge_mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    history: list[float] = []
    converged = False
    scene: Scene | None = None
    em: SolveResult | None = None
    Q = np.zeros((grid.ny, grid.nx), dtype=float)

    for it in range(cfg.max_iters):
        if materials is not None:
            scene = build_scene(grid, params, materials)
        else:
            scene = build_scene_at_T(grid, params, pair_label, T, freq_hz=freq)
        charge_mask = scene.target_mask | scene.gangue_mask
        em = solve_scene(grid, scene, source_amp=cfg.drive)
        Q = absorbed_power_density(em)

        k = _build_k_map(scene, cfg.thermal_props)
        T_new = solve_steady_heat(grid, k, Q, cfg.T_amb_K, bulk_cooling=cfg.bulk_cooling)
        T_new = np.clip(T_new, cfg.T_amb_K, cfg.T_max_K)
        # Ore bed only — air stays at ambient (negligible Q there anyway).
        T_coupled = T.copy()
        T_coupled[charge_mask] = (
            (1.0 - cfg.relax) * T[charge_mask] + cfg.relax * T_new[charge_mask]
        )
        max_delta = float(np.max(np.abs(T_coupled[charge_mask] - T[charge_mask]))) if charge_mask.any() else 0.0
        history.append(max_delta)
        T = T_coupled
        if max_delta < cfg.tol_K and it > 0:
            converged = True
            break

    assert scene is not None and em is not None
    em_report = evaluate(em, scene)
    thermal = _heat_report(T, Q, scene, em_report, converged, it + 1, history[-1] if history else 0.0)
    return CoupledResult(
        temperature_K=T,
        heat_generation=Q,
        em=em,
        scene=scene,
        em_report=em_report,
        thermal=thermal,
        history_max_delta=history,
    )


def isothermal_baseline(
    grid: Grid,
    pair_label: str,
    params: CavityParams | None = None,
    T_amb_K: float = 298.0,
    drive: float = 1.0,
) -> tuple[FomReport, Scene, SolveResult]:
    """Single EM solve at uniform T_amb (no thermal feedback)."""
    params = params or CavityParams()
    T = np.full((grid.ny, grid.nx), T_amb_K)
    scene = build_scene_at_T(grid, params, pair_label, T, freq_hz=params.freq_hz)
    em = solve_scene(grid, scene, source_amp=drive)
    return evaluate(em, scene), scene, em


def thermal_props_for_pair(pair_label: str) -> PhaseThermalProps:
    """Default k for each material pair."""
    if pair_label == "magnetite_in_quartz":
        return PhaseThermalProps(target_k=8.0, gangue_k=2.5)
    return PhaseThermalProps()


# ---------------------------------------------------------------------------
# Transient: dT/dt = (k∇²T − h(T−T_amb) + q) / (ρ cp), periodic EM refresh
# ---------------------------------------------------------------------------

@dataclass
class TransientConfig:
    dt_s: float = 0.4
    t_end_s: float = 90.0
    em_refresh_s: float = 4.0       # re-solve ε(T) field this often
    T_threshold_K: float = 773.0    # time-to-threshold metric
    T_amb_K: float = 298.0
    T_max_K: float = 1273.0
    drive: float = 8.0
    bulk_cooling: float = 4.0e4
    thermal_props: PhaseThermalProps = field(default_factory=PhaseThermalProps)


@dataclass
class TransientReport:
    t_target_s: float              # first time mean T_target >= threshold (inf if never)
    t_gangue_s: float
    delta_t_s: float               # t_gangue - t_target (negative => target runs away first)
    target_runaway_first: bool
    T_final_mean_target_K: float
    T_final_mean_gangue_K: float
    n_steps: int

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "t_target_s": self.t_target_s,
            "t_gangue_s": self.t_gangue_s,
            "delta_t_s": self.delta_t_s,
            "target_runaway_first": self.target_runaway_first,
            "T_final_mean_target_K": self.T_final_mean_target_K,
            "T_final_mean_gangue_K": self.T_final_mean_gangue_K,
            "n_steps": self.n_steps,
        }


@dataclass
class TransientResult:
    temperature_K: np.ndarray
    times_s: np.ndarray
    mean_T_target: np.ndarray
    mean_T_gangue: np.ndarray
    report: TransientReport


def _em_power_map(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    T: np.ndarray,
    cfg: TransientConfig,
) -> tuple[np.ndarray, Scene]:
    scene = build_scene_at_T(grid, params, pair_label, T, freq_hz=params.freq_hz)
    em = solve_scene(grid, scene, source_amp=cfg.drive)
    return absorbed_power_density(em), scene


def simulate_transient(
    grid: Grid,
    pair_label: str,
    config: TransientConfig | None = None,
    params: CavityParams | None = None,
) -> TransientResult:
    """Explicit transient heating with periodic ε(T) EM refresh."""
    cfg = config or TransientConfig()
    params = params or CavityParams()
    props = cfg.thermal_props

    T = np.full((grid.ny, grid.nx), cfg.T_amb_K, dtype=float)
    Q, scene = _em_power_map(grid, params, pair_label, T, cfg)
    k = _build_k_map(scene, props)
    rho_cp = _build_rho_cp_map(scene, props)
    charge = scene.target_mask | scene.gangue_mask

    n_steps = int(cfg.t_end_s / cfg.dt_s)
    times = np.empty(n_steps + 1)
    mean_t_hist = np.empty(n_steps + 1)
    mean_g_hist = np.empty(n_steps + 1)
    t_target = float("inf")
    t_gangue = float("inf")

    for step in range(n_steps + 1):
        t_s = step * cfg.dt_s
        times[step] = t_s
        mt = float(T[scene.target_mask].mean()) if scene.target_mask.any() else cfg.T_amb_K
        mg = float(T[scene.gangue_mask].mean()) if scene.gangue_mask.any() else cfg.T_amb_K
        mean_t_hist[step] = mt
        mean_g_hist[step] = mg
        if mt >= cfg.T_threshold_K and t_target == float("inf"):
            t_target = t_s
        if mg >= cfg.T_threshold_K and t_gangue == float("inf"):
            t_gangue = t_s
        if step == n_steps:
            break

        if step > 0 and step % max(1, int(round(cfg.em_refresh_s / cfg.dt_s))) == 0:
            Q, scene = _em_power_map(grid, params, pair_label, T, cfg)
            k = _build_k_map(scene, props)
            rho_cp = _build_rho_cp_map(scene, props)

        lap = _laplacian_dirichlet(grid, T, cfg.T_amb_K)
        dT = cfg.dt_s * (k * lap - cfg.bulk_cooling * (T - cfg.T_amb_K) + Q) / rho_cp
        T[charge] = np.clip(T[charge] + dT[charge], cfg.T_amb_K, cfg.T_max_K)
        T[~charge] = cfg.T_amb_K

    mt_f = float(T[scene.target_mask].mean())
    mg_f = float(T[scene.gangue_mask].mean())
    report = TransientReport(
        t_target_s=t_target,
        t_gangue_s=t_gangue,
        delta_t_s=t_gangue - t_target,
        target_runaway_first=t_target < t_gangue,
        T_final_mean_target_K=mt_f,
        T_final_mean_gangue_K=mg_f,
        n_steps=n_steps,
    )
    return TransientResult(
        temperature_K=T,
        times_s=times,
        mean_T_target=mean_t_hist,
        mean_T_gangue=mean_g_hist,
        report=report,
    )
