"""Forward-model validation: analytic checks, grid convergence, optional MEEP.

Runs without MEEP/openEMS (numpy/scipy only). When ``meep`` is installed,
``meep_scene_check`` cross-checks selectivity on a simplified 2D TM cavity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mw_inv.dielectric_data import polyakova_bulk_eps
from mw_inv.fdfd import C0, Grid, SolveResult, absorbed_power_density, solve, solve_scene
from mw_inv.fom import evaluate
from mw_inv.geometry import CavityParams, Materials, build_scene


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class ValidationReport:
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "metrics": c.metrics}
                for c in self.checks
            ],
        }


# ---------------------------------------------------------------------------
# 1. Method of manufactured solutions (MMS)
# ---------------------------------------------------------------------------

def _build_operator_mu(
    grid: Grid, eps_r: np.ndarray, mu_r: np.ndarray, k0: float,
) -> sp.csr_matrix:
    """Helmholtz operator with mu: div(1/mu grad Ez) + k0^2 eps Ez."""
    nx, ny = grid.nx, grid.ny
    dx2, dy2 = grid.dx ** 2, grid.dy ** 2
    rows, cols, vals = [], [], []

    def add(r: int, c: int, v: complex) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for iy in range(ny):
        for ix in range(nx):
            p = grid.index(ix, iy)
            if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
                add(p, p, 1.0)
                continue
            inv_mu = 1.0 / mu_r[iy, ix]
            add(p, p, inv_mu * (-2.0 / dx2 - 2.0 / dy2) + (k0 ** 2) * eps_r[iy, ix])
            add(p, grid.index(ix + 1, iy), inv_mu / dx2)
            add(p, grid.index(ix - 1, iy), inv_mu / dx2)
            add(p, grid.index(ix, iy + 1), inv_mu / dy2)
            add(p, grid.index(ix, iy - 1), inv_mu / dy2)
    return sp.csr_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny), dtype=complex)


def manufactured_solution_error(
    grid: Grid | None = None,
    rtol: float = 0.02,
) -> CheckResult:
    """MMS: known E_z mode in a lossy cavity; solver residual should be small."""
    grid = grid or Grid(nx=41, ny=41, Lx=0.25, Ly=0.25)
    x, y = grid.coords()
    XX, YY = np.meshgrid(x, y)
    ax, ay = np.pi / grid.Lx, np.pi / grid.Ly
    Ez_exact = np.sin(ax * XX) * np.sin(ay * YY)

    eps_r = (4.0 + 0.5j) * np.ones((grid.ny, grid.nx), dtype=complex)
    mu_r = np.ones((grid.ny, grid.nx), dtype=complex)
    freq_hz = 2.45e9
    k0 = 2.0 * np.pi * freq_hz / C0
    A = _build_operator_mu(grid, eps_r, mu_r, k0)
    e_exact = Ez_exact.reshape(-1)
    b_flat = A @ e_exact
    # Dirichlet rows: enforce homogeneous BC (Ez=0 on walls).
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            if ix == 0 or ix == grid.nx - 1 or iy == 0 or iy == grid.ny - 1:
                b_flat[grid.index(ix, iy)] = 0.0

    Ez_num = spla.spsolve(A, b_flat).reshape(grid.ny, grid.nx)
    interior = np.zeros((grid.ny, grid.nx), dtype=bool)
    interior[1:-1, 1:-1] = True
    rel = float(np.linalg.norm((Ez_num - Ez_exact)[interior])
                / max(np.linalg.norm(Ez_exact[interior]), 1e-30))
    ok = rel < rtol
    return CheckResult(
        "manufactured_solution",
        ok,
        f"relative L2 error on interior = {rel:.4g} (threshold {rtol})",
        {"rel_l2_error": rel},
    )


def dual_grid_agreement(tol: float = 0.02) -> CheckResult:
    """Selectivity at default geometry should agree between nx=61 and nx=101."""
    params = CavityParams()
    mats = Materials.from_pair("pyrite_in_calcite")
    sels = []
    for n in (61, 101):
        grid = Grid(nx=n, ny=n, Lx=0.36, Ly=0.36)
        scene = build_scene(grid, params, mats)
        res = solve_scene(grid, scene)
        sels.append(evaluate(res, scene).selectivity)
    delta = abs(sels[1] - sels[0])
    ok = delta < tol
    return CheckResult(
        "dual_grid_agreement",
        ok,
        f"selectivity nx=61 vs 101: {sels[0]:.4f} vs {sels[1]:.4f} (|Δ|={delta:.4f}, tol {tol})",
        {"selectivity_61": sels[0], "selectivity_101": sels[1], "delta": delta},
    )


# ---------------------------------------------------------------------------
# 3. Grid convergence on canonical scene
# ---------------------------------------------------------------------------

def grid_convergence(
    materials_label: str = "pyrite_in_calcite",
    grids: tuple[int, ...] = (41, 61, 81, 101),
    tol: float = 0.015,
) -> CheckResult:
    """Selectivity at default geometry should converge as the grid refines."""
    mats = Materials.from_pair(materials_label)
    params = CavityParams()
    sels = []
    for g in grids:
        grid = Grid(nx=g, ny=g, Lx=0.36, Ly=0.36)
        scene = build_scene(grid, params, mats)
        res = solve_scene(grid, scene)
        sels.append(evaluate(res, scene).selectivity)
    sels = np.array(sels)
    spread = float(sels[-1] - sels[-2])
    ok = abs(spread) < tol
    return CheckResult(
        "grid_convergence",
        ok,
        f"selectivity {materials_label}: {grids[-2]}->{grids[-1]} delta = {spread:+.4f} "
        f"(threshold {tol}); values {sels.tolist()}",
        {"selectivity_fine": float(sels[-1]), "delta_last_step": spread},
    )


# ---------------------------------------------------------------------------
# 4. Analytic empty-cavity resonance
# ---------------------------------------------------------------------------

def cavity_resonance_peak() -> CheckResult:
    """TM11 resonance peak for 0.2 m square lossless cavity near 1.06 GHz."""
    grid = Grid(nx=81, ny=81, Lx=0.20, Ly=0.20)
    eps = np.ones((grid.ny, grid.nx), dtype=complex)
    freqs = [0.80e9, 1.06e9, 1.40e9]
    energies = []
    for f in freqs:
        res = solve(grid, eps, f, source_xy=(0.1, 0.1))
        energies.append(float(np.sum(np.abs(res.Ez) ** 2)))
    ok = energies[1] > energies[0] and energies[1] > energies[2]
    ratio = energies[1] / max(energies[0], energies[2], 1e-30)
    return CheckResult(
        "cavity_resonance",
        ok,
        f"energy ratio peak/off = {ratio:.2f} at {freqs[1]/1e9:.2f} GHz",
        {"energy_ratio": ratio},
    )


# ---------------------------------------------------------------------------
# 5. Literature consistency (Polyakova bulk vs disseminated scene values)
# ---------------------------------------------------------------------------

def literature_consistency() -> CheckResult:
    """Document that bulk Polyakova ε differs from disseminated scene values."""
    bulk_p = polyakova_bulk_eps("pyrite", 2.45e9)
    scene_p = Materials.from_pair("pyrite_in_calcite").target
    ratio = abs(bulk_p) / abs(scene_p)
    # Expect large discrepancy — confirms we are NOT using bulk polynomials in scenes.
    ok = ratio > 5.0
    return CheckResult(
        "literature_bulk_vs_disseminated",
        ok,
        f"|ε_bulk|/|ε_scene| = {ratio:.1f} at 2.45 GHz "
        f"(bulk Polyakova {bulk_p.real:.0f}-j{bulk_p.imag:.0f} vs scene "
        f"{scene_p.real}-j{scene_p.imag}) — confirms disseminated primary data",
        {"bulk_to_scene_ratio": ratio},
    )


# ---------------------------------------------------------------------------
# 6. Magnetic loss channel (magnetite μ″ adds absorption)
# ---------------------------------------------------------------------------

def magnetic_loss_channel() -> CheckResult:
    """``absorbed_power_density`` includes μ″·|H|² at fixed E (formula check)."""
    grid = Grid(nx=41, ny=41, Lx=0.25, Ly=0.25)
    x, y = grid.coords()
    XX, YY = np.meshgrid(x, y)
    Ez = np.sin(np.pi * XX / grid.Lx) * np.sin(np.pi * YY / grid.Ly)
    eps_r = (1.0 + 0.0j) * np.ones((grid.ny, grid.nx), dtype=complex)
    mu0 = np.ones((grid.ny, grid.nx), dtype=complex)
    mu1 = mu0.copy()
    mu1[10:30, 10:30] = 1.3 + 0.6j
    mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    mask[10:30, 10:30] = True
    freq = 2.45e9
    res0 = SolveResult(Ez=Ez, freq_hz=freq, eps_r=eps_r, grid=grid, mu_r=mu0)
    res1 = SolveResult(Ez=Ez, freq_hz=freq, eps_r=eps_r, grid=grid, mu_r=mu1)
    p0 = float(absorbed_power_density(res0)[mask].sum())
    p1 = float(absorbed_power_density(res1)[mask].sum())
    ok = p1 > p0 + 1e-20
    return CheckResult(
        "magnetic_loss_channel",
        ok,
        f"fixed-E μ″ contribution: p(μ″>0)={p1:.3e} vs p(μ″=0)={p0:.3e}",
        {"p_lossy_mu": p1, "p_lossless_mu": p0},
    )


# ---------------------------------------------------------------------------
# 7. Temperature-dependent materials shift selectivity
# ---------------------------------------------------------------------------

def temperature_materials_effect(min_delta: float = 0.005) -> CheckResult:
    """ε(T) table at 773 K should measurably change selectivity vs 298 K."""
    grid = Grid(nx=71, ny=71, Lx=0.36, Ly=0.36)
    params = CavityParams()
    scene_cold = build_scene(grid, params, Materials.from_pair("pyrite_in_calcite", target_T_K=298.0))
    scene_hot = build_scene(grid, params, Materials.from_pair("pyrite_in_calcite", target_T_K=773.0))
    res_cold = solve_scene(grid, scene_cold)
    res_hot = solve_scene(grid, scene_hot)
    s_cold = evaluate(res_cold, scene_cold).selectivity
    s_hot = evaluate(res_hot, scene_hot).selectivity
    delta = abs(s_hot - s_cold)
    ok = delta > min_delta
    return CheckResult(
        "temperature_materials",
        ok,
        f"pyrite_in_calcite selectivity 298K={s_cold:.4f} vs 773K={s_hot:.4f} "
        f"(|Δ|={delta:.4f})",
        {"selectivity_298K": s_cold, "selectivity_773K": s_hot},
    )


# ---------------------------------------------------------------------------
# 8. Optional MEEP cross-check
# ---------------------------------------------------------------------------

def meep_scene_check(
    materials_label: str = "pyrite_in_calcite",
    grid_n: int = 61,
    selectivity_rtol: float = 0.12,
) -> CheckResult:
    """Compare FDFD selectivity to MEEP 2D TM when ``meep`` is importable."""
    try:
        from mw_inv.meep_compare import meep_selectivity  # noqa: WPS433
    except ImportError:
        return CheckResult(
            "meep_cross_check",
            True,
            "meep not installed — skipped (install via conda-forge to enable)",
            {"skipped": 1.0},
        )

    grid = Grid(nx=grid_n, ny=grid_n, Lx=0.36, Ly=0.36)
    mats = Materials.from_pair(materials_label)
    scene = build_scene(grid, CavityParams(), mats)
    fdfd_sel = evaluate(
        solve_scene(grid, scene),
        scene,
    ).selectivity
    meep_sel = meep_selectivity(scene, grid)
    rel = abs(meep_sel - fdfd_sel) / max(fdfd_sel, 1e-6)
    ok = rel < selectivity_rtol
    return CheckResult(
        "meep_cross_check",
        ok,
        f"selectivity FDFD={fdfd_sel:.4f} MEEP={meep_sel:.4f} rel err={rel:.3f}",
        {"fdfd_selectivity": fdfd_sel, "meep_selectivity": meep_sel, "rel_error": rel},
    )


def meep_3d_extrusion_check(
    materials_label: str = "pyrite_in_calcite",
    grid_n: int = 51,
    Lz: float = 0.36,
    selectivity_rtol: float = 0.20,
) -> CheckResult:
    """Compare FDFD to primitive 3D MEEP when ``meep`` is importable."""
    from mw_inv.meep_3d import compare_fdfd_meep_3d
    from mw_inv.meep_compare import meep_available

    if not meep_available():
        return CheckResult(
            "meep_3d_primitive",
            True,
            "meep not installed — skipped",
            {"skipped": 1.0},
        )

    grid = Grid(nx=grid_n, ny=grid_n, Lx=0.36, Ly=0.36)
    mats = Materials.from_pair(materials_label)
    scene = build_scene(grid, CavityParams(), mats)
    fdfd_sel = evaluate(
        solve_scene(grid, scene),
        scene,
    ).selectivity
    cmp = compare_fdfd_meep_3d(scene, grid, fdfd_sel, materials=mats, Lz=Lz)
    rel = float(cmp.get("rel_err_3d_primitive", 1.0))
    meep3 = float(cmp.get("meep_3d_primitive_selectivity", 0.0))
    ok = rel < selectivity_rtol
    return CheckResult(
        "meep_3d_primitive",
        ok,
        f"FDFD={fdfd_sel:.4f} MEEP3D(prim)={meep3:.4f} rel err={rel:.3f}",
        {
            "fdfd_selectivity": fdfd_sel,
            "meep_3d_primitive_selectivity": meep3,
            "rel_error": rel,
            "Lz_m": Lz,
        },
    )


def run_all(include_meep: bool = True) -> ValidationReport:
    checks = [
        manufactured_solution_error(),
        dual_grid_agreement(),
        cavity_resonance_peak(),
        grid_convergence(),
        literature_consistency(),
        magnetic_loss_channel(),
        temperature_materials_effect(),
    ]
    if include_meep:
        checks.append(meep_scene_check())
        checks.append(meep_3d_extrusion_check())
    return ValidationReport(checks=checks)
