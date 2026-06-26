"""2D frequency-domain FDFD solver (TM / E_z) for a metal-walled microwave cavity.

We solve the scalar Helmholtz equation for the out-of-plane field E_z on a uniform
grid, with a complex relative permittivity map ``eps_r(x, y)`` and a current source
``J_z`` (point or distributed line-port at the coax stub mouth):

    (d2/dx2 + d2/dy2) E_z + k0^2 * eps_r * E_z = -i * omega * mu0 * J_z

Boundary condition: PEC cavity walls -> E_z = 0 on the domain border (Dirichlet).
That is exactly a closed metal applicator, so no PML is needed for the thin slice.

Time convention: e^{-i omega t}. With that convention a *lossy* medium has
``Im(eps_r) > 0`` and the time-averaged absorbed power density is

    p(x, y) = 0.5 * omega * eps0 * Im(eps_r) * |E_z|^2      [W / m^3]

The solver returns the complex field; power/FOM live in ``fom.py``.

Why FDFD and not FDTD/MEEP: the figure of merit is a *single-frequency, steady-state*
absorbed-power distribution. FDFD gives that in one sparse solve with only
numpy/scipy, so the thin slice runs anywhere. The production path (broadband,
3D, EM-thermal coupling) is openEMS / MEEP / FDTDX -- see docs/FRONTIER.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Physical constants (SI).
C0 = 299_792_458.0           # speed of light [m/s]
EPS0 = 8.8541878128e-12      # vacuum permittivity [F/m]
MU0 = 1.25663706212e-6       # vacuum permeability [H/m]


@dataclass(frozen=True)
class Grid:
    """Uniform 2D grid over a rectangular cavity of size (Lx, Ly) metres."""

    nx: int
    ny: int
    Lx: float
    Ly: float

    @property
    def dx(self) -> float:
        return self.Lx / (self.nx - 1)

    @property
    def dy(self) -> float:
        return self.Ly / (self.ny - 1)

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, self.Lx, self.nx)
        y = np.linspace(0.0, self.Ly, self.ny)
        return x, y

    def index(self, ix: int, iy: int) -> int:
        return iy * self.nx + ix

    def nearest_node(self, x: float, y: float) -> tuple[int, int]:
        ix = int(round(x / self.dx))
        iy = int(round(y / self.dy))
        ix = min(max(ix, 0), self.nx - 1)
        iy = min(max(iy, 0), self.ny - 1)
        return ix, iy


@dataclass(frozen=True)
class SolveResult:
    Ez: np.ndarray          # complex field, shape (ny, nx)
    freq_hz: float
    eps_r: np.ndarray       # the permittivity map used, shape (ny, nx)
    grid: Grid
    mu_r: np.ndarray | None = None


def _build_operator(
    grid: Grid,
    eps_r: np.ndarray,
    k0: float,
    mu_r: np.ndarray | None = None,
    dirichlet_mask: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Assemble the sparse Helmholtz operator A with PEC (Dirichlet) walls.

    TM wave equation: div(1/mu grad Ez) + k0^2 eps Ez = source.  When ``mu_r``
    is omitted, mu = 1 everywhere (legacy scalar-permittivity mode).

    Optional *dirichlet_mask* enforces Ez=0 on internal metal (true PEC, backlog B2).
    """
    nx, ny = grid.nx, grid.ny
    dx2 = grid.dx ** 2
    dy2 = grid.dy ** 2
    n = nx * ny
    if mu_r is None:
        mu_r = np.ones_like(eps_r, dtype=complex)
    dmask = (
        dirichlet_mask
        if dirichlet_mask is not None
        else np.zeros((ny, nx), dtype=bool)
    )
    if dmask.shape != (ny, nx):
        raise ValueError(f"dirichlet_mask shape {dmask.shape} != ({ny}, {nx})")

    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []

    def add(r: int, c: int, v: complex) -> None:
        rows.append(r)
        cols.append(c)
        vals.append(v)

    for iy in range(ny):
        for ix in range(nx):
            p = grid.index(ix, iy)
            if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or dmask[iy, ix]:
                add(p, p, 1.0)
                continue
            inv_mu = 1.0 / mu_r[iy, ix]
            add(p, p, inv_mu * (-2.0 / dx2 - 2.0 / dy2) + (k0 ** 2) * eps_r[iy, ix])
            add(p, grid.index(ix + 1, iy), inv_mu / dx2)
            add(p, grid.index(ix - 1, iy), inv_mu / dx2)
            add(p, grid.index(ix, iy + 1), inv_mu / dy2)
            add(p, grid.index(ix, iy - 1), inv_mu / dy2)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex)


def solve(
    grid: Grid,
    eps_r: np.ndarray,
    freq_hz: float,
    source_xy: tuple[float, float] | None = None,
    source_amp: float = 1.0,
    source_j: np.ndarray | None = None,
    mu_r: np.ndarray | None = None,
    dirichlet_mask: np.ndarray | None = None,
) -> SolveResult:
    """Solve for E_z given permittivity maps, frequency, and a current source.

    Provide either ``source_xy`` (legacy point feed) or ``source_j`` (A/m² per cell,
    shape (ny, nx)) from ``geometry.build_source_jz``.  Scene-based callers should use
    ``solve_scene``.
    """
    if (source_xy is None) == (source_j is None):
        raise ValueError("provide exactly one of source_xy or source_j")

    if eps_r.shape != (grid.ny, grid.nx):
        raise ValueError(f"eps_r shape {eps_r.shape} != ({grid.ny}, {grid.nx})")
    if mu_r is not None and mu_r.shape != eps_r.shape:
        raise ValueError(f"mu_r shape {mu_r.shape} != eps_r shape {eps_r.shape}")
    if source_j is not None and source_j.shape != eps_r.shape:
        raise ValueError(f"source_j shape {source_j.shape} != eps_r shape {eps_r.shape}")

    omega = 2.0 * np.pi * freq_hz
    k0 = omega / C0
    A = _build_operator(grid, eps_r, k0, mu_r, dirichlet_mask)

    b = np.zeros(grid.nx * grid.ny, dtype=complex)
    cell_area = grid.dx * grid.dy
    dmask = (
        dirichlet_mask
        if dirichlet_mask is not None
        else np.zeros((grid.ny, grid.nx), dtype=bool)
    )

    if source_j is not None:
        j = np.asarray(source_j, dtype=complex) * source_amp
        for iy in range(1, grid.ny - 1):
            for ix in range(1, grid.nx - 1):
                if dmask[iy, ix]:
                    continue
                if j[iy, ix] != 0:
                    b[grid.index(ix, iy)] = -1j * omega * MU0 * j[iy, ix] * cell_area
    else:
        assert source_xy is not None
        sx, sy = source_xy
        six, siy = grid.nearest_node(sx, sy)
        six = min(max(six, 1), grid.nx - 2)
        siy = min(max(siy, 1), grid.ny - 2)
        if not dmask[siy, six]:
            b[grid.index(six, siy)] = -1j * omega * MU0 * source_amp / cell_area

    Ez_flat = spla.spsolve(A, b)
    Ez = Ez_flat.reshape(grid.ny, grid.nx)
    return SolveResult(Ez=Ez, freq_hz=freq_hz, eps_r=eps_r, grid=grid, mu_r=mu_r)


def _scene_dirichlet_mask(scene) -> np.ndarray | None:
    """Internal metal as Ez=0 when structure_model is dirichlet (default)."""
    pec = getattr(scene, "pec_mask", None)
    if pec is None or not np.any(pec):
        return None
    params = getattr(scene, "params", None)
    model = getattr(params, "structure_model", "dirichlet") if params else "dirichlet"
    if model != "dirichlet":
        return None
    return np.asarray(pec, dtype=bool)


def solve_scene(
    grid: Grid,
    scene,
    *,
    source_amp: float = 1.0,
) -> SolveResult:
    """Solve using distributed ``scene.source_j`` when present, else point ``source_xy``."""
    dmask = _scene_dirichlet_mask(scene)
    source_j = getattr(scene, "source_j", None)
    if source_j is not None and np.any(source_j):
        return solve(
            grid,
            scene.eps_r,
            scene.freq_hz,
            source_j=source_j,
            source_amp=source_amp,
            mu_r=scene.mu_r,
            dirichlet_mask=dmask,
        )
    return solve(
        grid,
        scene.eps_r,
        scene.freq_hz,
        source_xy=scene.source_xy,
        source_amp=source_amp,
        mu_r=scene.mu_r,
        dirichlet_mask=dmask,
    )


def magnetic_field_components(result: SolveResult) -> tuple[np.ndarray, np.ndarray]:
    """In-plane H from E_z for TM mode (e^{-i omega t} convention).

    H_x = (1/(i omega mu0 mu)) dEz/dy,  H_y = -(1/(i omega mu0 mu)) dEz/dx.
    """
    grid = result.grid
    mu = result.mu_r if result.mu_r is not None else np.ones_like(result.eps_r)
    omega = 2.0 * np.pi * result.freq_hz
    coeff = 1.0 / (1j * omega * MU0 * mu)
    dEz_dy = np.zeros_like(result.Ez)
    dEz_dx = np.zeros_like(result.Ez)
    dEz_dy[1:-1, :] = (result.Ez[2:, :] - result.Ez[:-2, :]) / (2.0 * grid.dy)
    dEz_dx[:, 1:-1] = (result.Ez[:, 2:] - result.Ez[:, :-2]) / (2.0 * grid.dx)
    return coeff * dEz_dy, -coeff * dEz_dx


def absorbed_power_density(result: SolveResult) -> np.ndarray:
    """Time-averaged absorbed power density including dielectric and magnetic loss.

    p = 0.5 omega (eps0 eps'' |E|^2 + mu0 mu'' |H|^2)  [W/m^3 in 2D per unit z].
    """
    omega = 2.0 * np.pi * result.freq_hz
    eps_imag = np.clip(result.eps_r.imag, 0.0, None)
    p_e = 0.5 * omega * EPS0 * eps_imag * np.abs(result.Ez) ** 2
    if result.mu_r is None or np.allclose(result.mu_r.imag, 0.0):
        return p_e
    hx, hy = magnetic_field_components(result)
    mu_imag = np.clip(result.mu_r.imag, 0.0, None)
    p_m = 0.5 * omega * MU0 * mu_imag * (np.abs(hx) ** 2 + np.abs(hy) ** 2)
    return p_e + p_m
