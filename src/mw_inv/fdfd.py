"""2D frequency-domain FDFD solver (TM / E_z) for a metal-walled microwave cavity.

We solve the scalar Helmholtz equation for the out-of-plane field E_z on a uniform
grid, with a complex relative permittivity map ``eps_r(x, y)`` and a point current
source (a stylised feed / antenna stub):

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


def _build_operator(grid: Grid, eps_r: np.ndarray, k0: float) -> sp.csr_matrix:
    """Assemble the sparse Helmholtz operator A with PEC (Dirichlet) walls."""
    nx, ny = grid.nx, grid.ny
    dx2 = grid.dx ** 2
    dy2 = grid.dy ** 2
    n = nx * ny

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
            # PEC wall nodes: E_z = 0.
            if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
                add(p, p, 1.0)
                continue
            # Interior 5-point Laplacian + k0^2 eps term.
            add(p, p, -2.0 / dx2 - 2.0 / dy2 + (k0 ** 2) * eps_r[iy, ix])
            add(p, grid.index(ix + 1, iy), 1.0 / dx2)
            add(p, grid.index(ix - 1, iy), 1.0 / dx2)
            add(p, grid.index(ix, iy + 1), 1.0 / dy2)
            add(p, grid.index(ix, iy - 1), 1.0 / dy2)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex)


def solve(
    grid: Grid,
    eps_r: np.ndarray,
    freq_hz: float,
    source_xy: tuple[float, float],
    source_amp: float = 1.0,
) -> SolveResult:
    """Solve for E_z given a permittivity map, frequency and a point feed.

    ``eps_r`` is shape (ny, nx), complex (Im > 0 == lossy). ``source_xy`` is the
    feed location in metres. Returns the complex field on the grid.
    """
    if eps_r.shape != (grid.ny, grid.nx):
        raise ValueError(f"eps_r shape {eps_r.shape} != ({grid.ny}, {grid.nx})")

    omega = 2.0 * np.pi * freq_hz
    k0 = omega / C0
    A = _build_operator(grid, eps_r, k0)

    b = np.zeros(grid.nx * grid.ny, dtype=complex)
    sx, sy = source_xy
    six, siy = grid.nearest_node(sx, sy)
    # Keep the feed off the PEC wall (where E_z is pinned to 0).
    six = min(max(six, 1), grid.nx - 2)
    siy = min(max(siy, 1), grid.ny - 2)
    cell_area = grid.dx * grid.dy
    b[grid.index(six, siy)] = -1j * omega * MU0 * source_amp / cell_area

    Ez_flat = spla.spsolve(A, b)
    Ez = Ez_flat.reshape(grid.ny, grid.nx)
    return SolveResult(Ez=Ez, freq_hz=freq_hz, eps_r=eps_r, grid=grid)


def absorbed_power_density(result: SolveResult) -> np.ndarray:
    """Time-averaged absorbed power density p = 0.5 * omega * eps0 * Im(eps) * |E|^2.

    Units W/m^3 (2D, per unit length in z). Absolute scale is arbitrary for the
    selectivity FOM since constant prefactors cancel in the ratio.
    """
    omega = 2.0 * np.pi * result.freq_hz
    eps_imag = np.clip(result.eps_r.imag, 0.0, None)
    return 0.5 * omega * EPS0 * eps_imag * np.abs(result.Ez) ** 2
