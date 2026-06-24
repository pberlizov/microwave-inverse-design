"""Build the permittivity map for a microwave applicator + mineral charge.

The scene is deliberately minimal and low-dimensional -- the whole premise (and the
Salakhi/Thomson relaxed-uniformity argument) is that we do NOT need a perfectly
uniform field over a large volume, only a field that *concentrates absorbed power in
the target mineral phase* rather than the surrounding gangue. A handful of geometric
knobs is enough to test whether geometry can move that contrast at all.

Materials are complex relative permittivities ``eps' + i eps''`` (eps'' > 0 = lossy).

The default materials are REAL literature values (magnetite absorber in transparent
quartz gangue) defined with citations in ``mw_inv.materials``; see docs/MATERIALS.md.
Use ``Materials.from_pair("pyrite_in_calcite")`` for the Salsman liberation system.
Values still carry uncertainty (grain size, form, temperature) -- representative, not
exact for a specific ore.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mw_inv.fdfd import Grid
from mw_inv.materials import DEFAULT_PAIR, PAIRS


@dataclass(frozen=True)
class Materials:
    # Background inside the cavity (air / vacuum).
    background: complex = DEFAULT_PAIR.background
    # Target mineral phase: strong microwave absorber (default: magnetite, ~12 - j1).
    target: complex = DEFAULT_PAIR.target
    # Gangue / waste rock: transparent low-loss (default: quartz, ~4.6 - j5e-4).
    gangue: complex = DEFAULT_PAIR.gangue

    @classmethod
    def from_pair(cls, label: str) -> "Materials":
        p = PAIRS[label]
        return cls(background=p.background, target=p.target, gangue=p.gangue)


@dataclass
class CavityParams:
    """Geometry knobs the optimiser is allowed to move.

    All positions are fractions of the cavity size in [0, 1] so bounds are uniform.
    """

    freq_hz: float = 2.45e9            # ISM band industrial microwave frequency
    feed_x_frac: float = 0.5           # feed position along x
    feed_y_frac: float = 0.08          # feed position along y (near a wall = a stub)
    baffle_x_frac: float = 0.5         # internal PEC baffle x-position
    baffle_len_frac: float = 0.0       # baffle length (0 = no baffle)
    baffle_gap_frac: float = 0.5       # where along the baffle the gap sits
    # Fixed charge geometry (the ore bed) -- not optimised in the thin slice.
    charge_cx_frac: float = 0.5
    charge_cy_frac: float = 0.62
    charge_w_frac: float = 0.42
    charge_h_frac: float = 0.30
    # Target inclusions inside the charge (disseminated mineral grains).
    inclusion_centers: tuple[tuple[float, float], ...] = (
        (0.40, 0.58), (0.60, 0.58), (0.50, 0.70),
    )
    inclusion_radius_frac: float = 0.05


@dataclass
class Scene:
    grid: Grid
    eps_r: np.ndarray
    target_mask: np.ndarray      # bool (ny, nx): target mineral pixels
    gangue_mask: np.ndarray      # bool (ny, nx): gangue pixels
    pec_mask: np.ndarray         # bool (ny, nx): internal PEC (baffle) pixels
    source_xy: tuple[float, float]
    freq_hz: float
    params: CavityParams = field(default_factory=CavityParams)


# A large positive imaginary permittivity approximates a PEC region cheaply
# without changing the solver (it forces |E|->0 inside the baffle).
_PEC_EPS = 1.0 + 1.0e6j


def build_scene(grid: Grid, params: CavityParams, materials: Materials | None = None) -> Scene:
    mats = materials or Materials()
    x, y = grid.coords()
    XX, YY = np.meshgrid(x, y)  # shape (ny, nx)

    eps_r = np.full((grid.ny, grid.nx), mats.background, dtype=complex)
    target_mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    gangue_mask = np.zeros((grid.ny, grid.nx), dtype=bool)

    # --- Charge bed (gangue rectangle) ---
    cx = params.charge_cx_frac * grid.Lx
    cy = params.charge_cy_frac * grid.Ly
    hw = 0.5 * params.charge_w_frac * grid.Lx
    hh = 0.5 * params.charge_h_frac * grid.Ly
    charge = (np.abs(XX - cx) <= hw) & (np.abs(YY - cy) <= hh)
    eps_r[charge] = mats.gangue
    gangue_mask |= charge

    # --- Target mineral inclusions inside the charge ---
    r = params.inclusion_radius_frac * min(grid.Lx, grid.Ly)
    for fx, fy in params.inclusion_centers:
        ix0, iy0 = fx * grid.Lx, fy * grid.Ly
        disk = ((XX - ix0) ** 2 + (YY - iy0) ** 2) <= r ** 2
        disk &= charge  # inclusions only live inside the bed
        eps_r[disk] = mats.target
        target_mask |= disk
        gangue_mask &= ~disk

    # --- Optional internal PEC baffle (a tuning vane) ---
    pec_mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    if params.baffle_len_frac > 1e-3:
        bx = params.baffle_x_frac * grid.Lx
        blen = params.baffle_len_frac * grid.Ly
        gap_center = params.baffle_gap_frac * grid.Ly
        gap_half = 0.06 * grid.Ly
        col = np.abs(XX - bx) <= (0.6 * grid.dx)
        within = (YY <= blen)
        not_gap = np.abs(YY - gap_center) > gap_half
        baffle = col & within & not_gap
        eps_r[baffle] = _PEC_EPS
        pec_mask |= baffle
        # A PEC baffle overrides any material it crosses.
        target_mask &= ~baffle
        gangue_mask &= ~baffle

    source_xy = (params.feed_x_frac * grid.Lx, params.feed_y_frac * grid.Ly)
    return Scene(
        grid=grid,
        eps_r=eps_r,
        target_mask=target_mask,
        gangue_mask=gangue_mask,
        pec_mask=pec_mask,
        source_xy=source_xy,
        freq_hz=params.freq_hz,
        params=params,
    )
