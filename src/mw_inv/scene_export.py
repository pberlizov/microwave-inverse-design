"""Analytic 3D primitives mirroring ``build_scene`` (for FDTD exporters).

Coordinates: metres, origin at cavity corner (0,0,0), x ∈ [0,Lx], y ∈ [0,Ly], z ∈ [0,Lz].
openEMS uses centred mm coordinates — convert with ``to_openems_mm``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mw_inv.fdfd import EPS0
from mw_inv.geometry import CavityParams, Materials, resolve_feed
from mw_inv.fdfd import Grid


@dataclass(frozen=True)
class BoxPrim:
    tag: str  # gangue | target | pec | air
    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float


@dataclass(frozen=True)
class CylinderPrim:
    tag: str
    cx: float
    cy: float
    z0: float
    z1: float
    radius: float


@dataclass(frozen=True)
class FeedPrim:
    wall: str
    x_m: float
    y_m: float
    z_m: float
    width_m: float
    depth_m: float


@dataclass(frozen=True)
class ScenePrimitives:
    boxes: tuple[BoxPrim, ...]
    cylinders: tuple[CylinderPrim, ...]
    feed: FeedPrim
    freq_hz: float
    Lx: float
    Ly: float
    Lz: float


def _plate_bbox(params: CavityParams, Lx: float, Ly: float, Lz: float) -> BoxPrim | None:
    if params.plate_len_frac <= 1e-3:
        return None
    cx = params.plate_cx_frac * Lx
    cy = params.plate_cy_frac * Ly
    half_len = 0.5 * params.plate_len_frac * min(Lx, Ly)
    angle = math.radians(params.plate_angle_deg)
    dx, dy = math.cos(angle), math.sin(angle)
    thickness = 0.003 * min(Lx, Ly)
    corners = []
    for s in (-half_len, half_len):
        for t in (-thickness, thickness):
            px = cx + s * dx + t * (-dy)
            py = cy + s * dy + t * dx
            corners.append((px, py))
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return BoxPrim("pec", min(xs), min(ys), 0.0, max(xs), max(ys), Lz)


def build_primitives(
    params: CavityParams,
    materials: Materials | None = None,
    *,
    Lx: float = 0.36,
    Ly: float = 0.36,
    Lz: float = 0.36,
) -> ScenePrimitives:
    """Analytic boxes/cylinders for gangue bed, target grains, PEC plate."""
    mats = materials or Materials()
    _ = mats  # materials used by exporters for ε values

    cx = params.charge_cx_frac * Lx
    cy = params.charge_cy_frac * Ly
    hw = 0.5 * params.charge_w_frac * Lx
    hh = 0.5 * params.charge_h_frac * Ly
    boxes: list[BoxPrim] = [
        BoxPrim("gangue", cx - hw, cy - hh, 0.0, cx + hw, cy + hh, Lz),
    ]
    plate = _plate_bbox(params, Lx, Ly, Lz)
    if plate is not None:
        boxes.append(plate)

    r = params.inclusion_radius_frac * min(Lx, Ly)
    cylinders: list[CylinderPrim] = []
    for ox, oy in params.inclusion_offsets_frac:
        gx = (params.charge_cx_frac + ox) * Lx
        gy = (params.charge_cy_frac + oy) * Ly
        cylinders.append(CylinderPrim("target", gx, gy, 0.0, Lz, r))

    grid = Grid(nx=51, ny=51, Lx=Lx, Ly=Ly)
    fx, fy = resolve_feed(params, grid)
    fw = params.stub_width_frac * min(Lx, Ly)
    fd = params.stub_depth_frac * (Ly if params.feed_wall in ("bottom", "top") else Lx)
    feed = FeedPrim(params.feed_wall or "bottom", fx, fy, Lz * 0.5, fw, fd)

    return ScenePrimitives(
        boxes=tuple(boxes),
        cylinders=tuple(cylinders),
        feed=feed,
        freq_hz=params.freq_hz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
    )


def to_openems_mm(x_m: float, L: float, unit: float = 1e-3) -> float:
    """Convert corner-frame metres to centred openEMS mm coordinates."""
    return (x_m - 0.5 * L) / unit


def eps_to_kappa(eps: complex, freq_hz: float) -> tuple[float, float]:
    """Return (epsilon_real, conductivity_Sm) for openEMS at single frequency."""
    er = float(eps.real)
    ei = max(float(eps.imag), 0.0)
    sigma = 2.0 * math.pi * freq_hz * EPS0 * ei
    return er, sigma
