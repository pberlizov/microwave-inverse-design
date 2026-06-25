"""Primitive-based 3D MEEP FDTD (gangue box, target cylinders, PEC plate).

Unlike the legacy extrusion-only path, this builds explicit 3D shapes matching
``scene_export.build_primitives`` — metal plate, volumetric bed, and grain cylinders.
"""

from __future__ import annotations

import math

import numpy as np

from mw_inv.fdfd import EPS0, Grid
from mw_inv.geometry import CavityParams, Materials, Scene, build_scene, resolve_feed
from mw_inv.meep_compare import meep_available
from mw_inv.scene_export import build_primitives


def _medium(eps: complex, freq_hz: float):
    import meep as mp

    omega = 2.0 * math.pi * freq_hz
    sigma = omega * EPS0 * max(eps.imag, 0.0)
    return mp.Medium(epsilon=eps.real, D_conductivity=sigma)


def _corner_to_meep(x: float, y: float, z: float, Lx: float, Ly: float, Lz: float):
    import meep as mp

    return mp.Vector3(x - Lx / 2, y - Ly / 2, z - Lz / 2)


def build_meep_geometry(
    params: CavityParams,
    materials: Materials,
    *,
    Lz: float = 0.36,
):
    """Return (geometry list, cell_size, resolution) for a 3D MEEP simulation."""
    import meep as mp

    Lx = Ly = 0.36
    prims = build_primitives(params, materials, Lx=Lx, Ly=Ly, Lz=Lz)
    freq = params.freq_hz
    geometry: list = []

    for box in prims.boxes:
        if box.tag == "gangue":
            c = _corner_to_meep(
                (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, (box.z0 + box.z1) / 2,
                Lx, Ly, Lz,
            )
            s = mp.Vector3(box.x1 - box.x0, box.y1 - box.y0, box.z1 - box.z0)
            geometry.append(mp.Block(center=c, size=s, material=_medium(materials.gangue, freq)))
        elif box.tag == "pec":
            c = _corner_to_meep(
                (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, Lz / 2, Lx, Ly, Lz,
            )
            s = mp.Vector3(
                max(box.x1 - box.x0, 0.002),
                max(box.y1 - box.y0, 0.002),
                Lz,
            )
            geometry.append(mp.Block(center=c, size=s, material=mp.metal))

    for cyl in prims.cylinders:
        c = _corner_to_meep(cyl.cx, cyl.cy, Lz / 2, Lx, Ly, Lz)
        geometry.append(
            mp.Cylinder(
                center=c,
                radius=cyl.radius,
                height=Lz,
                axis=mp.Vector3(0, 0, 1),
                material=_medium(materials.target, freq),
            )
        )

    res = max(14, int(24.0 / min(Lx, Ly, Lz)))
    return geometry, mp.Vector3(Lx, Ly, Lz), res


def build_feed_source(params: CavityParams, grid: Grid, Lz: float, materials: Materials | None = None):
    """MEEP source with stub-width extent (closer to coax mouth than delta point)."""
    import meep as mp

    Lx = Ly = 0.36
    sx, sy = resolve_feed(params, grid)
    sw = max(params.stub_width_frac * min(Lx, Ly), 0.012)
    sd = max(params.stub_depth_frac * min(Lx, Ly), 0.008)
    center = _corner_to_meep(sx, sy, Lz / 2, Lx, Ly, Lz)
    size = mp.Vector3(sw, sd, Lz * 0.15)
    return mp.Source(
        mp.ContinuousSource(frequency=params.freq_hz, width=20),
        component=mp.Ez,
        center=center,
        size=size,
    )


def meep_3d_selectivity(
    params: CavityParams | None = None,
    materials: Materials | None = None,
    *,
    Lz: float = 0.36,
    until: int = 500,
    extended_source: bool = True,
) -> float:
    """3D MEEP selectivity from explicit primitives + 2D masks for integration."""
    import meep as mp

    p = params or CavityParams()
    mats = materials or Materials()
    Lx = Ly = 0.36
    geometry, cell, res = build_meep_geometry(p, mats, Lz=Lz)

    grid = Grid(nx=61, ny=61, Lx=Lx, Ly=Ly)
    scene = build_scene(grid, p, mats)
    if extended_source:
        src = build_feed_source(p, grid, Lz, mats)
    else:
        sx, sy = resolve_feed(p, grid)
        src = mp.Source(
            mp.ContinuousSource(frequency=p.freq_hz),
            component=mp.Ez,
            center=_corner_to_meep(sx, sy, Lz / 2, Lx, Ly, Lz),
        )

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[src],
        boundary_layers=[],
        resolution=res,
        force_complex_fields=True,
    )
    sim.run(until=until)

    ez = np.asarray(sim.get_efield_z())
    if ez.ndim == 3:
        ez = np.transpose(ez, (1, 0, 2))
    return _selectivity_from_3d_field(ez, scene, p.freq_hz, Lz)


def _selectivity_from_3d_field(
    ez: np.ndarray,
    scene: Scene,
    freq_hz: float,
    Lz: float,
) -> float:
    grid = scene.grid
    omega = 2.0 * math.pi * freq_hz
    eps_imag = np.clip(np.imag(scene.eps_r), 0.0, None)
    nz = ez.shape[2]
    cell = grid.dx * grid.dy * (Lz / max(nz, 1))
    p_xy = 0.5 * omega * EPS0 * eps_imag[:, :, np.newaxis] * np.abs(ez) ** 2
    p_t = float(p_xy[scene.target_mask, :].sum() * cell)
    p_g = float(p_xy[scene.gangue_mask, :].sum() * cell)
    total = p_t + p_g
    return p_t / total if total > 0 else 0.0


def compare_fdfd_meep_3d(
    scene: Scene,
    grid: Grid,
    fdfd_selectivity: float,
    *,
    materials: Materials | None = None,
    Lz: float = 0.36,
) -> dict:
    """FDFD vs primitive 3D MEEP vs legacy extrusion."""
    from mw_inv.meep_compare import meep_selectivity, meep_selectivity_3d_extruded

    if not meep_available():
        return {"meep_available": False, "skipped": True}

    params = scene.params
    mats = materials or Materials(
        target=scene.eps_r[scene.target_mask][0] if scene.target_mask.any() else 8 + 0.3j,
        gangue=scene.eps_r[scene.gangue_mask][0] if scene.gangue_mask.any() else 5 + 0.05j,
    )
    meep2 = meep_selectivity(scene, grid)
    meep3_prim = meep_3d_selectivity(params, mats, Lz=Lz)
    meep3_ext = meep_selectivity_3d_extruded(scene, grid, Lz=Lz)
    return {
        "meep_available": True,
        "skipped": False,
        "fdfd_selectivity": fdfd_selectivity,
        "meep_2d_selectivity": meep2,
        "meep_3d_primitive_selectivity": meep3_prim,
        "meep_3d_extrusion_selectivity": meep3_ext,
        "rel_err_3d_primitive": abs(meep3_prim - fdfd_selectivity) / max(fdfd_selectivity, 1e-6),
        "Lz_m": Lz,
    }
