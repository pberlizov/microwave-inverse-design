"""MEEP FDTD cross-checks for the 2D FDFD forward model.

**MEEP 2D**: maturity ``experimental`` (component ``meep_2d_crosscheck``).
**MEEP 3D extrusion**: maturity ``wip`` (component ``meep_3d_extrusion``) — quasi-3D
only: 2D ε map extruded in *z*, point Ez source, PEC box.  Not a production 3D port.

See ``docs/MATURITY.md``.
"""

from __future__ import annotations

import numpy as np

from mw_inv.fdfd import EPS0, Grid
from mw_inv.geometry import Scene


def _selectivity_from_power(p: np.ndarray, scene: Scene, grid: Grid) -> float:
    """Target / (target + gangue) absorbed power from a 2D power-density slice."""
    cell = grid.dx * grid.dy
    p_t = float(p[scene.target_mask].sum() * cell)
    p_g = float(p[scene.gangue_mask].sum() * cell)
    total = p_t + p_g
    return p_t / total if total > 0 else 0.0


def _power_density_2d(ez: np.ndarray, scene: Scene, omega: float) -> np.ndarray:
    eps_imag = np.clip(np.imag(scene.eps_r), 0.0, None)
    return 0.5 * omega * EPS0 * eps_imag * np.abs(ez) ** 2


def _eps_lookup(scene: Scene, grid: Grid):
    data = scene.eps_r
    ny, nx = data.shape
    ox, oy = grid.Lx / 2.0, grid.Ly / 2.0

    def eps_at(x: float, y: float) -> complex:
        ix = int(round((x + ox) / grid.dx))
        iy = int(round((y + oy) / grid.dy))
        ix = int(np.clip(ix, 0, nx - 1))
        iy = int(np.clip(iy, 0, ny - 1))
        return complex(data[iy, ix])

    return eps_at, ox, oy


def meep_available() -> bool:
    try:
        import meep  # noqa: F401

        return True
    except ImportError:
        return False


def meep_selectivity(scene: Scene, grid: Grid, *, until: int = 300) -> float:
    """Run MEEP 2D TM and return target/(target+gangue) absorbed power."""
    import meep as mp

    freq = scene.freq_hz
    omega = 2.0 * np.pi * freq
    eps_at, ox, oy = _eps_lookup(scene, grid)

    def _material_func(r: mp.Vector3) -> mp.Medium:
        e = eps_at(r.x, r.y)
        sigma_d = omega * EPS0 * max(e.imag, 0.0)
        return mp.Medium(epsilon=e.real, D_conductivity=sigma_d)

    sx, sy = scene.source_xy
    src = mp.Source(
        mp.ContinuousSource(frequency=freq),
        component=mp.Ez,
        center=mp.Vector3(sx - ox, sy - oy),
    )
    res = max(25, int(max(grid.nx, grid.ny) / min(grid.Lx, grid.Ly)))
    sim = mp.Simulation(
        cell_size=mp.Vector3(grid.Lx, grid.Ly, 0),
        default_material=mp.Material(_material_func),
        sources=[src],
        boundary_layers=[],
        resolution=res,
        force_complex_fields=True,
    )
    sim.run(until=until)

    ez = np.array(sim.get_efield_z()).reshape(grid.ny, grid.nx)
    p = _power_density_2d(ez, scene, omega)
    return _selectivity_from_power(p, scene, grid)


def meep_selectivity_3d_extruded(
    scene: Scene,
    grid: Grid,
    *,
    Lz: float = 0.36,
    until: int = 400,
    resolution: int | None = None,
) -> float:
    """3D MEEP: extrude the 2D ``scene`` uniformly in *z* **[WIP / quasi-3D]**.

    Not a full 3D applicator model — see ``maturity.meep_3d_extrusion``.
    """
    import meep as mp

    freq = scene.freq_hz
    omega = 2.0 * np.pi * freq
    eps_at, ox, oy = _eps_lookup(scene, grid)

    def _material_func(r: mp.Vector3) -> mp.Medium:
        e = eps_at(r.x, r.y)
        sigma_d = omega * EPS0 * max(e.imag, 0.0)
        return mp.Medium(epsilon=e.real, D_conductivity=sigma_d)

    sx, sy = scene.source_xy
    src = mp.Source(
        mp.ContinuousSource(frequency=freq),
        component=mp.Ez,
        center=mp.Vector3(sx - ox, sy - oy, 0.0),
    )
    res = resolution or max(12, int(20.0 / min(grid.Lx, grid.Ly, Lz)))
    sim = mp.Simulation(
        cell_size=mp.Vector3(grid.Lx, grid.Ly, Lz),
        default_material=mp.Material(_material_func),
        sources=[src],
        boundary_layers=[],
        resolution=res,
        force_complex_fields=True,
    )
    sim.run(until=until)

    ez = np.asarray(sim.get_efield_z())  # (nx, ny, nz) in MEEP storage order
    # MEEP array axes: first dim x, second y, third z — map to (ny, nx, nz)
    if ez.ndim == 3:
        ez = np.transpose(ez, (1, 0, 2))
    nz = ez.shape[2]
    cell = grid.dx * grid.dy * (Lz / max(nz, 1))
    eps_imag = np.clip(np.imag(scene.eps_r), 0.0, None)
    p_xy = 0.5 * omega * EPS0 * eps_imag[:, :, np.newaxis] * np.abs(ez) ** 2
    p_t = float(p_xy[scene.target_mask, :].sum() * cell)
    p_g = float(p_xy[scene.gangue_mask, :].sum() * cell)
    total = p_t + p_g
    return p_t / total if total > 0 else 0.0


def compare_fdfd_meep(
    scene: Scene,
    grid: Grid,
    fdfd_selectivity: float,
    *,
    Lz: float = 0.36,
    until_2d: int = 300,
    until_3d: int = 400,
) -> dict[str, float | bool | str]:
    """Run MEEP 2D and 3D extrusion; return comparison dict (skipped if no MEEP)."""
    if not meep_available():
        return {"meep_available": False, "skipped": True}

    meep_2d = meep_selectivity(scene, grid, until=until_2d)
    meep_3d = meep_selectivity_3d_extruded(scene, grid, Lz=Lz, until=until_3d)
    return {
        "meep_available": True,
        "skipped": False,
        "meep_3d_maturity": "wip",
        "meep_3d_note": "quasi-3D extrusion — not production FDTD port",
        "fdfd_selectivity": fdfd_selectivity,
        "meep_2d_selectivity": meep_2d,
        "meep_3d_selectivity": meep_3d,
        "rel_err_2d": abs(meep_2d - fdfd_selectivity) / max(fdfd_selectivity, 1e-6),
        "rel_err_3d": abs(meep_3d - fdfd_selectivity) / max(fdfd_selectivity, 1e-6),
        "Lz_m": Lz,
    }
