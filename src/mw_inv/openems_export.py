"""Generate runnable openEMS (Octave) models from ``CavityParams``.

Produces a self-contained ``.m`` script with CSXCAD geometry (PEC cavity, gangue bed,
target cylinders, tuning plate), a lumped coax port, sinusoidal excitation at
``freq_hz``, field dumps, and Octave post-processing that reports selectivity.

Requires openEMS + CSXCAD on the user's machine: ``conda install -c conda-forge openems csxcad``.
"""

from __future__ import annotations

from pathlib import Path

from mw_inv.fdfd import EPS0
from mw_inv.geometry import CavityParams, Materials, build_scene
from mw_inv.fdfd import Grid
from mw_inv.scene_export import build_primitives, eps_to_kappa, to_openems_mm


def _box_lines(tag: str, b, priority: int, unit: float, Lx: float, Ly: float) -> str:
    s = [
        to_openems_mm(b.x0, Lx, unit),
        to_openems_mm(b.y0, Ly, unit),
        b.z0 / unit,
    ]
    e = [
        to_openems_mm(b.x1, Lx, unit),
        to_openems_mm(b.y1, Ly, unit),
        b.z1 / unit,
    ]
    prop = "pec" if tag == "pec" else tag
    return f"CSX = AddBox(CSX, '{prop}', {priority}, [{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}], [{e[0]:.4f} {e[1]:.4f} {e[2]:.4f}]);\n"


def _cylinder_lines(c, priority: int, unit: float, Lx: float, Ly: float) -> str:
    cx = to_openems_mm(c.cx, Lx, unit)
    cy = to_openems_mm(c.cy, Ly, unit)
    r = c.radius / unit
    z0 = c.z0 / unit
    z1 = c.z1 / unit
    return (
        f"CSX = AddCylinder(CSX, 'target', {priority}, "
        f"[{cx:.4f} {cy:.4f} {z0:.4f}], [{cx:.4f} {cy:.4f} {z1:.4f}], {r:.4f});\n"
    )


def generate_openems_script(
    params: CavityParams | None = None,
    materials: Materials | None = None,
    *,
    Lz: float = 0.36,
    unit: float = 1e-3,
    n_timesteps: int = 30_000,
    function_name: str = "mw_inv_openems_cavity",
    port_mode: str = "coax_gap",
    sim_path: str = "./openems_tmp",
    sim_csx: str = "mw_inv_cavity",
) -> str:
    """Full Octave/openEMS script (InitCSX → RunOpenEMS → selectivity post-process)."""
    p = params or CavityParams()
    mats = materials or Materials()
    Lx = Ly = 0.36
    prims = build_primitives(p, mats, Lx=Lx, Ly=Ly, Lz=Lz)
    freq = p.freq_hz
    er_g, k_g = eps_to_kappa(mats.gangue, freq)
    er_t, k_t = eps_to_kappa(mats.target, freq)

    geom = "% --- geometry primitives (from mw_inv.scene_export) ---\n"
    for box in prims.boxes:
        geom += _box_lines(box.tag, box, 10 if box.tag == "pec" else 5, unit, Lx, Ly)
    for i, cyl in enumerate(prims.cylinders):
        geom += _cylinder_lines(cyl, 8 + i, unit, Lx, Ly)

    target_mask_octave = "target_mask = false(size(E2));\n"
    for ox, oy in p.inclusion_offsets_frac:
        target_mask_octave += (
            f"target_mask = target_mask | ((X - ({p.charge_cx_frac + ox})*Lx).^2 + "
            f"(Y - ({p.charge_cy_frac + oy})*Ly).^2 <= r_grain^2);\n"
        )

    # Lumped port at feed — small gap through cavity wall
    grid = Grid(nx=51, ny=51, Lx=Lx, Ly=Ly)
    from mw_inv.geometry import resolve_feed

    fx, fy = resolve_feed(p, grid)
    pw = max(p.stub_width_frac * min(Lx, Ly), 0.008)
    pfx = to_openems_mm(fx, Lx, unit)
    pfy = to_openems_mm(fy, Ly, unit)
    pd = max(p.stub_depth_frac * Ly, 0.005) / unit

    if p.feed_wall == "bottom":
        port_start = f"[{pfx - pw/unit/2:.4f} {to_openems_mm(0, Ly, unit):.4f} {pd:.4f}]"
        port_stop = f"[{pfx + pw/unit/2:.4f} {to_openems_mm(pd * unit, Ly, unit):.4f} {pd + 2:.4f}]"
        port_dir = "[0 1 0]"
    else:
        port_start = f"[{to_openems_mm(0, Lx, unit):.4f} {pfy - pw/unit/2:.4f} {pd:.4f}]"
        port_stop = f"[{to_openems_mm(pd * unit, Lx, unit):.4f} {pfy + pw/unit/2:.4f} {pd + 2:.4f}]"
        port_dir = "[1 0 0]"

    bx, by, bz = Lx / unit, Ly / unit, Lz / unit

    # --- Port / excitation (coax_gap: pin-in-vacuum; lumped: legacy stub gap) ---
    pin_mm = 5.0
    pw_mm = max(p.stub_width_frac * min(Lx, Ly) / unit, 4.0)
    if port_mode == "coax_gap":
        # SMA-style coax on top face — pin does NOT reach floor (openEMS AddLumpedPort constraint)
        cx_mm = to_openems_mm(Lx / 2, Lx, unit)
        cy_mm = to_openems_mm(Ly / 2, Ly, unit)
        port_block = f"""% Coax gap port (top face, {pin_mm:.1f} mm pin — matched-port style)
[CSX port{{1}}] = AddLumpedPort(CSX, 50, 1, 50, ...
  [{cx_mm - pw_mm/2:.4f} {cy_mm - pw_mm/2:.4f} {bz - pin_mm - 1:.4f}], ...
  [{cx_mm + pw_mm/2:.4f} {cy_mm + pw_mm/2:.4f} {bz - 1:.4f}], [0 0 1], true);
"""
    else:
        port_block = f"""% Coax lumped port (legacy stub gap)
[CSX port{{1}}] = AddLumpedPort(CSX, 50, 1, 50, {port_start}, {port_stop}, {port_dir}, true);
"""

    port_post = """
port = calcPort(port, Sim_Path, freq);
s11 = port{1}.uf.ref ./ port{1}.uf.inc;
s11_mag = abs(s11);
coupling_eff = 1 - s11_mag.^2;
fprintf('openEMS |S11| = %.4f  coupling_eff = %.4f\\n', s11_mag, coupling_eff);
fid = fopen([Sim_Path '/port_metrics.json'], 'w');
fprintf(fid, ['{{"s11_mag": %.6f, "coupling_eff": %.6f, "selectivity": %.6f, "freq_hz": %.6e}}\\n'], ...
  s11_mag, coupling_eff, selectivity, freq);
fclose(fid);
"""

    return f"""function selectivity = {function_name}()
% Auto-generated openEMS 3D cavity model from mw_inv CavityParams.
% Requires: openEMS, CSXCAD (conda-forge). Run in Octave from this file's directory.
%
%   selectivity = {function_name}()

physical_constants;
unit = {unit};
	Sim_Path = '{sim_path}';
	Sim_CSX = '{sim_csx}';
freq = {freq:.6e};
EPS0 = {EPS0:.12e};

%% --- FDTD ---
FDTD = InitFDTD('NrTS', {n_timesteps}, 'EndCriteria', 1e-5);
FDTD = SetSinusExcite(FDTD, freq);
BC = {{'PEC','PEC','PEC','PEC','PEC','PEC'}};
FDTD = SetBoundaryCond(FDTD, BC);

%% --- CSX geometry ---
CSX = InitCSX();
mesh.x = linspace(-{bx/2:.2f}, {bx/2:.2f}, 37);
mesh.y = linspace(-{by/2:.2f}, {by/2:.2f}, 37);
mesh.z = linspace(0, {bz:.2f}, 25);
CSX = DefineMesh(CSX, mesh);

CSX = AddMaterial(CSX, 'gangue');
CSX = SetMaterialProperty(CSX, 'gangue', 'Epsilon', {er_g:.6f}, 'Kappa', {k_g:.6e});
CSX = AddMaterial(CSX, 'target');
CSX = SetMaterialProperty(CSX, 'target', 'Epsilon', {er_t:.6f}, 'Kappa', {k_t:.6e});
CSX = AddMetal(CSX, 'pec');

{geom}
{port_block}
% Field dump for post-processing (steady-state amplitude at freq)
CSX = AddDump(CSX, 'Et', 'DumpType', 10, 'DumpMode', 2, 'Frequency', freq);
CSX = AddBox(CSX, 'Et', 0, [-{bx/2:.2f} -{by/2:.2f} 0], [{bx/2:.2f} {by/2:.2f} {bz:.2f}]);

WriteOpenEMS([Sim_Path '/' Sim_CSX], FDTD, CSX);
RunOpenEMS(Sim_Path, Sim_CSX);

%% --- Post-process: dissipated power selectivity in target vs gangue boxes ---
% Integration regions match scene_export (metres → mm, centred)
u = {unit};
Lx = {Lx}; Ly = {Ly}; Lz = {Lz};
cx = {p.charge_cx_frac} * Lx; cy = {p.charge_cy_frac} * Ly;
hw = 0.5 * {p.charge_w_frac} * Lx; hh = 0.5 * {p.charge_h_frac} * Ly;
r_grain = {p.inclusion_radius_frac} * min(Lx, Ly);

	dump_file = [Sim_Path '/Et/Et_0000.h5'];
if exist(dump_file, 'file') ~= 2
  warning('No field dump found — returning NaN');
  selectivity = NaN;
  return;
end
Et = ReadHDF5Dump(dump_file);
E2 = abs(Et(:,:,:,1)).^2 + abs(Et(:,:,:,2)).^2 + abs(Et(:,:,:,3)).^2;
omega = 2*pi*freq;
% Approximate cell volume from mesh
dx = Lx/(size(E2,1)-1); dy = Ly/(size(E2,2)-1); dz = Lz/(size(E2,3)-1);
cell_vol = dx*dy*dz;

% Build coordinate grids (corner frame)
nx=size(E2,1); ny=size(E2,2); nz=size(E2,3);
xg = linspace(0, Lx, nx); yg = linspace(0, Ly, ny); zg = linspace(0, Lz, nz);
[X,Y,Z] = ndgrid(xg, yg, zg);

gangue_mask = abs(X-cx)<=hw & abs(Y-cy)<=hh;
{target_mask_octave}target_mask = target_mask & gangue_mask;

eps_im_g = {max(mats.gangue.imag, 0.0):.6f};
eps_im_t = {max(mats.target.imag, 0.0):.6f};
p_g = 0.5*omega*EPS0*eps_im_g * sum(E2(gangue_mask & ~target_mask)) * cell_vol;
p_t = 0.5*omega*EPS0*eps_im_t * sum(E2(target_mask)) * cell_vol;
selectivity = p_t / (p_t + p_g + 1e-30);
fprintf('openEMS selectivity (target/charge) = %.4f\\n', selectivity);
{port_post}
end
"""


def generate_calibration_script(
    *,
    Lx: float = 0.36,
    Lz: float = 0.36,
    freq_hz: float = 2.45e9,
    unit: float = 1e-3,
    n_timesteps: int = 20_000,
    function_name: str = "mw_inv_calibration_cavity",
    sim_path: str = "./openems_cal_tmp",
    sim_csx: str = "cal_cavity",
) -> str:
    """Empty PEC cavity + top coax gap — cross-solver S11 calibration fixture."""
    bx, by, bz = Lx / unit, Lx / unit, Lz / unit
    pin_mm = 5.0
    cx = 0.0
    cy = 0.0
    z_port_lo = bz - pin_mm - 1.0
    z_port_hi = bz - 1.0
    return f"""function result = {function_name}()
% Calibration fixture: empty PEC box + coax gap (no ore charge).
% Run before ore models to verify openEMS port + mesh on your install.
physical_constants;
unit = {unit};
	Sim_Path = '{sim_path}';
	Sim_CSX = '{sim_csx}';
freq = {freq_hz:.6e};

FDTD = InitFDTD('NrTS', {n_timesteps}, 'EndCriteria', 1e-5);
FDTD = SetSinusExcite(FDTD, freq);
BC = {{'PEC','PEC','PEC','PEC','PEC','PEC'}};
FDTD = SetBoundaryCond(FDTD, BC);

CSX = InitCSX();
mesh.x = linspace(-{bx/2:.2f}, {bx/2:.2f}, 31);
mesh.y = linspace(-{by/2:.2f}, {by/2:.2f}, 31);
mesh.z = linspace(0, {bz:.2f}, 21);
CSX = DefineMesh(CSX, mesh);
CSX = AddMetal(CSX, 'pec');
CSX = AddBox(CSX, 'pec', 10, [-{bx/2:.2f} -{by/2:.2f} 0], [{bx/2:.2f} {by/2:.2f} {bz:.2f}]);

[CSX port{{1}}] = AddLumpedPort(CSX, 50, 1, 50, ...
  [{cx - 2:.4f} {cy - 2:.4f} {z_port_lo:.4f}], ...
  [{cx + 2:.4f} {cy + 2:.4f} {z_port_hi:.4f}], [0 0 1], true);

WriteOpenEMS([Sim_Path '/' Sim_CSX], FDTD, CSX);
RunOpenEMS(Sim_Path, Sim_CSX);

port = calcPort(port, Sim_Path, freq);
s11 = port{{1}}.uf.ref ./ port{{1}}.uf.inc;
s11_mag = abs(s11);
coupling_eff = 1 - s11_mag.^2;
result.s11_mag = s11_mag;
result.coupling_eff = coupling_eff;
result.freq_hz = freq;
fprintf('Calibration |S11| = %.4f at %.3f GHz  coupling_eff = %.4f\\n', ...
  result.s11_mag, freq/1e9, result.coupling_eff);
fid = fopen([Sim_Path '/port_metrics.json'], 'w');
fprintf(fid, ['{{"s11_mag": %.6f, "coupling_eff": %.6f, "freq_hz": %.6e}}\\n'], ...
  s11_mag, coupling_eff, freq);
fclose(fid);
end
"""


def write_calibration_model(path: Path | str, **kwargs) -> Path:
    path = Path(path)
    func = "mw_inv_" + path.stem.replace("-", "_").replace(".", "_")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_calibration_script(function_name=func, **kwargs))
    return path


def write_openems_model(
    path: Path | str,
    params: CavityParams | None = None,
    materials: Materials | None = None,
    **kwargs,
) -> Path:
    path = Path(path)
    func = "mw_inv_" + path.stem.replace("-", "_").replace(".", "_")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generate_openems_script(params, materials, function_name=func, port_mode="coax_gap", **kwargs))
    return path


def export_scene_npz(
    path: Path | str,
    params: CavityParams | None = None,
    materials: Materials | None = None,
    *,
    grid_n: int = 81,
) -> Path:
    """Export FDFD scene arrays alongside openEMS model for cross-check."""
    import numpy as np

    p = params or CavityParams()
    mats = materials or Materials()
    grid = Grid(nx=grid_n, ny=grid_n, Lx=0.36, Ly=0.36)
    scene = build_scene(grid, p, mats)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        eps_r=scene.eps_r,
        target_mask=scene.target_mask,
        gangue_mask=scene.gangue_mask,
        freq_hz=scene.freq_hz,
        source_xy=scene.source_xy,
    )
    return path
