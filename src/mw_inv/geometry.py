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

Manufacturable parametrization (step 4): wall-mounted coax / waveguide feed, coax
stub depth, movable PEC tuning plate, and bed position.

**Maturity: EXPERIMENTAL** — manufacturable geometry with **distributed line-port**
excitation in FDFD (stub-mouth ``J_z`` band matching coax/waveguide width). openEMS
remains the matched-port truth solver (backlog A1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mw_inv.fdfd import Grid
from mw_inv.materials import PAIRS, Materials

FEED_WALLS: tuple[str, ...] = ("bottom", "left", "right", "top")
EXCITATION_TYPES: tuple[str, ...] = ("coax", "waveguide")


@dataclass
class CavityParams:
    """Geometry knobs the optimiser is allowed to move.

    All positions are fractions of the cavity size in [0, 1] so bounds are uniform.

    **Manufacturable (default search path):**
    - ``feed_wall`` + ``feed_along_frac`` + ``stub_depth_frac`` — coax or waveguide
      injection along a cavity wall (point source at stub tip in FDFD).
    - ``plate_*`` — movable PEC tuning plate (metal vane).
    - ``charge_cx/cy_frac`` — ore bed position inside the cavity.

    **Legacy (abstract):** ``feed_x/y_frac``, ``baffle_*``, ``tuner_field``.
    """

    freq_hz: float = 2.45e9            # ISM band industrial microwave frequency

    # --- Manufacturable excitation ---
    feed_wall: str = "bottom"          # bottom | left | right | top
    feed_along_frac: float = 0.5       # position along chosen wall [0, 1]
    stub_depth_frac: float = 0.08      # coax / port inset from wall (fraction of span)
    stub_width_frac: float = 0.04      # coax OD or waveguide mouth width
    excitation: str = "coax"           # coax | waveguide — WIP: width only, not TE10 port

    # Legacy feed (used only when feed_wall is empty string)
    feed_x_frac: float = 0.5
    feed_y_frac: float = 0.08

    # --- Movable PEC tuning plate (replaces abstract baffle in search) ---
    plate_cx_frac: float = 0.5
    plate_cy_frac: float = 0.35
    plate_len_frac: float = 0.0        # 0 = no plate
    plate_angle_deg: float = 90.0       # 90° ≈ vertical vane

    # --- Legacy internal PEC baffle (slot vane; kept for old scripts) ---
    baffle_x_frac: float = 0.5
    baffle_len_frac: float = 0.0
    baffle_gap_frac: float = 0.5
    # Internal metal model: ``dirichlet`` (Ez=0 rows, backlog B2) or legacy ``lossy_imag``.
    structure_model: str = "dirichlet"

    # --- Ore bed (charge) geometry ---
    charge_cx_frac: float = 0.5
    charge_cy_frac: float = 0.62
    charge_w_frac: float = 0.42
    charge_h_frac: float = 0.30
    # Target grains as offsets from bed centre (move with charge_cx/cy_frac).
    inclusion_offsets_frac: tuple[tuple[float, float], ...] = (
        (-0.10, -0.04), (0.10, -0.04), (0.0, 0.08),
    )
    inclusion_radius_frac: float = 0.05
    # Per-grain radii (fraction of min(Lx,Ly)); when empty, use inclusion_radius_frac for all.
    inclusion_radii_frac: tuple[float, ...] = ()

    # --- High-dimensional dielectric tuner (non-physical upper bound) ---
    tuner_field: tuple[float, ...] = ()
    tuner_eps_max: float = 12.0
    tuner_y_frac: float = 0.90
    tuner_h_frac: float = 0.06
    tuner_x0_frac: float = 0.10
    tuner_x1_frac: float = 0.90


@dataclass
class Scene:
    grid: Grid
    eps_r: np.ndarray
    mu_r: np.ndarray
    target_mask: np.ndarray      # bool (ny, nx): target mineral pixels
    gangue_mask: np.ndarray      # bool (ny, nx): gangue pixels
    pec_mask: np.ndarray         # bool (ny, nx): internal PEC (plate/baffle) pixels
    source_xy: tuple[float, float]
    freq_hz: float
    source_j: np.ndarray | None = None   # A/m² per cell — distributed stub-mouth feed
    params: CavityParams = field(default_factory=CavityParams)


# A large positive imaginary permittivity approximates a PEC region cheaply
# without changing the solver (it forces |E|->0 inside the plate).
_PEC_EPS = 1.0 + 1.0e6j


def resolve_feed(params: CavityParams, grid: Grid) -> tuple[float, float]:
    """Map wall + along-wall position + stub depth to a source (x, y) in metres."""
    if not params.feed_wall:
        return params.feed_x_frac * grid.Lx, params.feed_y_frac * grid.Ly
    wall = params.feed_wall
    if wall not in FEED_WALLS:
        raise ValueError(f"feed_wall must be one of {FEED_WALLS}, got {wall!r}")
    along = float(np.clip(params.feed_along_frac, 0.0, 1.0))
    stub = float(np.clip(params.stub_depth_frac, 0.01, 0.45))
    Lx, Ly = grid.Lx, grid.Ly
    if wall == "bottom":
        return along * Lx, stub * Ly
    if wall == "top":
        return along * Lx, (1.0 - stub) * Ly
    if wall == "left":
        return stub * Lx, along * Ly
    return (1.0 - stub) * Lx, along * Ly


def _stub_width(params: CavityParams, grid: Grid) -> float:
    w = params.stub_width_frac
    if params.excitation == "waveguide":
        w = max(w, 0.08)
    return w * min(grid.Lx, grid.Ly)


def build_source_jz(
    params: CavityParams,
    grid: Grid,
    *,
    source_amp: float = 1.0,
) -> np.ndarray:
    """Distributed ``J_z`` at the coax/waveguide stub mouth (A/m² per cell).

    Spreads the feed across the stub width at the tip — closer to a lumped port than a
    single grid-node point source.  Total discrete RHS matches the legacy point feed
    ``solve(..., source_xy=...)`` convention (``source_amp`` is the same scalar knob).
    """
    jz = np.zeros((grid.ny, grid.nx), dtype=float)
    sx, sy = resolve_feed(params, grid)
    half_w = 0.5 * _stub_width(params, grid)
    x, y = grid.coords()
    XX, YY = np.meshgrid(x, y)

    wall = params.feed_wall or "bottom"
    if wall in ("bottom", "top"):
        mask = (np.abs(XX - sx) <= half_w) & (np.abs(YY - sy) <= 1.5 * grid.dy)
    else:
        mask = (np.abs(YY - sy) <= half_w) & (np.abs(XX - sx) <= 1.5 * grid.dx)

    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    n = int(mask.sum())
    cell_area = grid.dx * grid.dy
    if n == 0:
        ix, iy = grid.nearest_node(sx, sy)
        ix = min(max(ix, 1), grid.nx - 2)
        iy = min(max(iy, 1), grid.ny - 2)
        jz[iy, ix] = source_amp / (cell_area * cell_area)
        return jz.astype(complex)

    jz[mask] = source_amp / (n * cell_area * cell_area)
    return jz.astype(complex)


def _rasterize_stub(
    XX: np.ndarray,
    YY: np.ndarray,
    grid: Grid,
    params: CavityParams,
) -> np.ndarray:
    """Air channel from wall to feed tip (coax jacket / waveguide mouth)."""
    wall = params.feed_wall or "bottom"
    half_w = 0.5 * _stub_width(params, grid)
    along = params.feed_along_frac * (grid.Lx if wall in ("bottom", "top") else grid.Ly)
    stub = params.stub_depth_frac
    if wall == "bottom":
        return (
            (np.abs(XX - along) <= half_w)
            & (YY >= 0.0)
            & (YY <= stub * grid.Ly + grid.dy)
        )
    if wall == "top":
        y0 = (1.0 - stub) * grid.Ly
        return (np.abs(XX - along) <= half_w) & (YY >= y0 - grid.dy) & (YY <= grid.Ly)
    if wall == "left":
        return (
            (np.abs(YY - along) <= half_w)
            & (XX >= 0.0)
            & (XX <= stub * grid.Lx + grid.dx)
        )
    x0 = (1.0 - stub) * grid.Lx
    return (np.abs(YY - along) <= half_w) & (XX >= x0 - grid.dx) & (XX <= grid.Lx)


def _rasterize_plate(
    XX: np.ndarray,
    YY: np.ndarray,
    grid: Grid,
    params: CavityParams,
) -> np.ndarray:
    """Movable PEC tuning plate as a thick line segment."""
    if params.plate_len_frac <= 1e-3:
        return np.zeros((grid.ny, grid.nx), dtype=bool)
    cx = params.plate_cx_frac * grid.Lx
    cy = params.plate_cy_frac * grid.Ly
    half_len = 0.5 * params.plate_len_frac * min(grid.Lx, grid.Ly)
    angle = math.radians(params.plate_angle_deg)
    dx, dy = math.cos(angle), math.sin(angle)
    px, py = XX - cx, YY - cy
    along = px * dx + py * dy
    perp = np.abs(px * (-dy) + py * dx)
    half_w = max(0.6 * grid.dx, 0.004 * min(grid.Lx, grid.Ly))
    return (np.abs(along) <= half_len) & (perp <= half_w)


def _rasterize_legacy_baffle(
    XX: np.ndarray,
    YY: np.ndarray,
    grid: Grid,
    params: CavityParams,
) -> np.ndarray:
    """Legacy vertical slot baffle (abstract geometry)."""
    if params.baffle_len_frac <= 1e-3:
        return np.zeros((grid.ny, grid.nx), dtype=bool)
    bx = params.baffle_x_frac * grid.Lx
    blen = params.baffle_len_frac * grid.Ly
    gap_center = params.baffle_gap_frac * grid.Ly
    gap_half = 0.06 * grid.Ly
    col = np.abs(XX - bx) <= (0.6 * grid.dx)
    within = YY <= blen
    not_gap = np.abs(YY - gap_center) > gap_half
    return col & within & not_gap


STRUCTURE_MODELS: tuple[str, ...] = ("dirichlet", "lossy_imag")


def _apply_pec(
    eps_r: np.ndarray,
    mu_r: np.ndarray,
    mask: np.ndarray,
    target_mask: np.ndarray,
    gangue_mask: np.ndarray,
    structure_model: str = "dirichlet",
) -> None:
    if not mask.any():
        return
    target_mask &= ~mask
    gangue_mask &= ~mask
    if structure_model == "lossy_imag":
        eps_r[mask] = _PEC_EPS
        mu_r[mask] = 1.0 + 0.0j
    # dirichlet: leave background ε (air); metal enforced in FDFD via Ez=0 rows


def build_scene(grid: Grid, params: CavityParams, materials: Materials | None = None) -> Scene:
    mats = materials or Materials()
    x, y = grid.coords()
    XX, YY = np.meshgrid(x, y)  # shape (ny, nx)

    eps_r = np.full((grid.ny, grid.nx), mats.background, dtype=complex)
    mu_r = np.full((grid.ny, grid.nx), mats.background_mu, dtype=complex)
    target_mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    gangue_mask = np.zeros((grid.ny, grid.nx), dtype=bool)

    # --- Charge bed (gangue rectangle) ---
    cx = params.charge_cx_frac * grid.Lx
    cy = params.charge_cy_frac * grid.Ly
    hw = 0.5 * params.charge_w_frac * grid.Lx
    hh = 0.5 * params.charge_h_frac * grid.Ly
    charge = (np.abs(XX - cx) <= hw) & (np.abs(YY - cy) <= hh)
    eps_r[charge] = mats.gangue
    mu_r[charge] = mats.gangue_mu
    gangue_mask |= charge

    # --- Target mineral inclusions inside the charge ---
    span = min(grid.Lx, grid.Ly)
    radii_frac = params.inclusion_radii_frac
    for i, (ox, oy) in enumerate(params.inclusion_offsets_frac):
        r_frac = radii_frac[i] if i < len(radii_frac) else params.inclusion_radius_frac
        r = r_frac * span
        ix0 = (params.charge_cx_frac + ox) * grid.Lx
        iy0 = (params.charge_cy_frac + oy) * grid.Ly
        disk = ((XX - ix0) ** 2 + (YY - iy0) ** 2) <= r ** 2
        disk &= charge
        eps_r[disk] = mats.target
        mu_r[disk] = mats.target_mu
        target_mask |= disk
        gangue_mask &= ~disk

    # --- Optional reconfigurable dielectric tuner band (lossless, non-manufacturable) ---
    if params.tuner_field:
        n = len(params.tuner_field)
        y0 = (params.tuner_y_frac - 0.5 * params.tuner_h_frac) * grid.Ly
        y1 = (params.tuner_y_frac + 0.5 * params.tuner_h_frac) * grid.Ly
        x0 = params.tuner_x0_frac * grid.Lx
        x1 = params.tuner_x1_frac * grid.Lx
        band_y = (YY >= y0) & (YY <= y1)
        for k, v in enumerate(params.tuner_field):
            cx0 = x0 + (x1 - x0) * k / n
            cx1 = x0 + (x1 - x0) * (k + 1) / n
            cell = band_y & (XX >= cx0) & (XX < cx1)
            eps_val = 1.0 + float(np.clip(v, 0.0, 1.0)) * (params.tuner_eps_max - 1.0)
            eps_r[cell] = complex(eps_val, 0.0)
            mu_r[cell] = 1.0 + 0.0j

    # --- Coax / waveguide stub (air path from wall to feed) ---
    if params.feed_wall:
        stub = _rasterize_stub(XX, YY, grid, params)
        eps_r[stub] = mats.background
        mu_r[stub] = mats.background_mu

    # --- Internal PEC: movable plate (preferred) or legacy baffle ---
    pec_mask = np.zeros((grid.ny, grid.nx), dtype=bool)
    plate = _rasterize_plate(XX, YY, grid, params)
    if plate.any():
        pec_mask |= plate
    elif params.baffle_len_frac > 1e-3:
        pec_mask |= _rasterize_legacy_baffle(XX, YY, grid, params)
    _apply_pec(
        eps_r, mu_r, pec_mask, target_mask, gangue_mask, params.structure_model,
    )

    source_xy = resolve_feed(params, grid)
    source_j = build_source_jz(params, grid)
    return Scene(
        grid=grid,
        eps_r=eps_r,
        mu_r=mu_r,
        target_mask=target_mask,
        gangue_mask=gangue_mask,
        pec_mask=pec_mask,
        source_xy=source_xy,
        source_j=source_j,
        freq_hz=params.freq_hz,
        params=params,
    )


def sample_inclusion_offsets(
    params: CavityParams,
    n_grains: int,
    rng: np.random.Generator,
    *,
    radii_frac: tuple[float, ...] = (),
    max_attempts: int = 300,
) -> tuple[tuple[float, float], ...]:
    """Random non-overlapping grain centres as offsets from the bed centre."""
    if n_grains <= 0:
        return ()
    default_r = params.inclusion_radius_frac
    r_list = [
        radii_frac[i] if i < len(radii_frac) else default_r
        for i in range(n_grains)
    ]
    max_r = max(r_list) if r_list else default_r
    hw = 0.5 * params.charge_w_frac - max_r
    hh = 0.5 * params.charge_h_frac - max_r
    if hw <= 0 or hh <= 0:
        return ()
    placed: list[tuple[float, float]] = []
    placed_r: list[float] = []
    for gi in range(n_grains):
        ri = r_list[gi]
        for _attempt in range(max_attempts):
            ox = float(rng.uniform(-hw, hw))
            oy = float(rng.uniform(-hh, hh))
            ok = all(
                (ox - px) ** 2 + (oy - py) ** 2 >= (1.1 * (ri + pr)) ** 2
                for (px, py), pr in zip(placed, placed_r)
            )
            if ok:
                placed.append((ox, oy))
                placed_r.append(ri)
                break
        else:
            break
    return tuple(placed)


def sample_inclusion_layout(
    params: CavityParams,
    radii_frac: tuple[float, ...],
    n_grains: int,
    rng: np.random.Generator,
) -> tuple[tuple[float, float], ...]:
    """Place grains with size-dependent separation (PSD-driven layouts)."""
    if not radii_frac:
        return sample_inclusion_offsets(params, n_grains, rng)
    n = min(n_grains, len(radii_frac))
    return sample_inclusion_offsets(params, n, rng, radii_frac=radii_frac[:n])


def params_with_layout(
    params: CavityParams,
    offsets: tuple[tuple[float, float], ...],
    radii_frac: tuple[float, ...] = (),
) -> CavityParams:
    """Return a copy of *params* with the given inclusion layout."""
    from dataclasses import replace

    kw: dict = {"inclusion_offsets_frac": offsets}
    if radii_frac:
        kw["inclusion_radii_frac"] = radii_frac
    return replace(params, **kw)


def build_scene_at_T(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    T_K: np.ndarray,
    freq_hz: float | None = None,
) -> Scene:
    """Build a scene with spatially varying ε(T), μ(T) from the literature tables.

    Geometry/masks match ``build_scene``; permittivity at each charge pixel comes from
    ``MineralModel.eps(T_local, f)``. Temperature is clipped to [298, 1273] K.
    """
    if T_K.shape != (grid.ny, grid.nx):
        raise ValueError(f"T_K shape {T_K.shape} != ({grid.ny}, {grid.nx})")
    pair = PAIRS[pair_label]
    freq = freq_hz if freq_hz is not None else params.freq_hz
    T_clip = np.clip(T_K, 298.0, 1273.0)

    mats = Materials.from_pair(pair_label, freq_hz=freq)
    scene = build_scene(grid, params, mats)

    eps_r = scene.eps_r.copy()
    mu_r = scene.mu_r.copy()

    def _fill(mask: np.ndarray, model) -> None:
        if not mask.any() or model is None:
            return
        temps = T_clip[mask]
        eps_vals = np.empty(temps.shape, dtype=complex)
        mu_vals = np.empty(temps.shape, dtype=complex)
        for i, t in enumerate(temps.flat):
            eps_vals.flat[i] = model.eps(float(t), freq)
            mu_vals.flat[i] = model.mu(float(t), freq)
        eps_r[mask] = eps_vals
        mu_r[mask] = mu_vals

    _fill(scene.gangue_mask, pair.gangue_model)

    from mw_inv.dielectric_data import MINERAL_MODELS, PAIR_MINERALS
    from mw_inv.phase_transitions import mineral_key_at_T, rules_for_pair

    target_key, gangue_key = PAIR_MINERALS[pair_label]
    phase_rules = rules_for_pair(pair_label)

    if phase_rules and scene.target_mask.any():
        temps = T_clip[scene.target_mask]
        eps_vals = np.empty(temps.shape, dtype=complex)
        mu_vals = np.empty(temps.shape, dtype=complex)
        for i, t in enumerate(temps.flat):
            mkey = mineral_key_at_T(target_key, float(t), rules=phase_rules)
            model = MINERAL_MODELS[mkey]
            eps_vals.flat[i] = model.eps(float(t), freq)
            mu_vals.flat[i] = model.mu(float(t), freq)
        eps_r[scene.target_mask] = eps_vals
        mu_r[scene.target_mask] = mu_vals
    else:
        _fill(scene.target_mask, pair.target_model)

    return Scene(
        grid=scene.grid,
        eps_r=eps_r,
        mu_r=mu_r,
        target_mask=scene.target_mask,
        gangue_mask=scene.gangue_mask,
        pec_mask=scene.pec_mask,
        source_xy=scene.source_xy,
        source_j=scene.source_j,
        freq_hz=scene.freq_hz,
        params=scene.params,
    )
