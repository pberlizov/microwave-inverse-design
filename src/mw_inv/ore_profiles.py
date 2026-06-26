"""Ore-scale material grounding: HMAP content, heating classes, effective ε.

Anchored to Goldbaum et al. (Toronto / CEEComm 2022): highly microwave-amenable
phases (HMAPs) and four bench heating classes from 42 ores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from mw_inv.measured_dielectrics import load_measured_dielectrics, validate_library
from mw_inv.dielectric_data import (
    DEFAULT_GANGUE_MINERAL,
    PAIR_MINERALS,
    mineral_eps,
    mineral_mu,
)
from mw_inv.geometry import CavityParams
from mw_inv.materials import Materials, PAIRS

# Minerals classified as HMAPs (Goldbaum / Forster et al. IMPC 2022)
HMAP_MINERALS: tuple[str, ...] = (
    "bornite", "chalcopyrite", "galena", "hematite", "magnetite",
    "molybdenite", "pentlandite", "pyrite", "pyrrhotite",
)

# (class, hmap_lo, hmap_hi, rate_min, rate_max, rate_mean) °C/min first 30 s
HEATING_CLASSES: tuple[tuple[str, float, float, float, float, float], ...] = (
    ("I_poor", 0.0, 2.0, 13.0, 40.0, 23.0),
    ("II_fair", 2.0, 5.0, 21.0, 133.0, 76.0),
    ("III_good", 5.0, 20.0, 105.0, 339.0, 159.0),
    ("IV_excellent", 20.0, 100.0, 150.0, 829.0, 559.0),
)

# Pilot-scale arcing risk — representative ceiling (Goldbaum: massive sulfides, high σ)
MAX_SAFE_POWER_DENSITY_W_M3 = 5.0e7   # order-of-magnitude; calibrate on your system
MAX_SAFE_LOSS_TANGENT = 1.5           # tan δ above this → arcing risk flag


@dataclass(frozen=True)
class OrePSD:
    """Particle-size distribution characteristic diameters [m]."""

    d10_m: float | None = None
    d50_m: float | None = None
    d90_m: float | None = None


# Default solid packing by texture when not measured (typical crushed ore beds).
DEFAULT_PACKING_BY_TEXTURE: dict[str, float] = {
    "disseminated": 0.62,
    "massive": 0.88,
    "vein": 0.72,
}


@dataclass(frozen=True)
class OreTexture:
    """QEMSCAN-derived texture hints for geometry defaults."""

    texture_class: str | None = None  # disseminated | massive | vein
    mean_grain_radius_m: float | None = None
    packing_fraction: float | None = None  # solid volume fraction in charge (0–1)
    psd: OrePSD | None = None


@dataclass(frozen=True)
class OreComposition:
    """Weight fractions of minerals in an ore (sum ≤ 1; remainder = gangue silicate."""

    label: str
    fractions: dict[str, float]  # mineral name → wt fraction
    gangue_label: str = "calcite_silicate"
    gangue_mineral: str | None = None
    texture: OreTexture | None = None
    source: str | None = None
    measured_dielectrics: dict | None = None

    @property
    def hmap_wt_percent(self) -> float:
        return 100.0 * sum(
            self.fractions.get(m, 0.0) for m in HMAP_MINERALS
        )

    def heating_class(self) -> str:
        w = self.hmap_wt_percent
        for name, lo, hi, *_ in HEATING_CLASSES:
            if lo <= w < hi:
                return name
        return "IV_excellent"

    def predicted_heating_rate_C_per_min(self) -> float:
        """Piecewise linear within each Goldbaum class (min → mean → max vs HMAP %)."""
        w = self.hmap_wt_percent
        for _name, lo, hi, rmin, rmax, rmean in HEATING_CLASSES:
            if lo <= w < hi:
                mid = 0.5 * (lo + hi)
                if w <= mid:
                    return float(np.interp(w, [lo, mid], [rmin, rmean]))
                return float(np.interp(w, [mid, hi], [rmean, rmax]))
        _name, lo, hi, rmin, rmax, rmean = HEATING_CLASSES[-1]
        return float(rmax)


# Named example ores (illustrative — replace with QEMSCAN on your sample)
ORE_PROFILES: dict[str, OreComposition] = {
    "disseminated_pyrite_porphyry": OreComposition(
        "disseminated_pyrite_porphyry",
        {"pyrite": 0.04, "chalcopyrite": 0.02},
    ),
    "massive_pyrite": OreComposition(
        "massive_pyrite",
        {"pyrite": 0.65},
    ),
    "magnetite_skarn": OreComposition(
        "magnetite_skarn",
        {"magnetite": 0.25, "pyrite": 0.03},
    ),
    "barren_calcite": OreComposition(
        "barren_calcite",
        {},
    ),
}


_GANGUE_LABEL_MAP: dict[str, str] = {
    "calcite_silicate": "calcite",
    "quartz_feldspar": "quartz",
    "carbonate": "calcite",
}


def load_ore_profile(path: Path | str) -> OreComposition:
    """Load a deposit ore profile from JSON (QEMSCAN / assay modal analysis)."""
    ore_path = Path(path)
    data = json.loads(ore_path.read_text())
    if "label" not in data or "fractions" not in data:
        raise ValueError("ore JSON must include 'label' and 'fractions'")

    measured_block = data.get("measured_dielectrics")
    if measured_block is not None:
        _validate_measured_dielectrics_block(measured_block)

    texture: OreTexture | None = None
    if data.get("texture"):
        t = data["texture"]
        psd_block = t.get("psd")
        psd: OrePSD | None = None
        if isinstance(psd_block, dict):
            psd = OrePSD(
                d10_m=psd_block.get("d10_m"),
                d50_m=psd_block.get("d50_m"),
                d90_m=psd_block.get("d90_m"),
            )
        texture = OreTexture(
            texture_class=t.get("class") or t.get("texture_class"),
            mean_grain_radius_m=t.get("mean_grain_radius_m"),
            packing_fraction=t.get("packing_fraction"),
            psd=psd,
        )

    fractions = {str(k): float(v) for k, v in data["fractions"].items()}
    return OreComposition(
        label=str(data["label"]),
        fractions=fractions,
        gangue_label=str(data.get("gangue_label", "calcite_silicate")),
        gangue_mineral=data.get("gangue_mineral"),
        texture=texture,
        source=data.get("source"),
        measured_dielectrics=data.get("measured_dielectrics"),
    )


def _validate_measured_dielectrics_block(block: object) -> None:
    if not isinstance(block, dict):
        raise ValueError("measured_dielectrics must be an object")
    if not block.get("path"):
        raise ValueError("measured_dielectrics.path is required")


def resolve_measured_dielectrics_path(
    ore_profile_path: Path | str,
    measured_block: dict,
) -> Path:
    """Resolve measured ε JSON relative to ore profile dir, then cwd."""
    raw = measured_block.get("path")
    if not raw:
        raise ValueError("measured_dielectrics.path is required")
    p = Path(str(raw))
    if p.is_file():
        return p.resolve()
    ore_dir = Path(ore_profile_path).resolve().parent
    for candidate in (ore_dir / p, Path.cwd() / p):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"measured dielectrics path not found: {raw}")


def dominant_hmap(ore: OreComposition) -> str | None:
    """Return the HMAP mineral with the largest weight fraction, if any."""
    best: tuple[str, float] | None = None
    for m in HMAP_MINERALS:
        f = ore.fractions.get(m, 0.0)
        if f > 1e-9 and (best is None or f > best[1]):
            best = (m, f)
    return best[0] if best else None


def infer_gangue_mineral(ore: OreComposition) -> str:
    """Infer matrix gangue mineral from explicit field, fractions, or label."""
    from mw_inv.mineral_catalog import inactive_gangue

    if ore.gangue_mineral:
        return ore.gangue_mineral
    gangue_keys = set(inactive_gangue())
    matrix_fracs = [(m, ore.fractions[m]) for m in gangue_keys if ore.fractions.get(m, 0.0) > 1e-9]
    if matrix_fracs:
        return max(matrix_fracs, key=lambda x: x[1])[0]
    return _GANGUE_LABEL_MAP.get(ore.gangue_label, DEFAULT_GANGUE_MINERAL)


def suggest_material_pair(ore: OreComposition) -> str:
    """Pick the best cited MaterialPair for this ore composition."""
    from mw_inv.mineral_catalog import loss_contrast

    gangue = infer_gangue_mineral(ore)
    hmap = dominant_hmap(ore)
    if hmap:
        direct = f"{hmap}_in_{gangue}"
        if direct in PAIRS:
            return direct

    hmap_present = {m for m in HMAP_MINERALS if ore.fractions.get(m, 0.0) > 1e-9}
    best_label = "pyrite_in_calcite"
    best_score = -1.0
    for pair_label, (target, pair_gangue) in PAIR_MINERALS.items():
        if hmap_present and target not in hmap_present:
            continue
        score = loss_contrast(target, pair_gangue)
        if score > best_score:
            best_score = score
            best_label = pair_label
    return best_label


def resolve_packing_fraction(ore: OreComposition) -> float | None:
    """Solid volume fraction for mixing; explicit value or texture-class default."""
    if ore.texture is None:
        return None
    if ore.texture.packing_fraction is not None:
        return float(np.clip(ore.texture.packing_fraction, 0.05, 0.98))
    tc = (ore.texture.texture_class or "").lower()
    return DEFAULT_PACKING_BY_TEXTURE.get(tc)


def effective_grain_radius_m(texture: OreTexture) -> float | None:
    """Grain radius from explicit mean or PSD d50 (diameter → radius)."""
    if texture.mean_grain_radius_m is not None:
        return max(float(texture.mean_grain_radius_m), 1e-6)
    if texture.psd and texture.psd.d50_m is not None:
        return max(float(texture.psd.d50_m) * 0.5, 1e-6)
    return None


def sample_psd_radii_m(
    texture: OreTexture,
    n_grains: int,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    """Sample grain radii [m] uniform in [d10/2, d90/2] when PSD present."""
    if n_grains <= 0:
        return ()
    if texture.psd and texture.psd.d10_m is not None and texture.psd.d90_m is not None:
        lo = min(texture.psd.d10_m, texture.psd.d90_m) * 0.5
        hi = max(texture.psd.d10_m, texture.psd.d90_m) * 0.5
        if hi <= lo:
            hi = lo * 1.01
        diam = rng.uniform(lo * 2.0, hi * 2.0, size=n_grains)
        return tuple(float(d) * 0.5 for d in diam)
    r = effective_grain_radius_m(texture)
    if r is None:
        return ()
    return (r,) * n_grains


def psd_radii_frac(
    texture: OreTexture,
    n_grains: int,
    rng: np.random.Generator,
    *,
    cavity_span_m: float,
) -> tuple[float, ...]:
    """Grain radii as fractions of cavity span for scene layout."""
    radii_m = sample_psd_radii_m(texture, n_grains, rng)
    if not radii_m:
        return ()
    return tuple(float(np.clip(r / cavity_span_m, 0.008, 0.12)) for r in radii_m)


def layout_params_with_psd(
    params: CavityParams,
    ore: OreComposition,
    *,
    n_grains: int,
    rng: np.random.Generator,
    cavity_span_m: float = 0.36,
) -> CavityParams:
    """Random PSD-sized grain layout inside the charge bed."""
    from dataclasses import replace
    from mw_inv.geometry import params_with_layout, sample_inclusion_layout

    if ore.texture is None or ore.texture.psd is None:
        return params
    radii_frac = psd_radii_frac(ore.texture, n_grains, rng, cavity_span_m=cavity_span_m)
    if not radii_frac:
        return params
    offsets = sample_inclusion_layout(params, radii_frac, len(radii_frac), rng)
    if not offsets:
        return params
    n = len(offsets)
    return params_with_layout(params, offsets, radii_frac[:n])


def porosity_diluted_eps(solid_eps: complex, packing_fraction: float) -> complex:
    """Bruggeman mix of solid mineral with air voids at porosity (1 − packing)."""
    packing = float(np.clip(packing_fraction, 0.05, 0.98))
    void_eps = 1.0 + 0.0j
    return bruggeman_effective_eps([solid_eps, void_eps], [packing, 1.0 - packing])


def cavity_params_from_ore(
    ore: OreComposition,
    base: CavityParams | None = None,
    *,
    cavity_span_m: float = 0.36,
) -> CavityParams:
    """Map ore texture to default charge/grain geometry knobs."""
    p = CavityParams() if base is None else replace(base)
    if ore.texture is None:
        return p

    grain_r = effective_grain_radius_m(ore.texture)
    if grain_r is not None:
        p.inclusion_radius_frac = float(np.clip(grain_r / cavity_span_m, 0.01, 0.12))

    tc = (ore.texture.texture_class or "").lower()
    if tc == "disseminated":
        p.inclusion_offsets_frac = (
            (-0.10, -0.04), (0.10, -0.04), (0.0, 0.08),
            (-0.05, 0.05), (0.05, 0.05),
        )
    elif tc == "massive":
        p.inclusion_offsets_frac = ((0.0, 0.0),)
        if ore.texture.mean_grain_radius_m is None:
            p.inclusion_radius_frac = 0.08
    return p


def ore_summary(
    ore: OreComposition,
    *,
    ore_profile_path: Path | str | None = None,
    target_T_K: float = 298.0,
    gangue_T_K: float = 298.0,
    freq_hz: float = 2.45e9,
    moisture_wt_percent: float | None = None,
) -> dict[str, object]:
    """Compact report dict for ingest CLI and pipeline manifests."""
    from mw_inv.mineral_catalog import loss_contrast

    pair = suggest_material_pair(ore)
    gangue = infer_gangue_mineral(ore)
    target, pair_gangue = PAIR_MINERALS[pair]
    measured = ore.measured_dielectrics or {}
    mode = "bruggeman"
    measured_detail: dict | None = None
    if measured.get("path"):
        try:
            mp = resolve_measured_dielectrics_path(ore_profile_path or ".", measured)
            lib = load_measured_dielectrics(mp)
            issues = validate_library(lib)
            moisture = moisture_wt_percent
            if moisture is None and measured.get("moisture_wt_percent") is not None:
                moisture = float(measured["moisture_wt_percent"])
            target_phase = str(measured.get("target_phase", "target"))
            gangue_phase = str(measured.get("gangue_phase", "gangue"))
            mode = "measured"
            measured_detail = {
                "path": str(mp),
                "target_phase": target_phase,
                "gangue_phase": gangue_phase,
                "moisture_wt_percent": moisture,
                "dataset": lib.summary(),
                "eval": {
                    "target_T_K": target_T_K,
                    "gangue_T_K": gangue_T_K,
                    "freq_hz": freq_hz,
                    "target_eps": [
                        lib.eps(target_phase, temp_K=target_T_K, freq_hz=freq_hz, moisture_wt_percent=moisture).real,
                        lib.eps(target_phase, temp_K=target_T_K, freq_hz=freq_hz, moisture_wt_percent=moisture).imag,
                    ],
                    "gangue_eps": [
                        lib.eps(gangue_phase, temp_K=gangue_T_K, freq_hz=freq_hz, moisture_wt_percent=moisture).real,
                        lib.eps(gangue_phase, temp_K=gangue_T_K, freq_hz=freq_hz, moisture_wt_percent=moisture).imag,
                    ],
                },
                "validation_issues": issues,
            }
        except (FileNotFoundError, ValueError, KeyError) as exc:
            measured_detail = {"path": measured.get("path"), "error": str(exc)}
    return {
        "label": ore.label,
        "source": ore.source,
        "hmap_wt_percent": ore.hmap_wt_percent,
        "dominant_hmap": dominant_hmap(ore),
        "inferred_gangue": gangue,
        "suggested_pair": pair,
        "heating_class": ore.heating_class(),
        "predicted_rate_C_per_min": ore.predicted_heating_rate_C_per_min(),
        "loss_contrast": loss_contrast(target, pair_gangue),
        "materials_mode": mode,
        "measured_dielectrics": measured_detail,
        "texture": None if ore.texture is None else {
            "class": ore.texture.texture_class,
            "mean_grain_radius_m": ore.texture.mean_grain_radius_m,
            "effective_grain_radius_m": (
                effective_grain_radius_m(ore.texture) if ore.texture else None
            ),
            "packing_fraction": resolve_packing_fraction(ore),
            "psd": None if ore.texture.psd is None else {
                "d10_m": ore.texture.psd.d10_m,
                "d50_m": ore.texture.psd.d50_m,
                "d90_m": ore.texture.psd.d90_m,
            },
        },
    }

def bruggeman_effective_eps(eps_list: list[complex], vol_fracs: list[float]) -> complex:
    """Bruggeman symmetric effective medium (2–3 phases)."""
    f = np.array(vol_fracs, dtype=float)
    f = f / f.sum()
    eps_eff = complex(sum(f[i] * eps_list[i] for i in range(len(eps_list))))
    for _ in range(60):
        s = 0.0 + 0.0j
        for i in range(len(eps_list)):
            s += f[i] * (eps_list[i] - eps_eff) / (eps_list[i] + 2.0 * eps_eff)
        if abs(s) < 1e-14:
            break
        eps_eff = eps_eff * (1.0 + s / (1.0 - s))
    return complex(eps_eff)


def materials_from_ore(
    ore: OreComposition,
    *,
    pair_fallback: str | None = None,
    target_T_K: float = 298.0,
    gangue_T_K: float = 298.0,
    freq_hz: float = 2.45e9,
    gangue_mineral: str | None = None,
    ore_profile_path: Path | str | None = None,
    moisture_wt_percent: float | None = None,
) -> Materials:
    """Effective target/gangue ε from measured curves or Bruggeman mixing."""
    pair_label = pair_fallback or suggest_material_pair(ore)
    pair = PAIRS.get(pair_label)
    if pair is None:
        raise KeyError(pair_label)

    gangue_key = gangue_mineral or infer_gangue_mineral(ore)

    # Deposit/batch specific measured ε overrides mixing models.
    measured = ore.measured_dielectrics or {}
    if measured.get("path"):
        mp = resolve_measured_dielectrics_path(ore_profile_path or ".", measured)
        lib = load_measured_dielectrics(mp)
        issues = validate_library(lib)
        if issues:
            raise ValueError(f"invalid measured dielectrics: {issues[0]}")
        moisture = moisture_wt_percent
        if moisture is None and measured.get("moisture_wt_percent") is not None:
            moisture = float(measured["moisture_wt_percent"])
        target_phase = measured.get("target_phase", "target")
        gangue_phase = measured.get("gangue_phase", "gangue")
        t_eps = lib.eps(
            str(target_phase),
            temp_K=target_T_K,
            freq_hz=freq_hz,
            moisture_wt_percent=moisture,
        )
        g_eps = lib.eps(
            str(gangue_phase),
            temp_K=gangue_T_K,
            freq_hz=freq_hz,
            moisture_wt_percent=moisture,
        )
        return Materials(
            background=pair.background,
            target=t_eps,
            gangue=g_eps,
            target_mu=1.0 + 0.0j,
            gangue_mu=1.0 + 0.0j,
            target_T_K=target_T_K,
            gangue_T_K=gangue_T_K,
            freq_hz=freq_hz,
            pair_label=pair_label,
        )

    hmap_items = [
        (m, ore.fractions[m])
        for m in HMAP_MINERALS
        if ore.fractions.get(m, 0.0) > 1e-9
    ]
    hmap_frac = sum(f for _, f in hmap_items)

    gangue_eps = mineral_eps(gangue_key, gangue_T_K, freq_hz)
    gangue_mu = mineral_mu(gangue_key, gangue_T_K, freq_hz)
    packing = resolve_packing_fraction(ore)
    if packing is not None and packing < 0.999:
        gangue_eps = porosity_diluted_eps(gangue_eps, packing)
        gangue_mu = porosity_diluted_eps(gangue_mu, packing)

    if hmap_frac < 1e-6:
        return Materials(
            background=pair.background,
            target=gangue_eps,
            gangue=gangue_eps,
            target_mu=gangue_mu,
            gangue_mu=gangue_mu,
            target_T_K=target_T_K,
            gangue_T_K=gangue_T_K,
            freq_hz=freq_hz,
            pair_label=pair_label,
        )

    fracs = [f / hmap_frac for _, f in hmap_items]
    hmap_eps = [mineral_eps(m, target_T_K, freq_hz) for m, _ in hmap_items]
    hmap_mu = [mineral_mu(m, target_T_K, freq_hz) for m, _ in hmap_items]
    target_eps = bruggeman_effective_eps(hmap_eps, fracs)
    target_mu = bruggeman_effective_eps(hmap_mu, fracs)
    if packing is not None and packing < 0.999:
        target_eps = porosity_diluted_eps(target_eps, packing)
        target_mu = porosity_diluted_eps(target_mu, packing)

    return Materials(
        background=pair.background,
        target=target_eps,
        gangue=gangue_eps,
        target_mu=target_mu,
        gangue_mu=gangue_mu,
        target_T_K=target_T_K,
        gangue_T_K=gangue_T_K,
        freq_hz=freq_hz,
        pair_label=pair_label,
    )


def power_density_W_m3(p_total: float, charge_volume_m3: float) -> float:
    return p_total / max(charge_volume_m3, 1e-12)


def arcing_risk_flag(
    p_total: float,
    charge_volume_m3: float,
    materials: Materials,
    freq_hz: float = 2.45e9,
) -> dict[str, float | bool]:
    """Flag designs exceeding representative power-density / loss limits."""
    tan_d = max(materials.target.imag, materials.gangue.imag) / max(materials.target.real, 1e-6)
    pd = power_density_W_m3(p_total, charge_volume_m3)
    return {
        "power_density_W_m3": pd,
        "loss_tangent": tan_d,
        "exceeds_power_limit": pd > MAX_SAFE_POWER_DENSITY_W_M3,
        "exceeds_loss_limit": tan_d > MAX_SAFE_LOSS_TANGENT,
        "arcing_risk": pd > MAX_SAFE_POWER_DENSITY_W_M3 or tan_d > MAX_SAFE_LOSS_TANGENT,
    }


def charge_volume_m3(params, *, Lx: float = 0.36, Ly: float = 0.36, depth: float = 0.36) -> float:
    return params.charge_w_frac * Lx * params.charge_h_frac * Ly * depth
