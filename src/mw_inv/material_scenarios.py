"""Material uncertainty scenarios for robust evaluation (C1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mw_inv.ore_profiles import OreComposition, effective_grain_radius_m, resolve_measured_dielectrics_path


@dataclass(frozen=True)
class MaterialScenario:
    moisture_wt_percent: float | None
    target_T_K: float
    gangue_T_K: float
    freq_hz: float
    grain_radius_m: float | None = None

    def to_dict(self) -> dict:
        return {
            "moisture_wt_percent": self.moisture_wt_percent,
            "target_T_K": self.target_T_K,
            "gangue_T_K": self.gangue_T_K,
            "freq_hz": self.freq_hz,
            "grain_radius_m": self.grain_radius_m,
        }


def _moisture_levels(ore: OreComposition, ore_profile_path: str | None) -> list[float]:
    measured = ore.measured_dielectrics or {}
    if not measured.get("path"):
        return [0.0, 3.0]
    from mw_inv.measured_dielectrics import load_measured_dielectrics

    mp = resolve_measured_dielectrics_path(ore_profile_path or ".", measured)
    lib = load_measured_dielectrics(mp)
    summary = lib.summary()
    target = summary.get("phases", {}).get(
        str(measured.get("target_phase", "target")), {},
    )
    levels = target.get("moisture_wt_percent_levels") or []
    return [float(x) for x in levels] if levels else [0.0, 3.0]


def sample_material_scenarios(
    ore: OreComposition,
    *,
    n_scenarios: int,
    seed: int,
    ore_profile_path: str | None = None,
    target_T_K: float = 298.0,
    gangue_T_K: float = 298.0,
    freq_hz: float = 2.45e9,
    moisture_lo: float | None = None,
    moisture_hi: float | None = None,
    psd_sample: bool = True,
) -> list[MaterialScenario]:
    """Draw moisture / PSD scenarios for worst-case material robustness."""
    rng = np.random.default_rng(seed)
    levels = _moisture_levels(ore, ore_profile_path)
    m_lo = moisture_lo if moisture_lo is not None else min(levels)
    m_hi = moisture_hi if moisture_hi is not None else max(levels)
    scenarios: list[MaterialScenario] = []
    for _ in range(n_scenarios):
        moisture = float(rng.uniform(m_lo, m_hi))
        grain_r: float | None = None
        if psd_sample and ore.texture and ore.texture.psd:
            from mw_inv.ore_profiles import sample_psd_radii_m

            radii = sample_psd_radii_m(ore.texture, 1, rng)
            grain_r = radii[0] if radii else effective_grain_radius_m(ore.texture)
        scenarios.append(MaterialScenario(
            moisture_wt_percent=moisture,
            target_T_K=target_T_K,
            gangue_T_K=gangue_T_K,
            freq_hz=freq_hz,
            grain_radius_m=grain_r,
        ))
    return scenarios
