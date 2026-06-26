"""Ensemble evaluation over random ore grain layouts (roadmap step 5).

A single fixed inclusion pattern is optimistic — real ore has stochastic grain
positions.  We average the FOM over many random layouts inside the charge bed to
get a robust estimate and to optimise geometry that works across realizations.

Also supports **frequency-robust** scoring across the ISM band (magnetron drift)
and **thermal ensemble** evaluation (coupled EM–heat over layouts).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from mw_inv.fdfd import Grid, solve_scene
from mw_inv.fom import FomReport, evaluate
from mw_inv.geometry import (
    CavityParams,
    Materials,
    build_scene,
    params_with_layout,
    sample_inclusion_offsets,
)

if TYPE_CHECKING:
    from mw_inv.ore_profiles import OreComposition

# Industrial ISM band — magnetron frequency drift target.
ISM_FREQ_HZ = (2.40e9, 2.50e9)
DEFAULT_N_FREQS = 5


def default_ism_freqs(n: int = DEFAULT_N_FREQS) -> np.ndarray:
    lo, hi = ISM_FREQ_HZ
    return np.linspace(lo, hi, n)


@dataclass(frozen=True)
class EnsembleReport:
    mean_selectivity: float
    std_selectivity: float
    min_selectivity: float
    max_selectivity: float
    mean_contrast: float
    mean_p_total: float
    n_realizations: int
    n_grains: int
    per_realization: tuple[FomReport, ...] = ()

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mean_selectivity": self.mean_selectivity,
            "std_selectivity": self.std_selectivity,
            "min_selectivity": self.min_selectivity,
            "max_selectivity": self.max_selectivity,
            "mean_contrast": self.mean_contrast,
            "mean_p_total": self.mean_p_total,
            "n_realizations": self.n_realizations,
            "n_grains": self.n_grains,
        }


def evaluate_single_layout(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None,
    offsets: tuple[tuple[float, float], ...],
    *,
    radii_frac: tuple[float, ...] = (),
) -> FomReport:
    scene = build_scene(
        grid,
        params_with_layout(params, offsets, radii_frac),
        materials,
    )
    result = solve_scene(grid, scene)
    return evaluate(result, scene)


def evaluate_ensemble(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None = None,
    *,
    n_realizations: int = 8,
    n_grains: int = 5,
    seed: int = 0,
    ore: "OreComposition | None" = None,
) -> EnsembleReport:
    """Average FOM over *n_realizations* random grain layouts."""
    from mw_inv.ore_profiles import layout_params_with_psd

    reports: list[FomReport] = []
    use_psd = (
        ore is not None
        and ore.texture is not None
        and ore.texture.psd is not None
        and ore.texture.psd.d10_m is not None
        and ore.texture.psd.d90_m is not None
    )
    for i in range(n_realizations):
        rng = np.random.default_rng(seed + i * 10_007)
        if use_psd and ore is not None:
            layout = layout_params_with_psd(
                params, ore, n_grains=n_grains, rng=rng, cavity_span_m=grid.Lx,
            )
            offsets = layout.inclusion_offsets_frac
            radii = layout.inclusion_radii_frac
            reports.append(
                evaluate_single_layout(grid, params, materials, offsets, radii_frac=radii),
            )
        else:
            offsets = sample_inclusion_offsets(params, n_grains, rng)
            if not offsets:
                offsets = params.inclusion_offsets_frac
            reports.append(evaluate_single_layout(grid, params, materials, offsets))

    sels = np.array([r.selectivity for r in reports])
    return EnsembleReport(
        mean_selectivity=float(sels.mean()),
        std_selectivity=float(sels.std()) if len(sels) > 1 else 0.0,
        min_selectivity=float(sels.min()),
        max_selectivity=float(sels.max()),
        mean_contrast=float(np.mean([r.contrast for r in reports])),
        mean_p_total=float(np.mean([r.p_total_charge for r in reports])),
        n_realizations=n_realizations,
        n_grains=n_grains,
        per_realization=tuple(reports),
    )


@dataclass(frozen=True)
class MaterialRobustReport:
    mean_selectivity: float
    min_selectivity: float
    std_selectivity: float
    n_scenarios: int
    scenarios: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "mean_selectivity": self.mean_selectivity,
            "min_selectivity": self.min_selectivity,
            "std_selectivity": self.std_selectivity,
            "n_scenarios": self.n_scenarios,
            "scenarios": list(self.scenarios),
        }


def evaluate_material_robust(
    grid: Grid,
    params: CavityParams,
    ore: "OreComposition",
    *,
    ore_profile_path: str | None = None,
    target_T_K: float = 298.0,
    gangue_T_K: float = 298.0,
    freq_hz: float = 2.45e9,
    n_scenarios: int = 6,
    seed: int = 0,
    psd_sample: bool = True,
) -> MaterialRobustReport:
    """Worst-case selectivity over sampled moisture / PSD material scenarios."""
    from dataclasses import replace

    from mw_inv.material_scenarios import sample_material_scenarios
    from mw_inv.ore_profiles import materials_from_ore
    from mw_inv.search import evaluate_params

    scenarios = sample_material_scenarios(
        ore,
        n_scenarios=n_scenarios,
        seed=seed,
        ore_profile_path=ore_profile_path,
        target_T_K=target_T_K,
        gangue_T_K=gangue_T_K,
        freq_hz=freq_hz,
        psd_sample=psd_sample,
    )
    sels: list[float] = []
    meta: list[dict] = []
    for scen in scenarios:
        mats = materials_from_ore(
            ore,
            ore_profile_path=ore_profile_path,
            target_T_K=scen.target_T_K,
            gangue_T_K=scen.gangue_T_K,
            freq_hz=scen.freq_hz,
            moisture_wt_percent=scen.moisture_wt_percent,
        )
        p = params
        if scen.grain_radius_m is not None:
            p = replace(
                p,
                inclusion_radius_frac=float(np.clip(scen.grain_radius_m / grid.Lx, 0.008, 0.12)),
            )
        rep = evaluate_params(grid, p, mats)
        sels.append(rep.selectivity)
        meta.append({**scen.to_dict(), "selectivity": rep.selectivity})
    arr = np.array(sels)
    return MaterialRobustReport(
        mean_selectivity=float(arr.mean()),
        min_selectivity=float(arr.min()),
        std_selectivity=float(arr.std()) if len(arr) > 1 else 0.0,
        n_scenarios=n_scenarios,
        scenarios=tuple(meta),
    )


@dataclass(frozen=True)
class FrequencyRobustReport:
    mean_selectivity: float
    min_selectivity: float
    std_selectivity: float
    mean_p_total: float
    freqs_hz: tuple[float, ...]
    n_freqs: int

    def score(self, metric: str = "mean") -> float:
        if metric == "min":
            return self.min_selectivity
        if metric == "mean":
            return self.mean_selectivity
        raise ValueError("metric must be 'mean' or 'min'")

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "mean_selectivity": self.mean_selectivity,
            "min_selectivity": self.min_selectivity,
            "std_selectivity": self.std_selectivity,
            "mean_p_total": self.mean_p_total,
            "freqs_hz": list(self.freqs_hz),
            "n_freqs": self.n_freqs,
        }


@dataclass(frozen=True)
class ThermalEnsembleReport:
    mean_delta_T_K: float
    min_delta_T_K: float
    std_delta_T_K: float
    mean_heat_selectivity: float
    min_heat_selectivity: float
    n_realizations: int
    n_grains: int
    freq_robust: bool = False

    def score(self, objective: str = "delta_T", metric: str = "mean") -> float:
        if objective == "delta_T":
            return self.min_delta_T_K if metric == "min" else self.mean_delta_T_K
        if objective in ("heat_selectivity", "em_selectivity"):
            return self.min_heat_selectivity if metric == "min" else self.mean_heat_selectivity
        raise ValueError(f"unknown objective {objective!r}")

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "mean_delta_T_K": self.mean_delta_T_K,
            "min_delta_T_K": self.min_delta_T_K,
            "std_delta_T_K": self.std_delta_T_K,
            "mean_heat_selectivity": self.mean_heat_selectivity,
            "min_heat_selectivity": self.min_heat_selectivity,
            "n_realizations": self.n_realizations,
            "n_grains": self.n_grains,
            "freq_robust": self.freq_robust,
        }


def _materials_at_freq(
    materials: Materials | None,
    pair_label: str | None,
    freq_hz: float,
) -> Materials:
    if pair_label:
        return Materials.from_pair(pair_label, freq_hz=freq_hz)
    return materials or Materials()


def evaluate_at_freq(
    grid: Grid,
    params: CavityParams,
    freq_hz: float,
    materials: Materials | None = None,
    pair_label: str | None = None,
    offsets: tuple[tuple[float, float], ...] | None = None,
) -> FomReport:
    p = replace(params, freq_hz=float(freq_hz))
    if offsets is not None:
        p = params_with_layout(p, offsets)
    mats = _materials_at_freq(materials, pair_label, freq_hz)
    scene = build_scene(grid, p, mats)
    result = solve_scene(grid, scene)
    return evaluate(result, scene)


def evaluate_frequency_robust(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None = None,
    *,
    pair_label: str | None = None,
    freqs_hz: np.ndarray | None = None,
    n_freqs: int = DEFAULT_N_FREQS,
    offsets: tuple[tuple[float, float], ...] | None = None,
) -> FrequencyRobustReport:
    """Selectivity statistics across the ISM frequency band."""
    freqs = freqs_hz if freqs_hz is not None else default_ism_freqs(n_freqs)
    reports = [
        evaluate_at_freq(grid, params, float(f), materials, pair_label, offsets)
        for f in freqs
    ]
    sels = np.array([r.selectivity for r in reports])
    return FrequencyRobustReport(
        mean_selectivity=float(sels.mean()),
        min_selectivity=float(sels.min()),
        std_selectivity=float(sels.std()) if len(sels) > 1 else 0.0,
        mean_p_total=float(np.mean([r.p_total_charge for r in reports])),
        freqs_hz=tuple(float(f) for f in freqs),
        n_freqs=len(freqs),
    )


def evaluate_frequency_robust_ensemble(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None = None,
    *,
    pair_label: str | None = None,
    n_realizations: int = 6,
    n_grains: int = 5,
    seed: int = 0,
    n_freqs: int = DEFAULT_N_FREQS,
) -> FrequencyRobustReport:
    """Mean/min selectivity over layouts × frequencies."""
    freqs = default_ism_freqs(n_freqs)
    sels: list[float] = []
    p_tot: list[float] = []
    for i in range(n_realizations):
        rng = np.random.default_rng(seed + i * 10_007)
        offsets = sample_inclusion_offsets(params, n_grains, rng) or params.inclusion_offsets_frac
        for f in freqs:
            r = evaluate_at_freq(grid, params, float(f), materials, pair_label, offsets)
            sels.append(r.selectivity)
            p_tot.append(r.p_total_charge)
    arr = np.array(sels)
    return FrequencyRobustReport(
        mean_selectivity=float(arr.mean()),
        min_selectivity=float(arr.min()),
        std_selectivity=float(arr.std()) if len(arr) > 1 else 0.0,
        mean_p_total=float(np.mean(p_tot)),
        freqs_hz=tuple(float(f) for f in freqs),
        n_freqs=len(freqs),
    )


def evaluate_thermal_ensemble(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    *,
    n_realizations: int = 6,
    n_grains: int = 5,
    seed: int = 0,
    thermal_cfg=None,
    freq_robust: bool = False,
    n_freqs: int = DEFAULT_N_FREQS,
) -> ThermalEnsembleReport:
    """Coupled EM–thermal FOM averaged over random grain layouts (optional freq band)."""
    from mw_inv.thermal import ThermalConfig, coupled_steady_state, thermal_props_for_pair

    cfg = thermal_cfg or ThermalConfig(
        drive=8.0,
        thermal_props=thermal_props_for_pair(pair_label),
        max_iters=12,
        tol_K=4.0,
    )
    freqs = list(default_ism_freqs(n_freqs)) if freq_robust else [params.freq_hz]
    delta_ts: list[float] = []
    heat_sels: list[float] = []

    for i in range(n_realizations):
        rng = np.random.default_rng(seed + i * 10_007)
        offsets = sample_inclusion_offsets(params, n_grains, rng) or params.inclusion_offsets_frac
        for f in freqs:
            p = params_with_layout(replace(params, freq_hz=float(f)), offsets)
            res = coupled_steady_state(grid, pair_label, config=cfg, params=p)
            delta_ts.append(res.thermal.delta_T_K)
            heat_sels.append(res.thermal.heat_selectivity)

    d = np.array(delta_ts)
    h = np.array(heat_sels)
    return ThermalEnsembleReport(
        mean_delta_T_K=float(d.mean()),
        min_delta_T_K=float(d.min()),
        std_delta_T_K=float(d.std()) if len(d) > 1 else 0.0,
        mean_heat_selectivity=float(h.mean()),
        min_heat_selectivity=float(h.min()),
        n_realizations=n_realizations,
        n_grains=n_grains,
        freq_robust=freq_robust,
    )
