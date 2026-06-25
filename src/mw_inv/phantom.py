"""Gel-phantom lab workflow: recipe-linked ε, thermal predictions, bench compare.

Uses ``phantom_data.saline_eps`` (Gabriel-scaled salt anchors) rather than hand-picked
constants.  Predictions include coupled EM–thermal ΔT where applicable, and
``compare_lab_measurement`` scores model vs bench data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.fdfd import Grid, solve
from mw_inv.fom import FomReport, evaluate
from mw_inv.geometry import CavityParams, Materials, build_scene
from mw_inv.phantom_data import (
    GEL_K_W_MK,
    GEL_RHO_CP,
    PHANTOM_RECIPES,
    PhantomRecipePair,
)
from mw_inv.search import best, optuna_search
from mw_inv.thermal import PhaseThermalProps, ThermalConfig, coupled_steady_state


def materials_from_recipe(pair: PhantomRecipePair) -> Materials:
    return Materials(
        target=pair.target.eps,
        gangue=pair.gangue.eps,
        background=1.0 + 0.0j,
    )


def phantom_thermal_props() -> PhaseThermalProps:
    """Gel-appropriate thermal properties (water-dominated)."""
    return PhaseThermalProps(
        target_k=GEL_K_W_MK,
        gangue_k=GEL_K_W_MK,
        target_rho_cp=GEL_RHO_CP,
        gangue_rho_cp=GEL_RHO_CP,
    )


@dataclass(frozen=True)
class LabPrediction:
    phantom: str
    mineral_analog: str
    target_salt_wt: float
    gangue_salt_wt: float
    target_eps: complex
    gangue_eps: complex
    untuned_selectivity: float
    untuned_contrast: float
    untuned_delta_T_K: float
    optimized_selectivity: float
    optimized_contrast: float
    optimized_delta_T_K: float
    optimized_params: dict
    measurement_protocol: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "phantom": self.phantom,
            "mineral_analog": self.mineral_analog,
            "target_salt_wt_percent": self.target_salt_wt,
            "gangue_salt_wt_percent": self.gangue_salt_wt,
            "target_eps": [self.target_eps.real, self.target_eps.imag],
            "gangue_eps": [self.gangue_eps.real, self.gangue_eps.imag],
            "untuned_selectivity": self.untuned_selectivity,
            "untuned_contrast": self.untuned_contrast,
            "untuned_delta_T_K": self.untuned_delta_T_K,
            "optimized_selectivity": self.optimized_selectivity,
            "optimized_contrast": self.optimized_contrast,
            "optimized_delta_T_K": self.optimized_delta_T_K,
            "optimized_params": self.optimized_params,
            "measurement_protocol": list(self.measurement_protocol),
        }


@dataclass(frozen=True)
class LabComparison:
    phantom: str
    predicted_selectivity: float
    predicted_delta_T_K: float
    measured_selectivity: float | None
    measured_delta_T_K: float
    selectivity_error: float | None
    delta_T_error_K: float
    rank_correct: bool | None  # True if optimized beats untuned on measured ΔT

    def to_dict(self) -> dict:
        return {
            "phantom": self.phantom,
            "predicted_selectivity": self.predicted_selectivity,
            "predicted_delta_T_K": self.predicted_delta_T_K,
            "measured_selectivity": self.measured_selectivity,
            "measured_delta_T_K": self.measured_delta_T_K,
            "selectivity_error": self.selectivity_error,
            "delta_T_error_K": self.delta_T_error_K,
            "rank_correct": self.rank_correct,
        }


DEFAULT_PROTOCOL: tuple[str, ...] = (
    "Prepare agar/glycerol gel batches at target and gangue NaCl wt% (see target_salt_wt_percent).",
    "Measure ε′, ε″ at 2.45 GHz on each batch (open coax probe or cavity perturbation) — compare to model anchors.",
    "Build 36 cm aluminium cavity; coax feed per optimized_params feed_wall / feed_along_frac / stub_depth_frac.",
    "Embed high-salt gel inclusions in low-salt bed (inclusion_radius_frac, charge geometry from params).",
    "Drive CW at 2.45 GHz; record S11, forward power, IR/fibre ΔT (target − gangue) at 60–120 s.",
    "Success: optimized geometry ΔT > untuned on same gel batch; model rank matches measurement.",
)


def _em_report(grid: Grid, params: CavityParams, mats: Materials) -> FomReport:
    scene = build_scene(grid, params, mats)
    res = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r)
    return evaluate(res, scene)


def _thermal_delta(
    grid: Grid,
    params: CavityParams,
    mats: Materials,
    *,
    drive: float = 8.0,
) -> float:
    """Coupled ΔT using gel thermal props and mineral ε(T) tables as fallback."""
    # Use pyrite pair only for ε(T) shape if needed; gel ε is fixed in mats via build_scene
    cfg = ThermalConfig(
        drive=drive,
        thermal_props=phantom_thermal_props(),
        max_iters=15,
        tol_K=4.0,
    )
    # coupled_steady_state expects pair_label for ε(T); use fixed materials via custom path
    # Build with static mats: pass a dummy label but override via scene build — use pyrite with
    # materials embedded in params by building scene manually in thermal loop.
    # Simplest: use pyrite_in_calcite tables for T-feedback shape on similar ε magnitude.
    res = coupled_steady_state(
        grid, "", config=cfg, params=params, materials=mats,
    )
    return res.thermal.delta_T_K


def predict_lab_outcome(
    phantom_label: str,
    grid: Grid | None = None,
    *,
    n_opt_trials: int = 40,
    seed: int = 7701,
    drive: float = 8.0,
    measured_eps_path: Path | str | None = None,
) -> LabPrediction:
    recipe = PHANTOM_RECIPES[phantom_label]
    grid = grid or Grid(nx=81, ny=81, Lx=0.36, Ly=0.36)
    if measured_eps_path:
        from mw_inv.phantom_calibration import materials_from_measured_recipe
        mats = materials_from_measured_recipe(phantom_label, measured_eps_path)
    else:
        mats = materials_from_recipe(recipe)

    untuned_em = _em_report(grid, CavityParams(), mats)
    untuned_dT = _thermal_delta(grid, CavityParams(), mats, drive=drive)

    trials = optuna_search(grid, n_opt_trials, seed, materials=mats)
    opt = best(trials)
    opt_params = CavityParams(**opt.params)
    opt_em = _em_report(grid, opt_params, mats)
    opt_dT = _thermal_delta(grid, opt_params, mats, drive=drive)

    return LabPrediction(
        phantom=phantom_label,
        mineral_analog=recipe.mineral_analog,
        target_salt_wt=recipe.target.salt_wt_percent,
        gangue_salt_wt=recipe.gangue.salt_wt_percent,
        target_eps=recipe.target.eps,
        gangue_eps=recipe.gangue.eps,
        untuned_selectivity=untuned_em.selectivity,
        untuned_contrast=untuned_em.contrast,
        untuned_delta_T_K=untuned_dT,
        optimized_selectivity=opt_em.selectivity,
        optimized_contrast=opt_em.contrast,
        optimized_delta_T_K=opt_dT,
        optimized_params=opt.params,
        measurement_protocol=DEFAULT_PROTOCOL,
    )


def predict_all_phantoms(grid: Grid | None = None, **kwargs) -> list[LabPrediction]:
    return [predict_lab_outcome(label, grid, **kwargs) for label in sorted(PHANTOM_RECIPES)]


def compare_lab_measurement(
    prediction: LabPrediction,
    measured_delta_T_K: float,
    measured_selectivity: float | None = None,
    *,
    untuned_measured_delta_T_K: float | None = None,
) -> LabComparison:
    """Score model prediction against bench results."""
    sel_err = None
    if measured_selectivity is not None:
        sel_err = prediction.optimized_selectivity - measured_selectivity
    rank = None
    if untuned_measured_delta_T_K is not None:
        rank = measured_delta_T_K > untuned_measured_delta_T_K
    return LabComparison(
        phantom=prediction.phantom,
        predicted_selectivity=prediction.optimized_selectivity,
        predicted_delta_T_K=prediction.optimized_delta_T_K,
        measured_selectivity=measured_selectivity,
        measured_delta_T_K=measured_delta_T_K,
        selectivity_error=sel_err,
        delta_T_error_K=prediction.optimized_delta_T_K - measured_delta_T_K,
        rank_correct=rank,
    )


def load_lab_measurements(path: Path | str) -> list[dict]:
    """Load bench results JSON: list of {phantom, measured_delta_T_K, ...}."""
    data = json.loads(Path(path).read_text())
    return data if isinstance(data, list) else data.get("measurements", [])


# Backward-compatible alias
PHANTOMS = PHANTOM_RECIPES


def materials_from_phantom(label: str) -> Materials:
    return materials_from_recipe(PHANTOM_RECIPES[label])
