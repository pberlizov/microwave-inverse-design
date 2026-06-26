"""Geometry search: optimise applicator knobs to maximise selective heating.

This mirrors the photonics project's reusable pattern -- propose geometry, evaluate
with the forward model, keep the best -- here with the forward model cheap enough to
act as its own verifier. We always run a random-search baseline alongside the
optimiser so any reported gain is stated against a control (the same honesty
discipline as the nanophotonics repo's baselines).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from mw_inv.fdfd import Grid, solve_scene
from mw_inv.fom import evaluate
from mw_inv.geometry import CavityParams, FEED_WALLS, Materials, build_scene

if TYPE_CHECKING:
    import optuna

    from mw_inv.ore_profiles import OreComposition
    from mw_inv.thermal import ThermalConfig


# Legacy abstract geometry (point feed + slot baffle) — kept for regression comparisons.
LEGACY_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "freq_hz": (2.40e9, 2.50e9),
    "feed_x_frac": (0.15, 0.85),
    "feed_y_frac": (0.04, 0.30),
    "baffle_x_frac": (0.20, 0.80),
    "baffle_len_frac": (0.0, 0.55),
    "baffle_gap_frac": (0.25, 0.85),
}

# Manufacturable applicator knobs (step 4): wall feed, stub, movable plate, bed position.
MANUFACTURABLE_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "freq_hz": (2.40e9, 2.50e9),
    "feed_along_frac": (0.15, 0.85),
    "stub_depth_frac": (0.03, 0.12),
    "stub_width_frac": (0.02, 0.08),
    "plate_cx_frac": (0.15, 0.85),
    "plate_cy_frac": (0.10, 0.55),
    "plate_len_frac": (0.0, 0.50),
    "plate_angle_deg": (30.0, 150.0),
    "charge_cx_frac": (0.35, 0.65),
    "charge_cy_frac": (0.45, 0.75),
}

# Default search uses manufacturable geometry.
SEARCH_SPACE = MANUFACTURABLE_SEARCH_SPACE

# Frequency-robust search: geometry only — score averaged over ISM band (no tuned freq).
FREQ_ROBUST_SEARCH_SPACE: dict[str, tuple[float, float]] = {
    k: v for k, v in MANUFACTURABLE_SEARCH_SPACE.items() if k != "freq_hz"
}

ROBUST_METRICS = ("mean", "min")


def get_search_space(*, legacy: bool = False, freq_robust: bool = False) -> dict[str, tuple[float, float]]:
    if legacy:
        return LEGACY_SEARCH_SPACE
    if freq_robust:
        return FREQ_ROBUST_SEARCH_SPACE
    return MANUFACTURABLE_SEARCH_SPACE


def _knobs_from_params(params: CavityParams, space: dict[str, tuple[float, float]]) -> dict:
    knobs = {k: getattr(params, k) for k in space}
    if space is not LEGACY_SEARCH_SPACE:
        knobs["feed_wall"] = params.feed_wall
    return knobs


def _sample_vec(
    rng: np.random.Generator,
    space: dict[str, tuple[float, float]],
) -> dict:
    vec = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()}
    if space is not LEGACY_SEARCH_SPACE:
        vec["feed_wall"] = str(rng.choice(FEED_WALLS))
    return vec


def _suggest_vec(trial: "optuna.Trial", space: dict[str, tuple[float, float]]) -> dict:
    import optuna  # noqa: F401

    vec = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in space.items()}
    if space is not LEGACY_SEARCH_SPACE:
        vec["feed_wall"] = trial.suggest_categorical("feed_wall", list(FEED_WALLS))
    return vec


def _params_from_vector(base: CavityParams, vec: dict) -> CavityParams:
    if "feed_x_frac" in vec and "feed_wall" not in vec:
        return replace(base, feed_wall="", **vec)
    return replace(base, **vec)


def params_from_dict(vec: dict, base: CavityParams | None = None) -> CavityParams:
    """Build ``CavityParams`` from a search/phantom JSON param dict."""
    return _params_from_vector(base or CavityParams(), vec)


@dataclass
class Trial:
    params: dict
    selectivity: float
    contrast: float
    p_target: float
    p_total: float = 0.0           # total absorbed in charge (target + gangue)
    coupling_eff: float = 1.0


def evaluate_params(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None = None,
    *,
    legacy: bool = False,
) -> Trial:
    from mw_inv.design_evaluator import DesignEvaluator, preset_config

    cfg = preset_config("em", materials=materials, legacy=legacy)
    rep = DesignEvaluator(grid, cfg, preset="em").evaluate(params)
    return Trial(
        params=rep.params,
        selectivity=rep.em_selectivity,
        contrast=rep.em_contrast,
        p_target=rep.p_target,
        p_total=rep.p_total,
        coupling_eff=rep.coupling_eff,
    )


def random_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    base: CavityParams | None = None,
    materials: Materials | None = None,
    *,
    legacy: bool = False,
) -> list[Trial]:
    rng = np.random.default_rng(seed)
    base = base or CavityParams()
    space = get_search_space(legacy=legacy)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[Trial] = []
    for _ in range(n_trials):
        vec = _sample_vec(rng, space)
        trials.append(evaluate_params(grid, _params_from_vector(base, vec), materials, legacy=legacy))
    return trials


def optuna_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    base: CavityParams | None = None,
    materials: Materials | None = None,
    *,
    legacy: bool = False,
) -> list[Trial]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = base or CavityParams()
    space = get_search_space(legacy=legacy)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[Trial] = []

    def objective(trial: "optuna.Trial") -> float:
        vec = _suggest_vec(trial, space)
        t = evaluate_params(grid, _params_from_vector(base, vec), materials, legacy=legacy)
        trial.set_user_attr("contrast", t.contrast)
        trial.set_user_attr("p_target", t.p_target)
        trials.append(t)
        return t.selectivity

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return trials


def best(trials: list[Trial]) -> Trial:
    return max(trials, key=lambda t: t.selectivity)


def trial_to_dict(trial: Trial) -> dict:
    return {
        "selectivity": trial.selectivity,
        "contrast": trial.contrast,
        "p_target": trial.p_target,
        "p_total": trial.p_total,
        "coupling_eff": trial.coupling_eff,
        "params": trial.params,
    }


def top_k_trials(trials: list[Trial], k: int) -> list[Trial]:
    """Return up to *k* unique trials ranked by selectivity (FDFD pre-screen for openEMS)."""
    if k <= 0:
        return []
    import json as _json

    ranked = sorted(trials, key=lambda t: t.selectivity, reverse=True)
    out: list[Trial] = []
    seen: set[str] = set()
    for trial in ranked:
        key = _json.dumps(trial.params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(trial)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# High-dimensional tuner-field search.
#
# The 6 named knobs above are low-dimensional enough that random search ties the
# surrogate optimiser. The reconfigurable dielectric tuner (K lossless cells) is the
# regime where a surrogate earns its keep: K ~ 16-24 continuous knobs, each cheap to
# evaluate but with enough interaction that uniform random sampling wastes budget.
# ---------------------------------------------------------------------------

def _evaluate_field(
    grid: Grid,
    field: list[float],
    base: CavityParams,
    materials: Materials | None,
) -> Trial:
    params = replace(base, tuner_field=tuple(field))
    scene = build_scene(grid, params, materials)
    result = solve_scene(grid, scene)
    report = evaluate(result, scene)
    return Trial(
        params={"tuner_field": list(field)},
        selectivity=report.selectivity,
        contrast=report.contrast,
        p_target=report.p_target,
        p_total=report.p_total_charge,
    )


def random_field_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    k: int = 16,
    base: CavityParams | None = None,
    materials: Materials | None = None,
) -> list[Trial]:
    rng = np.random.default_rng(seed)
    base = base or CavityParams()
    return [
        _evaluate_field(grid, list(rng.uniform(0.0, 1.0, size=k)), base, materials)
        for _ in range(n_trials)
    ]


def optuna_field_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    k: int = 16,
    base: CavityParams | None = None,
    materials: Materials | None = None,
) -> list[Trial]:
    import warnings

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = base or CavityParams()
    trials: list[Trial] = []

    def objective(trial: "optuna.Trial") -> float:
        field = [trial.suggest_float(f"c{i}", 0.0, 1.0) for i in range(k)]
        t = _evaluate_field(grid, field, base, materials)
        trial.set_user_attr("contrast", t.contrast)
        trials.append(t)
        return t.selectivity

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        )
        study.optimize(objective, n_trials=n_trials)
    return trials


# ---------------------------------------------------------------------------
# Thermal-coupled search (optimise ΔT or heat selectivity, not EM alone)
# ---------------------------------------------------------------------------

THERMAL_OBJECTIVES = ("delta_T", "heat_selectivity", "em_selectivity")


@dataclass
class ThermalTrial:
    params: dict
    delta_T_K: float
    heat_selectivity: float
    em_selectivity: float
    em_contrast: float
    converged: bool
    score: float                 # objective value that was maximised


def evaluate_thermal_params(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    thermal_cfg: "ThermalConfig | None" = None,
    objective: str = "delta_T",
    *,
    legacy: bool = False,
) -> ThermalTrial:
    from mw_inv.design_evaluator import DesignEvaluator, preset_config

    preset = f"thermal:{objective}"
    cfg = preset_config(
        preset, pair_label=pair_label, legacy=legacy, thermal_cfg=thermal_cfg,
    )
    rep = DesignEvaluator(grid, cfg, preset=preset).evaluate(params)
    return ThermalTrial(
        params=rep.params,
        delta_T_K=float(rep.delta_T_K or 0.0),
        heat_selectivity=float(rep.heat_selectivity or 0.0),
        em_selectivity=rep.em_selectivity,
        em_contrast=rep.em_contrast,
        converged=bool(rep.thermal_converged),
        score=rep.score,
    )


def random_thermal_search(
    grid: Grid,
    pair_label: str,
    n_trials: int,
    seed: int,
    objective: str = "delta_T",
    base: CavityParams | None = None,
    thermal_cfg: "ThermalConfig | None" = None,
    *,
    legacy: bool = False,
) -> list[ThermalTrial]:
    rng = np.random.default_rng(seed)
    base = base or CavityParams()
    space = get_search_space(legacy=legacy)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[ThermalTrial] = []
    for _ in range(n_trials):
        vec = _sample_vec(rng, space)
        trials.append(
            evaluate_thermal_params(
                grid, _params_from_vector(base, vec), pair_label, thermal_cfg, objective,
                legacy=legacy,
            )
        )
    return trials


def optuna_thermal_search(
    grid: Grid,
    pair_label: str,
    n_trials: int,
    seed: int,
    objective: str = "delta_T",
    base: CavityParams | None = None,
    thermal_cfg: "ThermalConfig | None" = None,
    *,
    legacy: bool = False,
) -> list[ThermalTrial]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = base or CavityParams()
    space = get_search_space(legacy=legacy)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[ThermalTrial] = []

    def obj(trial: "optuna.Trial") -> float:
        vec = _suggest_vec(trial, space)
        t = evaluate_thermal_params(
            grid, _params_from_vector(base, vec), pair_label, thermal_cfg, objective,
            legacy=legacy,
        )
        trial.set_user_attr("delta_T_K", t.delta_T_K)
        trial.set_user_attr("heat_selectivity", t.heat_selectivity)
        trial.set_user_attr("em_selectivity", t.em_selectivity)
        trials.append(t)
        return t.score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(obj, n_trials=n_trials)
    return trials


def best_thermal(trials: list[ThermalTrial]) -> ThermalTrial:
    return max(trials, key=lambda t: t.score)


# ---------------------------------------------------------------------------
# Ensemble robust search (step 5): optimise mean selectivity over grain layouts
# ---------------------------------------------------------------------------

@dataclass
class RobustTrial:
    params: dict
    mean_selectivity: float
    min_selectivity: float
    std_selectivity: float
    mean_p_total: float
    score: float


def evaluate_robust_params(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None = None,
    *,
    n_realizations: int = 6,
    n_grains: int = 5,
    seed: int = 0,
    legacy: bool = False,
    ore: "OreComposition | None" = None,
) -> RobustTrial:
    from mw_inv.ensemble import evaluate_ensemble

    space = get_search_space(legacy=legacy)
    rep = evaluate_ensemble(
        grid, params, materials,
        n_realizations=n_realizations, n_grains=n_grains, seed=seed, ore=ore,
    )
    return RobustTrial(
        params=_knobs_from_params(params, space),
        mean_selectivity=rep.mean_selectivity,
        min_selectivity=rep.min_selectivity,
        std_selectivity=rep.std_selectivity,
        mean_p_total=rep.mean_p_total,
        score=rep.mean_selectivity,
    )


def evaluate_material_robust_params(
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
    legacy: bool = False,
) -> RobustTrial:
    from mw_inv.ensemble import evaluate_material_robust

    space = get_search_space(legacy=legacy)
    rep = evaluate_material_robust(
        grid,
        params,
        ore,
        ore_profile_path=ore_profile_path,
        target_T_K=target_T_K,
        gangue_T_K=gangue_T_K,
        freq_hz=freq_hz,
        n_scenarios=n_scenarios,
        seed=seed,
    )
    return RobustTrial(
        params=_knobs_from_params(params, space),
        mean_selectivity=rep.mean_selectivity,
        min_selectivity=rep.min_selectivity,
        std_selectivity=rep.std_selectivity,
        mean_p_total=0.0,
        score=rep.min_selectivity,
    )


def optuna_robust_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    materials: Materials | None = None,
    *,
    n_realizations: int = 6,
    n_grains: int = 5,
    legacy: bool = False,
) -> list[RobustTrial]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = CavityParams()
    space = get_search_space(legacy=legacy)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[RobustTrial] = []

    def objective(trial: "optuna.Trial") -> float:
        vec = _suggest_vec(trial, space)
        t = evaluate_robust_params(
            grid, _params_from_vector(base, vec), materials,
            n_realizations=n_realizations, n_grains=n_grains, seed=seed + trial.number,
            legacy=legacy,
        )
        trial.set_user_attr("min_selectivity", t.min_selectivity)
        trial.set_user_attr("std_selectivity", t.std_selectivity)
        trials.append(t)
        return t.score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return trials


def best_robust(trials: list[RobustTrial]) -> RobustTrial:
    return max(trials, key=lambda t: t.score)


# ---------------------------------------------------------------------------
# Multi-objective search (C0): selectivity vs coupling_eff (+ optional arcing filter)
# ---------------------------------------------------------------------------

DEFAULT_MAX_HOTSPOT_DELTA_T_K = 475.0  # T_amb + 475 K ≈ 773 K runaway proxy threshold


@dataclass
class MultiTrial:
    params: dict
    selectivity: float
    coupling_eff: float
    p_total: float
    contrast: float
    arcing_risk: bool = False
    hotspot_delta_T_K: float | None = None
    hotspot_violation: bool = False


def _hotspot_delta_T_K(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None,
    pair_label: str | None,
) -> float | None:
    """Peak target-phase rise above ambient at coupled steady state (runaway proxy)."""
    if not pair_label:
        return None
    from mw_inv.thermal import ThermalConfig, coupled_steady_state, thermal_props_for_pair

    cfg = ThermalConfig(
        max_iters=12,
        tol_K=4.0,
        thermal_props=thermal_props_for_pair(pair_label),
    )
    coupled = coupled_steady_state(
        grid, pair_label, config=cfg, params=params, materials=materials,
    )
    return float(coupled.thermal.T_max_target_K - cfg.T_amb_K)


def _evaluate_multi_trial(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None,
    *,
    legacy: bool,
    check_arcing: bool,
    check_hotspot: bool = False,
    max_hotspot_delta_T_K: float = DEFAULT_MAX_HOTSPOT_DELTA_T_K,
) -> MultiTrial:
    from mw_inv.design_evaluator import DesignEvaluator, preset_config

    cfg = preset_config("em", materials=materials, legacy=legacy, check_arcing=check_arcing)
    rep = DesignEvaluator(grid, cfg, preset="em").evaluate(params)
    pair_label = materials.pair_label if materials else cfg.pair_label
    hotspot_dt: float | None = None
    hotspot_bad = False
    if check_hotspot:
        hotspot_dt = _hotspot_delta_T_K(grid, params, materials, pair_label)
        if hotspot_dt is not None:
            hotspot_bad = hotspot_dt > max_hotspot_delta_T_K
    return MultiTrial(
        params=rep.params,
        selectivity=rep.em_selectivity,
        coupling_eff=rep.coupling_eff,
        p_total=rep.p_total,
        contrast=rep.em_contrast,
        arcing_risk=bool(rep.arcing_risk),
        hotspot_delta_T_K=hotspot_dt,
        hotspot_violation=hotspot_bad,
    )


def optuna_multi_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    base: CavityParams | None = None,
    materials: Materials | None = None,
    *,
    legacy: bool = False,
    check_arcing: bool = False,
    check_hotspot: bool = False,
    max_hotspot_delta_T_K: float = DEFAULT_MAX_HOTSPOT_DELTA_T_K,
) -> tuple[list[MultiTrial], "optuna.Study"]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = base or CavityParams()
    space = get_search_space(legacy=legacy)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[MultiTrial] = []

    def objective(trial: "optuna.Trial") -> tuple[float, float]:
        vec = _suggest_vec(trial, space)
        mt = _evaluate_multi_trial(
            grid,
            _params_from_vector(base, vec),
            materials,
            legacy=legacy,
            check_arcing=check_arcing,
            check_hotspot=check_hotspot,
            max_hotspot_delta_T_K=max_hotspot_delta_T_K,
        )
        trial.set_user_attr("contrast", mt.contrast)
        trial.set_user_attr("p_total", mt.p_total)
        trial.set_user_attr("arcing_risk", mt.arcing_risk)
        if mt.hotspot_delta_T_K is not None:
            trial.set_user_attr("hotspot_delta_T_K", mt.hotspot_delta_T_K)
        trial.set_user_attr("hotspot_violation", mt.hotspot_violation)
        trials.append(mt)
        unsafe = (check_arcing and mt.arcing_risk) or (check_hotspot and mt.hotspot_violation)
        coupling = mt.coupling_eff if not unsafe else 0.0
        return mt.selectivity, coupling

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return trials, study


def pareto_front_trials(trials: list[MultiTrial], study: "optuna.Study") -> list[MultiTrial]:
    """Trials on the recorded Pareto front (falls back to all trials if empty)."""
    front = [trials[t.number] for t in study.best_trials if t.number < len(trials)]
    return front or list(trials)


def pareto_best_selectivity(trials: list[MultiTrial]) -> MultiTrial:
    """Best selectivity on the recorded trial set."""
    return max(trials, key=lambda t: t.selectivity)


def pareto_best_coupling(trials: list[MultiTrial]) -> MultiTrial:
    """Best coupling efficiency on the recorded trial set."""
    return max(trials, key=lambda t: t.coupling_eff)


def pareto_recommend(
    trials: list[MultiTrial],
    study: "optuna.Study",
    *,
    weight_selectivity: float = 0.6,
    weight_coupling: float = 0.4,
    exclude_arcing: bool = True,
    exclude_hotspot: bool = True,
) -> MultiTrial:
    """Pick a balanced design from the Pareto front using weighted objectives."""
    front = pareto_front_trials(trials, study)
    if exclude_arcing:
        safe = [t for t in front if not t.arcing_risk]
        if safe:
            front = safe
    if exclude_hotspot:
        cool = [t for t in front if not t.hotspot_violation]
        if cool:
            front = cool
    total_w = weight_selectivity + weight_coupling
    ws = weight_selectivity / total_w
    wc = weight_coupling / total_w
    return max(front, key=lambda t: ws * t.selectivity + wc * t.coupling_eff)


def multi_trial_to_dict(trial: MultiTrial) -> dict:
    out = {
        "selectivity": trial.selectivity,
        "coupling_eff": trial.coupling_eff,
        "p_total": trial.p_total,
        "contrast": trial.contrast,
        "arcing_risk": trial.arcing_risk,
        "params": trial.params,
    }
    if trial.hotspot_delta_T_K is not None:
        out["hotspot_delta_T_K"] = trial.hotspot_delta_T_K
        out["hotspot_violation"] = trial.hotspot_violation
    return out


def top_k_multi_trials(trials: list[MultiTrial], k: int) -> list[MultiTrial]:
    """Rank multi-objective trials by selectivity for openEMS pre-screen export."""
    if k <= 0:
        return []
    import json as _json

    ranked = sorted(trials, key=lambda t: t.selectivity, reverse=True)
    out: list[MultiTrial] = []
    seen: set[str] = set()
    for trial in ranked:
        key = _json.dumps(trial.params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        out.append(trial)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------------------
# Frequency-robust search: stable selectivity across ISM band (step 7-lite)
# ---------------------------------------------------------------------------

@dataclass
class FreqRobustTrial:
    params: dict
    mean_selectivity: float
    min_selectivity: float
    std_selectivity: float
    score: float


def evaluate_freq_robust_params(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None = None,
    *,
    pair_label: str | None = None,
    metric: str = "min",
    n_freqs: int = 5,
    legacy: bool = False,
    freq_robust: bool = True,
) -> FreqRobustTrial:
    from mw_inv.ensemble import evaluate_frequency_robust

    space = get_search_space(legacy=legacy, freq_robust=freq_robust)
    rep = evaluate_frequency_robust(
        grid, params, materials, pair_label=pair_label, n_freqs=n_freqs,
    )
    return FreqRobustTrial(
        params=_knobs_from_params(params, space),
        mean_selectivity=rep.mean_selectivity,
        min_selectivity=rep.min_selectivity,
        std_selectivity=rep.std_selectivity,
        score=rep.score(metric),
    )


def optuna_freq_robust_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    materials: Materials | None = None,
    *,
    pair_label: str | None = None,
    metric: str = "min",
    n_freqs: int = 5,
    legacy: bool = False,
) -> list[FreqRobustTrial]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = CavityParams()
    space = get_search_space(freq_robust=True, legacy=legacy)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[FreqRobustTrial] = []

    def objective(trial: "optuna.Trial") -> float:
        vec = _suggest_vec(trial, space)
        t = evaluate_freq_robust_params(
            grid, _params_from_vector(base, vec), materials,
            pair_label=pair_label, metric=metric, n_freqs=n_freqs,
            legacy=legacy,
        )
        trial.set_user_attr("mean_selectivity", t.mean_selectivity)
        trials.append(t)
        return t.score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return trials


def best_freq_robust(trials: list[FreqRobustTrial]) -> FreqRobustTrial:
    return max(trials, key=lambda t: t.score)


# ---------------------------------------------------------------------------
# Thermal ensemble + optional frequency-robust coupled search
# ---------------------------------------------------------------------------

@dataclass
class ThermalEnsembleTrial:
    params: dict
    mean_delta_T_K: float
    min_delta_T_K: float
    std_delta_T_K: float
    mean_heat_selectivity: float
    score: float
    freq_robust: bool


def evaluate_thermal_ensemble_params(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    *,
    objective: str = "delta_T",
    metric: str = "mean",
    n_realizations: int = 4,
    n_grains: int = 5,
    seed: int = 0,
    thermal_cfg=None,
    freq_robust: bool = False,
    n_freqs: int = 5,
    legacy: bool = False,
) -> ThermalEnsembleTrial:
    from mw_inv.ensemble import evaluate_thermal_ensemble

    space = get_search_space(legacy=legacy, freq_robust=freq_robust)
    rep = evaluate_thermal_ensemble(
        grid, params, pair_label,
        n_realizations=n_realizations, n_grains=n_grains, seed=seed,
        thermal_cfg=thermal_cfg, freq_robust=freq_robust, n_freqs=n_freqs,
    )
    return ThermalEnsembleTrial(
        params=_knobs_from_params(params, space),
        mean_delta_T_K=rep.mean_delta_T_K,
        min_delta_T_K=rep.min_delta_T_K,
        std_delta_T_K=rep.std_delta_T_K,
        mean_heat_selectivity=rep.mean_heat_selectivity,
        score=rep.score(objective, metric),
        freq_robust=freq_robust,
    )


def optuna_thermal_ensemble_search(
    grid: Grid,
    pair_label: str,
    n_trials: int,
    seed: int,
    *,
    objective: str = "delta_T",
    metric: str = "mean",
    n_realizations: int = 4,
    n_grains: int = 5,
    thermal_cfg=None,
    freq_robust: bool = False,
    n_freqs: int = 5,
    legacy: bool = False,
) -> list[ThermalEnsembleTrial]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = CavityParams()
    space = get_search_space(legacy=legacy, freq_robust=freq_robust)
    if legacy:
        base = replace(base, feed_wall="")
    trials: list[ThermalEnsembleTrial] = []

    def obj(trial: "optuna.Trial") -> float:
        vec = _suggest_vec(trial, space)
        t = evaluate_thermal_ensemble_params(
            grid, _params_from_vector(base, vec), pair_label,
            objective=objective, metric=metric,
            n_realizations=n_realizations, n_grains=n_grains,
            seed=seed + trial.number, thermal_cfg=thermal_cfg,
            freq_robust=freq_robust, n_freqs=n_freqs, legacy=legacy,
        )
        trial.set_user_attr("min_delta_T_K", t.min_delta_T_K)
        trials.append(t)
        return t.score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(obj, n_trials=n_trials)
    return trials


def best_thermal_ensemble(trials: list[ThermalEnsembleTrial]) -> ThermalEnsembleTrial:
    return max(trials, key=lambda t: t.score)


# --- Liberation stress objectives (step 4) ---

STRESS_OBJECTIVES = ("stress_selectivity", "mean_interface_stress", "stress_score")


@dataclass
class StressTrial:
    params: dict
    stress_selectivity: float
    mean_interface_stress_Pa: float
    em_selectivity: float
    delta_T_K: float
    grain_penalty: float
    score: float
    converged: bool


def evaluate_stress_params(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    objective: str = "stress_score",
    *,
    thermal_cfg: "ThermalConfig | None" = None,
    legacy: bool = False,
) -> StressTrial:
    from mw_inv.design_evaluator import DesignEvaluator, preset_config

    preset = f"stress:{objective}"
    cfg = preset_config(
        preset, pair_label=pair_label, legacy=legacy, thermal_cfg=thermal_cfg,
    )
    rep = DesignEvaluator(grid, cfg, preset=preset).evaluate(params)
    return StressTrial(
        params=rep.params,
        stress_selectivity=float(rep.stress_selectivity or 0.0),
        mean_interface_stress_Pa=float(rep.mean_interface_stress_Pa or 0.0),
        em_selectivity=rep.em_selectivity,
        delta_T_K=float(rep.delta_T_K or 0.0),
        grain_penalty=float(rep.grain_penalty or 1.0),
        score=rep.score,
        converged=bool(rep.thermal_converged),
    )


def optuna_stress_search(
    grid: Grid,
    pair_label: str,
    n_trials: int,
    seed: int,
    objective: str = "stress_score",
    thermal_cfg: "ThermalConfig | None" = None,
    *,
    legacy: bool = False,
) -> list[StressTrial]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = CavityParams()
    space = get_search_space(legacy=legacy)
    trials: list[StressTrial] = []

    def obj(trial: "optuna.Trial") -> float:
        vec = _suggest_vec(trial, space)
        t = evaluate_stress_params(
            grid, _params_from_vector(base, vec), pair_label, objective,
            thermal_cfg=thermal_cfg, legacy=legacy,
        )
        trials.append(t)
        return t.score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(obj, n_trials=n_trials)
    return trials


def best_stress(trials: list[StressTrial]) -> StressTrial:
    return max(trials, key=lambda t: t.score)


def evaluate_with_ore_constraints(
    grid: Grid,
    params: CavityParams,
    materials: Materials,
    pair_label: str,
    ore_hmap_wt: float | None = None,
) -> Trial:
    """EM trial with arcing-risk penalty for massive / high-HMAP ores."""
    from mw_inv.design_evaluator import DesignEvaluator, preset_config

    cfg = preset_config(
        "em",
        materials=materials,
        pair_label=pair_label,
        check_arcing=True,
    )
    rep = DesignEvaluator(grid, cfg, preset="em").evaluate(params)
    return Trial(
        params=rep.params,
        selectivity=rep.em_selectivity if not rep.arcing_penalty_applied else rep.score,
        contrast=rep.em_contrast,
        p_target=rep.p_target,
        p_total=rep.p_total,
    )
