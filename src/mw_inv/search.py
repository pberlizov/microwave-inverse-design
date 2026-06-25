"""Geometry search: optimise applicator knobs to maximise selective heating.

This mirrors the photonics project's reusable pattern -- propose geometry, evaluate
with the forward model, keep the best -- here with the forward model cheap enough to
act as its own verifier. We always run a random-search baseline alongside the
optimiser so any reported gain is stated against a control (the same honesty
discipline as the nanophotonics repo's baselines).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from mw_inv.fdfd import Grid, solve
from mw_inv.fom import evaluate
from mw_inv.geometry import CavityParams, FEED_WALLS, Materials, build_scene


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
    result = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r)
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
) -> RobustTrial:
    from mw_inv.ensemble import evaluate_ensemble

    space = get_search_space(legacy=legacy)
    rep = evaluate_ensemble(
        grid, params, materials,
        n_realizations=n_realizations, n_grains=n_grains, seed=seed,
    )
    return RobustTrial(
        params=_knobs_from_params(params, space),
        mean_selectivity=rep.mean_selectivity,
        min_selectivity=rep.min_selectivity,
        std_selectivity=rep.std_selectivity,
        mean_p_total=rep.mean_p_total,
        score=rep.mean_selectivity,
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
# Multi-objective search (step 6): selectivity vs charge coupling (P_total)
# ---------------------------------------------------------------------------

@dataclass
class MultiTrial:
    params: dict
    selectivity: float
    p_total: float
    contrast: float


def optuna_multi_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    base: CavityParams | None = None,
    materials: Materials | None = None,
    *,
    legacy: bool = False,
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
        t = evaluate_params(grid, _params_from_vector(base, vec), materials, legacy=legacy)
        mt = MultiTrial(
            params=t.params,
            selectivity=t.selectivity,
            p_total=t.p_total,
            contrast=t.contrast,
        )
        trial.set_user_attr("contrast", t.contrast)
        trials.append(mt)
        return t.selectivity, t.p_total

    study = optuna.create_study(
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return trials, study


def pareto_best_selectivity(trials: list[MultiTrial]) -> MultiTrial:
    """Best selectivity on the recorded Pareto set."""
    return max(trials, key=lambda t: t.selectivity)


def pareto_best_coupling(trials: list[MultiTrial]) -> MultiTrial:
    """Best charge absorption on the recorded trial set."""
    return max(trials, key=lambda t: t.p_total)


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
