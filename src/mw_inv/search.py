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
from mw_inv.geometry import CavityParams, Materials, build_scene


# Search space: (name, low, high). Frequency is allowed a modest band around 2.45 GHz.
SEARCH_SPACE: dict[str, tuple[float, float]] = {
    "freq_hz": (2.40e9, 2.50e9),
    "feed_x_frac": (0.15, 0.85),
    "feed_y_frac": (0.04, 0.30),
    "baffle_x_frac": (0.20, 0.80),
    "baffle_len_frac": (0.0, 0.55),
    "baffle_gap_frac": (0.25, 0.85),
}


@dataclass
class Trial:
    params: dict
    selectivity: float
    contrast: float
    p_target: float


def _params_from_vector(base: CavityParams, vec: dict) -> CavityParams:
    return replace(base, **vec)


def evaluate_params(
    grid: Grid,
    params: CavityParams,
    materials: Materials | None = None,
) -> Trial:
    scene = build_scene(grid, params, materials)
    result = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy)
    report = evaluate(result, scene)
    knobs = {k: getattr(params, k) for k in SEARCH_SPACE}
    return Trial(
        params=knobs,
        selectivity=report.selectivity,
        contrast=report.contrast,
        p_target=report.p_target,
    )


def random_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    base: CavityParams | None = None,
    materials: Materials | None = None,
) -> list[Trial]:
    rng = np.random.default_rng(seed)
    base = base or CavityParams()
    trials: list[Trial] = []
    for _ in range(n_trials):
        vec = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in SEARCH_SPACE.items()}
        trials.append(evaluate_params(grid, _params_from_vector(base, vec), materials))
    return trials


def optuna_search(
    grid: Grid,
    n_trials: int,
    seed: int,
    base: CavityParams | None = None,
    materials: Materials | None = None,
) -> list[Trial]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = base or CavityParams()
    trials: list[Trial] = []

    def objective(trial: "optuna.Trial") -> float:
        vec = {
            k: trial.suggest_float(k, lo, hi)
            for k, (lo, hi) in SEARCH_SPACE.items()
        }
        t = evaluate_params(grid, _params_from_vector(base, vec), materials)
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
