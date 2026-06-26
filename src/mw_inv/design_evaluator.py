"""Unified design evaluation — one forward pass, all FOMs, composable scoring.

Every search script, ore profile, and gate workflow should call
``DesignEvaluator.evaluate`` (or ``evaluate_design``) instead of ad-hoc
``evaluate_params`` / ``evaluate_thermal_params`` / ``evaluate_stress_params``.

Presets map to the legacy single-objective searches; ``composite:*`` presets
blend EM, thermal, and liberation stress for ore-realistic optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from mw_inv.fdfd import Grid, solve_scene
from mw_inv.fom import evaluate as evaluate_fom
from mw_inv.geometry import CavityParams, Materials, build_scene

# Re-export objective name tuples for scripts/tests (search.py aliases these).
EM_OBJECTIVES = ("em_selectivity", "em_contrast", "p_target")
THERMAL_OBJECTIVES = ("delta_T", "heat_selectivity", "em_selectivity")
STRESS_OBJECTIVES = ("stress_selectivity", "mean_interface_stress", "stress_score")

COMPOSITE_PRESETS = ("composite:liberation", "composite:thermal_em")

__all__ = [
    "COMPOSITE_PRESETS",
    "DesignEvaluator",
    "DesignReport",
    "EM_OBJECTIVES",
    "EvaluationConfig",
    "FomWeights",
    "STRESS_OBJECTIVES",
    "THERMAL_OBJECTIVES",
    "evaluate_design",
    "preset_config",
    "resolve_materials",
]


@dataclass(frozen=True)
class FomWeights:
    """Non-negative weights for composite scoring (need not sum to 1)."""

    em_selectivity: float = 0.0
    heat_selectivity: float = 0.0
    delta_T_K: float = 0.0
    stress_score: float = 0.0


LIBERATION_WEIGHTS = FomWeights(
    em_selectivity=0.15,
    heat_selectivity=0.20,
    delta_T_K=0.30,
    stress_score=0.35,
)

THERMAL_EM_WEIGHTS = FomWeights(
    em_selectivity=0.40,
    heat_selectivity=0.30,
    delta_T_K=0.30,
)


@dataclass
class EvaluationConfig:
    """What to compute and how to score a single cavity design."""

    pair_label: str | None = None
    materials: Materials | None = None
    mode: str = "em"  # em | thermal | stress | full
    objective: str = "em_selectivity"
    composite_weights: FomWeights | None = None
    thermal_cfg: Any = None  # ThermalConfig | None — lazy import in evaluate
    legacy: bool = False
    check_arcing: bool = False
    arcing_score_factor: float = 0.5
    # Coupling floor (backlog A0): designs below this coupling_eff are penalised so the
    # search cannot "win" by routing power into structure instead of the charge.
    coupling_floor: float = 0.0
    coupling_score_factor: float = 0.25


@dataclass
class DesignReport:
    """Structured FOM bundle for one (grid, params, materials) evaluation."""

    params: dict
    objective_key: str
    score: float

    em_selectivity: float
    em_contrast: float
    p_target: float
    p_total: float

    # Coupling (backlog A0)
    coupling_eff: float = 1.0
    p_abs_total: float | None = None
    pec_loss_fraction: float | None = None
    coupling_floor_applied: bool = False

    delta_T_K: float | None = None
    heat_selectivity: float | None = None
    thermal_converged: bool | None = None

    stress_selectivity: float | None = None
    mean_interface_stress_Pa: float | None = None
    max_interface_stress_Pa: float | None = None
    grain_penalty: float | None = None
    stress_score: float | None = None

    arcing_risk: bool | None = None
    arcing_penalty_applied: bool = False
    power_density_W_m3: float | None = None
    loss_tangent: float | None = None

    pair_label: str | None = None
    preset: str | None = None
    foms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "params": self.params,
            "objective_key": self.objective_key,
            "score": self.score,
            "em_selectivity": self.em_selectivity,
            "em_contrast": self.em_contrast,
            "p_target": self.p_target,
            "p_total": self.p_total,
            "coupling_eff": self.coupling_eff,
            "foms": dict(self.foms),
        }
        for key in (
            "p_abs_total", "pec_loss_fraction", "coupling_floor_applied",
            "delta_T_K", "heat_selectivity", "thermal_converged",
            "stress_selectivity", "mean_interface_stress_Pa", "max_interface_stress_Pa",
            "grain_penalty", "stress_score",
            "arcing_risk", "arcing_penalty_applied",
            "power_density_W_m3", "loss_tangent",
            "pair_label", "preset",
        ):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d


def resolve_materials(config: EvaluationConfig) -> Materials:
    if config.materials is not None:
        return config.materials
    if config.pair_label is None:
        from mw_inv.materials import DEFAULT_PAIR

        return Materials.from_pair(DEFAULT_PAIR.label)
    return Materials.from_pair(config.pair_label)


def preset_config(
    preset: str,
    *,
    materials: Materials | None = None,
    pair_label: str | None = None,
    legacy: bool = False,
    check_arcing: bool = False,
    thermal_cfg: Any = None,
) -> EvaluationConfig:
    """Named evaluation presets used by CLI scripts."""
    label = pair_label or (materials.pair_label if materials else None)

    if preset == "em":
        return EvaluationConfig(
            pair_label=label,
            materials=materials,
            mode="em",
            objective="em_selectivity",
            legacy=legacy,
            check_arcing=check_arcing,
        )
    if preset.startswith("thermal:"):
        obj = preset.split(":", 1)[1]
        if obj not in THERMAL_OBJECTIVES:
            raise ValueError(f"unknown thermal objective {obj!r}")
        return EvaluationConfig(
            pair_label=label,
            materials=materials,
            mode="thermal",
            objective=obj,
            legacy=legacy,
            thermal_cfg=thermal_cfg,
            check_arcing=check_arcing,
        )
    if preset.startswith("stress:"):
        obj = preset.split(":", 1)[1]
        if obj not in STRESS_OBJECTIVES:
            raise ValueError(f"unknown stress objective {obj!r}")
        return EvaluationConfig(
            pair_label=label,
            materials=materials,
            mode="stress",
            objective=obj,
            legacy=legacy,
            thermal_cfg=thermal_cfg,
            check_arcing=check_arcing,
        )
    if preset == "composite:liberation":
        return EvaluationConfig(
            pair_label=label,
            materials=materials,
            mode="full",
            objective="composite",
            composite_weights=LIBERATION_WEIGHTS,
            legacy=legacy,
            thermal_cfg=thermal_cfg,
            check_arcing=check_arcing,
        )
    if preset == "composite:thermal_em":
        return EvaluationConfig(
            pair_label=label,
            materials=materials,
            mode="full",
            objective="composite",
            composite_weights=THERMAL_EM_WEIGHTS,
            legacy=legacy,
            thermal_cfg=thermal_cfg,
            check_arcing=check_arcing,
        )
    raise ValueError(
        f"unknown preset {preset!r}; use em, thermal:*, stress:*, or {COMPOSITE_PRESETS}"
    )


def _params_dict(params: CavityParams, legacy: bool) -> dict:
    from mw_inv.search import _knobs_from_params, get_search_space

    space = get_search_space(legacy=legacy)
    return _knobs_from_params(params, space)


def _normalize_foms(report: DesignReport) -> dict[str, float]:
    """Map raw metrics to ~[0, 1] for weighted sums."""
    f: dict[str, float] = {
        "em_selectivity": float(report.em_selectivity),
        "em_contrast": float(min(report.em_contrast / 20.0, 1.0)),
    }
    if report.heat_selectivity is not None:
        f["heat_selectivity"] = float(report.heat_selectivity)
    if report.delta_T_K is not None:
        f["delta_T_K"] = float(min(max(report.delta_T_K, 0.0) / 450.0, 1.0))
    if report.stress_score is not None:
        f["stress_score"] = float(min(max(report.stress_score, 0.0) / 1.0e8, 1.0))
    if report.stress_selectivity is not None:
        f["stress_selectivity"] = float(report.stress_selectivity)
    if report.mean_interface_stress_Pa is not None:
        f["mean_interface_stress"] = float(
            min(max(report.mean_interface_stress_Pa, 0.0) / 1.0e8, 1.0)
        )
    return f


def _composite_score(foms: dict[str, float], weights: FomWeights) -> float:
    total_w = 0.0
    score = 0.0
    for name, w in (
        ("em_selectivity", weights.em_selectivity),
        ("heat_selectivity", weights.heat_selectivity),
        ("delta_T_K", weights.delta_T_K),
        ("stress_score", weights.stress_score),
    ):
        if w <= 0.0:
            continue
        if name not in foms:
            continue
        score += w * foms[name]
        total_w += w
    if total_w <= 0.0:
        return foms.get("em_selectivity", 0.0)
    return score / total_w


def _primary_score(report: DesignReport, config: EvaluationConfig) -> tuple[str, float]:
    obj = config.objective
    if obj == "composite":
        w = config.composite_weights or LIBERATION_WEIGHTS
        return "composite", _composite_score(report.foms, w)

    mapping: dict[str, float | None] = {
        "em_selectivity": report.em_selectivity,
        "em_contrast": report.em_contrast,
        "p_target": report.p_target,
        "delta_T": report.delta_T_K,
        "heat_selectivity": report.heat_selectivity,
        "stress_selectivity": report.stress_selectivity,
        "mean_interface_stress": report.mean_interface_stress_Pa,
        "stress_score": report.stress_score,
    }
    if obj not in mapping or mapping[obj] is None:
        raise ValueError(f"objective {obj!r} unavailable in mode={config.mode!r}")
    return obj, float(mapping[obj])


def evaluate_design(
    grid: Grid,
    params: CavityParams,
    config: EvaluationConfig,
    *,
    preset: str | None = None,
) -> DesignReport:
    """Single entry point: EM ± thermal ± stress ± arcing for one design."""
    materials = resolve_materials(config)
    pair_label = config.pair_label or materials.pair_label
    if pair_label is None and config.mode in ("thermal", "stress", "full"):
        raise ValueError("pair_label required for thermal/stress/full evaluation")

    scene = build_scene(grid, params, materials)
    result = solve_scene(grid, scene)
    fom = evaluate_fom(result, scene)

    delta_T: float | None = None
    heat_sel: float | None = None
    th_conv: bool | None = None
    stress_sel: float | None = None
    mean_stress: float | None = None
    max_stress: float | None = None
    grain_pen: float | None = None
    stress_sc: float | None = None

    need_thermal = config.mode in ("thermal", "stress", "full")
    if need_thermal:
        from mw_inv.thermal import ThermalConfig, coupled_steady_state, thermal_props_for_pair

        assert pair_label is not None
        tcfg = config.thermal_cfg or ThermalConfig(
            drive=8.0,
            thermal_props=thermal_props_for_pair(pair_label),
            max_iters=15,
            tol_K=3.0,
        )
        coupled = coupled_steady_state(
            grid, pair_label, config=tcfg, params=params, materials=materials,
        )
        th = coupled.thermal
        delta_T = th.delta_T_K
        heat_sel = th.heat_selectivity
        th_conv = th.converged
        fom = coupled.em_report

        if config.mode in ("stress", "full"):
            from mw_inv.stress import evaluate_stress_from_coupled, grain_size_penalty_factor

            srep = evaluate_stress_from_coupled(coupled)
            stress_sel = srep.stress_selectivity
            mean_stress = srep.mean_interface_stress_Pa
            max_stress = srep.max_interface_stress_Pa
            r = params.inclusion_radius_frac * min(grid.Lx, grid.Ly)
            grain_pen = grain_size_penalty_factor(r)
            stress_sc = mean_stress * grain_pen

    report = DesignReport(
        params=_params_dict(params, config.legacy),
        objective_key=config.objective,
        score=0.0,
        em_selectivity=fom.selectivity,
        em_contrast=fom.contrast,
        p_target=fom.p_target,
        p_total=fom.p_total_charge,
        coupling_eff=fom.coupling_eff,
        p_abs_total=fom.p_abs_total,
        pec_loss_fraction=fom.pec_loss_fraction,
        delta_T_K=delta_T,
        heat_selectivity=heat_sel,
        thermal_converged=th_conv,
        stress_selectivity=stress_sel,
        mean_interface_stress_Pa=mean_stress,
        max_interface_stress_Pa=max_stress,
        grain_penalty=grain_pen,
        stress_score=stress_sc,
        pair_label=pair_label,
        preset=preset,
    )
    report.foms = _normalize_foms(report)

    obj_key, score = _primary_score(report, config)

    if config.check_arcing:
        from mw_inv.ore_profiles import arcing_risk_flag, charge_volume_m3

        vol = charge_volume_m3(params, Lx=grid.Lx, Ly=grid.Ly)
        risk = arcing_risk_flag(fom.p_total_charge, vol, materials)
        report.arcing_risk = bool(risk["arcing_risk"])
        report.power_density_W_m3 = float(risk["power_density_W_m3"])
        report.loss_tangent = float(risk["loss_tangent"])
        if report.arcing_risk:
            score *= config.arcing_score_factor
            report.arcing_penalty_applied = True

    if config.coupling_floor > 0.0 and report.coupling_eff < config.coupling_floor:
        score *= config.coupling_score_factor
        report.coupling_floor_applied = True

    report.objective_key = obj_key
    report.score = score
    return report


class DesignEvaluator:
    """Stateful wrapper: fixed grid + config, many param vectors."""

    def __init__(self, grid: Grid, config: EvaluationConfig, *, preset: str | None = None):
        self.grid = grid
        self.config = config
        self.preset = preset

    @classmethod
    def from_preset(
        cls,
        grid: Grid,
        preset: str,
        *,
        materials: Materials | None = None,
        pair_label: str | None = None,
        **kwargs: Any,
    ) -> DesignEvaluator:
        cfg = preset_config(preset, materials=materials, pair_label=pair_label, **kwargs)
        return cls(grid, cfg, preset=preset)

    def evaluate(self, params: CavityParams) -> DesignReport:
        return evaluate_design(self.grid, params, self.config, preset=self.preset)

    def evaluate_dict(self, params_dict: dict, base: CavityParams | None = None) -> DesignReport:
        from mw_inv.search import params_from_dict

        return self.evaluate(params_from_dict(params_dict, base))


def optuna_design_search(
    grid: Grid,
    config: EvaluationConfig,
    n_trials: int,
    seed: int,
    base: CavityParams | None = None,
    *,
    preset: str | None = None,
) -> list[DesignReport]:
    """TPE search maximising ``config.objective`` via unified evaluator."""
    import optuna

    from mw_inv.search import _params_from_vector, _suggest_vec, get_search_space

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = base or CavityParams()
    space = get_search_space(legacy=config.legacy)
    if config.legacy:
        base = replace(base, feed_wall="")
    ev = DesignEvaluator(grid, config, preset=preset)
    reports: list[DesignReport] = []

    def objective(trial: optuna.Trial) -> float:
        vec = _suggest_vec(trial, space)
        rep = ev.evaluate(_params_from_vector(base, vec))
        trial.set_user_attr("em_selectivity", rep.em_selectivity)
        if rep.delta_T_K is not None:
            trial.set_user_attr("delta_T_K", rep.delta_T_K)
        if rep.stress_score is not None:
            trial.set_user_attr("stress_score", rep.stress_score)
        reports.append(rep)
        return rep.score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    return reports


def best_design(reports: list[DesignReport]) -> DesignReport:
    return max(reports, key=lambda r: r.score)
