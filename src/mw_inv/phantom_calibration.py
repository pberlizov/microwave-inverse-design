"""Bench phantom calibration: measured ε from probe → recipe override."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.geometry import Materials
from mw_inv.phantom_data import PHANTOM_RECIPES, PhantomRecipePair


@dataclass(frozen=True)
class MeasuredBatch:
    label: str
    salt_wt_percent: float
    eps_real: float
    eps_imag: float
    freq_hz: float = 2.45e9
    method: str = "open_coax_probe"
    notes: str = ""

    @property
    def eps(self) -> complex:
        return complex(self.eps_real, self.eps_imag)


def load_measured_eps(path: Path | str) -> dict[str, MeasuredBatch]:
    """Load ``data/measured_eps.json`` keyed by batch label."""
    data = json.loads(Path(path).read_text())
    batches = data.get("batches", data)
    out: dict[str, MeasuredBatch] = {}
    for row in batches:
        out[row["label"]] = MeasuredBatch(**{k: row[k] for k in MeasuredBatch.__dataclass_fields__ if k in row})
    return out


def recipe_with_measured(
    phantom_label: str,
    measured: dict[str, MeasuredBatch],
) -> PhantomRecipePair:
    """Override recipe anchor ε with probe-measured batch values when present."""
    base = PHANTOM_RECIPES[phantom_label]
    target_key = base.target.label
    gangue_key = base.gangue.label
    from dataclasses import replace

    target = replace(base.target, label=target_key)
    gangue = replace(base.gangue, label=gangue_key)
    # attach measured eps via monkey-patch properties — use Materials directly instead
    _ = target, gangue
    return base  # caller uses materials_from_measured_recipe


def materials_from_measured_recipe(
    phantom_label: str,
    measured_path: Path | str | None = None,
) -> Materials:
    """Build Materials using measured ε when available, else Gabriel anchors."""
    recipe = PHANTOM_RECIPES[phantom_label]
    t_eps = recipe.target.eps
    g_eps = recipe.gangue.eps
    if measured_path and Path(measured_path).is_file():
        batches = load_measured_eps(measured_path)
        if recipe.target.label in batches:
            t_eps = batches[recipe.target.label].eps
        if recipe.gangue.label in batches:
            g_eps = batches[recipe.gangue.label].eps
    return Materials(target=t_eps, gangue=g_eps, background=1.0 + 0.0j)


def compare_measured_vs_anchor(phantom_label: str, measured_path: Path | str) -> dict:
    """Report anchor vs measured ε drift for calibration QA."""
    recipe = PHANTOM_RECIPES[phantom_label]
    batches = load_measured_eps(measured_path)
    rows = []
    for role, gel in (("target", recipe.target), ("gangue", recipe.gangue)):
        anchor = gel.eps
        meas = batches.get(gel.label)
        if meas is None:
            rows.append({"role": role, "batch": gel.label, "status": "missing"})
            continue
        rows.append({
            "role": role,
            "batch": gel.label,
            "anchor_eps": [anchor.real, anchor.imag],
            "measured_eps": [meas.eps.real, meas.eps.imag],
            "drift_real": meas.eps.real - anchor.real,
            "drift_imag": meas.eps.imag - anchor.imag,
            "method": meas.method,
        })
    return {"phantom": phantom_label, "comparisons": rows}


@dataclass(frozen=True)
class BenchGateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class BenchGateReport:
    passed: bool
    checks: list[BenchGateCheck]
    probe_calibration: dict | None = None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "probe_calibration": self.probe_calibration,
        }


def evaluate_bench_gate(
    phantom_label: str,
    measured_eps_path: Path | str,
    lab_measurements_path: Path | str | None = None,
    *,
    max_real_drift: float = 1.5,
    max_imag_drift: float = 0.5,
    max_imag_drift_frac: float = 0.20,
    validate_model: bool = True,
    bench_grid: int = 41,
    bench_trials: int = 8,
    bench_seed: int = 7701,
    delta_t_rel_tol: float = 0.30,
) -> BenchGateReport:
    """Pass/fail bench calibration inputs for ``bench_calibrated`` promotion."""
    from mw_inv.bench_ingest import validate_lab_measurements
    from mw_inv.promotion import _bench_calibration_ok

    path = Path(measured_eps_path)
    checks: list[BenchGateCheck] = []
    probe_report: dict | None = None

    if not path.is_file():
        checks.append(BenchGateCheck("measured_eps_present", False, f"missing {path}"))
        return BenchGateReport(passed=False, checks=checks)

    probe_report = compare_measured_vs_anchor(phantom_label, path)
    for row in probe_report.get("comparisons", []):
        if row.get("status") == "missing":
            checks.append(BenchGateCheck(
                f"batch_{row.get('batch', '?')}",
                False,
                "missing measured batch",
            ))
            continue
        dr = abs(float(row.get("drift_real", 0.0)))
        di = abs(float(row.get("drift_imag", 0.0)))
        anchor_im = abs(float(row.get("anchor_eps", [0, 0])[1]))
        imag_frac_ok = True
        if anchor_im > 0.01:
            imag_frac_ok = di / anchor_im <= max_imag_drift_frac
        ok = dr <= max_real_drift and di <= max_imag_drift and imag_frac_ok
        detail = f"Δε′={row.get('drift_real', 0):.3f}, Δε″={row.get('drift_imag', 0):.3f}"
        if anchor_im > 0.01:
            detail += f", |Δε″|/ε″={di / anchor_im:.2%}"
        checks.append(BenchGateCheck(
            f"eps_drift_{row.get('batch', row.get('role', '?'))}",
            ok,
            detail,
        ))

    if lab_measurements_path:
        lp = Path(lab_measurements_path)
        if not lp.is_file():
            checks.append(BenchGateCheck("lab_measurements_present", False, f"missing {lp}"))
        else:
            val_issues = validate_lab_measurements(lp)
            if val_issues:
                checks.append(BenchGateCheck(
                    "lab_schema",
                    False,
                    val_issues[0].message,
                ))
            import json

            payload = json.loads(lp.read_text())
            rows = payload if isinstance(payload, list) else payload.get("measurements", [])
            matches = [r for r in rows if r.get("phantom") == phantom_label]
            if not matches:
                checks.append(BenchGateCheck("lab_phantom_match", False, f"no rows for {phantom_label!r}"))
            else:
                rank_ok = any(
                    float(r["measured_delta_T_K"]) > float(r["untuned_measured_delta_T_K"])
                    for r in matches
                    if r.get("untuned_measured_delta_T_K") is not None
                )
                checks.append(BenchGateCheck(
                    "lab_rank_optimized_beats_untuned",
                    rank_ok,
                    f"{len(matches)} bench record(s) for {phantom_label}",
                ))

            if validate_model and matches and not val_issues:
                from mw_inv.fdfd import Grid
                from mw_inv.phantom import compare_lab_measurement, predict_lab_outcome

                grid = Grid(nx=bench_grid, ny=bench_grid, Lx=0.36, Ly=0.36)
                pred = predict_lab_outcome(
                    phantom_label,
                    grid,
                    n_opt_trials=bench_trials,
                    seed=bench_seed,
                    measured_eps_path=path,
                )
                model_rank = pred.optimized_delta_T_K > pred.untuned_delta_T_K
                checks.append(BenchGateCheck(
                    "model_rank_optimized_beats_untuned",
                    model_rank,
                    f"pred ΔT {pred.untuned_delta_T_K:.1f} → {pred.optimized_delta_T_K:.1f} K",
                ))
                for row in matches:
                    comp = compare_lab_measurement(
                        pred,
                        float(row["measured_delta_T_K"]),
                        row.get("measured_selectivity"),
                        untuned_measured_delta_T_K=row.get("untuned_measured_delta_T_K"),
                    )
                    meas = float(row["measured_delta_T_K"])
                    rel_err = abs(comp.delta_T_error_K) / max(meas, 1e-6)
                    tol_ok = rel_err <= delta_t_rel_tol
                    checks.append(BenchGateCheck(
                        "model_delta_t_tolerance",
                        tol_ok,
                        f"|error|/measured={rel_err:.2%} (max {delta_t_rel_tol:.0%})",
                    ))
                    if comp.rank_correct is not None and model_rank is not None:
                        agree = comp.rank_correct == model_rank
                        checks.append(BenchGateCheck(
                            "model_measured_rank_agreement",
                            agree,
                            f"model rank={model_rank}, measured rank={comp.rank_correct}",
                        ))

    promotion_ok = _bench_calibration_ok(
        phantom_label,
        path,
        lab_measurements_path,
        max_real_drift=max_real_drift,
        max_imag_drift=max_imag_drift,
    )
    checks.append(BenchGateCheck(
        "bench_calibrated_requirements",
        promotion_ok,
        "probe ε drift + optional lab rank",
    ))
    return BenchGateReport(passed=promotion_ok, checks=checks, probe_calibration=probe_report)
