"""Evaluate designs over a deposit ore envelope (backlog D3).

Scores min/mean selectivity and coupling across many ore JSON profiles
(e.g. Forster 42-ore manifest, campaign directory).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataclasses import replace

from mw_inv.fdfd import Grid, solve_scene
from mw_inv.fom import evaluate as evaluate_fom
from mw_inv.geometry import CavityParams, build_scene
from mw_inv.industrial_metrics import IndustrialMetrics
from mw_inv.ore_profiles import load_ore_profile, materials_from_ore


@dataclass(frozen=True)
class EnvelopeOreResult:
    label: str
    path: str
    selectivity: float
    coupling_eff: float
    gangue_power_fraction: float
    pec_loss_fraction: float
    error: str | None = None

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "path": self.path,
            "selectivity": self.selectivity,
            "coupling_eff": self.coupling_eff,
            "gangue_power_fraction": self.gangue_power_fraction,
            "pec_loss_fraction": self.pec_loss_fraction,
        }
        if self.error:
            d["error"] = self.error
        return d


@dataclass(frozen=True)
class DepositEnvelopeReport:
    n_ores: int
    n_ok: int
    min_selectivity: float
    mean_selectivity: float
    min_coupling_eff: float
    max_gangue_power_fraction: float
    max_pec_loss_fraction: float
    results: tuple[EnvelopeOreResult, ...]

    def passes(
        self,
        *,
        min_selectivity: float,
        min_coupling_eff: float = 0.0,
        max_gangue_power_fraction: float = 1.0,
        max_pec_loss_fraction: float = 1.0,
    ) -> bool:
        return (
            self.min_selectivity >= min_selectivity
            and self.min_coupling_eff >= min_coupling_eff
            and self.max_gangue_power_fraction <= max_gangue_power_fraction
            and self.max_pec_loss_fraction <= max_pec_loss_fraction
        )

    def to_dict(self) -> dict:
        return {
            "n_ores": self.n_ores,
            "n_ok": self.n_ok,
            "min_selectivity": self.min_selectivity,
            "mean_selectivity": self.mean_selectivity,
            "min_coupling_eff": self.min_coupling_eff,
            "max_gangue_power_fraction": self.max_gangue_power_fraction,
            "max_pec_loss_fraction": self.max_pec_loss_fraction,
            "results": [r.to_dict() for r in self.results],
        }


@dataclass(frozen=True)
class DepositEnvelopeThresholds:
    """Acceptance bands for min-over-ores envelope promotion gate."""

    min_selectivity: float = 0.0
    min_coupling_eff: float = 0.10
    max_gangue_power_fraction: float = 0.85
    max_pec_loss_fraction: float = 0.15
    min_ores_ok: int = 1


@dataclass(frozen=True)
class DepositEnvelopeCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class DepositEnvelopeGateReport:
    passed: bool
    checks: tuple[DepositEnvelopeCheck, ...]
    thresholds: DepositEnvelopeThresholds

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
            "thresholds": {
                "min_selectivity": self.thresholds.min_selectivity,
                "min_coupling_eff": self.thresholds.min_coupling_eff,
                "max_gangue_power_fraction": self.thresholds.max_gangue_power_fraction,
                "max_pec_loss_fraction": self.thresholds.max_pec_loss_fraction,
                "min_ores_ok": self.thresholds.min_ores_ok,
            },
        }


def evaluate_deposit_envelope_gate(
    report: DepositEnvelopeReport,
    thresholds: DepositEnvelopeThresholds | None = None,
) -> DepositEnvelopeGateReport:
    """Pass/fail from min-over-ores envelope metrics (backlog D3 promotion hook)."""
    th = thresholds or DepositEnvelopeThresholds()
    checks: list[DepositEnvelopeCheck] = []

    ok_n = report.n_ok >= th.min_ores_ok
    checks.append(DepositEnvelopeCheck(
        "envelope_ores_evaluated",
        ok_n,
        f"{report.n_ok}/{report.n_ores} ores ok (min {th.min_ores_ok})",
    ))

    ok_sel = report.min_selectivity >= th.min_selectivity
    checks.append(DepositEnvelopeCheck(
        "envelope_min_selectivity",
        ok_sel,
        f"min selectivity={report.min_selectivity:.4f} (floor {th.min_selectivity})",
    ))

    ok_coup = report.min_coupling_eff >= th.min_coupling_eff
    checks.append(DepositEnvelopeCheck(
        "envelope_min_coupling",
        ok_coup,
        f"min coupling_eff={report.min_coupling_eff:.4f} (floor {th.min_coupling_eff})",
    ))

    ok_gangue = report.max_gangue_power_fraction <= th.max_gangue_power_fraction
    checks.append(DepositEnvelopeCheck(
        "envelope_gangue_budget",
        ok_gangue,
        f"max gangue_power_fraction={report.max_gangue_power_fraction:.4f} "
        f"(limit {th.max_gangue_power_fraction})",
    ))

    ok_pec = report.max_pec_loss_fraction <= th.max_pec_loss_fraction
    checks.append(DepositEnvelopeCheck(
        "envelope_pec_loss",
        ok_pec,
        f"max pec_loss_fraction={report.max_pec_loss_fraction:.4f} "
        f"(limit {th.max_pec_loss_fraction})",
    ))

    passed = all(c.passed for c in checks)
    return DepositEnvelopeGateReport(passed=passed, checks=tuple(checks), thresholds=th)


def discover_ore_json_paths(root: Path) -> list[Path]:
    """All ore profiles under *root* (recursive), skipping manifests."""
    paths: list[Path] = []
    for p in sorted(root.rglob("*.json")):
        if p.name.startswith("_"):
            continue
        if p.name.endswith(".example.json"):
            continue
        paths.append(p)
    return paths


def evaluate_deposit_envelope(
    ore_paths: list[Path],
    grid: Grid,
    params: CavityParams | None = None,
    *,
    freq_hz: float = 2.45e9,
    target_T_K: float = 298.15,
    gangue_T_K: float = 298.15,
) -> DepositEnvelopeReport:
    """Min/mean metrics over a list of ore profile JSON files."""
    base_params = params or CavityParams()
    results: list[EnvelopeOreResult] = []

    for path in ore_paths:
        try:
            ore = load_ore_profile(path)
            mats = materials_from_ore(
                ore,
                ore_profile_path=path,
                target_T_K=target_T_K,
                gangue_T_K=gangue_T_K,
                freq_hz=freq_hz,
            )
            p = replace(base_params, freq_hz=freq_hz)
            scene = build_scene(grid, p, mats)
            fom = evaluate_fom(solve_scene(grid, scene), scene)
            ind = IndustrialMetrics.from_fom(fom)
            results.append(EnvelopeOreResult(
                label=ore.label,
                path=str(path.resolve()),
                selectivity=fom.selectivity,
                coupling_eff=fom.coupling_eff,
                gangue_power_fraction=ind.gangue_power_fraction,
                pec_loss_fraction=fom.pec_loss_fraction,
            ))
        except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            results.append(EnvelopeOreResult(
                label=path.stem,
                path=str(path.resolve()),
                selectivity=0.0,
                coupling_eff=0.0,
                gangue_power_fraction=1.0,
                pec_loss_fraction=1.0,
                error=str(exc),
            ))

    ok = [r for r in results if r.error is None]
    if not ok:
        return DepositEnvelopeReport(
            n_ores=len(results),
            n_ok=0,
            min_selectivity=0.0,
            mean_selectivity=0.0,
            min_coupling_eff=0.0,
            max_gangue_power_fraction=1.0,
            max_pec_loss_fraction=1.0,
            results=tuple(results),
        )

    sels = [r.selectivity for r in ok]
    coups = [r.coupling_eff for r in ok]
    gangue = [r.gangue_power_fraction for r in ok]
    pec = [r.pec_loss_fraction for r in ok]
    return DepositEnvelopeReport(
        n_ores=len(results),
        n_ok=len(ok),
        min_selectivity=float(min(sels)),
        mean_selectivity=float(sum(sels) / len(sels)),
        min_coupling_eff=float(min(coups)),
        max_gangue_power_fraction=float(max(gangue)),
        max_pec_loss_fraction=float(max(pec)),
        results=tuple(results),
    )
