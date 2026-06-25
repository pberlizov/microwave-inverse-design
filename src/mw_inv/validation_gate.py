"""Solver validation gate — pass/fail criteria for FDFD ↔ FDTD triangulation.

The gate checks that (1) optimised geometry beats untuned on FDFD for
``pyrite_in_calcite``, and (2) when MEEP/openEMS data exist, rank order is
preserved and relative errors stay within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mw_inv.solver_triangulation import SolverRow, rank_agreement


@dataclass(frozen=True)
class GateThresholds:
    """Tunable acceptance bands for cross-solver agreement."""

    meep_2d_rel_err_max: float = 0.25
    meep_3d_rel_err_max: float = 0.30
    openems_rel_err_max: float = 0.35
    require_rank_match: bool = True
    require_fdfd_improvement: bool = True
    min_fdfd_improvement: float = 0.01  # optimised − untuned selectivity


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class ValidationGateReport:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    rank_agreement: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "rank_agreement": self.rank_agreement,
        }


def _row_by_label(rows: list[SolverRow], label: str) -> SolverRow | None:
    for r in rows:
        if r.label == label:
            return r
    return None


def evaluate_gate(
    rows: list[SolverRow],
    thresholds: GateThresholds | None = None,
) -> ValidationGateReport:
    """Evaluate pass/fail from triangulation rows (untuned, random_best, tpe_best)."""
    th = thresholds or GateThresholds()
    checks: list[GateCheck] = []
    rank = rank_agreement(rows)

    untuned = _row_by_label(rows, "untuned")
    tpe = _row_by_label(rows, "tpe_best")
    rnd = _row_by_label(rows, "random_best")

    if untuned is None or tpe is None:
        checks.append(GateCheck("cases_present", False, "need untuned and tpe_best rows"))
        return ValidationGateReport(passed=False, checks=checks, rank_agreement=rank)

    if th.require_fdfd_improvement:
        delta = tpe.fdfd_selectivity - untuned.fdfd_selectivity
        ok = delta >= th.min_fdfd_improvement
        checks.append(GateCheck(
            "fdfd_optimised_beats_untuned",
            ok,
            f"Δsel={delta:.4f} (tpe={tpe.fdfd_selectivity:.4f}, untuned={untuned.fdfd_selectivity:.4f})",
        ))

    best = max((tpe, rnd), key=lambda r: r.fdfd_selectivity if r else 0.0)
    if best and best.label != "tpe_best":
        checks.append(GateCheck(
            "tpe_is_fdfd_best",
            False,
            f"TPE ({tpe.fdfd_selectivity:.4f}) < {best.label} ({best.fdfd_selectivity:.4f})",
        ))
    else:
        checks.append(GateCheck("tpe_is_fdfd_best", True, "TPE best on FDFD"))

    for solver, max_err, attr in (
        ("meep_2d", th.meep_2d_rel_err_max, "rel_err_meep_2d"),
        ("meep_3d_primitive", th.meep_3d_rel_err_max, "rel_err_meep_3d"),
        ("openems", th.openems_rel_err_max, "rel_err_openems"),
    ):
        errs = [getattr(r, attr) for r in rows if getattr(r, attr) is not None]
        if not errs:
            checks.append(GateCheck(f"{solver}_available", True, f"{solver} skipped — no data"))
            continue
        worst = max(errs)
        ok = worst <= max_err
        checks.append(GateCheck(
            f"{solver}_rel_err",
            ok,
            f"max rel err={worst:.3f} (limit {max_err})",
        ))

    if th.require_rank_match:
        for key in ("meep_2d_selectivity", "meep_3d_primitive_selectivity", "openems_selectivity"):
            match = rank.get(f"{key}_rankings_match_fdfd")
            if match is None:
                continue
            checks.append(GateCheck(
                f"{key}_rank_match",
                bool(match),
                "rank order matches FDFD" if match else "rank order diverges from FDFD",
            ))

    passed = all(c.passed for c in checks)
    return ValidationGateReport(passed=passed, checks=checks, rank_agreement=rank)
