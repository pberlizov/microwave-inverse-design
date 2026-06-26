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
    openems_s11_max: float = 0.92
    openems_coupling_floor: float = 0.08
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
    openems_diagnosis: str | None = None

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
            "rank_agreement": self.rank_agreement,
            "openems_diagnosis": self.openems_diagnosis,
        }


def _row_by_label(rows: list[SolverRow], label: str) -> SolverRow | None:
    for r in rows:
        if r.label == label:
            return r
    return None


def _openems_port_rows(rows: list[SolverRow]) -> list[SolverRow]:
    return [r for r in rows if r.openems_s11_mag is not None or r.openems_coupling_eff is not None]


def diagnose_openems_failure(rows: list[SolverRow], rank: dict) -> str | None:
    """Explain openEMS gate failure: ranking mismatch vs coupling collapse."""
    if rank.get("openems_selectivity_rankings_match_fdfd") is not False:
        return None
    optimised = _row_by_label(rows, "tpe_best") or _row_by_label(rows, "tpe_k1")
    untuned = _row_by_label(rows, "untuned")
    if optimised is None or untuned is None:
        return "openems_rank_mismatch"
    opt_coupling = optimised.openems_coupling_eff
    untuned_coupling = untuned.openems_coupling_eff
    if opt_coupling is not None and untuned_coupling is not None:
        if opt_coupling < 0.5 * untuned_coupling:
            return "coupling_collapse_on_optimised"
    if opt_coupling is not None and opt_coupling < 0.08:
        return "coupling_collapse_on_optimised"
    return "ranking_mismatch_acceptable_coupling"


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

    port_rows = _openems_port_rows(rows)
    if port_rows:
        s11_vals = [r.openems_s11_mag for r in port_rows if r.openems_s11_mag is not None]
        if s11_vals:
            worst_s11 = max(s11_vals)
            ok_s11 = worst_s11 <= th.openems_s11_max
            checks.append(GateCheck(
                "openems_s11",
                ok_s11,
                f"max |S11|={worst_s11:.3f} (limit {th.openems_s11_max})",
            ))
        coupling_vals = [r.openems_coupling_eff for r in port_rows if r.openems_coupling_eff is not None]
        if coupling_vals:
            worst_coupling = min(coupling_vals)
            ok_coupling = worst_coupling >= th.openems_coupling_floor
            checks.append(GateCheck(
                "openems_coupling_floor",
                ok_coupling,
                f"min coupling_eff={worst_coupling:.3f} (floor {th.openems_coupling_floor})",
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
    diagnosis = diagnose_openems_failure(rows, rank) if not passed else None
    return ValidationGateReport(passed=passed, checks=checks, rank_agreement=rank, openems_diagnosis=diagnosis)
