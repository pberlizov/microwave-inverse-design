"""M4 pilot-ready gate — safety, repeatability, and throughput checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MIN_COUPLING_EFF = 0.25


@dataclass(frozen=True)
class PilotCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class PilotGateReport:
    passed: bool
    checks: tuple[PilotCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


def evaluate_pilot_gate(
    evaluation: dict[str, Any],
    search_summary: dict[str, Any] | None = None,
    *,
    min_coupling_eff: float = DEFAULT_MIN_COUPLING_EFF,
    require_robust_gate: bool = True,
    require_safety_flags: bool = True,
) -> PilotGateReport:
    """Pass when bench-calibrated design is repeatable, coupled, and safety-screened."""
    checks: list[PilotCheck] = []

    robust = evaluation.get("robust_gate") or {}
    if require_robust_gate:
        ok = bool(robust.get("passed"))
        checks.append(PilotCheck(
            "robust_repeatability",
            ok,
            str(robust.get("detail", "robust_gate missing — run pipeline with --robust")),
        ))
    else:
        checks.append(PilotCheck("robust_repeatability", True, "not required"))

    tpe = evaluation.get("tpe_best") or {}
    coupling = tpe.get("coupling_eff")
    if coupling is None:
        checks.append(PilotCheck("throughput_coupling", False, "tpe_best.coupling_eff missing"))
    else:
        c = float(coupling)
        checks.append(PilotCheck(
            "throughput_coupling",
            c >= min_coupling_eff,
            f"coupling_eff={c:.4f} (min {min_coupling_eff})",
        ))

    summary = search_summary or {}
    multi = summary.get("multi_search") or {}
    mode = summary.get("search_mode")
    if require_safety_flags and mode == "multi_objective":
        if not multi.get("check_arcing"):
            checks.append(PilotCheck(
                "safety_arcing_screen",
                False,
                "multi_objective run without --check-arcing",
            ))
        else:
            rec = multi.get("recommended") or {}
            arcing = bool(rec.get("arcing_risk"))
            checks.append(PilotCheck(
                "safety_arcing_screen",
                not arcing,
                "recommended arcing_risk" if arcing else "recommended passes arcing screen",
            ))
        if multi.get("check_hotspot"):
            rec = multi.get("recommended") or {}
            hotspot = bool(rec.get("hotspot_violation"))
            checks.append(PilotCheck(
                "safety_hotspot_screen",
                not hotspot,
                "recommended hotspot_violation" if hotspot else "recommended within hotspot limit",
            ))
        else:
            checks.append(PilotCheck(
                "safety_hotspot_screen",
                False,
                "multi_objective run without --check-hotspot",
            ))
    elif require_safety_flags:
        checks.append(PilotCheck(
            "safety_multi_objective",
            False,
            "pilot_ready expects --multi-objective with --check-arcing and --check-hotspot",
        ))
    else:
        checks.append(PilotCheck("safety_multi_objective", True, "not required"))

    passed = all(c.passed for c in checks)
    return PilotGateReport(passed=passed, checks=tuple(checks))
