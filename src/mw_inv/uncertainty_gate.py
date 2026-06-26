"""Uncertainty propagation gates — fail on lower confidence bounds (backlog C3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UncertaintyCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class UncertaintyGateReport:
    passed: bool
    checks: tuple[UncertaintyCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


def evaluate_uncertainty_gate(
    robust_block: dict[str, Any],
    *,
    min_p05_selectivity: float = 0.0,
    min_p05_coupling: float = 0.0,
    mode: str | None = None,
) -> UncertaintyGateReport:
    """Pass when material/freq robust reports meet percentile floors."""
    mode = mode or str(robust_block.get("mode", "none"))
    checks: list[UncertaintyCheck] = []

    def pick(prefix: str) -> dict | None:
        rep = robust_block.get(prefix)
        return rep if isinstance(rep, dict) else None

    if mode == "material":
        for label, key in (("untuned", "untuned_material"), ("best", "best_material")):
            rep = pick(key)
            if rep is None:
                checks.append(UncertaintyCheck(f"{label}_material_present", False, "missing block"))
                continue
            p05 = rep.get("p05_selectivity")
            if p05 is None:
                checks.append(UncertaintyCheck(f"{label}_p05_selectivity", False, "p05 missing"))
            else:
                ok = float(p05) >= min_p05_selectivity
                checks.append(UncertaintyCheck(
                    f"{label}_p05_selectivity",
                    ok,
                    f"p05={float(p05):.4f} (floor {min_p05_selectivity})",
                ))
            p05c = rep.get("p05_coupling_eff")
            if p05c is not None and min_p05_coupling > 0:
                okc = float(p05c) >= min_p05_coupling
                checks.append(UncertaintyCheck(
                    f"{label}_p05_coupling",
                    okc,
                    f"p05 coupling={float(p05c):.4f} (floor {min_p05_coupling})",
                ))
    elif mode in ("freq", "freq_ensemble"):
        for label, key in (("untuned", "untuned_freq"), ("best", "best_freq")):
            if mode == "freq_ensemble":
                key = key.replace("_freq", "_freq_ensemble")
            rep = pick(key)
            if rep is None:
                checks.append(UncertaintyCheck(f"{label}_freq_present", False, "missing block"))
                continue
            mn = rep.get("min_selectivity")
            if mn is None:
                checks.append(UncertaintyCheck(f"{label}_min_selectivity", False, "min missing"))
            else:
                ok = float(mn) >= min_p05_selectivity
                checks.append(UncertaintyCheck(
                    f"{label}_min_selectivity",
                    ok,
                    f"min={float(mn):.4f} (floor {min_p05_selectivity})",
                ))
    else:
        checks.append(UncertaintyCheck("uncertainty_mode", True, f"skipped for mode={mode!r}"))

    passed = all(c.passed for c in checks)
    return UncertaintyGateReport(passed=passed, checks=tuple(checks))
