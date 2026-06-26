"""Particle-level tail-risk gate (backlog D2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParticleTailCheck:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ParticleTailGateReport:
    passed: bool
    checks: tuple[ParticleTailCheck, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


def evaluate_particle_tail_gate(
    foms: dict,
    *,
    min_p05_particle_fraction: float = 0.0,
    max_gangue_power_fraction: float = 1.0,
    max_p95_particle_fraction: float = 1.0,
) -> ParticleTailGateReport:
    """Gate on per-particle tail statistics from ``DesignReport.foms``."""
    checks: list[ParticleTailCheck] = []
    p05 = foms.get("p05_particle_fraction")
    if min_p05_particle_fraction > 0:
        if p05 is None:
            checks.append(ParticleTailCheck("p05_particle_fraction", False, "missing"))
        else:
            ok = float(p05) >= min_p05_particle_fraction
            checks.append(
                ParticleTailCheck(
                    "p05_particle_fraction",
                    ok,
                    f"p05={float(p05):.4f} (floor {min_p05_particle_fraction})",
                )
            )
    gangue = foms.get("gangue_power_fraction")
    if max_gangue_power_fraction < 1.0:
        if gangue is None:
            checks.append(ParticleTailCheck("gangue_power_fraction", False, "missing"))
        else:
            ok = float(gangue) <= max_gangue_power_fraction
            checks.append(
                ParticleTailCheck(
                    "gangue_power_fraction",
                    ok,
                    f"gangue={float(gangue):.4f} (cap {max_gangue_power_fraction})",
                )
            )
    p95 = foms.get("p95_particle_fraction")
    if max_p95_particle_fraction < 1.0 and p95 is not None:
        ok = float(p95) <= max_p95_particle_fraction
        checks.append(
            ParticleTailCheck(
                "p95_particle_hotspot",
                ok,
                f"p95 particle={float(p95):.4f} (cap {max_p95_particle_fraction})",
            )
        )
    if not checks:
        checks.append(ParticleTailCheck("particle_tail_disabled", True, "no thresholds"))
    passed = all(c.passed for c in checks)
    return ParticleTailGateReport(passed=passed, checks=tuple(checks))
