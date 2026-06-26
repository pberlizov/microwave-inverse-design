"""Promotion tiers — trust boundary for exports and claims.

Tiers (cumulative requirements):

  literature_grounded  — benchmark suite passes (materials forward model)
  fdfd_optimised       — validation gate passes (optimised beats untuned on FDFD)
  deposit_calibrated   — named deposit with validated measured ore ε(f,T,moisture)
  solver_triangulated  — external solver data present and gate solver checks pass
  bench_calibrated     — phantom probe ε drift within tolerance (optional bench JSON)
  pilot_ready          — safety-screened multi-objective + robust repeatability + coupling floor
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from mw_inv.validation_gate import ValidationGateReport


class PromotionTier(str, Enum):
    UNRANKED = "unranked"
    LITERATURE_GROUNDED = "literature_grounded"
    FDFD_OPTIMISED = "fdfd_optimised"
    DEPOSIT_CALIBRATED = "deposit_calibrated"
    SOLVER_TRIANGULATED = "solver_triangulated"
    BENCH_CALIBRATED = "bench_calibrated"
    PILOT_READY = "pilot_ready"


TIER_ORDER: tuple[PromotionTier, ...] = (
    PromotionTier.UNRANKED,
    PromotionTier.LITERATURE_GROUNDED,
    PromotionTier.FDFD_OPTIMISED,
    PromotionTier.DEPOSIT_CALIBRATED,
    PromotionTier.SOLVER_TRIANGULATED,
    PromotionTier.BENCH_CALIBRATED,
    PromotionTier.PILOT_READY,
)


class PromotionError(PermissionError):
    """Raised when an action requires a higher promotion tier."""


def tier_rank(tier: PromotionTier | str) -> int:
    t = PromotionTier(tier) if isinstance(tier, str) else tier
    return TIER_ORDER.index(t)


def meets_tier(current: PromotionTier | str, required: PromotionTier | str) -> bool:
    return tier_rank(current) >= tier_rank(required)


def assert_tier_at_least(
    current: PromotionTier | str,
    required: PromotionTier | str,
    *,
    action: str = "proceed",
) -> None:
    if not meets_tier(current, required):
        raise PromotionError(
            f"Cannot {action}: promotion tier {current!s} "
            f"is below required {required!s}. "
            f"Run scripts/run_pipeline.py or improve gate/benchmark/bench inputs."
        )


@dataclass(frozen=True)
class PromotionAssessment:
    tier: PromotionTier
    requirements: dict[str, bool]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "requirements": dict(self.requirements),
            "notes": list(self.notes),
        }


def _external_solver_data_present(triangulation_rows: list[Any] | None) -> bool:
    if not triangulation_rows:
        return False
    for row in triangulation_rows:
        for attr in ("meep_2d_selectivity", "meep_3d_primitive_selectivity", "openems_selectivity"):
            if getattr(row, attr, None) is not None:
                return True
    return False


def _solver_gate_checks_ok(gate: ValidationGateReport | None) -> bool:
    if gate is None:
        return False
    for check in gate.checks:
        if check.name.endswith("_rel_err") or check.name.endswith("_rank_match"):
            if not check.passed:
                return False
    return True


def _bench_calibration_ok(
    phantom_label: str | None,
    measured_eps_path: str | Path | None,
    lab_measurements_path: str | Path | None = None,
    *,
    max_real_drift: float = 1.5,
    max_imag_drift: float = 0.5,
) -> bool:
    if not phantom_label or not measured_eps_path:
        return False
    from pathlib import Path

    path = Path(measured_eps_path)
    if not path.is_file():
        return False
    from mw_inv.phantom_calibration import compare_measured_vs_anchor

    report = compare_measured_vs_anchor(phantom_label, path)
    for row in report.get("comparisons", []):
        if row.get("status") == "missing":
            return False
        if abs(float(row.get("drift_real", 0.0))) > max_real_drift:
            return False
        if abs(float(row.get("drift_imag", 0.0))) > max_imag_drift:
            return False
    if not report.get("comparisons"):
        return False

    if lab_measurements_path:
        lp = Path(lab_measurements_path)
        if not lp.is_file():
            return False
        import json

        payload = json.loads(lp.read_text())
        rows = payload if isinstance(payload, list) else payload.get("measurements", [])
        matches = [r for r in rows if r.get("phantom") == phantom_label]
        if not matches:
            return False
        # Require at least one record with explicit untuned baseline where optimized beats it.
        ok_rank = False
        for r in matches:
            if r.get("untuned_measured_delta_T_K") is None:
                continue
            ok_rank = float(r["measured_delta_T_K"]) > float(r["untuned_measured_delta_T_K"])
            if ok_rank:
                break
        if not ok_rank:
            return False

    return True


def _deposit_bruggeman_calibration_ok(cal_block: dict | None) -> bool:
    """When Bruggeman calibration was evaluated, it must pass declared tolerance."""
    if cal_block is None:
        return True
    if cal_block.get("error"):
        return False
    if "passes_calibration" in cal_block:
        return bool(cal_block["passes_calibration"])
    return bool(cal_block.get("passes_default_tolerance", True))


def _deposit_calibration_ok(ore_block: dict | None) -> bool:
    """Deposit tier: pipeline run used validated measured ore ε, not Bruggeman-only."""
    if not ore_block:
        return False
    if ore_block.get("materials_mode") != "measured":
        return False
    md = ore_block.get("measured_dielectrics") or {}
    if md.get("error"):
        return False
    issues = md.get("validation_issues") or []
    if issues:
        return False
    if not md.get("path"):
        return False
    dataset = md.get("dataset") or {}
    return bool(dataset.get("dataset_id") or dataset.get("phases"))


def _campaign_deposit_ok(campaign_block: dict | None) -> bool:
    """Campaign path: linked measured dielectrics library on the campaign manifest."""
    if not campaign_block:
        return False
    return bool(campaign_block.get("measured_dielectrics"))


def _deposit_envelope_gate_ok(gate_block: dict | None) -> bool:
    """When an envelope gate was evaluated, it must pass for deposit tier."""
    if gate_block is None:
        return True
    return bool(gate_block.get("passed"))


def _pilot_ready_ok(pilot_gate: dict | None) -> bool:
    """Pilot tier: explicit pilot_gate block from pipeline evaluation."""
    if not pilot_gate:
        return False
    return bool(pilot_gate.get("passed"))


def assess_promotion(
    *,
    benchmarks_passed: bool | None = None,
    gate: ValidationGateReport | None = None,
    triangulation_rows: list[Any] | None = None,
    ore_block: dict | None = None,
    campaign_block: dict | None = None,
    deposit_envelope_gate: dict | None = None,
    deposit_calibration: dict | None = None,
    phantom_label: str | None = None,
    measured_eps_path: str | None = None,
    lab_measurements_path: str | None = None,
    pilot_gate: dict | None = None,
) -> PromotionAssessment:
    """Compute highest tier satisfied by available evidence."""

    lit = benchmarks_passed is True
    fdfd = lit and gate is not None and gate.passed
    deposit_material = _deposit_calibration_ok(ore_block) or _campaign_deposit_ok(campaign_block)
    envelope_ok = _deposit_envelope_gate_ok(deposit_envelope_gate)
    bruggeman_ok = _deposit_bruggeman_calibration_ok(deposit_calibration)
    deposit_ok = fdfd and deposit_material and envelope_ok and bruggeman_ok
    has_ext = _external_solver_data_present(triangulation_rows)
    solver_ok = fdfd and has_ext and _solver_gate_checks_ok(gate)
    bench_ok = solver_ok and _bench_calibration_ok(
        phantom_label,
        measured_eps_path if measured_eps_path else None,
        lab_measurements_path if lab_measurements_path else None,
    )
    pilot_ok = bench_ok and _pilot_ready_ok(pilot_gate)

    reqs = {
        "literature_benchmarks": lit,
        "fdfd_gate": fdfd,
        "deposit_measured_eps": deposit_material,
        "deposit_envelope_gate": envelope_ok if deposit_envelope_gate is not None else True,
        "deposit_bruggeman_calibration": bruggeman_ok if deposit_calibration is not None else True,
        "external_solver_validation": solver_ok,
        "bench_phantom_calibration": bench_ok,
        "pilot_safety_repeatability": pilot_ok,
    }
    notes: list[str] = []
    if fdfd and not has_ext:
        notes.append("solver_triangulated requires MEEP/openEMS data — tier capped below solver")
    if fdfd and not deposit_material:
        notes.append(
            "deposit_calibrated requires --ore with validated measured_dielectrics "
            "or --campaign with measured_dielectrics"
        )
    if deposit_envelope_gate is not None and not envelope_ok:
        notes.append("deposit_calibrated requires passing deposit envelope gate (min-over-ores)")
    if deposit_calibration is not None and not bruggeman_ok:
        notes.append(
            "deposit_calibrated requires passing Bruggeman calibration "
            "(run --calibrate-deposit; max rel error within tolerance)"
        )
    if solver_ok and not bench_ok:
        notes.append("bench_calibrated requires measured_eps.json within drift tolerance")
    if bench_ok and not pilot_ok:
        notes.append(
            "pilot_ready requires --multi-objective --check-arcing --check-hotspot, "
            "--robust, and passing pilot_gate"
        )

    if pilot_ok:
        tier = PromotionTier.PILOT_READY
    elif bench_ok:
        tier = PromotionTier.BENCH_CALIBRATED
    elif solver_ok:
        tier = PromotionTier.SOLVER_TRIANGULATED
    elif deposit_ok:
        tier = PromotionTier.DEPOSIT_CALIBRATED
    elif fdfd:
        tier = PromotionTier.FDFD_OPTIMISED
    elif lit:
        tier = PromotionTier.LITERATURE_GROUNDED
    else:
        tier = PromotionTier.UNRANKED
        notes = ("Run benchmarks and validation gate to establish promotion tier.",)

    return PromotionAssessment(tier=tier, requirements=reqs, notes=tuple(notes))


def tier_from_manifest(manifest: dict[str, Any]) -> PromotionTier:
    """Read tier from a run manifest (or compute from embedded reports)."""
    if "promotion" in manifest and "tier" in manifest["promotion"]:
        return PromotionTier(manifest["promotion"]["tier"])
    return assess_promotion(
        benchmarks_passed=manifest.get("benchmarks", {}).get("passed"),
        gate=_gate_from_dict(manifest.get("gate")),
        triangulation_rows=_rows_from_dict(manifest.get("triangulation", {})),
        ore_block=manifest.get("ore"),
        campaign_block=manifest.get("evaluation", {}).get("campaign"),
        deposit_envelope_gate=manifest.get("evaluation", {}).get("deposit_envelope_gate"),
        deposit_calibration=manifest.get("evaluation", {}).get("deposit_calibration"),
        phantom_label=manifest.get("bench", {}).get("phantom_label"),
        measured_eps_path=manifest.get("bench", {}).get("measured_eps_path"),
        lab_measurements_path=manifest.get("bench", {}).get("lab_measurements_path"),
        pilot_gate=manifest.get("evaluation", {}).get("pilot_gate"),
    ).tier


def _gate_from_dict(block: dict | None) -> ValidationGateReport | None:
    if not block or "passed" not in block:
        return None
    from mw_inv.validation_gate import GateCheck

    checks = [
        GateCheck(name=c["name"], passed=c["passed"], detail=c.get("detail", ""))
        for c in block.get("checks", [])
    ]
    return ValidationGateReport(
        passed=bool(block["passed"]),
        checks=checks,
        rank_agreement=block.get("rank_agreement", {}),
        openems_diagnosis=block.get("openems_diagnosis"),
    )


def _rows_from_dict(block: dict) -> list[Any] | None:
    rows = block.get("rows")
    if not rows:
        return None

    from mw_inv.solver_triangulation import SolverRow

    out: list[SolverRow] = []
    for r in rows:
        out.append(SolverRow(
            label=r["label"],
            fdfd_selectivity=float(r["fdfd_selectivity"]),
            fdfd_coupling_eff=r.get("fdfd_coupling_eff"),
            fdfd_pec_loss_fraction=r.get("fdfd_pec_loss_fraction"),
            meep_2d_selectivity=r.get("meep_2d_selectivity"),
            meep_3d_primitive_selectivity=r.get("meep_3d_primitive_selectivity"),
            openems_selectivity=r.get("openems_selectivity"),
            openems_s11_mag=r.get("openems_s11_mag"),
            openems_coupling_eff=r.get("openems_coupling_eff"),
        ))
    return out
