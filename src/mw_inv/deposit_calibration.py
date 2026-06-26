"""Auto-calibration: probe + assay → effective ε diff (backlog D4).

Compares Bruggeman-effective permittivity from an ore profile against the
referenced measured_dielectrics library at the same (T, f) probe conditions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataclasses import replace

from mw_inv.measured_dielectrics import load_measured_dielectrics
from mw_inv.ore_profiles import (
    load_ore_profile,
    materials_from_ore,
    resolve_measured_dielectrics_path,
)


@dataclass(frozen=True)
class PhaseCalibrationDiff:
    phase: str
    mineral: str
    predicted_eps_real: float
    predicted_eps_imag: float
    measured_eps_real: float
    measured_eps_imag: float
    delta_eps_real: float
    delta_eps_imag: float
    rel_error_real: float
    rel_error_imag: float

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "mineral": self.mineral,
            "predicted_eps_real": self.predicted_eps_real,
            "predicted_eps_imag": self.predicted_eps_imag,
            "measured_eps_real": self.measured_eps_real,
            "measured_eps_imag": self.measured_eps_imag,
            "delta_eps_real": self.delta_eps_real,
            "delta_eps_imag": self.delta_eps_imag,
            "rel_error_real": self.rel_error_real,
            "rel_error_imag": self.rel_error_imag,
        }


@dataclass(frozen=True)
class DepositCalibrationReport:
    ore_label: str
    ore_path: str
    measured_library: str
    target_T_K: float
    gangue_T_K: float
    freq_hz: float
    phases: tuple[PhaseCalibrationDiff, ...]
    max_rel_error_real: float
    max_rel_error_imag: float

    def passes(self, *, max_rel_error: float = 0.25) -> bool:
        return (
            self.max_rel_error_real <= max_rel_error
            and self.max_rel_error_imag <= max_rel_error
        )

    def to_dict(self) -> dict:
        return {
            "ore_label": self.ore_label,
            "ore_path": self.ore_path,
            "measured_library": self.measured_library,
            "target_T_K": self.target_T_K,
            "gangue_T_K": self.gangue_T_K,
            "freq_hz": self.freq_hz,
            "max_rel_error_real": self.max_rel_error_real,
            "max_rel_error_imag": self.max_rel_error_imag,
            "passes_default_tolerance": self.passes(),
            "phases": [p.to_dict() for p in self.phases],
        }


def _phase_diff(
    phase: str,
    mineral: str,
    predicted: complex,
    measured: complex,
) -> PhaseCalibrationDiff:
    pr, pi = float(predicted.real), float(predicted.imag)
    mr, mi = float(measured.real), float(measured.imag)
    dr, di = pr - mr, pi - mi
    rel_r = abs(dr / mr) if abs(mr) > 1e-12 else abs(dr)
    rel_i = abs(di / mi) if abs(mi) > 1e-12 else abs(di)
    return PhaseCalibrationDiff(
        phase=phase,
        mineral=mineral,
        predicted_eps_real=pr,
        predicted_eps_imag=pi,
        measured_eps_real=mr,
        measured_eps_imag=mi,
        delta_eps_real=dr,
        delta_eps_imag=di,
        rel_error_real=float(rel_r),
        rel_error_imag=float(rel_i),
    )


def calibrate_ore_profile(
    ore_path: Path | str,
    *,
    target_T_K: float = 298.15,
    gangue_T_K: float = 298.15,
    freq_hz: float = 2.45e9,
    moisture_wt_percent: float | None = None,
) -> DepositCalibrationReport:
    """Compare effective ε from ore assay vs measured library at probe conditions."""
    path = Path(ore_path)
    ore = load_ore_profile(path)
    measured_block = ore.measured_dielectrics or {}
    if not measured_block.get("path"):
        raise ValueError("ore profile lacks measured_dielectrics.path for calibration")
    lib_path = resolve_measured_dielectrics_path(path, measured_block)
    lib = load_measured_dielectrics(lib_path)

    moisture = moisture_wt_percent
    if moisture is None and measured_block.get("default_moisture_wt_percent") is not None:
        moisture = float(measured_block["default_moisture_wt_percent"])
    if moisture is None and measured_block.get("moisture_wt_percent") is not None:
        moisture = float(measured_block["moisture_wt_percent"])

    target_phase = str(measured_block.get("target_phase", "target"))
    gangue_phase = str(measured_block.get("gangue_phase", "gangue"))

    brugg = materials_from_ore(
        replace(ore, measured_dielectrics=None),
        ore_profile_path=path,
        target_T_K=target_T_K,
        gangue_T_K=gangue_T_K,
        freq_hz=freq_hz,
        moisture_wt_percent=moisture,
    )

    phases: list[PhaseCalibrationDiff] = []
    for phase, mineral, pred in (
        ("target", target_phase, brugg.target),
        ("gangue", gangue_phase, brugg.gangue),
    ):
        meas = lib.eps(
            mineral,
            temp_K=target_T_K if phase == "target" else gangue_T_K,
            freq_hz=freq_hz,
            moisture_wt_percent=moisture,
        )
        phases.append(_phase_diff(phase, mineral, pred, meas))

    rel_r = max(p.rel_error_real for p in phases)
    rel_i = max(p.rel_error_imag for p in phases)
    return DepositCalibrationReport(
        ore_label=ore.label,
        ore_path=str(path.resolve()),
        measured_library=str(lib_path.resolve()),
        target_T_K=target_T_K,
        gangue_T_K=gangue_T_K,
        freq_hz=freq_hz,
        phases=tuple(phases),
        max_rel_error_real=float(rel_r),
        max_rel_error_imag=float(rel_i),
    )


def write_calibration_diff(
    report: DepositCalibrationReport,
    out_path: Path | str,
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=2))
    return p


def diff_calibration_reports(
    baseline: DepositCalibrationReport | dict,
    current: DepositCalibrationReport | dict,
) -> dict:
    """Changelog between two calibration runs (D4 manifest diff)."""
    if isinstance(baseline, DepositCalibrationReport):
        b = baseline.to_dict()
    else:
        b = baseline
    if isinstance(current, DepositCalibrationReport):
        c = current.to_dict()
    else:
        c = current

    phase_deltas: list[dict] = []
    b_phases = {p["phase"]: p for p in b.get("phases", [])}
    for cp in c.get("phases", []):
        bp = b_phases.get(cp["phase"], {})
        phase_deltas.append(
            {
                "phase": cp["phase"],
                "mineral": cp.get("mineral"),
                "delta_eps_real": cp.get("delta_eps_real", 0.0) - bp.get("delta_eps_real", 0.0),
                "delta_eps_imag": cp.get("delta_eps_imag", 0.0) - bp.get("delta_eps_imag", 0.0),
                "rel_error_real": cp.get("rel_error_real", 0.0) - bp.get("rel_error_real", 0.0),
                "rel_error_imag": cp.get("rel_error_imag", 0.0) - bp.get("rel_error_imag", 0.0),
            }
        )

    return {
        "baseline_ore": b.get("ore_label"),
        "current_ore": c.get("ore_label"),
        "baseline_max_rel_error_real": b.get("max_rel_error_real"),
        "current_max_rel_error_real": c.get("max_rel_error_real"),
        "max_rel_error_real_shift": (
            float(c.get("max_rel_error_real", 0.0)) - float(b.get("max_rel_error_real", 0.0))
        ),
        "baseline_passes": b.get("passes_default_tolerance"),
        "current_passes": c.get("passes_default_tolerance"),
        "passes_flipped": bool(b.get("passes_default_tolerance")) != bool(c.get("passes_default_tolerance")),
        "phase_deltas": phase_deltas,
    }


def write_calibration_changelog(
    baseline: DepositCalibrationReport | dict,
    current: DepositCalibrationReport,
    out_path: Path | str,
) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(diff_calibration_reports(baseline, current), indent=2))
    return p
