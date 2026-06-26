"""Bench RF port report: compare measured VNA S11 vs openEMS port metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from mw_inv.openems_postprocess import OpenemsPortMetrics, load_port_metrics
from mw_inv.vna_s11 import S1PTrace, load_touchstone_s11, summary_s11_metrics


@dataclass(frozen=True)
class PortS11Report:
    unloaded: dict
    loaded: dict | None
    openems: dict | None

    def to_dict(self) -> dict:
        out = {
            "unloaded": self.unloaded,
            "loaded": self.loaded,
            "openems": self.openems,
        }
        if self.loaded is not None:
            out["delta"] = {
                "s11_mag_loaded_minus_unloaded": float(self.loaded["s11_mag"] - self.unloaded["s11_mag"]),
                "s11_db_loaded_minus_unloaded": float(self.loaded["s11_db"] - self.unloaded["s11_db"]),
            }
        if self.openems is not None:
            # Compare measured unloaded S11 at the openEMS simulated frequency.
            f = float(self.openems.get("freq_hz") or self.unloaded.get("freq_eval_hz") or 2.45e9)
            mag_meas = float(self.unloaded.get("s11_mag"))
            mag_sim = float(self.openems.get("s11_mag"))
            out["openems_compare"] = {
                "freq_hz": f,
                "measured_unloaded_s11_mag": mag_meas,
                "openems_s11_mag": mag_sim,
                "delta_s11_mag": float(mag_meas - mag_sim),
            }
        return out


@dataclass(frozen=True)
class RFGateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RFGateReport:
    passed: bool
    checks: tuple[RFGateCheck, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
        }


def _maybe_load_openems(metrics_path: Path | str | None) -> OpenemsPortMetrics | None:
    if metrics_path is None:
        return None
    p = Path(metrics_path)
    return load_port_metrics(p) if p.is_file() else None


def build_port_report(
    *,
    unloaded_s1p: Path | str,
    loaded_s1p: Path | str | None = None,
    openems_port_metrics: Path | str | None = None,
    freq_hz: float = 2.45e9,
    band_lo_hz: float | None = None,
    band_hi_hz: float | None = None,
) -> PortS11Report:
    unloaded = load_touchstone_s11(unloaded_s1p)
    loaded: S1PTrace | None = load_touchstone_s11(loaded_s1p) if loaded_s1p else None
    openems = _maybe_load_openems(openems_port_metrics)

    unloaded_summary = summary_s11_metrics(
        unloaded,
        freq_hz=freq_hz,
        band_lo_hz=band_lo_hz,
        band_hi_hz=band_hi_hz,
    )
    loaded_summary = (
        summary_s11_metrics(
            loaded,
            freq_hz=freq_hz,
            band_lo_hz=band_lo_hz,
            band_hi_hz=band_hi_hz,
        )
        if loaded is not None
        else None
    )
    openems_summary = openems.to_dict() if openems is not None else None

    # Add a convenience coupling estimate from measured |S11| (if user measured at same reference plane).
    unloaded_summary["coupling_eff_est"] = float(1.0 - unloaded_summary["s11_mag"] ** 2)
    if loaded_summary is not None:
        loaded_summary["coupling_eff_est"] = float(1.0 - loaded_summary["s11_mag"] ** 2)
        # Prevent silly rounding artifacts.
        loaded_summary["coupling_eff_est"] = float(max(0.0, min(1.0, loaded_summary["coupling_eff_est"])))

    if openems_summary is not None and openems_summary.get("s11_mag") is not None:
        openems_summary["s11_db"] = float(20.0 * math.log10(max(float(openems_summary["s11_mag"]), 1e-12)))

    return PortS11Report(unloaded=unloaded_summary, loaded=loaded_summary, openems=openems_summary)


def evaluate_rf_gate(
    report: PortS11Report,
    *,
    max_unloaded_s11: float = 0.92,
    max_loaded_s11: float = 0.92,
    max_loaded_minus_unloaded: float = 0.10,
) -> RFGateReport:
    """Simple Stage-A RF acceptance checks from VNA |S11| traces."""
    checks: list[RFGateCheck] = []

    u = float(report.unloaded.get("s11_mag", 1.0))
    checks.append(RFGateCheck(
        "unloaded_s11",
        u <= max_unloaded_s11,
        f"|S11|={u:.3f} (max {max_unloaded_s11})",
    ))

    if report.loaded is None:
        checks.append(RFGateCheck("loaded_trace_present", False, "missing loaded trace"))
    else:
        s11 = float(report.loaded.get("s11_mag", 1.0))
        checks.append(RFGateCheck(
            "loaded_s11",
            s11 <= max_loaded_s11,
            f"|S11|={s11:.3f} (max {max_loaded_s11})",
        ))
        checks.append(RFGateCheck(
            "loaded_minus_unloaded",
            (s11 - u) <= max_loaded_minus_unloaded,
            f"Δ|S11|={s11 - u:+.3f} (max {max_loaded_minus_unloaded})",
        ))

    passed = all(c.passed for c in checks)
    return RFGateReport(passed=passed, checks=tuple(checks))
