"""Hotspot / runaway proxy using evolved vs frozen ε(T) (backlog E2)."""

from __future__ import annotations

from dataclasses import dataclass

from mw_inv.fdfd import Grid
from mw_inv.geometry import CavityParams
from mw_inv.thermal import ThermalConfig, coupled_steady_state, thermal_props_for_pair


@dataclass(frozen=True)
class HotspotGateReport:
    pair_label: str
    max_hotspot_delta_T_K: float
    evolved_delta_T_K: float
    frozen_delta_T_K: float
    evolved_violation: bool
    frozen_violation: bool
    uses_evolved_for_gate: bool
    passed: bool

    def to_dict(self) -> dict:
        return {
            "pair_label": self.pair_label,
            "max_hotspot_delta_T_K": self.max_hotspot_delta_T_K,
            "evolved_delta_T_K": self.evolved_delta_T_K,
            "frozen_delta_T_K": self.frozen_delta_T_K,
            "evolved_violation": self.evolved_violation,
            "frozen_violation": self.frozen_violation,
            "uses_evolved_for_gate": self.uses_evolved_for_gate,
            "passed": self.passed,
        }


def _peak_target_delta_T(
    grid: Grid,
    pair_label: str,
    params: CavityParams,
    *,
    evolve_properties: bool,
    T_amb_K: float = 298.0,
) -> float:
    cfg = ThermalConfig(
        max_iters=12,
        tol_K=4.0,
        thermal_props=thermal_props_for_pair(pair_label),
        evolve_properties=evolve_properties,
        T_amb_K=T_amb_K,
    )
    coupled = coupled_steady_state(
        grid, pair_label, config=cfg, params=params, materials=None,
    )
    return float(coupled.thermal.T_max_target_K - T_amb_K)


def evaluate_hotspot_gate(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    *,
    max_hotspot_delta_T_K: float = 475.0,
    uses_evolved: bool = True,
    T_amb_K: float = 298.0,
) -> HotspotGateReport:
    """Compare peak target ΔT with evolved ε(T)+phase rules vs frozen RT ε."""
    evolved_dt = _peak_target_delta_T(
        grid, pair_label, params, evolve_properties=True, T_amb_K=T_amb_K,
    )
    frozen_dt = _peak_target_delta_T(
        grid, pair_label, params, evolve_properties=False, T_amb_K=T_amb_K,
    )
    ev_viol = evolved_dt > max_hotspot_delta_T_K
    fr_viol = frozen_dt > max_hotspot_delta_T_K
    return HotspotGateReport(
        pair_label=pair_label,
        max_hotspot_delta_T_K=max_hotspot_delta_T_K,
        evolved_delta_T_K=evolved_dt,
        frozen_delta_T_K=frozen_dt,
        evolved_violation=ev_viol,
        frozen_violation=fr_viol,
        uses_evolved_for_gate=uses_evolved,
        passed=not (ev_viol if uses_evolved else fr_viol),
    )
