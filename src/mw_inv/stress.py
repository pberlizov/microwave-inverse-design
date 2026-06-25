"""Thermoelastic stress proxy at target–gangue interfaces (Salsman-style FOM).

Models interface tensile stress from differential thermal expansion when the
target grain heats faster than gangue — the mechanism cited for thermally
assisted liberation (Salsman et al. 1996; Kingman texture studies).

This is a **2D quasi-static proxy**, not a full FE solid-mechanics solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from mw_inv.fdfd import Grid, SolveResult, absorbed_power_density, solve
from mw_inv.fom import FomReport, evaluate
from mw_inv.geometry import CavityParams, Scene, build_scene
from mw_inv.materials import Materials
from mw_inv.thermal import CoupledResult, ThermalConfig, coupled_steady_state


@dataclass(frozen=True)
class ThermoelasticProps:
    """Linear thermoelastic constants (representative, room-T)."""

    target_alpha: float = 14.0e-6   # 1/K — pyrite (order of magnitude)
    gangue_alpha: float = 6.0e-6    # 1/K — calcite
    target_E_Pa: float = 150e9
    gangue_E_Pa: float = 79e9
    nu: float = 0.25


@dataclass(frozen=True)
class StressReport:
    """Interface stress figures of merit."""

    mean_interface_stress_Pa: float
    max_interface_stress_Pa: float
    stress_selectivity: float       # target interface / (target + gangue interface)
    em_selectivity: float
    delta_T_K: float
    grain_radius_m: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean_interface_stress_Pa": self.mean_interface_stress_Pa,
            "max_interface_stress_Pa": self.max_interface_stress_Pa,
            "stress_selectivity": self.stress_selectivity,
            "em_selectivity": self.em_selectivity,
            "delta_T_K": self.delta_T_K,
            "grain_radius_m": self.grain_radius_m,
        }


def _interface_mask(target_mask: np.ndarray) -> np.ndarray:
    """Pixels on target boundary (interface with gangue or exterior)."""
    eroded = ndimage.binary_erosion(target_mask, iterations=1)
    return target_mask & ~eroded


def interface_stress_field(
    T: np.ndarray,
    scene: Scene,
    props: ThermoelasticProps,
    T_amb_K: float = 298.0,
) -> np.ndarray:
    """Radial constraint stress proxy σ ≈ E_g Δα ΔT / (1−ν) at target rim."""
    dT = np.clip(T - T_amb_K, 0.0, None)
    delta_alpha = props.target_alpha - props.gangue_alpha
    coeff = props.gangue_E_Pa * abs(delta_alpha) / max(1.0 - props.nu, 0.01)
    stress = np.zeros_like(T, dtype=float)
    iface = _interface_mask(scene.target_mask)
    stress[iface] = coeff * dT[iface]
    return stress


def evaluate_stress_from_coupled(
    coupled: CoupledResult,
    props: ThermoelasticProps | None = None,
) -> StressReport:
    props = props or ThermoelasticProps()
    scene = coupled.scene
    stress = interface_stress_field(coupled.temperature_K, scene, props)
    iface_t = _interface_mask(scene.target_mask)
    iface_g = _interface_mask(scene.gangue_mask)
    s_t = float(stress[iface_t].sum()) if iface_t.any() else 0.0
    s_g = float(stress[iface_g].sum()) if iface_g.any() else 0.0
    total = s_t + s_g
    return StressReport(
        mean_interface_stress_Pa=float(stress[iface_t].mean()) if iface_t.any() else 0.0,
        max_interface_stress_Pa=float(stress.max()),
        stress_selectivity=s_t / total if total > 0 else 0.0,
        em_selectivity=coupled.thermal.em_selectivity,
        delta_T_K=coupled.thermal.delta_T_K,
        grain_radius_m=scene.params.inclusion_radius_frac * min(scene.grid.Lx, scene.grid.Ly),
    )


def evaluate_stress(
    grid: Grid,
    params: CavityParams,
    pair_label: str,
    *,
    materials: Materials | None = None,
    thermal_cfg: ThermalConfig | None = None,
    mech: ThermoelasticProps | None = None,
) -> tuple[StressReport, CoupledResult]:
    """Coupled EM–thermal → interface stress report."""
    cfg = thermal_cfg or ThermalConfig(drive=8.0, max_iters=15, tol_K=3.0)
    if materials is not None:
        coupled = coupled_steady_state(grid, "", config=cfg, params=params, materials=materials)
    else:
        coupled = coupled_steady_state(grid, pair_label, config=cfg, params=params)
    return evaluate_stress_from_coupled(coupled, mech), coupled


def grain_size_penalty_factor(radius_m: float, *, r_opt_m: float = 2.5e-3) -> float:
    """Kingman-style penalty: fine grains bleed heat → lower effective stress.

    Peak near r_opt (~2.5 mm for 2.45 GHz skin-depth scale); falls for very
    fine (<100 µm) or very coarse grains.
    """
    r = max(radius_m, 1e-5)
    ratio = r / r_opt_m
    return float(np.exp(-0.5 * (np.log(ratio) ** 2)))
