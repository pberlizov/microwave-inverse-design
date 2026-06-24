"""Figures of merit for selective mineral heating.

The central quantity is *selectivity*: of all the microwave power absorbed in the
charge, what fraction lands in the target mineral phase rather than the gangue?
That is the design target the photonics splitter FOM (flux split ratio) is replaced
by -- same "where does the energy go" question, different physics (absorption, not
routing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mw_inv.fdfd import SolveResult, absorbed_power_density
from mw_inv.geometry import Scene


@dataclass(frozen=True)
class FomReport:
    selectivity: float          # P_target / (P_target + P_gangue), in [0, 1]
    contrast: float             # mean power density target / gangue (per-area)
    p_target: float             # total absorbed power in target (arb. units)
    p_gangue: float             # total absorbed power in gangue
    p_total_charge: float       # p_target + p_gangue

    def to_dict(self) -> dict[str, float]:
        return {
            "selectivity": self.selectivity,
            "contrast": self.contrast,
            "p_target": self.p_target,
            "p_gangue": self.p_gangue,
            "p_total_charge": self.p_total_charge,
        }


def evaluate(result: SolveResult, scene: Scene) -> FomReport:
    p = absorbed_power_density(result)
    cell = scene.grid.dx * scene.grid.dy

    p_target = float(p[scene.target_mask].sum() * cell)
    p_gangue = float(p[scene.gangue_mask].sum() * cell)
    p_total = p_target + p_gangue

    selectivity = p_target / p_total if p_total > 0 else 0.0

    n_t = int(scene.target_mask.sum())
    n_g = int(scene.gangue_mask.sum())
    mean_t = (p[scene.target_mask].mean() if n_t else 0.0)
    mean_g = (p[scene.gangue_mask].mean() if n_g else 0.0)
    contrast = float(mean_t / mean_g) if mean_g > 0 else float("inf")

    return FomReport(
        selectivity=float(selectivity),
        contrast=contrast,
        p_target=p_target,
        p_gangue=p_gangue,
        p_total_charge=p_total,
    )
