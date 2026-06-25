"""Figures of merit for selective mineral heating.

The central quantity is *selectivity*: of all the microwave power absorbed in the
charge, what fraction lands in the target mineral phase rather than the gangue?
That is the design target the photonics splitter FOM (flux split ratio) is replaced
by -- same "where does the energy go" question, different physics (absorption, not
routing).

Selectivity alone is not actionable, though (backlog A0): a design can route a high
*fraction* of charge power into the target while coupling almost no power into the
charge at all. So we also report **coupling efficiency** = (power absorbed in the
charge) / (total power absorbed anywhere). In this lossless-walled cavity, total
absorbed power equals the power delivered by the feed (energy conservation), so
coupling_eff is the fraction of delivered power that lands in the ore rather than
being dissipated in internal structure (notably the Im(eps)=1e6 "PEC" baffle, which is
a strong *absorber* in this approximation, not a lossless reflector -- see
``pec_loss_fraction``).

NOTE: a true scattering parameter |S11| / reflected power needs a matched port; the
grid-node point source has no usable input impedance (its driven-node field is
dominated by the source self-term). That metric is deferred to the openEMS port-truth
path (backlog A1); here we report the energy-consistent coupling efficiency only.
"""

from __future__ import annotations

from dataclasses import dataclass

from mw_inv.fdfd import SolveResult, absorbed_power_density
from mw_inv.geometry import Scene


@dataclass(frozen=True)
class FomReport:
    selectivity: float          # P_target / (P_target + P_gangue), in [0, 1]
    contrast: float             # mean power density target / gangue (per-area)
    p_target: float             # total absorbed power in target (arb. units)
    p_gangue: float             # total absorbed power in gangue
    p_total_charge: float       # p_target + p_gangue
    # --- Coupling (backlog A0) ---
    p_abs_total: float = 0.0    # total absorbed power anywhere (= delivered power)
    p_structural: float = 0.0   # absorbed outside the charge (PEC baffle, lossy tuner)
    coupling_eff: float = 0.0   # p_total_charge / p_abs_total, in [0, 1]
    pec_loss_fraction: float = 0.0  # share of absorbed power dumped in "PEC" cells

    def to_dict(self) -> dict[str, float]:
        return {
            "selectivity": self.selectivity,
            "contrast": self.contrast,
            "p_target": self.p_target,
            "p_gangue": self.p_gangue,
            "p_total_charge": self.p_total_charge,
            "p_abs_total": self.p_abs_total,
            "p_structural": self.p_structural,
            "coupling_eff": self.coupling_eff,
            "pec_loss_fraction": self.pec_loss_fraction,
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

    # Coupling: total absorbed power = delivered power (lossless walls). What fraction
    # of it lands in the ore charge vs internal structure (notably the lossy "PEC" baffle)?
    p_abs_total = float(p.sum() * cell)
    p_structural = max(p_abs_total - p_total, 0.0)
    coupling_eff = (p_total / p_abs_total) if p_abs_total > 0 else 0.0
    pec_mask = getattr(scene, "pec_mask", None)
    p_pec = float(p[pec_mask].sum() * cell) if pec_mask is not None else 0.0
    pec_loss_fraction = (p_pec / p_abs_total) if p_abs_total > 0 else 0.0

    return FomReport(
        selectivity=float(selectivity),
        contrast=contrast,
        p_target=p_target,
        p_gangue=p_gangue,
        p_total_charge=p_total,
        p_abs_total=p_abs_total,
        p_structural=p_structural,
        coupling_eff=float(coupling_eff),
        pec_loss_fraction=float(pec_loss_fraction),
    )
