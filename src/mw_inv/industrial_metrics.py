"""Industrial decision KPIs derived from FDFD FOM reports (backlog I0).

These are **proxies** for plant metrics — useful for ranking and gates, not
substitutes for pilot kW measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

from mw_inv.fom import FomReport

# Representative ROM bulk density for energy proxy (t/m³).
DEFAULT_BULK_DENSITY_T_M3 = 2.7
# Nominal forward power used when converting absorbed power to kWh/t proxy.
DEFAULT_FORWARD_POWER_KW = 100.0
# Nominal residence time for throughput proxy (continuous bed, seconds).
DEFAULT_RESIDENCE_TIME_S = 120.0


@dataclass(frozen=True)
class IndustrialMetrics:
    """Charge-level KPIs complementing selectivity."""

    gangue_power_fraction: float   # P_gangue / P_charge — waste heating budget
    target_power_fraction: float   # P_target / P_charge (= selectivity on charge)
    structural_power_fraction: float  # P_structural / P_abs_total
    coupling_eff: float
    pec_loss_fraction: float
    # Energy proxy: kWh per tonne if all delivered power were absorbed in charge
    # at DEFAULT_FORWARD_POWER_KW for one steady-state snapshot.
    specific_energy_proxy_kwh_per_t: float | None = None
    charge_tonnes_proxy: float | None = None
    throughput_proxy_t_per_h: float | None = None
    delivered_kw_proxy: float | None = None

    @classmethod
    def from_fom(
        cls,
        fom: FomReport,
        *,
        charge_volume_m3: float | None = None,
        bulk_density_t_m3: float = DEFAULT_BULK_DENSITY_T_M3,
        forward_power_kw: float = DEFAULT_FORWARD_POWER_KW,
        residence_time_s: float | None = DEFAULT_RESIDENCE_TIME_S,
    ) -> "IndustrialMetrics":
        p_charge = fom.p_total_charge
        gangue_frac = (fom.p_gangue / p_charge) if p_charge > 0 else 0.0
        target_frac = (fom.p_target / p_charge) if p_charge > 0 else 0.0
        struct_frac = (fom.p_structural / fom.p_abs_total) if fom.p_abs_total > 0 else 0.0

        energy_kwh_t: float | None = None
        tonnes: float | None = None
        throughput_t_h: float | None = None
        delivered_kw: float | None = None
        if charge_volume_m3 is not None and charge_volume_m3 > 0:
            tonnes = charge_volume_m3 * bulk_density_t_m3
            if tonnes > 0 and p_charge > 0 and fom.p_abs_total > 0:
                # Fraction of forward power coupling into charge × hours per tonne at 1 h exposure.
                frac_coupled = fom.coupling_eff
                energy_kwh_t = forward_power_kw * frac_coupled / tonnes
            if tonnes and residence_time_s and residence_time_s > 0:
                throughput_t_h = tonnes * 3600.0 / residence_time_s
            delivered_kw = forward_power_kw * fom.coupling_eff

        return cls(
            gangue_power_fraction=float(gangue_frac),
            target_power_fraction=float(target_frac),
            structural_power_fraction=float(struct_frac),
            coupling_eff=float(fom.coupling_eff),
            pec_loss_fraction=float(fom.pec_loss_fraction),
            specific_energy_proxy_kwh_per_t=energy_kwh_t,
            charge_tonnes_proxy=tonnes,
            throughput_proxy_t_per_h=throughput_t_h,
            delivered_kw_proxy=delivered_kw,
        )

    def passes_gangue_budget(self, max_gangue_fraction: float) -> bool:
        return self.gangue_power_fraction <= max_gangue_fraction

    def to_dict(self) -> dict:
        d = {
            "gangue_power_fraction": self.gangue_power_fraction,
            "target_power_fraction": self.target_power_fraction,
            "structural_power_fraction": self.structural_power_fraction,
            "coupling_eff": self.coupling_eff,
            "pec_loss_fraction": self.pec_loss_fraction,
        }
        if self.specific_energy_proxy_kwh_per_t is not None:
            d["specific_energy_proxy_kwh_per_t"] = self.specific_energy_proxy_kwh_per_t
        if self.charge_tonnes_proxy is not None:
            d["charge_tonnes_proxy"] = self.charge_tonnes_proxy
        if self.throughput_proxy_t_per_h is not None:
            d["throughput_proxy_t_per_h"] = self.throughput_proxy_t_per_h
        if self.delivered_kw_proxy is not None:
            d["delivered_kw_proxy"] = self.delivered_kw_proxy
        return d
