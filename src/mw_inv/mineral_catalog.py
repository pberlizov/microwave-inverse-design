"""Microwave-amenable mineral catalog and heating-response classification.

Anchors: Goldbaum/Forster HMAP list (IMPC/CEEComm 2022); Chen/Standish/Harrison
heating tiers; MDPI Sensors 22(3):1138 disseminated ε; Cumbane 2008 sulphide ε(T);
Goldbaum thesis Ch. 4 cavity-perturbation curves @ 912–2466 MHz.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mw_inv.dielectric_data import MINERAL_MODELS, MineralModel, mineral_eps
from mw_inv.ore_profiles import HMAP_MINERALS

__all__ = [
    "MicrowaveClass",
    "MineralEntry",
    "CATALOG",
    "catalog_entry",
    "hmap_minerals",
    "inactive_gangue",
    "loss_contrast",
    "microwave_class",
]


class MicrowaveClass(str, Enum):
    """Heating-response tier (Chen / Standish / Harrison / Goldbaum synthesis)."""

    HMAP = "hmap"           # highly microwave-amenable (Goldbaum list)
    ACTIVE = "active"       # heats well but not primary HMAP target
    INACTIVE = "inactive"   # transparent gangue (ε″ ≲ 0.1)


@dataclass(frozen=True)
class MineralEntry:
    key: str
    model: MineralModel
    mw_class: MicrowaveClass
    citation: str

    def eps(self, temp_K: float = 298.0, freq_hz: float = 2.45e9) -> complex:
        return self.model.eps(temp_K, freq_hz)

    def mu(self, temp_K: float = 298.0, freq_hz: float = 2.45e9) -> complex:
        return self.model.mu(temp_K, freq_hz)


def _entry(key: str, mw_class: MicrowaveClass, citation: str) -> MineralEntry:
    return MineralEntry(key=key, model=MINERAL_MODELS[key], mw_class=mw_class, citation=citation)


CATALOG: dict[str, MineralEntry] = {
    # --- Goldbaum HMAPs ---
    "bornite": _entry("bornite", MicrowaveClass.HMAP,
                      "Goldbaum HMAP; Harrison good-MW heater"),
    "chalcopyrite": _entry("chalcopyrite", MicrowaveClass.HMAP,
                           "Goldbaum HMAP; Cumbane/Lovas ε(T) sulphide"),
    "galena": _entry("galena", MicrowaveClass.HMAP,
                     "Goldbaum HMAP; stable ε to 500°C (Cumbane)"),
    "hematite": _entry("hematite", MicrowaveClass.HMAP,
                       "Goldbaum HMAP; Nelson/Blake 1989 oxide"),
    "magnetite": _entry("magnetite", MicrowaveClass.HMAP,
                        "MDPI 2022 + Hotta μ, ε(T) @ 2.45 GHz"),
    "molybdenite": _entry("molybdenite", MicrowaveClass.HMAP,
                          "Chen hyperactive MoS2; high ε″"),
    "pentlandite": _entry("pentlandite", MicrowaveClass.HMAP,
                          "Goldbaum et al. 2020 ε(T) Ni-Cu ores"),
    "pyrite": _entry("pyrite", MicrowaveClass.HMAP,
                     "MDPI 2022 disseminated 8-j0.3; Peng ε(T)"),
    "pyrrhotite": _entry("pyrrhotite", MicrowaveClass.HMAP,
                         "Peng 2013 high ε″ below 200°C; magnetic"),
    # --- Gangue / matrix ---
    "quartz": _entry("quartz", MicrowaveClass.INACTIVE,
                     "MDPI 2022; Church/Webb inactive silicate"),
    "calcite": _entry("calcite", MicrowaveClass.INACTIVE,
                      "MDPI 2022; carbonate gangue"),
    "dolomite": _entry("dolomite", MicrowaveClass.INACTIVE,
                       "Carbonate gangue — Pipe/OK ultramafic ores"),
    "feldspar": _entry("feldspar", MicrowaveClass.INACTIVE,
                       "MDPI 2022 plagioclase reference"),
    "serpentine": _entry("serpentine", MicrowaveClass.INACTIVE,
                         "Goldbaum Ch. 4 ultramafic silicate"),
}


def catalog_entry(name: str) -> MineralEntry:
    if name not in CATALOG:
        raise KeyError(f"unknown catalog mineral {name!r}; keys: {sorted(CATALOG)}")
    return CATALOG[name]


def hmap_minerals() -> tuple[str, ...]:
    return HMAP_MINERALS


def inactive_gangue() -> tuple[str, ...]:
    return tuple(k for k, e in CATALOG.items() if e.mw_class == MicrowaveClass.INACTIVE)


def microwave_class(name: str) -> MicrowaveClass:
    return catalog_entry(name).mw_class


def loss_contrast(
    target: str,
    gangue: str,
    *,
    temp_K: float = 298.0,
    freq_hz: float = 2.45e9,
) -> float:
    """ε″ ratio target/gangue — primary selective-heating figure of merit."""
    t = mineral_eps(target, temp_K, freq_hz).imag
    g = max(mineral_eps(gangue, temp_K, freq_hz).imag, 1e-12)
    return float(t / g)
