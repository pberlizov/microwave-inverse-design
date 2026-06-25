"""Measured microwave dielectric properties of minerals -- with citations.

Complex relative permittivity convention: ``eps = eps' + i*eps''`` with ``eps'' > 0``
for a lossy (absorbing) medium and time dependence e^{-i omega t}. Room-temperature
values come from cited measurements; ``Materials.from_pair(..., target_T_K=...)`` uses
tabulated ε(T, f) anchors in ``mw_inv.dielectric_data`` (see docs/MATERIALS.md).

Key qualitative frame (Chen et al.; Standish & Worner; reproduced in Kingman reviews):
  - "Hyperactive" microwave absorbers: Fe3O4 (magnetite), FeS2 (pyrite), MoS2, UO2
  - "Inactive" (transparent gangue):  SiO2 (quartz), CaCO3 (calcite), CaO
"""

from __future__ import annotations

from dataclasses import dataclass

from mw_inv.dielectric_data import (
    CALCITE,
    MAGNETITE,
    MINERAL_MODELS,
    PYRITE,
    QUARTZ,
    DielectricAnchor,
    EpsTModel,
    MineralModel,
    PAIR_MINERALS,
)

__all__ = [
    "DielectricAnchor",
    "EpsTModel",
    "MAGNETITE",
    "MAGNETITE_IN_QUARTZ",
    "CHALCOPYRITE_IN_CALCITE",
    "GALENA_IN_CALCITE",
    "MOLYBDENITE_IN_QUARTZ",
    "Mineral",
    "MineralModel",
    "MaterialPair",
    "Materials",
    "PAIRS",
    "PAIR_LABELS",
    "PYRITE",
    "PYRITE_IN_CALCITE",
    "PYRRHOTITE_IN_QUARTZ",
    "QUARTZ",
    "CALCITE",
    "DEFAULT_PAIR",
]


@dataclass(frozen=True)
class Mineral:
    name: str
    eps: complex
    note: str
    source: str
    model: MineralModel | None = None
    mu: complex = 1.0 + 0.0j


MAGNETITE_MINERAL = Mineral(
    name=MAGNETITE.name,
    eps=MAGNETITE.eps(298.0),
    mu=MAGNETITE.mu(298.0),
    note="~12-j1, μ~1.3-j0.55 at 2.45 GHz; magnetic loss channel included.",
    source="MDPI 2022 ε; Hotta et al. ISIJ Int. 49(9)/51(3) μ, ε(T) @ 2.45 GHz",
    model=MAGNETITE,
)

PYRITE_DISSEMINATED = Mineral(
    name=PYRITE.name,
    eps=PYRITE.eps(298.0),
    mu=PYRITE.mu(298.0),
    note="Measured 8-j0.3 at 2.45 GHz disseminated grains; ε rises with T (Cumbane/Peng).",
    source="MDPI Sensors 22(3):1138 (2022); ε(T) anchors Peng/Cumbane @ 2.45 GHz",
    model=PYRITE,
)

QUARTZ_MINERAL = Mineral(
    name=QUARTZ.name,
    eps=QUARTZ.eps(298.0),
    note="Essentially microwave transparent ('inactive').",
    source="MDPI 2022; Church/Webb low-loss tables",
    model=QUARTZ,
)

CALCITE_MINERAL = Mineral(
    name=CALCITE.name,
    eps=CALCITE.eps(298.0),
    note="Low-loss carbonate gangue; ε′~8–9.",
    source="MDPI 2022; inactive in Chen/Standish heating tables",
    model=CALCITE,
)


def _mineral_from_model(key: str, *, note: str, source: str) -> Mineral:
    model = MINERAL_MODELS[key]
    return Mineral(
        name=model.name,
        eps=model.eps(298.0),
        mu=model.mu(298.0),
        note=note,
        source=source,
        model=model,
    )


CHALCOPYRITE_MINERAL = _mineral_from_model(
    "chalcopyrite",
    note="HMAP Cu sulphide; ε″ > disseminated pyrite at RT (impure ore).",
    source="Goldbaum HMAP; Cumbane/Lovas ε(T) @ ~2 GHz",
)
PYRRHOTITE_MINERAL = _mineral_from_model(
    "pyrrhotite",
    note="HMAP Fe sulphide; high ε″ below 200°C (Peng 2013).",
    source="Goldbaum HMAP; Peng 2013 @ 2.45 GHz",
)
GALENA_MINERAL = _mineral_from_model(
    "galena",
    note="HMAP Pb sulphide; moderate ε″ stable to ~500°C.",
    source="Goldbaum HMAP; Cumbane 2008",
)
MOLYBDENITE_MINERAL = _mineral_from_model(
    "molybdenite",
    note="Hyperactive MoS2; high ε″ — arcing risk in massive ore.",
    source="Chen/Standish hyperactive; Goldbaum HMAP",
)


def _pair_from_keys(label: str, target_key: str, gangue_key: str, provenance: str) -> MaterialPair:
    t = _mineral_from_model(
        target_key,
        note=MINERAL_MODELS[target_key].name,
        source=provenance,
    )
    g = _mineral_from_model(
        gangue_key,
        note=MINERAL_MODELS[gangue_key].name,
        source=provenance,
    )
    return MaterialPair(
        target=t.eps,
        gangue=g.eps,
        background=1.0 + 0.0j,
        label=label,
        provenance=provenance,
        target_mu=t.mu,
        gangue_mu=g.mu,
        target_model=t.model,
        gangue_model=g.model,
    )


@dataclass(frozen=True)
class MaterialPair:
    target: complex
    gangue: complex
    background: complex
    label: str
    provenance: str
    target_mu: complex = 1.0 + 0.0j
    gangue_mu: complex = 1.0 + 0.0j
    target_model: MineralModel | None = None
    gangue_model: MineralModel | None = None


MAGNETITE_IN_QUARTZ = MaterialPair(
    target=MAGNETITE_MINERAL.eps,
    gangue=QUARTZ_MINERAL.eps,
    background=1.0 + 0.0j,
    label="magnetite_in_quartz",
    provenance="magnetite ~12-j1 + μ″ (Hotta 2009); quartz 4.6-j5e-4 (MDPI 2022)",
    target_mu=MAGNETITE_MINERAL.mu,
    gangue_mu=1.0 + 0.0j,
    target_model=MAGNETITE,
    gangue_model=QUARTZ,
)

PYRITE_IN_CALCITE = MaterialPair(
    target=PYRITE_DISSEMINATED.eps,
    gangue=CALCITE_MINERAL.eps,
    background=1.0 + 0.0j,
    label="pyrite_in_calcite",
    provenance="pyrite 8-j0.3 + ε(T) (MDPI/Peng); calcite ~8.5 low-loss",
    target_mu=PYRITE_DISSEMINATED.mu,
    gangue_mu=1.0 + 0.0j,
    target_model=PYRITE,
    gangue_model=CALCITE,
)

CHALCOPYRITE_IN_CALCITE = _pair_from_keys(
    "chalcopyrite_in_calcite",
    "chalcopyrite",
    "calcite",
    "chalcopyrite ~10-j0.45 (HMAP); calcite carbonate gangue — matched ε′ regime",
)
PYRRHOTITE_IN_QUARTZ = _pair_from_keys(
    "pyrrhotite_in_quartz",
    "pyrrhotite",
    "quartz",
    "pyrrhotite ~11-j0.65 + μ″; quartz transparent gangue",
)
GALENA_IN_CALCITE = _pair_from_keys(
    "galena_in_calcite",
    "galena",
    "calcite",
    "galena ~9.5-j0.35 (HMAP); calcite gangue — Pb-Zn porphyry style",
)
MOLYBDENITE_IN_QUARTZ = _pair_from_keys(
    "molybdenite_in_quartz",
    "molybdenite",
    "quartz",
    "molybdenite ~7.5-j1.0 hyperactive; quartz gangue",
)

PAIRS: dict[str, MaterialPair] = {
    MAGNETITE_IN_QUARTZ.label: MAGNETITE_IN_QUARTZ,
    PYRITE_IN_CALCITE.label: PYRITE_IN_CALCITE,
    CHALCOPYRITE_IN_CALCITE.label: CHALCOPYRITE_IN_CALCITE,
    PYRRHOTITE_IN_QUARTZ.label: PYRRHOTITE_IN_QUARTZ,
    GALENA_IN_CALCITE.label: GALENA_IN_CALCITE,
    MOLYBDENITE_IN_QUARTZ.label: MOLYBDENITE_IN_QUARTZ,
}

PAIR_LABELS: tuple[str, ...] = tuple(PAIR_MINERALS.keys())

DEFAULT_PAIR = MAGNETITE_IN_QUARTZ


@dataclass(frozen=True)
class Materials:
    background: complex = DEFAULT_PAIR.background
    target: complex = DEFAULT_PAIR.target
    gangue: complex = DEFAULT_PAIR.gangue
    background_mu: complex = 1.0 + 0.0j
    target_mu: complex = DEFAULT_PAIR.target_mu
    gangue_mu: complex = DEFAULT_PAIR.gangue_mu
    target_T_K: float = 298.0
    gangue_T_K: float = 298.0
    freq_hz: float = 2.45e9
    pair_label: str | None = None

    @classmethod
    def from_pair(
        cls,
        label: str,
        *,
        target_T_K: float = 298.0,
        gangue_T_K: float = 298.0,
        freq_hz: float = 2.45e9,
    ) -> "Materials":
        p = PAIRS[label]
        t_eps, t_mu = cls._phase_at(p.target_model, p.target, p.target_mu,
                                    target_T_K, freq_hz)
        g_eps, g_mu = cls._phase_at(p.gangue_model, p.gangue, p.gangue_mu,
                                    gangue_T_K, freq_hz)
        return cls(
            background=p.background,
            target=t_eps,
            gangue=g_eps,
            target_mu=t_mu,
            gangue_mu=g_mu,
            target_T_K=target_T_K,
            gangue_T_K=gangue_T_K,
            freq_hz=freq_hz,
            pair_label=label,
        )

    @staticmethod
    def _phase_at(
        model: MineralModel | None,
        default_eps: complex,
        default_mu: complex,
        temp_K: float,
        freq_hz: float,
    ) -> tuple[complex, complex]:
        if model is None:
            return default_eps, default_mu
        return model.eps(temp_K, freq_hz), model.mu(temp_K, freq_hz)

    def eps_t_model(self, phase: str = "target") -> EpsTModel:
        """Parametric ε″(T) ramp for thermal sweeps, anchored to table at T_ref."""
        if self.pair_label is not None:
            p = PAIRS[self.pair_label]
            model = p.target_model if phase == "target" else p.gangue_model
            T_ref = self.target_T_K if phase == "target" else self.gangue_T_K
            if model is not None:
                return EpsTModel.from_mineral(model, T_ref_K=T_ref)
        eps = self.target if phase == "target" else self.gangue
        return EpsTModel(eps_real=eps.real, eps_imag_ref=eps.imag, ramps_with_T=False)
