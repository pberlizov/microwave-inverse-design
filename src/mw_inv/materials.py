"""Measured microwave dielectric properties of minerals -- with citations.

Complex relative permittivity convention: ``eps = eps' + i*eps''`` with ``eps'' > 0``
for a lossy (absorbing) medium and time dependence e^{-i omega t}. All values are at or
near 2.45 GHz, room temperature, unless noted.

These are REAL literature values, not placeholders. They still carry meaningful
uncertainty: mineral permittivity depends strongly on grain size, porosity, bulk vs.
disseminated form, purity, and temperature (loss generally rises with T, which is what
drives thermal runaway). Treat the numbers as representative, not exact for a given ore.
Each entry cites where it comes from; see docs/MATERIALS.md for the full discussion.

Key qualitative frame (Chen et al.; Standish & Worner; reproduced in Kingman reviews):
  - "Hyperactive" microwave absorbers: Fe3O4 (magnetite), FeS2 (pyrite), MoS2, UO2
  - "Inactive" (transparent gangue):  SiO2 (quartz), CaCO3 (calcite), CaO
The best microwave-assisted-liberation response (Kingman 2000) comes from good
absorbers finely dispersed in a transparent gangue -- exactly the target/gangue split
modelled here, and the pyrite-in-calcite system Salsman (1996) simulated for stress.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Mineral:
    name: str
    eps: complex          # eps' + i eps''  (eps'' > 0 = lossy)
    note: str
    source: str


# --- Strong microwave absorbers (candidate "target" phases) ---

MAGNETITE = Mineral(
    name="magnetite (Fe3O4)",
    eps=12.0 + 1.0j,
    note="~12 - j1 at 2.45 GHz; also magnetic (mu' ~ 1.5). Penetration depth ~80 um.",
    source="cavity-perturbation studies; consistent with 2.45 GHz values reported in "
           "magnetite/maghemite microwave permittivity literature",
)

PYRITE_DISSEMINATED = Mineral(
    name="pyrite (FeS2), finely disseminated",
    eps=8.0 + 0.3j,
    note="Measured 8 - j0.3 at 2.45 GHz; authors note this is LOWER than bulk-pyrite "
         "literature, attributed to small dispersed grain size.",
    source="Near-field scanning microwave microscope study of rock-forming minerals, "
           "Sensors 22(3):1138 (MDPI, 2022), PMC8840724",
)

# --- Transparent / low-loss gangue (candidate "gangue" phases) ---

QUARTZ = Mineral(
    name="quartz (SiO2)",
    eps=4.6 + 0.0005j,
    note="eps' ~ 4.6, tan(delta) ~ 1e-4 -> eps'' ~ 5e-4. Essentially microwave "
         "transparent ('inactive').",
    source="widely established; eps' = 4.6 cited in Sensors 22(3):1138 (MDPI, 2022)",
)

CALCITE = Mineral(
    name="calcite (CaCO3)",
    eps=8.5 + 0.05j,
    note="eps' ~ 8-9 (near twice quartz); low loss ('inactive'). eps'' taken small/"
         "representative -- carbonate loss is low but less precisely reported than quartz.",
    source="eps' ~ 8-9 cited in Sensors 22(3):1138 (MDPI, 2022); classed 'inactive' in "
           "Chen/Standish heating-response tables",
)


@dataclass(frozen=True)
class MaterialPair:
    """A target/gangue pairing used to build a scene."""

    target: complex
    gangue: complex
    background: complex
    label: str
    provenance: str


# Headline pair: strong absorber in transparent gangue (largest, best-supported
# loss contrast -- the Kingman "best response" regime).
MAGNETITE_IN_QUARTZ = MaterialPair(
    target=MAGNETITE.eps,
    gangue=QUARTZ.eps,
    background=1.0 + 0.0j,
    label="magnetite_in_quartz",
    provenance="magnetite ~12-j1 (cavity perturbation); quartz 4.6-j5e-4 (MDPI 2022)",
)

# Canonical thermally-assisted-liberation system Salsman (1996) modelled.
# Lower loss-contrast (and nearly matched eps') -- a deliberately harder, literature-
# faithful case.
PYRITE_IN_CALCITE = MaterialPair(
    target=PYRITE_DISSEMINATED.eps,
    gangue=CALCITE.eps,
    background=1.0 + 0.0j,
    label="pyrite_in_calcite",
    provenance="pyrite 8-j0.3 (MDPI 2022, disseminated); calcite ~8.5 low-loss",
)

PAIRS: dict[str, MaterialPair] = {
    MAGNETITE_IN_QUARTZ.label: MAGNETITE_IN_QUARTZ,
    PYRITE_IN_CALCITE.label: PYRITE_IN_CALCITE,
}

DEFAULT_PAIR = MAGNETITE_IN_QUARTZ
