"""Cited saline/gel phantom dielectric models at 2.45 GHz.

Salt-in-agar phantoms follow the Gabriel (1996) / industrial saline-gel scaling:
ε′ rises and ε″ grows strongly with NaCl weight fraction.  Anchor points are
order-of-magnitude consistent with Lazebnik et al. tissue-mimicking gels and
typical bench recipes (0–3 wt% NaCl in agar/glycerol at 2.45 GHz).

Use ``saline_eps(wt_percent)`` for recipe-linked permittivity — not hand-picked constants.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (NaCl wt%, eps_real, eps_imag) @ ~2.45 GHz — monotonic with salinity
_SALINE_ANCHORS: tuple[tuple[float, float, float], ...] = (
    (0.0, 4.5, 0.06),    # low-salt agar / deionised gel
    (0.5, 7.0, 0.35),
    (1.0, 10.5, 1.2),
    (2.0, 16.0, 3.5),
    (3.0, 22.0, 7.0),    # high-salt target mimic
)

# Thermal: water-dominated gel (~80% water) — J/m³/K
GEL_RHO_CP = 4.2e6
GEL_K_W_MK = 0.55


def saline_eps(salt_wt_percent: float) -> complex:
    """Interpolate complex ε for saline agar at 2.45 GHz."""
    w = float(np.clip(salt_wt_percent, 0.0, 3.5))
    pts = np.array(_SALINE_ANCHORS)
    er = float(np.interp(w, pts[:, 0], pts[:, 1]))
    ei = float(np.interp(w, pts[:, 0], pts[:, 2]))
    return complex(er, ei)


@dataclass(frozen=True)
class GelRecipe:
    label: str
    salt_wt_percent: float
    mineral_analog: str
    provenance: str

    @property
    def eps(self) -> complex:
        return saline_eps(self.salt_wt_percent)


@dataclass(frozen=True)
class PhantomRecipePair:
    label: str
    target: GelRecipe
    gangue: GelRecipe
    mineral_analog: str


PHANTOM_RECIPES: dict[str, PhantomRecipePair] = {
    "saline_3_vs_0": PhantomRecipePair(
        label="saline_3_vs_0",
        target=GelRecipe("high_salt_3pct", 3.0, "magnetite_in_quartz",
                         "3 wt% NaCl agar @ 2.45 GHz (Gabriel-scaled anchors)"),
        gangue=GelRecipe("low_salt_0pct", 0.0, "transparent gangue",
                         "0 wt% NaCl agar @ 2.45 GHz"),
        mineral_analog="strong contrast",
    ),
    "saline_2_vs_0.5": PhantomRecipePair(
        label="saline_2_vs_0.5",
        target=GelRecipe("salt_2pct", 2.0, "pyrite_in_calcite",
                         "2 wt% NaCl — matched ε′ trend, loss contrast"),
        gangue=GelRecipe("salt_0.5pct", 0.5, "pyrite_in_calcite gangue",
                         "0.5 wt% NaCl — low loss gangue mimic"),
        mineral_analog="pyrite_in_calcite (matched ε′ regime)",
    ),
    "saline_1_vs_0": PhantomRecipePair(
        label="saline_1_vs_0",
        target=GelRecipe("salt_1pct", 1.0, "intermediate",
                         "1 wt% NaCl"),
        gangue=GelRecipe("salt_0pct", 0.0, "intermediate",
                         "0 wt% NaCl"),
        mineral_analog="moderate contrast sanity check",
    ),
}
