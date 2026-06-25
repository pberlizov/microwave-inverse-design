"""Primary-literature dielectric/permeability data and ε(T, f) models.

Room-temperature values used in scenes come from cited measurements (MDPI 2022
near-field microscope for disseminated grains; Hotta et al. 2009 for magnetite
powder μ at 2.45 GHz). Temperature and frequency dependence is piecewise-linear
in T between tabulated anchors, with sources noted per anchor.

Bulk ore-mineral polynomials from Polyakova et al. (PIER B 25, 2010) are
provided for cross-validation only — they describe pressed mineral plates at
12–145 GHz and do *not* replace the disseminated-scene values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Tabulated anchors: (T_K, freq_hz, eps_real, eps_imag, citation snippet)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DielectricAnchor:
    temp_K: float
    freq_hz: float
    eps_real: float
    eps_imag: float
    source: str


@dataclass(frozen=True)
class PermeabilityAnchor:
    temp_K: float
    freq_hz: float
    mu_real: float
    mu_imag: float
    source: str


@dataclass(frozen=True)
class MineralModel:
    """Temperature/frequency model for one mineral phase."""

    name: str
    eps_anchors: tuple[DielectricAnchor, ...]
    mu_anchors: tuple[PermeabilityAnchor, ...] = ()
    # Inert gangue: eps'' fixed, eps' weakly T-dependent.
    ramps_with_T: bool = True

    def eps(self, temp_K: float, freq_hz: float = 2.45e9) -> complex:
        er = _interp_scalar(self.eps_anchors, temp_K, freq_hz, "eps_real")
        ei = _interp_scalar(self.eps_anchors, temp_K, freq_hz, "eps_imag")
        return complex(er, ei)

    def mu(self, temp_K: float, freq_hz: float = 2.45e9) -> complex:
        if not self.mu_anchors:
            return 1.0 + 0.0j
        mr = _interp_scalar(self.mu_anchors, temp_K, freq_hz, "mu_real")
        mi = _interp_scalar(self.mu_anchors, temp_K, freq_hz, "mu_imag")
        return complex(mr, mi)


def _interp_scalar(
    anchors: tuple[DielectricAnchor, ...] | tuple[PermeabilityAnchor, ...],
    temp_K: float,
    freq_hz: float,
    field: str,
) -> float:
    """Bilinear-style: interpolate in T at the nearest tabulated frequency."""
    if not anchors:
        return 1.0 if field.endswith("real") and field.startswith("mu") else 0.0
    freqs = sorted({a.freq_hz for a in anchors})
    f_near = min(freqs, key=lambda f: abs(f - freq_hz))
    pts = [a for a in anchors if a.freq_hz == f_near]
    pts = sorted(pts, key=lambda a: a.temp_K)
    temps = np.array([a.temp_K for a in pts], dtype=float)
    vals = np.array([getattr(a, field) for a in pts], dtype=float)
    return float(np.interp(temp_K, temps, vals))


# --- Pyrite (disseminated), 2.45 GHz -----------------------------------------
# RT anchor: MDPI Sensors 22(3):1138 (2022), PMC8840724 — disseminated grains.
# Higher-T anchors: trend from Peng et al. / Pickles cavity-perturbation pyrite
# heating curves at 2.45 GHz (J. Korean Soc. Mineral & Energy Res. 2011), scaled
# to preserve the disseminated RT point while capturing Cumbane (2008) qualitative
# finding that ε′ and ε″ rise strongly with T for pyrite/chalcopyrite.

PYRITE = MineralModel(
    name="pyrite (FeS2), disseminated",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 8.0, 0.30,
                         "MDPI Sensors 22(3):1138 (2022) disseminated @ 2.45 GHz"),
        DielectricAnchor(573.0, 2.45e9, 8.3, 0.42,
                         "Peng et al. 2011 pyrite ε(T) trend @ 2.45 GHz (~300°C)"),
        DielectricAnchor(773.0, 2.45e9, 8.8, 0.55,
                         "Peng et al. 2011 pre-phase-transform rise @ 2.45 GHz"),
        DielectricAnchor(973.0, 2.45e9, 9.5, 0.85,
                         "Cumbane 2008 qualitative strong ε″ rise toward 650°C"),
        DielectricAnchor(1273.0, 2.45e9, 10.0, 1.20,
                         "loss-tangent ~1 ceiling (Kingman-review magnitude)"),
    ),
    mu_anchors=(
        PermeabilityAnchor(298.0, 2.45e9, 1.0, 0.05,
                           "Peng et al. 2011: weak μ″ contribution @ 2.45 GHz"),
    ),
)

# --- Magnetite ----------------------------------------------------------------
# ε: MDPI 2022 aggregated ~12-j1 @ 2.45 GHz; ε(T) from Hotta et al. ISIJ Int.
# 51(3) 2011 high-temperature coaxial measurements (38–62 μm powder, dr≈63%).
# μ: Hotta et al. ISIJ Int. 49(9) 2009 — ferromagnetic-resonance peak near 2.45 GHz.

MAGNETITE = MineralModel(
    name="magnetite (Fe3O4)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 12.0, 1.0,
                         "MDPI Sensors 22(3):1138 (2022) @ 2.45 GHz"),
        DielectricAnchor(573.0, 2.45e9, 12.5, 1.4,
                         "Hotta et al. 2011 ε″(T) rise @ 2.45 GHz (~300°C)"),
        DielectricAnchor(773.0, 2.45e9, 14.0, 2.5,
                         "Hotta et al. 2011 abrupt ε″ rise above ~400°C"),
        DielectricAnchor(848.0, 2.45e9, 13.0, 2.0,
                         "Hotta et al. 2011 ε′ peak ~450–500°C @ 2.45 GHz"),
    ),
    mu_anchors=(
        PermeabilityAnchor(298.0, 2.45e9, 1.30, 0.55,
                           "Hotta et al. 2009 powder Fe3O4 near μ″ peak @ 2.45 GHz"),
        PermeabilityAnchor(573.0, 2.45e9, 1.15, 0.65,
                           "Hotta et al. 2011 μ″ rises to ~500°C below 3.5 GHz"),
        PermeabilityAnchor(773.0, 2.45e9, 1.05, 0.35,
                           "Hotta et al. 2011 μ′→1 approaching Curie (~580°C)"),
    ),
)

# --- Gangue: quartz & calcite (low-loss, weak T dependence) -------------------
# Campbell & Davidson, CANMET Report 1989 / Church-Webb-Salsman low-loss tables;
# MDPI 2022 for ε′ at 2.45 GHz.

QUARTZ = MineralModel(
    name="quartz (SiO2)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 4.6, 5.0e-4,
                         "MDPI 2022; tan δ~1e-4 (Church/Webb low-loss)"),
        DielectricAnchor(1273.0, 2.45e9, 4.6, 6.0e-4,
                         "inactive gangue — ε″ remains negligible (Chen/Standish)"),
    ),
    ramps_with_T=False,
)

CALCITE = MineralModel(
    name="calcite (CaCO3)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 8.5, 0.05,
                         "MDPI 2022 ε′~8–9; low carbonate loss"),
        DielectricAnchor(1273.0, 2.45e9, 8.5, 0.06,
                         "inactive gangue — weak T dependence (Cumbane: carbonates stable)"),
    ),
    ramps_with_T=False,
)

# --- HMAP sulphides & oxides (Goldbaum 2022 / Cumbane 2008 / Peng 2013) --------
# Scene-scale values at 2.45 GHz — disseminated or pressed-powder anchors, *not*
# Polyakova bulk plates. Impure/natural samples often sit below bulk references
# (Goldbaum thesis Ch. 4; Wang et al. 2018 conductivity–purity correlation).

CHALCOPYRITE = MineralModel(
    name="chalcopyrite (CuFeS2)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 10.0, 0.45,
                         "Impure CuFeS2; conductivity > pyrite (Ferrari-John 2016); "
                         "Cumbane/Lovas ε(T) @ ~2 GHz"),
        DielectricAnchor(573.0, 2.45e9, 10.5, 0.65,
                         "Lovas 2010 / Cumbane 2008 rise toward 400°C"),
        DielectricAnchor(773.0, 2.45e9, 11.5, 0.95,
                         "Lovas hematite transition band ~450–650°C"),
        DielectricAnchor(973.0, 2.45e9, 12.0, 1.10,
                         "oxidation products — high-loss ceiling"),
    ),
)

PYRRHOTITE = MineralModel(
    name="pyrrhotite (Fe1-xS)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 11.0, 0.65,
                         "Peng 2013 Fe1-xS high ε″ below 200°C @ 2.45 GHz"),
        DielectricAnchor(473.0, 2.45e9, 12.0, 0.85,
                         "Goldbaum et al. 2020 pyrrhotite ε(T) @ 912 MHz trend"),
        DielectricAnchor(773.0, 2.45e9, 13.0, 1.20,
                         "sulphide melting band — high loss (Goldbaum Ch. 4)"),
    ),
    mu_anchors=(
        PermeabilityAnchor(298.0, 2.45e9, 1.05, 0.15,
                           "monoclinic pyrrhotite — weak ferromagnetic channel"),
    ),
)

GALENA = MineralModel(
    name="galena (PbS)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 9.5, 0.35,
                         "Cumbane 2008 stable to ~500°C; moderate HMAP (Goldbaum Fig 4-2)"),
        DielectricAnchor(573.0, 2.45e9, 9.8, 0.40,
                         "weak T rise to 400°C (Galena PbS Table 3 IntechOpen)"),
        DielectricAnchor(773.0, 2.45e9, 10.5, 0.55,
                         "Lovas 2010 PbSO4 oxidation band above ~650°C"),
    ),
    ramps_with_T=False,
)

MOLYBDENITE = MineralModel(
    name="molybdenite (MoS2)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 7.5, 1.0,
                         "Chen/Standish hyperactive MoS2; high ε″ HMAP (Harrison class I)"),
        DielectricAnchor(573.0, 2.45e9, 8.0, 1.3,
                         "Goldbaum 94.6% MoS2 — orders above concentrate (Ch. 4)"),
        DielectricAnchor(773.0, 2.45e9, 8.5, 1.6,
                         "high-loss sulphide — pilot arcing risk at high wt.%"),
    ),
)

BORNITE = MineralModel(
    name="bornite (Cu5FeS4)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 10.5, 0.55,
                         "Harrison 1997 good-MW heater; Goldbaum HMAP list"),
        DielectricAnchor(573.0, 2.45e9, 11.0, 0.75,
                         "sulphide ε(T) rise — qualitative (Cumbane family)"),
        DielectricAnchor(773.0, 2.45e9, 11.5, 0.95,
                         "pre-decomposition high-loss band"),
    ),
)

PENTLANDITE = MineralModel(
    name="pentlandite ((Fe,Ni)9S8)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 10.0, 0.50,
                         "Goldbaum et al. 2020 novel pentlandite ε(T) @ 912 MHz"),
        DielectricAnchor(423.0, 2.45e9, 10.5, 0.85,
                         "S dissociation rise ~150°C (Goldbaum Fig 4-1)"),
        DielectricAnchor(623.0, 2.45e9, 11.0, 1.10,
                         "liquefaction decline above ~350°C — anchor pre-melt"),
    ),
)

HEMATITE = MineralModel(
    name="hematite (Fe2O3)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 11.0, 0.40,
                         "Nelson/Blake 1989 oxide; Zheng 2020 medium–high loss class"),
        DielectricAnchor(573.0, 2.45e9, 12.0, 0.65,
                         "Pickles goethite→hematite dehydroxylation ε rise @ ~2.45 GHz"),
        DielectricAnchor(773.0, 2.45e9, 13.0, 0.90,
                         "high-T oxide — active absorber (Chen class II)"),
    ),
)

# --- Additional gangue (MDPI 2022 / Goldbaum Ch. 4 silicates) ----------------

DOLOMITE = MineralModel(
    name="dolomite (CaMg(CO3)2)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 6.8, 0.04,
                         "carbonate gangue — inactive (Pipe/OK ores, Goldbaum Ch. 5)"),
        DielectricAnchor(1273.0, 2.45e9, 6.8, 0.05,
                         "weak T dependence — carbonate family"),
    ),
    ramps_with_T=False,
)

FELDSPAR = MineralModel(
    name="plagioclase feldspar",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 6.0, 0.09,
                         "MDPI 2022 Table 1 labradorite 6.01-j0.09 @ 2.45 GHz"),
        DielectricAnchor(1273.0, 2.45e9, 6.0, 0.10,
                         "inactive silicate gangue"),
    ),
    ramps_with_T=False,
)

SERPENTINE = MineralModel(
    name="serpentine (lizardite)",
    eps_anchors=(
        DielectricAnchor(298.0, 2.45e9, 5.2, 0.02,
                         "Goldbaum Ch. 4 ultramafic silicate — low loss @ 912 MHz"),
        DielectricAnchor(773.0, 2.45e9, 5.5, 0.08,
                         "dehydroxylation onset — weak rise toward olivine"),
    ),
    ramps_with_T=False,
)


MINERAL_MODELS: dict[str, MineralModel] = {
    "pyrite": PYRITE,
    "magnetite": MAGNETITE,
    "quartz": QUARTZ,
    "calcite": CALCITE,
    "chalcopyrite": CHALCOPYRITE,
    "pyrrhotite": PYRRHOTITE,
    "galena": GALENA,
    "molybdenite": MOLYBDENITE,
    "bornite": BORNITE,
    "pentlandite": PENTLANDITE,
    "hematite": HEMATITE,
    "dolomite": DOLOMITE,
    "feldspar": FELDSPAR,
    "serpentine": SERPENTINE,
}


def mineral_eps(
    name: str,
    temp_K: float = 298.0,
    freq_hz: float = 2.45e9,
) -> complex:
    """Look up ε(T, f) for a catalog mineral key."""
    return MINERAL_MODELS[name].eps(temp_K, freq_hz)


def mineral_mu(
    name: str,
    temp_K: float = 298.0,
    freq_hz: float = 2.45e9,
) -> complex:
    """Look up μ(T, f) for a catalog mineral key."""
    return MINERAL_MODELS[name].mu(temp_K, freq_hz)


PAIR_MINERALS: dict[str, tuple[str, str]] = {
    "magnetite_in_quartz": ("magnetite", "quartz"),
    "pyrite_in_calcite": ("pyrite", "calcite"),
    "chalcopyrite_in_calcite": ("chalcopyrite", "calcite"),
    "pyrrhotite_in_quartz": ("pyrrhotite", "quartz"),
    "galena_in_calcite": ("galena", "calcite"),
    "molybdenite_in_quartz": ("molybdenite", "quartz"),
}

# Default gangue mineral when ore fractions do not specify silicate/carbonate matrix.
DEFAULT_GANGUE_MINERAL = "calcite"


# ---------------------------------------------------------------------------
# Legacy Arrhenius ramp (kept for sweeps thermal lumped model compatibility)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EpsTModel:
    """Arrhenius-like ramp of ε″ with temperature cap — parametric fallback."""

    eps_real: float
    eps_imag_ref: float
    T_ref_K: float = 298.0
    activation_K: float = 1000.0
    max_loss_tangent: float = 0.6
    ramps_with_T: bool = True

    def eps_imag(self, T_K: float) -> float:
        if not self.ramps_with_T:
            return self.eps_imag_ref
        factor = np.exp(self.activation_K * (1.0 / self.T_ref_K - 1.0 / T_K))
        e_imag = self.eps_imag_ref * factor
        return float(min(e_imag, self.max_loss_tangent * self.eps_real))

    def eps(self, T_K: float) -> complex:
        return self.eps_real + 1j * self.eps_imag(T_K)

    @classmethod
    def from_mineral(cls, mineral: MineralModel, T_ref_K: float = 298.0) -> "EpsTModel":
        """Build a parametric ramp anchored to the mineral table at T_ref."""
        eps_ref = mineral.eps(T_ref_K)
        return cls(
            eps_real=eps_ref.real,
            eps_imag_ref=eps_ref.imag,
            T_ref_K=T_ref_K,
            ramps_with_T=mineral.ramps_with_T,
        )


# ---------------------------------------------------------------------------
# Polyakova et al. 2010 bulk mineral polynomials (validation cross-check)
# ---------------------------------------------------------------------------

def _horner(coeffs: tuple[float, ...], f: float) -> float:
    """Evaluate a0 + a1*f + a2*f^2 + ... (coeffs low-to-high degree)."""
    v = 0.0
    for c in reversed(coeffs):
        v = v * f + c
    return v


_POLYAKOVA_N0 = {
    "magnetite": (
        32.02744215, -0.5259548602, -0.06491391718, 3.608474987e-3,
        -7.125902056e-5, 6.751320507e-7, -3.113051295e-9, 5.632545142e-12,
    ),
    "pyrite": (
        45.10123824, 0.4750837566, -0.2770431487, 1.360410159e-2,
        -2.920310577e-4, 3.327239518e-6, -2.093060363e-8, 6.862839841e-11,
        -9.151452873e-14,
    ),
}

_POLYAKOVA_N00 = {
    "magnetite": (
        2.520151324, -0.3127286757, 0.01875754576, -5.916173864e-4,
        1.067064015e-5, -1.144566958e-7, 7.234628417e-10, -2.491158837e-12,
        3.606967497e-15,
    ),
    "pyrite": (
        0.7031046195, 1.990480306e-2, -1.619627954e-3, 3.351891491e-5,
        -3.167624788e-7, 1.430090796e-9, -2.501858921e-12,
    ),
}


def polyakova_bulk_eps(mineral: str, freq_hz: float) -> complex:
    """Complex ε for *bulk* pressed mineral plates (Polyakova et al. 2010).

    Valid 12–145 GHz. Values at 2.45 GHz are orders of magnitude above
    disseminated-grain measurements — use only as an independent reference."""
    f = freq_hz / 1e9
    if mineral not in _POLYAKOVA_N0:
        raise KeyError(f"no Polyakova data for {mineral!r}")
    n = complex(_horner(_POLYAKOVA_N0[mineral], f), _horner(_POLYAKOVA_N00[mineral], f))
    return n * n
