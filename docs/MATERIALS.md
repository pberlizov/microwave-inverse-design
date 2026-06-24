# Mineral dielectric data used in this model

All values are complex relative permittivity `eps = eps' + i·eps''` (eps'' > 0 = lossy),
at/near **2.45 GHz, room temperature** unless noted. Encoded in
[src/mw_inv/materials.py](../src/mw_inv/materials.py). These are real literature values,
not placeholders — but they carry genuine uncertainty (grain size, bulk vs. disseminated
form, porosity, purity, temperature), so treat them as *representative*, not exact for a
specific ore.

## Values

| Mineral | Role | eps' | eps'' | Notes | Source |
|---|---|---|---|---|---|
| Magnetite (Fe₃O₄) | absorber | ~12 | ~1 | also magnetic (μ'~1.5); penetration depth ~80 µm | cavity-perturbation / 2.45 GHz magnetite microwave studies |
| Pyrite (FeS₂), disseminated | absorber | ~8 | ~0.3 | explicitly noted as *lower* than bulk pyrite, due to small dispersed grains | Sensors 22(3):1138, MDPI 2022 (near-field scanning microwave microscope), PMC8840724 |
| Quartz (SiO₂) | gangue | ~4.6 | ~5×10⁻⁴ | tan δ ~ 1e-4; essentially transparent ("inactive") | eps'=4.6 in Sensors 22(3):1138; widely established |
| Calcite (CaCO₃) | gangue | ~8–9 | ~0.05 | low loss ("inactive"); eps'' representative (carbonate loss less precisely reported) | eps'~8–9 in Sensors 22(3):1138; Chen/Standish heating-response tables |

### Qualitative heating-response classification
From Chen et al. / Standish & Worner, reproduced in Kingman-group reviews and the
[IntechOpen chapter](https://www.intechopen.com/chapters/40687):

- **Hyperactive** absorbers: MoS₂, UO₂, **Fe₃O₄ (magnetite)**, **FeS₂ (pyrite)**, CuCl
- **Active:** Fe₂O₃ (hematite), FeS, CuS, NiO, CuO
- **Inactive** (transparent gangue): **SiO₂ (quartz)**, **CaCO₃ (calcite)**, CaO, MgO

## The two pairs (and why both matter)

The model ships two cited `MaterialPair`s. They are not interchangeable — they probe
opposite regimes, which is the point.

### `magnetite_in_quartz` (default) — good absorber in transparent gangue
Huge loss contrast (eps''_target / eps''_gangue ≈ 2000). This is Kingman (2000)'s
**best-response** regime: selectivity is ~0.996 *before any tuning* because almost no
power can deposit in the near-lossless quartz. Geometry barely moves selectivity here —
material contrast already does the work.

### `pyrite_in_calcite` — Salsman's liberation system, nearly matched eps'
Pyrite (8 − j0.3) disseminated in calcite (8.5 − j0.05). The real parts are nearly
equal, so the field does *not* preferentially concentrate in the absorber, and the loss
contrast is only ~6×. This is close to Kingman's **worst-response** regime (finely
disseminated absorber in a dielectrically similar gangue) and is exactly the
pyrite-in-calcite system Salsman (1996) modelled for thermally-assisted liberation.
Untuned selectivity is only ~0.54.

**This is where inverse design earns its keep.** Optimising applicator geometry lifts
selectivity 0.54 → ~0.67 and per-area contrast ~5 → ~9 — recovering selectivity that
material contrast alone cannot provide, in precisely the ore class conventional
microwave treatment handles worst.

## Temperature dependence (what `run_sweeps.py` is anchored to)

Primary tables of ε″(T) for these minerals are mostly paywalled, so the ε″(T) ramp in
[src/mw_inv/sweeps.py](../src/mw_inv/sweeps.py) is a **parametric Arrhenius model, not a
digitised measurement**. It is anchored to these qualitative primary facts:

- **Pyrite, chalcopyrite, chalcocite: ε′ and ε″ vary *significantly* with temperature**,
  measured ambient → 650 °C at 615/1410/2210 MHz. Cumbane et al. (2008), via the
  [IntechOpen review](https://www.intechopen.com/chapters/40687).
- **Galena, sphalerite: little variation up to ~500 °C** — so not all sulphides ramp.
- Some ores reach **loss tangent ~1 by ~1000 °C** at 2.45 GHz (a magnitude ceiling; used
  as the cap in `EpsTModel`).
- Phase transformations cause jumps (e.g. goethite→hematite dehydroxylation raises ε′,ε″).
- Carbonate/silicate gangue (calcite, quartz) stays low-loss, so the ramp is **asymmetric**
  — target heats, gangue does not. This asymmetry is the selective-heating mechanism
  (Salsman et al. 1996: a pyrite grain in calcite generates interface tensile stress
  exceeding rock strength).

### Two non-obvious results from the sweeps
1. **Frequency is a strong free knob.** Across ±4% of the ISM band, `pyrite_in_calcite`
   selectivity moves 0.38–0.64 — retuning the source rivals the geometry optimizer.
2. **Absorption is self-limiting in ε″ (hence in T).** Absorbed power peaks at an
   impedance/skin-depth-matched ε″\* ≈ 0.4–0.6 for grains of this size and then *falls*
   (field expulsion). So a strongly-heating grain does **not** run away unboundedly — it
   self-regulates once ε″ passes the optimum. Unbounded runaway requires grains ≪ skin
   depth, where absorption keeps rising with ε″. Disseminated pyrite (ε″≈0.3) sits just
   below ε″\*, i.e. near-optimally matched already.

## Important caveats

- **Temperature dependence is not modelled.** Loss factors generally rise with
  temperature (the mechanism behind thermal runaway). Galena/sphalerite are reported
  roughly stable to ~500 °C; many absorbers are not. Any real design must sweep T.
- **Magnetite is magnetic** (μ ≠ 1); this scalar-permittivity model ignores magnetic
  loss, which is a real heating channel for magnetite specifically.
- **Single frequency.** No broadband or dispersion.
- Numbers were gathered from secondary extraction of the cited papers; before any
  quantitative engineering claim, pull the primary tables (Cumbane et al. 2008;
  Church/Webb/Salsman; Pickles et al.) and measure the specific ore.

## Primary references worth pulling next
- Near-field scanning microwave microscope mineral permittivity — *Sensors* 22(3):1138 (2022), open access (PMC8840724).
- Salsman, Williamson, Tolley, Rice (1996) — FE model, pyrite particle in calcite, thermally assisted liberation.
- Kingman et al. (2000) — microwave treatment response vs. mineral texture/dissemination.
- Church, Webb, Salsman — *Dielectric Properties of Low-loss Minerals* (US Bureau of Mines).
- Cumbane et al. (2008) — dielectric properties of selected minerals 1–22 GHz.
- "Twenty years of … microwave-assisted breakage of rocks and minerals — a review," [arXiv:2011.14624](https://arxiv.org/abs/2011.14624).
