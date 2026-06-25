# Mineral dielectric data used in this model

All values are complex relative permittivity `eps = eps' + i·eps''` (eps'' > 0 = lossy),
at/near **2.45 GHz** unless noted. Room-temperature scene values live in
[src/mw_inv/materials.py](../src/mw_inv/materials.py); **ε(T, f) anchor tables** and
**μ for magnetite** live in [src/mw_inv/dielectric_data.py](../src/mw_inv/dielectric_data.py).
Use `Materials.from_pair("pyrite_in_calcite", target_T_K=773)` to re-solve with heated target ε.

## Values

| Mineral | Role | eps' | eps'' | Notes | Source |
|---|---|---|---|---|---|
| Magnetite (Fe₃O₄) | absorber | ~12 | ~1 | **μ ≈ 1.3 − j0.55 @ 2.45 GHz** (Hotta et al. 2009/2011); ε(T) anchors to 848 K | MDPI 2022 ε; Hotta ISIJ Int. 49(9), 51(3) |
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

## Temperature dependence

### In the forward model (new)
Piecewise-linear **ε(T) anchors at 2.45 GHz** in `dielectric_data.py`:

| Mineral | T (K) | ε′ | ε″ | Source |
|---|---|---|---|---|
| Pyrite (disseminated) | 298 | 8.0 | 0.30 | MDPI 2022 |
| Pyrite | 573 | 8.3 | 0.42 | Peng et al. 2011 trend @ 2.45 GHz |
| Pyrite | 773 | 8.8 | 0.55 | Peng / Cumbane qualitative |
| Pyrite | 973 | 9.5 | 0.85 | Cumbane 2008 strong rise |
| Magnetite | 298 | 12.0 | 1.0 | MDPI 2022 |
| Magnetite | 773 | 14.0 | 2.5 | Hotta et al. 2011 @ 2.45 GHz |
| Quartz / calcite | 298–1273 | ~constant | ~constant | inactive gangue |

Gangue phases (quartz, calcite) stay low-loss across the table — the asymmetry that drives
selective heating is preserved.

### In thermal sweeps (`run_sweeps.py`)
The lumped ε″(T) ramp remains a **parametric Arrhenius fallback** (`EpsTModel`), now
anchored to the mineral table at T_ref via `materials.eps_t_model()`. It is still **not**
coupled EM–thermal time stepping (held for step 3).

Primary literature anchors:

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

### Three non-obvious results from the sweeps
1. **Frequency is a strong free knob.** Across ±4% of the ISM band, `pyrite_in_calcite`
   selectivity moves 0.38–0.64 — retuning the source rivals the geometry optimizer.
2. **Absorption is self-limiting in ε″ (hence in T).** Absorbed power peaks at an
   impedance/skin-depth-matched ε″\* ≈ 0.4–0.6 for grains of this size and then *falls*
   (field expulsion). So a strongly-heating grain does **not** run away unboundedly — it
   self-regulates once ε″ passes the optimum. Disseminated pyrite (ε″≈0.3) sits just
   below ε″\*, i.e. near-optimally matched already.
3. **The runaway/self-limiting boundary is grain size vs skin depth**
   (`scripts/run_grain_sweep.py`). The power penetration (skin) depth at 2.45 GHz in
   pyrite ranges ~184 mm (ε″=0.3) down to **~7.6 mm at loss tangent 1**. Sweeping ε″ at
   fixed grain size traverses the grain/skin-depth ratio, and the absorption turnover
   lands at **grain diameter ≈ skin depth** (measured: d/δ collapses to **1.8 ± 0.4**
   across grain sizes). Consequence:
   - **Large grains (≳ a few cm): self-limiting** — they turn over at low ε″ and never
     heat away.
   - **Small grains (≲ skin depth at max loss, ~mm scale — i.e. real disseminated
     grains): monotonic** — absorption keeps rising with ε″ (with temperature) through
     the whole physical range, so the feedback stays positive and the grain is
     **runaway-prone**. This is exactly why finely disseminated sulphides (Salsman's
     pyrite-in-calcite) are the canonical thermally-assisted-liberation target: they sit
     on the runaway-prone side of the boundary.

## Important caveats

- **ε(T) in the forward model uses interpolated anchors**, not digitised full curves.
  Phase-transformation jumps (pyrite→pyrrhotite ~500 °C) are not resolved.
- **Magnetite μ(T) is tabulated** at 2.45 GHz (Hotta); magnetic loss is included in
  `absorbed_power_density` via μ″·|H|².
- **Bulk vs disseminated:** Polyakova et al. (2010) bulk-mineral ε at 2.45 GHz is orders
  of magnitude above disseminated scene values — used only in `run_validation.py` cross-checks.
- **Single frequency** in scenes (2.45 GHz default); table supports freq interpolation
  when multiple anchors exist.
- Before quantitative engineering claims, pull primary tables for the specific ore.

## Forward-model validation

`python scripts/run_validation.py` runs analytic and convergence checks in
[src/mw_inv/validation.py](../src/mw_inv/validation.py):

- Method-of-manufactured-solutions (Helmholtz operator)
- Dual-grid selectivity agreement (nx=61 vs 101)
- Empty-cavity resonance peak
- Literature bulk-vs-disseminated ε consistency (Polyakova 2010)
- Magnetic-loss power channel (μ″ term)
- ε(T) selectivity shift (298 K vs 773 K)

Optional **MEEP** cross-check when `meep` is installed (`conda install -c conda-forge meep`):
[src/mw_inv/meep_compare.py](../src/mw_inv/meep_compare.py).

## Spatial EM–thermal coupling (`run_thermal.py`)

[src/mw_inv/thermal.py](../src/mw_inv/thermal.py) closes the feedback loop the lumped model
approximates:

1. **FDFD** at local ε(T), μ(T) from `build_scene_at_T`.
2. Absorbed power density **q(x, y)** as heat source.
3. **Steady heat equation** k∇²T − h(T−T_amb) + q = 0 (Dirichlet T_amb on cavity walls).
4. Iterate until T converges in the ore charge.

Example result for `pyrite_in_calcite` at drive=8 (representative k, h=4×10⁴ W/m³/K):

| Quantity | Isothermal @298 K | Coupled steady state |
|---|---|---|
| EM selectivity | ~0.75 | ~0.86 |
| ΔT (target − gangue) | 0 | ~400 K |

Thermal FOMs: `delta_T_K`, `heat_selectivity`, `T_mean_target_K`, convergence history.

## HMAP mineral catalog (state of the art)

Nine **highly microwave-amenable phases** (Goldbaum/Forster IMPC 2022) now have
ε(T) anchor tables in `dielectric_data.py`, with microwave class tags in
`mineral_catalog.py`:

| Key | Class | ε @ 2.45 GHz (298 K) | Primary source |
|-----|-------|----------------------|----------------|
| molybdenite | HMAP | ~7.5 − j1.0 | Chen hyperactive MoS₂ |
| pyrrhotite | HMAP | ~11 − j0.65 | Peng 2013 @ 2.45 GHz |
| chalcopyrite | HMAP | ~10 − j0.45 | Cumbane/Lovas; impure ore |
| bornite | HMAP | ~10.5 − j0.55 | Harrison good-MW heater |
| pentlandite | HMAP | ~10 − j0.50 | Goldbaum et al. 2020 |
| galena | HMAP | ~9.5 − j0.35 | Cumbane stable to 500°C |
| pyrite | HMAP | ~8 − j0.3 | MDPI 2022 disseminated |
| hematite | HMAP | ~11 − j0.40 | Nelson/Blake 1989 oxide |
| magnetite | HMAP | ~12 − j1.0 (+ μ″) | MDPI + Hotta |

Gangue extensions: **dolomite**, **feldspar** (MDPI plagioclase ref), **serpentine**
(Goldbaum ultramafic silicate).

New `MaterialPair`s for search/export:

- `chalcopyrite_in_calcite` — Cu porphyry matched-ε′ regime
- `pyrrhotite_in_quartz` — Ni-Cu / massive sulphide high contrast
- `galena_in_calcite` — Pb-Zn style
- `molybdenite_in_quartz` — Mo concentrate / hyperactive

`materials_from_ore()` **Bruggeman-mixes** HMAP fractions into effective target ε and
**auto-selects** the best cited `MaterialPair` via `suggest_material_pair()`.

### Deposit ore JSON (Tier 2)

Place QEMSCAN / assay modal fractions in `data/ores/*.json`:

```json
{
  "label": "deposit_sample_12",
  "source": "QEMSCAN",
  "fractions": {"pyrite": 0.04, "quartz": 0.55, "feldspar": 0.20},
  "gangue_mineral": "quartz",
  "texture": {"class": "disseminated", "mean_grain_radius_m": 0.0025}
}
```

```bash
python3 scripts/ingest_ore_profile.py data/ores/disseminated_pyrite_porphyry.json
python3 scripts/run_search.py --ore data/ores/massive_pyrite.json --trials 40
python3 scripts/run_pipeline.py --ore data/ores/massive_pyrite.json --trials 24
```

Texture `class` maps to default grain layout (`disseminated` → many small grains;
`massive` → single large inclusion). `mean_grain_radius_m` sets `inclusion_radius_frac`.

```bash
python3 scripts/run_materials_catalog.py --pairs
```

## Primary references worth pulling next
- Near-field scanning microwave microscope mineral permittivity — *Sensors* 22(3):1138 (2022), open access (PMC8840724).
- Salsman, Williamson, Tolley, Rice (1996) — FE model, pyrite particle in calcite, thermally assisted liberation.
- Kingman et al. (2000) — microwave treatment response vs. mineral texture/dissemination.
- Church, Webb, Salsman — *Dielectric Properties of Low-loss Minerals* (US Bureau of Mines).
- Cumbane et al. (2008) — dielectric properties of selected minerals 1–22 GHz.
- "Twenty years of … microwave-assisted breakage of rocks and minerals — a review," [arXiv:2011.14624](https://arxiv.org/abs/2011.14624).
