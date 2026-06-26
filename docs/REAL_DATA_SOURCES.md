# Open & online datasets for microwave ore / mineral evaluation

Curated catalog for ingesting **real measured ε**, **ore composition**, **bench phantoms**,
and **literature anchors** into `mw-inv`. Status key:

| Status | Meaning |
|--------|---------|
| **In repo** | Already versioned or cited in code |
| **Ready** | Open download; schema/mapping known |
| **Extract** | PDF/thesis tables — needs manual or OCR ingest |
| **Indirect** | Composition/geometry only; ε must be linked separately |

Last reviewed: 2026-06.

---

## 1. Microwave ore & mineral dielectrics (highest priority)

### 1.1 U.S. Bureau of Mines / NIOSH (public domain PDFs)

| Resource | Content | f range | T range | Format | URL |
|----------|---------|---------|---------|--------|-----|
| Open-ended coax mineral survey | **60+ minerals**, powdered, density series | 300 MHz–3 GHz | 25–325 °C | PDF tables + figures | [CDC 10047](https://stacks.cdc.mil/view/cdc/10047) |
| Low-loss minerals + 915 MHz heating | Quartz, feldspar, calcite, etc. | 1 MHz–1 GHz (+ 915 MHz tests) | RT–elevated | PDF | [CDC 10413](https://stacks.cdc.mil/view/cdc/10413) |

**In repo:** anchors in `dielectric_data.py` / `mineral_catalog.py` (partial).  
**Ingest:** scrape tables → `data/measured_dielectrics/usbm_minerals_v1.json` (one phase per mineral, points per T,f).

### 1.2 Hartlieb et al. — hard-rock ε(T,f) @ ~2.45 GHz

| Resource | Content | Notes |
|----------|---------|-------|
| Zenodo **4835664** | Granite, basalt, sandstone ε vs T (RT–1000 °C), 397–2986 MHz | PDF dataset bundle |
| Minerals Engineering 2016 paper | Same data in Table 2 (ε′, ε″ @ 2450 MHz) | [ScienceDirect S0892687515301278](https://doi.org/10.1016/j.mineng.2015.11.008) |

**In repo:** `granite_hartlieb16`, `basalt_hartlieb16`, `sandstone_hartlieb16` in [SMRT `make_bedrock.py`](https://github.com/smrt-model/smrt/blob/master/smrt/inputs/make_bedrock.py) @ 2450 MHz RT only.  
**Ingest:** digitize Table 2 + Zenodo PDF → gangue/target phases for `materials_from_ore` Bruggeman gangue.

### 1.3 Goldbaum / Forster / Pickles (Toronto, HMAP ores)

| Resource | Content | Access |
|----------|---------|--------|
| **Forster PhD 2023** | Cavity-perturbation ε(T,f) for sulphides + **42 ore** heating classes | [TSpace item](https://utoronto.scholaris.ca/items/11badaa6-2b38-48be-90a4-7366e19aab4e) — thesis PDF + supplementary Excel |
| Goldbaum et al. 2020 | Pyrrhotite ε(T) @ 912 MHz | [DOI 10.1016/j.mineng.2019.106152](https://doi.org/10.1016/j.mineng.2019.106152) |
| Forster et al. IMPC 2022 | 42-ore heating-class summary | Conference paper |
| Pickles / Peng | Pyrite, FeS₂ in coal ε(T) @ **915 & 2450 MHz** | [JKSMER](https://www.jksmer.or.kr/articles/xml/Dx0o/) — figures digitizable |

**In repo:** HMAP list, heating-class ladder, partial ε(T) anchors (`dielectric_data.py`).  
**Ingest:** Forster supplementary → `data/measured_dielectrics/forster_hmap_minerals.json` + `data/ores/forster_42ores/` (assay fractions if provided).

### 1.4 MDPI / journal concentrate studies

| Paper | Minerals / ore | f | T | URL |
|-------|----------------|---|---|-----|
| ZnS concentrate roasting | Sphalerite concentrate | 915, 2450 MHz | RT–850 °C | [Minerals 7(2):31](https://doi.org/10.3390/min7020031) |
| Carbonaceous sulphidic Au concentrate | Refractory flotation concentrate | 2450 MHz region | High-T roasting | [DOI S0892687509000697](https://doi.org/10.1016/j.mineng.2009.04.010) |
| Tikhonov et al. 2010 | Magnetite, hematite, sphalerite, chalcopyrite, pyrite, ilmenite | **12–145 GHz** | RT | [PIER B Vol 25](https://doi.org/10.2528/PIERB10072404) — not ISM band but useful cross-check |

### 1.5 EuroPeg petrophysics (pegmatite ores)

| Resource | Content | Format |
|----------|---------|--------|
| **EuroPeg_PetroDB v3** (Zenodo) | European LCT/NYF pegmatite ores + wall rocks; **dielectric** among petrophysical properties | [Zenodo 14203353](https://zenodo.org/records/14203353) — 126 MB ZIP |

**Ingest:** map deposit samples → `data/ores/europe_pegmatite_*.json` + linked `measured_dielectrics/`.

---

## 2. Computational / DFT dielectric databases (literature fallback)

| Database | N entries | f | Notes | URL |
|----------|-----------|---|-------|-----|
| **Materials Project** | 10k+ with dielectric flag | static / DFPT | API `materials.dielectric`; free key | [materialsproject.org](https://materialsproject.org/) |
| **Zenodo dielectric tensors v3** | 1,056 inorganic | static | JSON tensors | [Zenodo 4987552](https://zenodo.org/records/4987552) |
| **oxi_diel_db** (GitHub) | Oxides, MP-linked | static | JSON per structure | [takahashi-akira-36m/oxi_diel_db](https://github.com/takahashi-akira-36m/oxi_diel_db) |
| **WURM** | Minerals | static (DFPT) | Raman + **dielectric** + IR | [wurm.info](https://www.wurm.info/) |

**Caveat:** static/DFPT ε ≠ lossy microwave ore at 2.45 GHz — use for gangue/mineral **ranking**, not promotion-tier measured ε.

---

## 3. Bench phantoms & probe calibration (E0 path)

| Resource | Content | License | URL |
|----------|---------|---------|-----|
| **Gabriel 1996** | Tissue ε(f) Cole–Cole; **salt gel anchors** for saline phantoms | Open mirror | [IFAC-CNR tissprop](https://niremf.ifac.cnr.it/tissprop/) |
| **IT'IS Tissue DB** | Same dispersion; downloadable Excel/ASCII | Academic | [itis.swiss/.../database](https://itis.swiss/virtual-population/tissue-properties/database/) |
| **In repo** | `measured_eps.example.json`, `lab_measurements.example.json` | Example fixtures | `data/*.example.json` |

**Live bench:** copy examples → `data/measured_eps.json`, `data/lab_measurements.json`.

---

## 4. Rock / regolith ε(T,f) — gangue & simulant

| Resource | Rocks | f | T | Size | URL |
|----------|-------|---|---|------|-----|
| Zenodo **14629408** | Etna basalt, L5 chondrite, CI simulant | Broadband (~1601 pts) | Variable | **774 MB** MATLAB | [Zenodo 14629408](https://zenodo.org/records/14629408) |
| USGS **Olhoeft 1979** | Selected rocks/minerals RT statistics | lab | RT | PDF scan | [OF 79-993](https://doi.org/10.3133/ofr79993) |
| SMRT bedrock table | granite/basalt/sandstone @ 2.45 GHz + frozen bedrock conductivity | 2.45 GHz | coded | Python dict | [smrt/inputs/make_bedrock.py](https://github.com/smrt-model/smrt/blob/master/smrt/inputs/make_bedrock.py) |

---

## 5. Ore composition & mineralogy (link to Bruggeman / QEMSCAN)

No public “QEMSCAN + ε” unified DB exists. Composition-only sources:

| Resource | Content | Format | URL |
|----------|---------|--------|-----|
| **Global mine production DB** | 1171 mines, 80 materials, reserves, coords | CSV/GPKG ZIP | [Zenodo 7369478](https://zenodo.org/records/7369478) |
| **BGS NGDC QEMSCAN compilations** | Porphyry/deposit mineral maps | Excel | e.g. [DOI 10.5285/5a4bc758-ac88-4715-a665-81f69108f854](https://doi.org/10.5285/5a4bc758-ac88-4715-a665-81f69108f854) |
| **data.gov.uk QEMSCAN** | Volcanic tephra, IODP sediments | Various | [Tajogaite dataset](https://www.data.gov.uk/dataset/1d833eba-79d4-43c6-8281-67e1686cb0a4) |
| **RRUFF** | Chemistry, XRD, Raman — **no ε** | ZIP downloads | [rruff.info](https://rruff.info/) |

**In repo:** `data/ores/*.json` (4 profiles), `ORE_PROFILES` builtins.  
**Ingest:** assay/QEMSCAN CSV → `fractions` block; link measured ε separately.

---

## 6. Moisture / frequency robustness (C1 scenarios)

| Resource | Content | Relevance | URL |
|----------|---------|-----------|-----|
| **DDOAS** | Organic Arctic soils ε(f,T,moisture) | Moisture interpolation patterns | [Zenodo 3819912](https://doi.org/10.5281/zenodo.3819912) |
| **In repo deposit example** | Pyrite porphyry ε(T,f,moisture) | Template schema | `data/measured_dielectrics/disseminated_pyrite_porphyry_deposit.json` |

---

## 7. Geometry / microstructure (D1 — not ε)

| Resource | Content | URL |
|----------|---------|-----|
| **Digital Rocks Portal DRP-372** | 3D pore structures + simulated transport | [OSTI 1975026](https://doi.org/10.2172/1975026) |
| EuroPeg / QEMSCAN | PSD, porosity, liberation | See §1.5, §5 |

---

## 8. Already in this repository

| Path | Type | Eval command |
|------|------|--------------|
| `data/benchmarks/*.json` | Literature regression | `scripts/run_benchmarks.py` |
| `data/ores/*.json` | Ore profiles | `scripts/run_real_data_eval.py` |
| `data/measured_dielectrics/*.json` | Deposit ε(f,T,m) | `--ore` + `scripts/ingest_deposit.py` |
| `data/measured_eps.example.json` | Bench probe | `scripts/run_bench_calibration.py` |
| `src/mw_inv/dielectric_data.py` | 14+ mineral ε(T) anchors | `scripts/run_materials_catalog.py --pairs` |
| `src/mw_inv/materials.py` | 6 material pairs @ 2.45 GHz | pipeline default |

Batch evaluation across all of the above:

```bash
python scripts/run_real_data_eval.py          # full
python scripts/run_real_data_eval.py --quick  # CI-scale
```

---

Run **`python scripts/ingest_literature_datasets.py`** to (re)build Hartlieb/USBM/mineral JSON under `data/measured_dielectrics/`.

## Ingest adapters (automated)

| Adapter | Output | Source |
|---------|--------|--------|
| `hartlieb` | `measured_dielectrics/hartlieb_bedrock_v1.json` | Hartlieb 2016 Table 2 |
| `usbm_low_loss` | `measured_dielectrics/usbm_low_loss_gangue_v1.json` | USBM RI 9035 |
| `literature_minerals` | `measured_dielectrics/literature_hmap_minerals_v1.json` | `dielectric_data.py` |
| `usbm_coax` | `measured_dielectrics/usbm_coax_minerals_v1.json` | USBM CDC 10047 subset |
| `forster_hmap` | `measured_dielectrics/forster_hmap_minerals_v1.json` | Forster PhD Ch. 4–5 + gangue merge |
| `forster_ores` | `ores/forster/*.json` (42 profiles) | Forster heating-class manifest |
| `europeg` | `measured_dielectrics/europeg_pegmatite_v1.json` | EuroPeg subset |
| `computed_static` | `measured_dielectrics/computed_dielectric_subset_v1.json` | MP/Zenodo static ε |
| `gabriel` | `measured_dielectrics/gabriel_saline_phantoms_v1.json` | Gabriel saline gel anchors |

```bash
python scripts/prepare_test_data.py              # ingest all + validate ores
python scripts/prepare_test_data.py --eval         # + quick FDFD sweep
python scripts/ingest_literature_datasets.py --status
python scripts/run_real_data_eval.py --quick
```

Machine-readable registry: `data/datasets_catalog.json`.

1. **USBM 60-mineral coax dataset** (CDC PDFs) — fills `measured_dielectrics/` for Bruggeman minerals.
2. **Hartlieb Zenodo 4835664** — real gangue ε(T) at 2.45 GHz (granite/sandstone/basalt).
3. **Forster 2023 thesis supplements** — HMAP minerals + 42-ore assay/heating (deposit-calibrated tier).
4. **EuroPeg_PetroDB** — European pegmatite ores with measured petrophysics including ε.
5. **Materials Project API** — gap-fill minerals missing from USBM (static ε ranking only).
6. **Global mine production DB** — mine-level metadata for provenance (not ε).
7. **Your lab:** `measured_eps.json`, `lab_measurements.json`, deposit-specific `measured_dielectrics/*.json`.

---

## 10. Gaps (no open bulk download found)

- **Single CSV** of pyrite/chalcopyrite ε(T) @ 915/2450 MHz from Pickles/Goldbaum (papers only).
- **QEMSCAN + matched probe ε** on the same drill core (commercial / thesis-only).
- **OpenEMS/VNA S-parameter archives** tied to cavity geometries in this repo (bring your own `.s1p`).
- **Pilot-scale high-power microwave** time series (proprietary).

For these, use thesis supplementary files, author email, or `scripts/ingest_deposit.py` with your probe JSON.
