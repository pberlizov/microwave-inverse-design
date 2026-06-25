# Public literature benchmarks

Curated reference data in [`data/benchmarks/`](../data/benchmarks/) for regression
against published mining-microwave and dielectric literature. **There is no standard
open benchmark for inverse applicator design** — these tiers validate forward-model
grounding only.

Run:

```bash
python3 scripts/run_benchmarks.py
python3 scripts/run_benchmarks.py --tier dielectric heating_class
```

Output: `data/benchmark_report.json`

## Tiers

| Tier | File | What it checks |
|------|------|----------------|
| `dielectric` | `literature_dielectric.json` | ε′, ε″ @ 2.45 GHz vs MDPI 2022, Nelson, pairs; ε(T) trends |
| `heating_class` | `goldbaum_heating_classes.json` | HMAP wt% → Goldbaum class I–IV; predicted °C/min bands |
| `phantom` | `phantom_saline_gabriel.json` | `saline_eps()` vs Gabriel-scaled anchors |
| `stress` | `stress_qualitative.json` | Grain-size penalty order; α_pyrite > α_calcite |
| `solver` | `solver_internal.json` | Empty-cavity resonance; Polyakova bulk ≠ disseminated |

## Sources (open access)

- MDPI Sensors 22(3):1138 / [PMC8840724](https://pmc.ncbi.nlm.nih.gov/articles/PMC8840724/)
- Nelson & Lindroth 1989 — [US Bureau of Mines](https://stacks.cdc.gov/view/cdc/10413/cdc_10413_DS1.pdf)
- Cumbane et al. 2008 — IntechOpen mineral processing chapter
- Goldbaum / Forster et al. 2022 — [CEEComm PDF](https://www.ceecthefuture.org/component/cck/?file=publication_file&id=3999&task=download)
- Goldbaum PhD thesis — [TSpace](https://utoronto.scholaris.ca/items/11badaa6-2b38-48be-90a4-7366e19aab4e)

## Adding benchmarks

1. Digitize a literature table into the appropriate JSON file.
2. Add an entry with `id`, reference values, and `rtol_*` tolerances.
3. Run `pytest tests/test_benchmarks.py` — new checks must pass on current code.

## What this does *not* benchmark

- Geometry optimisation rankings (no public ground truth)
- openEMS/MEEP selectivity vs FDFD (see `run_validation_gate.py`)
- Bench phantom ΔT (see `docs/BENCH_PROTOCOL.md`)
