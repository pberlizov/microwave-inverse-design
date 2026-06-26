# `data/` — inputs vs generated artifacts

This directory intentionally mixes **small, versioned inputs** (benchmarks, example ore
profiles, templates) with **generated outputs** (figures, JSON summaries, exports).

## Versioned inputs (commit these)

- `data/benchmarks/` — curated public-literature benchmark JSON used by
  `scripts/run_benchmarks.py` and CI.
- `data/templates/` — templates for measured inputs (copy to your own dataset paths).
- `data/ores/` — example ore profiles (QEMSCAN/assay-style JSON) used by ingest and
  `--ore` flows.
- `data/measured_eps.template.json` — template for bench dielectric-probe ingestion.
- `data/measured_dielectrics/` — versioned deposit ε(f,T,moisture) datasets (commit curated examples).
- `data/literature_tables/` — curated source tables for literature ingest adapters.
- `data/datasets_catalog.json` — open/online dataset registry (`docs/REAL_DATA_SOURCES.md`).
  Run `python scripts/ingest_literature_datasets.py` or `python scripts/prepare_test_data.py` before testing.
- `data/*.example.json`, `data/*.template.json` — small fixtures for CI and copy-paste.
- `data/lab_measurements.example.json` — example bench measurement payload.
- `data/third_party/touchstone/` — vendored Touchstone examples for Stage-A RF ingest.

Run **`python scripts/run_real_data_eval.py`** to FDFD-evaluate every source above
(ores, deposit ε points, benchmarks, pairs, phantoms, bench JSON). Copy
`measured_eps.example.json` → `measured_eps.json` (and lab JSON likewise) to include
your probe/bench measurements instead of CI fixtures.

## Generated artifacts (do not commit by default)

- `data/runs/` — timestamped pipeline outputs from `scripts/run_pipeline.py`.
- `data/design_exports/` — openEMS export bundles written by export scripts/pipeline.
- Top-level `data/*.json`, `data/*.png`, `data/*.npz` — experiment summaries/figures.

If you want to publish a specific result, prefer copying it into a dedicated
`results/` or `papers/` folder with an accompanying write-up rather than committing
raw run directories.
