# Bench phantom protocol (gel cavity validation)

This protocol closes **step 2** of the production path: validate that FDFD-ranked
designs produce the expected **thermal rank** on a physical bench, before claiming
ore applicability.

## Prerequisites

- 36 cm aluminium cavity (or exported openEMS dimensions)
- 2.45 GHz CW source (magnetron + circulator or solid-state, ≥100 W forward)
- IR camera or fibre thermocouples
- Open coax dielectric probe (or cavity perturbation kit)
- Agar/glycerol gel batches per [phantom_data.py](../src/mw_inv/phantom_data.py) recipes

## 1. Measure gel permittivity

For each batch (target and gangue salt wt%):

1. Prepare gel, rest 24 h, measure at 20–25 °C.
2. Record ε′, ε″ at **2.45 GHz**.
3. Save to `data/measured_eps.json` (copy from [measured_eps.template.json](../data/measured_eps.template.json)).

```bash
python3 scripts/ingest_probe_measurements.py data/measured_eps.json --phantom saline_2_vs_0.5
```

Drift vs Gabriel anchors >20% on ε″ should trigger recipe update before comparing ΔT.

## 2. Export optimised designs

```bash
python3 scripts/run_validation_gate.py --materials pyrite_in_calcite
python3 scripts/export_design.py --phantom saline_2_vs_0.5 --trials 16
python3 scripts/run_pipeline.py --materials pyrite_in_calcite --trials 12 --phantom saline_2_vs_0.5 --measured-eps data/measured_eps.json --lab-measurements data/lab_measurements.json --bench-study
```

Build cavity using `optimized_params` from manifest JSON (feed_wall, stub_depth_frac, plate_*).

## 3. Phantom layout

- Low-salt gel bed filling charge region (`charge_w/h_frac` from params).
- High-salt cylindrical inclusions at `inclusion_offsets_frac` radii.
- Match `inclusion_radius_frac` to mould size (~18 mm for default 0.05 × 0.36 m).

## 4. Measurement procedure

| Step | Action |
|------|--------|
| A | Untuned geometry baseline — record ΔT(target−gangue) at 60–120 s |
| B | Optimised geometry — same gel batch, same forward power |
| C | Optional: S11 at port before/after loading charge (VNA) |
| D | Save JSON per [lab_measurements.example.json](../data/lab_measurements.example.json) |

```bash
python3 scripts/run_phantom_study.py --phantom saline_2_vs_0.5 --compare data/lab_measurements.json
```

Optional: turn VNA Touchstone traces into a compact report (for manifests / QA):

```bash
python3 scripts/ingest_vna_s11.py --unloaded data/vna/unloaded.s1p --loaded data/vna/loaded.s1p --out data/vna_s11_report.json
python3 scripts/build_rf_port_report.py --unloaded-s1p data/vna/unloaded.s1p --loaded-s1p data/vna/loaded.s1p --out data/rf_port_report.json
```

## 5. Success criteria

- **Rank correct:** optimised ΔT > untuned on the **same** gel batch.
- Model rank agreement even if absolute ΔT differs by ~20–30%.
- Probe-measured ε improves prediction vs anchors alone.

## 6. openEMS cross-check (optional)

If openEMS is installed:

```bash
cd data/design_exports/pyrite_gate/tpe_best
octave --eval "selectivity = mw_inv_tpe_best_cavity();"
python3 scripts/run_validation_gate.py --openems-dump-dir data/openems_runs
```

Run [calibration_cavity.m](../data/design_exports/pyrite_gate/calibration_cavity.m) first; |S11| should be finite (not NaN — indicates shorted port).
