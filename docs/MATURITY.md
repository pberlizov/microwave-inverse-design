# Implementation maturity

Honest status of each major component. **Do not cite quantitative claims from
WIP paths** in grants or papers without completing the gaps.

| Component | Status | Module / script |
|---|---|---|
| 2D FDFD forward model | **CORE** | `fdfd.py`, `run_demo.py` |
| Cited mineral materials + ε(T) | **CORE** | `materials.py`, `dielectric_data.py` |
| Geometry search (6-knob legacy) | **CORE** | `search.py`, `run_search.py --legacy` |
| Validation suite (MMS, grid, etc.) | **CORE** | `validation.py`, `run_validation.py` |
| Coupling efficiency metric (A0) | **CORE** | `fom.py` — energy-consistent `coupling_eff` |
| Reflected power / \|S11\| (A0 cont.) | **NOT STARTED** | needs a matched port — deferred to A1 (openEMS) |
| Manufacturable geometry params | **EXPERIMENTAL** | `geometry.py` — point source + Im(ε) PEC, not real ports |
| EM–thermal coupling (2D) | **EXPERIMENTAL** | `thermal.py` — representative k, h |
| Ensemble / freq-robust / multi-obj search | **EXPERIMENTAL** | `ensemble.py`, `run_*_search.py` |
| MEEP 2D cross-check | **EXPERIMENTAL** | `meep_compare.py` (optional dep) |
| MEEP 3D primitive FDTD | **EXPERIMENTAL** | `meep_3d.py` — explicit boxes/cylinders, point source |
| MEEP 3D extrusion (legacy) | **WIP** | `meep_compare.meep_selectivity_3d_extruded` — quasi-3D only |
| openEMS export | **EXPERIMENTAL** | `openems_export.py`, `export_openems.py` — runnable Octave script |
| Lab gel phantom validation | **EXPERIMENTAL** | `phantom.py`, `phantom_data.py` — recipe ε + thermal ΔT; bench compare via JSON |
| Design export + solver triangulation | **EXPERIMENTAL** | `design_export.py`, `solver_triangulation.py` |
| Validation gate (pyrite canonical) | **EXPERIMENTAL** | `validation_gate.py`, `run_validation_gate.py` |
| Stress / liberation FOM | **EXPERIMENTAL** | `stress.py`, `run_stress_search.py` |
| Ore HMAP profiles | **EXPERIMENTAL** | `ore_profiles.py`, `run_ore_profile.py` |
| Bench phantom protocol | **EXPERIMENTAL** | [docs/BENCH_PROTOCOL.md](BENCH_PROTOCOL.md), `phantom_calibration.py` |
| 16-cell dielectric tuner | **EXPERIMENTAL** (deprecated) | `run_field_search.py` — non-physical bound |

Status constants live in [`src/mw_inv/maturity.py`](../src/mw_inv/maturity.py).
Scripts call `warn_if_below()` at startup for WIP components.

## Coupling efficiency (backlog A0) — and a pathology it exposes

Every evaluation now reports an energy-consistent **coupling efficiency**
(`fom.py`, surfaced on `DesignReport`): in a lossless-walled cavity the total absorbed
power equals the power the feed delivers, so

    coupling_eff = P_abs_charge / P_abs_total            (in [0, 1])

is the fraction of delivered power that lands in the ore rather than internal structure.
`p_structural` and `pec_loss_fraction` break out where the rest goes.

This immediately surfaces a real pitfall in selectivity-only optimization: the internal
**baffle is modelled as `Im(eps)=1e6`, which is a near-perfect *absorber*, not a lossless
PEC reflector.** A long baffle can *raise* selectivity (0.748 → 0.763 on
`pyrite_in_calcite`) while collapsing charge power by ~12 orders of magnitude
(`coupling_eff → 0`, `pec_loss_fraction → 1`). Selectivity alone does not warn you; the
coupling metric does. Two consequences:

- `EvaluationConfig(coupling_floor=...)` penalises designs below a coupling floor
  (`coupling_score_factor`, default 0.25), so search cannot win by routing power into
  structure. Off by default (floor = 0) to avoid silently rejecting legacy baffle runs.
- The lossy-`Im(eps)` PEC approximation should be replaced by a true PEC (Dirichlet) or
  validated against an openEMS reflector before any baffle result is trusted.

A *true* scattering `|S11|` / reflected-power figure needs a matched port; the grid-node
point source has no usable input impedance (its driven-node field is dominated by the
source self-term, identical across feed positions). That metric is deferred to the
openEMS port-truth path (backlog A1).

## What “EXPERIMENTAL” means for 3D / lab paths

**openEMS** (`export_openems.py`): writes a self-contained Octave script with CSXCAD
geometry (gangue box, target cylinders, PEC plate), lumped port, field dump, and
selectivity post-processing. Requires openEMS installed locally; validate HDF5 dump
layout on your version.

**MEEP 3D primitive** (`meep_3d.py`): builds explicit 3D shapes matching
`scene_export.build_primitives`. Still uses a point Ez source — not a matched coax port.

**Lab phantoms** (`run_phantom_study.py`): saline gel recipes with Gabriel-scaled ε
anchors, FDFD selectivity + coupled thermal ΔT predictions, and optional comparison
to bench JSON (`data/lab_measurements.example.json`). Calibrate ε with a probe on your
gel batch before trusting absolute numbers.

## What “WIP” means here

**MEEP 3D extrusion**: extrudes the **2D** permittivity map in *z* with a point Ez
source. Superseded by primitive 3D where possible; kept for cross-check.

## Production path (from [FRONTIER.md](FRONTIER.md))

openEMS or MEEP/FDTDX with matched ports → measured ore ε → coupled EM–thermal →
manufacturable 3D geometry → bench phantom → ore trial.
