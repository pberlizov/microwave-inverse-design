# Backlog — from thin-slice to industry-relevant

This repo already proves the **thin-slice** claim (“geometry can move selectivity”) and
has a working promotion pipeline (`scripts/run_pipeline.py`). The backlog below captures
what blocks (a) **state-of-the-art inverse EM design** and (b) **industry usefulness**
for a real microwave applicator.

Conventions:
- **P0/P1/P2**: priority (P0 blocks credibility or any industrial decision).
- Each item has **Implementation steps** and **Done when** acceptance criteria.
- Where possible, items map onto existing **promotion tiers** in `src/mw_inv/promotion.py`.

## Milestones (recommended order)

1. **M1 — Solver-triangulated** (reach `solver_triangulated` tier consistently) — **one-command path done** (`--run-openems` / `--synthesize-openems-dumps`)
2. **M2 — Bench-calibrated** (reach `bench_calibrated` on gel phantoms) — **pipeline path done** (`--phantom`, `--measured-eps`, `--lab-measurements`, `evaluate_bench_gate`)
3. **M3 — Deposit-calibrated** (new tier: measured ore ε(f,T,moisture) + validation) — **done** (`DEPOSIT_CALIBRATED` tier, `--ore` manifest block)
4. **M4 — Pilot-ready** (new tier: safety constraints + repeatability + throughput metrics) — **done** (`PILOT_READY` tier, `pilot_gate.py`, `--pilot-enforce`)

## Backlog mapping (deferred from “ports + productization” scope)

- **Industrial spec + constraints**: see Epics **C** (objectives/constraints/safety) and **H** (manufacturing handoff).
- **Coupled physics where it changes decisions**: see Epic **E** (EM–thermal–mechanical).
- **Staged calibration/validation loop**: see Epics **E** (phantoms) and **D** (measured ore ε).
- **Robust / uncertainty-aware optimization**: see Epics **C** and **D** (materials uncertainty).

## Epic A — Real ports, power, and coupling (SOTA + industry blocker)

### A0 (P0) Add coupling efficiency and reflected power metrics to every evaluation
**Why:** Selectivity without coupling is not actionable; industry needs “how much of
source power lands in the charge” and “how sensitive is matching to load/geometry”.

**Status (partial):** energy-consistent `coupling_eff` + `p_abs_total` + `p_structural`
+ `pec_loss_fraction` are in `fom.py`, surfaced on `DesignReport`, recorded in the
pipeline manifest (untuned + tpe_best), and enforceable via `EvaluationConfig.coupling_floor`.
This already exposed the lossy-`Im(eps)` baffle pathology (see docs/MATURITY.md).
**Remaining:** true `|S11|` / reflected power needs a matched port → moved to A1 (openEMS).

Implementation steps:
1. Define standard metrics: `P_abs_charge`, `P_abs_target`, `P_incident`, `|S11|`,
   and derived `coupling_eff = P_abs_charge / P_incident`.
2. Add these to evaluation outputs (`DesignEvaluator` reports and JSON manifests).
3. Update pipeline manifests (`src/mw_inv/run_manifest.py`) to record coupling metrics
   for `untuned` and `tpe_best`.

Done when:
- `scripts/run_pipeline.py` writes coupling metrics for untuned + best.
- A “good” design cannot regress coupling below a user-configured floor.

### A1 (P0) Replace “point source” excitation with a port model for *truth* solvers

**Status (partial):** openEMS exports use a **wall-mounted lumped port** (``AddLumpedPort``)
that tracks the optimised feed (wall + along-wall + stub depth) and reports matched-port
metrics via ``calcPort``; each run writes ``port_metrics.json`` (|S11|, coupling_eff, selectivity).
Python ingests
via ``openems_postprocess.ingest_openems_case``; triangulation rows carry
``openems_s11_mag`` / ``openems_coupling_eff``; gate adds coupling-floor checks and
``openems_diagnosis`` (ranking mismatch vs coupling collapse).
Harness: ``scripts/run_port_validation.py``.

**M1 one-command (done):** ``mw-inv-pipeline --run-openems`` runs Octave on the export
bundle and refreshes triangulation/gate/promotion; ``--synthesize-openems-dumps`` for
CI without Octave. Shared refresh: ``run_refresh.apply_triangulation_refresh``;
``scripts/update_run_with_openems.py`` for post-hoc ingest.

**Remaining:** automated Octave runner in CI (optional job with openEMS container).

Implementation steps:
1. Treat openEMS as the first “port-truth” engine (already exportable via
   `src/mw_inv/openems_export.py`).
2. Add a runnable “port validation harness”:
   - generate model for 2–3 canonical geometries (untuned/random_best/tpe_best),
   - run openEMS,
   - extract `|S11|` and absorbed-power selectivity via dump post-processing.
3. Wire results into `solver_triangulation` (`src/mw_inv/solver_triangulation.py`)
   so the promotion tier can reach `solver_triangulated` without manual glue.

Done when:
- `scripts/run_solver_triangulation.py` can consume an openEMS run directory and fill
  `openems_selectivity` + `|S11|` + coupling in the triangulation report.
- Gate failures explain whether the issue is “ranking mismatch” vs “coupling collapse”.

### A2 (P1) Improve FDFD excitation beyond a point source (keep FDFD as fast pre-screen)

**Status (line-port done):** ``build_source_jz`` distributes ``J_z`` across the stub mouth
with the same discrete RHS scale as the legacy point feed; ``Scene.source_j`` +
``solve_scene()`` is the default in search/evaluator/thermal. Legacy point feed remains
via ``solve(..., source_xy=...)`` for regression tests.

**Remaining:** TE10 waveguide boundary mode; adjoint-ready source API.

Implementation steps:
1. Extend `fdfd.solve` to accept either:
   - a sparse distributed current `Jz(x,y)` source term, or
   - a boundary-condition driven excitation (e.g., waveguide edge mode).
2. Update `geometry.build_scene` to emit the new excitation representation.
3. Add unit tests in `tests/test_fdfd.py` asserting invariants (symmetries, scaling,
   stability vs grid refinement).

Done when:
- “Manufacturable” geometry no longer requires the “grid-node point feed” caveat.

## Epic B — 3D physics and boundary conditions (industry blocker)

### B0 (P0) Make a 3D solver path the default “next step” in promotion
**Why:** Applicators are 3D; 2D results are only screening.

Implementation steps:
1. Define a minimal 3D geometry family that matches a buildable cavity + feed + plate.
2. Ensure openEMS export supports these 3D primitives with consistent naming and dump layout.
3. Expand `validation_gate` to optionally require 3D agreement once openEMS data exists.

Done when:
- The repo can produce 3D openEMS models for the same param family used in optimization,
  and the gate can enforce rank agreement.

### B1 (P1) Boundary realism: vents/windows, non-PEC features, and loss in metals
Implementation steps:
1. Identify which boundaries are truly PEC vs apertures (couplers, windows, conveyance).
2. Add models for apertures and finite conductivity where it matters (openEMS first).

Done when:
- “Pilot-ready” models reflect the actual applicator topology (not sealed PEC only).

## Epic C — Objectives, constraints, and safety (industry blocker)

### C0 (P0) Multi-objective search: selectivity × coupling × safety
**Why:** Industrial designs trade off selectivity, throughput, matching, and safety.

**Status (partial):** ``optuna_multi_search`` maximises selectivity + ``coupling_eff``;
``pareto_recommend()`` weighted picker; ``--check-arcing`` filters risky trials.
``scripts/run_multi_search.py`` writes Pareto front + recommendation JSON.
Pipeline: ``--multi-objective`` maps Pareto recommendation into ``tpe_search`` for gate/export.
``--check-hotspot`` / ``--max-hotspot-dt`` apply coupled thermal peak-ΔT runaway proxy filter.

**Remaining:** none for core C0 loop (hotspot ΔT proxy wired).

Implementation steps:
1. Add objectives/constraints:
   - maximize selectivity,
   - maximize coupling efficiency,
   - limit peak E-field and/or loss tangent (arcing proxy),
   - optionally cap hotspot ΔT (runaway proxy).
2. Implement as a Pareto search in Optuna (multi-objective) and store the Pareto front.
3. Add selection logic for a “recommended design” based on user weights.

Done when:
- `scripts/run_multi_search.py` can output a Pareto front with coupling + safety metrics.

### C1 (P1) Robust optimization under uncertainty (frequency, load placement, materials)

**Status (partial):** ``--robust material`` on pipeline samples moisture/PSD scenarios via
``evaluate_material_robust()``; min-selectivity gate reuses ``_robust_gate``.

Implementation steps:
1. Frequency robustness (already partially present): expand from mean/min selectivity to
   coupling + safety robustness over ISM band.
2. Load/layout robustness: integrate `ensemble.py` realizations into objectives.
3. Material uncertainty: represent ε as distributions or scenarios (moisture, PSD, T).

Done when:
- A design is promoted only if it stays acceptable across a declared uncertainty set.

## Epic D — Materials: from cited anchors to measured ore (industry blocker)

### D0 (P0) Measured ore dielectric ingest (freq, temperature, moisture)

**Status (done):** JSON schema in `measured_dielectrics.py` with ε(f,T,moisture) interpolation;
`data/measured_dielectrics/<deposit>.json` example dataset; ore JSON `measured_dielectrics.path`
(relative to ore file); `materials_from_ore()` prefers measured curves; pipeline/search
`--ore-*` eval knobs; `scripts/ingest_deposit.py`; manifest provenance via `ore_summary`.

Implementation steps:
1. Define a data schema (JSON/CSV) for measured ε(f,T,moisture) for ore + gangue.
2. Add ingest tooling (`scripts/ingest_*`) to validate units, interpolate, and version data.
3. Update `materials_from_ore()` to prefer measured curves over literature anchors when present.

Done when:
- The pipeline can run end-to-end with a “named deposit” dataset and record provenance.

### D1 (P1) Microstructure realism: packing fraction, PSD, and mixing models

**Status (partial):** ore JSON `texture.packing_fraction` + `texture.psd` {d10,d50,d90};
`resolve_packing_fraction()` / `porosity_diluted_eps()` in Bruggeman gangue+target mixing;
PSD d50 → grain radius when `mean_grain_radius_m` absent.
**Scene:** per-grain `inclusion_radii_frac`, `sample_inclusion_layout()`, PSD layouts in `evaluate_ensemble`.

Implementation steps:
1. Add packing fraction + PSD to ore profiles and scene generators.
2. Implement mixing models beyond simple Bruggeman where appropriate (calibrated to measurement).

Done when:
- Effective ε used in simulation is justified by measurement, not only by literature.

## Epic E — EM–thermal–mechanical closure (industry relevance)

### E0 (P0) Calibrated thermal predictions on phantoms (close `bench_calibrated`)

**Status (partial):** ``data/measured_eps.example.json`` + ``evaluate_bench_gate()``;
pipeline records ``manifest.bench.gate``; ``--bench-enforce`` flag; M2 integration test
reaches ``bench_calibrated`` with synthetic openEMS + lab example JSON.
``validate_lab_measurements()``, model-vs-bench rank/ΔT tolerance in gate,
``scripts/run_bench_calibration.py``, ``scripts/ingest_lab_measurements.py``, CI M2 smoke.

Implementation steps:
1. Follow `docs/BENCH_PROTOCOL.md` to collect probe ε + ΔT bench JSON.
2. Automate ingest and comparison (`phantom_calibration.py`, `run_phantom_study.py`).
3. Promote to `bench_calibrated` only when drift tolerances and rank agreement pass.

Done when:
- A “better” design in simulation is also better on the bench for the same phantom batch.

### E1 (P1) Stress / liberation proxy tied to validated thermal gradients
Implementation steps:
1. Validate that the stress proxy correlates with measured fracture/weakening in a controlled test.
2. Add stress as an objective/constraint in composite presets (`DesignEvaluator`).

Done when:
- The “liberation” objective is anchored to a measurable outcome, not only a proxy.

## Epic F — Optimization methodology upgrades (research SOTA)

### F0 (P1) Adjoint / gradient-based optimization for high-dimensional geometry
Implementation steps:
1. Choose an adjoint-capable engine (FDTDX, MEEP adjoint, or differentiable surrogate).
2. Define a topology-optimization parameterization that respects manufacturing constraints.
3. Compare against Optuna baselines on the same objectives and budgets.

Done when:
- Demonstrated sample-efficiency improvement on a clearly high-dimensional actuator.

### F1 (P2) Multi-fidelity optimization (FDFD pre-screen → openEMS truth)

**Status (partial):** ``--openems-top-k`` on `mw-inv-pipeline` / `export_design.py` stores
`tpe_top_k` in search JSON and exports untuned + top-K FDFD winners for openEMS validation.
Use `scripts/update_run_with_openems.py` after Octave runs to refresh promotion tier.
**Promotion-aware:** openEMS run/synthesize skipped when FDFD gate fails (``--openems-force`` to override).

Implementation steps:
1. Use FDFD to filter candidates; periodically validate with openEMS and update a surrogate.
2. Add promotion-aware scheduling: don’t waste openEMS runs before FDFD gates pass.

Done when:
- OpenEMS calls are budgeted and measurably improve final designs vs pure black-box search.

## Epic G — Reproducibility, packaging, and operational readiness

### G0 (P0) Package the library + CLIs (reduce “sys.path” script pattern)
Implementation steps:
1. Add `pyproject.toml` and make `mw_inv` installable.
2. Convert key scripts into console entrypoints (keep scripts as thin wrappers if desired).
3. Pin or constrain dependency versions for reproducibility.

Done when:
- Users can run `pip install -e .` and call `mw-inv-pipeline ...` without path hacks.

### G1 (P0) Data hygiene: ignore generated artifacts by default

**Status (done):** ``.gitignore`` ignores ``data/runs/``, ``data/design_exports/``, and
top-level ``data/*.{json,npz,png}`` while keeping ``*.example.json``, ``*.template.json``,
and ``data/benchmarks|ores|templates/``. See ``data/README.md``.

Implementation steps:
1. Update `.gitignore` to ignore `data/runs/`, `data/design_exports/`, and other generated dirs.
2. Add a `data/README.md` describing what’s source vs generated.

Done when:
- A clean run does not produce untracked files that look like “source artifacts”.

### G2 (P1) Container / environment lock for CI parity
Implementation steps:
1. Provide a `Dockerfile` (or `uv.lock`/`poetry.lock`) that matches CI Python and deps.
2. Add a “reproduce CI locally” doc snippet.

Done when:
- A new user can reproduce CI results with one command.

## Epic H — Manufacturing handoff and tolerances (industry blocker)

### H0 (P1) CAD/param export with tolerances and build notes
Implementation steps:
1. Define the authoritative parameter set for the buildable cavity family.
2. Emit CAD-friendly exports (at minimum: dimensioned drawings + openEMS CSXCAD geometry).
3. Add tolerance propagation in robustness checks (± manufacturing and placement errors).

Done when:
- A selected design can be handed to a mechanical build with tolerances and acceptance checks.
