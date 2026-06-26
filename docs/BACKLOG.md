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
3. **M3 — Deposit-calibrated** (measured ore ε(f,T,moisture) + validation) — **tier machinery done**; **live deposit closure not done**
4. **M4 — Pilot-ready** (safety + repeatability + throughput) — **tier machinery done**; **pilot-scale evidence not done**

> **Note:** M3/M4 mean the promotion ladder and gates exist — not that a real deposit or pilot has passed them. See [Thin-slice → useful gap map](#thin-slice--useful-gap-map) below.

## Thin-slice → useful gap map

Cross-reference from [docs/MATURITY.md](MATURITY.md) / expert review — what still separates “research slice” from “field-useful codebase” **beyond bench hardware**.

| Gap | Backlog |
|-----|---------|
| 3D port-matched FDTD as **primary** truth (FDFD prescreen only) | **A1**, **B0**, **F1** |
| True PEC / finite conductivity (not lossy-Im(ε) walls) | **A2**, **B1** |
| Multi-mode cavity + load-sensitive matching | **A1**, **B0** |
| Discrete / statistical particle bed (not smeared two-phase ε) | **D2** |
| Moving bed / residence time / throughput | **I1** |
| Industrial KPIs: energy/t, gangue ΔT budget, worst-case particle | **I0** |
| Selectivity as secondary to cost/throughput/safety objectives | **I0**, **C0** |
| Nonlinear ε(T,f, chemistry) + μ(T) closed loop in thermal | **E2** |
| Phase transitions (pyrite→pyrrhotite, Curie, moisture loss) | **E2** |
| Frequency as **constrained** control (magnetron band, not free knob) | **C2** |
| Robust design over deposit **envelope** (grade/moisture/PSD box) | **C1**, **D3** |
| Uncertainty propagation (ε error → ΔT / selectivity CI) | **C3**, **D3** |
| Manufacturing tolerances + placement errors in robust search | **H0**, **C1** |
| High-D / topology optimization with build constraints | **F0**, **H0** |
| Hand-param geometry → authoritative CAD family | **H0**, **B0** |
| Auto-calibration: probe + assay → effective ε → manifest diff | **D4** |
| Deposit campaigns / versioned mine-block models | **D4**, **G3** |
| Manufacturing handoff: STEP/drawings + tuning procedure | **H0** |
| Effective-medium validation vs bulk probe on ROM | **D1**, **D3** |
| Liberation / comminution / economics linkage | **E1**, **I2** (P2) |
| Pilot-scale power scaling | **I1** |

## Backlog mapping (deferred from “ports + productization” scope)

- **Industrial spec + constraints**: Epics **C**, **I** (objectives/constraints/safety/energy/throughput).
- **Coupled physics where it changes decisions**: Epic **E** (phantoms, nonlinear ε–thermal, stress/liberation).
- **Staged calibration/validation loop**: Epics **E**, **D** (measured ore ε, auto-calibration, campaigns).
- **Robust / uncertainty-aware optimization**: Epics **C**, **D** (materials, envelopes, uncertainty CI).
- **Particle bed + discrete heating**: Epic **D2** (not only effective medium).
- **3D truth-first workflow**: Epics **A**, **B**, **F1**.

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

**Status (partial):** openEMS export + triangulation + ``--run-openems`` exist; FDFD remains the **default optimization evaluator**. ``validation_gate`` adds ``openems_fdfd_coupling_ratio``; ``metal_model.py`` + ``scripts/run_metal_model_check.py`` for canonical plate alignment; synthetic dumps track FDFD ``coupling_eff``.

**Remaining (P0 for usefulness):** invert the workflow — openEMS (or MEEP 3D) as **sign-off**, FDFD as **cheap prescreen only**; gate blocks export when 3D port metrics absent.

Implementation steps:
1. Define a minimal 3D geometry family that matches a buildable cavity + feed + plate.
2. Ensure openEMS export supports these 3D primitives with consistent naming and dump layout.
3. Expand `validation_gate` to optionally require 3D agreement once openEMS data exists.
4. Pipeline default: FDFD top-K → openEMS truth → promote only on 3D metrics (extend `--openems-top-k`).

Done when:
- The repo can produce 3D openEMS models for the same param family used in optimization,
  and the gate can enforce rank agreement.
- Documented policy: **no external “design recommendation” without solver_triangulated data**.

### B2 (P0) Dirichlet PEC and replace lossy-Im(ε) structural boundaries
**Why:** Lossy “PEC” absorbs power and can fake selectivity gains (see docs/MATURITY.md coupling pathology).

**Status (partial):** ``CavityParams.structure_model`` — ``dirichlet`` (default, Ez=0 rows on ``pec_mask`` in ``fdfd.py``) vs legacy ``lossy_imag``; pipeline ``--structure-model``; gate ``fdfd_pec_loss_fraction_max`` wired. Legacy baffle pathology preserved under ``lossy_imag`` + point feed (``tests/test_coupling.py``).

**Remaining:** openEMS metal vs FDFD alignment (B0); deprecate ``lossy_imag`` for promotion runs.
1. Implement true PEC (Ez=0) rows in FDFD or deprecate baffle path for promotion.
2. openEMS: verify metal walls vs Im(ε) plate; align FDFD pre-screen with openEMS metal model.
3. Gate: fail designs where `pec_loss_fraction` exceeds tolerance.

Done when:
- Optimized designs cannot win by routing power into structural Im(ε) absorbers.
- FDFD/openEMS structural loss metrics agree within declared tolerance on canonical cases.

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
``MultiTrial`` records ``gangue_power_fraction`` + ``min_particle_fraction``; pipeline
``--weight-gangue-budget`` / ``--weight-particle-floor`` extend Pareto recommendation weights.
``--multi-industrial`` runs 4-objective NSGA-II (selectivity, coupling, gangue budget, particle floor).

**Remaining:** weighted Pareto picker for >2 objectives when not using ``--multi-industrial``; ``composite:industrial`` preset available for single-design scoring.

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

### C2 (P1) Constrained frequency control (magnetron-realistic ISM band)

**Status (partial):** ``ism_band.py`` — ``IsmBandMode.FIXED | FULL | TUNABLE``; ``evaluate_frequency_robust(..., band=)``; pipeline ``--ism-band`` + ``--ism-band-samples``.
Search: ``--search-ism-band fixed|full|none`` constrains Optuna ``freq_hz`` (fixed=2.45 GHz geometry-only; full=legacy band).
``optuna_search`` / ``optuna_multi_search`` accept ``freq_robust=True`` to optimise min-selectivity over ISM samples during search.

**Why:** Frequency sweeps can move selectivity as much as geometry; free continuous f is not an industrial actuator.

Implementation steps:
1. Replace unconstrained `freq_hz` search with band constraints: fixed 2.45 GHz, or ±δ MHz, or stub-tuner discrete steps.
2. Model magnetron load-pull sensitivity (optional: S11 vs f coupling from openEMS sweeps).
3. Frequency-robust search optimizes **worst-case over allowed band**, not mean over arbitrary range.

Done when:
- Pipeline/search document which frequency DOFs are physical for a given applicator class.
- Robust designs pass min-selectivity over **declared ISM tolerance**, not best-case f.

### C3 (P1) Uncertainty propagation to outcomes (ε error → ΔT / selectivity CI)

**Status (partial):** ``MaterialRobustReport`` p05/p95 selectivity + min/mean coupling over scenarios; ``evaluate_material_robust``.
``evaluate_uncertainty_gate()`` — pipeline ``--robust-p05-floor`` + ``--robust-p05-enforce`` (exit 8) on material or freq robust blocks.

**Why:** Literature and probe ε have spread; point designs are fragile.

Implementation steps:
1. Represent ε inputs as intervals or scenarios (already partial in `material_scenarios.py`).
2. Propagate through evaluation to report percentiles on selectivity, coupling, ΔT.
3. Promotion requires acceptable **lower confidence bound**, not mean only.

Done when:
- Manifest records uncertainty bands; gate can fail on 5th-percentile selectivity or ΔT.

### C1 (P1) Robust optimization under uncertainty (frequency, load placement, materials)

**Status (partial):** ``--robust material|freq|ensemble`` on pipeline; ``--ism-band fixed|full|tunable`` for frequency robustness; min-selectivity ``_robust_gate``.

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

**Remaining:** validate effective ε against bulk probe; mixing beyond Bruggeman.

Implementation steps:
1. Add packing fraction + PSD to ore profiles and scene generators.
2. Implement mixing models beyond simple Bruggeman where appropriate (calibrated to measurement).

Done when:
- Effective ε used in simulation is justified by measurement, not only by literature.

### D2 (P0) Discrete particle bed (statistical or resolved), not smeared two-phase ε alone

**Status (partial):** ``evaluate_particle_power()`` — per-inclusion disk absorbed power + charge fractions; ``ParticlePowerReport``.
``p05_particle_fraction`` / ``p95_particle_fraction`` in ``DesignReport.foms``; ``particle_tail_gate.py`` + pipeline ``--particle-p05-floor`` / ``--particle-tail-enforce`` (exit 11).

**Why:** Sorting, liberation, and runaway depend on **particle-level** heating variance; bulk selectivity hides hot/cold grains.

Implementation steps:
1. Represent charge as N discrete inclusions with PSD-sampled radii, positions, and optional contact gaps.
2. Report distribution of per-particle absorbed power / ΔT (mean, p95, worst-case).
3. Add objectives on tail risk (e.g. minimize gangue p95 heating, maximize target p05 heating).

Done when:
- Ensemble evaluation reports particle-level statistics, not only bulk selectivity.
- Robust search can optimize worst-case particle over layout realizations.

### D3 (P1) Deposit envelope: one design vs grade / moisture / PSD box

**Status (partial):** ``deposit_envelope.py`` + ``scripts/evaluate_deposit_envelope.py`` — min/mean selectivity & coupling over ore directory; pipeline ``--ore-envelope`` / ``--campaign`` writes ``deposit_envelope_report.json`` + ``deposit_envelope_gate``; promotion ``deposit_calibrated`` requires envelope pass when gate present; ``--envelope-enforce``.

**Why:** ROM variability is the norm; point-ore JSON optimization is fragile.

Implementation steps:
1. Define deposit envelopes (ranges on fractions, moisture, PSD) from campaign data or QEMSCAN batches.
2. Evaluate designs over envelope scenarios; require min performance across box.
3. Wire to promotion: `deposit_calibrated` requires envelope pass, not single ore file.

Done when:
- Pipeline accepts `--ore-envelope` or ore directory + manifest; gate uses min-over-scenarios. **(done: envelope gate + promotion hook; `--target-tier deposit_calibrated` requires campaign/envelope + tier enforce exit 7)**

### D4 (P1) Auto-calibration loop: probe + assay → effective ε → manifest diff
**Why:** Useful codebase is the system of record for a deposit, not a one-shot ingest script.

**Status (partial):** ``deposit_calibration.py`` — Bruggeman vs ``measured_dielectrics`` diff; pipeline ``--calibrate-deposit`` with ``--ore`` writes ``deposit_calibration_report.json``.
``--calibrate-baseline`` writes ``deposit_calibration_changelog.json``; ``--calibrate-deposit-enforce`` (exit 9); promotion checks ``passes_calibration`` when report present.

Implementation steps:
1. Fit effective medium (or update measured library) from open-coax bulk + QEMSCAN/assay fractions.
2. Version datasets (`dataset_id`, date, operator); diff manifests when calibration updates.
3. Regression: alert when new lab data moves ranked designs or gate outcomes.

Done when:
- `scripts/ingest_deposit.py` (or successor) can **update** a deposit model from new probe rows and re-run gate with changelog.

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

### E2 (P0) Nonlinear EM–thermal–material evolution (ε, μ, chemistry vs T)

**Status (partial):** ``phase_transitions.py`` — pyrite→pyrrhotite at 600 K; ``build_scene_at_T`` applies ``mineral_key_at_T`` to target mask ε.
``TransientConfig.evolve_properties`` toggles frozen RT ε vs periodic ε(T)+phase refresh in ``simulate_transient``.
``ThermalConfig.evolve_properties`` + ``hotspot_gate.py`` — evolved vs frozen peak ΔT; ``--hotspot-frozen`` / ``--hotspot-gate-enforce`` (exit 10).

**Why:** Pyrite oxidation, sulphide melting, Curie loss on magnetite, and moisture loss change ε during heating — illustrative ε(T) tables are not a closed loop.

Implementation steps:
1. Extend thermal stepping to update ε **and** μ from temperature-dependent tables / phase rules (not only Arrhenius ε″ ramp).
2. Optional: simple phase-transition hooks (e.g. pyrite→pyrrhotite band) with literature anchors.
3. Re-evaluate EM field periodically (or use surrogate) when ε change exceeds threshold.
4. Report whether runaway is **model-predicted** vs self-limited under evolving properties.

Done when:
- Transient thermal + periodic EM refresh shows qualitatively different outcomes vs frozen-ε run for at least one canonical sulphide case.
- Hotspot/runaway gate uses evolved properties, not RT ε only. **(partial: evolved default; frozen comparison in hotspot_gate)**

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
``openems_schedule.py`` — gate-aware case filter; ``--openems-budget``, ``--openems-include-untuned``; schedule metadata in export summary.

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

### G3 (P1) Deposit campaigns: versioned mine-block models as first-class inputs
**Why:** Literature scenario libraries (e.g. Forster manifest) test software; operations need campaign-scoped ore models.

**Status (partial):** ``campaign.py`` + ``data/campaigns/forster_literature_v1/campaign.json``; pipeline ``--campaign`` resolves ore globs and runs deposit envelope.

Implementation steps:
1. Schema for campaign id, date range, block ids, linked measured_dielectrics versions.
2. `discover_real_data_catalog` / pipeline resolve “active campaign” ore set.
3. Manifest records campaign provenance; diff across campaigns.

Done when:
- User can point pipeline at `data/campaigns/<id>/` and get reproducible promotion for that campaign only.

## Epic H — Manufacturing handoff and tolerances (industry blocker)

### H0 (P1) CAD/param export with tolerances and build notes

**Status (partial):** ``tuning_procedure.py`` — JSON + Markdown build/tuning steps from ``CavityParams``; ``write_tuning_procedure`` called from ``design_export`` per export case.
``manufacturing_tolerance.py`` — ``jitter_cavity_params`` + ``evaluate_manufacturing_robust``; pipeline ``--robust manufacturing`` + ``--manufacturing-tol``.

Implementation steps:
1. Define the authoritative parameter set for the buildable cavity family.
2. Emit CAD-friendly exports (at minimum: dimensioned drawings + openEMS CSXCAD geometry; stretch: STEP/STL for plate/feed).
3. Add tolerance propagation in robustness checks (± manufacturing and placement errors).
4. Emit **tuning procedure** (stub adjustment, plate position sequence, acceptance S11/ΔT checks).

Done when:
- A selected design can be handed to a mechanical build with tolerances and acceptance checks.

## Epic I — Industrial decision relevance (beyond selectivity)

### I0 (P0) Primary KPIs: energy per tonne, gangue budget, worst-case particle

**Status (partial):** ``industrial_metrics.py`` — ``gangue_power_fraction``, ``specific_energy_proxy_kwh_per_t``, ``throughput_proxy_t_per_h``, ``delivered_kw_proxy``; surfaced in ``DesignReport.foms``; ``composite:industrial`` preset weights coupling + gangue budget.

**Why:** Plants optimize cost and risk; absorbed-power selectivity alone does not drive CAPEX/OPEX.

Implementation steps:
1. Define derived metrics: `specific_energy_kWh_per_t`, gangue ΔT cap, target ΔT floor, magnetron utilization proxy.
2. Add composite presets (`DesignEvaluator`, `run_design_eval.py`) where selectivity is one term, not the headline.
3. Pipeline manifest defaults to industrial summary when `--preset industrial` (or similar).

Done when:
- A design recommendation JSON leads with energy/throughput/safety; selectivity is supporting data.
- Multi-objective front (C0) includes at least one industrial KPI.

### I1 (P1) Throughput, residence time, and pilot-scale power scaling
**Why:** Applicator sizing requires power × time × bed depth, not single steady-state field snapshot.

**Status (partial):** ``throughput_proxy_t_per_h`` and ``delivered_kw_proxy`` in ``IndustrialMetrics`` (nominal residence time + forward kW assumptions).

Implementation steps:
1. Model residence time (conveyor speed, bed depth, exposure window) → cumulative ΔT.
2. Scale absorbed power from simulation geometry to pilot kW (forward power, circulator loss).
3. Document what is validated vs extrapolated when moving from 100 W bench to kW pilot.

Done when:
- Report includes throughput estimate and power scaling assumptions; pilot_gate can check declared kW band.

### I2 (P2) Economics and comminution linkage (NPV / Bond / liberation index)
**Why:** Microwave-assisted comminution projects live or die on energy balance vs baseline mill.

Implementation steps:
1. Optional module: link ΔT/stress proxy to Bond work index shift or liberation metric (literature or user CSV).
2. Simple NPV / energy-cost calculator fed from manifest metrics (not blocking core physics).

Done when:
- Optional `--economics` block in manifest with documented inputs; no claim without user-supplied plant data.
