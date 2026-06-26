from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mw_inv.benchmarks import run_benchmarks, write_report
from mw_inv.design_evaluator import DesignEvaluator
from mw_inv.design_export import export_all_cases, load_search_cases
from mw_inv.fdfd import Grid
from mw_inv.geometry import CavityParams, Materials
from mw_inv.materials import DEFAULT_PAIR, PAIRS
from mw_inv.ore_profiles import (
    cavity_params_from_ore,
    load_ore_profile,
    materials_from_ore,
    ore_summary,
)
from mw_inv.openems_export import write_calibration_model
from mw_inv.openems_runner import (
    octave_available,
    run_openems_exports,
    synthesize_port_dumps,
)
from mw_inv.promotion import PromotionTier, meets_tier
from mw_inv.provenance import default_provenance
from mw_inv.run_manifest import RunManifest, default_run_dir
from mw_inv.run_refresh import apply_triangulation_refresh
from mw_inv.search import best, evaluate_params, optuna_search, random_search, top_k_trials, trial_to_dict
from mw_inv.validation_gate import GateThresholds


def _robustness_block(
    grid: Grid,
    materials: Materials,
    *,
    untuned: CavityParams,
    best_params: CavityParams,
    mode: str,
    n_realizations: int,
    n_grains: int,
    n_freqs: int,
    seed: int,
) -> dict:
    """Compute optional robustness summaries for untuned vs best."""
    from mw_inv.ensemble import (
        evaluate_ensemble,
        evaluate_frequency_robust,
        evaluate_frequency_robust_ensemble,
    )

    out: dict[str, object] = {"mode": mode}
    if mode in ("ensemble", "freq_ensemble"):
        out["untuned_ensemble"] = evaluate_ensemble(
            grid, untuned, materials, n_realizations=n_realizations, n_grains=n_grains, seed=seed,
        ).to_dict()
        out["best_ensemble"] = evaluate_ensemble(
            grid, best_params, materials, n_realizations=n_realizations, n_grains=n_grains, seed=seed,
        ).to_dict()
    if mode in ("freq", "freq_ensemble"):
        if mode == "freq":
            out["untuned_freq"] = evaluate_frequency_robust(
                grid, untuned, materials, pair_label=None, n_freqs=n_freqs,
            ).to_dict()
            out["best_freq"] = evaluate_frequency_robust(
                grid, best_params, materials, pair_label=None, n_freqs=n_freqs,
            ).to_dict()
        else:
            out["untuned_freq_ensemble"] = evaluate_frequency_robust_ensemble(
                grid,
                untuned,
                materials,
                pair_label=None,
                n_realizations=n_realizations,
                n_grains=n_grains,
                seed=seed,
                n_freqs=n_freqs,
            ).to_dict()
            out["best_freq_ensemble"] = evaluate_frequency_robust_ensemble(
                grid,
                best_params,
                materials,
                pair_label=None,
                n_realizations=n_realizations,
                n_grains=n_grains,
                seed=seed,
                n_freqs=n_freqs,
            ).to_dict()
    return out


def _robust_gate(block: dict, *, min_improvement: float, floor: float) -> dict:
    """Compute a simple pass/fail from a robustness block."""
    mode = block.get("mode")
    if mode == "none" or mode is None:
        return {"passed": True, "detail": "robustness disabled"}

    def pick_min(key: str) -> float | None:
        rep = block.get(key)
        if not isinstance(rep, dict):
            return None
        v = rep.get("min_selectivity")
        return None if v is None else float(v)

    if mode == "ensemble":
        u = pick_min("untuned_ensemble")
        b = pick_min("best_ensemble")
    elif mode == "freq":
        u = pick_min("untuned_freq")
        b = pick_min("best_freq")
    elif mode == "freq_ensemble":
        u = pick_min("untuned_freq_ensemble")
        b = pick_min("best_freq_ensemble")
    else:
        return {"passed": False, "detail": f"unknown robustness mode {mode!r}"}

    if u is None or b is None:
        return {"passed": False, "detail": "missing robustness min_selectivity fields"}

    delta = b - u
    ok = (b >= floor) and (delta >= min_improvement)
    return {
        "passed": bool(ok),
        "mode": mode,
        "untuned_min_selectivity": u,
        "best_min_selectivity": b,
        "delta_min_selectivity": delta,
        "floor": float(floor),
        "min_improvement": float(min_improvement),
        "detail": f"best min sel {b:.4f}, untuned {u:.4f}, Δ={delta:.4f} (floor {floor}, min Δ {min_improvement})",
    }


def _write_bench_artifacts(
    run_dir: Path,
    *,
    phantom_label: str,
    measured_eps_path: str | None,
    lab_measurements_path: str | None,
    bench_grid: int,
    bench_trials: int,
    bench_seed: int,
    run_study: bool,
) -> dict[str, str | None]:
    """Write bench phantom calibration artifacts; returns paths for manifest.bench."""
    out: dict[str, str | None] = {
        "phantom_label": phantom_label,
        "measured_eps_path": measured_eps_path,
        "lab_measurements_path": lab_measurements_path,
        "probe_calibration_report_path": None,
        "phantom_study_report_path": None,
        "gate": None,
    }
    if measured_eps_path and Path(measured_eps_path).is_file():
        from mw_inv.phantom_calibration import compare_measured_vs_anchor, evaluate_bench_gate

        report = compare_measured_vs_anchor(phantom_label, measured_eps_path)
        p = run_dir / "probe_calibration_report.json"
        p.write_text(json.dumps(report, indent=2))
        out["probe_calibration_report_path"] = str(p)
        bench_gate = evaluate_bench_gate(
            phantom_label,
            measured_eps_path,
            lab_measurements_path,
        )
        out["gate"] = bench_gate.to_dict()

    if run_study or lab_measurements_path:
        from mw_inv.fdfd import Grid
        from mw_inv.phantom import compare_lab_measurement, load_lab_measurements, predict_lab_outcome

        grid = Grid(nx=bench_grid, ny=bench_grid, Lx=0.36, Ly=0.36)
        pred = predict_lab_outcome(
            phantom_label,
            grid,
            n_opt_trials=bench_trials,
            seed=bench_seed,
            measured_eps_path=measured_eps_path,
        )
        comps = []
        if lab_measurements_path:
            for row in load_lab_measurements(lab_measurements_path):
                if row.get("phantom") != phantom_label:
                    continue
                comps.append(
                    compare_lab_measurement(
                        pred,
                        float(row["measured_delta_T_K"]),
                        row.get("measured_selectivity"),
                        untuned_measured_delta_T_K=row.get("untuned_measured_delta_T_K"),
                    ).to_dict()
                )
        payload = {"prediction": pred.to_dict(), "comparisons": comps}
        p2 = run_dir / "phantom_study_report.json"
        p2.write_text(json.dumps(payload, indent=2))
        out["phantom_study_report_path"] = str(p2)

    return out


def _run_search(
    grid: Grid,
    materials: Materials,
    *,
    trials: int,
    seed: int,
    legacy: bool,
    base_params: CavityParams | None = None,
    store_top_k: int = 0,
) -> dict:
    params0 = base_params or CavityParams()
    base = evaluate_params(grid, params0, materials, legacy=legacy)
    t0 = time.time()
    rnd = random_search(grid, trials, seed=seed, base=params0, materials=materials, legacy=legacy)
    t1 = time.time()
    tpe = optuna_search(grid, trials, seed=seed, base=params0, materials=materials, legacy=legacy)
    rnd_best = best(rnd)
    tpe_best = best(tpe)
    summary = {
        "trials": trials,
        "seed": seed,
        "materials": materials.pair_label or "custom",
        "search_mode": "legacy" if legacy else "manufacturable",
        "baseline_untuned": {
            "selectivity": base.selectivity,
            "contrast": base.contrast,
        },
        "random_search": {
            "best_selectivity": rnd_best.selectivity,
            "best_contrast": rnd_best.contrast,
            "best_params": rnd_best.params,
            "seconds": round(t1 - t0, 1),
        },
        "tpe_search": {
            "best_selectivity": tpe_best.selectivity,
            "best_contrast": tpe_best.contrast,
            "best_params": tpe_best.params,
            "seconds": round(time.time() - t1, 1),
        },
    }
    if store_top_k > 0:
        summary["tpe_top_k"] = [trial_to_dict(t) for t in top_k_trials(tpe, store_top_k)]
        summary["openems_top_k"] = store_top_k
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Tier-1 promotion pipeline")
    ap.add_argument("--materials", choices=sorted(PAIRS), default=None)
    ap.add_argument("--ore", type=str, default=None, help="QEMSCAN/assay ore JSON (data/ores/)")
    ap.add_argument("--preset", default="em")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--grid", type=int, default=71)
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--run-dir", default=None, help="output directory (default: data/runs/<timestamp>)")
    ap.add_argument("--search", default=None, help="reuse existing search_summary.json")
    ap.add_argument("--skip-benchmarks", action="store_true")
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument(
        "--export-tier",
        default="fdfd_optimised",
        choices=[t.value for t in PromotionTier],
        help="minimum tier required to write openEMS export bundle",
    )
    ap.add_argument("--phantom", default=None, help="phantom label for bench_calibrated tier")
    ap.add_argument(
        "--measured-eps",
        default=None,
        help="probe-measured ε JSON (e.g. data/measured_eps.example.json)",
    )
    ap.add_argument("--lab-measurements", default=None, help="bench measurements JSON for phantom compare")
    ap.add_argument("--bench-study", action="store_true", help="run phantom prediction/compare step")
    ap.add_argument("--bench-enforce", action="store_true", help="fail pipeline if bench gate fails")
    ap.add_argument("--bench-grid", type=int, default=61)
    ap.add_argument("--bench-trials", type=int, default=12)
    ap.add_argument("--bench-seed", type=int, default=7701)
    ap.add_argument(
        "--robust",
        choices=("none", "freq", "ensemble", "freq_ensemble"),
        default="none",
        help="optional robustness evaluation for untuned vs best",
    )
    ap.add_argument("--robust-realizations", type=int, default=6)
    ap.add_argument("--robust-grains", type=int, default=5)
    ap.add_argument("--robust-n-freqs", type=int, default=5)
    ap.add_argument("--robust-seed", type=int, default=2206)
    ap.add_argument("--robust-enforce", action="store_true", help="fail the pipeline if robustness gate fails")
    ap.add_argument("--robust-floor", type=float, default=0.0, help="minimum acceptable robust min selectivity")
    ap.add_argument("--robust-min-improvement", type=float, default=0.0, help="minimum robust min-selectivity improvement vs untuned")
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--openems-dump-dir", default=None, help="openEMS dump root (expects <case>/Et/Et_0000.h5)")
    ap.add_argument(
        "--openems-top-k",
        type=int,
        default=0,
        help="FDFD pre-screen: export/triangulate untuned + top-K TPE trials (0 = legacy 3-case)",
    )
    ap.add_argument(
        "--run-openems",
        action="store_true",
        help="after export, run openEMS via Octave (run_openems_all.m) and refresh triangulation",
    )
    ap.add_argument(
        "--synthesize-openems-dumps",
        action="store_true",
        help="after export, write synthetic port_metrics.json per case (CI/dev; no Octave)",
    )
    ap.add_argument(
        "--openems-octave",
        default="octave",
        help="Octave executable for --run-openems",
    )
    ap.add_argument(
        "--openems-timeout",
        type=float,
        default=None,
        help="timeout in seconds for --run-openems (default: none)",
    )
    ap.add_argument(
        "--openems-force",
        action="store_true",
        help="run openEMS even when the FDFD validation gate failed",
    )
    ap.add_argument(
        "--gate-min-improvement",
        type=float,
        default=0.01,
        help="minimum FDFD selectivity gain (tpe_best − untuned) for validation gate",
    )
    args = ap.parse_args(argv)
    if args.materials and args.ore:
        ap.error("--materials and --ore are mutually exclusive")
    if args.run_openems and args.synthesize_openems_dumps:
        ap.error("--run-openems and --synthesize-openems-dumps are mutually exclusive")
    if (args.run_openems or args.synthesize_openems_dumps) and args.skip_export:
        ap.error("openEMS ingest requires export; omit --skip-export")
    if args.run_openems and args.openems_dump_dir:
        ap.error("use either --run-openems or --openems-dump-dir, not both")

    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    ore_block: dict | None = None
    base_params = CavityParams()

    if args.ore:
        ore = load_ore_profile(args.ore)
        materials = materials_from_ore(ore)
        base_params = cavity_params_from_ore(ore, cavity_span_m=grid.Lx)
        ore_block = {**ore_summary(ore), "json_path": str(args.ore)}
        materials_label = materials.pair_label or "custom"
    else:
        materials_label = args.materials or DEFAULT_PAIR.label
        materials = Materials.from_pair(materials_label)

    manifest = RunManifest(
        run_id=run_id,
        materials=materials_label,
        preset=args.preset,
        cli={
            "argv": list(argv) if argv is not None else None,
            "args": {k: getattr(args, k) for k in vars(args)},
        },
        ore=ore_block or {},
        bench={},
    )
    # Repo root is .../src/mw_inv/cli/pipeline.py -> parents[3] is workspace root.
    manifest.provenance = default_provenance(Path(__file__).resolve().parents[3])
    if args.phantom:
        manifest.bench = _write_bench_artifacts(
            run_dir,
            phantom_label=args.phantom,
            measured_eps_path=args.measured_eps,
            lab_measurements_path=args.lab_measurements,
            bench_grid=args.bench_grid,
            bench_trials=args.bench_trials,
            bench_seed=args.bench_seed,
            run_study=bool(args.bench_study),
        )

    # --- benchmarks ---
    if not args.skip_benchmarks:
        bench_report = run_benchmarks()
        bench_path = run_dir / "benchmark_report.json"
        write_report(bench_path, bench_report)
        manifest.benchmarks_path = str(bench_path)
        manifest.benchmarks_passed = bench_report.passed
        print(f"  benchmarks: {'PASS' if bench_report.passed else 'FAIL'} -> {bench_path}")

    # --- search ---
    search_path = Path(args.search) if args.search else run_dir / "search_summary.json"
    if args.search:
        manifest.search_summary = json.loads(search_path.read_text())
        manifest.search_path = str(search_path)
        print(f"  search: reused {search_path}")
    else:
        manifest.search_summary = _run_search(
            grid,
            materials,
            trials=args.trials,
            seed=args.seed,
            legacy=args.legacy,
            base_params=base_params,
            store_top_k=args.openems_top_k,
        )
        manifest.search_summary["grid"] = args.grid
        if ore_block:
            manifest.search_summary["ore"] = ore_block
        search_path.write_text(json.dumps(manifest.search_summary, indent=2))
        manifest.search_path = str(search_path)
        print(
            f"  search: TPE sel={manifest.search_summary['tpe_search']['best_selectivity']:.4f} -> {search_path}"
        )

    # --- evaluation snapshot (untuned + TPE best) ---
    ev = DesignEvaluator.from_preset(grid, args.preset, materials=materials, pair_label=materials_label)
    tpe_params = manifest.search_summary.get("tpe_search", {}).get("best_params", {})
    untuned_params = base_params
    manifest.evaluation = {
        "untuned": ev.evaluate(untuned_params).to_dict(),
        "tpe_best": ev.evaluate_dict(tpe_params, base=untuned_params).to_dict() if tpe_params else {},
    }
    if args.robust != "none" and tpe_params:
        from mw_inv.search import params_from_dict

        best_params = params_from_dict(tpe_params, base=untuned_params)
        rb = _robustness_block(
            grid,
            materials,
            untuned=untuned_params,
            best_params=best_params,
            mode=args.robust,
            n_realizations=args.robust_realizations,
            n_grains=args.robust_grains,
            n_freqs=args.robust_n_freqs,
            seed=args.robust_seed,
        )
        manifest.evaluation["robustness"] = rb
        gate = _robust_gate(rb, min_improvement=args.robust_min_improvement, floor=args.robust_floor)
        manifest.evaluation["robust_gate"] = gate
        if args.robust_enforce and not gate.get("passed", False):
            manifest.notes.append(f"robustness gate failed: {gate.get('detail')}")
            manifest_path = run_dir / "manifest.json"
            manifest.write(manifest_path)
            print(f"  robustness: FAIL -> {gate.get('detail')}")
            print(f"  manifest: {manifest_path}")
            raise SystemExit(3)

    # --- gate / triangulation (FDFD pre-check for export tier; openEMS refresh later) ---
    dump_dir = Path(args.openems_dump_dir) if args.openems_dump_dir else None
    top_k = args.openems_top_k if args.openems_top_k > 0 else None
    gate_thresholds = GateThresholds(min_fdfd_improvement=args.gate_min_improvement)
    refresh = apply_triangulation_refresh(
        manifest,
        run_dir,
        search_path=search_path,
        grid=grid,
        materials=materials,
        materials_label=materials_label,
        openems_dump_dir=dump_dir,
        top_k=top_k,
        gate_thresholds=gate_thresholds,
        triangulation_meta={
            "openems_top_k": args.openems_top_k if args.openems_top_k > 0 else None,
        },
    )
    gate = refresh.gate
    assessment = refresh.assessment
    print(f"  gate: {'PASS' if gate.passed else 'FAIL'}  promotion tier: {assessment.tier.value}")

    # --- export (tier-guarded) ---
    export_bundles = []
    export_tier = PromotionTier(args.export_tier)
    if not args.skip_export:
        if meets_tier(assessment.tier, export_tier):
            export_dir = run_dir / "design_exports"
            write_calibration_model(export_dir / "calibration_cavity.m")
            cases = load_search_cases(search_path, top_k=top_k)
            export_bundles = export_all_cases(export_dir, cases, materials, grid_n=args.grid)
            manifest.export_dir = str(export_dir)
            manifest.export_summary = {
                "openems_top_k": args.openems_top_k if args.openems_top_k > 0 else None,
                "bundles": [
                    {
                        "label": b.label,
                        "fdfd_selectivity": b.fdfd_selectivity,
                        "openems_model": str(b.openems_path.name),
                        "manifest": str(b.manifest_path.name),
                    }
                    for b in export_bundles
                ],
            }
            print(f"  export: {len(export_bundles)} cases -> {export_dir}")
        else:
            manifest.notes.append(
                f"export skipped: tier {assessment.tier.value} < required {export_tier.value}"
            )
            print(f"  export: SKIPPED (tier {assessment.tier.value} < {export_tier.value})")

    # --- openEMS run / synthetic dumps → refresh triangulation ---
    post_openems = dump_dir
    if args.run_openems or args.synthesize_openems_dumps:
        if not gate.passed and not args.openems_force:
            manifest.notes.append("openEMS skipped: FDFD gate failed (use --openems-force to override)")
            print("  openEMS: SKIPPED (FDFD gate failed)")
        elif not manifest.export_dir:
            manifest.notes.append("openEMS step skipped: export bundle missing")
            print("  openEMS: SKIPPED (no export bundle)")
        else:
            export_dir = Path(manifest.export_dir)
            openems_record: dict[str, object] = {
                "mode": "octave" if args.run_openems else "synthetic",
                "export_dir": str(export_dir),
            }
            if args.run_openems:
                if not octave_available(args.openems_octave):
                    raise SystemExit(
                        f"Octave not found ({args.openems_octave!r}). "
                        "Install Octave + openEMS, or use --synthesize-openems-dumps for CI."
                    )
                result = run_openems_exports(
                    export_dir,
                    octave_cmd=args.openems_octave,
                    timeout_s=args.openems_timeout,
                )
                openems_record.update({
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-4000:] if result.stdout else "",
                    "stderr_tail": result.stderr[-4000:] if result.stderr else "",
                })
                if result.returncode != 0:
                    manifest.notes.append(f"openEMS batch failed (rc={result.returncode})")
                    print(f"  openEMS: FAIL (rc={result.returncode})")
                else:
                    print(f"  openEMS: batch complete -> {result.dump_dir}")
                post_openems = result.dump_dir
            else:
                post_openems = synthesize_port_dumps(export_dir, export_bundles)
                print(f"  openEMS: synthetic port dumps -> {post_openems}")

            manifest.cli.setdefault("openems", []).append(openems_record)
            refresh = apply_triangulation_refresh(
                manifest,
                run_dir,
                search_path=search_path,
                grid=grid,
                materials=materials,
                materials_label=materials_label,
                openems_dump_dir=post_openems,
                top_k=top_k,
                gate_thresholds=gate_thresholds,
                triangulation_meta={
                    "openems_top_k": args.openems_top_k if args.openems_top_k > 0 else None,
                    "openems_mode": openems_record["mode"],
                },
            )
            gate = refresh.gate
            assessment = refresh.assessment
            print(
                f"  gate (openEMS): {'PASS' if gate.passed else 'FAIL'}  "
                f"promotion tier: {assessment.tier.value}"
            )

    if args.phantom and manifest.bench.get("gate"):
        bench_passed = bool(manifest.bench["gate"].get("passed"))
        print(f"  bench gate: {'PASS' if bench_passed else 'FAIL'}")
        if args.bench_enforce and not bench_passed:
            manifest_path = run_dir / "manifest.json"
            manifest.write(manifest_path)
            raise SystemExit(4)

    manifest_path = run_dir / "manifest.json"
    manifest.write(manifest_path)
    print(f"  manifest: {manifest_path}")
    print(f"  promotion tier: {assessment.tier.value}")


if __name__ == "__main__":
    main()
