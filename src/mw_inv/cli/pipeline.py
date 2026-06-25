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
from mw_inv.promotion import PromotionTier, meets_tier
from mw_inv.run_manifest import RunManifest, default_run_dir, finalize_promotion
from mw_inv.search import best, evaluate_params, optuna_search, random_search
from mw_inv.solver_triangulation import triangulate_from_search, write_triangulation_report
from mw_inv.validation_gate import evaluate_gate


def _run_search(
    grid: Grid,
    materials: Materials,
    *,
    trials: int,
    seed: int,
    legacy: bool,
    base_params: CavityParams | None = None,
) -> dict:
    params0 = base_params or CavityParams()
    base = evaluate_params(grid, params0, materials, legacy=legacy)
    t0 = time.time()
    rnd = random_search(grid, trials, seed=seed, materials=materials, legacy=legacy)
    t1 = time.time()
    tpe = optuna_search(grid, trials, seed=seed, materials=materials, legacy=legacy)
    rnd_best = best(rnd)
    tpe_best = best(tpe)
    return {
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
    ap.add_argument("--measured-eps", default="data/measured_eps.json")
    ap.add_argument("--legacy", action="store_true")
    args = ap.parse_args(argv)
    if args.materials and args.ore:
        ap.error("--materials and --ore are mutually exclusive")

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
        ore=ore_block or {},
        bench={
            "phantom_label": args.phantom,
            "measured_eps_path": args.measured_eps if args.phantom else None,
        },
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
    manifest.evaluation = {
        "untuned": ev.evaluate(CavityParams()).to_dict(),
        "tpe_best": ev.evaluate_dict(tpe_params).to_dict() if tpe_params else {},
    }

    # --- gate / triangulation ---
    rows = triangulate_from_search(search_path, grid, materials)
    tri_path = run_dir / "solver_triangulation.json"
    write_triangulation_report(tri_path, rows, materials_label=materials_label)
    gate = evaluate_gate(rows)
    gate_path = run_dir / "validation_gate_report.json"
    gate_payload = {
        "materials": materials_label,
        "search_source": str(search_path),
        "gate": gate.to_dict(),
        "triangulation": [r.to_dict() for r in rows],
    }
    gate_path.write_text(json.dumps(gate_payload, indent=2))

    manifest.triangulation_path = str(tri_path)
    manifest.triangulation = {
        "rows": [r.to_dict() for r in rows],
        "rank_agreement": gate.rank_agreement,
    }
    manifest.gate_path = str(gate_path)
    manifest.gate = gate.to_dict()

    assessment = finalize_promotion(manifest)
    print(f"  gate: {'PASS' if gate.passed else 'FAIL'}  promotion tier: {assessment.tier.value}")

    # --- export (tier-guarded) ---
    export_tier = PromotionTier(args.export_tier)
    if not args.skip_export:
        if meets_tier(assessment.tier, export_tier):
            export_dir = run_dir / "design_exports"
            write_calibration_model(export_dir / "calibration_cavity.m")
            cases = load_search_cases(search_path)
            bundles = export_all_cases(export_dir, cases, materials, grid_n=args.grid)
            manifest.export_dir = str(export_dir)
            manifest.export_summary = {
                "bundles": [
                    {
                        "label": b.label,
                        "fdfd_selectivity": b.fdfd_selectivity,
                        "openems_model": str(b.openems_path.name),
                        "manifest": str(b.manifest_path.name),
                    }
                    for b in bundles
                ],
            }
            print(f"  export: {len(bundles)} cases -> {export_dir}")
        else:
            manifest.notes.append(
                f"export skipped: tier {assessment.tier.value} < required {export_tier.value}"
            )
            print(f"  export: SKIPPED (tier {assessment.tier.value} < {export_tier.value})")

    manifest_path = run_dir / "manifest.json"
    manifest.write(manifest_path)
    print(f"  manifest: {manifest_path}")
    print(f"  promotion tier: {assessment.tier.value}")


if __name__ == "__main__":
    main()

