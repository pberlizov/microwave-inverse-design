from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mw_inv.design_evaluator import (
    COMPOSITE_PRESETS,
    best_design,
    optuna_design_search,
    preset_config,
)
from mw_inv.fdfd import Grid
from mw_inv.geometry import CavityParams, Materials
from mw_inv.materials import DEFAULT_PAIR, PAIRS
from mw_inv.ore_profiles import (
    cavity_params_from_ore,
    load_ore_profile,
    materials_from_ore,
    ore_summary,
)
from mw_inv.search import best, evaluate_params, optuna_search, random_search


PRESET_CHOICES = ("em", *COMPOSITE_PRESETS)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--grid", type=int, default=101)
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument(
        "--materials",
        choices=sorted(PAIRS),
        default=None,
        help="cited mineral pair (see mw_inv.materials / docs/MATERIALS.md)",
    )
    ap.add_argument(
        "--ore",
        type=str,
        default=None,
        help="QEMSCAN/assay ore JSON — auto pair + measured or Bruggeman ε (data/ores/)",
    )
    ap.add_argument("--ore-target-t", type=float, default=None, help="target phase T [K] for measured ε")
    ap.add_argument("--ore-gangue-t", type=float, default=None, help="gangue phase T [K] for measured ε")
    ap.add_argument("--ore-freq", type=float, default=None, help="frequency [Hz] for measured ε")
    ap.add_argument("--ore-moisture", type=float, default=None, help="moisture wt%% for measured ε")
    ap.add_argument("--out", type=str, default="data/search_summary.json")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="use abstract baffle geometry (pre step-4 search space)",
    )
    ap.add_argument(
        "--preset",
        choices=PRESET_CHOICES,
        default="em",
        help="evaluation preset (composite:* uses unified DesignEvaluator)",
    )
    ap.add_argument("--check-arcing", action="store_true")
    args = ap.parse_args(argv)
    if args.materials and args.ore:
        ap.error("--materials and --ore are mutually exclusive")

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    legacy = args.legacy
    ore_block: dict | None = None
    base_params = CavityParams()

    if args.ore:
        ore_path = Path(args.ore)
        ore = load_ore_profile(ore_path)
        ore_kw = dict(
            ore_profile_path=ore_path,
            target_T_K=args.ore_target_t if args.ore_target_t is not None else 298.0,
            gangue_T_K=args.ore_gangue_t if args.ore_gangue_t is not None else 298.0,
            freq_hz=args.ore_freq if args.ore_freq is not None else 2.45e9,
            moisture_wt_percent=args.ore_moisture,
        )
        materials = materials_from_ore(ore, **ore_kw)
        base_params = cavity_params_from_ore(ore, cavity_span_m=grid.Lx)
        ore_block = {
            **ore_summary(ore, **ore_kw),
            "json_path": str(ore_path.resolve()),
            "eval_conditions": {
                "target_T_K": ore_kw["target_T_K"],
                "gangue_T_K": ore_kw["gangue_T_K"],
                "freq_hz": ore_kw["freq_hz"],
                "moisture_wt_percent": ore_kw["moisture_wt_percent"],
            },
        }
        materials_label = materials.pair_label or "custom"
    else:
        pair_label = args.materials or DEFAULT_PAIR.label
        materials = Materials.from_pair(pair_label)
        materials_label = pair_label

    base_trial = evaluate_params(grid, base_params, materials, legacy=legacy)

    # Composite presets use DesignEvaluator-based objectives (EM, thermal, stress, etc.)
    if args.preset != "em":
        cfg = preset_config(
            args.preset,
            materials=materials,
            pair_label=materials_label,
            legacy=legacy,
            check_arcing=args.check_arcing,
        )
        t0 = time.time()
        comp = optuna_design_search(
            grid, cfg, args.trials, args.seed, preset=args.preset,
        )
        t1 = time.time()
        comp_best = best_design(comp)
        summary = {
            "grid": args.grid,
            "trials": args.trials,
            "seed": args.seed,
            "materials": materials_label,
            "preset": args.preset,
            "search_mode": "legacy" if legacy else "manufacturable",
            "materials_provenance": (
                PAIRS[materials_label].provenance
                if materials_label in PAIRS
                else "ore_bruggeman"
            ),
            "baseline_untuned": {
                "selectivity": base_trial.selectivity,
                "contrast": base_trial.contrast,
            },
            "composite_search": {
                "best_score": comp_best.score,
                "objective_key": comp_best.objective_key,
                "best_params": comp_best.params,
                "report": comp_best.to_dict(),
                "seconds": round(t1 - t0, 1),
            },
        }
        if ore_block:
            summary["ore"] = ore_block
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print("=== Composite search result ===")
        print(f"  preset                : {args.preset}")
        print(f"  untuned EM sel        : {base_trial.selectivity:.4f}")
        print(f"  composite best score  : {comp_best.score:.4f} ({comp_best.objective_key})")
        print(f"  EM sel @ composite    : {comp_best.em_selectivity:.4f}")
        if comp_best.delta_T_K is not None:
            print(f"  ΔT @ composite        : {comp_best.delta_T_K:.1f} K")
        print(f"  wrote {out}")
        return

    t0 = time.time()
    rnd = random_search(grid, args.trials, seed=args.seed, base=base_params, materials=materials, legacy=legacy)
    t1 = time.time()
    tpe = optuna_search(grid, args.trials, seed=args.seed, base=base_params, materials=materials, legacy=legacy)
    t2 = time.time()

    rnd_best = best(rnd)
    tpe_best = best(tpe)

    summary = {
        "grid": args.grid,
        "trials": args.trials,
        "seed": args.seed,
        "materials": materials_label,
        "search_mode": "legacy" if legacy else "manufacturable",
        "materials_provenance": (
            PAIRS[materials_label].provenance if materials_label in PAIRS else "ore_bruggeman"
        ),
        "baseline_untuned": {
            "selectivity": base_trial.selectivity,
            "contrast": base_trial.contrast,
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
            "seconds": round(t2 - t1, 1),
        },
    }
    if ore_block:
        summary["ore"] = ore_block

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("=== Thin-slice search result ===")
    print(f"  mode                  : {'legacy baffle' if legacy else 'manufacturable'}")
    print(f"  materials             : {materials_label}")
    if ore_block:
        print(f"  ore profile           : {ore_block['label']}  pair={ore_block['suggested_pair']}")
    print(f"  untuned selectivity   : {base_trial.selectivity:.4f}")
    print(f"  random  best          : {rnd_best.selectivity:.4f}  ({summary['random_search']['seconds']}s)")
    print(f"  TPE     best          : {tpe_best.selectivity:.4f}  ({summary['tpe_search']['seconds']}s)")
    print(f"  contrast untuned/TPE  : {base_trial.contrast:.2f} -> {tpe_best.contrast:.2f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
