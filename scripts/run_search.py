"""Thin-slice experiment: can geometry move selective-absorption contrast?

Optimise applicator knobs (feed position, frequency, internal baffle) to maximise
the fraction of absorbed power that lands in the target mineral phase, and compare
against a random-search control. Writes results/summary JSON.

Run:  python scripts/run_search.py --trials 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.materials import DEFAULT_PAIR, PAIRS  # noqa: E402
from mw_inv.search import best, evaluate_params, optuna_search, random_search  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--grid", type=int, default=101)
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--materials", choices=sorted(PAIRS), default=DEFAULT_PAIR.label,
                    help="cited mineral pair (see mw_inv.materials / docs/MATERIALS.md)")
    ap.add_argument("--out", type=str, default="data/search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair(args.materials)

    # Untuned baseline (default geometry, no optimisation).
    base_trial = evaluate_params(grid, CavityParams(), materials)

    t0 = time.time()
    rnd = random_search(grid, args.trials, seed=args.seed, materials=materials)
    t1 = time.time()
    tpe = optuna_search(grid, args.trials, seed=args.seed, materials=materials)
    t2 = time.time()

    rnd_best = best(rnd)
    tpe_best = best(tpe)

    summary = {
        "grid": args.grid,
        "trials": args.trials,
        "seed": args.seed,
        "materials": args.materials,
        "materials_provenance": PAIRS[args.materials].provenance,
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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("=== Thin-slice search result ===")
    print(f"  materials             : {args.materials}")
    print(f"  untuned selectivity   : {base_trial.selectivity:.4f}")
    print(f"  random  best          : {rnd_best.selectivity:.4f}  ({summary['random_search']['seconds']}s)")
    print(f"  TPE     best          : {tpe_best.selectivity:.4f}  ({summary['tpe_search']['seconds']}s)")
    print(f"  contrast untuned/TPE  : {base_trial.contrast:.2f} -> {tpe_best.contrast:.2f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
