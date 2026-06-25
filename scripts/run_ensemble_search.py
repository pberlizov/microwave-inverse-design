"""Robust geometry search over random grain layouts (roadmap step 5).

Optimises mean selectivity averaged over stochastic ore realizations instead of a
single fixed inclusion pattern.

    python scripts/run_ensemble_search.py --materials pyrite_in_calcite --trials 20
    python scripts/run_ensemble_search.py --realizations 8 --grains 6 --grid 61

Writes data/ensemble_search_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.ensemble import evaluate_ensemble  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.search import best_robust, evaluate_params, evaluate_robust_params, optuna_robust_search  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--realizations", type=int, default=6, help="layouts per evaluation")
    ap.add_argument("--grains", type=int, default=5, help="target grains per layout")
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--seed", type=int, default=2206)
    ap.add_argument("--out", type=str, default="data/ensemble_search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair(args.materials)

    fixed = evaluate_params(grid, CavityParams(), materials)
    base_ens = evaluate_ensemble(
        grid, CavityParams(), materials,
        n_realizations=args.realizations, n_grains=args.grains, seed=args.seed,
    )

    t0 = time.time()
    trials = optuna_robust_search(
        grid, args.trials, args.seed, materials,
        n_realizations=args.realizations, n_grains=args.grains,
    )
    elapsed = time.time() - t0
    best = best_robust(trials)

    summary = {
        "materials": args.materials,
        "n_realizations": args.realizations,
        "n_grains": args.grains,
        "trials": args.trials,
        "fixed_layout": {
            "selectivity": fixed.selectivity,
            "p_total": fixed.p_total,
        },
        "untuned_ensemble": base_ens.to_dict(),
        "tpe_robust_best": {
            "mean_selectivity": best.mean_selectivity,
            "min_selectivity": best.min_selectivity,
            "std_selectivity": best.std_selectivity,
            "mean_p_total": best.mean_p_total,
            "params": best.params,
            "seconds": round(elapsed, 1),
        },
        "note": "Optimises mean selectivity over random grain layouts in the charge bed.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== Ensemble robust search ({args.materials}) ===")
    print(f"  fixed layout selectivity     : {fixed.selectivity:.4f}")
    print(f"  untuned mean ± std           : {base_ens.mean_selectivity:.4f} ± {base_ens.std_selectivity:.4f}")
    print(f"  TPE robust mean selectivity  : {best.mean_selectivity:.4f}  (min {best.min_selectivity:.4f})")
    print(f"  seconds                      : {elapsed:.1f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
