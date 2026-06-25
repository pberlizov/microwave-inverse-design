"""Multi-objective search: selectivity vs charge coupling (roadmap step 6).

Pareto-optimal tradeoff between selective heating and total absorbed power in the
charge — avoids geometries that 'win' by absorbing almost nothing.

    python scripts/run_multi_search.py --materials pyrite_in_calcite --trials 40

Writes data/multi_search_summary.json with Pareto highlights.
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
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.search import (  # noqa: E402
    evaluate_params,
    optuna_multi_search,
    pareto_best_coupling,
    pareto_best_selectivity,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--seed", type=int, default=3307)
    ap.add_argument("--out", type=str, default="data/multi_search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair(args.materials)
    baseline = evaluate_params(grid, CavityParams(), materials)

    t0 = time.time()
    trials, study = optuna_multi_search(grid, args.trials, args.seed, materials=materials)
    elapsed = time.time() - t0

    best_sel = pareto_best_selectivity(trials)
    best_p = pareto_best_coupling(trials)
    pareto = study.best_trials

    summary = {
        "materials": args.materials,
        "trials": args.trials,
        "baseline": {
            "selectivity": baseline.selectivity,
            "p_total": baseline.p_total,
            "contrast": baseline.contrast,
        },
        "best_selectivity": {
            "selectivity": best_sel.selectivity,
            "p_total": best_sel.p_total,
            "contrast": best_sel.contrast,
            "params": best_sel.params,
        },
        "best_coupling": {
            "selectivity": best_p.selectivity,
            "p_total": best_p.p_total,
            "contrast": best_p.contrast,
            "params": best_p.params,
        },
        "pareto_count": len(pareto),
        "pareto_front": [
            {
                "selectivity": t.values[0],
                "p_total": t.values[1],
                "params": trials[t.number].params if t.number < len(trials) else {},
            }
            for t in pareto[:12]
        ],
        "seconds": round(elapsed, 1),
        "note": "Two-objective: maximise selectivity AND total charge absorption.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== Multi-objective search ({args.materials}) ===")
    print(f"  baseline sel / P_charge   : {baseline.selectivity:.4f} / {baseline.p_total:.3e}")
    print(f"  best selectivity          : {best_sel.selectivity:.4f}  (P={best_sel.p_total:.3e})")
    print(f"  best coupling             : sel={best_p.selectivity:.4f}  (P={best_p.p_total:.3e})")
    print(f"  Pareto points             : {len(pareto)}")
    print(f"  seconds                   : {elapsed:.1f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
