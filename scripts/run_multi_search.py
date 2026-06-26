"""Multi-objective search: selectivity vs coupling efficiency (backlog C0).

Pareto tradeoff between selective heating and energy-consistent coupling — avoids
geometries that 'win' on selectivity while routing power into structure.

    python scripts/run_multi_search.py --materials pyrite_in_calcite --trials 40
    python scripts/run_multi_search.py --check-arcing --trials 40

Writes data/multi_search_summary.json with Pareto highlights and a weighted recommendation.
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
    pareto_front_trials,
    pareto_recommend,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--seed", type=int, default=3307)
    ap.add_argument("--check-arcing", action="store_true", help="penalise / filter arcing-risk trials")
    ap.add_argument("--weight-selectivity", type=float, default=0.6)
    ap.add_argument("--weight-coupling", type=float, default=0.4)
    ap.add_argument("--out", type=str, default="data/multi_search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair(args.materials)
    baseline = evaluate_params(grid, CavityParams(), materials)

    t0 = time.time()
    trials, study = optuna_multi_search(
        grid,
        args.trials,
        args.seed,
        materials=materials,
        check_arcing=args.check_arcing,
    )
    elapsed = time.time() - t0

    best_sel = pareto_best_selectivity(trials)
    best_coupling = pareto_best_coupling(trials)
    recommended = pareto_recommend(
        trials,
        study,
        weight_selectivity=args.weight_selectivity,
        weight_coupling=args.weight_coupling,
        exclude_arcing=args.check_arcing,
    )
    pareto = pareto_front_trials(trials, study)

    summary = {
        "materials": args.materials,
        "trials": args.trials,
        "objectives": ["em_selectivity", "coupling_eff"],
        "check_arcing": args.check_arcing,
        "baseline": {
            "selectivity": baseline.selectivity,
            "coupling_eff": baseline.coupling_eff,
            "p_total": baseline.p_total,
            "contrast": baseline.contrast,
        },
        "best_selectivity": {
            "selectivity": best_sel.selectivity,
            "coupling_eff": best_sel.coupling_eff,
            "p_total": best_sel.p_total,
            "contrast": best_sel.contrast,
            "arcing_risk": best_sel.arcing_risk,
            "params": best_sel.params,
        },
        "best_coupling": {
            "selectivity": best_coupling.selectivity,
            "coupling_eff": best_coupling.coupling_eff,
            "p_total": best_coupling.p_total,
            "contrast": best_coupling.contrast,
            "arcing_risk": best_coupling.arcing_risk,
            "params": best_coupling.params,
        },
        "recommended": {
            "weights": {
                "selectivity": args.weight_selectivity,
                "coupling": args.weight_coupling,
            },
            "selectivity": recommended.selectivity,
            "coupling_eff": recommended.coupling_eff,
            "p_total": recommended.p_total,
            "arcing_risk": recommended.arcing_risk,
            "params": recommended.params,
        },
        "pareto_count": len(pareto),
        "pareto_front": [
            {
                "selectivity": t.selectivity,
                "coupling_eff": t.coupling_eff,
                "p_total": t.p_total,
                "arcing_risk": t.arcing_risk,
                "params": t.params,
            }
            for t in pareto[:12]
        ],
        "seconds": round(elapsed, 1),
        "note": "Two-objective: maximise selectivity AND coupling_eff (charge power fraction).",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== Multi-objective search ({args.materials}) ===")
    print(
        f"  baseline sel / coupling : {baseline.selectivity:.4f} / {baseline.coupling_eff:.4f}"
    )
    print(
        f"  best selectivity        : {best_sel.selectivity:.4f}  "
        f"(coupling={best_sel.coupling_eff:.4f})"
    )
    print(
        f"  best coupling           : sel={best_coupling.selectivity:.4f}  "
        f"(coupling={best_coupling.coupling_eff:.4f})"
    )
    print(
        f"  recommended (weighted)  : sel={recommended.selectivity:.4f}  "
        f"coupling={recommended.coupling_eff:.4f}"
    )
    print(f"  Pareto points             : {len(pareto)}")
    print(f"  seconds                   : {elapsed:.1f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
