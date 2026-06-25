"""Gel-phantom predictions (recipe-linked ε + thermal ΔT) and optional lab compare.

    python scripts/run_phantom_study.py
    python scripts/run_phantom_study.py --compare data/lab_measurements.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.maturity import status_dict  # noqa: E402
from mw_inv.phantom import (  # noqa: E402
    PHANTOM_RECIPES,
    compare_lab_measurement,
    load_lab_measurements,
    predict_all_phantoms,
    predict_lab_outcome,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", choices=sorted(PHANTOM_RECIPES), default=None)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--seed", type=int, default=7701)
    ap.add_argument("--measured-eps", type=str, default=None, help="probe-measured eps JSON")
    ap.add_argument("--compare", type=str, default=None, help="JSON bench measurements")
    ap.add_argument("--out", default="data/phantom_lab_predictions.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    preds = (
        [predict_lab_outcome(args.phantom, grid, n_opt_trials=args.trials, seed=args.seed,
                           measured_eps_path=args.measured_eps)]
        if args.phantom
        else predict_all_phantoms(grid, n_opt_trials=args.trials, seed=args.seed)
    )

    comparisons = []
    if args.compare:
        by_phantom = {p.phantom: p for p in preds}
        for row in load_lab_measurements(args.compare):
            ph = row["phantom"]
            if ph in by_phantom:
                comparisons.append(
                    compare_lab_measurement(
                        by_phantom[ph],
                        float(row["measured_delta_T_K"]),
                        row.get("measured_selectivity"),
                        untuned_measured_delta_T_K=row.get("untuned_measured_delta_T_K"),
                    ).to_dict()
                )

    summary = {
        "maturity": status_dict("phantom_lab"),
        "predictions": [p.to_dict() for p in preds],
        "comparisons": comparisons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("=== Lab phantom study (recipe-linked ε + thermal ΔT) ===")
    for p in preds:
        print(f"  {p.phantom}  salt {p.gangue_salt_wt:.1f}/{p.target_salt_wt:.1f} wt%")
        print(f"    ε target {p.target_eps.real:.1f}-j{p.target_eps.imag:.2f}")
        print(f"    sel {p.untuned_selectivity:.3f} → {p.optimized_selectivity:.3f}")
        print(f"    ΔT  {p.untuned_delta_T_K:.1f} → {p.optimized_delta_T_K:.1f} K")
    if comparisons:
        print(f"  compared {len(comparisons)} bench record(s) from {args.compare}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
