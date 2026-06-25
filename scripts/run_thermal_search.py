"""Optimise applicator geometry for thermal FOM (coupled EM–heat), not EM alone.

    python scripts/run_thermal_search.py --materials pyrite_in_calcite --objective delta_T
    python scripts/run_thermal_search.py --trials 20 --grid 61 --objective heat_selectivity

Writes data/thermal_search_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.search import (  # noqa: E402
    THERMAL_OBJECTIVES,
    best_thermal,
    evaluate_thermal_params,
    optuna_thermal_search,
    random_thermal_search,
)
from mw_inv.thermal import ThermalConfig, thermal_props_for_pair  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--objective", choices=THERMAL_OBJECTIVES, default="delta_T")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--drive", type=float, default=8.0)
    ap.add_argument("--out", type=str, default="data/thermal_search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    tcfg = ThermalConfig(drive=args.drive, thermal_props=thermal_props_for_pair(args.materials))

    base = evaluate_thermal_params(
        grid, CavityParams(), args.materials, tcfg, args.objective,
    )

    t0 = time.time()
    rnd = random_thermal_search(
        grid, args.materials, args.trials, args.seed, args.objective, thermal_cfg=tcfg,
    )
    t1 = time.time()
    tpe = optuna_thermal_search(
        grid, args.materials, args.trials, args.seed, args.objective, thermal_cfg=tcfg,
    )
    t2 = time.time()

    rnd_best = best_thermal(rnd)
    tpe_best = best_thermal(tpe)

    summary = {
        "materials": args.materials,
        "objective": args.objective,
        "drive": args.drive,
        "grid": args.grid,
        "trials": args.trials,
        "baseline_untuned": {
            "score": base.score,
            "delta_T_K": base.delta_T_K,
            "heat_selectivity": base.heat_selectivity,
            "em_selectivity": base.em_selectivity,
        },
        "random_search": {
            "best_score": rnd_best.score,
            "delta_T_K": rnd_best.delta_T_K,
            "heat_selectivity": rnd_best.heat_selectivity,
            "em_selectivity": rnd_best.em_selectivity,
            "params": rnd_best.params,
            "seconds": round(t1 - t0, 1),
        },
        "tpe_search": {
            "best_score": tpe_best.score,
            "delta_T_K": tpe_best.delta_T_K,
            "heat_selectivity": tpe_best.heat_selectivity,
            "em_selectivity": tpe_best.em_selectivity,
            "params": tpe_best.params,
            "seconds": round(t2 - t1, 1),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== Thermal search ({args.materials}, objective={args.objective}) ===")
    print(f"  untuned score         : {base.score:.2f}  (ΔT={base.delta_T_K:.1f} K)")
    print(f"  random best           : {rnd_best.score:.2f}  ({summary['random_search']['seconds']}s)")
    print(f"  TPE best              : {tpe_best.score:.2f}  ({summary['tpe_search']['seconds']}s)")
    print(f"  EM selectivity (TPE)  : {tpe_best.em_selectivity:.4f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
