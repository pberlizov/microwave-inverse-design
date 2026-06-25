"""Optimise applicator for liberation-relevant interface stress (not EM alone).

    python scripts/run_stress_search.py --materials pyrite_in_calcite --trials 24
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
from mw_inv.search import STRESS_OBJECTIVES, best_stress, evaluate_stress_params, optuna_stress_search  # noqa: E402
from mw_inv.thermal import ThermalConfig, thermal_props_for_pair  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--objective", choices=STRESS_OBJECTIVES, default="stress_score")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--out", default="data/stress_search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    tcfg = ThermalConfig(drive=8.0, thermal_props=thermal_props_for_pair(args.materials), max_iters=15)
    base = evaluate_stress_params(grid, CavityParams(), args.materials, args.objective, thermal_cfg=tcfg)

    t0 = time.time()
    tpe = optuna_stress_search(
        grid, args.materials, args.trials, args.seed, args.objective, thermal_cfg=tcfg,
    )
    best = best_stress(tpe)

    summary = {
        "materials": args.materials,
        "objective": args.objective,
        "baseline": {
            "score": base.score,
            "stress_selectivity": base.stress_selectivity,
            "mean_interface_stress_Pa": base.mean_interface_stress_Pa,
            "em_selectivity": base.em_selectivity,
            "delta_T_K": base.delta_T_K,
        },
        "tpe_search": {
            "best_score": best.score,
            "stress_selectivity": best.stress_selectivity,
            "mean_interface_stress_Pa": best.mean_interface_stress_Pa,
            "em_selectivity": best.em_selectivity,
            "delta_T_K": best.delta_T_K,
            "params": best.params,
            "seconds": round(time.time() - t0, 1),
        },
        "em_vs_stress": {
            "note": "Compare em_selectivity of stress-optimal vs EM-only search",
            "stress_opt_em_selectivity": best.em_selectivity,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== Stress search ({args.materials}) ===")
    print(f"  untuned stress score : {base.score:.2e} Pa (EM sel={base.em_selectivity:.3f})")
    print(f"  TPE best             : {best.score:.2e} Pa (EM sel={best.em_selectivity:.3f})")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
