"""Thermal search robust to grain layouts and (optionally) frequency drift.

    python scripts/run_thermal_ensemble_search.py --objective delta_T
    python scripts/run_thermal_ensemble_search.py --freq-robust --metric min --realizations 4

Writes data/thermal_ensemble_search_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.ensemble import evaluate_thermal_ensemble  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.search import (  # noqa: E402
    ROBUST_METRICS,
    THERMAL_OBJECTIVES,
    best_thermal_ensemble,
    evaluate_thermal_ensemble_params,
    optuna_thermal_ensemble_search,
)
from mw_inv.thermal import ThermalConfig, thermal_props_for_pair  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--objective", choices=THERMAL_OBJECTIVES, default="delta_T")
    ap.add_argument("--metric", choices=ROBUST_METRICS, default="mean")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--realizations", type=int, default=4, help="random layouts per trial")
    ap.add_argument("--grains", type=int, default=5)
    ap.add_argument("--freq-robust", action="store_true", help="also sweep ISM band")
    ap.add_argument("--freqs", type=int, default=5)
    ap.add_argument("--grid", type=int, default=51)
    ap.add_argument("--seed", type=int, default=5509)
    ap.add_argument("--drive", type=float, default=8.0)
    ap.add_argument("--out", type=str, default="data/thermal_ensemble_search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    tcfg = ThermalConfig(
        drive=args.drive,
        thermal_props=thermal_props_for_pair(args.materials),
        max_iters=12,
        tol_K=4.0,
    )

    untuned = evaluate_thermal_ensemble(
        grid, CavityParams(), args.materials,
        n_realizations=args.realizations, n_grains=args.grains, seed=args.seed,
        thermal_cfg=tcfg, freq_robust=args.freq_robust, n_freqs=args.freqs,
    )
    baseline = evaluate_thermal_ensemble_params(
        grid, CavityParams(), args.materials,
        objective=args.objective, metric=args.metric,
        n_realizations=args.realizations, n_grains=args.grains, seed=args.seed,
        thermal_cfg=tcfg, freq_robust=args.freq_robust, n_freqs=args.freqs,
    )

    t0 = time.time()
    trials = optuna_thermal_ensemble_search(
        grid, args.materials, args.trials, args.seed,
        objective=args.objective, metric=args.metric,
        n_realizations=args.realizations, n_grains=args.grains,
        thermal_cfg=tcfg, freq_robust=args.freq_robust, n_freqs=args.freqs,
    )
    elapsed = time.time() - t0
    best = best_thermal_ensemble(trials)

    summary = {
        "materials": args.materials,
        "objective": args.objective,
        "metric": args.metric,
        "freq_robust": args.freq_robust,
        "n_realizations": args.realizations,
        "n_grains": args.grains,
        "trials": args.trials,
        "untuned": untuned.to_dict(),
        "baseline_score": baseline.score,
        "tpe_best": {
            "score": best.score,
            "mean_delta_T_K": best.mean_delta_T_K,
            "min_delta_T_K": best.min_delta_T_K,
            "mean_heat_selectivity": best.mean_heat_selectivity,
            "params": best.params,
            "seconds": round(elapsed, 1),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    mode = "thermal+layout"
    if args.freq_robust:
        mode += "+freq"
    print(f"=== Thermal ensemble search ({args.materials}, {mode}) ===")
    print(f"  untuned mean ΔT / min ΔT  : {untuned.mean_delta_T_K:.1f} / {untuned.min_delta_T_K:.1f} K")
    print(f"  TPE best score            : {best.score:.2f}")
    print(f"  TPE mean ΔT / min ΔT      : {best.mean_delta_T_K:.1f} / {best.min_delta_T_K:.1f} K")
    print(f"  seconds                   : {elapsed:.1f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
