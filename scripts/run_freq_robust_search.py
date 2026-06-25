"""Frequency-robust geometry search across the ISM band.

Scores each candidate by mean or worst-case selectivity over 2.40–2.50 GHz
(magnetron drift), without tuning a single centre frequency.

    python scripts/run_freq_robust_search.py --materials pyrite_in_calcite --metric min
    python scripts/run_freq_robust_search.py --trials 30 --metric mean --freqs 5

Writes data/freq_robust_search_summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.ensemble import evaluate_frequency_robust  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.search import (  # noqa: E402
    ROBUST_METRICS,
    best_freq_robust,
    evaluate_freq_robust_params,
    optuna_freq_robust_search,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--metric", choices=ROBUST_METRICS, default="min",
                    help="min = worst-case over band; mean = average over band")
    ap.add_argument("--freqs", type=int, default=5, help="ISM band sample points")
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--seed", type=int, default=4408)
    ap.add_argument("--out", type=str, default="data/freq_robust_search_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair(args.materials)

    untuned = evaluate_frequency_robust(
        grid, CavityParams(), materials, pair_label=args.materials, n_freqs=args.freqs,
    )
    baseline = evaluate_freq_robust_params(
        grid, CavityParams(), materials, pair_label=args.materials,
        metric=args.metric, n_freqs=args.freqs,
    )

    t0 = time.time()
    trials = optuna_freq_robust_search(
        grid, args.trials, args.seed, materials,
        pair_label=args.materials, metric=args.metric, n_freqs=args.freqs,
    )
    elapsed = time.time() - t0
    best = best_freq_robust(trials)

    summary = {
        "materials": args.materials,
        "metric": args.metric,
        "n_freqs": args.freqs,
        "trials": args.trials,
        "untuned_band": untuned.to_dict(),
        "baseline_score": baseline.score,
        "tpe_best": {
            "score": best.score,
            "mean_selectivity": best.mean_selectivity,
            "min_selectivity": best.min_selectivity,
            "std_selectivity": best.std_selectivity,
            "params": best.params,
            "seconds": round(elapsed, 1),
        },
        "note": "Geometry optimised for selectivity stable across ISM band; freq_hz not tuned.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"=== Frequency-robust search ({args.materials}, metric={args.metric}) ===")
    print(f"  untuned mean / min over band : {untuned.mean_selectivity:.4f} / {untuned.min_selectivity:.4f}")
    print(f"  TPE best score ({args.metric})     : {best.score:.4f}")
    print(f"  TPE mean / min               : {best.mean_selectivity:.4f} / {best.min_selectivity:.4f}")
    print(f"  seconds                      : {elapsed:.1f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
