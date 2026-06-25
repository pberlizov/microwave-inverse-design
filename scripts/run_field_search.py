"""High-dimensional tuner-field search (legacy upper bound).

**Deprecated for primary use** — prefer ``run_search.py`` (manufacturable plate +
wall feed + bed position).  This script remains as a surrogate upper bound: the
16-cell lossless dielectric tuner is not physically realizable but shows how much
mode shaping a non-manufacturable actuator could buy.

    python scripts/run_field_search.py --materials pyrite_in_calcite --trials 120 --k 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.search import optuna_field_search, random_field_search  # noqa: E402


def _best_so_far(trials) -> np.ndarray:
    return np.maximum.accumulate([t.selectivity for t in trials])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--k", type=int, default=16, help="tuner cells (search dimension)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--grid", type=int, default=101)
    ap.add_argument("--out", type=str, default="data/field_search.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair(args.materials)

    rnd_curves, tpe_curves = [], []
    t0 = time.time()
    for s in range(args.seeds):
        rnd = random_field_search(grid, args.trials, seed=1000 + s, k=args.k,
                                  materials=materials)
        tpe = optuna_field_search(grid, args.trials, seed=1000 + s, k=args.k,
                                  materials=materials)
        rnd_curves.append(_best_so_far(rnd))
        tpe_curves.append(_best_so_far(tpe))
    elapsed = time.time() - t0

    rnd_mean = np.mean(rnd_curves, axis=0)
    tpe_mean = np.mean(tpe_curves, axis=0)
    rnd_final = float(rnd_mean[-1])
    tpe_final = float(tpe_mean[-1])
    gap = tpe_final - rnd_final

    print(f"=== High-dim tuner-field search ({args.materials}, K={args.k}, "
          f"{args.trials} trials x {args.seeds} seeds, {elapsed:.1f}s) ===")
    print("  (deprecated upper bound — use run_search.py for manufacturable geometry)")
    print(f"  random best (mean) : {rnd_final:.4f}")
    print(f"  TPE    best (mean) : {tpe_final:.4f}")
    print(f"  gap (TPE - random) : {gap:+.4f}  "
          f"({'TPE wins' if gap > 0.003 else 'tie'})")
    # Budget to reach random's final quality:
    reach = np.argmax(tpe_mean >= rnd_final)
    if tpe_mean[-1] >= rnd_final and reach > 0:
        print(f"  TPE reaches random's final in {reach+1}/{args.trials} trials "
              f"({100*(reach+1)/args.trials:.0f}% of budget)")

    summary = {
        "materials": args.materials,
        "k": args.k,
        "trials": args.trials,
        "seeds": args.seeds,
        "random_best_mean": rnd_final,
        "tpe_best_mean": tpe_final,
        "gap": gap,
        "random_curve_mean": rnd_mean.tolist(),
        "tpe_curve_mean": tpe_mean.tolist(),
        "note": "Best-so-far selectivity averaged over seeds. TPE uses multivariate "
                "to model cell interactions. Contrast with run_search.py's 6-knob case "
                "where random ties TPE.",
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out}")
    _maybe_plot(rnd_curves, tpe_curves, args)


def _maybe_plot(rnd_curves, tpe_curves, args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping plot: {e})")
        return
    try:
        rnd = np.array(rnd_curves)
        tpe = np.array(tpe_curves)
        x = np.arange(1, rnd.shape[1] + 1)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arr, color, name in [(rnd, "C0", "random"), (tpe, "C3", "TPE (surrogate)")]:
            m = arr.mean(axis=0)
            ax.plot(x, m, color=color, label=name)
            if arr.shape[0] > 1:
                ax.fill_between(x, arr.min(axis=0), arr.max(axis=0), color=color, alpha=0.15)
        ax.set(xlabel="trial", ylabel="best-so-far selectivity",
               title=f"High-dim tuner field (K={args.k}): TPE vs random")
        ax.legend()
        fig.tight_layout()
        fig.savefig("data/field_search.png", dpi=110)
        print("  wrote data/field_search.png")
    except Exception as e:  # noqa: BLE001
        print(f"  (plot render failed, summary still written: {e})")


if __name__ == "__main__":
    main()
