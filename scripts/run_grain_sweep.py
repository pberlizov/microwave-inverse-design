"""Grain size vs skin depth: when does selective heating self-limit vs run away?

    python scripts/run_grain_sweep.py --materials pyrite_in_calcite

Sweeps inclusion size and loss factor (real EM) and shows that the absorbed-power
turnover (self-shielding onset) lands where the grain diameter ~ the microwave skin
depth. Small grains (diameter < skin depth across the whole eps'' range) never turn
over -- absorption keeps rising with eps'' (hence with temperature) -> positive feedback
-> runaway-prone. Large grains self-limit. Writes data/grain_sweep.json (+ png if
matplotlib works).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.sweeps import grain_size_sweep, skin_depth_m  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--grid", type=int, default=161)
    ap.add_argument("--out", type=str, default="data/grain_sweep.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    pair = PAIRS[args.materials]
    materials = Materials.from_pair(args.materials)
    eps_real = pair.target.real

    # Log-spaced eps'' (more resolution at the low end where big grains turn over).
    epps = np.geomspace(0.05, 8.0, 24)
    fracs = np.array([0.018, 0.025, 0.035, 0.05, 0.07, 0.09, 0.11])
    rows = grain_size_sweep(grid, fracs, epps, eps_real=eps_real, base_materials=materials)

    freq_hz = CavityParams().freq_hz
    delta_max_loss = skin_depth_m(freq_hz, eps_real, epps[-1])
    print(f"=== Grain size vs skin depth ({args.materials}, eps'={eps_real:.1f}) ===")
    print(f"  skin depth at eps''={epps[-1]:.1f} (loss tangent~1): {delta_max_loss*1000:.1f} mm")
    print(f"  {'diam_mm':>8} {'eps_star':>9} {'skin_mm':>8} {'d/delta':>8}  regime")
    for r in rows:
        es = "  monoton" if r.monotonic else f"{r.turnover_eps_imag:8.2f}"
        regime = "runaway-prone (no self-shielding in range)" if r.monotonic \
            else "self-limiting"
        print(f"  {r.diameter_m*1000:8.1f} {es} {r.skin_depth_at_turnover_m*1000:8.1f} "
              f"{r.ratio_d_over_delta:8.2f}  {regime}")

    finite = [r for r in rows if not r.monotonic]
    if finite:
        ratios = np.array([r.ratio_d_over_delta for r in finite])
        print(f"\n  turnover collapses onto d/skin-depth = {ratios.mean():.2f} "
              f"+/- {ratios.std():.2f}  (order unity -> grain ~ skin depth)")
    print("  => smaller grains stay in the absorption-rising (positive-feedback) regime "
          "to higher eps''/temperature: more runaway-prone.")

    summary = {
        "materials": args.materials,
        "eps_real": eps_real,
        "skin_depth_at_max_loss_mm": delta_max_loss * 1000.0,
        "eps_imag": epps.tolist(),
        "rows": [
            {
                "diameter_mm": r.diameter_m * 1000.0,
                "turnover_eps_imag": (None if r.monotonic else r.turnover_eps_imag),
                "skin_depth_at_turnover_mm": r.skin_depth_at_turnover_m * 1000.0,
                "ratio_d_over_delta": r.ratio_d_over_delta,
                "monotonic_runaway_prone": r.monotonic,
                "mean_power_density": r.mean_power_density.tolist(),
            }
            for r in rows
        ],
        "note": "Real EM. Turnover (self-shielding onset) ~ grain diameter ~ skin depth. "
                "Monotonic rows never self-shield within eps'' <= loss tangent 1, so "
                "absorption keeps rising with temperature -> runaway-prone.",
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out}")
    _maybe_plot(rows, epps)


def _maybe_plot(rows, epps) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping plot: {e})")
        return
    try:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for r in rows:
            lbl = f"{r.diameter_m*1000:.0f} mm"
            ax[0].plot(r.eps_imag, r.mean_power_density / r.mean_power_density.max(),
                       "-o", ms=2, label=lbl)
        ax[0].set(xlabel="target loss factor eps'' (rises with T)",
                  ylabel="mean absorbed power density (norm.)",
                  title="Self-shielding turnover shifts with grain size")
        ax[0].set_xscale("log")
        ax[0].legend(fontsize=7, title="grain diameter")

        fin = [r for r in rows if not r.monotonic]
        if fin:
            d = np.array([r.diameter_m * 1000 for r in fin])
            es = np.array([r.turnover_eps_imag for r in fin])
            ax[1].plot(d, es, "-o", color="C3")
            ax[1].set(xlabel="grain diameter (mm)", ylabel="turnover eps''*",
                      title="Larger grains self-limit at lower loss")
        fig.suptitle("Grain size vs skin depth: positive-feedback (runaway) vs self-limiting")
        fig.tight_layout()
        fig.savefig("data/grain_sweep.png", dpi=110)
        print("  wrote data/grain_sweep.png")
    except Exception as e:  # noqa: BLE001
        print(f"  (plot render failed, summary still written: {e})")


if __name__ == "__main__":
    main()
