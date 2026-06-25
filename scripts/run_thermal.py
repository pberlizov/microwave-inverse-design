"""Spatial EM–thermal coupled steady-state solve.

    python scripts/run_thermal.py --materials pyrite_in_calcite
    python scripts/run_thermal.py --materials pyrite_in_calcite --drive 5 --grid 81

Writes data/thermal_summary.json and, if matplotlib works, data/thermal.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.thermal import (  # noqa: E402
    ThermalConfig,
    TransientConfig,
    coupled_steady_state,
    isothermal_baseline,
    simulate_transient,
    thermal_props_for_pair,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--grid", type=int, default=81)
    ap.add_argument("--drive", type=float, default=8.0, help="microwave power scale factor")
    ap.add_argument("--transient", action="store_true", help="run transient runaway timing simulation")
    ap.add_argument("--t-end", type=float, default=90.0, help="transient duration (s)")
    ap.add_argument("--threshold", type=float, default=773.0, help="T threshold (K) for time-to-heat")
    ap.add_argument("--out", type=str, default="data/thermal_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    pair = PAIRS[args.materials]

    iso, _, _ = isothermal_baseline(grid, args.materials, drive=args.drive)
    cfg = ThermalConfig(
        drive=args.drive,
        thermal_props=thermal_props_for_pair(args.materials),
    )
    coupled = coupled_steady_state(grid, args.materials, config=cfg)
    th = coupled.thermal

    summary = {
        "materials": args.materials,
        "provenance": pair.provenance,
        "drive": args.drive,
        "grid": args.grid,
        "isothermal": iso.to_dict(),
        "coupled_steady": th.to_dict(),
        "history_max_delta_K": coupled.history_max_delta,
        "note": "Quasi-steady 2D: k∇²T=−q from FDFD; ε(T) from dielectric_data tables; "
                "Dirichlet T_amb on cavity walls. Representative k, h, not measured ore.",
    }

    if args.transient:
        tr_cfg = TransientConfig(
            drive=args.drive,
            t_end_s=args.t_end,
            T_threshold_K=args.threshold,
            thermal_props=thermal_props_for_pair(args.materials),
        )
        tr = simulate_transient(grid, args.materials, config=tr_cfg)
        summary["transient"] = tr.report.to_dict()
        summary["transient"]["times_s"] = tr.times_s.tolist()
        summary["transient"]["mean_T_target_K"] = tr.mean_T_target.tolist()
        summary["transient"]["mean_T_gangue_K"] = tr.mean_T_gangue.tolist()

    print(f"=== EM–thermal coupling ({args.materials}, drive={args.drive}) ===")
    print(f"  isothermal EM selectivity @298K : {iso.selectivity:.4f}")
    print(f"  coupled EM selectivity           : {th.em_selectivity:.4f}")
    print(f"  heat selectivity (spatial q)      : {th.heat_selectivity:.4f}")
    print(f"  T_mean target / gangue (K)       : {th.T_mean_target_K:.1f} / {th.T_mean_gangue_K:.1f}")
    print(f"  ΔT (target − gangue)             : {th.delta_T_K:.1f} K")
    print(f"  T_max in target (K)              : {th.T_max_target_K:.1f}")
    print(f"  converged in {th.n_iters} iters   : {th.converged}  (max |ΔT|={th.max_delta_K:.2f} K)")

    if args.transient:
        r = summary["transient"]
        print(f"\n=== Transient runaway timing (threshold={args.threshold:.0f} K) ===")
        print(f"  t_target reaches threshold : {r['t_target_s']:.1f} s")
        print(f"  t_gangue reaches threshold : {r['t_gangue_s']:.1f} s")
        print(f"  target runaway first       : {r['target_runaway_first']}")
        print(f"  Δt (gangue − target)       : {r['delta_t_s']:.1f} s")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out}")

    _maybe_plot(coupled)


def _maybe_plot(coupled) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (matplotlib unavailable: {e})")
        return
    try:
        T = coupled.temperature_K
        q = coupled.heat_generation
        scene = coupled.scene
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        im0 = ax[0].imshow(T, origin="lower", cmap="inferno")
        ax[0].set_title(f"T (K)  ΔT={coupled.thermal.delta_T_K:.0f} K")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)

        q_show = np.zeros_like(q)
        charge = scene.target_mask | scene.gangue_mask
        q_show[charge] = q[charge]
        im1 = ax[1].imshow(q_show, origin="lower", cmap="magma")
        ax[1].set_title("absorbed power density (charge)")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)

        ax[2].plot(coupled.history_max_delta, "-o", ms=4)
        ax[2].set(xlabel="iteration", ylabel="max |ΔT| in charge (K)", title="coupling convergence")
        fig.suptitle("EM–thermal steady state (2D spatial)")
        fig.tight_layout()
        fig.savefig("data/thermal.png", dpi=110)
        print("  wrote data/thermal.png")
    except Exception as e:  # noqa: BLE001
        print(f"  (plot failed: {e})")


if __name__ == "__main__":
    main()
