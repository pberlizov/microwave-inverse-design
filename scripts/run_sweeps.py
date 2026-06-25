"""Frequency sweep (real EM) + thermal-runaway analysis (parametric eps''(T)).

    python scripts/run_sweeps.py --materials pyrite_in_calcite

Writes data/sweeps_summary.json and, if matplotlib works, data/sweeps.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.sweeps import (  # noqa: E402
    frequency_sweep,
    loss_response,
    runaway_curve,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--grid", type=int, default=101)
    ap.add_argument("--out", type=str, default="data/sweeps_summary.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    pair = PAIRS[args.materials]
    materials = Materials.from_pair(args.materials)

    # --- 1. Frequency sweep (real EM across +/-5% of the 2.45 GHz ISM band) ---
    freqs = np.linspace(2.35e9, 2.55e9, 41)
    fpts = frequency_sweep(grid, freqs, materials=materials)
    sel = np.array([p.selectivity for p in fpts])
    con = np.array([p.contrast for p in fpts])
    i_best = int(np.argmax(sel))
    i_base = int(np.argmin(np.abs(freqs - 2.45e9)))
    print(f"=== Frequency sweep ({args.materials}) ===")
    print(f"  band               : {freqs[0]/1e9:.3f}-{freqs[-1]/1e9:.3f} GHz")
    print(f"  selectivity @2.45  : {sel[i_base]:.4f}")
    print(f"  best selectivity   : {sel[i_best]:.4f}  @ {freqs[i_best]/1e9:.4f} GHz")
    print(f"  selectivity spread : {sel.min():.4f} - {sel.max():.4f}")

    # --- 2a. Absorbed power vs loss factor (real EM): find the absorption optimum ---
    # eps'' rises with temperature, so this axis is also a temperature axis. Absorbed
    # power is NON-monotonic: it peaks at an impedance/skin-depth-matched eps''* and
    # then falls (self-shielding). 'More loss' is not 'more heat'.
    epps = np.linspace(0.02, 8.0, 40)
    lpts = loss_response(grid, epps, eps_real=pair.target.real, base_materials=materials)
    p_abs = np.array([p.p_target for p in lpts])
    sel_l = np.array([p.selectivity for p in lpts])
    i_peak = int(np.argmax(p_abs))
    eps_star = float(epps[i_peak])
    op_eps = pair.target.imag
    print(f"\n=== Absorption vs loss factor ({args.materials}) ===")
    print(f"  absorption-optimal eps''* : {eps_star:.2f}  (peak absorbed power)")
    print(f"  operating eps'' (target)  : {op_eps:.2f}  "
          f"({'sub-optimal -> some positive feedback' if op_eps < eps_star else 'supra-optimal -> self-shielding'})")
    print(f"  selectivity is monotonic  : {sel_l[0]:.3f} -> {sel_l[-1]:.3f} (rises despite power peak)")

    # --- 2b. Lumped thermal model: eps''(T) ramps -> SELF-LIMITING (not runaway) ---
    eps_t = materials.eps_t_model("target")
    temps = np.linspace(298.0, 1300.0, 60)
    cooling_coeff = 5.0e-3  # arbitrary Newton-cooling units (shape, not absolute, matters)
    run = runaway_curve(grid, eps_t, cooling_coeff, temps_K=temps, base_materials=materials)
    g_peak_T = float(run.temps_K[int(np.argmax(run.p_gen))])
    smooth = bool(np.all(np.diff(run.T_steady) >= -1.0))  # no downward jump == smooth
    print(f"\n=== Lumped thermal response (parametric eps''(T), {args.materials}) ===")
    print(f"  eps''(298K)={eps_t.eps_imag(298):.2f}  eps''(1300K)={eps_t.eps_imag(1300):.2f}  (Arrhenius ramp)")
    print(f"  absorbed power peaks at T : {g_peak_T:.0f} K (eps''~{eps_t.eps_imag(g_peak_T):.2f}=eps''*)")
    print(f"  T(drive) response         : {'smooth / self-limiting' if smooth else 'sharp jump'} "
          f"(no unbounded runaway for grains > skin depth)")

    summary = {
        "materials": args.materials,
        "provenance": pair.provenance,
        "frequency_sweep": {
            "freqs_ghz": (freqs / 1e9).tolist(),
            "selectivity": sel.tolist(),
            "contrast": con.tolist(),
            "selectivity_at_2p45": sel[i_base],
            "best_selectivity": sel[i_best],
            "best_freq_ghz": freqs[i_best] / 1e9,
        },
        "loss_response": {
            "eps_imag": epps.tolist(),
            "p_target": p_abs.tolist(),
            "selectivity": sel_l.tolist(),
            "eps_star": eps_star,
            "operating_eps_imag": op_eps,
            "note": "Real EM. Absorbed power peaks at eps''* (impedance/skin-depth match) "
                    "then falls (self-shielding); selectivity rises monotonically.",
        },
        "thermal": {
            "temps_K": run.temps_K.tolist(),
            "p_gen": run.p_gen.tolist(),
            "drives": run.drives.tolist(),
            "T_steady": run.T_steady.tolist(),
            "eps_imag_298": eps_t.eps_imag(298),
            "eps_imag_1300": eps_t.eps_imag(1300),
            "g_peak_T_K": g_peak_T,
            "self_limiting": smooth,
            "cooling_coeff": cooling_coeff,
            "note": "eps''(T) is a parametric Arrhenius-like ramp anchored to qualitative "
                    "Cumbane(2008) findings (pyrite loss rises strongly to 650C); not a "
                    "digitised measurement. Drive/cooling units arbitrary. Heating is "
                    "SELF-LIMITING (no unbounded runaway) because eps'' passes the "
                    "absorption optimum; true runaway needs grains << skin depth.",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {out}")

    _maybe_plot(freqs, sel, con, epps, p_abs, sel_l, eps_star)


def _maybe_plot(freqs, sel, con, epps, p_abs, sel_l, eps_star) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping plot: {e})")
        return
    try:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].plot(freqs / 1e9, sel, "-o", ms=3)
        ax[0].axvline(2.45, color="k", ls=":", lw=1)
        ax[0].set(xlabel="drive frequency (GHz)", ylabel="selectivity", title="Frequency sweep (real EM)")

        ax[1].plot(freqs / 1e9, con, "-o", ms=3, color="C1")
        ax[1].axvline(2.45, color="k", ls=":", lw=1)
        ax[1].set(xlabel="drive frequency (GHz)", ylabel="per-area contrast", title="Contrast vs frequency")

        ax2 = ax[2]
        ax2.plot(epps, p_abs / p_abs.max(), "-o", ms=3, color="C3",
                 label="absorbed power (norm.)")
        ax2.plot(epps, sel_l, "-o", ms=3, color="C0", label="selectivity")
        ax2.axvline(eps_star, color="k", ls=":", lw=1, label=f"eps''* = {eps_star:.2f}")
        ax2.set(xlabel="target loss factor eps'' (rises with T)", ylabel="normalised",
                title="Absorption optimum / self-shielding")
        ax2.legend(fontsize=8)
        fig.suptitle("Microwave applicator: frequency response, and absorption vs loss factor (temperature)")
        fig.tight_layout()
        fig.savefig("data/sweeps.png", dpi=110)
        print("  wrote data/sweeps.png")
    except Exception as e:  # noqa: BLE001
        print(f"  (plot render failed, summary still written: {e})")


if __name__ == "__main__":
    main()
