"""Export permittivity, |E_z|, and absorbed-power maps for untuned vs tuned applicators.

Always writes data/fields.npz (raw arrays, inspectable in any environment). Also
attempts data/fields.png; on some broken matplotlib/Python combos savefig raises a
RecursionError (e.g. matplotlib 3.10 on CPython 3.14) -- the PNG is then skipped and
the npz is still produced.

Run:  python scripts/plot_fields.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid, absorbed_power_density, solve  # noqa: E402
from mw_inv.fom import evaluate  # noqa: E402
from mw_inv.geometry import CavityParams, Materials, build_scene  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.search import best, optuna_search  # noqa: E402


def fields_for(params: CavityParams, grid: Grid, materials: Materials) -> dict:
    scene = build_scene(grid, params, materials)
    res = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r)
    rep = evaluate(res, scene)
    return {
        "eps_imag": scene.eps_r.imag,
        "absE": np.abs(res.Ez),
        "power": absorbed_power_density(res),
        "target_mask": scene.target_mask,
        "selectivity": rep.selectivity,
        "contrast": rep.contrast,
    }


def try_png(untuned: dict, tuned: dict, grid: Grid, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        ext = [0, grid.Lx, 0, grid.Ly]
        rows = [("untuned", untuned), ("tuned", tuned)]
        cols = [("Im(eps) [loss]", "eps_imag", "viridis"),
                ("|E_z|", "absE", "magma"),
                ("absorbed power", "power", "inferno")]
        for r, (tag, d) in enumerate(rows):
            for c, (label, key, cmap) in enumerate(cols):
                ax = axes[r][c]
                title = f"{tag}: {label}"
                if key == "power":
                    title += f"\nsel={d['selectivity']:.3f}"
                ax.imshow(d[key], origin="lower", cmap=cmap, extent=ext)
                ax.set_title(title, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle("Microwave applicator inverse design (thin slice): "
                     "selective heating of target mineral phase", fontsize=11)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        return True
    except Exception as exc:  # broken matplotlib/Python combo, etc.
        print(f"PNG skipped ({type(exc).__name__}): {str(exc)[:80]}")
        return False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    args = ap.parse_args()
    materials = Materials.from_pair(args.materials)

    grid = Grid(nx=121, ny=121, Lx=0.36, Ly=0.36)
    untuned = fields_for(CavityParams(), grid, materials)

    trials = optuna_search(grid, n_trials=80, seed=1903, materials=materials)
    b = best(trials)
    tuned_params = CavityParams(**b.params)
    tuned = fields_for(tuned_params, grid, materials)

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "fields.npz"
    np.savez_compressed(
        npz,
        untuned_power=untuned["power"], untuned_absE=untuned["absE"],
        untuned_eps_imag=untuned["eps_imag"], target_mask=untuned["target_mask"],
        tuned_power=tuned["power"], tuned_absE=tuned["absE"],
        tuned_eps_imag=tuned["eps_imag"],
        untuned_selectivity=untuned["selectivity"], tuned_selectivity=tuned["selectivity"],
        untuned_contrast=untuned["contrast"], tuned_contrast=tuned["contrast"],
    )
    print(f"wrote {npz}")
    print(f"  untuned selectivity {untuned['selectivity']:.3f}  contrast {untuned['contrast']:.2f}")
    print(f"  tuned   selectivity {tuned['selectivity']:.3f}  contrast {tuned['contrast']:.2f}")

    if try_png(untuned, tuned, grid, out_dir / "fields.png"):
        print("wrote data/fields.png")


if __name__ == "__main__":
    main()
