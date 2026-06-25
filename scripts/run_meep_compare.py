"""Compare FDFD vs MEEP 2D and primitive 3D FDTD."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid, solve  # noqa: E402
from mw_inv.fom import evaluate  # noqa: E402
from mw_inv.geometry import CavityParams, Materials, build_scene  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.meep_3d import compare_fdfd_meep_3d  # noqa: E402
from mw_inv.meep_compare import meep_available  # noqa: E402
from mw_inv.maturity import status_dict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--Lz", type=float, default=0.36)
    ap.add_argument("--out", default="data/meep_compare.json")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    mats = Materials.from_pair(args.materials)
    scene = build_scene(grid, CavityParams(), mats)
    fdfd_sel = evaluate(
        solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r),
        scene,
    ).selectivity

    result = compare_fdfd_meep_3d(scene, grid, fdfd_sel, materials=mats, Lz=args.Lz)
    result["materials"] = args.materials
    result["grid"] = args.grid
    result["maturity"] = {
        "meep_2d": status_dict("meep_2d_crosscheck"),
        "meep_3d_primitive": status_dict("meep_3d_primitive"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(f"=== FDFD vs MEEP ({args.materials}) ===")
    print(f"  FDFD 2D              : {fdfd_sel:.4f}")
    if not meep_available():
        print("  MEEP not installed")
    else:
        print(f"  MEEP 2D              : {result.get('meep_2d_selectivity', 0):.4f}")
        print(f"  MEEP 3D (primitives) : {result.get('meep_3d_primitive_selectivity', 0):.4f}")
        print(f"  MEEP 3D (extrusion)  : {result.get('meep_3d_extrusion_selectivity', 0):.4f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
