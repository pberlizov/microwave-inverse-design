"""Triangulate FDFD vs MEEP (and optional openEMS dumps) on canonical designs.

    python scripts/run_solver_triangulation.py --search data/search_summary.json
    python scripts/run_solver_triangulation.py --search data/search_summary.json \\
        --openems-dump-dir data/design_exports/search_summary/openems_runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.maturity import status_dict  # noqa: E402
from mw_inv.meep_compare import meep_available  # noqa: E402
from mw_inv.solver_triangulation import (  # noqa: E402
    triangulate_from_search,
    write_triangulation_report,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", default="data/search_summary.json")
    ap.add_argument("--materials", choices=sorted(PAIRS), default=None)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--Lz", type=float, default=0.36)
    ap.add_argument("--openems-dump-dir", default=None)
    ap.add_argument("--top-k", type=int, default=0, help="use untuned + top-K TPE trials from search summary")
    ap.add_argument("--out", default="data/solver_triangulation.json")
    args = ap.parse_args()

    search_data = json.loads(Path(args.search).read_text())
    materials_label = args.materials or search_data.get("materials", "pyrite_in_calcite")
    materials = Materials.from_pair(materials_label)
    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    dump_dir = Path(args.openems_dump_dir) if args.openems_dump_dir else None

    top_k = args.top_k if args.top_k > 0 else None
    rows = triangulate_from_search(
        args.search, grid, materials, Lz=args.Lz, openems_dump_dir=dump_dir, top_k=top_k,
    )
    write_triangulation_report(
        args.out,
        rows,
        materials_label=materials_label,
        meta={
            "search_source": args.search,
            "maturity": {
                "meep_2d": status_dict("meep_2d_crosscheck"),
                "meep_3d_primitive": status_dict("meep_3d_primitive"),
                "openems_port": status_dict("openems_port"),
            },
        },
    )

    print("=== Solver triangulation ===")
    print(f"  materials : {materials_label}")
    print(f"  MEEP      : {'installed' if meep_available() else 'not installed (FDFD only)'}")
    for row in rows:
        line = f"  {row.label:12s}  FDFD={row.fdfd_selectivity:.4f}"
        if row.meep_2d_selectivity is not None:
            line += f"  MEEP2D={row.meep_2d_selectivity:.4f} (Δ={row.rel_err_meep_2d:.3f})"
        if row.meep_3d_primitive_selectivity is not None:
            line += f"  MEEP3D={row.meep_3d_primitive_selectivity:.4f}"
        if row.openems_selectivity is not None:
            line += f"  openEMS={row.openems_selectivity:.4f}"
        print(line)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
