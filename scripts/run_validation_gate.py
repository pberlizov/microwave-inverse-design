"""Full validation gate workflow for pyrite_in_calcite canonical designs.

    python scripts/run_validation_gate.py
    python scripts/run_validation_gate.py --search data/pyrite_search_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_export import export_all_cases, load_search_cases  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.maturity import status_dict  # noqa: E402
from mw_inv.openems_export import write_calibration_model  # noqa: E402
from mw_inv.solver_triangulation import triangulate_from_search, write_triangulation_report  # noqa: E402
from mw_inv.validation_gate import evaluate_gate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--search", default=None, help="search JSON (default: run fresh search)")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--grid", type=int, default=71)
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--openems-dump-dir", default=None)
    ap.add_argument("--out", default="data/validation_gate_report.json")
    ap.add_argument("--export-dir", default="data/design_exports/pyrite_gate")
    args = ap.parse_args()

    search_path = Path(args.search) if args.search else Path("data/pyrite_search_summary.json")
    if not search_path.is_file():
        print(f"No search file at {search_path} — running search ({args.trials} trials)...")
        from mw_inv.search import best, optuna_search, random_search, evaluate_params
        from mw_inv.geometry import CavityParams

        grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
        mats = Materials.from_pair(args.materials)
        t0 = time.time()
        rnd = random_search(grid, args.trials, args.seed, materials=mats)
        tpe = optuna_search(grid, args.trials, args.seed, materials=mats)
        base = evaluate_params(grid, CavityParams(), mats)
        summary = {
            "materials": args.materials,
            "trials": args.trials,
            "seed": args.seed,
            "grid": args.grid,
            "baseline_untuned": {"selectivity": base.selectivity, "contrast": base.contrast},
            "random_search": {
                "best_selectivity": best(rnd).selectivity,
                "best_params": best(rnd).params,
                "seconds": round(time.time() - t0, 1),
            },
            "tpe_search": {
                "best_selectivity": best(tpe).selectivity,
                "best_params": best(tpe).params,
            },
        }
        search_path.parent.mkdir(parents=True, exist_ok=True)
        search_path.write_text(json.dumps(summary, indent=2))
        print(f"  wrote {search_path}")

    materials = Materials.from_pair(args.materials)
    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    dump_dir = Path(args.openems_dump_dir) if args.openems_dump_dir else None

    cal_path = write_calibration_model(Path(args.export_dir) / "calibration_cavity.m")
    cases = load_search_cases(search_path)
    export_all_cases(args.export_dir, cases, materials, grid_n=args.grid)

    rows = triangulate_from_search(search_path, grid, materials, openems_dump_dir=dump_dir)
    gate = evaluate_gate(rows)

    report = {
        "materials": args.materials,
        "search_source": str(search_path),
        "calibration_model": str(cal_path),
        "export_dir": args.export_dir,
        "maturity": {
            "solver_triangulation": status_dict("solver_triangulation"),
            "openems_port": status_dict("openems_port"),
        },
        "triangulation": [r.to_dict() for r in rows],
        "gate": gate.to_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    write_triangulation_report(out.with_name("solver_triangulation_pyrite.json"), rows, materials_label=args.materials)

    print("=== Validation gate ===")
    print(f"  materials     : {args.materials}")
    print(f"  calibration   : {cal_path}")
    print(f"  gate passed   : {gate.passed}")
    for c in gate.checks:
        mark = "OK" if c.passed else "FAIL"
        print(f"    [{mark}] {c.name}: {c.detail}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
