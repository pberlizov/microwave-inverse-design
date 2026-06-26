"""openEMS port-truth validation harness (A1): export + ingest port metrics.

This is the CLI version of ``scripts/run_port_validation.py`` so users can run:

  mw-inv-port-validation --out-dir data/port_validation
  mw-inv-port-validation --ingest-dir data/port_validation/openems_runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mw_inv.design_export import cases_from_search_summary, export_all_cases
from mw_inv.fdfd import Grid
from mw_inv.geometry import Materials
from mw_inv.openems_export import write_calibration_model
from mw_inv.openems_postprocess import load_port_metrics
from mw_inv.solver_triangulation import triangulate_cases
from mw_inv.validation_gate import evaluate_gate


def _export_bundle(out_dir: Path, materials_label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    materials = Materials.from_pair(materials_label)
    write_calibration_model(out_dir / "calibration_cavity.m", sim_path="./openems_runs/calibration")
    cases = cases_from_search_summary({
        "tpe_search": {"best_params": {"feed_wall": "bottom", "feed_along_frac": 0.55}},
    })
    export_all_cases(out_dir, cases, materials, grid_n=61)
    print(f"  exported calibration + {len(cases)} ore cases -> {out_dir}")
    print("  run in Octave:  cd", out_dir, "&& octave -qf run_openems_all.m")


def _ingest_runs(ingest_dir: Path, materials_label: str, grid_n: int) -> dict:
    materials = Materials.from_pair(materials_label)
    grid = Grid(nx=grid_n, ny=grid_n, Lx=0.36, Ly=0.36)
    cases = cases_from_search_summary({
        "tpe_search": {"best_params": {"feed_wall": "bottom", "feed_along_frac": 0.55}},
    })
    rows = triangulate_cases(cases, grid, materials, openems_dump_dir=ingest_dir)
    gate = evaluate_gate(rows)

    cal_metrics = None
    cal_dir = ingest_dir / "calibration"
    if cal_dir.is_dir():
        pm = cal_dir / "port_metrics.json"
        if pm.is_file():
            cal_metrics = load_port_metrics(pm).to_dict()

    report = {
        "materials": materials_label,
        "ingest_dir": str(ingest_dir),
        "calibration": cal_metrics,
        "cases": [r.to_dict() for r in rows],
        "gate": gate.to_dict(),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="openEMS port validation harness (A1)")
    ap.add_argument("--out-dir", default="data/port_validation", help="export bundle directory")
    ap.add_argument("--ingest-dir", default=None, help="openEMS run root (openems_runs/)")
    ap.add_argument("--out", default="data/port_validation_report.json")
    ap.add_argument("--materials", default="pyrite_in_calcite")
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args(argv)

    if args.ingest_dir:
        report = _ingest_runs(Path(args.ingest_dir), args.materials, args.grid)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print("=== Port validation ingest ===")
        for row in report["cases"]:
            s11 = row.get("openems_s11_mag")
            coup = row.get("openems_coupling_eff")
            sel = row.get("openems_selectivity")
            extra = ""
            if s11 is not None and coup is not None:
                extra = f"  |S11|={s11:.3f}  coupling={coup:.3f}"
            print(f"  {row['label']:12s}  sel={sel}{extra}")
        if report["calibration"]:
            c = report["calibration"]
            print(f"  calibration   |S11|={c['s11_mag']:.3f}  coupling={c['coupling_eff']:.3f}")
        print(f"  gate: {'PASS' if report['gate']['passed'] else 'FAIL'}")
        if report["gate"].get("openems_diagnosis"):
            print(f"  diagnosis: {report['gate']['openems_diagnosis']}")
        print(f"  wrote {out}")
        return

    _export_bundle(Path(args.out_dir), args.materials)
    if not args.export_only:
        print("Re-run with --ingest-dir after openEMS completes.")


if __name__ == "__main__":
    main()

