"""Refresh an existing pipeline run with openEMS port/field dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mw_inv.fdfd import Grid
from mw_inv.geometry import Materials
from mw_inv.provenance import default_provenance
from mw_inv.run_manifest import RunManifest
from mw_inv.run_refresh import apply_triangulation_refresh


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="pipeline run directory (contains manifest.json)")
    ap.add_argument("--openems-dump-dir", required=True, help="openEMS dump root (<case>/port_metrics.json)")
    ap.add_argument("--grid", type=int, default=None, help="override grid size (defaults to manifest search grid)")
    ap.add_argument("--Lz", type=float, default=0.36)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest.json in {run_dir}")

    m = RunManifest.load(manifest_path)
    if not m.provenance:
        m.provenance = default_provenance(Path(__file__).resolve().parents[3])
    m.cli.setdefault("updates", [])
    m.cli["updates"].append(
        {
            "tool": "update_run_with_openems",
            "openems_dump_dir": str(args.openems_dump_dir),
            "grid": args.grid,
            "Lz": args.Lz,
        }
    )
    search_path = Path(m.search_path) if m.search_path else run_dir / "search_summary.json"
    if not search_path.is_file():
        raise FileNotFoundError(f"missing search summary at {search_path}")

    search_summary = json.loads(search_path.read_text())
    materials_label = search_summary.get("materials", m.materials)
    materials = Materials.from_pair(materials_label)

    grid_n = int(args.grid or search_summary.get("grid") or 71)
    grid = Grid(nx=grid_n, ny=grid_n, Lx=0.36, Ly=0.36)
    dump_dir = Path(args.openems_dump_dir)
    top_k = int(search_summary["openems_top_k"]) if search_summary.get("openems_top_k") else None

    refresh = apply_triangulation_refresh(
        m,
        run_dir,
        search_path=search_path,
        grid=grid,
        materials=materials,
        materials_label=materials_label,
        openems_dump_dir=dump_dir,
        Lz=args.Lz,
        top_k=top_k,
    )
    m.search_summary = search_summary
    m.write(manifest_path)

    print("=== Updated run with openEMS dumps ===")
    print(f"  run_dir       : {run_dir}")
    print(f"  openems dumps : {dump_dir}")
    print("  wrote         : solver_triangulation.json, validation_gate_report.json, manifest.json")
    print(f"  promotion tier: {refresh.assessment.tier.value}")


if __name__ == "__main__":
    main()

