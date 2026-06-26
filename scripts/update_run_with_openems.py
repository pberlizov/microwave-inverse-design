"""Update an existing pipeline run with openEMS dumps, then recompute promotion tier.

Typical flow:
1) Run the pipeline to produce a run dir + export bundle:
     python scripts/run_pipeline.py --materials pyrite_in_calcite --trials 24
2) Export openEMS models (if you didn't already):
     python scripts/export_design.py --search data/runs/<RUN>/search_summary.json --out-dir data/runs/<RUN>/design_exports
3) Run openEMS in Octave from the export folder (creates openems_runs/<case>/Et/Et_0000.h5):
     cd data/runs/<RUN>/design_exports
     octave -qf run_openems_all.m
4) Update the run's triangulation/gate/manifest using those dumps:
     python scripts/update_run_with_openems.py --run-dir data/runs/<RUN> --openems-dump-dir data/runs/<RUN>/design_exports/openems_runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import Materials  # noqa: E402
from mw_inv.provenance import default_provenance  # noqa: E402
from mw_inv.run_manifest import RunManifest, finalize_promotion  # noqa: E402
from mw_inv.solver_triangulation import triangulate_from_search, write_triangulation_report  # noqa: E402
from mw_inv.validation_gate import evaluate_gate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="pipeline run directory (contains manifest.json)")
    ap.add_argument("--openems-dump-dir", required=True, help="openEMS dump root (<case>/Et/Et_0000.h5)")
    ap.add_argument("--grid", type=int, default=None, help="override grid size (defaults to manifest search grid)")
    ap.add_argument("--Lz", type=float, default=0.36)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest.json in {run_dir}")

    m = RunManifest.load(manifest_path)
    if not m.provenance:
        # Best-effort: attach provenance when the manifest predates this feature.
        m.provenance = default_provenance(Path(__file__).resolve().parents[1])
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

    rows = triangulate_from_search(
        search_path, grid, materials, Lz=args.Lz, openems_dump_dir=dump_dir,
        top_k=int(search_summary["openems_top_k"]) if search_summary.get("openems_top_k") else None,
    )
    tri_path = run_dir / "solver_triangulation.json"
    write_triangulation_report(
        tri_path,
        rows,
        materials_label=materials_label,
        meta={"search_source": str(search_path), "openems_dump_dir": str(dump_dir)},
    )

    gate = evaluate_gate(rows)
    gate_path = run_dir / "validation_gate_report.json"
    gate_payload = {
        "materials": materials_label,
        "search_source": str(search_path),
        "openems_dump_dir": str(dump_dir),
        "gate": gate.to_dict(),
        "triangulation": [r.to_dict() for r in rows],
    }
    gate_path.write_text(json.dumps(gate_payload, indent=2))

    m.triangulation_path = str(tri_path)
    m.triangulation = {"rows": [r.to_dict() for r in rows], "rank_agreement": gate.rank_agreement, "openems_dump_dir": str(dump_dir)}
    m.gate_path = str(gate_path)
    m.gate = gate.to_dict()
    m.search_path = str(search_path)
    m.search_summary = search_summary

    assessment = finalize_promotion(m)
    m.write(manifest_path)

    print("=== Updated run with openEMS dumps ===")
    print(f"  run_dir       : {run_dir}")
    print(f"  openems dumps : {dump_dir}")
    print(f"  wrote         : {tri_path.name}, {gate_path.name}, manifest.json")
    print(f"  promotion tier: {assessment.tier.value}")


if __name__ == "__main__":
    main()
