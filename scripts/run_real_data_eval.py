"""Evaluate on all versioned real-data inputs in data/.

Discovers ore profiles, deposit ε libraries, literature benchmarks, material pairs,
phantom recipes, and bench JSON (example or live copies), then writes a consolidated
FDFD evaluation report.

    python scripts/run_real_data_eval.py
    python scripts/run_real_data_eval.py --quick --out data/real_data_eval_report.json

For live bench data, copy:
    data/measured_eps.example.json  → data/measured_eps.json
    data/lab_measurements.example.json → data/lab_measurements.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.real_data_eval import discover_real_data_catalog, evaluate_real_data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch evaluation on all real/versioned data")
    ap.add_argument("--data-root", type=str, default=None, help="default: repo data/")
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--quick", action="store_true", help="smaller grid, fewer phantom trials")
    ap.add_argument("--catalog-only", action="store_true", help="list sources without FDFD eval")
    ap.add_argument(
        "--ingest-literature",
        action="store_true",
        help="write literature measured_dielectrics JSON before eval",
    )
    ap.add_argument("--out", type=str, default="data/real_data_eval_report.json")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else None
    if args.ingest_literature:
        from mw_inv.literature_ingest import ingest_all_auto

        for p in ingest_all_auto(root):
            print(f"  ingested literature → {p}")

    catalog = discover_real_data_catalog(root)

    print("=== Real data catalog ===")
    print(f"  sources: {catalog.to_dict()['n_sources']}")
    for s in catalog.sources:
        print(f"    [{s.kind:16s}] {s.label:40s} ({s.provenance})")
    if catalog.missing_user_inputs:
        print("\n  To add live bench data:")
        for hint in catalog.missing_user_inputs:
            print(f"    • {hint}")
    if catalog.external_datasets:
        n_ing = sum(1 for r in catalog.external_datasets if r.get("status") == "ingested")
        n_pend = sum(1 for r in catalog.external_datasets if r.get("status") == "pending")
        print(f"\n  external catalog: {n_ing} ingested, {n_pend} pending (see data/datasets_catalog.json)")

    if args.catalog_only:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"catalog": catalog.to_dict()}, indent=2))
        print(f"\n  wrote catalog → {out}")
        return

    report = evaluate_real_data(root, grid_n=args.grid, quick=args.quick)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    sm = report["summary"]
    print("\n=== Evaluation summary ===")
    print(f"  benchmarks       : {'PASS' if sm['benchmarks_passed'] else 'FAIL'}")
    print(f"  ore profiles     : {sm['n_ores']}")
    print(f"  deposit libraries: {sm['n_deposit_libraries']} ({sm['n_deposit_points']} ε points)")
    print(f"  material pairs   : {sm['n_material_pairs']}")
    print(f"  phantoms         : {sm['n_phantoms']}")
    print(f"  live probe JSON  : {sm['using_live_probe']}")
    print(f"  live lab JSON    : {sm['using_live_lab']}")
    for row in report.get("ore_profiles", []):
        if "error" in row:
            print(f"    ore ERROR {row.get('ore_path')}: {row['error']}")
            continue
        print(
            f"    {row['label']:32s} mode={row.get('materials_mode')} "
            f"sel={row.get('fdfd_selectivity', 0):.3f} arcing={row.get('arcing_risk')}"
        )
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
