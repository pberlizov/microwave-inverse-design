"""Ingest curated literature tables into data/measured_dielectrics/.

Writes Hartlieb bedrock, USBM low-loss gangue, and dielectric_data mineral exports.
No network fetch — tables are embedded in mw_inv.literature_ingest.

    python scripts/ingest_literature_datasets.py
    python scripts/ingest_literature_datasets.py --adapter hartlieb
    python scripts/ingest_literature_datasets.py --status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.external_datasets import ingest_status, load_datasets_catalog  # noqa: E402
from mw_inv.literature_ingest import ingest_all_auto, ingest_literature_dataset, list_adapters  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest literature dielectric datasets")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument(
        "--adapter",
        choices=list_adapters(),
        default=None,
        help="run one adapter (default: all auto from datasets_catalog.json)",
    )
    ap.add_argument("--status", action="store_true", help="print ingest status and exit")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else None

    if args.status:
        cat = load_datasets_catalog(root)
        print(f"=== Datasets catalog ({cat.version}) ===")
        for row in ingest_status(root):
            print(f"  [{row['status']:8s}] {row['id']:32s} {row.get('path') or ''}")
        return

    if args.adapter:
        out = ingest_literature_dataset(args.adapter, root)
        print(f"  wrote {out}")
        return

    paths = ingest_all_auto(root)
    print("=== Literature ingest ===")
    for p in paths:
        print(f"  wrote {p}")
    print(f"  total: {len(paths)} files")

    status = ingest_status(root)
    pending = [r for r in status if r["status"] == "pending" and r.get("auto")]
    if pending:
        print("\n  WARNING: auto entries still pending:")
        for r in pending:
            print(f"    - {r['id']}")


if __name__ == "__main__":
    main()
