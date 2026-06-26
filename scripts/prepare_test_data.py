"""Prepare all literature datasets and validate before testing.

Runs every auto-ingest adapter, checks ore profiles load, and optionally
runs a quick real-data FDFD sweep.

    python scripts/prepare_test_data.py
    python scripts/prepare_test_data.py --eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.external_datasets import ingest_status  # noqa: E402
from mw_inv.literature_ingest import ingest_all_auto  # noqa: E402
from mw_inv.ore_profiles import load_ore_profile, ore_summary  # noqa: E402
from mw_inv.real_data_eval import discover_real_data_catalog, evaluate_real_data  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _validate_ores(data_root: Path) -> tuple[int, list[str]]:
    ores_dir = data_root / "ores"
    errors: list[str] = []
    n_ok = 0
    for p in sorted(ores_dir.rglob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            ore = load_ore_profile(p)
            ore_summary(ore, ore_profile_path=p)
            n_ok += 1
        except (ValueError, FileNotFoundError, KeyError) as exc:
            errors.append(f"{p}: {exc}")
    return n_ok, errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest literature data and validate for testing")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--eval", action="store_true", help="run quick real_data_eval after ingest")
    ap.add_argument("--out", type=str, default="data/test_data_prep_report.json")
    args = ap.parse_args()

    root = Path(args.data_root) if args.data_root else _repo_root() / "data"

    print("=== Literature ingest (all adapters) ===")
    paths = ingest_all_auto(root)
    print(f"  wrote {len(paths)} files")

    status = ingest_status(root)
    pending = [r for r in status if r.get("auto") and r["status"] != "ingested"]
    if pending:
        print("  ERROR: pending auto-ingest entries:")
        for r in pending:
            print(f"    - {r['id']}")
        sys.exit(2)

    for row in status:
        if row.get("auto"):
            print(f"  [{row['status']:8s}] {row['id']}")

    print("\n=== Ore profile validation ===")
    n_ores, ore_errors = _validate_ores(root)
    print(f"  validated {n_ores} ore profiles")
    if ore_errors:
        print("  ERRORS:")
        for msg in ore_errors[:15]:
            print(f"    - {msg}")
        sys.exit(2)

    cat = discover_real_data_catalog(root)
    report: dict[str, object] = {
        "n_ingest_files": len(paths),
        "ingest_status": status,
        "n_ore_profiles": n_ores,
        "catalog_n_sources": cat.to_dict()["n_sources"],
    }

    if args.eval:
        print("\n=== Quick real-data evaluation ===")
        eval_report = evaluate_real_data(root, quick=True)
        sm = eval_report["summary"]
        print(f"  ores={sm['n_ores']}  deposits={sm['n_deposit_libraries']}  "
              f"deposit_points={sm['n_deposit_points']}  benchmarks={'PASS' if sm['benchmarks_passed'] else 'FAIL'}")
        ore_err = [r for r in eval_report.get("ore_profiles", []) if "error" in r]
        if ore_err:
            print(f"  ore eval errors: {len(ore_err)}")
            for r in ore_err[:5]:
                print(f"    - {r.get('ore_path')}: {r['error']}")
            sys.exit(2)
        report["eval_summary"] = sm

    out = _repo_root() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
