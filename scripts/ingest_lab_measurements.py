"""Validate lab bench ΔT JSON (E0 ingest)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.bench_ingest import validate_lab_measurements  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate lab_measurements.json schema")
    ap.add_argument("lab_json", type=str, help="path to lab measurements JSON")
    ap.add_argument("--out", type=str, default=None, help="optional validated copy path")
    args = ap.parse_args()

    path = Path(args.lab_json)
    issues = validate_lab_measurements(path)
    if issues:
        print("VALIDATION FAILED:")
        for issue in issues:
            print(f"  {issue.path}: {issue.message}")
        raise SystemExit(1)

    payload = json.loads(path.read_text())
    print(f"OK: {path} ({len(payload if isinstance(payload, list) else payload.get('measurements', []))} records)")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
