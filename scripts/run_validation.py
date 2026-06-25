"""Run forward-model validation suite and write JSON report.

    python scripts/run_validation.py
    python scripts/run_validation.py --no-meep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.validation import run_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/validation_report.json")
    ap.add_argument("--no-meep", action="store_true")
    args = ap.parse_args()

    report = run_all(include_meep=not args.no_meep)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))

    print(f"Validation {'PASSED' if report.passed else 'FAILED'} — {len(report.checks)} checks")
    for c in report.checks:
        mark = "ok" if c.passed else "FAIL"
        print(f"  [{mark}] {c.name}: {c.detail}")
    print(f"\n  wrote {out}")
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
