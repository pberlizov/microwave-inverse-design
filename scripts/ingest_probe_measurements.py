"""Ingest open-coax probe measurements and compare to Gabriel anchors.

    python scripts/ingest_probe_measurements.py data/measured_eps.json
    python scripts/ingest_probe_measurements.py data/measured_eps.json --phantom saline_2_vs_0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.phantom_calibration import compare_measured_vs_anchor, load_measured_eps  # noqa: E402
from mw_inv.phantom_data import PHANTOM_RECIPES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("measured", type=str)
    ap.add_argument("--phantom", choices=sorted(PHANTOM_RECIPES), default="saline_2_vs_0.5")
    ap.add_argument("--out", default="data/probe_calibration_report.json")
    args = ap.parse_args()

    batches = load_measured_eps(args.measured)
    cmp = compare_measured_vs_anchor(args.phantom, args.measured)
    report = {"batches": {k: {"eps": [v.eps.real, v.eps.imag], "method": v.method} for k, v in batches.items()}, "comparison": cmp}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"=== Probe calibration ({args.phantom}) ===")
    for row in cmp["comparisons"]:
        if row.get("status") == "missing":
            print(f"  MISSING batch {row['batch']} — measure before bench run")
        else:
            print(f"  {row['role']:6s} {row['batch']:16s}  drift ε′={row['drift_real']:+.2f}  ε″={row['drift_imag']:+.3f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
