"""Build a bench RF port report from VNA (.s1p) + openEMS port_metrics.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.rf_port_report import build_port_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unloaded-s1p", required=True)
    ap.add_argument("--loaded-s1p", default=None)
    ap.add_argument("--openems-port-metrics", default=None, help="port_metrics.json from openEMS")
    ap.add_argument("--out", default="data/rf_port_report.json")
    ap.add_argument("--freq", type=float, default=2.45e9)
    ap.add_argument("--band-lo", type=float, default=None)
    ap.add_argument("--band-hi", type=float, default=None)
    args = ap.parse_args()

    report = build_port_report(
        unloaded_s1p=args.unloaded_s1p,
        loaded_s1p=args.loaded_s1p,
        openems_port_metrics=args.openems_port_metrics,
        freq_hz=args.freq,
        band_lo_hz=args.band_lo,
        band_hi_hz=args.band_hi,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

