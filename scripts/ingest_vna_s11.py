"""Ingest VNA S11 Touchstone files into a compact JSON report.

Example:
  python scripts/ingest_vna_s11.py --unloaded data/vna/unloaded.s1p --loaded data/vna/loaded.s1p \\
    --out data/vna_s11_report.json --freq 2.45e9 --band-lo 2.40e9 --band-hi 2.50e9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.vna_s11 import load_s1p, summary_s11_metrics  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unloaded", required=True, help="Touchstone .s1p (empty cavity)")
    ap.add_argument("--loaded", default=None, help="Touchstone .s1p (with charge)")
    ap.add_argument("--out", default="data/vna_s11_report.json")
    ap.add_argument("--freq", type=float, default=2.45e9)
    ap.add_argument("--band-lo", type=float, default=None)
    ap.add_argument("--band-hi", type=float, default=None)
    args = ap.parse_args()

    unloaded = load_s1p(args.unloaded)
    payload = {
        "unloaded": summary_s11_metrics(
            unloaded,
            freq_hz=args.freq,
            band_lo_hz=args.band_lo,
            band_hi_hz=args.band_hi,
        ),
        "loaded": None,
    }
    if args.loaded:
        loaded = load_s1p(args.loaded)
        payload["loaded"] = summary_s11_metrics(
            loaded,
            freq_hz=args.freq,
            band_lo_hz=args.band_lo,
            band_hi_hz=args.band_hi,
        )
        payload["delta"] = {
            "s11_mag_loaded_minus_unloaded": float(payload["loaded"]["s11_mag"] - payload["unloaded"]["s11_mag"]),
            "s11_db_loaded_minus_unloaded": float(payload["loaded"]["s11_db"] - payload["unloaded"]["s11_db"]),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

