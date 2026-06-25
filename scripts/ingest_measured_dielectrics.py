"""Validate and summarize a measured dielectric dataset for ore/gangue.

Example:
  python scripts/ingest_measured_dielectrics.py data/templates/measured_dielectrics.template.json
  python scripts/ingest_measured_dielectrics.py my_deposit_eps.json --phase ore_bulk --temp 298 --freq 2.45e9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.measured_dielectrics import load_measured_dielectrics, validate_library  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--phase", default=None, help="phase label to evaluate at a point")
    ap.add_argument("--temp", type=float, default=298.0, help="temperature [K]")
    ap.add_argument("--freq", type=float, default=2.45e9, help="frequency [Hz]")
    ap.add_argument("--moisture", type=float, default=None, help="moisture (wt%%, nearest selection)")
    args = ap.parse_args()

    lib = load_measured_dielectrics(args.path)
    issues = validate_library(lib)
    if issues:
        print("INVALID measured dielectrics:")
        for msg in issues[:20]:
            print(f"  - {msg}")
        sys.exit(2)

    print("=== Measured dielectrics ===")
    if lib.description:
        print(f"  description: {lib.description}")
    print(f"  phases: {', '.join(sorted(lib.phases))}")
    for label, phase in sorted(lib.phases.items()):
        temps = sorted({p.temp_K for p in phase.points})
        freqs = sorted({p.freq_hz for p in phase.points})
        moist = sorted({p.moisture_wt_percent for p in phase.points if p.moisture_wt_percent is not None})
        print(f"  - {label}: n={len(phase.points)}  T=[{temps[0]:.0f}..{temps[-1]:.0f}]K  f=[{freqs[0]/1e9:.3f}..{freqs[-1]/1e9:.3f}]GHz"
              + (f"  moisture={moist}" if moist else ""))

    if args.phase:
        eps = lib.eps(
            args.phase, temp_K=args.temp, freq_hz=args.freq, moisture_wt_percent=args.moisture,
        )
        print(f"  eval: phase={args.phase}  T={args.temp:.1f}K  f={args.freq/1e9:.3f}GHz  eps={eps.real:.4g}-j{eps.imag:.4g}")


if __name__ == "__main__":
    main()
