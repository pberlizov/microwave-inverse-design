"""Print cited mineral catalog @ 2.45 GHz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.mineral_catalog import CATALOG, microwave_class  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Literature mineral ε catalog")
    ap.add_argument("--pairs", action="store_true", help="Also print MaterialPair loss contrast")
    args = ap.parse_args()

    print("=== Mineral catalog @ 2.45 GHz, 298 K ===")
    for key in sorted(CATALOG):
        e = CATALOG[key]
        eps = e.eps()
        cls = microwave_class(key).value
        print(f"  {key:14s}  {cls:8s}  ε={eps.real:5.2f}-j{eps.imag:.3f}  — {e.citation[:60]}")

    if args.pairs:
        print("\n=== Material pairs (target/gangue ε″ contrast) ===")
        for label in sorted(PAIRS):
            p = PAIRS[label]
            t_key, g_key = label.split("_in_")
            # label uses first mineral only — show actual pair eps
            tc = p.target.imag / max(p.gangue.imag, 1e-12)
            print(f"  {label:28s}  target ε={p.target.real:.1f}-j{p.target.imag:.3f}  "
                  f"gangue ε={p.gangue.real:.1f}-j{p.gangue.imag:.4f}  ε″ ratio={tc:.0f}x")


if __name__ == "__main__":
    main()
