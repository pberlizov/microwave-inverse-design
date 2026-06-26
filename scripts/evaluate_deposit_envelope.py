"""Evaluate a cavity design over an ore envelope directory (backlog D3).

    python scripts/evaluate_deposit_envelope.py data/ores/forster
    python scripts/evaluate_deposit_envelope.py data/ores --min-selectivity 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.deposit_envelope import discover_ore_json_paths, evaluate_deposit_envelope  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Min/mean FOM over many ore JSON profiles")
    ap.add_argument("ore_root", type=str, help="directory containing ore profile JSON files")
    ap.add_argument("--min-selectivity", type=float, default=0.0)
    ap.add_argument("--min-coupling", type=float, default=0.0)
    ap.add_argument("--max-gangue-frac", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="data/deposit_envelope_report.json")
    args = ap.parse_args()

    root = Path(args.ore_root)
    paths = discover_ore_json_paths(root)
    if not paths:
        print(f"No ore JSON under {root}")
        sys.exit(2)

    grid = Grid(nx=51, ny=51, Lx=0.36, Ly=0.36)
    report = evaluate_deposit_envelope(paths, grid, CavityParams())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2) + "\n")

    ok = report.passes(
        min_selectivity=args.min_selectivity,
        min_coupling_eff=args.min_coupling,
        max_gangue_power_fraction=args.max_gangue_frac,
    )
    print(f"=== Deposit envelope ({root}) ===")
    print(f"  ores evaluated : {report.n_ok}/{report.n_ores}")
    print(f"  min selectivity: {report.min_selectivity:.4f}")
    print(f"  mean selectivity: {report.mean_selectivity:.4f}")
    print(f"  min coupling   : {report.min_coupling_eff:.4f}")
    print(f"  max gangue frac: {report.max_gangue_power_fraction:.4f}")
    print(f"  gate           : {'PASS' if ok else 'FAIL'}")
    print(f"  wrote {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
