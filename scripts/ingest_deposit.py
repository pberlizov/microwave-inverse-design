"""Validate a named deposit: ore JSON + measured ε(f,T,moisture) library.

Example:
  python scripts/ingest_deposit.py data/ores/disseminated_pyrite_porphyry_measured_example.json
  python scripts/ingest_deposit.py data/ores/disseminated_pyrite_porphyry_measured_example.json \\
      --temp 373 --freq 2.45e9 --moisture 1.2 --out data/deposit_ingest_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.ore_profiles import (  # noqa: E402
    cavity_params_from_ore,
    load_ore_profile,
    materials_from_ore,
    ore_summary,
    resolve_measured_dielectrics_path,
)
from mw_inv.measured_dielectrics import load_measured_dielectrics, validate_library  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate deposit ore + measured dielectric curves")
    ap.add_argument("ore_json", type=str, help="ore profile JSON under data/ores/")
    ap.add_argument("--target-t", type=float, default=298.0, help="target phase temperature [K]")
    ap.add_argument("--gangue-t", type=float, default=298.0, help="gangue phase temperature [K]")
    ap.add_argument("--freq", type=float, default=2.45e9, help="evaluation frequency [Hz]")
    ap.add_argument("--moisture", type=float, default=None, help="moisture wt%% (overrides ore JSON default)")
    ap.add_argument("--out", default=None, help="optional JSON report path")
    args = ap.parse_args()

    ore_path = Path(args.ore_json)
    ore = load_ore_profile(ore_path)
    kw = dict(
        ore_profile_path=ore_path,
        target_T_K=args.target_t,
        gangue_T_K=args.gangue_t,
        freq_hz=args.freq,
        moisture_wt_percent=args.moisture,
    )
    summary = ore_summary(ore, **kw)
    mats = materials_from_ore(ore, **kw)
    params = cavity_params_from_ore(ore)

    measured_block = ore.measured_dielectrics or {}
    library_report = None
    if measured_block.get("path"):
        mp = resolve_measured_dielectrics_path(ore_path, measured_block)
        lib = load_measured_dielectrics(mp)
        library_report = {
            "path": str(mp),
            "validation_issues": validate_library(lib),
            **lib.summary(),
        }

    report = {
        **summary,
        "ore_json": str(ore_path.resolve()),
        "eval_conditions": {
            "target_T_K": args.target_t,
            "gangue_T_K": args.gangue_t,
            "freq_hz": args.freq,
            "moisture_wt_percent": args.moisture,
        },
        "materials": {
            "pair_label": mats.pair_label,
            "target_eps": [mats.target.real, mats.target.imag],
            "gangue_eps": [mats.gangue.real, mats.gangue.imag],
        },
        "geometry": {
            "inclusion_radius_frac": params.inclusion_radius_frac,
            "grain_count": len(params.inclusion_offsets_frac),
        },
        "measured_library": library_report,
    }

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))

    print(f"=== Deposit ingest ({ore.label}) ===")
    print(f"  materials mode   : {summary['materials_mode']}")
    if summary.get("materials_mode") == "measured":
        md = summary.get("measured_dielectrics") or {}
        ds = (md.get("dataset") or {})
        print(f"  dataset          : {ds.get('dataset_id') or md.get('path')}")
        print(f"  eval ε target    : {mats.target.real:.3f}-j{mats.target.imag:.3f}")
        print(f"  eval ε gangue    : {mats.gangue.real:.3f}-j{mats.gangue.imag:.4f}")
    else:
        print(f"  suggested pair   : {summary['suggested_pair']}")
        print(f"  ε″ target/gangue : {mats.target.imag:.3f} / {mats.gangue.imag:.4f}")
    if library_report and library_report.get("validation_issues"):
        print("  VALIDATION ISSUES:")
        for msg in library_report["validation_issues"]:
            print(f"    - {msg}")
        sys.exit(2)
    if args.out:
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
