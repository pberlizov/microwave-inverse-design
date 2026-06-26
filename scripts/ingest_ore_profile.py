"""Ingest QEMSCAN / assay ore JSON and report heating class + suggested pair.

    python scripts/ingest_ore_profile.py data/ores/disseminated_pyrite_porphyry.json
    python scripts/ingest_ore_profile.py data/ores/massive_pyrite.json --out data/ore_ingest_report.json
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
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ore_json", type=str, help="ore profile JSON (see data/ores/)")
    ap.add_argument("--out", default="data/ore_ingest_report.json")
    args = ap.parse_args()

    ore = load_ore_profile(args.ore_json)
    summary = ore_summary(ore)
    mats = materials_from_ore(ore)
    params = cavity_params_from_ore(ore)
    report = {
        **summary,
        "materials_pair_label": mats.pair_label,
        "target_eps_imag": mats.target.imag,
        "gangue_eps_imag": mats.gangue.imag,
        "inclusion_radius_frac": params.inclusion_radius_frac,
        "grain_count": len(params.inclusion_offsets_frac),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"=== Ore ingest ({ore.label}) ===")
    print(f"  source           : {ore.source or 'unspecified'}")
    print(f"  HMAP wt%         : {summary['hmap_wt_percent']:.1f}%")
    print(f"  heating class    : {summary['heating_class']}")
    print(f"  dominant HMAP    : {summary['dominant_hmap']}")
    print(f"  inferred gangue  : {summary['inferred_gangue']}")
    print(f"  suggested pair   : {summary['suggested_pair']}")
    if summary.get("materials_mode") == "measured":
        md = summary.get("measured_dielectrics") or {}
        print(f"  materials mode   : measured ({md.get('path')})")
    else:
        print("  materials mode   : bruggeman (HMAP mix + gangue mineral)")
    print(f"  loss contrast    : {summary['loss_contrast']:.1f}")
    print(f"  ε″ target/gangue : {mats.target.imag:.3f} / {mats.gangue.imag:.4f}")
    print(f"  grain radius frac: {params.inclusion_radius_frac:.3f}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
