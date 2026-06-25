"""Export optimised designs to openEMS + FDFD bundles for lab / FDTD cross-check.

    python scripts/export_design.py --search data/search_summary.json
    python scripts/export_design.py --phantom saline_2_vs_0.5 --trials 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_export import (  # noqa: E402
    cases_from_phantom_prediction,
    export_all_cases,
    load_search_cases,
)
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.maturity import status_dict  # noqa: E402
from mw_inv.phantom import predict_lab_outcome  # noqa: E402
from mw_inv.promotion import PromotionError, PromotionTier, assert_tier_at_least  # noqa: E402
from mw_inv.run_manifest import RunManifest, finalize_promotion  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", type=str, default=None, help="search_summary.json path")
    ap.add_argument("--phantom", type=str, default=None, help="phantom recipe label")
    ap.add_argument("--materials", choices=sorted(PAIRS), default="pyrite_in_calcite")
    ap.add_argument("--trials", type=int, default=12, help="opt trials when --phantom")
    ap.add_argument("--grid", type=int, default=81)
    ap.add_argument("--out-dir", default="data/design_exports")
    ap.add_argument("--Lz", type=float, default=0.36)
    ap.add_argument(
        "--manifest",
        default=None,
        help="run manifest.json — export allowed only if promotion tier meets --require-tier",
    )
    ap.add_argument(
        "--require-tier",
        default="fdfd_optimised",
        choices=[t.value for t in PromotionTier if t != PromotionTier.UNRANKED],
        help="minimum promotion tier for export (ignored with --allow-unranked)",
    )
    ap.add_argument(
        "--allow-unranked",
        action="store_true",
        help="skip promotion tier check (dev only)",
    )
    args = ap.parse_args()

    manifest: RunManifest | None = None
    if args.manifest and not args.allow_unranked:
        manifest = RunManifest.load(args.manifest)
        if not manifest.promotion:
            finalize_promotion(manifest)
        tier = PromotionTier(manifest.promotion["tier"])
        required = PromotionTier(args.require_tier)
        try:
            assert_tier_at_least(tier, required, action="export design bundle")
        except PromotionError as exc:
            print(f"EXPORT BLOCKED: {exc}")
            sys.exit(2)

    if not args.search and not args.phantom:
        args.search = "data/search_summary.json"

    out_dir = Path(args.out_dir)
    if args.search:
        cases = load_search_cases(args.search)
        materials = Materials.from_pair(
            json.loads(Path(args.search).read_text()).get("materials", args.materials)
        )
        tag = Path(args.search).stem
        out_dir = out_dir / tag
    else:
        grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
        pred = predict_lab_outcome(args.phantom, grid, n_opt_trials=args.trials, seed=7701)
        cases = cases_from_phantom_prediction(pred.to_dict())
        from mw_inv.phantom import materials_from_phantom

        materials = materials_from_phantom(args.phantom)
        tag = args.phantom
        out_dir = out_dir / tag

    bundles = export_all_cases(out_dir, cases, materials, grid_n=args.grid, Lz=args.Lz)
    summary = {
        "maturity": {
            "openems_port": status_dict("openems_port"),
            "meep_3d_primitive": status_dict("meep_3d_primitive"),
        },
        "promotion_tier": manifest.promotion.get("tier") if manifest else None,
        "tag": tag,
        "exports": [
            {
                "label": b.label,
                "fdfd_selectivity": b.fdfd_selectivity,
                "openems_model": str(b.openems_path),
                "manifest": str(b.manifest_path),
            }
            for b in bundles
        ],
    }
    summary_path = out_dir / "export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"=== Design export ({tag}) ===")
    for b in bundles:
        print(f"  {b.label:12s}  FDFD sel={b.fdfd_selectivity:.4f}  -> {b.openems_path.name}")
    print(f"  wrote {summary_path}")


if __name__ == "__main__":
    main()
