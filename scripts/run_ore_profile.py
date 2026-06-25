"""Ore profile heating-class prediction and arcing-risk screening.

    python scripts/run_ore_profile.py disseminated_pyrite_porphyry
    python scripts/run_ore_profile.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_evaluator import DesignEvaluator  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams  # noqa: E402
from mw_inv.ore_profiles import ORE_PROFILES, charge_volume_m3, load_ore_profile, materials_from_ore, ore_summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ore", nargs="?", default=None, choices=sorted(ORE_PROFILES))
    ap.add_argument("--json", type=str, default=None, help="ore profile JSON path (overrides positional ore)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="data/ore_profile_report.json")
    args = ap.parse_args()

    if args.json:
        ore = load_ore_profile(args.json)
        labels = [ore.label]
        ore_sources = {ore.label: ore}
    else:
        labels = sorted(ORE_PROFILES) if args.all or args.ore is None else [args.ore]
        ore_sources = {k: ORE_PROFILES[k] for k in labels}

    grid = Grid(nx=61, ny=61, Lx=0.36, Ly=0.36)
    rows = []
    for label in labels:
        ore = ore_sources[label]
        summary = ore_summary(ore)
        mats = materials_from_ore(ore)
        ev = DesignEvaluator.from_preset(
            grid,
            "em",
            materials=mats,
            check_arcing=True,
        )
        rep = ev.evaluate(CavityParams())
        vol = charge_volume_m3(CavityParams())
        rows.append({
            "ore": label,
            "suggested_pair": summary["suggested_pair"],
            "inferred_gangue": summary["inferred_gangue"],
            "dominant_hmap": summary["dominant_hmap"],
            "hmap_wt_percent": ore.hmap_wt_percent,
            "heating_class": ore.heating_class(),
            "predicted_rate_C_per_min": ore.predicted_heating_rate_C_per_min(),
            "fdfd_selectivity": rep.em_selectivity,
            "em_contrast": rep.em_contrast,
            "p_total": rep.p_total,
            "charge_volume_m3": vol,
            "arcing_risk": rep.arcing_risk,
            "power_density_W_m3": rep.power_density_W_m3,
            "loss_tangent": rep.loss_tangent,
            "evaluator_score": rep.score,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"profiles": rows}, indent=2))
    for r in rows:
        print(f"  {r['ore']:30s}  HMAP={r['hmap_wt_percent']:5.1f}%  class={r['heating_class']:14s}  "
              f"pair={r['suggested_pair']:22s}  "
              f"rate~{r['predicted_rate_C_per_min']:.0f} C/min  sel={r['fdfd_selectivity']:.3f}  "
              f"arcing={r['arcing_risk']}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
