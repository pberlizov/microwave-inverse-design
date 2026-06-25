"""Evaluate one cavity design under a named preset (unified FOM report).

    python scripts/run_design_eval.py --preset em
    python scripts/run_design_eval.py --preset composite:liberation --materials pyrite_in_calcite
    python scripts/run_design_eval.py --params data/pyrite_search_summary.json --key tpe_search
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_evaluator import (  # noqa: E402
    COMPOSITE_PRESETS,
    DesignEvaluator,
    best_design,
    optuna_design_search,
    preset_config,
)
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.materials import DEFAULT_PAIR, PAIRS  # noqa: E402
from mw_inv.search import params_from_dict  # noqa: E402

PRESET_CHOICES = (
    "em",
    "thermal:delta_T",
    "thermal:heat_selectivity",
    "thermal:em_selectivity",
    "stress:stress_score",
    "stress:stress_selectivity",
    "stress:mean_interface_stress",
    *COMPOSITE_PRESETS,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified design evaluation")
    ap.add_argument("--preset", choices=PRESET_CHOICES, default="em")
    ap.add_argument("--materials", choices=sorted(PAIRS), default=DEFAULT_PAIR.label)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--params", default=None, help="JSON file with best_params")
    ap.add_argument("--key", default="tpe_search", help="key under JSON for best_params")
    ap.add_argument("--trials", type=int, default=0, help="if >0, run TPE search")
    ap.add_argument("--seed", type=int, default=1903)
    ap.add_argument("--out", default="data/design_eval_report.json")
    ap.add_argument("--check-arcing", action="store_true")
    args = ap.parse_args()

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair(args.materials)
    cfg = preset_config(
        args.preset,
        materials=materials,
        pair_label=args.materials,
        check_arcing=args.check_arcing,
    )
    ev = DesignEvaluator(grid, cfg, preset=args.preset)

    if args.trials > 0:
        reports = optuna_design_search(grid, cfg, args.trials, args.seed, preset=args.preset)
        rep = best_design(reports)
        payload = {
            "preset": args.preset,
            "materials": args.materials,
            "n_trials": args.trials,
            "best": rep.to_dict(),
        }
    else:
        if args.params:
            data = json.loads(Path(args.params).read_text())
            block = data.get(args.key, data)
            pdict = block.get("best_params", block)
            rep = ev.evaluate(params_from_dict(pdict))
        else:
            rep = ev.evaluate(CavityParams())
        payload = {
            "preset": args.preset,
            "materials": args.materials,
            "report": rep.to_dict(),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"=== Design eval ({args.preset}) ===")
    print(f"  score           : {rep.score:.4f} ({rep.objective_key})")
    print(f"  EM selectivity  : {rep.em_selectivity:.4f}")
    if rep.delta_T_K is not None:
        print(f"  ΔT              : {rep.delta_T_K:.1f} K  heat_sel={rep.heat_selectivity:.3f}")
    if rep.stress_score is not None:
        print(f"  stress score    : {rep.stress_score:.2e} Pa")
    if rep.arcing_risk is not None:
        print(f"  arcing risk     : {rep.arcing_risk}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
