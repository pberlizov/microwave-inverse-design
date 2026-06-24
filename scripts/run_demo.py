"""Single forward solve: build a default applicator + ore charge, report selectivity.

Run:  python scripts/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402

from mw_inv.fdfd import Grid, solve  # noqa: E402
from mw_inv.fom import evaluate  # noqa: E402
from mw_inv.geometry import CavityParams, Materials, build_scene  # noqa: E402
from mw_inv.materials import DEFAULT_PAIR, PAIRS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default=DEFAULT_PAIR.label)
    args = ap.parse_args()

    # ~3 wavelengths across at 2.45 GHz (lambda0 ~= 12.2 cm).
    grid = Grid(nx=121, ny=121, Lx=0.36, Ly=0.36)
    params = CavityParams()
    materials = Materials.from_pair(args.materials)
    scene = build_scene(grid, params, materials)
    result = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy)
    report = evaluate(result, scene)

    print(f"=== Default applicator (no tuning), materials={args.materials} ===")
    print(f"  provenance       : {PAIRS[args.materials].provenance}")
    print(f"  frequency        : {scene.freq_hz/1e9:.3f} GHz")
    print(f"  target pixels    : {int(scene.target_mask.sum())}")
    print(f"  gangue pixels    : {int(scene.gangue_mask.sum())}")
    print(f"  selectivity      : {report.selectivity:.4f}   (P_target / P_charge)")
    print(f"  contrast         : {report.contrast:.3f}   (mean p_target / p_gangue)")
    print(f"  P_target (arb)   : {report.p_target:.3e}")
    print(f"  P_gangue (arb)   : {report.p_gangue:.3e}")


if __name__ == "__main__":
    main()
