"""Canonical FDFD Dirichlet plate vs openEMS AddMetal alignment check (B0/B2).

    python3 scripts/run_metal_model_check.py
    python3 scripts/run_metal_model_check.py --synthesize-openems --out data/metal_model_report.json

With Octave + openEMS installed:
    python3 scripts/run_metal_model_check.py --run-openems
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_export import DesignCase, export_design_bundle  # noqa: E402
from mw_inv.fdfd import Grid  # noqa: E402
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.metal_model import evaluate_metal_model_alignment, write_metal_model_report  # noqa: E402
from mw_inv.openems_runner import (  # noqa: E402
    octave_available,
    run_openems_exports,
    synthesize_port_dumps,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="FDFD/openEMS metal model alignment")
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--plate-len", type=float, default=0.30, help="plate_len_frac")
    ap.add_argument("--structure-model", choices=("dirichlet", "lossy_imag"), default="dirichlet")
    ap.add_argument("--out", default="data/metal_model_report.json")
    ap.add_argument("--export-dir", default="data/metal_model_export")
    ap.add_argument("--run-openems", action="store_true")
    ap.add_argument("--synthesize-openems", action="store_true")
    ap.add_argument("--openems-octave", default="octave")
    args = ap.parse_args()

    if args.run_openems and args.synthesize_openems:
        ap.error("use either --run-openems or --synthesize-openems")

    grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
    materials = Materials.from_pair("pyrite_in_calcite")
    params = replace(
        CavityParams(),
        plate_len_frac=args.plate_len,
        plate_angle_deg=45.0,
        structure_model=args.structure_model,
    )

    case_dir: Path | None = None
    if args.run_openems or args.synthesize_openems:
        export_dir = Path(args.export_dir)
        bundle = export_design_bundle(export_dir, DesignCase("plate_check", params, "metal_model"), materials)
        if args.run_openems:
            if not octave_available(args.openems_octave):
                raise SystemExit(f"Octave not found ({args.openems_octave!r})")
            result = run_openems_exports(export_dir, octave_cmd=args.openems_octave)
            if result.returncode != 0:
                raise SystemExit(f"openEMS failed (rc={result.returncode})")
            case_dir = result.dump_dir / bundle.label
        else:
            dump_root = synthesize_port_dumps(export_dir, [bundle])
            case_dir = dump_root / bundle.label

    report = evaluate_metal_model_alignment(grid, params, materials, openems_case_dir=case_dir)
    out = write_metal_model_report(args.out, report)

    print("=== Metal model alignment ===")
    print(f"  structure_model     : {report.structure_model}")
    print(f"  FDFD coupling_eff   : {report.fdfd_coupling_eff:.4f}")
    print(f"  FDFD pec_loss_frac  : {report.fdfd_pec_loss_fraction:.4f}")
    if report.openems_coupling_eff is not None:
        print(f"  openEMS coupling_eff: {report.openems_coupling_eff:.4f}")
        print(f"  coupling ratio      : {report.coupling_ratio:.4f}")
        print(f"  ratio gate          : {'PASS' if report.gate_passed else 'FAIL'}")
    print(f"  -> {out}")

    if report.structure_model == "dirichlet" and report.fdfd_pec_loss_fraction > 0.10:
        raise SystemExit(2)
    if report.gate_passed is False:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
