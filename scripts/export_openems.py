"""Export runnable openEMS model + FDFD scene arrays.

    python scripts/export_openems.py --materials pyrite_in_calcite
    python scripts/export_openems.py --phantom saline_2_vs_0.5

Requires openEMS at run time (Octave):  mw_inv_openems_cavity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.materials import PAIRS  # noqa: E402
from mw_inv.openems_export import export_scene_npz, write_openems_model  # noqa: E402
from mw_inv.phantom import materials_from_phantom  # noqa: E402
from mw_inv.phantom_data import PHANTOM_RECIPES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", choices=sorted(PAIRS), default=None)
    ap.add_argument("--phantom", choices=sorted(PHANTOM_RECIPES), default=None)
    ap.add_argument("--out-dir", default="data/openems")
    ap.add_argument("--Lz", type=float, default=0.36)
    args = ap.parse_args()

    if args.phantom:
        mats = materials_from_phantom(args.phantom)
        tag = args.phantom
    else:
        label = args.materials or "pyrite_in_calcite"
        mats = Materials.from_pair(label)
        tag = label

    out_dir = Path(args.out_dir)
    m_path = write_openems_model(out_dir / f"{tag}_cavity.m", CavityParams(), mats, Lz=args.Lz)
    npz_path = export_scene_npz(out_dir / f"{tag}_scene.npz", CavityParams(), mats)

    print(f"Exported openEMS model: {m_path}")
    print(f"Exported FDFD scene:    {npz_path}")
    print("Run in Octave (with openEMS installed):")
    print(f"  cd {out_dir.resolve()}")
    func = "mw_inv_" + m_path.stem.replace("-", "_").replace(".", "_")
    print(f"  selectivity = {func}();")


if __name__ == "__main__":
    main()
