"""Generate runnable openEMS model — delegates to ``openems_export``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.openems_export import write_openems_model  # noqa: E402


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/openems_cavity.m")
    ap.add_argument("--Lz", type=float, default=0.36)
    args = ap.parse_args()
    path = write_openems_model(args.out, Lz=args.Lz)
    func = "mw_inv_" + path.stem.replace("-", "_").replace(".", "_")
    print(f"Wrote runnable openEMS model: {path}")
    print(f"  Run in Octave with openEMS:  selectivity = {func}();")


if __name__ == "__main__":
    main()
