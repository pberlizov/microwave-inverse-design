"""End-to-end Tier-1 pipeline: benchmarks → search → gate → export → manifest.

    python scripts/run_pipeline.py --materials pyrite_in_calcite --trials 24
    python scripts/run_pipeline.py --trials 8 --grid 41 --skip-export   # CI smoke

Or (installed):
    mw-inv-pipeline --materials pyrite_in_calcite --trials 24
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
