"""Run public-literature benchmark suite.

    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --tier dielectric heating_class
    python scripts/run_benchmarks.py --out data/benchmark_report.json

Or (installed):
    mw-inv-benchmarks --tier dielectric
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.benchmarks import main  # noqa: E402


if __name__ == "__main__":
    main()
