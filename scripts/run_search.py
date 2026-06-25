"""Thin-slice experiment: can geometry move selective-absorption contrast?

Optimise **manufacturable** applicator knobs (wall feed, coax stub, movable plate,
bed position, frequency) to maximise absorbed power in the target mineral phase.
Use ``--legacy`` for the old abstract baffle parametrization.

Run:  python scripts/run_search.py --trials 60

Or (installed):
  mw-inv-search --trials 60
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.search import main  # noqa: E402


if __name__ == "__main__":
    main()
