"""Single forward solve: build a default applicator + ore charge, report selectivity.

Run:
  python scripts/run_demo.py
Or (installed):
  mw-inv-demo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.demo import main  # noqa: E402

if __name__ == "__main__":
    main()
