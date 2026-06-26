"""Wrapper for ``mw_inv.cli.update_run_with_openems`` (kept for backwards compatibility)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.cli.update_run_with_openems import main  # noqa: E402


if __name__ == "__main__":
    main()
