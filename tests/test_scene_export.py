"""Tests for scene_export primitives."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.scene_export import build_primitives, eps_to_kappa  # noqa: E402


def test_build_primitives_has_gangue_and_targets():
    params = CavityParams(plate_len_frac=0.12)
    prims = build_primitives(params, Materials(), Lx=0.36, Ly=0.36, Lz=0.36)
    tags = {b.tag for b in prims.boxes}
    assert "gangue" in tags
    assert "pec" in tags
    assert len(prims.cylinders) >= 1


def test_eps_to_kappa_nonnegative():
    er, k = eps_to_kappa(5.0 + 0.2j, 2.45e9)
    assert er == 5.0
    assert k >= 0.0
