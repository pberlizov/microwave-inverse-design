"""Regression tests for openEMS export scripts (port placement matters)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.openems_export import generate_openems_script  # noqa: E402


def test_openems_wall_port_tracks_feed_position() -> None:
    mats = Materials.from_pair("pyrite_in_calcite")
    p1 = CavityParams(feed_wall="bottom", feed_along_frac=0.25, stub_depth_frac=0.10, stub_width_frac=0.04)
    s1 = generate_openems_script(p1, mats, port_mode="wall_lumped")
    assert "Wall lumped port" in s1
    # x-range should reflect feed_along_frac=0.25 (≈ -97.2 mm to -82.8 mm in centred coords)
    assert "-97.2000" in s1
    assert "-82.8000" in s1

    p2 = CavityParams(feed_wall="bottom", feed_along_frac=0.75, stub_depth_frac=0.10, stub_width_frac=0.04)
    s2 = generate_openems_script(p2, mats, port_mode="wall_lumped")
    # x-range should shift with feed_along_frac (≈ +82.8 mm to +97.2 mm)
    assert "82.8000" in s2
    assert "97.2000" in s2


def test_openems_port_mode_alias_kept() -> None:
    mats = Materials.from_pair("pyrite_in_calcite")
    p = CavityParams()
    s = generate_openems_script(p, mats, port_mode="coax_gap")
    assert "Top-face coax gap port" in s

