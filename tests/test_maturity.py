"""Tests for maturity labelling."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.maturity import Maturity, get, status_dict, warn_if_below  # noqa: E402


def test_experimental_components_exist():
    assert get("openems_port").maturity == Maturity.EXPERIMENTAL
    assert get("phantom_lab").maturity == Maturity.EXPERIMENTAL
    assert get("meep_3d_primitive").maturity == Maturity.EXPERIMENTAL
    assert get("meep_3d_extrusion").maturity == Maturity.WIP
    assert get("fdfd_2d").maturity == Maturity.CORE


def test_warn_if_below_emits_for_wip():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn_if_below("meep_3d_extrusion", minimum=Maturity.CORE)
    assert any("WIP" in str(x.message) for x in w)


def test_status_dict_serialisable():
    d = status_dict("phantom_lab")
    assert d["maturity"] == "experimental"
    assert isinstance(d["gaps"], list)
