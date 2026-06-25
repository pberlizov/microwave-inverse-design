"""Tests for design export bundles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.design_export import (  # noqa: E402
    export_design_bundle,
    load_search_cases,
    cases_from_search_summary,
)
from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.search import params_from_dict  # noqa: E402


def test_params_from_dict_roundtrip():
    p = params_from_dict({"feed_along_frac": 0.3, "feed_wall": "left", "freq_hz": 2.44e9})
    assert p.feed_wall == "left"
    assert abs(p.feed_along_frac - 0.3) < 1e-9


def test_load_search_cases(tmp_path):
    data = {
        "random_search": {"best_params": {"feed_along_frac": 0.4, "feed_wall": "bottom"}},
        "tpe_search": {"best_params": {"feed_along_frac": 0.6, "feed_wall": "right"}},
    }
    path = tmp_path / "search.json"
    path.write_text(json.dumps(data))
    cases = load_search_cases(path)
    labels = [c.label for c in cases]
    assert labels == ["untuned", "random_best", "tpe_best"]


def test_export_design_bundle(tmp_path):
    case = cases_from_search_summary({"tpe_search": {"best_params": {"feed_wall": "bottom"}}})[0]
    mats = Materials.from_pair("pyrite_in_calcite")
    bundle = export_design_bundle(tmp_path, case, mats, grid_n=41)
    assert bundle.openems_path.is_file()
    assert bundle.manifest_path.is_file()
    manifest = json.loads(bundle.manifest_path.read_text())
    assert "fdfd_selectivity" in manifest
    assert manifest["openems_function"].startswith("mw_inv_")
