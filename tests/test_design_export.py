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
from mw_inv.geometry import Materials  # noqa: E402
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


def test_load_search_cases_top_k(tmp_path):
    data = {
        "tpe_top_k": [
            {"selectivity": 0.7, "params": {"feed_along_frac": 0.4, "feed_wall": "bottom"}},
            {"selectivity": 0.65, "params": {"feed_along_frac": 0.5, "feed_wall": "left"}},
        ],
        "tpe_search": {"best_params": {"feed_along_frac": 0.4, "feed_wall": "bottom"}},
    }
    path = tmp_path / "search.json"
    path.write_text(json.dumps(data))
    cases = load_search_cases(path, top_k=2)
    assert [c.label for c in cases] == ["untuned", "tpe_k1", "tpe_k2"]


def test_top_k_trials_dedupes():
    from mw_inv.search import Trial, top_k_trials

    trials = [
        Trial({"a": 1}, 0.5, 1.0, 1.0),
        Trial({"a": 1}, 0.6, 1.0, 1.0),
        Trial({"a": 2}, 0.55, 1.0, 1.0),
    ]
    picked = top_k_trials(trials, 3)
    assert len(picked) == 2
    assert picked[0].selectivity == 0.6


def test_export_design_bundle(tmp_path):
    case = cases_from_search_summary({"tpe_search": {"best_params": {"feed_wall": "bottom"}}})[0]
    mats = Materials.from_pair("pyrite_in_calcite")
    bundle = export_design_bundle(tmp_path, case, mats, grid_n=41)
    assert bundle.openems_path.is_file()
    assert bundle.manifest_path.is_file()
    manifest = json.loads(bundle.manifest_path.read_text())
    assert "fdfd_selectivity" in manifest
    assert manifest["openems_function"].startswith("mw_inv_")
    # Exported openEMS script writes dumps to a per-case directory for triangulation ingestion.
    text = bundle.openems_path.read_text()
    assert "openems_runs" in text
    assert "port_metrics.json" in text
    assert "calcPort" in text
