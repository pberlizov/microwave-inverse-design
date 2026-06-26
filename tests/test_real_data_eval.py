"""Real-data catalog discovery and batch evaluation smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.real_data_eval import discover_real_data_catalog, evaluate_real_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "data"


def test_catalog_discovers_ores_and_deposits() -> None:
    cat = discover_real_data_catalog(ROOT)
    kinds = {s.kind for s in cat.sources}
    assert "ore_profile" in kinds
    assert "deposit_eps" in kinds
    assert "benchmark" in kinds
    assert "material_pair" in kinds
    assert any(s.label == "disseminated_pyrite_porphyry" for s in cat.sources)


def test_quick_real_data_eval_runs() -> None:
    report = evaluate_real_data(ROOT, quick=True)
    assert report["summary"]["n_ores"] >= 3
    assert report["summary"]["n_deposit_points"] >= 5
    assert report["benchmarks"]["passed"] is True
    assert len(report["material_pairs"]) >= 5
    assert len(report["phantoms"]) >= 2
