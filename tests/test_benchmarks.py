"""Tests for literature benchmark harness."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.benchmarks import (  # noqa: E402
    benchmark_dir,
    check_goldbaum_heating,
    check_literature_dielectric,
    check_phantom_saline,
    check_solver_internal,
    check_stress_qualitative,
    load_benchmark,
    run_benchmarks,
)


def test_benchmark_files_exist():
    for name in (
        "literature_dielectric",
        "goldbaum_heating_classes",
        "phantom_saline_gabriel",
        "stress_qualitative",
        "solver_internal",
    ):
        assert (benchmark_dir() / f"{name}.json").is_file()


def test_literature_dielectric_passes():
    results = check_literature_dielectric()
    assert results
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_goldbaum_heating_passes():
    results = check_goldbaum_heating()
    assert all(r.passed for r in results), [r for r in results if not r.passed]


def test_phantom_and_stress_pass():
    assert all(r.passed for r in check_phantom_saline())
    assert all(r.passed for r in check_stress_qualitative())


def test_solver_internal_passes():
    assert all(r.passed for r in check_solver_internal())


def test_run_all_benchmarks():
    report = run_benchmarks()
    assert report.passed
    assert len(report.results) >= 20


def test_load_benchmark_has_entries():
    data = load_benchmark("literature_dielectric")
    assert len(data["entries"]) >= 5
