"""Tests for openEMS port-truth ingest (backlog A1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.geometry import CavityParams, Materials  # noqa: E402
from mw_inv.openems_postprocess import (  # noqa: E402
    ingest_openems_case,
    load_port_metrics,
    resolve_openems_case_paths,
)
from mw_inv.solver_triangulation import SolverRow, triangulate_case  # noqa: E402
from mw_inv.validation_gate import GateThresholds, diagnose_openems_failure, evaluate_gate  # noqa: E402


def test_load_port_metrics(tmp_path: Path) -> None:
    path = tmp_path / "port_metrics.json"
    path.write_text(json.dumps({
        "s11_mag": 0.42,
        "coupling_eff": 0.8236,
        "selectivity": 0.61,
        "freq_hz": 2.45e9,
    }))
    m = load_port_metrics(path)
    assert abs(m.s11_mag - 0.42) < 1e-9
    assert abs(m.coupling_eff - (1 - 0.42**2)) < 0.01


def test_ingest_openems_case_port_only(tmp_path: Path) -> None:
    case_dir = tmp_path / "untuned"
    case_dir.mkdir()
    (case_dir / "port_metrics.json").write_text(json.dumps({
        "s11_mag": 0.35,
        "coupling_eff": 0.8775,
        "selectivity": 0.55,
        "freq_hz": 2.45e9,
    }))
    field, port = resolve_openems_case_paths(case_dir)
    assert field is None
    assert port is not None
    mats = Materials.from_pair("pyrite_in_calcite")
    metrics = ingest_openems_case(case_dir, CavityParams(), mats)
    assert metrics.s11_mag == 0.35
    assert metrics.selectivity == 0.55


def test_triangulate_case_reads_port_metrics(tmp_path: Path) -> None:
    from mw_inv.design_export import DesignCase

    case_dir = tmp_path / "tpe_best"
    case_dir.mkdir()
    (case_dir / "port_metrics.json").write_text(json.dumps({
        "s11_mag": 0.25,
        "coupling_eff": 0.9375,
        "selectivity": 0.68,
        "freq_hz": 2.45e9,
    }))
    from mw_inv.fdfd import Grid

    grid = Grid(nx=41, ny=41, Lx=0.36, Ly=0.36)
    mats = Materials.from_pair("pyrite_in_calcite")
    case = DesignCase("tpe_best", CavityParams(), "test")
    row = triangulate_case(case, grid, mats, openems_case_dir=case_dir)
    assert row.openems_s11_mag == 0.25
    assert row.openems_coupling_eff == 0.9375
    assert row.openems_selectivity == 0.68


def test_gate_openems_coupling_diagnosis() -> None:
    rows = [
        SolverRow("untuned", 0.50, openems_selectivity=0.48, openems_coupling_eff=0.40, rel_err_openems=0.04),
        SolverRow("tpe_best", 0.62, openems_selectivity=0.45, openems_coupling_eff=0.05, rel_err_openems=0.27),
    ]
    gate = evaluate_gate(rows, GateThresholds(min_fdfd_improvement=0.0, openems_coupling_floor=0.08))
    assert gate.openems_diagnosis == "coupling_collapse_on_optimised"
    assert any(c.name == "openems_coupling_floor" and not c.passed for c in gate.checks)


def test_gate_rank_mismatch_without_coupling_collapse() -> None:
    rows = [
        SolverRow("untuned", 0.50, openems_selectivity=0.48, openems_coupling_eff=0.35, rel_err_openems=0.04),
        SolverRow("tpe_best", 0.62, openems_selectivity=0.55, openems_coupling_eff=0.30, rel_err_openems=0.11),
    ]
    rank = {"openems_selectivity_rankings_match_fdfd": False}
    assert diagnose_openems_failure(rows, rank) == "ranking_mismatch_acceptable_coupling"
