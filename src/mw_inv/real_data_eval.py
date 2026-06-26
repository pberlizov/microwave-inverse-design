"""Discover and evaluate all available real / versioned data in the workspace."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mw_inv.benchmarks import run_benchmarks
from mw_inv.design_evaluator import DesignEvaluator
from mw_inv.fdfd import Grid
from mw_inv.geometry import CavityParams, Materials
from mw_inv.materials import PAIRS
from mw_inv.measured_dielectrics import load_measured_dielectrics
from mw_inv.ore_profiles import (
    cavity_params_from_ore,
    load_ore_profile,
    materials_from_ore,
    ore_summary,
)
from mw_inv.external_datasets import ingest_status, load_datasets_catalog
from mw_inv.phantom_data import PHANTOM_RECIPES
from mw_inv.search import evaluate_params


def _repo_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _provenance_for(path: Path, *, example_suffixes: tuple[str, ...] = (".example.json", ".template.json")) -> str:
    name = path.name
    if any(name.endswith(s) for s in example_suffixes):
        return "versioned_example"
    if "template" in path.parts or name.endswith(".template.json"):
        return "template"
    return "versioned"


@dataclass(frozen=True)
class DataSource:
    kind: str
    path: str
    label: str
    provenance: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "label": self.label,
            "provenance": self.provenance,
        }


@dataclass
class RealDataCatalog:
    sources: list[DataSource] = field(default_factory=list)
    missing_user_inputs: list[str] = field(default_factory=list)
    external_datasets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_sources": len(self.sources),
            "sources": [s.to_dict() for s in self.sources],
            "missing_user_inputs": list(self.missing_user_inputs),
            "external_datasets": list(self.external_datasets),
        }


def discover_real_data_catalog(data_root: Path | None = None) -> RealDataCatalog:
    """Scan versioned inputs and optional local user measurement files."""
    root = data_root or _repo_data_root()
    catalog = RealDataCatalog()

    bench_dir = root / "benchmarks"
    if bench_dir.is_dir():
        for p in sorted(bench_dir.glob("*.json")):
            catalog.sources.append(DataSource("benchmark", str(p.resolve()), p.stem, _provenance_for(p)))

    ores_dir = root / "ores"
    if ores_dir.is_dir():
        for p in sorted(ores_dir.rglob("*.json")):
            if p.name.startswith("_"):
                continue
            catalog.sources.append(DataSource("ore_profile", str(p.resolve()), p.stem, _provenance_for(p)))

    from mw_inv.campaign import discover_campaign_files, load_campaign

    for camp_path in discover_campaign_files(root):
        try:
            camp = load_campaign(camp_path)
            catalog.sources.append(DataSource(
                "campaign",
                str(camp_path.resolve()),
                camp.campaign_id,
                "versioned",
            ))
        except (ValueError, json.JSONDecodeError, OSError):
            continue

    dep_dir = root / "measured_dielectrics"
    if dep_dir.is_dir():
        for p in sorted(dep_dir.glob("*.json")):
            if "template" in p.name:
                continue
            catalog.sources.append(DataSource("deposit_eps", str(p.resolve()), p.stem, _provenance_for(p)))

    for name, kind in (
        ("measured_eps.json", "bench_probe"),
        ("measured_eps.example.json", "bench_probe"),
        ("lab_measurements.json", "bench_lab"),
        ("lab_measurements.example.json", "bench_lab"),
    ):
        p = root / name
        if p.is_file():
            catalog.sources.append(DataSource(kind, str(p.resolve()), p.stem, _provenance_for(p)))

    for p in sorted(root.glob("*.s1p")):
        catalog.sources.append(DataSource("vna_touchstone", str(p.resolve()), p.stem, "user_local"))

    for label in sorted(PAIRS):
        catalog.sources.append(DataSource(
            "material_pair",
            f"mw_inv.materials.PAIRS[{label!r}]",
            label,
            "literature",
        ))

    for label in sorted(PHANTOM_RECIPES):
        catalog.sources.append(DataSource(
            "phantom_recipe",
            f"mw_inv.phantom_data.PHANTOM_RECIPES[{label!r}]",
            label,
            "literature",
        ))

    for hint in ("measured_eps.json", "lab_measurements.json"):
        if not (root / hint).is_file():
            catalog.missing_user_inputs.append(
                f"Copy data/{hint.replace('.json', '.example.json')} → data/{hint} for live bench data",
            )

    cat_path = root / "datasets_catalog.json"
    if cat_path.is_file():
        catalog.sources.append(DataSource(
            "datasets_catalog",
            str(cat_path.resolve()),
            "datasets_catalog",
            "versioned",
        ))
        ds_catalog = load_datasets_catalog(root)
        catalog.external_datasets = ingest_status(root)
        for entry in ds_catalog.entries:
            provenance = "external_online"
            if entry.ingest_output:
                ingested = (root / entry.ingest_output).is_file()
                kind = "literature_ingest" if ingested else "external_pending"
                if ingested:
                    p = root / entry.ingest_output
                    catalog.sources.append(DataSource(
                        kind,
                        str(p.resolve()),
                        entry.id,
                        "literature",
                    ))
            else:
                catalog.sources.append(DataSource(
                    "external_dataset",
                    entry.url or entry.id,
                    entry.id,
                    provenance,
                ))

    return catalog


def _eval_ore_row(
    ore_path: Path,
    grid: Grid,
    *,
    quick: bool,
) -> dict[str, Any]:
    ore = load_ore_profile(ore_path)
    params = cavity_params_from_ore(ore, cavity_span_m=grid.Lx)
    base_kw = dict(ore_profile_path=ore_path, target_T_K=298.0, gangue_T_K=298.0, freq_hz=2.45e9)
    summary = ore_summary(ore, **base_kw)
    mats = materials_from_ore(ore, **base_kw)
    ev = DesignEvaluator.from_preset(grid, "em", materials=mats, check_arcing=True)
    rep = ev.evaluate(params)
    row: dict[str, Any] = {
        "ore_path": str(ore_path),
        "label": ore.label,
        "materials_mode": summary.get("materials_mode"),
        "heating_class": summary.get("heating_class"),
        "hmap_wt_percent": summary.get("hmap_wt_percent"),
        "suggested_pair": summary.get("suggested_pair"),
        "fdfd_selectivity": rep.em_selectivity,
        "coupling_eff": rep.coupling_eff,
        "arcing_risk": rep.arcing_risk,
        "provenance": _provenance_for(ore_path),
    }
    md = ore.measured_dielectrics or {}
    if md.get("path") and not quick:
        from mw_inv.ensemble import evaluate_material_robust

        try:
            mrep = evaluate_material_robust(
                grid, params, ore, ore_profile_path=str(ore_path), n_scenarios=3, seed=11,
            )
            row["material_robust_min_selectivity"] = mrep.min_selectivity
            row["material_robust_mean_selectivity"] = mrep.mean_selectivity
        except (ValueError, FileNotFoundError, KeyError) as exc:
            row["material_robust_error"] = str(exc)
    return row


def _reference_gangue_label(lib) -> str | None:
    if "gangue" in lib.phases:
        return "gangue"
    best_label: str | None = None
    best_loss = float("inf")
    for label, phase in lib.phases.items():
        if not phase.points:
            continue
        try:
            eps = phase.eps(temp_K=298.15, freq_hz=2.45e9)
        except (ValueError, KeyError):
            continue
        if eps.imag < best_loss:
            best_loss = eps.imag
            best_label = label
    return best_label


def _eval_deposit_library(path: Path, grid: Grid) -> dict[str, Any]:
    lib = load_measured_dielectrics(path)
    summary = lib.summary()
    points_out: list[dict[str, Any]] = []
    params = CavityParams()
    paired = "target" in lib.phases and "gangue" in lib.phases
    ref_gangue = _reference_gangue_label(lib)

    for phase_label, phase in lib.phases.items():
        if paired and phase_label != "target":
            continue
        if not paired and phase_label == ref_gangue:
            continue
        for pt in phase.points:
            g_moist = pt.moisture_wt_percent
            try:
                if paired:
                    t_eps = lib.eps(
                        "target", temp_K=pt.temp_K, freq_hz=pt.freq_hz,
                        moisture_wt_percent=pt.moisture_wt_percent,
                    )
                    g_eps = lib.eps(
                        "gangue", temp_K=pt.temp_K, freq_hz=pt.freq_hz,
                        moisture_wt_percent=g_moist,
                    )
                else:
                    if ref_gangue is None:
                        continue
                    t_eps = lib.eps(
                        phase_label, temp_K=pt.temp_K, freq_hz=pt.freq_hz,
                        moisture_wt_percent=pt.moisture_wt_percent,
                    )
                    g_eps = lib.eps(
                        ref_gangue, temp_K=pt.temp_K, freq_hz=pt.freq_hz,
                        moisture_wt_percent=g_moist,
                    )
            except KeyError:
                continue
            mats = Materials(
                target=t_eps,
                gangue=g_eps,
                background=1.0 + 0.0j,
                pair_label="measured_deposit",
            )
            rep = evaluate_params(grid, params, mats)
            points_out.append({
                "target_phase": "target" if paired else phase_label,
                "gangue_phase": "gangue" if paired else ref_gangue,
                "temp_K": pt.temp_K,
                "freq_hz": pt.freq_hz,
                "moisture_wt_percent": pt.moisture_wt_percent,
                "target_eps": [t_eps.real, t_eps.imag],
                "gangue_eps": [g_eps.real, g_eps.imag],
                "selectivity": rep.selectivity,
                "coupling_eff": rep.coupling_eff,
                "source": pt.source,
            })
    return {
        "path": str(path),
        "dataset_id": summary.get("dataset_id"),
        "version": summary.get("version"),
        "eval_mode": "target_gangue" if paired else "multi_phase",
        "n_points_evaluated": len(points_out),
        "points": points_out,
        "provenance": _provenance_for(path),
    }


def _eval_material_pairs(grid: Grid) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    params = CavityParams()
    for label in sorted(PAIRS):
        mats = Materials.from_pair(label)
        rep = evaluate_params(grid, params, mats)
        rows.append({
            "pair": label,
            "selectivity": rep.selectivity,
            "coupling_eff": rep.coupling_eff,
            "contrast": rep.contrast,
            "provenance": "literature",
        })
    return rows


def _eval_phantoms(
    grid: Grid,
    measured_eps_path: Path | None,
    *,
    quick: bool,
) -> list[dict[str, Any]]:
    from mw_inv.phantom import predict_lab_outcome

    n_trials = 4 if quick else 12
    rows: list[dict[str, Any]] = []
    for label in sorted(PHANTOM_RECIPES):
        pred = predict_lab_outcome(
            label,
            grid,
            n_opt_trials=n_trials,
            seed=7701,
            measured_eps_path=measured_eps_path,
        )
        rows.append({
            "phantom": label,
            "untuned_selectivity": pred.untuned_selectivity,
            "optimized_selectivity": pred.optimized_selectivity,
            "untuned_delta_T_K": pred.untuned_delta_T_K,
            "optimized_delta_T_K": pred.optimized_delta_T_K,
            "model_rank_correct": pred.optimized_delta_T_K > pred.untuned_delta_T_K,
            "used_measured_eps": measured_eps_path is not None,
            "provenance": "literature",
        })
    return rows


def _eval_bench_gates(
    probe_path: Path | None,
    lab_path: Path | None,
    *,
    quick: bool,
) -> list[dict[str, Any]]:
    from mw_inv.phantom_calibration import evaluate_bench_gate

    if probe_path is None:
        return []
    rows: list[dict[str, Any]] = []
    grid_n = 31 if quick else 41
    trials = 4 if quick else 8
    for phantom in sorted(PHANTOM_RECIPES):
        report = evaluate_bench_gate(
            phantom,
            probe_path,
            lab_path,
            bench_grid=grid_n,
            bench_trials=trials,
            validate_model=lab_path is not None,
        )
        strict = all(c.passed for c in report.checks)
        rows.append({
            "phantom": phantom,
            "gate_passed": report.passed,
            "strict_passed": strict,
            "probe_path": str(probe_path),
            "lab_path": str(lab_path) if lab_path else None,
            "checks": [c.name for c in report.checks if not c.passed],
            "provenance": _provenance_for(probe_path),
        })
    return rows


def evaluate_real_data(
    data_root: Path | None = None,
    *,
    grid_n: int = 41,
    quick: bool = False,
) -> dict[str, Any]:
    """Run FDFD / bench / benchmark evaluation across discovered data."""
    root = data_root or _repo_data_root()
    if quick:
        grid_n = min(grid_n, 31)
    grid = Grid(nx=grid_n, ny=grid_n, Lx=0.36, Ly=0.36)
    catalog = discover_real_data_catalog(root)

    bench_report = run_benchmarks().to_dict()

    ore_rows: list[dict[str, Any]] = []
    ores_dir = root / "ores"
    if ores_dir.is_dir():
        for p in sorted(ores_dir.rglob("*.json")):
            if p.name.startswith("_"):
                continue
            try:
                ore_rows.append(_eval_ore_row(p, grid, quick=quick))
            except (ValueError, FileNotFoundError, KeyError) as exc:
                ore_rows.append({"ore_path": str(p), "error": str(exc)})

    deposit_rows: list[dict[str, Any]] = []
    dep_dir = root / "measured_dielectrics"
    if dep_dir.is_dir():
        for p in sorted(dep_dir.glob("*.json")):
            if "template" in p.name:
                continue
            try:
                deposit_rows.append(_eval_deposit_library(p, grid))
            except (ValueError, json.JSONDecodeError, KeyError) as exc:
                deposit_rows.append({"path": str(p), "error": str(exc)})

    probe_path = root / "measured_eps.json"
    if not probe_path.is_file():
        probe_path = root / "measured_eps.example.json"
    probe = probe_path if probe_path.is_file() else None

    lab_path = root / "lab_measurements.json"
    if not lab_path.is_file():
        lab_path = root / "lab_measurements.example.json"
    lab = lab_path if lab_path.is_file() else None

    return {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "grid": grid_n,
        "quick": quick,
        "catalog": catalog.to_dict(),
        "benchmarks": bench_report,
        "material_pairs": _eval_material_pairs(grid),
        "ore_profiles": ore_rows,
        "deposit_libraries": deposit_rows,
        "phantoms": _eval_phantoms(grid, probe, quick=quick),
        "bench_gates": _eval_bench_gates(probe, lab, quick=quick),
        "summary": {
            "n_ores": len(ore_rows),
            "n_deposit_libraries": len(deposit_rows),
            "n_deposit_points": sum(r.get("n_points_evaluated", 0) for r in deposit_rows),
            "n_material_pairs": len(PAIRS),
            "n_phantoms": len(PHANTOM_RECIPES),
            "benchmarks_passed": bench_report.get("passed"),
            "using_live_probe": probe is not None and probe.name == "measured_eps.json",
            "using_live_lab": lab is not None and lab.name == "lab_measurements.json",
        },
    }
