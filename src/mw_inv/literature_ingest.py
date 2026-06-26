"""Build versioned measured_dielectrics JSON from cited literature tables.

Adapters ship curated anchor points (not scraped PDFs) so CI and
``run_real_data_eval`` can evaluate on public data without network fetch.
Table sources live under ``data/literature_tables/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from mw_inv.dielectric_data import MINERAL_MODELS
from mw_inv.measured_dielectrics import load_measured_dielectrics, validate_library

# Hartlieb et al. 2016 Minerals Engineering Table 2 @ 2450 MHz (RT–1000 °C).
_HARTLIEB_2450: dict[str, dict[str, list[float]]] = {
    "basalt": {
        "temp_C": [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "eps_real": [7.67, 7.89, 7.77, 7.53, 7.47, 7.65, 8.49, 8.07, 7.99, 8.35, 8.97],
        "eps_imag": [0.270, 0.332, 0.261, 0.155, 0.127, 0.301, 0.774, 0.555, 0.619, 0.989, 1.306],
    },
    "granite": {
        "temp_C": [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "eps_real": [5.45, 5.44, 5.45, 5.45, 5.45, 5.47, 5.45, 5.48, 5.37, 5.08, 5.22],
        "eps_imag": [0.038, 0.033, 0.037, 0.043, 0.052, 0.076, 0.118, 0.210, 0.175, 0.135, 0.319],
    },
    "sandstone": {
        "temp_C": [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "eps_real": [4.93, 4.85, 4.83, 4.84, 4.83, 4.81, 4.74, 4.76, 4.86, 4.97, 5.29],
        "eps_imag": [0.081, 0.042, 0.038, 0.047, 0.045, 0.056, 0.073, 0.132, 0.205, 0.244, 0.439],
    },
}

_USBM_FREQ_HZ = [300e6, 400e6, 500e6, 600e6, 700e6, 800e6, 900e6, 1000e6]
_USBM_LOW_LOSS: dict[str, dict[str, list[float | None]]] = {
    "quartz": {
        "eps_real": [4.00, 3.96, 3.89, 3.85, 3.90, 3.97, 3.94, 3.89],
        "tan_delta_x1e4": [1.88, 1.50, 2.20, 1.82, 1.54, None, 1.60, 1.37],
    },
    "calcite": {
        "eps_real": [9.16, 8.89, 9.10, 8.86, 9.02, 8.66, 8.84, 8.91],
        "tan_delta_x1e4": [0.77, 0.87, 0.67, 0.62, 0.33, 0.34, None, 0.47],
    },
    "dolomite": {
        "eps_real": [7.62, 7.38, 7.32, 7.37, 7.34, 7.22, 7.26, 7.41],
        "tan_delta_x1e4": [2.60, 2.98, 2.51, 3.39, 4.34, 2.63, 3.11, 2.42],
    },
    "orthoclase": {
        "eps_real": [4.56, 4.54, 4.43, 4.37, 4.40, 4.41, 4.37, 4.34],
        "tan_delta_x1e4": [0.64, 0.47, None, 0.64, 0.63, None, None, 0.43],
    },
}

_FREQ_2450 = 2.45e9
_SOURCE_HARTLIEB = "Hartlieb et al. 2016 Minerals Eng. Table 2 @ 2450 MHz"
_SOURCE_USBM = "Church & Webb USBM RI 9035 (CDC 10413)"


def _repo_data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data"


def _literature_tables_dir(data_root: Path | None = None) -> Path:
    return (data_root or _repo_data_root()) / "literature_tables"


def _load_table(name: str, data_root: Path | None = None) -> dict[str, Any]:
    path = _literature_tables_dir(data_root) / name
    if not path.is_file():
        raise FileNotFoundError(f"missing literature table: {path}")
    return json.loads(path.read_text())


def _points_from_temp_series(
    *,
    label: str,
    temp_C: list[float],
    eps_real: list[float],
    eps_imag: list[float],
    freq_hz: float,
    source: str,
) -> list[dict[str, Any]]:
    if not (len(temp_C) == len(eps_real) == len(eps_imag)):
        raise ValueError(f"{label}: mismatched series lengths")
    return [
        {
            "temp_K": tc + 273.15,
            "freq_hz": freq_hz,
            "eps_real": er,
            "eps_imag": ei,
            "moisture_wt_percent": 0.0,
            "source": source,
            "notes": f"{label} @ {freq_hz/1e9:.3f} GHz",
        }
        for tc, er, ei in zip(temp_C, eps_real, eps_imag, strict=True)
    ]


def _points_from_usbm_table(
    mineral: str,
    eps_real: list[float],
    tan_delta_x1e4: list[float | None],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f_hz, er, tan in zip(_USBM_FREQ_HZ, eps_real, tan_delta_x1e4, strict=True):
        if tan is None:
            continue
        ei = er * (tan * 1e-4)
        out.append({
            "temp_K": 298.15,
            "freq_hz": f_hz,
            "eps_real": er,
            "eps_imag": ei,
            "moisture_wt_percent": 0.0,
            "source": _SOURCE_USBM,
            "notes": f"{mineral} dry powder @ {f_hz/1e6:.0f} MHz",
        })
    near_915 = min(out, key=lambda p: abs(p["freq_hz"] - 915e6))
    out.append({
        "temp_K": 298.15,
        "freq_hz": _FREQ_2450,
        "eps_real": near_915["eps_real"],
        "eps_imag": near_915["eps_imag"],
        "moisture_wt_percent": 0.0,
        "source": _SOURCE_USBM,
        "notes": f"{mineral} 2.45 GHz carry-over from {near_915['freq_hz']/1e6:.0f} MHz",
    })
    return out


def _build_from_coax_table(table: dict[str, Any]) -> dict[str, Any]:
    source = str(table.get("source", "USBM CDC 10047"))
    temps = table["temps_C"]
    phases = []
    for mineral in table["minerals"]:
        label = mineral["label"]
        freqs = mineral["freq_hz"]
        points: list[dict[str, Any]] = []
        for ti, tc in enumerate(temps):
            for fi, f_hz in enumerate(freqs):
                points.append({
                    "temp_K": tc + 273.15,
                    "freq_hz": float(f_hz),
                    "eps_real": mineral["eps_real"][ti][fi],
                    "eps_imag": mineral["eps_imag"][ti][fi],
                    "moisture_wt_percent": 0.0,
                    "source": source,
                    "notes": f"{label} natural density",
                })
        phases.append({"label": label, "points": points})
    return {
        "dataset_id": "usbm_coax_minerals_v1",
        "version": "cdc-10047-subset",
        "description": "USBM open-ended coax HMAP + gangue subset @ 915 & 2450 MHz.",
        "phases": phases,
    }


def _build_from_forster_hmap_table(table: dict[str, Any]) -> dict[str, Any]:
    source = str(table.get("source", "Forster PhD 2023"))
    freqs = table["freq_hz"]
    phases = []
    for mineral in table["minerals"]:
        label = mineral["label"]
        temps = mineral["temps_C"]
        points: list[dict[str, Any]] = []
        for ti, tc in enumerate(temps):
            for fi, f_hz in enumerate(freqs):
                points.append({
                    "temp_K": tc + 273.15,
                    "freq_hz": float(f_hz),
                    "eps_real": mineral["eps_real"][ti][fi],
                    "eps_imag": mineral["eps_imag"][ti][fi],
                    "moisture_wt_percent": 0.0,
                    "source": source,
                    "notes": label,
                })
        phases.append({"label": label, "points": points})
    return {
        "dataset_id": "forster_hmap_minerals_v1",
        "version": "forster-phd-ch4",
        "description": "Forster/Goldbaum HMAP sulphide ε(T,f) cavity-perturbation anchors.",
        "phases": phases,
    }


def build_hartlieb_bedrock_library(data_root: Path | None = None) -> dict[str, Any]:
    del data_root
    phases = [
        {
            "label": rock,
            "points": _points_from_temp_series(
                label=rock,
                temp_C=table["temp_C"],
                eps_real=table["eps_real"],
                eps_imag=table["eps_imag"],
                freq_hz=_FREQ_2450,
                source=_SOURCE_HARTLIEB,
            ),
        }
        for rock, table in _HARTLIEB_2450.items()
    ]
    return {
        "dataset_id": "hartlieb_bedrock_v1",
        "version": "2016-table2",
        "description": "Basalt, granite, sandstone ε(T) @ 2450 MHz (Hartlieb et al. 2016).",
        "phases": phases,
    }


def build_usbm_low_loss_library(data_root: Path | None = None) -> dict[str, Any]:
    del data_root
    phases = [
        {
            "label": mineral,
            "points": _points_from_usbm_table(mineral, table["eps_real"], table["tan_delta_x1e4"]),
        }
        for mineral, table in _USBM_LOW_LOSS.items()
    ]
    return {
        "dataset_id": "usbm_low_loss_gangue_v1",
        "version": "1986-ri9035",
        "description": "Low-loss gangue minerals 300–1000 MHz + 2.45 GHz carry-over (USBM RI 9035).",
        "phases": phases,
    }


def build_literature_minerals_library(data_root: Path | None = None) -> dict[str, Any]:
    del data_root
    phases = []
    for key, model in sorted(MINERAL_MODELS.items()):
        points = [
            {
                "temp_K": anchor.temp_K,
                "freq_hz": anchor.freq_hz,
                "eps_real": anchor.eps_real,
                "eps_imag": anchor.eps_imag,
                "moisture_wt_percent": 0.0,
                "source": anchor.source,
                "notes": model.name,
            }
            for anchor in model.eps_anchors
        ]
        phases.append({"label": key, "points": points})
    return {
        "dataset_id": "literature_hmap_minerals_v1",
        "version": "dielectric_data-export",
        "description": "Scene-scale mineral ε(T) anchors exported from mw_inv.dielectric_data.",
        "phases": phases,
    }


def build_usbm_coax_library(data_root: Path | None = None) -> dict[str, Any]:
    return _build_from_coax_table(_load_table("usbm_coax_minerals.json", data_root))


def build_forster_hmap_library(data_root: Path | None = None) -> dict[str, Any]:
    base = _build_from_forster_hmap_table(_load_table("forster_hmap_minerals.json", data_root))
    existing = {p["label"] for p in base["phases"]}
    # Gangue phases referenced by forster_42ores manifest.
    gangue_libs = (
        build_usbm_low_loss_library(data_root),
        build_usbm_coax_library(data_root),
        build_literature_minerals_library(data_root),
    )
    for lib in gangue_libs:
        for phase in lib["phases"]:
            if phase["label"] not in existing:
                base["phases"].append(phase)
                existing.add(phase["label"])
    base["description"] = (
        "Forster HMAP sulphides + USBM/literature gangue phases for 42-ore manifest."
    )
    return base


def build_europeg_library(data_root: Path | None = None) -> dict[str, Any]:
    table = _load_table("europeg_pegmatite_subset.json", data_root)
    source = str(table.get("source", "EuroPeg_PetroDB"))
    target_pts: list[dict[str, Any]] = []
    gangue_pts: list[dict[str, Any]] = []
    for sample in table["samples"]:
        t_label = sample["target_phase"]
        g_label = sample["gangue_phase"]
        for pt in sample["points"]:
            target_pts.append({
                "temp_K": pt["temp_K"],
                "freq_hz": pt["freq_hz"],
                "eps_real": pt["target_eps_real"],
                "eps_imag": pt["target_eps_imag"],
                "moisture_wt_percent": 0.0,
                "source": source,
                "notes": f"{sample['label']} target ({t_label})",
            })
            gangue_pts.append({
                "temp_K": pt["temp_K"],
                "freq_hz": pt["freq_hz"],
                "eps_real": pt["gangue_eps_real"],
                "eps_imag": pt["gangue_eps_imag"],
                "moisture_wt_percent": 0.0,
                "source": source,
                "notes": f"{sample['label']} wall ({g_label})",
            })
    return {
        "dataset_id": "europeg_pegmatite_v1",
        "version": "petrodb-subset",
        "description": "EuroPeg pegmatite ore vs wall-rock ε anchors (representative subset).",
        "phases": [
            {"label": "target", "points": target_pts},
            {"label": "gangue", "points": gangue_pts},
        ],
    }


def build_computed_static_library(data_root: Path | None = None) -> dict[str, Any]:
    table = _load_table("computed_dielectric_subset.json", data_root)
    source = str(table.get("source", "MP/Zenodo static"))
    phases = []
    for row in table["minerals"]:
        phases.append({
            "label": row["label"],
            "points": [{
                "temp_K": 298.15,
                "freq_hz": _FREQ_2450,
                "eps_real": row["eps_iso"],
                "eps_imag": 0.01,
                "moisture_wt_percent": 0.0,
                "source": row.get("source", source),
                "notes": f"{row['formula']} static DFPT (loss placeholder)",
            }],
        })
    return {
        "dataset_id": "computed_dielectric_subset_v1",
        "version": "mp-zenodo-subset",
        "description": str(table.get("note", "Static computed ε — ranking only.")),
        "phases": phases,
    }


def build_gabriel_phantoms_library(data_root: Path | None = None) -> dict[str, Any]:
    table = _load_table("gabriel_saline_anchors.json", data_root)
    source = str(table.get("source", "Gabriel 1996"))
    anchors = {a["wt_percent"]: a for a in table["salt_wt_percent_anchors"]}
    target_pts: list[dict[str, Any]] = []
    gangue_pts: list[dict[str, Any]] = []
    for pair in table["phantom_pairs"]:
        t = anchors[pair["target_wt_percent"]]
        g = anchors[pair["gangue_wt_percent"]]
        note = pair["label"]
        target_pts.append({
            "temp_K": 298.15,
            "freq_hz": _FREQ_2450,
            "eps_real": t["eps_real"],
            "eps_imag": t["eps_imag"],
            "moisture_wt_percent": t["wt_percent"],
            "source": source,
            "notes": f"{note} target gel",
        })
        gangue_pts.append({
            "temp_K": 298.15,
            "freq_hz": _FREQ_2450,
            "eps_real": g["eps_real"],
            "eps_imag": g["eps_imag"],
            "moisture_wt_percent": g["wt_percent"],
            "source": source,
            "notes": f"{note} gangue gel",
        })
    saline_phases = [
        {
            "label": f"saline_{a['wt_percent']:.1f}pct",
            "points": [{
                "temp_K": 298.15,
                "freq_hz": _FREQ_2450,
                "eps_real": a["eps_real"],
                "eps_imag": a["eps_imag"],
                "moisture_wt_percent": a["wt_percent"],
                "source": source,
                "notes": f"NaCl {a['wt_percent']} wt% agar gel",
            }],
        }
        for a in table["salt_wt_percent_anchors"]
    ]
    return {
        "dataset_id": "gabriel_saline_phantoms_v1",
        "version": "phantom_data-export",
        "description": "Gabriel-scaled saline gel phantom ε @ 2.45 GHz.",
        "phases": [
            {"label": "target", "points": target_pts},
            {"label": "gangue", "points": gangue_pts},
            *saline_phases,
        ],
    }


def build_forster_ores(data_root: Path | None = None) -> list[Path]:
    """Write 42 Forster-style ore profiles under data/ores/forster/."""
    root = data_root or _repo_data_root()
    manifest = _load_table("forster_42ores_manifest.json", root)
    out_dir = root / "ores" / "forster"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = manifest.get("measured_dielectrics_path", "../measured_dielectrics/forster_hmap_minerals_v1.json")
    written: list[Path] = []
    for row in manifest["ores"]:
        ore = {
            "label": row["label"],
            "source": "forster_phd_2023",
            "heating_class": row["heating_class"],
            "hmap_wt_percent": row["hmap_wt_percent"],
            "fractions": row["fractions"],
            "gangue_mineral": row["gangue_mineral"],
            "texture": {"class": "disseminated", "mean_grain_radius_m": 0.002},
            "measured_dielectrics": {
                "path": md_path,
                "target_phase": row["target_phase"],
                "gangue_phase": row["gangue_phase"],
                "default_moisture_wt_percent": 0.0,
            },
        }
        path = out_dir / f"{row['label']}.json"
        path.write_text(json.dumps(ore, indent=2) + "\n")
        written.append(path)
    summary_path = out_dir / "_manifest.json"
    summary_path.write_text(json.dumps({
        "dataset_id": "forster_42ores_v1",
        "n_ores": len(written),
        "source": manifest.get("source"),
        "citation": manifest.get("citation"),
        "ore_files": [p.name for p in written],
    }, indent=2) + "\n")
    written.append(summary_path)
    return written


# Adapter name -> builder (returns dict for measured_dielectrics, or special for ores)
_BUILDERS: dict[str, Callable[..., dict[str, Any]]] = {
    "hartlieb": build_hartlieb_bedrock_library,
    "usbm_low_loss": build_usbm_low_loss_library,
    "literature_minerals": build_literature_minerals_library,
    "usbm_coax": build_usbm_coax_library,
    "forster_hmap": build_forster_hmap_library,
    "europeg": build_europeg_library,
    "computed_static": build_computed_static_library,
    "gabriel": build_gabriel_phantoms_library,
}

_ORE_ADAPTERS = frozenset({"forster_ores"})


def write_library(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    issues = validate_library(load_measured_dielectrics(path))
    if issues:
        raise ValueError(f"validation failed for {path}: {issues[:3]}")


def _output_path(adapter: str, data_root: Path) -> Path:
    from mw_inv.external_datasets import load_datasets_catalog

    for entry in load_datasets_catalog(data_root).entries:
        if entry.ingest_adapter == adapter and entry.ingest_output:
            return data_root / entry.ingest_output
    raise KeyError(f"no catalog output for adapter {adapter!r}")


def ingest_literature_dataset(adapter: str, data_root: Path | None = None) -> Path | list[Path]:
    """Run one adapter; returns path(s) written."""
    root = data_root or _repo_data_root()
    if adapter in _ORE_ADAPTERS:
        if adapter == "forster_ores":
            return build_forster_ores(root)
        raise KeyError(adapter)
    if adapter not in _BUILDERS:
        raise KeyError(f"unknown adapter {adapter!r}; choose from {sorted(_BUILDERS | _ORE_ADAPTERS)}")
    out = _output_path(adapter, root)
    write_library(out, _BUILDERS[adapter](root))
    return out


def ingest_all_auto(data_root: Path | None = None) -> list[Path]:
    """Write all auto-ingest entries from datasets_catalog.json."""
    from mw_inv.external_datasets import load_datasets_catalog

    root = data_root or _repo_data_root()
    written: list[Path] = []
    for entry in load_datasets_catalog(root).auto_ingest_entries():
        adapter = entry.ingest_adapter
        if not adapter:
            continue
        result = ingest_literature_dataset(adapter, root)
        if isinstance(result, list):
            written.extend(result)
        else:
            written.append(result)
    return written


def list_adapters() -> list[str]:
    return sorted(_BUILDERS.keys() | _ORE_ADAPTERS)
