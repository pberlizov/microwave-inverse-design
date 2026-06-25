"""Export manufacturable designs to FDTD/FDFD cross-check bundles.

Reads optimised params from search or phantom JSON summaries and writes:
openEMS ``.m`` model, FDFD ``.npz`` scene, and a manifest with FDFD selectivity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.fdfd import Grid, solve
from mw_inv.fom import evaluate
from mw_inv.geometry import CavityParams, Materials, build_scene
from mw_inv.openems_export import export_scene_npz, write_openems_model
from mw_inv.search import params_from_dict


@dataclass(frozen=True)
class DesignCase:
    label: str
    params: CavityParams
    source: str  # e.g. "search_summary:tpe_best"


def cases_from_search_summary(data: dict) -> list[DesignCase]:
    """Untuned baseline + random/TPE bests from ``run_search.py`` output."""
    cases = [DesignCase("untuned", CavityParams(), "search_summary:baseline")]
    rnd = data.get("random_search", {}).get("best_params")
    if rnd:
        cases.append(DesignCase("random_best", params_from_dict(rnd), "search_summary:random"))
    tpe = data.get("tpe_search", {}).get("best_params")
    if tpe:
        cases.append(DesignCase("tpe_best", params_from_dict(tpe), "search_summary:tpe"))
    return cases


def cases_from_phantom_prediction(data: dict) -> list[DesignCase]:
    """Untuned + optimised from a single phantom prediction dict."""
    phantom = data.get("phantom", "phantom")
    return [
        DesignCase("untuned", CavityParams(), f"{phantom}:baseline"),
        DesignCase(
            "optimized",
            params_from_dict(data["optimized_params"]),
            f"{phantom}:optimized",
        ),
    ]


def load_search_cases(path: Path | str) -> list[DesignCase]:
    return cases_from_search_summary(json.loads(Path(path).read_text()))


def fdfd_selectivity(
    grid: Grid,
    params: CavityParams,
    materials: Materials,
) -> float:
    scene = build_scene(grid, params, materials)
    res = solve(grid, scene.eps_r, scene.freq_hz, scene.source_xy, mu_r=scene.mu_r)
    return evaluate(res, scene).selectivity


@dataclass(frozen=True)
class ExportBundle:
    label: str
    openems_path: Path
    scene_npz_path: Path
    manifest_path: Path
    fdfd_selectivity: float


def export_design_bundle(
    out_dir: Path | str,
    case: DesignCase,
    materials: Materials,
    *,
    grid_n: int = 81,
    Lz: float = 0.36,
) -> ExportBundle:
    """Write openEMS + FDFD sidecar + JSON manifest for one design case."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = case.label.replace(" ", "_")

    grid = Grid(nx=grid_n, ny=grid_n, Lx=0.36, Ly=0.36)
    fdfd_sel = fdfd_selectivity(grid, case.params, materials)

    m_path = write_openems_model(
        out_dir / f"{tag}_cavity.m", case.params, materials, Lz=Lz,
    )
    npz_path = export_scene_npz(
        out_dir / f"{tag}_scene.npz", case.params, materials, grid_n=grid_n,
    )
    func = "mw_inv_" + m_path.stem.replace("-", "_").replace(".", "_")
    manifest = {
        "label": case.label,
        "source": case.source,
        "fdfd_selectivity": fdfd_sel,
        "params": {
            k: getattr(case.params, k)
            for k in case.params.__dataclass_fields__
            if k != "tuner_field"
        },
        "openems_function": func,
        "openems_model": str(m_path.name),
        "scene_npz": str(npz_path.name),
        "materials": {
            "target_eps": [materials.target.real, materials.target.imag],
            "gangue_eps": [materials.gangue.real, materials.gangue.imag],
        },
        "Lz_m": Lz,
        "grid_n": grid_n,
    }
    manifest_path = out_dir / f"{tag}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return ExportBundle(
        label=case.label,
        openems_path=m_path,
        scene_npz_path=npz_path,
        manifest_path=manifest_path,
        fdfd_selectivity=fdfd_sel,
    )


def export_all_cases(
    out_dir: Path | str,
    cases: list[DesignCase],
    materials: Materials,
    **kwargs,
) -> list[ExportBundle]:
    return [export_design_bundle(out_dir, case, materials, **kwargs) for case in cases]
