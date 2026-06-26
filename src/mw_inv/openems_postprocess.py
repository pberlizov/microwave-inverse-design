"""Python post-processing for openEMS field dumps (selectivity in target vs gangue).

Mirrors the Octave logic in ``openems_export.generate_openems_script``.  Requires
``h5py`` when reading real dumps — optional dependency, not in ``requirements.txt``.

Port-truth metrics (backlog A1): matched-port ``|S11|`` and ``coupling_eff = 1 - |S11|²``
are written by exported openEMS scripts to ``port_metrics.json`` beside field dumps.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mw_inv.fdfd import EPS0
from mw_inv.geometry import CavityParams, Materials

__all__ = [
    "OpenemsCaseMetrics",
    "OpenemsPortMetrics",
    "h5py_available",
    "ingest_openems_case",
    "load_port_metrics",
    "resolve_openems_case_paths",
    "selectivity_from_e2",
    "selectivity_from_openems_dump",
]


@dataclass(frozen=True)
class OpenemsPortMetrics:
    """Matched-port figures from openEMS ``calcPort`` (truth-solver path)."""

    s11_mag: float
    coupling_eff: float
    selectivity: float | None = None
    freq_hz: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "s11_mag": self.s11_mag,
            "coupling_eff": self.coupling_eff,
            "selectivity": self.selectivity,
            "freq_hz": self.freq_hz,
        }


@dataclass(frozen=True)
class OpenemsCaseMetrics:
    """Combined field-dump + port metrics for one openEMS case directory."""

    selectivity: float | None
    s11_mag: float | None
    coupling_eff: float | None
    field_dump: str | None
    port_metrics_path: str | None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "openems_selectivity": self.selectivity,
            "openems_s11_mag": self.s11_mag,
            "openems_coupling_eff": self.coupling_eff,
            "field_dump": self.field_dump,
            "port_metrics_path": self.port_metrics_path,
        }


def h5py_available() -> bool:
    try:
        import h5py  # noqa: F401

        return True
    except ImportError:
        return False


def _charge_masks(
    params: CavityParams,
    shape: tuple[int, int, int],
    *,
    Lx: float = 0.36,
    Ly: float = 0.36,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner-frame coordinate masks matching openEMS Octave post-process."""
    nx, ny, nz = shape
    xg = np.linspace(0.0, Lx, nx)
    yg = np.linspace(0.0, Ly, ny)
    X, Y, _Z = np.meshgrid(xg, yg, np.linspace(0, 1, nz), indexing="ij")

    cx = params.charge_cx_frac * Lx
    cy = params.charge_cy_frac * Ly
    hw = 0.5 * params.charge_w_frac * Lx
    hh = 0.5 * params.charge_h_frac * Ly
    r_grain = params.inclusion_radius_frac * min(Lx, Ly)

    gangue_mask = (np.abs(X - cx) <= hw) & (np.abs(Y - cy) <= hh)
    target_mask = np.zeros(shape, dtype=bool)
    for ox, oy in params.inclusion_offsets_frac:
        gx = (params.charge_cx_frac + ox) * Lx
        gy = (params.charge_cy_frac + oy) * Ly
        target_mask |= (X - gx) ** 2 + (Y - gy) ** 2 <= r_grain ** 2
    target_mask &= gangue_mask
    return gangue_mask, target_mask


def selectivity_from_e2(
    e2: np.ndarray,
    params: CavityParams,
    materials: Materials,
    freq_hz: float,
    *,
    Lx: float = 0.36,
    Ly: float = 0.36,
    Lz: float = 0.36,
) -> float:
    """Dissipated-power selectivity from |E|² volume field (openEMS convention)."""
    if e2.ndim != 3:
        raise ValueError(f"expected 3D |E|² field, got shape {e2.shape}")
    gangue_mask, target_mask = _charge_masks(params, e2.shape, Lx=Lx, Ly=Ly)
    nx, ny, nz = e2.shape
    dx = Lx / max(nx - 1, 1)
    dy = Ly / max(ny - 1, 1)
    dz = Lz / max(nz - 1, 1)
    cell_vol = dx * dy * dz
    omega = 2.0 * math.pi * freq_hz
    eps_im_g = max(materials.gangue.imag, 0.0)
    eps_im_t = max(materials.target.imag, 0.0)
    p_g = 0.5 * omega * EPS0 * eps_im_g * float(e2[gangue_mask & ~target_mask].sum()) * cell_vol
    p_t = 0.5 * omega * EPS0 * eps_im_t * float(e2[target_mask].sum()) * cell_vol
    total = p_t + p_g
    return p_t / total if total > 0 else 0.0


def load_port_metrics(path: Path | str) -> OpenemsPortMetrics:
    """Load ``port_metrics.json`` written by exported openEMS Octave scripts."""
    data = json.loads(Path(path).read_text())
    return OpenemsPortMetrics(
        s11_mag=float(data["s11_mag"]),
        coupling_eff=float(data["coupling_eff"]),
        selectivity=float(data["selectivity"]) if data.get("selectivity") is not None else None,
        freq_hz=float(data["freq_hz"]) if data.get("freq_hz") is not None else None,
    )


def resolve_openems_case_paths(case_dir: Path) -> tuple[Path | None, Path | None]:
    """Return ``(field_dump_h5, port_metrics_json)`` for an openEMS case folder."""
    case_dir = Path(case_dir)
    metrics = case_dir / "port_metrics.json"
    port_path = metrics if metrics.is_file() else None

    candidates = (
        case_dir / "Et" / "Et_0000.h5",
        case_dir / "Et_0000.h5",
        case_dir / "Et" / "Et_0000",
    )
    field_path: Path | None = None
    for p in candidates:
        if p.is_file():
            field_path = p
            break
    return field_path, port_path


def ingest_openems_case(
    case_dir: Path | str,
    params: CavityParams,
    materials: Materials,
    *,
    Lz: float = 0.36,
) -> OpenemsCaseMetrics:
    """Read field dump + port JSON from an openEMS run directory."""
    case_dir = Path(case_dir)
    field_path, port_path = resolve_openems_case_paths(case_dir)

    selectivity: float | None = None
    if field_path is not None and h5py_available():
        selectivity = selectivity_from_openems_dump(field_path, params, materials, Lz=Lz)

    s11_mag: float | None = None
    coupling_eff: float | None = None
    if port_path is not None:
        port = load_port_metrics(port_path)
        s11_mag = port.s11_mag
        coupling_eff = port.coupling_eff
        if selectivity is None and port.selectivity is not None:
            selectivity = port.selectivity

    return OpenemsCaseMetrics(
        selectivity=selectivity,
        s11_mag=s11_mag,
        coupling_eff=coupling_eff,
        field_dump=str(field_path) if field_path else None,
        port_metrics_path=str(port_path) if port_path else None,
    )


def _load_e2_from_hdf5(path: Path) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        # openEMS ReadHDF5Dump layout varies by version; try common paths.
        candidates: list[str] = []
        def visit(name: str, obj) -> None:
            if hasattr(obj, "shape") and len(getattr(obj, "shape", ())) >= 3:
                candidates.append(name)

        f.visititems(lambda name, obj: visit(name, obj) if isinstance(obj, h5py.Dataset) else None)

        for key in ("FieldData/E", "Et", "E"):
            if key in f:
                candidates.insert(0, key)

        for key in candidates:
            ds = f[key]
            arr = np.asarray(ds[()])
            if arr.ndim == 4 and arr.shape[-1] >= 3:
                # (nx, ny, nz, 3) complex or real components
                if np.iscomplexobj(arr):
                    e2 = np.sum(np.abs(arr[..., :3]) ** 2, axis=-1)
                else:
                    e2 = np.sum(arr[..., :3] ** 2, axis=-1)
                return e2.astype(float)
            if arr.ndim == 3:
                return np.abs(arr).astype(float) ** 2

    raise ValueError(f"could not locate E-field dataset in {path}")


def selectivity_from_openems_dump(
    dump_path: Path | str,
    params: CavityParams,
    materials: Materials,
    *,
    Lz: float = 0.36,
) -> float:
    """Read openEMS HDF5 dump and return target/charge selectivity."""
    if not h5py_available():
        raise ImportError("h5py required for openEMS dump ingestion (pip install h5py)")
    e2 = _load_e2_from_hdf5(Path(dump_path))
    return selectivity_from_e2(e2, params, materials, params.freq_hz, Lz=Lz)
