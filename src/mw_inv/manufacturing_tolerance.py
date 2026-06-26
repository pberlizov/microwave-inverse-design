"""Manufacturing placement tolerance jitter for robust evaluation (backlog H0)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mw_inv.geometry import CavityParams

# Fractional placement knobs subject to build tolerance (not freq or discrete walls).
MANUFACTURING_JITTER_FIELDS: tuple[str, ...] = (
    "feed_along_frac",
    "stub_depth_frac",
    "plate_cx_frac",
    "plate_cy_frac",
    "plate_len_frac",
    "plate_angle_deg",
    "charge_cx_frac",
    "charge_cy_frac",
)


def jitter_cavity_params(
    params: CavityParams,
    rng: np.random.Generator,
    placement_tol_frac: float,
) -> CavityParams:
    """Apply uniform ± tolerance jitter to manufacturable placement knobs."""
    from dataclasses import replace

    updates: dict[str, float] = {}
    for field in MANUFACTURING_JITTER_FIELDS:
        val = float(getattr(params, field))
        delta = float(rng.uniform(-placement_tol_frac, placement_tol_frac))
        if field == "plate_angle_deg":
            updates[field] = val + delta * 90.0
        elif field.endswith("_frac"):
            updates[field] = float(np.clip(val + delta, 0.02, 0.98))
        else:
            updates[field] = val + delta
    return replace(params, **updates)


@dataclass(frozen=True)
class ManufacturingRobustReport:
    n_samples: int
    placement_tol_frac: float
    min_selectivity: float
    mean_selectivity: float
    std_selectivity: float

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "placement_tol_frac": self.placement_tol_frac,
            "min_selectivity": self.min_selectivity,
            "mean_selectivity": self.mean_selectivity,
            "std_selectivity": self.std_selectivity,
        }


def evaluate_manufacturing_robust(
    grid,
    params: CavityParams,
    materials,
    *,
    n_samples: int = 8,
    placement_tol_frac: float = 0.02,
    seed: int = 0,
    legacy: bool = False,
) -> ManufacturingRobustReport:
    """Worst-case selectivity over ± placement tolerance realizations."""
    from mw_inv.search import evaluate_params

    rng = np.random.default_rng(seed)
    sels: list[float] = []
    for i in range(n_samples):
        jittered = jitter_cavity_params(params, rng, placement_tol_frac)
        sels.append(
            evaluate_params(grid, jittered, materials, legacy=legacy).selectivity
        )
    arr = np.asarray(sels, dtype=float)
    return ManufacturingRobustReport(
        n_samples=n_samples,
        placement_tol_frac=placement_tol_frac,
        min_selectivity=float(arr.min()),
        mean_selectivity=float(arr.mean()),
        std_selectivity=float(arr.std()),
    )
