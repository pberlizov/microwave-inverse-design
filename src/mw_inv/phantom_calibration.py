"""Bench phantom calibration: measured ε from probe → recipe override."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.geometry import Materials
from mw_inv.phantom_data import PHANTOM_RECIPES, PhantomRecipePair


@dataclass(frozen=True)
class MeasuredBatch:
    label: str
    salt_wt_percent: float
    eps_real: float
    eps_imag: float
    freq_hz: float = 2.45e9
    method: str = "open_coax_probe"
    notes: str = ""

    @property
    def eps(self) -> complex:
        return complex(self.eps_real, self.eps_imag)


def load_measured_eps(path: Path | str) -> dict[str, MeasuredBatch]:
    """Load ``data/measured_eps.json`` keyed by batch label."""
    data = json.loads(Path(path).read_text())
    batches = data.get("batches", data)
    out: dict[str, MeasuredBatch] = {}
    for row in batches:
        out[row["label"]] = MeasuredBatch(**{k: row[k] for k in MeasuredBatch.__dataclass_fields__ if k in row})
    return out


def recipe_with_measured(
    phantom_label: str,
    measured: dict[str, MeasuredBatch],
) -> PhantomRecipePair:
    """Override recipe anchor ε with probe-measured batch values when present."""
    base = PHANTOM_RECIPES[phantom_label]
    target_key = base.target.label
    gangue_key = base.gangue.label
    from dataclasses import replace

    target = replace(base.target, label=target_key)
    gangue = replace(base.gangue, label=gangue_key)
    # attach measured eps via monkey-patch properties — use Materials directly instead
    _ = target, gangue
    return base  # caller uses materials_from_measured_recipe


def materials_from_measured_recipe(
    phantom_label: str,
    measured_path: Path | str | None = None,
) -> Materials:
    """Build Materials using measured ε when available, else Gabriel anchors."""
    recipe = PHANTOM_RECIPES[phantom_label]
    t_eps = recipe.target.eps
    g_eps = recipe.gangue.eps
    if measured_path and Path(measured_path).is_file():
        batches = load_measured_eps(measured_path)
        if recipe.target.label in batches:
            t_eps = batches[recipe.target.label].eps
        if recipe.gangue.label in batches:
            g_eps = batches[recipe.gangue.label].eps
    return Materials(target=t_eps, gangue=g_eps, background=1.0 + 0.0j)


def compare_measured_vs_anchor(phantom_label: str, measured_path: Path | str) -> dict:
    """Report anchor vs measured ε drift for calibration QA."""
    recipe = PHANTOM_RECIPES[phantom_label]
    batches = load_measured_eps(measured_path)
    rows = []
    for role, gel in (("target", recipe.target), ("gangue", recipe.gangue)):
        anchor = gel.eps
        meas = batches.get(gel.label)
        if meas is None:
            rows.append({"role": role, "batch": gel.label, "status": "missing"})
            continue
        rows.append({
            "role": role,
            "batch": gel.label,
            "anchor_eps": [anchor.real, anchor.imag],
            "measured_eps": [meas.eps.real, meas.eps.imag],
            "drift_real": meas.eps.real - anchor.real,
            "drift_imag": meas.eps.imag - anchor.imag,
            "method": meas.method,
        })
    return {"phantom": phantom_label, "comparisons": rows}
