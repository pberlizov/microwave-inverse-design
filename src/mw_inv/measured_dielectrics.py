"""Measured dielectric datasets for deposit-/batch-specific materials.

This complements `mw_inv.dielectric_data`, which provides cited literature anchors for
scene-scale mineral phases. Industry use needs *measured* ε(f, T, moisture, PSD, …) for
the actual ore and gangue on-hand.

This module provides:
- a JSON schema for measured complex permittivity points
- light validation
- simple interpolation over (T, f) with optional nearest-moisture selection

Convention: complex relative permittivity is `eps = eps' + i*eps''` with `eps'' > 0`
for lossy media (e^{-i ωt}).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MeasuredDielectricPoint:
    temp_K: float
    freq_hz: float
    eps_real: float
    eps_imag: float
    moisture_wt_percent: float | None = None
    source: str = ""
    notes: str = ""

    @property
    def eps(self) -> complex:
        return complex(float(self.eps_real), float(self.eps_imag))


@dataclass(frozen=True)
class PhaseDielectricDataset:
    label: str
    points: tuple[MeasuredDielectricPoint, ...]

    def eps(self, *, temp_K: float, freq_hz: float, moisture_wt_percent: float | None = None) -> complex:
        pts = list(self.points)
        if not pts:
            raise ValueError(f"phase {self.label!r} has no points")

        moist_values = sorted({p.moisture_wt_percent for p in pts if p.moisture_wt_percent is not None})
        if moist_values and moisture_wt_percent is not None and len(moist_values) >= 2:
            eps_by_m = [
                (m, _eps_at_temp_freq([p for p in pts if p.moisture_wt_percent == m], temp_K, freq_hz))
                for m in moist_values
            ]
            return _interp_moisture(eps_by_m, moisture_wt_percent)

        if moist_values and moisture_wt_percent is not None:
            m_near = min(moist_values, key=lambda m: abs(m - moisture_wt_percent))
            pts = [p for p in pts if p.moisture_wt_percent == m_near]

        return _eps_at_temp_freq(pts, temp_K, freq_hz)


@dataclass(frozen=True)
class MeasuredDielectricLibrary:
    """A set of named phases (e.g. ore_bulk, gangue_bulk, concentrate_pyrite, …)."""

    phases: dict[str, PhaseDielectricDataset]
    description: str = ""
    dataset_id: str = ""
    version: str = ""

    def eps(
        self,
        phase: str,
        *,
        temp_K: float,
        freq_hz: float,
        moisture_wt_percent: float | None = None,
    ) -> complex:
        if phase not in self.phases:
            raise KeyError(f"unknown measured phase {phase!r}; available: {sorted(self.phases)}")
        return self.phases[phase].eps(
            temp_K=temp_K, freq_hz=freq_hz, moisture_wt_percent=moisture_wt_percent,
        )

    def summary(self) -> dict[str, object]:
        """Compact metadata for manifests and ingest reports."""
        phases: dict[str, object] = {}
        for label, phase in sorted(self.phases.items()):
            temps = sorted({p.temp_K for p in phase.points})
            freqs = sorted({p.freq_hz for p in phase.points})
            moist = sorted({p.moisture_wt_percent for p in phase.points if p.moisture_wt_percent is not None})
            sources = sorted({p.source for p in phase.points if p.source})
            phases[label] = {
                "n_points": len(phase.points),
                "temp_K_range": [temps[0], temps[-1]] if temps else [],
                "freq_hz_range": [freqs[0], freqs[-1]] if freqs else [],
                "moisture_wt_percent_levels": moist,
                "sources": sources,
            }
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "description": self.description,
            "phases": phases,
        }


def load_measured_dielectrics(path: Path | str) -> MeasuredDielectricLibrary:
    """Load measured dielectrics from JSON.

    Supported JSON shapes:
    1) {"description": "...", "phases": [{"label": "...", "points": [...]}, ...]}
    2) {"description": "...", "phases": {"phase_label": [...points...], ...}}
    3) {"phase_label": [...points...], ...}  (legacy/minimal)
    """
    data = json.loads(Path(path).read_text())
    description = str(data.get("description", "")) if isinstance(data, dict) else ""
    dataset_id = str(data.get("dataset_id", "")) if isinstance(data, dict) else ""
    version = str(data.get("version", "")) if isinstance(data, dict) else ""

    phases_block = data.get("phases") if isinstance(data, dict) and "phases" in data else data
    phases: dict[str, PhaseDielectricDataset] = {}

    if isinstance(phases_block, list):
        for row in phases_block:
            label = str(row["label"])
            points = tuple(_parse_point(p) for p in row.get("points", []))
            phases[label] = PhaseDielectricDataset(label=label, points=points)
    elif isinstance(phases_block, dict):
        for label, points_list in phases_block.items():
            if label == "description":
                continue
            if not isinstance(points_list, list):
                continue
            points = tuple(_parse_point(p) for p in points_list)
            phases[str(label)] = PhaseDielectricDataset(label=str(label), points=points)
    else:
        raise ValueError("measured dielectrics JSON must contain a phases mapping or list")

    if not phases:
        raise ValueError("no phases found in measured dielectrics JSON")
    return MeasuredDielectricLibrary(
        phases=phases,
        description=description,
        dataset_id=dataset_id,
        version=version,
    )


def validate_library(lib: MeasuredDielectricLibrary) -> list[str]:
    """Return a list of human-readable validation issues (empty => OK)."""
    issues: list[str] = []
    for label, phase in lib.phases.items():
        if not phase.points:
            issues.append(f"{label}: no points")
            continue
        for p in phase.points:
            if p.temp_K <= 0:
                issues.append(f"{label}: temp_K must be > 0 (got {p.temp_K})")
            if p.freq_hz <= 0:
                issues.append(f"{label}: freq_hz must be > 0 (got {p.freq_hz})")
            if p.eps_real <= 0:
                issues.append(f"{label}: eps_real must be > 0 (got {p.eps_real})")
            if p.eps_imag < 0:
                issues.append(f"{label}: eps_imag must be >= 0 (got {p.eps_imag})")
    return issues


def _parse_point(d: dict) -> MeasuredDielectricPoint:
    return MeasuredDielectricPoint(
        temp_K=float(d["temp_K"]),
        freq_hz=float(d["freq_hz"]),
        eps_real=float(d["eps_real"]),
        eps_imag=float(d["eps_imag"]),
        moisture_wt_percent=(None if d.get("moisture_wt_percent") is None else float(d["moisture_wt_percent"])),
        source=str(d.get("source", "")),
        notes=str(d.get("notes", "")),
    )


def _eps_at_temp_freq(points: list[MeasuredDielectricPoint], temp_K: float, freq_hz: float) -> complex:
    """Interpolate ε in T at bracketed frequencies, then in f."""
    if not points:
        raise ValueError("no points for (T, f) interpolation")
    freqs = sorted({p.freq_hz for p in points})
    if len(freqs) == 1:
        return _interp_temp([p for p in points if p.freq_hz == freqs[0]], temp_K)

    f0, f1 = _bracket(freq_hz, freqs)
    e0 = _interp_temp([p for p in points if p.freq_hz == f0], temp_K)
    e1 = _interp_temp([p for p in points if p.freq_hz == f1], temp_K)
    if abs(f1 - f0) < 1e-12:
        return e0
    t = float(np.clip((freq_hz - f0) / (f1 - f0), 0.0, 1.0))
    return (1.0 - t) * e0 + t * e1


def _interp_moisture(eps_by_m: list[tuple[float, complex]], moisture_wt_percent: float) -> complex:
    """Linear interpolation in moisture between measured levels."""
    if not eps_by_m:
        raise ValueError("empty moisture table")
    if len(eps_by_m) == 1:
        return eps_by_m[0][1]
    levels = sorted(eps_by_m, key=lambda x: x[0])
    ms = [m for m, _ in levels]
    if moisture_wt_percent <= ms[0]:
        return levels[0][1]
    if moisture_wt_percent >= ms[-1]:
        return levels[-1][1]
    for (m0, e0), (m1, e1) in zip(levels[:-1], levels[1:], strict=False):
        if m0 <= moisture_wt_percent <= m1:
            if abs(m1 - m0) < 1e-12:
                return e0
            t = (moisture_wt_percent - m0) / (m1 - m0)
            return (1.0 - t) * e0 + t * e1
    return levels[-1][1]


def _interp_temp(points: list[MeasuredDielectricPoint], temp_K: float) -> complex:
    if not points:
        raise ValueError("no points for frequency slice")
    pts = sorted(points, key=lambda p: p.temp_K)
    temps = np.array([p.temp_K for p in pts], dtype=float)
    reals = np.array([p.eps_real for p in pts], dtype=float)
    imags = np.array([p.eps_imag for p in pts], dtype=float)
    er = float(np.interp(temp_K, temps, reals))
    ei = float(np.interp(temp_K, temps, imags))
    return complex(er, ei)


def _bracket(x: float, xs: list[float]) -> tuple[float, float]:
    """Return (lo, hi) from sorted `xs` bracketing x, clamped to endpoints."""
    if not xs:
        raise ValueError("empty bracket set")
    if x <= xs[0]:
        return xs[0], xs[0]
    if x >= xs[-1]:
        return xs[-1], xs[-1]
    for lo, hi in zip(xs[:-1], xs[1:], strict=False):
        if lo <= x <= hi:
            return lo, hi
    return xs[-1], xs[-1]

