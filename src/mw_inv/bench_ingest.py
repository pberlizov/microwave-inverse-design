"""Validate bench JSON inputs (probe ε and lab ΔT records)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.phantom_data import PHANTOM_RECIPES


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def validate_lab_measurements(path: Path | str) -> list[ValidationIssue]:
    """Schema check for lab_measurements.json (list or {measurements: [...]})."""
    p = Path(path)
    if not p.is_file():
        return [ValidationIssue(str(p), "file not found")]
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        return [ValidationIssue(str(p), f"invalid JSON: {exc}")]

    rows = payload if isinstance(payload, list) else payload.get("measurements")
    if not isinstance(rows, list) or not rows:
        return [ValidationIssue(str(p), "expected non-empty list or measurements[]")]

    issues: list[ValidationIssue] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(ValidationIssue(f"[{i}]", "row must be an object"))
            continue
        phantom = row.get("phantom")
        if not phantom:
            issues.append(ValidationIssue(f"[{i}].phantom", "required"))
        elif phantom not in PHANTOM_RECIPES:
            issues.append(ValidationIssue(
                f"[{i}].phantom",
                f"unknown phantom {phantom!r}; known: {sorted(PHANTOM_RECIPES)}",
            ))
        if "measured_delta_T_K" not in row:
            issues.append(ValidationIssue(f"[{i}].measured_delta_T_K", "required"))
        else:
            try:
                float(row["measured_delta_T_K"])
            except (TypeError, ValueError):
                issues.append(ValidationIssue(f"[{i}].measured_delta_T_K", "must be numeric"))
        if row.get("untuned_measured_delta_T_K") is not None:
            try:
                float(row["untuned_measured_delta_T_K"])
            except (TypeError, ValueError):
                issues.append(ValidationIssue(f"[{i}].untuned_measured_delta_T_K", "must be numeric"))
    return issues
