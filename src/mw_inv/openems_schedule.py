"""Promotion-aware openEMS case scheduling (backlog F1)."""

from __future__ import annotations

from mw_inv.design_export import DesignCase


def _case_sort_key(label: str) -> int:
    order = {"tpe_best": 0, "random_best": 1, "untuned": 99}
    if label in order:
        return order[label]
    if label.startswith("tpe_k"):
        try:
            return 10 + int(label[5:])
        except ValueError:
            return 50
    return 50


def schedule_openems_cases(
    cases: list[DesignCase],
    *,
    gate_passed: bool,
    budget: int | None = None,
    include_untuned: bool = False,
    force: bool = False,
) -> tuple[list[DesignCase], dict]:
    """Filter export cases so openEMS budget targets FDFD gate winners only.

    When the FDFD gate fails, only ``untuned`` is kept for diagnostic runs unless
    *force* is True. When it passes, ``tpe_best`` and ``tpe_k*`` top-K cases are
    kept; ``untuned`` is dropped unless explicitly requested.
    """
    meta: dict = {
        "gate_passed": gate_passed,
        "force": force,
        "budget": budget,
        "input_count": len(cases),
    }
    if not cases:
        meta["scheduled_count"] = 0
        meta["skipped_labels"] = []
        return [], meta

    if not gate_passed and not force:
        diagnostic = [c for c in cases if c.label == "untuned"]
        scheduled = diagnostic[:1]
        meta["skipped_labels"] = [c.label for c in cases if c not in scheduled]
        meta["reason"] = "fdfd_gate_failed_diagnostic_only"
    else:
        winners = [c for c in cases if c.label != "untuned" or include_untuned]
        if not include_untuned:
            winners = [c for c in winners if c.label != "untuned"]
        if not winners:
            winners = [c for c in cases if c.label == "tpe_best"] or cases[:1]
        # Prefer tpe_best first, then top-K order.
        winners.sort(key=lambda c: _case_sort_key(c.label))
        scheduled = winners
        meta["skipped_labels"] = [c.label for c in cases if c not in scheduled]
        meta["reason"] = "fdfd_gate_passed_winners"

    if budget is not None and budget > 0:
        scheduled = scheduled[:budget]

    meta["scheduled_count"] = len(scheduled)
    meta["scheduled_labels"] = [c.label for c in scheduled]
    return scheduled, meta
