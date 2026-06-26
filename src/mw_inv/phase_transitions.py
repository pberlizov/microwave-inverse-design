"""Simple phase/chemistry rules for ε(T) evolution during heating (backlog E2).

These are **qualitative hooks** — not a full kinetic model. They switch the mineral
key used in ``build_scene_at_T`` when local temperature crosses literature bands.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pyrite oxidative decomposition onset (literature band ~500–650 °C).
PYRITE_TO_PYRRHOTITE_T_K = 600.0

# Magnetite Curie transition (~580 °C) — μ collapses via MineralModel μ(T); no mineral-key swap.
MAGNETITE_CURIE_T_K = 853.0

# Free moisture evaporation band (gangue ε shift handled via measured moisture tables when present).
MOISTURE_EVAPORATION_T_K = 373.0


@dataclass(frozen=True)
class PhaseRule:
    source_mineral: str
    product_mineral: str
    threshold_T_K: float
    description: str = ""


DEFAULT_PHASE_RULES: tuple[PhaseRule, ...] = (
    PhaseRule(
        "pyrite",
        "pyrrhotite",
        PYRITE_TO_PYRRHOTITE_T_K,
        "disseminated pyrite → pyrrhotite proxy above ~600 K (oxidation onset)",
    ),
)


def mineral_key_at_T(
    mineral: str,
    temp_K: float,
    *,
    rules: tuple[PhaseRule, ...] = DEFAULT_PHASE_RULES,
) -> str:
    """Return effective mineral catalog key at *temp_K* after phase rules."""
    key = mineral
    for rule in rules:
        if key == rule.source_mineral and temp_K >= rule.threshold_T_K:
            key = rule.product_mineral
    return key


def rules_for_pair(pair_label: str) -> tuple[PhaseRule, ...]:
    """Pair-specific phase rules (extend as deposit chemistry is validated)."""
    if pair_label == "pyrite_in_calcite":
        return DEFAULT_PHASE_RULES
    # magnetite_in_quartz: Curie loss on μ(T) is in dielectric_data MAGNETITE anchors.
    return ()
