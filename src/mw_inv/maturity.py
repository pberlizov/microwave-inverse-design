"""Component maturity labels — what is production-ready vs stub/WIP.

The thin-slice core (2D FDFD + search + cited mineral materials) is intentionally
*experimental* but exercised by tests.  Later roadmap steps vary from *experimental*
(simplified but runnable) to *stub* (scaffolding only).  Scripts should call
``warn_if_below()`` so users see the gap at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import warnings


class Maturity(str, Enum):
    """Increasing honesty about implementation depth."""

    CORE = "core"              # main thin-slice loop; tested, used in README results
    EXPERIMENTAL = "experimental"  # runnable simplification — do not over-interpret
    WIP = "wip"                # partial implementation; API may change
    STUB = "stub"              # placeholder / checklist only — not a solver port


@dataclass
class ComponentInfo:
    name: str
    maturity: Maturity
    summary: str
    gaps: tuple[str, ...] = ()


# fmt: off
COMPONENTS: dict[str, ComponentInfo] = {
    "fdfd_2d": ComponentInfo(
        "fdfd_2d", Maturity.CORE,
        "2D TM Helmholtz FDFD with Dirichlet PEC walls and point current source.",
        ("Not FDTD; not 3D; no S-parameters or matched ports.",),
    ),
    "materials_mineral": ComponentInfo(
        "materials_mineral", Maturity.CORE,
        "Cited ε, μ, ε(T) anchor tables for pyrite/calcite and magnetite/quartz.",
        ("Not measured ore from a named deposit.",),
    ),
    "search_manufacturable": ComponentInfo(
        "search_manufacturable", Maturity.EXPERIMENTAL,
        "Optuna over wall feed, stub, plate, bed position (FDFD FOM).",
        (
            "Feed is a grid-node source, not a coax/waveguide port BC.",
            "Internal metal is Im(eps)→∞, not explicit PEC surfaces.",
        ),
    ),
    "thermal_coupled_2d": ComponentInfo(
        "thermal_coupled_2d", Maturity.EXPERIMENTAL,
        "Quasi-steady 2D diffusion + ε(T) feedback; explicit transient optional.",
        ("Representative k, ρcp, volumetric h — not measured ore.", "No fracture/stress FOM."),
    ),
    "ensemble_layouts": ComponentInfo(
        "ensemble_layouts", Maturity.EXPERIMENTAL,
        "Random non-overlapping grains in a static bed; mean/min FOM over layouts.",
        ("Still static 2D bed — not discrete moving particles or fluidized bed.",),
    ),
    "freq_robust_search": ComponentInfo(
        "freq_robust_search", Maturity.EXPERIMENTAL,
        "Score geometry over 2.40–2.50 GHz sample points (freq not a tuned knob).",
        ("Monochromatic FDFD per frequency — not broadband FDTD.",),
    ),
    "meep_2d_crosscheck": ComponentInfo(
        "meep_2d_crosscheck", Maturity.EXPERIMENTAL,
        "Optional MEEP 2D TM cross-check vs FDFD selectivity.",
        ("Same point source; convergence not auto-verified.",),
    ),
    "meep_3d_extrusion": ComponentInfo(
        "meep_3d_extrusion", Maturity.WIP,
        "Legacy MEEP 3D with 2D ε map extruded in z (quasi-3D).",
        ("Superseded by meep_3d_primitive where possible.",),
    ),
    "meep_3d_primitive": ComponentInfo(
        "meep_3d_primitive", Maturity.EXPERIMENTAL,
        "MEEP 3D FDTD with explicit gangue box, grain cylinders, PEC plate.",
        (
            "Point Ez source — not matched coax port.",
            "Requires MEEP; convergence not auto-checked.",
        ),
    ),
    "openems_port": ComponentInfo(
        "openems_port", Maturity.EXPERIMENTAL,
        "Runnable Octave/openEMS script: CSXCAD geometry, lumped port, field dump, selectivity post-process.",
        (
            "Requires openEMS + CSXCAD installed locally.",
            "Post-process assumes HDF5 dump layout; validate on your openEMS version.",
            "Not wired to Python — run Octave manually.",
        ),
    ),
    "phantom_lab": ComponentInfo(
        "phantom_lab", Maturity.EXPERIMENTAL,
        "Saline gel recipes (Gabriel-scaled ε), FDFD+thermal ΔT predictions, lab JSON compare.",
        (
            "Anchor ε not measured for your gel batch — calibrate with probe first.",
            "No automated lab data ingest hardware; compare via JSON file.",
        ),
    ),
    "solver_triangulation": ComponentInfo(
        "solver_triangulation", Maturity.EXPERIMENTAL,
        "FDFD vs MEEP vs openEMS dump comparison on canonical search designs.",
        (
            "openEMS compare requires manual Octave run + h5py dump ingest.",
            "Ranking agreement is necessary not sufficient for production sign-off.",
        ),
    ),
    "design_export": ComponentInfo(
        "design_export", Maturity.EXPERIMENTAL,
        "Bundle optimised params → openEMS .m + FDFD .npz + manifest for bench/FDTD.",
        ("Manifest FDFD selectivity is 2D reference — validate in 3D FDTD.",),
    ),
    "validation_gate": ComponentInfo(
        "validation_gate", Maturity.EXPERIMENTAL,
        "Pass/fail gate: FDFD improvement + MEEP/openEMS rank/error checks on pyrite_in_calcite.",
        ("Gate passes on FDFD alone if MEEP/openEMS not run.",),
    ),
    "stress_fom": ComponentInfo(
        "stress_fom", Maturity.EXPERIMENTAL,
        "Interface thermoelastic stress proxy (Salsman-style) + grain-size penalty.",
        ("Not full FE solid mechanics — 2D rim stress estimate.",),
    ),
    "ore_profiles": ComponentInfo(
        "ore_profiles", Maturity.CORE,
        "QEMSCAN JSON ingest, HMAP heating classes, Bruggeman ε, auto pair selection.",
        ("Matrix Bruggeman mixing of non-HMAP gangue still deferred.",),
    ),
    "ore_ingest": ComponentInfo(
        "ore_ingest", Maturity.CORE,
        "load_ore_profile() + ingest_ore_profile.py for deposit modal analysis JSON.",
        (),
    ),
    "phantom_calibration": ComponentInfo(
        "phantom_calibration", Maturity.EXPERIMENTAL,
        "Probe-measured ε ingest + anchor drift report for gel batches.",
        ("Requires bench probe data in measured_eps.json.",),
    ),
    "literature_benchmarks": ComponentInfo(
        "literature_benchmarks", Maturity.CORE,
        "Regression vs published ε, Goldbaum HMAP classes, phantom/stress/solver tiers.",
        ("Forward-model grounding only — not inverse-design ground truth.",),
    ),
    "mineral_catalog": ComponentInfo(
        "mineral_catalog", Maturity.CORE,
        "14-mineral ε(T) library: 9 Goldbaum HMAPs + gangue silicates/carbonates @ 2.45 GHz.",
        ("Scene-scale disseminated anchors — not deposit-specific QEMSCAN.",),
    ),
    "design_evaluator": ComponentInfo(
        "design_evaluator", Maturity.CORE,
        "Unified EM / thermal / stress FOM evaluation + composite presets for search.",
        ("Composite weights are illustrative — tune per deposit.",),
    ),
    "promotion_tiers": ComponentInfo(
        "promotion_tiers", Maturity.CORE,
        "literature_grounded → fdfd_optimised → solver_triangulated → bench_calibrated.",
        ("solver_triangulated needs MEEP/openEMS runs; bench needs measured_eps.json.",),
    ),
    "run_manifest": ComponentInfo(
        "run_manifest", Maturity.CORE,
        "Single JSON manifest per pipeline run (search, gate, export, promotion tier).",
        (),
    ),
    "field_search_tuner": ComponentInfo(
        "field_search_tuner", Maturity.EXPERIMENTAL,
        "16-cell lossless ε tuner — non-physical upper bound (deprecated path).",
        ("Not manufacturable.",),
    ),
}
# fmt: on


def get(name: str) -> ComponentInfo:
    if name not in COMPONENTS:
        raise KeyError(f"unknown component {name!r}; known: {sorted(COMPONENTS)}")
    return COMPONENTS[name]


def warn_if_below(name: str, minimum: Maturity = Maturity.EXPERIMENTAL) -> None:
    """Emit ``UserWarning`` when component maturity is stub/WIP below *minimum*."""
    info = get(name)
    order = (Maturity.STUB, Maturity.WIP, Maturity.EXPERIMENTAL, Maturity.CORE)
    if order.index(info.maturity) < order.index(minimum):
        gaps = " ".join(info.gaps)
        warnings.warn(
            f"[{info.maturity.value.upper()}] {info.name}: {info.summary} "
            f"Gaps: {gaps}",
            stacklevel=2,
        )


def status_dict(name: str) -> dict[str, str | list[str]]:
    """JSON-serialisable maturity block for script outputs."""
    info = get(name)
    return {
        "maturity": info.maturity.value,
        "summary": info.summary,
        "gaps": list(info.gaps),
    }


def banner(name: str) -> str:
    info = get(name)
    return f"*** {info.maturity.value.upper()}: {info.name} — {info.summary} ***"
