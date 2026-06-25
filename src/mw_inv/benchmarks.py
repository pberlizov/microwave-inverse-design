"""Public-literature benchmark harness for mw_inv.

Loads curated reference data from ``data/benchmarks/*.json`` and checks that
materials, ore heating classes, phantom recipes, stress trends, and internal
forward-model references remain aligned with published mining-microwave science.

There is no standard inverse-design benchmark suite in the literature — these
tiers validate *forward-model grounding*, not geometry-optimisation rankings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from mw_inv.dielectric_data import MINERAL_MODELS
from mw_inv.materials import Materials, PAIRS
from mw_inv.ore_profiles import HEATING_CLASSES, HMAP_MINERALS, OreComposition, ORE_PROFILES
from mw_inv.phantom_data import saline_eps
from mw_inv.stress import ThermoelasticProps, grain_size_penalty_factor

_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"

_MINERAL_MODELS = MINERAL_MODELS

ALL_TIERS = (
    "dielectric",
    "heating_class",
    "phantom",
    "stress",
    "solver",
)


@dataclass
class BenchmarkResult:
    tier: str
    name: str
    passed: bool
    detail: str
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "metrics": self.metrics,
        }


@dataclass
class BenchmarkReport:
    results: list[BenchmarkResult]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def by_tier(self) -> dict[str, list[BenchmarkResult]]:
        out: dict[str, list[BenchmarkResult]] = {t: [] for t in ALL_TIERS}
        for r in self.results:
            out.setdefault(r.tier, []).append(r)
        return out

    def to_dict(self) -> dict:
        tiers = self.by_tier()
        return {
            "passed": self.passed,
            "n_checks": len(self.results),
            "n_passed": sum(r.passed for r in self.results),
            "tiers": {
                t: {
                    "passed": all(x.passed for x in rs) if rs else True,
                    "checks": [x.to_dict() for x in rs],
                }
                for t, rs in tiers.items()
                if rs
            },
        }


def benchmark_dir() -> Path:
    return _BENCHMARK_DIR


def load_benchmark(name: str) -> dict:
    path = _BENCHMARK_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark file {path}")
    return json.loads(path.read_text())


def _rel_err(ref: float, val: float) -> float:
    return abs(val - ref) / max(abs(ref), 1e-12)


def _eps_from_entry(entry: dict) -> complex:
    if "pair_label" in entry:
        mats = Materials.from_pair(entry["pair_label"])
        role = entry.get("role", "target")
        return mats.target if role == "target" else mats.gangue
    model = _MINERAL_MODELS[entry["mineral_model"]]
    return model.eps(float(entry.get("T_K", 298.0)), float(entry.get("freq_hz", 2.45e9)))


def check_literature_dielectric(data: dict | None = None) -> list[BenchmarkResult]:
    data = data or load_benchmark("literature_dielectric")
    results: list[BenchmarkResult] = []

    for entry in data["entries"]:
        eps = _eps_from_entry(entry)
        if entry.get("check") == "eps_imag_below_upper":
            upper = float(entry["reference_eps_imag_upper"])
            ok = eps.imag <= upper
            results.append(BenchmarkResult(
                tier="dielectric",
                name=entry["id"],
                passed=ok,
                detail=f"ε″={eps.imag:.3f} ≤ bulk upper {upper:.3f}",
                metrics={"eps_imag": eps.imag},
            ))
            continue
        rr = float(entry["reference_eps_real"])
        ri = float(entry["reference_eps_imag"])
        rtol_r = float(entry.get("rtol_real", 0.05))
        rtol_i = float(entry.get("rtol_imag", 0.2))
        ok_r = _rel_err(rr, eps.real) <= rtol_r
        ok_i = _rel_err(ri, eps.imag) <= rtol_i
        ok = ok_r and ok_i
        results.append(BenchmarkResult(
            tier="dielectric",
            name=entry["id"],
            passed=ok,
            detail=(
                f"ε={eps.real:.3f}-j{eps.imag:.4f} vs ref {rr:.3f}-j{ri:.4f} "
                f"(rel err { _rel_err(rr, eps.real):.3f}, { _rel_err(ri, eps.imag):.3f})"
            ),
            metrics={"eps_real": eps.real, "eps_imag": eps.imag, "source": entry.get("source", "")},
        ))

    for chk in data.get("eps_T_checks", []):
        model = _MINERAL_MODELS[chk["mineral_model"]]
        f = float(chk.get("freq_hz", 2.45e9))
        tc = float(chk["T_cold_K"])
        th = float(chk["T_hot_K"])
        e_c = model.eps(tc, f)
        e_h = model.eps(th, f)
        ok = True
        detail_parts = [f"298→773 ε″ {e_c.imag:.3f}→{e_h.imag:.3f}"]
        if chk.get("expect_eps_imag_increase"):
            ok = e_h.imag > e_c.imag
            detail_parts.append("expect ε″ increase")
        if "max_eps_real_delta" in chk:
            ok = ok and abs(e_h.real - e_c.real) <= float(chk["max_eps_real_delta"])
            detail_parts.append(f"|Δε′|={abs(e_h.real - e_c.real):.3f}")
        results.append(BenchmarkResult(
            tier="dielectric",
            name=chk["id"],
            passed=ok,
            detail="; ".join(detail_parts),
            metrics={"eps_cold_imag": e_c.imag, "eps_hot_imag": e_h.imag},
        ))

    for ord_chk in data.get("eps_imag_ordering", []):
        f = float(ord_chk.get("freq_hz", 2.45e9))
        names = ord_chk["minerals"]
        imags = [float(_MINERAL_MODELS[n].eps(298.0, f).imag) for n in names]
        ok = imags == sorted(imags, reverse=True)
        detail = " > ".join(f"{n}({v:.3f})" for n, v in zip(names, imags))
        results.append(BenchmarkResult(
            tier="dielectric",
            name=ord_chk["id"],
            passed=ok,
            detail=f"ε″ order @298K: {detail}",
            metrics={f"eps_imag_{n}": v for n, v in zip(names, imags)},
        ))

    return results


def check_goldbaum_heating(data: dict | None = None) -> list[BenchmarkResult]:
    data = data or load_benchmark("goldbaum_heating_classes")
    results: list[BenchmarkResult] = []

    # Registry matches benchmark file
    for c in data["classes"]:
        cid = c["id"]
        match = next((x for x in HEATING_CLASSES if x[0] == cid), None)
        ok = match is not None and match[1] == c["hmap_wt_percent_min"]
        results.append(BenchmarkResult(
            tier="heating_class",
            name=f"registry_{cid}",
            passed=ok,
            detail=f"ore_profiles.HEATING_CLASSES aligned with Goldbaum {cid}",
            metrics={"hmap_lo": c["hmap_wt_percent_min"]},
        ))

    hmap_set = set(HMAP_MINERALS)
    ref_set = set(data["hmap_minerals"])
    results.append(BenchmarkResult(
        tier="heating_class",
        name="hmap_mineral_list",
        passed=hmap_set == ref_set,
        detail=f"HMAP minerals match ({len(hmap_set)} phases)",
    ))

    for ore in data["benchmark_ores"]:
        label = ore["label"]
        if label.startswith("synthetic_"):
            comp = OreComposition(label, ore.get("fractions", {}))
        elif label in ORE_PROFILES:
            comp = ORE_PROFILES[label]
        else:
            comp = OreComposition(label, ore.get("fractions", {}))

        predicted = comp.heating_class()
        expected = ore["expected_class"]
        ok = predicted == expected
        rate = comp.predicted_heating_rate_C_per_min()
        class_row = next(c for c in data["classes"] if c["id"] == expected)
        rate_ok = rate >= class_row["rate_min_C_per_min"] * 0.85
        rate_ok = rate_ok and rate <= class_row["rate_max_C_per_min"] * 1.5
        results.append(BenchmarkResult(
            tier="heating_class",
            name=f"ore_{label}",
            passed=ok and rate_ok,
            detail=(
                f"HMAP={comp.hmap_wt_percent:.1f}% class {predicted} (expect {expected}), "
                f"pred rate {rate:.0f} C/min"
            ),
            metrics={"hmap_wt_percent": comp.hmap_wt_percent, "predicted_rate": rate},
        ))

    return results


def check_phantom_saline(data: dict | None = None) -> list[BenchmarkResult]:
    data = data or load_benchmark("phantom_saline_gabriel")
    results: list[BenchmarkResult] = []
    rtol_r = float(data.get("rtol_real", 0.01))
    rtol_i = float(data.get("rtol_imag", 0.01))

    for anchor in data["anchors"]:
        w = float(anchor["salt_wt_percent"])
        eps = saline_eps(w)
        rr = float(anchor["eps_real"])
        ri = float(anchor["eps_imag"])
        ok = _rel_err(rr, eps.real) <= rtol_r and _rel_err(ri, eps.imag) <= rtol_i
        results.append(BenchmarkResult(
            tier="phantom",
            name=f"saline_{w:g}pct",
            passed=ok,
            detail=f"saline_eps({w})={eps.real:.2f}-j{eps.imag:.2f}",
        ))

    mono = data["monotonicity"]
    e0 = saline_eps(float(mono["salt_wt_percent_lo"]))
    e3 = saline_eps(float(mono["salt_wt_percent_hi"]))
    ok = True
    if mono.get("expect_eps_real_increase"):
        ok = ok and e3.real > e0.real
    if mono.get("expect_eps_imag_increase"):
        ok = ok and e3.imag > e0.imag
    results.append(BenchmarkResult(
        tier="phantom",
        name="saline_monotonic",
        passed=ok,
        detail=f"ε 0→3 wt%: {e0.real:.1f}-j{e0.imag:.2f} → {e3.real:.1f}-j{e3.imag:.2f}",
    ))
    return results


def check_stress_qualitative(data: dict | None = None) -> list[BenchmarkResult]:
    data = data or load_benchmark("stress_qualitative")
    results: list[BenchmarkResult] = []

    gs = data["grain_size_penalty"]
    r_f = float(gs["radius_fine_m"])
    r_o = float(gs["radius_opt_m"])
    r_c = float(gs["radius_coarse_m"])
    p_f = grain_size_penalty_factor(r_f)
    p_o = grain_size_penalty_factor(r_o)
    p_c = grain_size_penalty_factor(r_c)
    ok = p_f < p_o
    if gs.get("expect_opt_ge_coarse_fraction"):
        ok = ok and p_o >= p_c * float(gs["expect_opt_ge_coarse_fraction"])
    results.append(BenchmarkResult(
        tier="stress",
        name="grain_size_penalty_order",
        passed=ok,
        detail=f"penalty fine={p_f:.3f} opt={p_o:.3f} coarse={p_c:.3f}",
        metrics={"fine": p_f, "opt": p_o, "coarse": p_c},
    ))

    te = data["thermoelastic"]
    props = ThermoelasticProps()
    ok = props.target_alpha > props.gangue_alpha
    results.append(BenchmarkResult(
        tier="stress",
        name="differential_expansion",
        passed=ok,
        detail=f"α_target={props.target_alpha:.1e} > α_gangue={props.gangue_alpha:.1e} 1/K",
    ))
    return results


def check_solver_internal(data: dict | None = None) -> list[BenchmarkResult]:
    data = data or load_benchmark("solver_internal")
    results: list[BenchmarkResult] = []

    from mw_inv.validation import cavity_resonance_peak, literature_consistency

    cr = cavity_resonance_peak()
    results.append(BenchmarkResult(
        tier="solver",
        name="empty_cavity_resonance",
        passed=cr.passed,
        detail=cr.detail,
        metrics=dict(cr.metrics),
    ))

    lit = literature_consistency()
    min_ratio = float(data["polyakova_bulk_vs_scene"]["min_bulk_to_scene_ratio"])
    ratio = float(lit.metrics.get("bulk_to_scene_ratio", 0.0))
    results.append(BenchmarkResult(
        tier="solver",
        name="polyakova_bulk_vs_disseminated",
        passed=lit.passed and ratio >= min_ratio,
        detail=lit.detail,
        metrics=dict(lit.metrics),
    ))
    return results


_TIER_FUNCS = {
    "dielectric": check_literature_dielectric,
    "heating_class": check_goldbaum_heating,
    "phantom": check_phantom_saline,
    "stress": check_stress_qualitative,
    "solver": check_solver_internal,
}


def run_benchmarks(tiers: list[str] | None = None) -> BenchmarkReport:
    """Run one or all benchmark tiers."""
    selected = tiers or list(ALL_TIERS)
    results: list[BenchmarkResult] = []
    for tier in selected:
        if tier not in _TIER_FUNCS:
            raise ValueError(f"unknown tier {tier!r}; choose from {ALL_TIERS}")
        results.extend(_TIER_FUNCS[tier]())
    return BenchmarkReport(results)


def write_report(path: Path | str, report: BenchmarkReport) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
