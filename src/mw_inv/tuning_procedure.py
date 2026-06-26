"""Manufacturing tuning procedure export from cavity params (backlog H0 partial)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mw_inv.geometry import CavityParams


@dataclass(frozen=True)
class TuningStep:
    order: int
    action: str
    parameter: str
    nominal: str
    tolerance: str
    acceptance: str

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "action": self.action,
            "parameter": self.parameter,
            "nominal": self.nominal,
            "tolerance": self.tolerance,
            "acceptance": self.acceptance,
        }


@dataclass(frozen=True)
class TuningProcedure:
    label: str
    cavity_Lx_m: float
    cavity_Ly_m: float
    steps: tuple[TuningStep, ...]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "cavity_Lx_m": self.cavity_Lx_m,
            "cavity_Ly_m": self.cavity_Ly_m,
            "steps": [s.to_dict() for s in self.steps],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Tuning procedure — {self.label}",
            "",
            f"Cavity: {self.cavity_Lx_m:.3f} × {self.cavity_Ly_m:.3f} m (2D ROM slice)",
            "",
            "| Step | Action | Nominal | Tolerance | Acceptance |",
            "|------|--------|---------|-----------|------------|",
        ]
        for s in self.steps:
            lines.append(
                f"| {s.order} | {s.action} | {s.nominal} | {s.tolerance} | {s.acceptance} |"
            )
        return "\n".join(lines) + "\n"


def build_tuning_procedure(
    params: CavityParams,
    *,
    label: str = "design",
    Lx: float = 0.36,
    Ly: float = 0.36,
    placement_tol_frac: float = 0.02,
    angle_tol_deg: float = 5.0,
) -> TuningProcedure:
    """Nominal build + acceptance sequence for a manufacturable cavity family."""
    steps = [
        TuningStep(
            1,
            "Mount coax/waveguide feed",
            "feed_wall + along-wall position",
            f"{params.feed_wall} @ {params.feed_along_frac:.3f}",
            f"± {placement_tol_frac:.2f} (frac)",
            "|S11| unloaded ≤ 0.92 @ 2.45 GHz (VNA)",
        ),
        TuningStep(
            2,
            "Set stub depth",
            "stub_depth_frac",
            f"{params.stub_depth_frac:.3f}",
            f"± {placement_tol_frac:.2f} (frac of {params.feed_wall or 'wall'} span)",
            "Loaded |S11| improves vs unloaded; coupling_eff ≥ 0.25 (FDFD/openEMS pre-check)",
        ),
        TuningStep(
            3,
            "Install tuning plate",
            "plate centre + length + angle",
            (
                f"({params.plate_cx_frac:.3f}, {params.plate_cy_frac:.3f}) L={params.plate_len_frac:.3f} "
                f"θ={params.plate_angle_deg:.1f}°"
            ),
            f"± {angle_tol_deg:.0f}° placement; ± {placement_tol_frac:.2f} (frac) centre",
            "Selectivity gain vs no-plate baseline; pec_loss_fraction < 0.15 (Dirichlet metal)",
        ),
        TuningStep(
            4,
            "Position ore bed",
            "charge centre",
            f"({params.charge_cx_frac:.3f}, {params.charge_cy_frac:.3f})",
            f"± {placement_tol_frac:.2f} (frac)",
            "Repeat loaded S11 + phantom ΔT rank if bench data available",
        ),
        TuningStep(
            5,
            "Verify magnetron band",
            "freq_hz",
            f"{params.freq_hz/1e9:.4f} GHz",
            "ISM 2.40–2.50 GHz",
            "Min selectivity over ±50 MHz stub tolerance ≥ envelope floor",
        ),
    ]
    return TuningProcedure(label=label, cavity_Lx_m=Lx, cavity_Ly_m=Ly, steps=tuple(steps))


def write_tuning_procedure(
    out_dir: Path | str,
    params: CavityParams,
    *,
    label: str = "design",
    Lx: float = 0.36,
    Ly: float = 0.36,
) -> tuple[Path, Path]:
    """Write JSON + Markdown tuning procedure beside export bundles."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = build_tuning_procedure(params, label=label, Lx=Lx, Ly=Ly)
    json_path = out_dir / f"{label}_tuning_procedure.json"
    md_path = out_dir / f"{label}_tuning_procedure.md"
    json_path.write_text(json.dumps(proc.to_dict(), indent=2))
    md_path.write_text(proc.to_markdown())
    return json_path, md_path
