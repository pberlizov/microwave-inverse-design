"""One-shot bench calibration: probe ε + lab ΔT validate, model compare, gate (E0).

    python scripts/run_bench_calibration.py \\
        --phantom saline_2_vs_0.5 \\
        --measured-eps data/measured_eps.example.json \\
        --lab-measurements data/lab_measurements.example.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.bench_ingest import validate_lab_measurements  # noqa: E402
from mw_inv.phantom_calibration import evaluate_bench_gate  # noqa: E402
from mw_inv.phantom import compare_lab_measurement, load_lab_measurements, predict_lab_outcome  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Bench calibration orchestrator (E0)")
    ap.add_argument("--phantom", required=True)
    ap.add_argument("--measured-eps", required=True)
    ap.add_argument("--lab-measurements", default=None)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7701)
    ap.add_argument("--out", type=str, default="data/bench_calibration_report.json")
    ap.add_argument("--enforce", action="store_true", help="exit non-zero if gate fails")
    args = ap.parse_args()

    eps_path = Path(args.measured_eps)
    if args.lab_measurements:
        issues = validate_lab_measurements(args.lab_measurements)
        if issues:
            print("Lab JSON validation failed:")
            for i in issues:
                print(f"  {i.path}: {i.message}")
            raise SystemExit(2)

    report = evaluate_bench_gate(
        args.phantom,
        eps_path,
        args.lab_measurements,
        bench_grid=args.grid,
        bench_trials=args.trials,
        bench_seed=args.seed,
    )

    payload: dict = {"gate": report.to_dict(), "phantom": args.phantom}
    if args.lab_measurements:
        from mw_inv.fdfd import Grid

        grid = Grid(nx=args.grid, ny=args.grid, Lx=0.36, Ly=0.36)
        pred = predict_lab_outcome(
            args.phantom,
            grid,
            n_opt_trials=args.trials,
            seed=args.seed,
            measured_eps_path=eps_path,
        )
        comps = []
        for row in load_lab_measurements(args.lab_measurements):
            if row.get("phantom") != args.phantom:
                continue
            comps.append(
                compare_lab_measurement(
                    pred,
                    float(row["measured_delta_T_K"]),
                    row.get("measured_selectivity"),
                    untuned_measured_delta_T_K=row.get("untuned_measured_delta_T_K"),
                ).to_dict()
            )
        payload["prediction"] = pred.to_dict()
        payload["comparisons"] = comps

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    status = "PASS" if report.passed else "FAIL"
    strict = all(c.passed for c in report.checks)
    print(f"=== Bench calibration ({args.phantom}) ===")
    print(f"  gate (promotion): {status}")
    print(f"  strict (all checks): {'PASS' if strict else 'FAIL'}")
    for check in report.checks:
        mark = "ok" if check.passed else "FAIL"
        print(f"    [{mark}] {check.name}: {check.detail}")
    print(f"  wrote {out}")
    if args.enforce and not report.passed:
        raise SystemExit(3)
    if args.enforce and not strict:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
