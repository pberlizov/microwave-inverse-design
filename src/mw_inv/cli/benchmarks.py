from __future__ import annotations

import argparse

from mw_inv.benchmarks import ALL_TIERS, run_benchmarks, write_report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", nargs="*", choices=ALL_TIERS, default=None)
    ap.add_argument("--out", default="data/benchmark_report.json")
    args = ap.parse_args(argv)

    report = run_benchmarks(args.tier)
    write_report(args.out, report)

    print("=== Literature benchmarks ===")
    for tier in (args.tier or ALL_TIERS):
        checks = [r for r in report.results if r.tier == tier]
        if not checks:
            continue
        ok = all(c.passed for c in checks)
        print(f"  [{'PASS' if ok else 'FAIL'}] {tier} ({sum(c.passed for c in checks)}/{len(checks)})")
        for c in checks:
            mark = "ok" if c.passed else "XX"
            print(f"      {mark} {c.name}: {c.detail}")
    d = report.to_dict()
    print(f"  overall: {'PASS' if report.passed else 'FAIL'} ({d['n_passed']}/{d['n_checks']})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()

