"""Provenance helpers for reproducible runs.

Industry readiness requires that every run artifact records:
- code identity (git commit + dirty state when available)
- runtime (Python version, platform)
- dependency versions (numpy/scipy/optuna/mw-inv)

These functions are best-effort and never hard-fail a run.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def runtime_info() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def package_versions(names: tuple[str, ...] = ("mw-inv", "numpy", "scipy", "optuna")) -> dict[str, str | None]:
    try:
        from importlib.metadata import version
    except Exception:  # pragma: no cover
        return {n: None for n in names}

    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = version(n)
        except Exception:
            out[n] = None
    return out


def _run_git(args: list[str], *, cwd: Path) -> str | None:
    try:
        p = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            return None
        return p.stdout.strip()
    except Exception:
        return None


def git_info(cwd: Path | str = ".") -> dict[str, str | bool | None]:
    """Return git commit metadata if `cwd` is inside a git repo."""
    root = Path(cwd)
    inside = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=root)
    if inside != "true":
        return {"is_repo": False, "commit": None, "branch": None, "dirty": None}

    commit = _run_git(["rev-parse", "HEAD"], cwd=root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = _run_git(["status", "--porcelain"], cwd=root)
    dirty = None if status is None else (len(status) > 0)
    return {"is_repo": True, "commit": commit, "branch": branch, "dirty": dirty}


def default_provenance(cwd: Path | str = ".") -> dict:
    return {
        "runtime": runtime_info(),
        "packages": package_versions(),
        "git": git_info(cwd),
    }

