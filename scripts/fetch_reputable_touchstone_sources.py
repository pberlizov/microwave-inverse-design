"""Fetch reputable, permissively-licensed Touchstone examples into data/third_party/.

Sources:
- scikit-rf (BSD-3): skrf/data/*.s?p
- NIST MD_2-3404 (NIST Open License): V2.zip (Touchstone .s2p)

This is intended for exercising Stage-A RF ingest tooling (VNA S11 parsing) without
needing local hardware traces.
"""

from __future__ import annotations

import json
import time
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "third_party" / "touchstone"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mw-inv-fetch"})
            with urllib.request.urlopen(req, timeout=120) as r:
                path.write_bytes(r.read())
            return
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"failed to download {url} -> {path}") from last_err


def fetch_scikit_rf(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        repo = td / "scikit-rf"
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/scikit-rf/scikit-rf.git", str(repo)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        (dest / "UPSTREAM_SHA.txt").write_text(sha + "\n")
        shutil.copy2(repo / "LICENSE.txt", dest / "LICENSE.txt")

        data_dir = repo / "skrf" / "data"
        copied = 0
        for p in sorted(data_dir.glob("*.s?p")):
            shutil.copy2(p, dest / p.name)
            copied += 1
        (dest / "FETCH_REPORT.json").write_text(json.dumps(
            {"source": "scikit-rf", "sha": sha, "n_touchstone_files": copied},
            indent=2,
        ))


def fetch_nist_md_2_3404(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    # Dataset zip is stable at this URL (V2).
    zip_url = "https://data.nist.gov/od/ds/mds2-3404/V2.zip"
    lic_url = "https://www.nist.gov/open/license"

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        zpath = td / "V2.zip"
        last_err: Exception | None = None
        for attempt in range(1, 6):
            try:
                _download(zip_url, zpath)
                with zipfile.ZipFile(zpath) as zf:
                    names = [n for n in zf.namelist() if n.lower().endswith(".s2p")]
                    for n in sorted(names):
                        out = dest / Path(n).name
                        out.write_bytes(zf.read(n))
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                for p in dest.glob("*.s2p"):
                    p.unlink(missing_ok=True)
                time.sleep(0.5 * attempt)
        else:  # pragma: no cover
            raise RuntimeError("failed to fetch/extract NIST dataset zip") from last_err
        _download(lic_url, dest / "LICENSE-NIST-OPEN.txt")

        (dest / "FETCH_REPORT.json").write_text(json.dumps(
            {"source": "nist_md_2_3404", "zip_url": zip_url, "n_s2p_files": len(names)},
            indent=2,
        ))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scikit-rf").mkdir(parents=True, exist_ok=True)
    (OUT / "nist_md_2_3404").mkdir(parents=True, exist_ok=True)

    fetch_scikit_rf(OUT / "scikit-rf")
    fetch_nist_md_2_3404(OUT / "nist_md_2_3404")
    print(f"fetched into {OUT}")


if __name__ == "__main__":
    sys.exit(main())
