from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.vna_s11 import load_s1p, s11_at_freq, summary_s11_metrics  # noqa: E402


def test_load_s1p_ri_parses_complex(tmp_path: Path) -> None:
    p = tmp_path / "t.s1p"
    p.write_text(
        "\n".join([
            "! comment",
            "# GHZ S RI R 50",
            "2.40  0.10  -0.20",
            "2.50  0.20  -0.10",
        ])
    )
    tr = load_s1p(p)
    assert tr.z0_ohm == 50.0
    assert tr.format == "RI"
    assert tr.freq_hz[0] == 2.40e9
    assert abs(tr.s11[0] - complex(0.10, -0.20)) < 1e-12


def test_s11_at_freq_interpolates(tmp_path: Path) -> None:
    p = tmp_path / "t.s1p"
    p.write_text(
        "\n".join([
            "# GHZ S RI R 50",
            "2.40  0.10  0.00",
            "2.50  0.30  0.00",
        ])
    )
    tr = load_s1p(p)
    v = s11_at_freq(tr, 2.45e9)
    assert abs(v.real - 0.20) < 1e-12
    assert abs(v.imag) < 1e-12


def test_summary_s11_metrics_band_min(tmp_path: Path) -> None:
    p = tmp_path / "t.s1p"
    p.write_text(
        "\n".join([
            "# MHZ S MA R 50",
            "2400  0.50  0",
            "2450  0.20  0",
            "2500  0.40  0",
        ])
    )
    tr = load_s1p(p)
    rep = summary_s11_metrics(tr, freq_hz=2.45e9, band_lo_hz=2.40e9, band_hi_hz=2.50e9)
    assert abs(rep["s11_mag"] - 0.20) < 1e-12
    assert abs(rep["min_s11_mag_in_band"] - 0.20) < 1e-12

