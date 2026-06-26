from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.measured_dielectrics import load_measured_dielectrics, validate_library  # noqa: E402


def test_measured_dielectrics_interp_temp_and_freq(tmp_path: Path) -> None:
    path = tmp_path / "eps.json"
    payload = {
        "description": "synthetic",
        "phases": {
            "target": [
                {"temp_K": 300, "freq_hz": 2.0e9, "eps_real": 10.0, "eps_imag": 1.0},
                {"temp_K": 500, "freq_hz": 2.0e9, "eps_real": 14.0, "eps_imag": 2.0},
                {"temp_K": 300, "freq_hz": 3.0e9, "eps_real": 8.0, "eps_imag": 0.5},
                {"temp_K": 500, "freq_hz": 3.0e9, "eps_real": 12.0, "eps_imag": 1.5},
            ],
        },
    }
    path.write_text(json.dumps(payload))
    lib = load_measured_dielectrics(path)
    assert not validate_library(lib)

    # Temp interpolation at 2 GHz: halfway 300->500 should be 12-j1.5
    e = lib.eps("target", temp_K=400, freq_hz=2.0e9)
    assert abs(e.real - 12.0) < 1e-9
    assert abs(e.imag - 1.5) < 1e-9

    # Frequency interpolation at 400K between the two frequency slices:
    # at 400K: 2GHz => 12-j1.5 ; 3GHz => 10-j1.0 ; at 2.5GHz => 11-j1.25
    e2 = lib.eps("target", temp_K=400, freq_hz=2.5e9)
    assert abs(e2.real - 11.0) < 1e-9
    assert abs(e2.imag - 1.25) < 1e-9


def test_measured_dielectrics_nearest_moisture_single_level(tmp_path: Path) -> None:
    """With only two discrete moisture levels, mid-moisture is interpolated."""
    path = tmp_path / "eps.json"
    payload = {
        "phases": {
            "ore_bulk": [
                {"temp_K": 298, "freq_hz": 2.45e9, "eps_real": 5.0, "eps_imag": 0.2, "moisture_wt_percent": 0.0},
                {"temp_K": 298, "freq_hz": 2.45e9, "eps_real": 7.0, "eps_imag": 0.5, "moisture_wt_percent": 2.0},
            ],
        },
    }
    path.write_text(json.dumps(payload))
    lib = load_measured_dielectrics(path)
    e_dry = lib.eps("ore_bulk", temp_K=298, freq_hz=2.45e9, moisture_wt_percent=0.0)
    e_wet = lib.eps("ore_bulk", temp_K=298, freq_hz=2.45e9, moisture_wt_percent=2.0)
    e_mid = lib.eps("ore_bulk", temp_K=298, freq_hz=2.45e9, moisture_wt_percent=1.0)
    assert abs(e_dry.real - 5.0) < 1e-9
    assert abs(e_wet.real - 7.0) < 1e-9
    assert abs(e_mid.real - 6.0) < 1e-9

