from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mw_inv.rf_port_report import build_port_report  # noqa: E402


def test_rf_port_report_includes_openems_compare(tmp_path: Path) -> None:
    unloaded = tmp_path / "unloaded.s1p"
    unloaded.write_text(
        "\n".join([
            "# GHZ S RI R 50",
            "2.45  0.10  0.00",
        ])
    )
    port_metrics = tmp_path / "port_metrics.json"
    port_metrics.write_text(json.dumps({
        "s11_mag": 0.20,
        "coupling_eff": 0.96,
        "freq_hz": 2.45e9,
    }))
    rep = build_port_report(
        unloaded_s1p=unloaded,
        openems_port_metrics=port_metrics,
        freq_hz=2.45e9,
    ).to_dict()
    assert "openems_compare" in rep
    assert abs(rep["openems_compare"]["delta_s11_mag"] - (0.10 - 0.20)) < 1e-12

