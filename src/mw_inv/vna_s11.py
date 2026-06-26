"""VNA S11 ingestion (Touchstone .s1p) for bench calibration.

Stage-A bench RF asks: does the built applicator/port remain matched when the charge is
loaded? This module parses Touchstone v1 ``.s1p`` files and extracts |S11| metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "S1PTrace",
    "load_s1p",
    "s11_at_freq",
    "summary_s11_metrics",
]


_UNIT_SCALE = {
    "HZ": 1.0,
    "KHZ": 1e3,
    "MHZ": 1e6,
    "GHZ": 1e9,
}


@dataclass(frozen=True)
class S1PTrace:
    path: str
    freq_hz: np.ndarray
    s11: np.ndarray  # complex
    z0_ohm: float = 50.0
    format: str = "RI"  # RI | MA | DB

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "n_points": int(self.freq_hz.size),
            "freq_hz_min": float(self.freq_hz.min()) if self.freq_hz.size else None,
            "freq_hz_max": float(self.freq_hz.max()) if self.freq_hz.size else None,
            "z0_ohm": float(self.z0_ohm),
            "format": self.format,
        }


def _parse_header(line: str) -> tuple[float, str, float]:
    # Touchstone v1: "# <freq_unit> <parameter> <format> R <z0>"
    toks = [t.upper() for t in line.strip().split()]
    if len(toks) < 4 or toks[0] != "#":
        raise ValueError("invalid Touchstone header")
    unit = toks[1]
    if unit not in _UNIT_SCALE:
        raise ValueError(f"unsupported frequency unit {unit!r}")
    if toks[2] != "S":
        raise ValueError("only S-parameter files supported")
    fmt = toks[3]
    if fmt not in ("RI", "MA", "DB"):
        raise ValueError(f"unsupported data format {fmt!r}")
    z0 = 50.0
    if "R" in toks:
        i = toks.index("R")
        if i + 1 < len(toks):
            try:
                z0 = float(toks[i + 1])
            except ValueError:
                pass
    return _UNIT_SCALE[unit], fmt, z0


def load_s1p(path: Path | str) -> S1PTrace:
    """Parse a Touchstone v1 ``.s1p`` file into frequency and complex S11 arrays."""
    path = Path(path)
    text = path.read_text(errors="replace").splitlines()
    scale = 1e9  # default GHz if header absent
    fmt = "MA"   # common default if header absent
    z0 = 50.0

    freqs: list[float] = []
    s11: list[complex] = []

    for raw in text:
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if line.startswith("#"):
            scale, fmt, z0 = _parse_header(line)
            continue
        # Data row: freq  a  b
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            f = float(parts[0]) * scale
            a = float(parts[1])
            b = float(parts[2])
        except ValueError:
            continue
        if fmt == "RI":
            v = complex(a, b)
        elif fmt == "MA":
            mag = a
            ang = math.radians(b)
            v = complex(mag * math.cos(ang), mag * math.sin(ang))
        else:  # DB
            mag = 10 ** (a / 20.0)
            ang = math.radians(b)
            v = complex(mag * math.cos(ang), mag * math.sin(ang))
        freqs.append(f)
        s11.append(v)

    if not freqs:
        raise ValueError(f"no S11 points parsed from {path}")

    f_arr = np.asarray(freqs, dtype=float)
    s_arr = np.asarray(s11, dtype=complex)
    order = np.argsort(f_arr)
    f_arr = f_arr[order]
    s_arr = s_arr[order]
    return S1PTrace(path=str(path), freq_hz=f_arr, s11=s_arr, z0_ohm=z0, format=fmt)


def s11_at_freq(trace: S1PTrace, freq_hz: float) -> complex:
    """Linear interpolation of complex S11 at a given frequency (Hz)."""
    f = trace.freq_hz
    s = trace.s11
    if freq_hz <= float(f[0]):
        return complex(s[0])
    if freq_hz >= float(f[-1]):
        return complex(s[-1])
    re = float(np.interp(freq_hz, f, s.real))
    im = float(np.interp(freq_hz, f, s.imag))
    return complex(re, im)


def _min_mag_in_band(trace: S1PTrace, lo_hz: float, hi_hz: float) -> tuple[float, float]:
    lo = min(lo_hz, hi_hz)
    hi = max(lo_hz, hi_hz)
    mask = (trace.freq_hz >= lo) & (trace.freq_hz <= hi)
    if not bool(mask.any()):
        # fall back to endpoints
        mags = np.abs(trace.s11)
        i = int(np.argmin(mags))
        return float(trace.freq_hz[i]), float(mags[i])
    mags = np.abs(trace.s11[mask])
    idxs = np.flatnonzero(mask)
    j = int(idxs[int(np.argmin(mags))])
    return float(trace.freq_hz[j]), float(abs(trace.s11[j]))


def summary_s11_metrics(
    trace: S1PTrace,
    *,
    freq_hz: float = 2.45e9,
    band_lo_hz: float | None = None,
    band_hi_hz: float | None = None,
) -> dict:
    """Compact metrics for run manifests (keeps raw .s1p as source of truth)."""
    s = s11_at_freq(trace, freq_hz)
    mag = abs(s)
    db = 20.0 * math.log10(max(mag, 1e-12))
    ang_deg = math.degrees(math.atan2(s.imag, s.real))

    out = {
        **trace.to_dict(),
        "freq_eval_hz": float(freq_hz),
        "s11_mag": float(mag),
        "s11_db": float(db),
        "s11_ang_deg": float(ang_deg),
    }
    if band_lo_hz is not None and band_hi_hz is not None:
        f_min, mag_min = _min_mag_in_band(trace, band_lo_hz, band_hi_hz)
        out["band_lo_hz"] = float(min(band_lo_hz, band_hi_hz))
        out["band_hi_hz"] = float(max(band_lo_hz, band_hi_hz))
        out["min_s11_mag_in_band"] = float(mag_min)
        out["min_s11_freq_hz_in_band"] = float(f_min)
    return out

