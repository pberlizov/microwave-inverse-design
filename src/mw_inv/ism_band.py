"""ISM-band frequency constraints for magnetron-realistic evaluation (backlog C2).

Industrial magnetrons operate near 2.45 GHz with load-dependent drift; designs
should be scored over a **declared band**, not an arbitrary continuous frequency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

# Full ISM allocation commonly used in mineral MW literature.
ISM_BAND_HZ = (2.40e9, 2.50e9)
ISM_CENTER_HZ = 2.45e9


class IsmBandMode(str, Enum):
    """How frequency enters robustness evaluation."""

    FIXED = "fixed"          # single centre frequency (default 2.45 GHz)
    FULL = "full"            # entire 2.40–2.50 GHz ISM allocation
    TUNABLE = "tunable"      # ± tolerance around centre (stub/magnetron drift)


@dataclass(frozen=True)
class IsmBandConfig:
    mode: IsmBandMode = IsmBandMode.FULL
    center_hz: float = ISM_CENTER_HZ
    tolerance_mhz: float = 50.0
    n_samples: int = 5

    def freqs_hz(self) -> np.ndarray:
        n = max(int(self.n_samples), 1)
        if self.mode == IsmBandMode.FIXED:
            return np.array([float(self.center_hz)])
        if self.mode == IsmBandMode.FULL:
            lo, hi = ISM_BAND_HZ
            return np.linspace(lo, hi, n)
        half = self.tolerance_mhz * 1e6
        return np.linspace(self.center_hz - half, self.center_hz + half, n)

    def contains(self, freq_hz: float) -> bool:
        if self.mode == IsmBandMode.FIXED:
            return abs(freq_hz - self.center_hz) < 1.0
        if self.mode == IsmBandMode.FULL:
            lo, hi = ISM_BAND_HZ
            return lo <= freq_hz <= hi
        half = self.tolerance_mhz * 1e6
        return (self.center_hz - half) <= freq_hz <= (self.center_hz + half)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "center_hz": self.center_hz,
            "tolerance_mhz": self.tolerance_mhz,
            "n_samples": self.n_samples,
            "freqs_hz": [float(f) for f in self.freqs_hz()],
        }


DEFAULT_ISM_CONFIG = IsmBandConfig(mode=IsmBandMode.FULL, n_samples=5)
FIXED_ISM_CONFIG = IsmBandConfig(mode=IsmBandMode.FIXED)
TUNABLE_ISM_CONFIG = IsmBandConfig(mode=IsmBandMode.TUNABLE, tolerance_mhz=50.0, n_samples=5)


def ism_config_from_cli(
    mode: str,
    *,
    n_samples: int = 5,
    tolerance_mhz: float = 50.0,
    center_hz: float = ISM_CENTER_HZ,
) -> IsmBandConfig:
    """Parse pipeline/CLI ``--ism-band`` values."""
    try:
        band_mode = IsmBandMode(mode.lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in IsmBandMode)
        raise ValueError(f"ism-band must be one of {valid}, got {mode!r}") from exc
    return IsmBandConfig(
        mode=band_mode,
        center_hz=center_hz,
        tolerance_mhz=tolerance_mhz,
        n_samples=n_samples,
    )
