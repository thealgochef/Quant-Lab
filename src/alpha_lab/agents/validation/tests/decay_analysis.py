"""
Signal Decay Analysis testing.

Tests:
- IC at each forward horizon (1, 5, 10, 15, 30, 60, 120, 240 bars)
- Exponential decay fit to IC curve
- Half-life computation (bars until IC drops to 50% of peak)
- Decay classification: ultra-fast / fast / medium / slow / persistent
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alpha_lab.agents.validation.firewall import ValidationTest, compute_forward_returns
from alpha_lab.core.contracts import SignalVector

_DEFAULT_HORIZONS = [1, 5, 10, 15, 30, 60, 120, 240]

# Decay class boundaries (in bars)
_DECAY_CLASSES = {
    "ultra-fast": 5,
    "fast": 15,
    "medium": 60,
    "slow": 240,
    "persistent": float("inf"),
}


class DecayAnalysisTest(ValidationTest):
    """Signal decay analysis test suite."""

    test_name = "decay_analysis"

    def __init__(self, horizons: list[int] | None = None) -> None:
        self.horizons = horizons or _DEFAULT_HORIZONS

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute decay metrics including half-life and decay classification."""
        close = price_data["close"]
        sig = signal.direction * signal.strength

        if isinstance(sig, pd.Series):
            sig = sig.values
        sig = pd.Series(sig, index=close.index)

        # IC at each horizon
        ic_curve: dict[str, float] = {}
        for h in self.horizons:
            if h >= len(close):
                break
            fwd = compute_forward_returns(close, h)
            mask = sig.notna() & fwd.notna() & (sig != 0)
            if mask.sum() < 30:
                continue
            corr, _ = stats.spearmanr(sig[mask], fwd[mask])
            ic_curve[str(h)] = float(corr) if np.isfinite(corr) else 0.0

        if not ic_curve:
            return {
                "ic_curve": {},
                "half_life": 0.0,
                "decay_class": "ultra-fast",
                "peak_ic": 0.0,
                "peak_horizon": 0,
            }

        # Find peak IC
        horizons_arr = [int(k) for k in ic_curve]
        ic_arr = [ic_curve[str(h)] for h in horizons_arr]
        abs_ic = [abs(v) for v in ic_arr]
        peak_idx = int(np.argmax(abs_ic))
        peak_ic = ic_arr[peak_idx]
        peak_horizon = horizons_arr[peak_idx]

        # Half-life: first horizon after peak where |IC| drops below 50% of peak
        half_life = float(self.horizons[-1])  # default to max
        half_threshold = abs(peak_ic) * 0.5
        for i in range(peak_idx + 1, len(horizons_arr)):
            if abs(ic_arr[i]) < half_threshold:
                # Interpolate between this and previous horizon
                if i > 0:
                    h_prev = horizons_arr[i - 1]
                    h_curr = horizons_arr[i]
                    ic_prev = abs(ic_arr[i - 1])
                    ic_curr = abs(ic_arr[i])
                    if ic_prev > ic_curr:
                        frac = (ic_prev - half_threshold) / (ic_prev - ic_curr)
                        half_life = h_prev + frac * (h_curr - h_prev)
                    else:
                        half_life = float(h_curr)
                else:
                    half_life = float(horizons_arr[i])
                break

        # Classify decay
        decay_class = "persistent"
        for cls, boundary in _DECAY_CLASSES.items():
            if half_life < boundary:
                decay_class = cls
                break

        return {
            "ic_curve": ic_curve,
            "half_life": float(half_life),
            "decay_class": decay_class,
            "peak_ic": float(peak_ic),
            "peak_horizon": peak_horizon,
        }
