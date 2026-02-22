"""
Information Coefficient (IC) testing.

Tests:
- Spearman rank correlation: signal vs forward returns
- Rolling IC (252-bar window) for stability
- IC Information Ratio (IC_mean / IC_std)
- IC t-statistic (must be > 2.0 to pass)
- Test at horizons: 1, 5, 10, 15, 30, 60, 120, 240 bars
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alpha_lab.agents.validation.firewall import ValidationTest, compute_forward_returns
from alpha_lab.core.contracts import SignalVector

_DEFAULT_HORIZONS = [1, 5, 10, 15, 30, 60, 120, 240]
_ROLLING_WINDOW = 252


class ICTest(ValidationTest):
    """Information Coefficient test suite."""

    test_name = "information_coefficient"

    def __init__(self, horizons: list[int] | None = None) -> None:
        self.horizons = horizons or _DEFAULT_HORIZONS

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute IC metrics across all forward horizons."""
        close = price_data["close"]
        sig = signal.direction * signal.strength

        if isinstance(sig, pd.Series):
            sig = sig.values
        sig = pd.Series(sig, index=close.index)

        # IC at each horizon
        ic_by_horizon: dict[str, float] = {}
        for h in self.horizons:
            if h >= len(close):
                break
            fwd = compute_forward_returns(close, h)
            mask = sig.notna() & fwd.notna() & (sig != 0)
            if mask.sum() < 30:
                continue
            corr, _ = stats.spearmanr(sig[mask], fwd[mask])
            ic_by_horizon[str(h)] = float(corr) if np.isfinite(corr) else 0.0

        if not ic_by_horizon:
            return {
                "ic_mean": 0.0,
                "ic_std": 1.0,
                "ic_ir": 0.0,
                "ic_tstat": 0.0,
                "ic_by_horizon": {},
                "rolling_ic_mean": 0.0,
                "rolling_ic_std": 1.0,
            }

        ic_values = list(ic_by_horizon.values())
        ic_mean = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values)) if len(ic_values) > 1 else 1.0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # IC t-stat: mean / (std / sqrt(n))
        n = len(ic_values)
        ic_tstat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 and n > 1 else 0.0

        # Rolling IC at primary horizon (first available)
        primary_h = self.horizons[0]
        rolling_ic_mean = 0.0
        rolling_ic_std = 1.0
        if primary_h < len(close):
            fwd = compute_forward_returns(close, primary_h)
            rolling_ic = _rolling_spearman(sig, fwd, window=min(_ROLLING_WINDOW, len(close) // 2))
            valid_ic = rolling_ic.dropna()
            if len(valid_ic) > 0:
                rolling_ic_mean = float(valid_ic.mean())
                rolling_ic_std = float(valid_ic.std()) if len(valid_ic) > 1 else 1.0

        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_tstat": ic_tstat,
            "ic_by_horizon": ic_by_horizon,
            "rolling_ic_mean": rolling_ic_mean,
            "rolling_ic_std": rolling_ic_std,
        }


def _rolling_spearman(
    x: pd.Series, y: pd.Series, window: int = 252
) -> pd.Series:
    """Compute rolling Spearman correlation between two series."""
    result = pd.Series(np.nan, index=x.index)
    x_arr = x.values
    y_arr = y.values

    for i in range(window, len(x)):
        x_win = x_arr[i - window : i]
        y_win = y_arr[i - window : i]
        mask = np.isfinite(x_win) & np.isfinite(y_win) & (x_win != 0)
        if mask.sum() >= 20:
            corr, _ = stats.spearmanr(x_win[mask], y_win[mask])
            result.iloc[i] = corr if np.isfinite(corr) else np.nan

    return result
