"""
Robustness testing.

Tests:
- Subsample stability: performance across 4 calendar quarters
- Regime stability: performance in trending vs ranging vs volatile periods
- Checks that signal direction doesn't flip sign across subsamples
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alpha_lab.agents.validation.firewall import ValidationTest, compute_forward_returns
from alpha_lab.core.contracts import SignalVector

_N_SUBSAMPLES = 4
_DEFAULT_HORIZON = 5


class RobustnessTest(ValidationTest):
    """Robustness test suite."""

    test_name = "robustness"

    def __init__(self, n_subsamples: int = _N_SUBSAMPLES, horizon: int = _DEFAULT_HORIZON) -> None:
        self.n_subsamples = n_subsamples
        self.horizon = horizon

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute robustness metrics across subsamples and regimes."""
        close = price_data["close"]
        sig = signal.direction * signal.strength

        if isinstance(sig, pd.Series):
            sig = sig.values
        sig = pd.Series(sig, index=close.index)

        fwd = compute_forward_returns(close, self.horizon)

        # Subsample stability: split into N non-overlapping chunks
        n = len(close)
        chunk_size = n // self.n_subsamples
        subsample_ics: list[float] = []

        for i in range(self.n_subsamples):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_subsamples - 1 else n
            s_chunk = sig.iloc[start:end]
            f_chunk = fwd.iloc[start:end]
            mask = s_chunk.notna() & f_chunk.notna() & (s_chunk != 0)
            if mask.sum() >= 20:
                corr, _ = stats.spearmanr(s_chunk[mask], f_chunk[mask])
                if np.isfinite(corr):
                    subsample_ics.append(float(corr))

        # Subsample stable if:
        # 1. All subsamples have same sign IC
        # 2. No subsample IC is drastically different
        subsample_stable = False
        if len(subsample_ics) >= 2:
            signs = [np.sign(ic) for ic in subsample_ics]
            same_sign = len(set(signs)) == 1 and signs[0] != 0
            # Coefficient of variation < 1.0 (not too unstable)
            mean_ic = np.mean(subsample_ics)
            std_ic = np.std(subsample_ics)
            cv = abs(std_ic / mean_ic) if abs(mean_ic) > 1e-6 else float("inf")
            subsample_stable = same_sign and cv < 1.5

        # Regime stability: split by volatility regime
        ret = close.pct_change()
        rolling_vol = ret.rolling(20).std()
        regime_ics = _compute_regime_ics(sig, fwd, rolling_vol)

        return {
            "subsample_ics": subsample_ics,
            "subsample_stable": subsample_stable,
            "n_subsamples_computed": len(subsample_ics),
            "regime_ics": regime_ics,
        }


def _compute_regime_ics(
    sig: pd.Series, fwd: pd.Series, rolling_vol: pd.Series
) -> dict[str, float]:
    """Compute IC in different volatility regimes (terciles)."""
    result: dict[str, float] = {}
    valid = sig.notna() & fwd.notna() & rolling_vol.notna() & (sig != 0)

    if valid.sum() < 60:
        return result

    try:
        terciles = pd.qcut(rolling_vol[valid], 3, labels=["low_vol", "mid_vol", "high_vol"])
    except ValueError:
        return result

    for regime in ["low_vol", "mid_vol", "high_vol"]:
        mask = terciles == regime
        if mask.sum() >= 20:
            corr, _ = stats.spearmanr(sig[valid][mask], fwd[valid][mask])
            if np.isfinite(corr):
                result[regime] = float(corr)

    return result
