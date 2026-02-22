"""
Factor Orthogonality testing.

Tests:
- Correlation with known factors: momentum, mean-reversion, volatility, volume, calendar
- Incremental R² above existing factor exposures
- Maximum single-factor correlation must be < 0.30 to pass
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.agents.validation.firewall import ValidationTest
from alpha_lab.core.contracts import SignalVector


class OrthogonalityTest(ValidationTest):
    """Factor orthogonality test suite."""

    test_name = "orthogonality"

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute factor correlations and incremental R² metrics."""
        close = price_data["close"]
        volume = price_data.get("volume", pd.Series(0, index=close.index))
        sig = signal.direction * signal.strength

        if isinstance(sig, pd.Series):
            sig = sig.values
        sig = pd.Series(sig, index=close.index, dtype=float)

        # Build known factor proxies
        factors = _build_factor_proxies(close, volume)

        # Align signal with factors
        valid = sig.notna() & (sig != 0)
        for f in factors.values():
            valid = valid & f.notna()

        if valid.sum() < 30:
            return {
                "factor_correlations": {},
                "max_factor_corr": 0.0,
                "incremental_r2": 0.0,
                "is_orthogonal": True,
            }

        sig_v = sig[valid].values
        factor_corrs: dict[str, float] = {}

        for name, f in factors.items():
            f_v = f[valid].values
            corr = np.corrcoef(sig_v, f_v)[0, 1]
            factor_corrs[name] = float(corr) if np.isfinite(corr) else 0.0

        max_factor_corr = max(abs(v) for v in factor_corrs.values()) if factor_corrs else 0.0

        # Incremental R²: R² of signal vs forward returns ABOVE factor model
        incremental_r2 = _compute_incremental_r2(sig_v, factors, valid, close)

        is_orthogonal = max_factor_corr < 0.30 and incremental_r2 > 0.005

        return {
            "factor_correlations": factor_corrs,
            "max_factor_corr": float(max_factor_corr),
            "incremental_r2": float(incremental_r2),
            "is_orthogonal": is_orthogonal,
        }


def _build_factor_proxies(close: pd.Series, volume: pd.Series) -> dict[str, pd.Series]:
    """Build common factor proxies from price/volume data."""
    factors: dict[str, pd.Series] = {}

    # Momentum: 20-bar return
    factors["momentum"] = close.pct_change(20)

    # Mean-reversion: negative of 5-bar return
    factors["mean_reversion"] = -close.pct_change(5)

    # Volatility: 20-bar rolling std of returns
    ret = close.pct_change()
    factors["volatility"] = ret.rolling(20).std()

    # Volume: ratio to 20-bar SMA
    vol_sma = volume.rolling(20).mean()
    factors["volume"] = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)

    # Calendar: hour of day (normalized 0-1) as proxy for time-of-day effect
    if hasattr(close.index, "hour"):
        factors["calendar"] = pd.Series(
            close.index.hour / 24.0, index=close.index
        )

    return factors


def _compute_incremental_r2(
    sig: np.ndarray,
    factors: dict[str, pd.Series],
    valid_mask: pd.Series,
    close: pd.Series,
) -> float:
    """Compute incremental R² of signal above factor model.

    Uses OLS: R²(factors + signal) - R²(factors only) for 5-bar forward returns.
    """
    from alpha_lab.agents.validation.firewall import compute_forward_returns

    fwd = compute_forward_returns(close, 5)
    full_valid = valid_mask & fwd.notna()

    if full_valid.sum() < 30:
        return 0.0

    y = fwd[full_valid].values

    # Factor-only model
    factor_matrix = np.column_stack([f[full_valid].values for f in factors.values()])
    # Add intercept
    ones = np.ones((len(y), 1))
    x_factors = np.column_stack([ones, factor_matrix])

    r2_factors = _ols_r2(x_factors, y)

    # Factor + signal model
    x_full = np.column_stack([x_factors, sig[: len(y)] if len(sig) >= len(y) else sig])

    # Need to realign sig to the valid mask
    sig_aligned = pd.Series(sig, index=close.index[valid_mask])[full_valid].values
    x_full = np.column_stack([x_factors, sig_aligned])

    r2_full = _ols_r2(x_full, y)

    return max(0.0, r2_full - r2_factors)


def _ols_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Compute R² from OLS regression."""
    try:
        # Use least squares: beta = (X'X)^-1 X'y
        beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)
    except np.linalg.LinAlgError:
        return 0.0
