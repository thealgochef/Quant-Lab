"""
Performance metric collectors for live monitoring.

Continuous metrics (per architecture spec Section 7.1):
1. Rolling IC (20-bar window)
2. Rolling Hit Rate (50-trade window)
3. Cost-Adjusted Sharpe (20-day window)
4. Signal Decay Velocity
5. Slippage Tracking
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def compute_rolling_ic(
    signal_values: list[float], returns: list[float], window: int = 20
) -> list[float]:
    """Compute rolling Information Coefficient over a sliding window.

    IC is the Pearson correlation between signal values and forward returns
    over a rolling window.

    Args:
        signal_values: Signal strength/direction values
        returns: Corresponding forward returns
        window: Rolling window size (default 20 bars)

    Returns:
        List of rolling IC values (shorter than input by window-1)
    """
    if len(signal_values) != len(returns):
        raise ValueError("signal_values and returns must have same length")

    if len(signal_values) < window:
        return []

    sig = pd.Series(signal_values, dtype=float)
    ret = pd.Series(returns, dtype=float)

    rolling_ic = sig.rolling(window=window, min_periods=10).corr(ret)
    return rolling_ic.dropna().tolist()


def compute_rolling_hit_rate(
    predictions: list[int], actuals: list[int], window: int = 50
) -> list[float]:
    """Compute rolling directional hit rate over a trade window.

    Only counts non-neutral predictions (ignores 0s) in the denominator.

    Args:
        predictions: Predicted directions [-1, 0, +1]
        actuals: Actual directions [-1, 0, +1]
        window: Rolling window size (default 50 trades)

    Returns:
        List of rolling hit rate values
    """
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals must have same length")

    if len(predictions) < window:
        return []

    pred = np.array(predictions, dtype=float)
    act = np.array(actuals, dtype=float)

    results: list[float] = []
    for i in range(window, len(predictions) + 1):
        p_win = pred[i - window : i]
        a_win = act[i - window : i]

        # Only count non-neutral predictions
        mask = p_win != 0
        n_active = mask.sum()
        if n_active == 0:
            results.append(float("nan"))
            continue

        correct = ((p_win[mask] > 0) & (a_win[mask] > 0)) | (
            (p_win[mask] < 0) & (a_win[mask] < 0)
        )
        results.append(float(correct.sum() / n_active))

    return results


def compute_rolling_sharpe(
    daily_returns: list[float], window: int = 20
) -> list[float]:
    """Compute rolling annualized Sharpe ratio.

    Sharpe = sqrt(252) * mean(returns) / std(returns)

    Args:
        daily_returns: Daily return series
        window: Rolling window size (default 20 days)

    Returns:
        List of rolling Sharpe values
    """
    if len(daily_returns) < window:
        return []

    ret = pd.Series(daily_returns, dtype=float)
    rolling_mean = ret.rolling(window=window, min_periods=10).mean()
    rolling_std = ret.rolling(window=window, min_periods=10).std(ddof=1)

    # Avoid division by zero
    sharpe = rolling_mean / rolling_std.replace(0, float("nan"))
    sharpe = sharpe * math.sqrt(252)

    return sharpe.dropna().tolist()


def compute_decay_velocity(
    ic_series: list[float], expected_half_life: float
) -> dict[str, Any]:
    """Compare actual IC decay against expected decay curve.

    Fits a linear regression to log(|IC|) vs time to estimate actual
    half-life, then compares to the expected half-life.

    Args:
        ic_series: Time series of IC values (most recent last)
        expected_half_life: Expected half-life in bars from validation

    Returns:
        Dict with actual_half_life, velocity_ratio, classification
    """
    if len(ic_series) < 20 or expected_half_life <= 0:
        return {
            "actual_half_life": float("nan"),
            "expected_half_life": expected_half_life,
            "velocity_ratio": float("nan"),
            "classification": "insufficient_data",
        }

    ic_arr = np.array(ic_series, dtype=float)
    abs_ic = np.abs(ic_arr)

    # Filter out near-zero values to avoid log issues
    valid = abs_ic > 0.001
    if valid.sum() < 10:
        return {
            "actual_half_life": float("nan"),
            "expected_half_life": expected_half_life,
            "velocity_ratio": float("nan"),
            "classification": "insufficient_data",
        }

    t = np.arange(len(ic_arr))[valid].astype(float)
    log_ic = np.log(abs_ic[valid])

    # Linear regression: log(IC) = a - b*t → half_life = ln(2)/b
    t_mean = t.mean()
    log_mean = log_ic.mean()
    slope = (np.sum((t - t_mean) * (log_ic - log_mean))) / max(
        np.sum((t - t_mean) ** 2), 1e-12
    )

    if slope >= 0:
        # IC not decaying (increasing or flat)
        actual_half_life = float("inf")
        velocity_ratio = 0.0
        classification = "slower_than_expected"
    else:
        actual_half_life = float(math.log(2) / abs(slope))
        velocity_ratio = expected_half_life / actual_half_life
        if velocity_ratio < 0.5:
            classification = "slower_than_expected"
        elif velocity_ratio > 1.5:
            classification = "faster_than_expected"
        else:
            classification = "as_expected"

    return {
        "actual_half_life": actual_half_life,
        "expected_half_life": expected_half_life,
        "velocity_ratio": velocity_ratio,
        "classification": classification,
    }


def compute_slippage_tracking(
    expected_prices: list[float], realized_prices: list[float]
) -> dict[str, float]:
    """Track realized vs assumed slippage.

    Slippage = realized_price - expected_price (positive = worse fill).

    Args:
        expected_prices: Expected fill prices
        realized_prices: Actual fill prices

    Returns:
        Dict with avg_slippage, worst_slippage, slippage_std, avg_bps, num_trades
    """
    if len(expected_prices) != len(realized_prices):
        raise ValueError("expected_prices and realized_prices must have same length")

    if len(expected_prices) == 0:
        return {
            "avg_slippage": 0.0,
            "worst_slippage": 0.0,
            "slippage_std": 0.0,
            "avg_bps": 0.0,
            "num_trades": 0,
        }

    expected = np.array(expected_prices, dtype=float)
    realized = np.array(realized_prices, dtype=float)
    slippage = np.abs(realized - expected)

    # Basis points relative to expected price
    bps = (slippage / np.where(expected != 0, expected, 1.0)) * 10_000

    return {
        "avg_slippage": float(np.mean(slippage)),
        "worst_slippage": float(np.max(slippage)),
        "slippage_std": float(np.std(slippage, ddof=1)) if len(slippage) > 1 else 0.0,
        "avg_bps": float(np.mean(bps)),
        "num_trades": len(expected_prices),
    }
