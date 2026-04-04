"""
MS (Market Shift) momentum features from tick-level data.

Computes technical indicators directly on tick prices around each extremum:
RSI at multiple periods, MACD, volume momentum, and price velocity.

Based on MS feature set from Sokolovsky & Arnaboldi (2020).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import FeatureConfig
from alpha_lab.agents.data_infra.ml.extrema_detection import Extremum

_DEFAULT_CONFIG = FeatureConfig()


def extract_ms_features(
    extremum: Extremum,
    tick_prices: pd.Series | np.ndarray,
    tick_volumes: pd.Series | np.ndarray | None = None,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    """Extract MS (Market Shift) momentum features at an extremum.

    Looks backward from the extremum to compute momentum indicators
    on the raw tick price series. No look-ahead.

    Args:
        extremum: The detected price extremum.
        tick_prices: Full tick price series (Series or numpy array).
        tick_volumes: Optional tick volume series (Series or numpy array).
        config: Feature extraction config.

    Returns:
        Dict of feature name -> value.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    features: dict[str, float] = {}
    idx = extremum.index

    # Extract lookback window — numpy slicing is much faster than pandas .iloc
    lookback_start = max(0, idx - config.ms_window)
    if isinstance(tick_prices, np.ndarray):
        prices_window = tick_prices[lookback_start: idx + 1]
    else:
        prices_window = tick_prices.iloc[lookback_start: idx + 1].values.astype(float)

    n = len(prices_window)
    if n < 10:
        return features

    # RSI at multiple periods
    for period in config.rsi_periods:
        if period < n:
            rsi = _compute_tick_rsi(prices_window, period)
            if np.isfinite(rsi):
                features[f"ms_rsi_{period}"] = rsi

    # MACD at selected periods
    macd_periods = [(12, 26), (20, 50), (40, 100)]
    for fast_p, slow_p in macd_periods:
        if slow_p < n:
            macd_val = _compute_tick_macd(prices_window, fast_p, slow_p)
            if np.isfinite(macd_val):
                features[f"ms_macd_{fast_p}_{slow_p}"] = macd_val

    # Price velocity: rate of change approaching extremum
    for lookback in [10, 20, 50]:
        if lookback < n:
            start_price = prices_window[-(lookback + 1)]
            end_price = prices_window[-1]
            if start_price > 0:
                features[f"ms_price_velocity_{lookback}"] = (
                    (end_price - start_price) / start_price
                )

    # Volume momentum
    if tick_volumes is not None:
        if isinstance(tick_volumes, np.ndarray):
            vol_window = tick_volumes[lookback_start: idx + 1]
        else:
            vol_window = tick_volumes.iloc[lookback_start: idx + 1].values.astype(float)
        if len(vol_window) >= 20:
            recent_vol = np.nanmean(vol_window[-20:])
            overall_vol = np.nanmean(vol_window)
            if overall_vol > 0:
                features["ms_volume_momentum"] = recent_vol / overall_vol

    # Bid/ask volume fractions (from tick data if available)
    if n >= 2:
        price_changes = np.diff(prices_window)
        up_moves = np.sum(price_changes > 0)
        down_moves = np.sum(price_changes < 0)
        total_moves = up_moves + down_moves
        if total_moves > 0:
            features["ms_uptick_fraction"] = float(up_moves / total_moves)
            features["ms_downtick_fraction"] = float(down_moves / total_moves)

    # Volatility at extremum
    if n >= 20:
        returns = np.diff(prices_window) / prices_window[:-1]
        returns = returns[np.isfinite(returns)]
        if len(returns) >= 10:
            features["ms_volatility"] = float(np.std(returns[-20:]))

    return features


def _compute_tick_rsi(prices: np.ndarray, period: int) -> float:
    """Compute RSI on tick prices using the standard Wilder method.

    Args:
        prices: Array of tick prices.
        period: RSI lookback period.

    Returns:
        RSI value in [0, 100], or NaN if insufficient data.
    """
    if len(prices) < period + 1:
        return float("nan")

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Use exponential (Wilder) smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_tick_macd(
    prices: np.ndarray, fast_period: int, slow_period: int,
) -> float:
    """Compute MACD histogram value at the last tick.

    Args:
        prices: Array of tick prices.
        fast_period: Fast EMA period.
        slow_period: Slow EMA period.

    Returns:
        MACD histogram value (fast EMA - slow EMA), or NaN.
    """
    if len(prices) < slow_period:
        return float("nan")

    fast_ema = _ema_last(prices, fast_period)
    slow_ema = _ema_last(prices, slow_period)

    if np.isnan(fast_ema) or np.isnan(slow_ema):
        return float("nan")

    return fast_ema - slow_ema


def _ema_last(values: np.ndarray, span: int) -> float:
    """Compute EMA and return only the final value (memory-efficient)."""
    if len(values) < span:
        return float("nan")

    alpha = 2.0 / (span + 1)
    ema = float(values[0])
    for v in values[1:]:
        ema = alpha * v + (1 - alpha) * ema
    return ema
