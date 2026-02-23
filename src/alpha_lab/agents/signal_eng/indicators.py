"""
Shared technical indicators for signal detectors.

Provides ATR, KAMA, and session-anchored VWAP computations used across
multiple Tier 1 detectors.  All functions are pure (no side effects)
and operate on pandas Series / DataFrames with a DatetimeIndex.

Adapted from proven implementations in Claude-my-quant/ml/features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback window (default 14)

    Returns:
        pd.Series of ATR values (NaN for warmup period)
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(window=period).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average.

    Args:
        series: Input price series
        span: EMA span (e.g. 13, 48, 200)

    Returns:
        pd.Series of EMA values
    """
    return series.ewm(span=span, adjust=False).mean()


def compute_kama(
    close: pd.Series,
    period: int = 10,
    fast: int = 2,
    slow: int = 30,
) -> pd.Series:
    """Kaufman Adaptive Moving Average.

    Adapts smoothing speed based on the efficiency ratio (directional
    movement / volatility).  High ER -> fast smoothing, low ER -> slow.

    Args:
        close: Close price series
        period: Efficiency ratio lookback (default 10)
        fast: Fast smoothing constant (default 2)
        slow: Slow smoothing constant (default 30)

    Returns:
        pd.Series of KAMA values (NaN for warmup period)
    """
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(window=period).sum()

    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)

    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    close_vals = close.values
    sc_vals = sc.values
    kama_vals = np.full(len(close), np.nan)

    first_valid = period
    if first_valid < len(close):
        kama_vals[first_valid] = close_vals[first_valid]

    for i in range(first_valid + 1, len(close)):
        kama_vals[i] = kama_vals[i - 1] + sc_vals[i] * (
            close_vals[i] - kama_vals[i - 1]
        )

    return pd.Series(kama_vals, index=close.index)


def compute_kama_efficiency_ratio(
    close: pd.Series, period: int = 10
) -> pd.Series:
    """Kaufman Efficiency Ratio (directional movement / volatility).

    ER near 1.0 = strong trend, ER near 0.0 = choppy/ranging.

    Args:
        close: Close price series
        period: Lookback window

    Returns:
        pd.Series of ER values in [0, 1]
    """
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(window=period).sum()
    er = direction / volatility.replace(0, np.nan)
    return er.fillna(0).clip(0, 1)


def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP with daily reset.

    Resets cumulative sums at each new session (grouped by date in
    the DatetimeIndex).  Falls back to typical price when cumulative
    volume is zero.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
            and a DatetimeIndex

    Returns:
        pd.Series of VWAP values
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = typical_price * df["volume"]

    dates = df.index.date
    cum_tp_vol = tp_vol.groupby(dates).cumsum()
    cum_vol = df["volume"].groupby(dates).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.fillna(typical_price)


def compute_session_vwap_bands(
    df: pd.DataFrame, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """VWAP with upper and lower standard deviation bands.

    Args:
        df: DataFrame with OHLCV columns and DatetimeIndex
        num_std: Number of standard deviations for bands (default 2.0)

    Returns:
        Tuple of (vwap, upper_band, lower_band)
    """
    vwap = compute_session_vwap(df)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0

    dates = df.index.date
    # Running variance of typical price vs VWAP within each session
    sq_diff = (typical_price - vwap) ** 2

    # Cumulative count per session for running std dev
    cum_count = sq_diff.groupby(dates).cumcount() + 1
    cum_sq_diff = sq_diff.groupby(dates).cumsum()
    running_var = cum_sq_diff / cum_count
    running_std = np.sqrt(running_var)

    upper = vwap + num_std * running_std
    lower = vwap - num_std * running_std
    return vwap, upper, lower


def compute_swing_highs(
    highs: pd.Series, left: int = 3, right: int = 3
) -> pd.Series:
    """Detect swing highs using N-bar pivot logic.

    A swing high at bar *i* requires ``highs[i]`` to be the strict
    maximum of the window ``[i-left, i+right]``.  The swing is stamped
    at bar ``i + right`` (the earliest bar where the pivot is confirmed)
    to prevent look-ahead bias.

    Args:
        highs: Series of high prices
        left: Bars to the left of pivot
        right: Bars to the right of pivot

    Returns:
        pd.Series with swing high values where detected, NaN elsewhere.
    """
    result = pd.Series(np.nan, index=highs.index)
    vals = highs.values
    for i in range(left, len(vals) - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == window.max():
            # On tie, pick leftmost: only stamp if no earlier bar in window matches
            tie_indices = np.where(window == vals[i])[0]
            if tie_indices[0] == left:  # i is the leftmost max
                result.iloc[i + right] = vals[i]
    return result


def compute_swing_lows(
    lows: pd.Series, left: int = 3, right: int = 3
) -> pd.Series:
    """Detect swing lows using N-bar pivot logic.

    A swing low at bar *i* requires ``lows[i]`` to be the strict
    minimum of the window ``[i-left, i+right]``.  The swing is stamped
    at bar ``i + right`` (the earliest bar where the pivot is confirmed)
    to prevent look-ahead bias.

    Args:
        lows: Series of low prices
        left: Bars to the left of pivot
        right: Bars to the right of pivot

    Returns:
        pd.Series with swing low values where detected, NaN elsewhere.
    """
    result = pd.Series(np.nan, index=lows.index)
    vals = lows.values
    for i in range(left, len(vals) - right):
        window = vals[i - left: i + right + 1]
        if vals[i] == window.min():
            # On tie, pick leftmost: only stamp if no earlier bar in window matches
            tie_indices = np.where(window == vals[i])[0]
            if tie_indices[0] == left:  # i is the leftmost min
                result.iloc[i + right] = vals[i]
    return result
