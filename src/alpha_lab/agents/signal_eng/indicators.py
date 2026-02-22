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

    kama = pd.Series(np.nan, index=close.index)
    first_valid = period
    if first_valid < len(close):
        kama.iloc[first_valid] = close.iloc[first_valid]

    for i in range(first_valid + 1, len(close)):
        kama.iloc[i] = kama.iloc[i - 1] + sc.iloc[i] * (
            close.iloc[i] - kama.iloc[i - 1]
        )

    return kama


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
