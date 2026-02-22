"""
Bar aggregation logic.

Handles:
- Tick chart construction (987-tick, 2000-tick bars) — NOT YET IMPLEMENTED
- Time bar resampling from 1-minute bars (1m through 1D)
- Partial bar handling at session boundaries
- Exchange session alignment (not UTC midnight)
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.core.enums import Timeframe

# Timeframe -> pandas resample rule
_RESAMPLE_RULES: dict[str, str] = {
    Timeframe.M1: "1min",
    Timeframe.M3: "3min",
    Timeframe.M5: "5min",
    Timeframe.M10: "10min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1h",
    Timeframe.H4: "4h",
}


def aggregate_tick_bars(ticks: pd.DataFrame, tick_count: int) -> pd.DataFrame:
    """Aggregate raw ticks into tick-count bars.

    Args:
        ticks: DataFrame with columns [price, size, timestamp]
        tick_count: Number of ticks per bar (987 or 2000)

    Returns:
        DataFrame with columns [open, high, low, close, volume, tick_count, timestamp]
    """
    raise NotImplementedError("Tick bar aggregation not implemented — using time-based bars only")


def aggregate_time_bars(
    bars_1m: pd.DataFrame,
    timeframe: Timeframe,
    session_boundaries: dict[str, str],
) -> pd.DataFrame:
    """Resample 1-minute OHLCV bars into a higher timeframe.

    For 1m: returns input unchanged.
    For 3m-4H: standard pandas resample with proper OHLCV aggregation.
    For 1D: session-aware grouping by trading date (uses session_id
    column if present, otherwise groups by calendar date).

    Args:
        bars_1m: DataFrame with DatetimeIndex (US/Eastern) and columns
                 [open, high, low, close, volume]. May also have
                 session_id, session_type, killzone columns.
        timeframe: Target timeframe
        session_boundaries: Dict with rth_open, rth_close, etc.

    Returns:
        DataFrame with columns [open, high, low, close, volume]
    """
    if timeframe in (Timeframe.TICK_987, Timeframe.TICK_2000):
        raise NotImplementedError("Tick bar aggregation not implemented")

    if bars_1m.empty:
        return bars_1m[["open", "high", "low", "close", "volume"]].copy()

    if timeframe == Timeframe.M1:
        return bars_1m[["open", "high", "low", "close", "volume"]].copy()

    if timeframe == Timeframe.D1:
        return _aggregate_daily(bars_1m)

    rule = _RESAMPLE_RULES.get(timeframe)
    if rule is None:
        msg = f"No resample rule for timeframe {timeframe}"
        raise ValueError(msg)

    resampled = (
        bars_1m[["open", "high", "low", "close", "volume"]]
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna(subset=["open"])
    )
    return resampled


def _aggregate_daily(bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1m bars into daily bars using trading-date grouping.

    A CME trading day runs 18:00 ET to 17:00 ET next day.  If the
    bars already have a ``session_id`` column (format
    ``{INSTR}_{YYYY-MM-DD}_{TYPE}``), we extract the date from it.
    Otherwise we fall back to grouping by calendar date.
    """
    if "session_id" in bars_1m.columns:
        # Extract the date portion from session_id
        trading_dates = bars_1m["session_id"].str.extract(r"(\d{4}-\d{2}-\d{2})")[0]
    else:
        trading_dates = bars_1m.index.date.astype(str)

    ohlcv = bars_1m[["open", "high", "low", "close", "volume"]].copy()
    ohlcv["_td"] = trading_dates.values

    daily = ohlcv.groupby("_td").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    daily.index = pd.DatetimeIndex(daily.index, tz="US/Eastern")
    daily.index.name = "timestamp"
    return daily
