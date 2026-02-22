"""
Session tagging and killzone classification.

Tags every bar with:
- session_id: unique per trading day (e.g., 'NQ_2026-02-21_RTH')
- session_type: RTH | GLOBEX | POST_MARKET
- killzone: LONDON | NEW_YORK | ASIA | OVERLAP | NONE

Killzone boundaries (Eastern Time):
- London: 2:00-5:00 AM
- New York: 8:00-11:00 AM (09:30-11:00 after OVERLAP window)
- Overlap: 8:00-9:30 AM (London/NY overlap)
- Asia: 7:00-10:00 PM
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from alpha_lab.core.enums import Killzone, SessionType

# Session boundaries (Eastern Time)
_RTH_OPEN = dt.time(9, 30)
_RTH_CLOSE = dt.time(16, 15)
_GLOBEX_OPEN = dt.time(18, 0)

# Killzone boundaries (Eastern Time)
_LONDON_START = dt.time(2, 0)
_LONDON_END = dt.time(5, 0)
_OVERLAP_START = dt.time(8, 0)
_OVERLAP_END = dt.time(9, 30)
_NY_START = dt.time(8, 0)
_NY_END = dt.time(11, 0)
_ASIA_START = dt.time(19, 0)
_ASIA_END = dt.time(22, 0)


def tag_sessions(bars: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Add session_id and session_type columns to bars DataFrame.

    Trading date logic:
    - RTH bars (09:30-16:15): trading date = bar's calendar date
    - Evening GLOBEX bars (18:00-23:59): trading date = next calendar date
    - Morning GLOBEX bars (00:00-09:29): trading date = bar's calendar date
    - POST_MARKET bars (16:15-18:00): trading date = bar's calendar date

    Args:
        bars: DataFrame with DatetimeIndex (US/Eastern)
        instrument: 'NQ' or 'ES'

    Returns:
        DataFrame with added session_id and session_type columns
    """
    df = bars.copy()
    times = df.index.time

    # Vectorised session type classification
    is_rth = (times >= _RTH_OPEN) & (times < _RTH_CLOSE)
    is_evening_globex = times >= _GLOBEX_OPEN
    is_morning_globex = times < _RTH_OPEN
    session_types = np.where(
        is_rth,
        SessionType.RTH.value,
        np.where(
            is_evening_globex | is_morning_globex,
            SessionType.GLOBEX.value,
            SessionType.POST_MARKET.value,
        ),
    )

    # Trading date: evening GLOBEX bars belong to the next calendar date
    dates = df.index.date
    trading_dates = np.where(
        is_evening_globex,
        pd.to_datetime(dates) + pd.Timedelta(days=1),
        pd.to_datetime(dates),
    )
    trading_date_strs = pd.to_datetime(trading_dates).strftime("%Y-%m-%d")

    df["session_type"] = session_types
    df["session_id"] = [
        f"{instrument}_{d}_{s}"
        for d, s in zip(trading_date_strs, session_types, strict=True)
    ]
    return df


def classify_killzone(timestamp: pd.Timestamp) -> Killzone:
    """Classify a timestamp into its killzone.

    OVERLAP (08:00-09:30) is checked before NEW_YORK since it is a
    subset of the NY window.  Bars from 09:30-11:00 get NEW_YORK.

    Args:
        timestamp: Bar timestamp (assumed Eastern Time)

    Returns:
        Killzone enum value
    """
    t = timestamp.time()

    if _OVERLAP_START <= t < _OVERLAP_END:
        return Killzone.OVERLAP
    if _LONDON_START <= t < _LONDON_END:
        return Killzone.LONDON
    if _NY_START <= t < _NY_END:
        return Killzone.NEW_YORK
    if _ASIA_START <= t < _ASIA_END:
        return Killzone.ASIA
    return Killzone.NONE


def tag_killzones(bars: pd.DataFrame) -> pd.DataFrame:
    """Add killzone column to bars DataFrame.

    Args:
        bars: DataFrame with DatetimeIndex

    Returns:
        DataFrame with added killzone column
    """
    df = bars.copy()
    df["killzone"] = [classify_killzone(ts).value for ts in df.index]
    return df
