"""
Tick-to-bar aggregation logic.

Handles:
- Tick chart construction (987-tick, 2000-tick bars)
- Time bar construction (1m through 1D)
- Partial bar handling at session boundaries
- Exchange session alignment (not UTC midnight)
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.core.enums import Timeframe


def aggregate_tick_bars(ticks: pd.DataFrame, tick_count: int) -> pd.DataFrame:
    """
    Aggregate raw ticks into tick-count bars.

    Args:
        ticks: DataFrame with columns [price, size, timestamp]
        tick_count: Number of ticks per bar (987 or 2000)

    Returns:
        DataFrame with columns [open, high, low, close, volume, tick_count, timestamp]
    """
    raise NotImplementedError


def aggregate_time_bars(
    ticks: pd.DataFrame, timeframe: Timeframe, session_boundaries: dict[str, str]
) -> pd.DataFrame:
    """
    Aggregate raw ticks into time-based OHLCV bars.

    Aligns bars to exchange session boundaries (not UTC midnight).
    Handles overnight/globex session separately from RTH.

    Args:
        ticks: DataFrame with columns [price, size, timestamp]
        timeframe: Target timeframe
        session_boundaries: Dict with rth_open, rth_close, globex_open, globex_close

    Returns:
        DataFrame with columns [open, high, low, close, volume, session_id]
    """
    raise NotImplementedError
