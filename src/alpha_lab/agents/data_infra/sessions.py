"""
Session tagging and killzone classification.

Tags every bar with:
- session_id: unique per trading day (e.g., 'NQ_2026-02-21_RTH')
- session_type: RTH | GLOBEX | PRE_MARKET | POST_MARKET
- killzone: LONDON | NEW_YORK | ASIA | OVERLAP | NONE

Killzone boundaries (Eastern Time):
- London: 2:00-5:00 AM
- New York: 8:00-11:00 AM
- Asia: 7:00-10:00 PM
"""

from __future__ import annotations

import pandas as pd

from alpha_lab.core.enums import Killzone


def tag_sessions(bars: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Add session_id and session_type columns to bars DataFrame.

    Args:
        bars: DataFrame with DatetimeIndex
        instrument: 'NQ' or 'ES'

    Returns:
        DataFrame with added session_id and session_type columns
    """
    raise NotImplementedError


def classify_killzone(timestamp: pd.Timestamp) -> Killzone:
    """
    Classify a timestamp into its killzone.

    Args:
        timestamp: Bar timestamp (assumed Eastern Time)

    Returns:
        Killzone enum value
    """
    raise NotImplementedError


def tag_killzones(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Add killzone column to bars DataFrame.

    Args:
        bars: DataFrame with DatetimeIndex

    Returns:
        DataFrame with added killzone column
    """
    raise NotImplementedError
