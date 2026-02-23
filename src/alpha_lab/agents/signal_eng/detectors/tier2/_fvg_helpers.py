"""Shared Fair Value Gap detection logic.

Used by both FairValueGapsDetector and IFVGDetector to avoid
duplicating the 3-candle gap pattern detection algorithm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.indicators import compute_atr


def detect_fvgs(
    df: pd.DataFrame,
    min_gap_atr: float = 0.5,
) -> list[dict]:
    """Detect Fair Value Gaps in OHLCV data.

    A bullish FVG forms when ``high[i-1] < low[i+1]`` (gap up).
    A bearish FVG forms when ``low[i-1] > high[i+1]`` (gap down).

    Args:
        df: DataFrame with open/high/low/close columns
        min_gap_atr: Minimum gap size as multiple of ATR to qualify

    Returns:
        List of dicts, each with keys:
        - idx: integer position where FVG is confirmed (3rd candle, i+1)
        - bar_index: DatetimeIndex label of the confirmation candle
        - type: "bullish" or "bearish"
        - zone_low: lower bound of FVG zone
        - zone_high: upper bound of FVG zone
        - size_atr: gap size normalised by ATR
    """
    atr = compute_atr(df)
    atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

    highs = df["high"].values
    lows = df["low"].values
    atr_vals = atr_safe.values
    fvgs: list[dict] = []

    for i in range(1, len(df) - 1):
        atr_i = atr_vals[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        # Bullish FVG: candle before's high < candle after's low
        # FVG is confirmed at bar i+1 (the 3rd candle that completes the gap)
        if highs[i - 1] < lows[i + 1]:
            gap_size = lows[i + 1] - highs[i - 1]
            if gap_size / atr_i >= min_gap_atr:
                fvgs.append({
                    "idx": i + 1,
                    "bar_index": df.index[i + 1],
                    "type": "bullish",
                    "zone_low": highs[i - 1],
                    "zone_high": lows[i + 1],
                    "size_atr": gap_size / atr_i,
                })

        # Bearish FVG: candle before's low > candle after's high
        if lows[i - 1] > highs[i + 1]:
            gap_size = lows[i - 1] - highs[i + 1]
            if gap_size / atr_i >= min_gap_atr:
                fvgs.append({
                    "idx": i + 1,
                    "bar_index": df.index[i + 1],
                    "type": "bearish",
                    "zone_low": highs[i + 1],
                    "zone_high": lows[i - 1],
                    "size_atr": gap_size / atr_i,
                })

    return fvgs


def track_fvg_fills(
    fvgs: list[dict],
    df: pd.DataFrame,
    max_age: int = 200,
) -> list[dict]:
    """Track which FVGs have been filled by subsequent price action.

    An FVG is "filled" when price closes inside the gap zone after
    the FVG formed.

    Args:
        fvgs: List of FVG dicts from :func:`detect_fvgs`
        df: Same DataFrame used for detection
        max_age: Expire FVGs older than this many bars

    Returns:
        Same list with added keys per FVG:
        - filled: bool
        - fill_idx: integer position where fill occurred (or -1)
    """
    closes = df["close"].values
    n = len(df)

    for fvg in fvgs:
        fvg["filled"] = False
        fvg["fill_idx"] = -1
        start = fvg["idx"] + 1  # check from 1 bar after confirmation
        end = min(fvg["idx"] + max_age, n)

        for j in range(start, end):
            if fvg["type"] == "bullish":
                # Filled when close enters the zone from above
                if closes[j] <= fvg["zone_high"]:
                    fvg["filled"] = True
                    fvg["fill_idx"] = j
                    break
            else:
                # Filled when close enters the zone from below
                if closes[j] >= fvg["zone_low"]:
                    fvg["filled"] = True
                    fvg["fill_idx"] = j
                    break

    return fvgs
