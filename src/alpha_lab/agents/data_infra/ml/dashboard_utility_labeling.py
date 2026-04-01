"""Dashboard utility labeling (+15/-30) with explicit executable semantics.

Execution approximation used in this module:
- LONG events enter at best ask, exit checks use best bid.
- SHORT events enter at best bid, exit checks use best ask.

This module is mode-specific and intentionally separate from extrema
rebound/crossing labeling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_event_15_30(
    *,
    direction: str,
    event_ts: pd.Timestamp,
    forward_ticks: pd.DataFrame,
    tick_size: float = 0.25,
    tp_ticks: int = 15,
    sl_ticks: int = 30,
) -> dict[str, object]:
    """Label one touch-anchored event with +15/-30 utility semantics."""
    if forward_ticks.empty:
        return {
            "label_15_30": None,
            "outcome_type": "no_data",
            "time_to_tp": None,
            "time_to_sl": None,
            "mfe": 0.0,
            "mae": 0.0,
            "entry_price": None,
        }

    first = forward_ticks.iloc[0]
    is_long = direction.upper() == "LONG"

    if is_long:
        entry_price = float(first.get("ask_px_00", first.get("price", 0.0)))
        exit_series = forward_ticks.get("bid_px_00", forward_ticks["price"]).astype(float)
        tp_price = entry_price + tp_ticks * tick_size
        sl_price = entry_price - sl_ticks * tick_size
        tp_hit = exit_series >= tp_price
        sl_hit = exit_series <= sl_price
        mfe = float((exit_series - entry_price).max()) if len(exit_series) else 0.0
        mae = float((entry_price - exit_series).max()) if len(exit_series) else 0.0
    else:
        entry_price = float(first.get("bid_px_00", first.get("price", 0.0)))
        exit_series = forward_ticks.get("ask_px_00", forward_ticks["price"]).astype(float)
        tp_price = entry_price - tp_ticks * tick_size
        sl_price = entry_price + sl_ticks * tick_size
        tp_hit = exit_series <= tp_price
        sl_hit = exit_series >= sl_price
        mfe = float((entry_price - exit_series).max()) if len(exit_series) else 0.0
        mae = float((exit_series - entry_price).max()) if len(exit_series) else 0.0

    tp_idx = int(np.where(tp_hit.values)[0][0]) if tp_hit.any() else None
    sl_idx = int(np.where(sl_hit.values)[0][0]) if sl_hit.any() else None

    if tp_idx is not None and (sl_idx is None or tp_idx <= sl_idx):
        label = 1
        outcome = "tp"
        tp_ts = forward_ticks.iloc[tp_idx]["ts_event"]
        sl_ts = None
    elif sl_idx is not None:
        label = 0
        outcome = "sl"
        tp_ts = None
        sl_ts = forward_ticks.iloc[sl_idx]["ts_event"]
    else:
        label = 0
        outcome = "session_end"
        tp_ts = None
        sl_ts = None

    return {
        "label_15_30": label,
        "outcome_type": outcome,
        "time_to_tp": None if tp_ts is None else (pd.Timestamp(tp_ts) - event_ts).total_seconds(),
        "time_to_sl": None if sl_ts is None else (pd.Timestamp(sl_ts) - event_ts).total_seconds(),
        "mfe": mfe,
        "mae": mae,
        "entry_price": entry_price,
    }
