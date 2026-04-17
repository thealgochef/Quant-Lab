"""
Dashboard-utility labeling for level-touch events.

Assigns 3-class labels aligned to the dashboard execution semantics:
- tradeable_reversal (0): MFE >= tp_points before MAE >= sl_points
- trap_reversal (1):      MAE >= sl_points and MFE >= trap_mfe_min
- aggressive_blowthrough (2): MAE >= sl_points and MFE < trap_mfe_min

Resolution order: MAE checked first (conservative), matching both
Quant-Lab experiment/labeling.py and Trading-Dashboard outcome_tracker.py.

Entry reference: representative_price (level price), matching training labels.
"""

from __future__ import annotations

import logging
from datetime import time

import pandas as pd

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig

logger = logging.getLogger(__name__)

# Label constants (matching experiment path and Trading-Dashboard)
TRADEABLE_REVERSAL = "tradeable_reversal"
TRAP_REVERSAL = "trap_reversal"
AGGRESSIVE_BLOWTHROUGH = "aggressive_blowthrough"
NO_RESOLUTION = "no_resolution"

LABEL_ENCODING = {
    TRADEABLE_REVERSAL: 0,
    TRAP_REVERSAL: 1,
    AGGRESSIVE_BLOWTHROUGH: 2,
}

CLASS_NAMES = {v: k for k, v in LABEL_ENCODING.items()}

RTH_END = time(16, 15)
_ET = "US/Eastern"


def label_touch_event(
    event: dict | pd.Series,
    forward_bars: pd.DataFrame,
    config: DashboardUtilityConfig,
) -> dict:
    """Label a single touch event by scanning forward bars for MFE/MAE.

    Entry price is ``representative_price`` (the level itself).
    MAE checked first each bar (conservative ordering).

    Args:
        event: Row with at least ``representative_price`` and ``direction``.
        forward_bars: 1-min bars AFTER event (touch bar excluded).
        config: Utility config with TP/SL thresholds.

    Returns:
        Dict with label, label_encoded, max_mfe, max_mae, resolution_ts.
    """
    entry_price = event["representative_price"]
    direction = event["direction"]

    max_mfe = 0.0
    max_mae = 0.0
    label = NO_RESOLUTION
    resolution_ts = pd.NaT
    bars_to_resolution = -1

    for i, (bar_ts, bar) in enumerate(forward_bars.iterrows()):
        if direction == "LONG":
            bar_mfe = bar["high"] - entry_price
            bar_mae = entry_price - bar["low"]
        else:  # SHORT
            bar_mfe = entry_price - bar["low"]
            bar_mae = bar["high"] - entry_price

        max_mfe = max(max_mfe, bar_mfe)
        max_mae = max(max_mae, bar_mae)

        # Check adverse FIRST (conservative, matches experiment/labeling.py)
        if max_mae >= config.sl_points:
            if max_mfe >= config.trap_mfe_min:
                label = TRAP_REVERSAL
            else:
                label = AGGRESSIVE_BLOWTHROUGH
            resolution_ts = bar_ts
            bars_to_resolution = i
            break

        # Then check favorable
        if max_mfe >= config.tp_points:
            label = TRADEABLE_REVERSAL
            resolution_ts = bar_ts
            bars_to_resolution = i
            break

    label_encoded = LABEL_ENCODING.get(label)

    return {
        "label": label,
        "label_encoded": label_encoded,
        "max_mfe": round(max_mfe, 4),
        "max_mae": round(max_mae, 4),
        "resolution_ts": resolution_ts,
        "bars_to_resolution": bars_to_resolution,
    }
