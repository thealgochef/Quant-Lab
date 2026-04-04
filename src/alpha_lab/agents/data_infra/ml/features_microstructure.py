"""
PL (Price Level) microstructure features from MBP-10 book data.

Extracts order book depth features at each extremum timestamp using
TickStore's 10-level bid/ask data. Features capture volume distribution,
trade activity, and book shape around the extremum price.

Based on PL feature set from Sokolovsky & Arnaboldi (2020).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import FeatureConfig
from alpha_lab.agents.data_infra.ml.extrema_detection import Extremum

_DEFAULT_CONFIG = FeatureConfig()


def extract_pl_features(
    extremum: Extremum,
    ticks_at_extremum: pd.DataFrame,
    config: FeatureConfig | None = None,
    tick_size: float = 0.25,
) -> dict[str, float]:
    """Extract PL (Price Level) microstructure features at an extremum.

    Uses the tick data snapshot at/near the extremum to compute order
    book depth features.

    Args:
        extremum: The detected price extremum.
        ticks_at_extremum: DataFrame of ticks around the extremum timestamp.
            Expected columns: bid_px_00..09, ask_px_00..09,
            bid_sz_00..09, ask_sz_00..09, price, size.
        config: Feature extraction config.
        tick_size: Instrument tick size.

    Returns:
        Dict of feature name -> value.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    features: dict[str, float] = {}

    if ticks_at_extremum.empty:
        return features

    features.update(_book_volume_features(ticks_at_extremum, extremum.price, tick_size))
    features.update(_trade_features(ticks_at_extremum))
    features.update(_book_ratio_features(ticks_at_extremum, tick_size))
    features.update(_depth_shape_features(ticks_at_extremum))
    features.update(_peak_morphology_features(extremum))

    return features


def _book_volume_features(
    ticks: pd.DataFrame, extremum_price: float, tick_size: float,
) -> dict[str, float]:
    """Volume aggregates around the extremum price."""
    features: dict[str, float] = {}
    last = ticks.iloc[-1]

    total_bid_vol = 0.0
    total_ask_vol = 0.0
    near_bid_vol = 0.0
    near_ask_vol = 0.0

    for i in range(10):
        bid_sz_col = f"bid_sz_{i:02d}"
        ask_sz_col = f"ask_sz_{i:02d}"
        bid_px_col = f"bid_px_{i:02d}"
        ask_px_col = f"ask_px_{i:02d}"

        if bid_sz_col not in last.index:
            break

        bsz = float(last[bid_sz_col]) if not pd.isna(last[bid_sz_col]) else 0.0
        asz = float(last[ask_sz_col]) if not pd.isna(last[ask_sz_col]) else 0.0
        total_bid_vol += bsz
        total_ask_vol += asz

        # Near-price volume (within 5 ticks of extremum)
        if bid_px_col in last.index and not pd.isna(last[bid_px_col]):
            bp = float(last[bid_px_col])
            if abs(bp - extremum_price) <= 5 * tick_size:
                near_bid_vol += bsz
        if ask_px_col in last.index and not pd.isna(last[ask_px_col]):
            ap = float(last[ask_px_col])
            if abs(ap - extremum_price) <= 5 * tick_size:
                near_ask_vol += asz

    features["pl_total_bid_vol"] = total_bid_vol
    features["pl_total_ask_vol"] = total_ask_vol
    features["pl_total_vol"] = total_bid_vol + total_ask_vol
    features["pl_near_bid_vol"] = near_bid_vol
    features["pl_near_ask_vol"] = near_ask_vol
    features["pl_near_vol"] = near_bid_vol + near_ask_vol

    return features


def _trade_features(ticks: pd.DataFrame) -> dict[str, float]:
    """Trade count and size features from recent ticks."""
    features: dict[str, float] = {}

    if "size" not in ticks.columns or "price" not in ticks.columns:
        return features

    sizes = ticks["size"].values
    prices = ticks["price"].values

    features["pl_trade_count"] = float(len(ticks))
    features["pl_total_trade_vol"] = float(np.nansum(sizes))
    features["pl_max_trade_size"] = float(np.nanmax(sizes)) if len(sizes) > 0 else 0.0
    features["pl_mean_trade_size"] = float(np.nanmean(sizes)) if len(sizes) > 0 else 0.0

    # Classify trades as bid/ask using tick rule
    if len(prices) >= 2:
        price_diff = np.diff(prices)
        # Uptick = buy (ask), downtick = sell (bid)
        buy_mask = np.append([False], price_diff > 0)
        sell_mask = np.append([False], price_diff < 0)
        features["pl_buy_vol"] = float(np.nansum(sizes[buy_mask]))
        features["pl_sell_vol"] = float(np.nansum(sizes[sell_mask]))
        features["pl_buy_count"] = float(np.sum(buy_mask))
        features["pl_sell_count"] = float(np.sum(sell_mask))

    return features


def _book_ratio_features(ticks: pd.DataFrame, tick_size: float = 0.25) -> dict[str, float]:
    """Bid/ask ratio and depth imbalance features."""
    features: dict[str, float] = {}
    last = ticks.iloc[-1]

    # Close-bid vs far-bid volume ratio
    close_bid = 0.0
    far_bid = 0.0
    close_ask = 0.0
    far_ask = 0.0

    for i in range(10):
        bid_sz_col = f"bid_sz_{i:02d}"
        ask_sz_col = f"ask_sz_{i:02d}"
        if bid_sz_col not in last.index:
            break

        bsz = float(last[bid_sz_col]) if not pd.isna(last[bid_sz_col]) else 0.0
        asz = float(last[ask_sz_col]) if not pd.isna(last[ask_sz_col]) else 0.0

        if i < 3:
            close_bid += bsz
            close_ask += asz
        else:
            far_bid += bsz
            far_ask += asz

    # Bid/ask volume ratio
    total_bid = close_bid + far_bid
    total_ask = close_ask + far_ask
    total = total_bid + total_ask
    if total > 0:
        features["pl_bid_ask_ratio"] = total_bid / total
    else:
        features["pl_bid_ask_ratio"] = 0.5

    # Close/far concentration
    if total_bid > 0:
        features["pl_close_bid_ratio"] = close_bid / total_bid
    else:
        features["pl_close_bid_ratio"] = 0.0
    if total_ask > 0:
        features["pl_close_ask_ratio"] = close_ask / total_ask
    else:
        features["pl_close_ask_ratio"] = 0.0

    # Spread
    bid_px_col = "bid_px_00"
    ask_px_col = "ask_px_00"
    if bid_px_col in last.index and ask_px_col in last.index:
        bp = float(last[bid_px_col]) if not pd.isna(last[bid_px_col]) else 0.0
        ap = float(last[ask_px_col]) if not pd.isna(last[ask_px_col]) else 0.0
        if bp > 0 and ap > 0:
            features["pl_spread"] = ap - bp
            features["pl_spread_ticks"] = (ap - bp) / tick_size

    return features


def _depth_shape_features(ticks: pd.DataFrame) -> dict[str, float]:
    """Volume concentration across book levels."""
    features: dict[str, float] = {}
    last = ticks.iloc[-1]

    bid_vols: list[float] = []
    ask_vols: list[float] = []

    for i in range(10):
        bid_sz_col = f"bid_sz_{i:02d}"
        ask_sz_col = f"ask_sz_{i:02d}"
        if bid_sz_col not in last.index:
            break
        bsz = float(last[bid_sz_col]) if not pd.isna(last[bid_sz_col]) else 0.0
        asz = float(last[ask_sz_col]) if not pd.isna(last[ask_sz_col]) else 0.0
        bid_vols.append(bsz)
        ask_vols.append(asz)

    if bid_vols:
        bid_arr = np.array(bid_vols)
        ask_arr = np.array(ask_vols)
        total_bid = bid_arr.sum()
        total_ask = ask_arr.sum()

        # Concentration: fraction in top 3 levels
        if total_bid > 0:
            features["pl_bid_top3_concentration"] = bid_arr[:3].sum() / total_bid
        if total_ask > 0:
            features["pl_ask_top3_concentration"] = ask_arr[:3].sum() / total_ask

        # Depth slope (linear regression coefficient on cumulative volume)
        if total_bid > 0:
            cum_bid = np.cumsum(bid_arr) / total_bid
            x = np.arange(len(cum_bid))
            if len(x) >= 2:
                slope = np.polyfit(x, cum_bid, 1)[0]
                features["pl_bid_depth_slope"] = float(slope)
        if total_ask > 0:
            cum_ask = np.cumsum(ask_arr) / total_ask
            x = np.arange(len(cum_ask))
            if len(x) >= 2:
                slope = np.polyfit(x, cum_ask, 1)[0]
                features["pl_ask_depth_slope"] = float(slope)

    return features


def _peak_morphology_features(extremum: Extremum) -> dict[str, float]:
    """Features describing the shape of the extremum itself."""
    return {
        "pl_extremum_type": 1.0 if extremum.extremum_type == "peak" else 0.0,
        "pl_prominence": extremum.prominence,
        "pl_width": extremum.width,
    }
