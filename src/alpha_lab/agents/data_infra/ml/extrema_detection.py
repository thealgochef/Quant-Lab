"""
Price extrema detection using scipy peak-finding.

Identifies local price peaks (tops) and troughs (bottoms) in tick-level
price series. Uses a sliding window approach with configurable prominence
and width constraints.

This is tick-level peak-finding (not bar-level pivot detection like
``indicators.compute_swing_highs/lows``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from alpha_lab.agents.data_infra.ml.config import ExtremaConfig

_DEFAULT_CONFIG = ExtremaConfig()


@dataclass
class Extremum:
    """A single detected price extremum (peak or trough)."""

    index: int
    timestamp: pd.Timestamp
    price: float
    extremum_type: str  # "peak" | "trough"
    prominence: float
    width: float


def detect_extrema(
    tick_prices: pd.Series,
    tick_timestamps: pd.Series,
    config: ExtremaConfig | None = None,
    tick_size: float = 0.25,
) -> list[Extremum]:
    """Detect price peaks and troughs in a tick-level price series.

    Slides a window across the price series, runs ``scipy.signal.find_peaks``
    with prominence and width constraints, then deduplicates across windows.

    Args:
        tick_prices: Series of tick prices (numeric, no NaNs expected).
        tick_timestamps: Series of timestamps aligned to tick_prices.
        config: Detection parameters. Defaults to ExtremaConfig().
        tick_size: Instrument tick size for prominence scaling.

    Returns:
        List of Extremum objects sorted by index.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    prices = tick_prices.values.astype(float)
    n = len(prices)

    if n < config.window_size:
        # Not enough data for even one window — run on entire series
        return _detect_in_window(
            prices, tick_timestamps, 0, n, config, tick_size,
        )

    # Sliding window with 50% overlap
    step = config.window_size // 2
    raw_extrema: list[Extremum] = []

    for start in range(0, n - config.window_size + 1, step):
        end = min(start + config.window_size, n)
        window_extrema = _detect_in_window(
            prices, tick_timestamps, start, end, config, tick_size,
        )
        raw_extrema.extend(window_extrema)

    # Handle tail if last window didn't cover it
    last_start = ((n - config.window_size) // step) * step
    last_end = last_start + config.window_size
    if last_end < n:
        tail_extrema = _detect_in_window(
            prices, tick_timestamps, last_end - config.window_size, n,
            config, tick_size,
        )
        raw_extrema.extend(tail_extrema)

    # Deduplicate: keep highest-prominence extremum within dedup_window
    return _deduplicate(raw_extrema, config.dedup_window)


def _detect_in_window(
    prices: np.ndarray,
    timestamps: pd.Series,
    start: int,
    end: int,
    config: ExtremaConfig,
    tick_size: float,
) -> list[Extremum]:
    """Run peak/trough detection on a single window slice."""
    window = prices[start:end]
    if len(window) < 3:
        return []

    min_prom = config.min_prominence_ticks * tick_size
    width_range = (
        min(config.min_peak_width, len(window) - 1),
        min(config.max_peak_width, len(window) - 1),
    )

    results: list[Extremum] = []

    # Detect peaks (local maxima)
    peak_idx, peak_props = find_peaks(
        window,
        prominence=min_prom,
        width=width_range,
    )
    for i, idx in enumerate(peak_idx):
        abs_idx = start + idx
        results.append(Extremum(
            index=abs_idx,
            timestamp=pd.Timestamp(timestamps.iloc[abs_idx]),
            price=float(prices[abs_idx]),
            extremum_type="peak",
            prominence=float(peak_props["prominences"][i]),
            width=float(peak_props["widths"][i]),
        ))

    # Detect troughs (local minima) by negating prices
    trough_idx, trough_props = find_peaks(
        -window,
        prominence=min_prom,
        width=width_range,
    )
    for i, idx in enumerate(trough_idx):
        abs_idx = start + idx
        results.append(Extremum(
            index=abs_idx,
            timestamp=pd.Timestamp(timestamps.iloc[abs_idx]),
            price=float(prices[abs_idx]),
            extremum_type="trough",
            prominence=float(trough_props["prominences"][i]),
            width=float(trough_props["widths"][i]),
        ))

    return results


def _deduplicate(
    extrema: list[Extremum], dedup_window: int,
) -> list[Extremum]:
    """Deduplicate overlapping detections by keeping highest prominence.

    Within each group of extrema of the same type whose indices are within
    ``dedup_window`` of each other, keep only the one with highest prominence.
    """
    if not extrema:
        return []

    # Sort by type then index
    extrema.sort(key=lambda e: (e.extremum_type, e.index))

    result: list[Extremum] = []
    for etype in ("peak", "trough"):
        typed = [e for e in extrema if e.extremum_type == etype]
        if not typed:
            continue

        # Group nearby detections
        groups: list[list[Extremum]] = [[typed[0]]]
        for e in typed[1:]:
            if e.index - groups[-1][-1].index <= dedup_window:
                groups[-1].append(e)
            else:
                groups.append([e])

        # Keep best from each group
        for group in groups:
            best = max(group, key=lambda e: e.prominence)
            result.append(best)

    result.sort(key=lambda e: e.index)
    return result
