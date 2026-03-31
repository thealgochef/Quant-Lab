"""
Rebound/crossing label assignment for price extrema.

For each detected extremum, looks forward in the tick series and classifies:
- **Rebound (1)**: Price reverses by N ticks before continuing by M ticks
- **Crossing (0)**: Price continues by M ticks before reversing by N ticks
- **Ambiguous (None)**: Neither condition met within the forward window

Generates labels at multiple threshold levels (e.g. 7, 11, 15 ticks)
for multi-resolution analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import LabelingConfig
from alpha_lab.agents.data_infra.ml.extrema_detection import Extremum

_DEFAULT_CONFIG = LabelingConfig()


@dataclass
class LabeledExtremum:
    """An extremum with assigned rebound/crossing labels."""

    extremum: Extremum
    labels: dict[str, int | None]  # e.g. {"label_7t": 1, "label_11t": 0, "label_15t": None}
    reversal_ticks: float  # Max reversal observed within forward window
    continuation_ticks: float  # Max continuation observed within forward window


def label_extrema(
    extrema: list[Extremum],
    tick_prices: pd.Series,
    config: LabelingConfig | None = None,
    tick_size: float = 0.25,
) -> list[LabeledExtremum]:
    """Assign rebound/crossing labels to each extremum.

    For each extremum, scans forward up to ``config.forward_window`` ticks.
    At each threshold in ``config.rebound_thresholds``, determines whether
    a rebound or crossing occurred first.

    Args:
        extrema: Detected price extrema.
        tick_prices: Full tick price series.
        config: Labeling parameters.
        tick_size: Instrument tick size.

    Returns:
        List of LabeledExtremum with labels at each threshold.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    prices = tick_prices.values.astype(float)
    n = len(prices)
    results: list[LabeledExtremum] = []

    for ext in extrema:
        if ext.index >= n - 1:
            continue

        fw_end = min(ext.index + config.forward_window + 1, n)
        fw_prices = prices[ext.index + 1: fw_end]

        if len(fw_prices) == 0:
            results.append(LabeledExtremum(
                extremum=ext,
                labels={f"label_{t}t": None for t in config.rebound_thresholds},
                reversal_ticks=0.0,
                continuation_ticks=0.0,
            ))
            continue

        labels, rev_ticks, cont_ticks = _classify_forward(
            ext.price, ext.extremum_type, fw_prices,
            config.rebound_thresholds, config.crossing_threshold,
            tick_size,
        )

        results.append(LabeledExtremum(
            extremum=ext,
            labels=labels,
            reversal_ticks=rev_ticks,
            continuation_ticks=cont_ticks,
        ))

    return results


def _classify_forward(
    extremum_price: float,
    extremum_type: str,
    forward_prices: np.ndarray,
    rebound_thresholds: list[int],
    crossing_threshold: int,
    tick_size: float,
) -> tuple[dict[str, int | None], float, float]:
    """Classify forward price action as rebound or crossing.

    For a **peak**: rebound = price drops by N ticks; crossing = price rises by M ticks.
    For a **trough**: rebound = price rises by N ticks; crossing = price drops by M ticks.

    Returns:
        (labels_dict, max_reversal_ticks, max_continuation_ticks)
    """
    if extremum_type == "peak":
        # Reversal = price goes down; continuation = price goes up
        deltas = forward_prices - extremum_price
        reversal_deltas = -deltas  # positive when price drops
        continuation_deltas = deltas  # positive when price rises
    else:
        # Trough: reversal = price goes up; continuation = price goes down
        deltas = forward_prices - extremum_price
        reversal_deltas = deltas  # positive when price rises
        continuation_deltas = -deltas  # positive when price drops

    # Convert to tick units
    reversal_in_ticks = reversal_deltas / tick_size
    continuation_in_ticks = continuation_deltas / tick_size

    max_reversal = float(np.max(reversal_in_ticks)) if len(reversal_in_ticks) > 0 else 0.0
    max_continuation = (
        float(np.max(continuation_in_ticks)) if len(continuation_in_ticks) > 0 else 0.0
    )

    crossing_level = crossing_threshold

    labels: dict[str, int | None] = {}
    for threshold in rebound_thresholds:
        label = _label_at_threshold(
            reversal_in_ticks, continuation_in_ticks,
            threshold, crossing_level,
        )
        labels[f"label_{threshold}t"] = label

    return labels, max_reversal, max_continuation


def _label_at_threshold(
    reversal_in_ticks: np.ndarray,
    continuation_in_ticks: np.ndarray,
    rebound_threshold: int,
    crossing_threshold: int,
) -> int | None:
    """Determine label at a single rebound threshold.

    Scans forward tick-by-tick. The first condition to be met wins:
    - If reversal reaches ``rebound_threshold`` before continuation
      reaches ``crossing_threshold`` → rebound (1)
    - If continuation reaches ``crossing_threshold`` before reversal
      reaches ``rebound_threshold`` → crossing (0)
    - If neither met → ambiguous (None)
    """
    for i in range(len(reversal_in_ticks)):
        rev_hit = reversal_in_ticks[i] >= rebound_threshold
        cont_hit = continuation_in_ticks[i] >= crossing_threshold

        if rev_hit and cont_hit:
            # Both thresholds hit on same tick — rebound wins (conservative)
            return 1
        if rev_hit:
            return 1
        if cont_hit:
            return 0

    return None


def build_label_dataframe(
    labeled_extrema: list[LabeledExtremum],
) -> pd.DataFrame:
    """Convert labeled extrema into a DataFrame suitable for ML.

    Each row represents one extremum with its labels and metadata.

    Returns:
        DataFrame with columns: index, timestamp, price, type, prominence,
        width, reversal_ticks, continuation_ticks, label_Nt (per threshold).
    """
    rows: list[dict] = []
    for le in labeled_extrema:
        row = {
            "tick_index": le.extremum.index,
            "timestamp": le.extremum.timestamp,
            "price": le.extremum.price,
            "extremum_type": le.extremum.extremum_type,
            "prominence": le.extremum.prominence,
            "width": le.extremum.width,
            "reversal_ticks": le.reversal_ticks,
            "continuation_ticks": le.continuation_ticks,
        }
        row.update(le.labels)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)
