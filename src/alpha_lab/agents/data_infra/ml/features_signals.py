"""
Signal detector features for the hybrid ML pipeline.

Maps each extremum's tick timestamp to the nearest bar boundary, then
extracts direction and strength from all available SignalVectors.
Produces features like ``sig_ema_confluence_5m_direction`` and
``sig_ema_confluence_5m_strength``.

This is the hybrid component that combines microstructure features
with existing detector outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import FeatureConfig
from alpha_lab.agents.data_infra.ml.extrema_detection import Extremum
from alpha_lab.core.contracts import SignalBundle, SignalVector

_DEFAULT_CONFIG = FeatureConfig()


def extract_signal_features(
    extremum: Extremum,
    signal_bundle: SignalBundle,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    """Extract signal detector features at an extremum.

    For each SignalVector in the bundle, finds the bar corresponding to
    the extremum timestamp (using backward-looking alignment to prevent
    look-ahead) and extracts direction + strength.

    Args:
        extremum: The detected price extremum.
        signal_bundle: Bundle containing all SignalVectors.
        config: Feature extraction config.

    Returns:
        Dict of feature name -> value.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    features: dict[str, float] = {}
    ts = extremum.timestamp

    for sv in signal_bundle.signals:
        prefix = f"sig_{sv.category}_{sv.timeframe}"

        direction_val, strength_val = _lookup_signal_at_time(sv, ts)
        features[f"{prefix}_direction"] = direction_val
        features[f"{prefix}_strength"] = strength_val

    return features


def extract_signal_features_batch(
    extrema: list[Extremum],
    signal_bundle: SignalBundle,
    config: FeatureConfig | None = None,
) -> list[dict[str, float]]:
    """Extract signal features for a batch of extrema.

    More efficient than calling extract_signal_features one at a time
    because it pre-indexes each SignalVector once.

    Args:
        extrema: List of detected extrema.
        signal_bundle: Bundle containing all SignalVectors.
        config: Feature extraction config.

    Returns:
        List of feature dicts, one per extremum.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    if not extrema or not signal_bundle.signals:
        return [{} for _ in extrema]

    # Pre-index each signal vector's direction and strength by timestamp
    sv_indices: list[tuple[str, pd.DatetimeIndex, np.ndarray, np.ndarray]] = []
    for sv in signal_bundle.signals:
        prefix = f"sig_{sv.category}_{sv.timeframe}"

        direction = sv.direction
        strength = sv.strength

        if isinstance(direction, pd.Series) and isinstance(strength, pd.Series):
            sv_indices.append((
                prefix, direction.index, direction.values, strength.values,
            ))

    results: list[dict[str, float]] = []
    for ext in extrema:
        features: dict[str, float] = {}
        ts = ext.timestamp

        for prefix, dt_idx, dir_vals, str_vals in sv_indices:
            # Find the latest bar at or before the extremum timestamp
            bar_idx = _find_bar_index(dt_idx, ts)
            if bar_idx is not None and 0 <= bar_idx < len(dir_vals):
                features[f"{prefix}_direction"] = float(dir_vals[bar_idx])
                features[f"{prefix}_strength"] = float(str_vals[bar_idx])
            else:
                features[f"{prefix}_direction"] = 0.0
                features[f"{prefix}_strength"] = 0.0

        results.append(features)

    return results


def _lookup_signal_at_time(
    sv: SignalVector, ts: pd.Timestamp,
) -> tuple[float, float]:
    """Look up direction and strength at a given timestamp.

    Uses backward-looking alignment: finds the most recent bar at or
    before the given timestamp.
    """
    direction = sv.direction
    strength = sv.strength

    if not isinstance(direction, pd.Series) or not isinstance(strength, pd.Series):
        return 0.0, 0.0

    idx = direction.index
    bar_idx = _find_bar_index(idx, ts)

    if bar_idx is not None and 0 <= bar_idx < len(direction):
        d = float(direction.iloc[bar_idx])
        s = float(strength.iloc[bar_idx])
        return d, s

    return 0.0, 0.0


def _find_bar_index(
    bar_index: pd.Index, timestamp: pd.Timestamp,
) -> int | None:
    """Find the index of the latest bar at or before timestamp.

    Uses searchsorted for O(log n) lookup with no look-ahead.
    """
    if len(bar_index) == 0:
        return None

    # Handle timezone alignment
    ts = timestamp
    if hasattr(bar_index, "tz") and bar_index.tz is not None:
        if ts.tz is None:
            ts = ts.tz_localize(bar_index.tz)
        elif ts.tz != bar_index.tz:
            ts = ts.tz_convert(bar_index.tz)
    elif ts.tz is not None:
        ts = ts.tz_localize(None)

    pos = bar_index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return int(pos)
