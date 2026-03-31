"""
Walk-forward cross-validation splitter.

Provides time-series aware train/test splits that prevent data leakage.
Supports both rolling and expanding window modes with configurable gap
between train and test periods.

Compatible with sklearn's cross-validator interface for use with RFECV.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import WalkForwardConfig

_DEFAULT_CONFIG = WalkForwardConfig()


@dataclass
class WalkForwardSplit:
    """A single train/test split in walk-forward validation."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_indices: np.ndarray
    test_indices: np.ndarray


class WalkForwardSplitter:
    """Walk-forward cross-validation for time series data.

    Generates non-overlapping train/test splits that move forward
    through time. Each test period immediately follows (with optional gap)
    the training period.

    Args:
        config: Walk-forward configuration.
    """

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self._config = config or _DEFAULT_CONFIG

    def split(
        self, timestamps: pd.Series,
    ) -> list[WalkForwardSplit]:
        """Generate walk-forward splits from timestamps.

        Args:
            timestamps: Series of timestamps for each sample,
                sorted chronologically.

        Returns:
            List of WalkForwardSplit objects.
        """
        cfg = self._config

        if len(timestamps) == 0:
            return []

        ts = pd.to_datetime(timestamps)
        min_ts = ts.min()
        max_ts = ts.max()

        train_delta = timedelta(days=cfg.train_days)
        test_delta = timedelta(days=cfg.test_days)
        gap_delta = timedelta(days=cfg.gap_days)

        splits: list[WalkForwardSplit] = []
        fold = 0

        if cfg.expanding:
            # Expanding window: train_start is always min_ts
            train_start = min_ts
            test_start = min_ts + train_delta + gap_delta

            while test_start + test_delta <= max_ts + timedelta(days=1):
                train_end = test_start - gap_delta - timedelta(microseconds=1)
                test_end = test_start + test_delta - timedelta(microseconds=1)

                train_mask = (ts >= train_start) & (ts <= train_end)
                test_mask = (ts >= test_start) & (ts <= test_end)

                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]

                if len(train_idx) > 0 and len(test_idx) > 0:
                    splits.append(WalkForwardSplit(
                        fold=fold,
                        train_start=pd.Timestamp(train_start),
                        train_end=pd.Timestamp(train_end),
                        test_start=pd.Timestamp(test_start),
                        test_end=pd.Timestamp(test_end),
                        train_indices=train_idx,
                        test_indices=test_idx,
                    ))
                    fold += 1

                test_start += test_delta
        else:
            # Rolling window: fixed-size train window
            train_start = min_ts

            while True:
                train_end = train_start + train_delta - timedelta(microseconds=1)
                test_start = train_end + gap_delta + timedelta(microseconds=1)
                test_end = test_start + test_delta - timedelta(microseconds=1)

                if test_end > max_ts + timedelta(days=1):
                    break

                train_mask = (ts >= train_start) & (ts <= train_end)
                test_mask = (ts >= test_start) & (ts <= test_end)

                train_idx = np.where(train_mask)[0]
                test_idx = np.where(test_mask)[0]

                if len(train_idx) > 0 and len(test_idx) > 0:
                    splits.append(WalkForwardSplit(
                        fold=fold,
                        train_start=pd.Timestamp(train_start),
                        train_end=pd.Timestamp(train_end),
                        test_start=pd.Timestamp(test_start),
                        test_end=pd.Timestamp(test_end),
                        train_indices=train_idx,
                        test_indices=test_idx,
                    ))
                    fold += 1

                train_start += test_delta

        return splits

    def as_sklearn_cv(
        self, timestamps: pd.Series,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return sklearn-compatible (train_indices, test_indices) tuples.

        Suitable for use with ``sklearn.feature_selection.RFECV``.

        Args:
            timestamps: Series of timestamps for each sample.

        Returns:
            List of (train_idx, test_idx) array tuples.
        """
        splits = self.split(timestamps)
        return [(s.train_indices, s.test_indices) for s in splits]

    def get_n_splits(self, timestamps: pd.Series) -> int:
        """Return the number of splits."""
        return len(self.split(timestamps))
