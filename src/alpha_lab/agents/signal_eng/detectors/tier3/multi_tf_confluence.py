"""
Category 11: Multi-Timeframe Confluence Score.

Measures signal agreement across all timeframes.
Higher score = more timeframes agree on direction.

Signal composition:
- direction: +1 (majority bullish), -1 (majority bearish), 0 (tied)
- strength: combines agreement ratio across TFs with local EMA spread rank
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr, compute_ema
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 50


class MultiTFConfluenceDetector(SignalDetector):
    """Multi-TF Confluence: agreement score across all timeframes."""

    detector_id = "multi_tf_confluence"
    category = "multi_tf_confluence"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in Timeframe]

    def __init__(
        self,
        ema_fast: int = 13,
        ema_mid: int = 48,
        ema_slow: int = 200,
        min_timeframes: int = 2,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.min_timeframes = min_timeframes

    def validate_inputs(self, data: DataBundle) -> bool:
        count = 0
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > _MIN_BARS:
                count += 1
        return count >= self.min_timeframes

    def compute(self, data: DataBundle) -> list[SignalVector]:
        # First pass: compute per-bar EMA direction Series for each TF
        tf_dir_series: dict[str, pd.Series] = {}
        tf_dataframes: dict[str, pd.DataFrame] = {}

        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            tf_dataframes[tf] = df
            close = df["close"]
            ema_f = compute_ema(close, self.ema_fast)
            ema_m = compute_ema(close, min(self.ema_mid, len(df) - 1))

            # Per-bar direction: no look-ahead
            bullish = ema_f > ema_m
            bearish = ema_f < ema_m
            dir_series = pd.Series(0, index=df.index, dtype=int)
            dir_series = dir_series.where(~bullish, 1)
            dir_series = dir_series.where(~bearish, -1)
            tf_dir_series[tf] = dir_series

        if len(tf_dataframes) < self.min_timeframes:
            return []

        n_tfs = len(tf_dir_series)

        # Second pass: produce signals per TF using cross-TF agreement
        signals: list[SignalVector] = []
        for tf, df in tf_dataframes.items():
            sv = self._compute_timeframe(
                df, tf, data.instrument, tf_dir_series, n_tfs,
            )
            if sv is not None:
                signals.append(sv)

        return signals

    def _compute_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        instrument: str,
        tf_dir_series: dict[str, pd.Series],
        n_tfs: int,
    ) -> SignalVector | None:
        close = df["close"]
        ema_f = compute_ema(close, self.ema_fast)
        ema_m = compute_ema(close, min(self.ema_mid, len(df) - 1))

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # Local EMA alignment per bar
        bullish = ema_f > ema_m
        bearish = ema_f < ema_m
        local_dir = pd.Series(0, index=df.index, dtype=int)
        local_dir = local_dir.where(~bullish, 1)
        local_dir = local_dir.where(~bearish, -1)

        # Cross-TF vote: align each other TF's per-bar direction to this TF's index
        # using merge_asof so each bar sees only the most recent completed higher-TF bar
        other_tfs = [tf for tf in tf_dir_series if tf != timeframe]
        n_bull = pd.Series(0, index=df.index, dtype=int)
        n_bear = pd.Series(0, index=df.index, dtype=int)
        n_others_counted = 0

        for otf in other_tfs:
            otf_dir = tf_dir_series[otf]
            # Build a tz-naive index for merge_asof alignment
            has_tz = hasattr(df.index, "tz") and df.index.tz
            left_idx = df.index.tz_localize(None) if has_tz else df.index
            has_tz_r = hasattr(otf_dir.index, "tz") and otf_dir.index.tz
            right_idx = (
                otf_dir.index.tz_localize(None) if has_tz_r
                else otf_dir.index
            )

            left_df = pd.DataFrame({"_key": 0}, index=left_idx)
            right_df = pd.DataFrame({"dir": otf_dir.values}, index=right_idx)

            aligned = pd.merge_asof(
                left_df, right_df,
                left_index=True, right_index=True,
                direction="backward",
            )
            aligned_dir = aligned["dir"].fillna(0).astype(int).values
            n_bull += (aligned_dir > 0).astype(int)
            n_bear += (aligned_dir < 0).astype(int)
            n_others_counted += 1

        if n_others_counted > 0:
            agreement_bull = n_bull / n_others_counted
            agreement_bear = n_bear / n_others_counted
        else:
            agreement_bull = pd.Series(0.0, index=df.index)
            agreement_bear = pd.Series(0.0, index=df.index)

        # Direction = local, but weaken if cross-TF disagrees per bar
        direction = local_dir.copy()
        direction = direction.where(~((agreement_bear > 0.5) & (direction == 1)), 0)
        direction = direction.where(~((agreement_bull > 0.5) & (direction == -1)), 0)

        # Strength component 1: per-bar agreement ratio
        agreement_ratio = pd.concat([agreement_bull, agreement_bear], axis=1).max(axis=1)

        # Strength component 2: local EMA spread rank
        spread = (ema_f - ema_m).abs() / atr_safe
        spread_rank = spread.rolling(window=min(100, len(df))).rank(pct=True).fillna(0.0)

        strength = (0.60 * agreement_ratio + 0.40 * spread_rank).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        # Formation index
        dir_change = direction.diff().fillna(0) != 0
        formation_idx = pd.Series(np.nan, index=df.index)
        formation_idx[dir_change] = np.arange(len(df))[dir_change.values]
        formation_idx = formation_idx.ffill().fillna(0).astype(int)

        return SignalVector(
            signal_id=f"SIG_MULTI_TF_CONFLUENCE_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "ema_fast": self.ema_fast,
                "ema_mid": self.ema_mid,
                "ema_slow": self.ema_slow,
                "min_timeframes": self.min_timeframes,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Cross-TF EMA agreement voting with local confirmation",
                "bars_processed": len(df),
                "timeframes_available": list(tf_dir_series.keys()),
                "n_timeframes": len(tf_dir_series),
            },
        )
