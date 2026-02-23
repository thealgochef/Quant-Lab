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
        # First pass: compute EMA direction for each available TF
        tf_directions: dict[str, int] = {}
        tf_dataframes: dict[str, pd.DataFrame] = {}

        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            tf_dataframes[tf] = df
            close = df["close"]
            ema_f = compute_ema(close, self.ema_fast)
            ema_m = compute_ema(close, min(self.ema_mid, len(df) - 1))

            # Use last value for cross-TF voting
            bull = ema_f.iloc[-1] > ema_m.iloc[-1]
            bear = ema_f.iloc[-1] < ema_m.iloc[-1]
            tf_directions[tf] = 1 if bull else (-1 if bear else 0)

        if len(tf_dataframes) < self.min_timeframes:
            return []

        n_tfs = len(tf_directions)

        # Second pass: produce signals per TF using cross-TF agreement
        signals: list[SignalVector] = []
        for tf, df in tf_dataframes.items():
            sv = self._compute_timeframe(df, tf, data.instrument, tf_directions, n_tfs)
            if sv is not None:
                signals.append(sv)

        return signals

    def _compute_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        instrument: str,
        tf_directions: dict[str, int],
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

        # Cross-TF vote (static for the analysis window)
        other_dirs = [d for tf, d in tf_directions.items() if tf != timeframe and d != 0]
        n_bull_others = sum(1 for d in other_dirs if d > 0)
        n_bear_others = sum(1 for d in other_dirs if d < 0)

        # Direction: local direction confirmed by cross-TF majority
        n_others = len(other_dirs)
        if n_others > 0:
            agreement_bull = n_bull_others / n_others
            agreement_bear = n_bear_others / n_others
        else:
            agreement_bull = 0.0
            agreement_bear = 0.0

        # Direction = local, but weaken if cross-TF disagrees
        direction = local_dir.copy()
        # If majority of other TFs disagree, set to neutral
        if agreement_bear > 0.5:
            direction = direction.where(direction != 1, 0)
        if agreement_bull > 0.5:
            direction = direction.where(direction != -1, 0)

        # Strength component 1: agreement ratio
        agreement_ratio = max(agreement_bull, agreement_bear)

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
                "timeframes_available": list(tf_directions.keys()),
                "n_timeframes": len(tf_directions),
            },
        )
