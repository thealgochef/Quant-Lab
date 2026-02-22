"""
Category 1: EMA Confluence Detector.

Detects multi-EMA alignment (13/48/200), crossover velocity, and spread.
Bullish when fast > mid > slow with expanding spread.
Bearish when fast < mid < slow with expanding spread.

Signal composition:
- direction: +1 (bullish alignment), -1 (bearish alignment), 0 (mixed)
- strength: combines alignment score, spread expansion, and crossover velocity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr, compute_ema
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

# Default EMA spans
_EMA_FAST = 13
_EMA_MID = 48
_EMA_SLOW = 200


class EmaConfluenceDetector(SignalDetector):
    """EMA Confluence: 13/48/200 alignment, crossover velocity, spread."""

    detector_id = "ema_confluence"
    category = "ema_confluence"
    tier = SignalTier.CORE
    timeframes = [
        tf.value
        for tf in [
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1,
        ]
    ]

    def __init__(
        self,
        ema_fast: int = _EMA_FAST,
        ema_mid: int = _EMA_MID,
        ema_slow: int = _EMA_SLOW,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow

    def validate_inputs(self, data: DataBundle) -> bool:
        """Need at least one timeframe with enough bars for EMA 200."""
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > self.ema_slow:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        """Compute EMA confluence signals across all timeframes."""
        signals: list[SignalVector] = []

        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= self.ema_slow:
                continue

            sv = self._compute_timeframe(df, tf, data.instrument)
            if sv is not None:
                signals.append(sv)

        return signals

    def _compute_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        instrument: str,
    ) -> SignalVector | None:
        """Compute EMA confluence for a single timeframe."""
        close = df["close"]

        ema_f = compute_ema(close, self.ema_fast)
        ema_m = compute_ema(close, self.ema_mid)
        ema_s = compute_ema(close, self.ema_slow)

        # ATR for spread normalization
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # --- Alignment score ---
        # Full bullish: fast > mid > slow -> +1
        # Full bearish: fast < mid < slow -> -1
        # Mixed: 0
        bullish = (ema_f > ema_m) & (ema_m > ema_s)
        bearish = (ema_f < ema_m) & (ema_m < ema_s)
        direction = pd.Series(0, index=df.index, dtype=int)
        direction = direction.where(~bullish, 1)
        direction = direction.where(~bearish, -1)

        # --- Spread (fast-slow) normalized by ATR ---
        spread = (ema_f - ema_s) / atr_safe
        # Normalize to [0, 1]: use rolling percentile rank
        spread_abs = spread.abs()
        spread_rank = spread_abs.rolling(window=min(100, len(df))).rank(pct=True)
        spread_rank = spread_rank.fillna(0.0)

        # --- Crossover velocity (rate of change of fast-mid spread) ---
        fast_mid_spread = ema_f - ema_m
        velocity = fast_mid_spread.diff(3) / atr_safe
        velocity_abs = velocity.abs()
        velocity_rank = velocity_abs.rolling(window=min(100, len(df))).rank(pct=True)
        velocity_rank = velocity_rank.fillna(0.0)

        # --- Strength: weighted combo of spread rank + velocity rank ---
        strength = (0.6 * spread_rank + 0.4 * velocity_rank).clip(0.0, 1.0)
        # Zero out strength when direction is neutral
        strength = strength.where(direction != 0, 0.0)

        # Formation index: bar positions where alignment first forms
        alignment_change = direction.diff().fillna(0) != 0
        formation_idx = pd.Series(np.nan, index=df.index)
        formation_idx[alignment_change] = np.arange(len(df))[alignment_change.values]
        formation_idx = formation_idx.ffill().fillna(0).astype(int)

        return SignalVector(
            signal_id=f"SIG_EMA_CONFLUENCE_{timeframe}_v1",
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
            },
            metadata={
                "instrument": instrument,
                "intuition": "Multi-EMA alignment with expanding spread",
                "bars_processed": len(df),
            },
        )
