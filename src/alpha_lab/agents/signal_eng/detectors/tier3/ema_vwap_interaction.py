"""
Category 12: EMA-VWAP Interaction Detector.

Analyzes EMA alignment relative to VWAP, identifying triple zones
where price, EMAs, and VWAP converge.

Signal composition:
- direction: +1 (triple bullish: close > EMA13 > EMA48 and close > VWAP),
             -1 (triple bearish), 0 (mixed)
- strength: combines VWAP z-score, EMA spread rank, and alignment consistency
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_ema,
    compute_session_vwap_bands,
)
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class EmaVwapInteractionDetector(SignalDetector):
    """EMA-VWAP Interaction: alignment relative to VWAP, triple zones."""

    detector_id = "ema_vwap_interaction"
    category = "ema_vwap_interaction"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1]]

    def __init__(
        self,
        ema_fast: int = 13,
        ema_slow: int = 48,
        band_std: float = 2.0,
        consistency_window: int = 10,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.band_std = band_std
        self.consistency_window = consistency_window

    def validate_inputs(self, data: DataBundle) -> bool:
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if (
                isinstance(df, pd.DataFrame)
                and len(df) > _MIN_BARS
                and "volume" in df.columns
            ):
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            if "volume" not in df.columns:
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
        close = df["close"]

        ema_f = compute_ema(close, self.ema_fast)
        ema_s = compute_ema(close, self.ema_slow)
        vwap, upper, lower = compute_session_vwap_bands(df, self.band_std)

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # Triple alignment
        triple_bull = (close > ema_f) & (ema_f > ema_s) & (close > vwap)
        triple_bear = (close < ema_f) & (ema_f < ema_s) & (close < vwap)

        direction = pd.Series(0, index=df.index, dtype=int)
        direction = direction.where(~triple_bull, 1)
        direction = direction.where(~triple_bear, -1)

        # Strength component 1: VWAP z-score magnitude
        band_width = (upper - lower).replace(0, np.nan).fillna(1.0)
        zscore = ((close - vwap) / band_width).abs().clip(0.0, 3.0) / 3.0
        zscore = zscore.fillna(0.0)

        # Strength component 2: EMA spread rank
        spread = (ema_f - ema_s).abs() / atr_safe
        spread_rank = spread.rolling(window=min(100, len(df))).rank(pct=True).fillna(0.0)

        # Strength component 3: alignment consistency
        win = min(self.consistency_window, len(df))
        consistency = (
            direction.rolling(window=win, min_periods=1)
            .apply(lambda x: (x == x.iloc[-1]).mean() if x.iloc[-1] != 0 else 0.0, raw=False)
            .fillna(0.0)
        )

        strength = (0.35 * zscore + 0.35 * spread_rank + 0.30 * consistency).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        # Formation index
        dir_change = direction.diff().fillna(0) != 0
        formation_idx = pd.Series(np.nan, index=df.index)
        formation_idx[dir_change] = np.arange(len(df))[dir_change.values]
        formation_idx = formation_idx.ffill().fillna(0).astype(int)

        return SignalVector(
            signal_id=f"SIG_EMA_VWAP_INTERACTION_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
                "band_std": self.band_std,
                "consistency_window": self.consistency_window,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Triple alignment of price, EMAs, and VWAP",
                "bars_processed": len(df),
            },
        )
