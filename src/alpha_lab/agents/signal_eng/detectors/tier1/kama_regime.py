"""
Category 2: KAMA Regime Detector.

Uses Kaufman's Adaptive Moving Average to detect regime.
Signals based on KAMA slope, efficiency ratio, and price-KAMA divergence.

Signal composition:
- direction: +1 (trending up / bullish), -1 (trending down / bearish), 0 (ranging)
- strength: combines efficiency ratio, slope magnitude, and price-KAMA distance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_kama,
    compute_kama_efficiency_ratio,
)
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

# Defaults
_KAMA_PERIOD = 10
_KAMA_FAST = 2
_KAMA_SLOW = 30
_SLOPE_LOOKBACK = 5
_ER_TREND_THRESHOLD = 0.30


class KamaRegimeDetector(SignalDetector):
    """KAMA Regime: slope, price-KAMA divergence, adaptive smoothing."""

    detector_id = "kama_regime"
    category = "kama_regime"
    tier = SignalTier.CORE
    timeframes = [
        tf.value
        for tf in [
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.H1,
            Timeframe.H4,
            Timeframe.D1,
        ]
    ]

    def __init__(
        self,
        kama_period: int = _KAMA_PERIOD,
        kama_fast: int = _KAMA_FAST,
        kama_slow: int = _KAMA_SLOW,
        slope_lookback: int = _SLOPE_LOOKBACK,
        er_trend_threshold: float = _ER_TREND_THRESHOLD,
    ) -> None:
        self.kama_period = kama_period
        self.kama_fast = kama_fast
        self.kama_slow = kama_slow
        self.slope_lookback = slope_lookback
        self.er_trend_threshold = er_trend_threshold

    def validate_inputs(self, data: DataBundle) -> bool:
        """Need enough bars for KAMA warmup (period + slow)."""
        min_bars = self.kama_period + self.kama_slow
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > min_bars:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        """Compute KAMA regime signals across timeframes."""
        signals: list[SignalVector] = []

        for tf in self.timeframes:
            df = data.bars.get(tf)
            min_bars = self.kama_period + self.kama_slow
            if not isinstance(df, pd.DataFrame) or len(df) <= min_bars:
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
        """Compute KAMA regime for a single timeframe."""
        close = df["close"]

        kama = compute_kama(close, self.kama_period, self.kama_fast, self.kama_slow)
        er = compute_kama_efficiency_ratio(close, self.kama_period)
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # --- KAMA slope (normalized by ATR) ---
        kama_slope = kama.diff(self.slope_lookback) / atr_safe
        kama_slope = kama_slope.fillna(0.0)

        # --- Price-KAMA divergence (normalized by ATR) ---
        price_kama_dist = (close - kama) / atr_safe
        price_kama_dist = price_kama_dist.fillna(0.0)

        # --- Direction ---
        # Trending: ER above threshold
        # Bullish trend: positive slope + price above KAMA
        # Bearish trend: negative slope + price below KAMA
        is_trending = er > self.er_trend_threshold
        is_bullish = (kama_slope > 0) & (close > kama)
        is_bearish = (kama_slope < 0) & (close < kama)

        direction = pd.Series(0, index=df.index, dtype=int)
        direction = direction.where(~(is_trending & is_bullish), 1)
        direction = direction.where(~(is_trending & is_bearish), -1)

        # --- Strength ---
        # Combine: efficiency ratio (trend conviction) + slope magnitude + distance
        slope_rank = kama_slope.abs().rolling(window=min(100, len(df))).rank(pct=True)
        slope_rank = slope_rank.fillna(0.0)

        dist_rank = price_kama_dist.abs().rolling(window=min(100, len(df))).rank(pct=True)
        dist_rank = dist_rank.fillna(0.0)

        strength = (0.4 * er + 0.35 * slope_rank + 0.25 * dist_rank).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        # Formation index
        direction_change = direction.diff().fillna(0) != 0
        formation_idx = pd.Series(np.nan, index=df.index)
        formation_idx[direction_change] = np.arange(len(df))[direction_change.values]
        formation_idx = formation_idx.ffill().fillna(0).astype(int)

        return SignalVector(
            signal_id=f"SIG_KAMA_REGIME_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "kama_period": self.kama_period,
                "kama_fast": self.kama_fast,
                "kama_slow": self.kama_slow,
                "slope_lookback": self.slope_lookback,
                "er_trend_threshold": self.er_trend_threshold,
            },
            metadata={
                "instrument": instrument,
                "intuition": "KAMA adaptive regime: trending when ER high + directional slope",
                "bars_processed": len(df),
            },
        )
