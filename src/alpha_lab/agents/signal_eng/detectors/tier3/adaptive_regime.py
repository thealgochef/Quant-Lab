"""
Category 20: Adaptive Regime Filter.

Combines KAMA regime detection with EMA spread classification
to dynamically adjust signal weights based on market conditions.

Signal composition:
- direction: +1 (trending up), -1 (trending down), 0 (ranging)
- strength: combines efficiency ratio, EMA spread rank, and direction consistency
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_ema,
    compute_kama_efficiency_ratio,
)
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 50


class AdaptiveRegimeDetector(SignalDetector):
    """Adaptive Regime Filter: KAMA + EMA spread classification."""

    detector_id = "adaptive_regime"
    category = "adaptive_regime"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]]

    def __init__(
        self,
        er_period: int = 10,
        ema_fast: int = 13,
        ema_slow: int = 48,
        er_threshold: float = 0.30,
        consistency_window: int = 20,
    ) -> None:
        self.er_period = er_period
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.er_threshold = er_threshold
        self.consistency_window = consistency_window

    def validate_inputs(self, data: DataBundle) -> bool:
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > _MIN_BARS:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
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

        # Efficiency ratio: 1.0 = strong trend, 0.0 = choppy
        er = compute_kama_efficiency_ratio(close, self.er_period)

        # EMAs for trend direction
        ema_f = compute_ema(close, self.ema_fast)
        ema_s = compute_ema(close, self.ema_slow)

        # ATR for normalization
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # KAMA-like slope via EMA fast
        slope = ema_f.diff(5) / atr_safe

        # Direction: trending when ER > threshold and EMAs directional
        is_trending = er > self.er_threshold
        is_bullish = (ema_f > ema_s) & (slope > 0)
        is_bearish = (ema_f < ema_s) & (slope < 0)

        direction = pd.Series(0, index=df.index, dtype=int)
        direction = direction.where(~(is_trending & is_bullish), 1)
        direction = direction.where(~(is_trending & is_bearish), -1)

        # Strength component 1: ER value (already in [0,1])
        er_component = er.clip(0.0, 1.0).fillna(0.0)

        # Strength component 2: EMA spread rank
        spread = (ema_f - ema_s).abs() / atr_safe
        spread_rank = spread.rolling(window=min(100, len(df))).rank(pct=True).fillna(0.0)

        # Strength component 3: direction consistency
        win = min(self.consistency_window, len(df))
        dir_sign = direction.replace(0, np.nan)

        def _dir_consistency(x: pd.Series) -> float:
            vals = x.dropna()
            if len(vals) == 0:
                return 0.0
            return float((vals == vals.iloc[-1]).mean())

        consistency = dir_sign.rolling(window=win, min_periods=1).apply(
            _dir_consistency, raw=False,
        ).fillna(0.0)

        strength = (0.40 * er_component + 0.30 * spread_rank + 0.30 * consistency).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        # Formation index
        dir_change = direction.diff().fillna(0) != 0
        formation_idx = pd.Series(np.nan, index=df.index)
        formation_idx[dir_change] = np.arange(len(df))[dir_change.values]
        formation_idx = formation_idx.ffill().fillna(0).astype(int)

        return SignalVector(
            signal_id=f"SIG_ADAPTIVE_REGIME_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "er_period": self.er_period,
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
                "er_threshold": self.er_threshold,
                "consistency_window": self.consistency_window,
            },
            metadata={
                "instrument": instrument,
                "intuition": "KAMA efficiency ratio + EMA spread regime classification",
                "bars_processed": len(df),
            },
        )
