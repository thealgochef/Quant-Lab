"""
Category 13: Displacement Signals.

Detects large body candles after consolidation, especially when
accompanied by a Fair Value Gap formation.

Signal composition:
- direction: +1 (bullish displacement), -1 (bearish displacement), 0 (none)
- strength: combines body size rank, volume, FVG presence, consolidation tightness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.detectors.tier2._fvg_helpers import detect_fvgs
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class DisplacementDetector(SignalDetector):
    """Displacement: large body after consolidation + FVG."""

    detector_id = "displacement"
    category = "displacement"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]]

    def __init__(
        self,
        consolidation_window: int = 10,
        consolidation_threshold: float = 1.5,
        displacement_multiplier: float = 2.0,
        fvg_lookback: int = 3,
        min_gap_atr: float = 0.3,
    ) -> None:
        self.consolidation_window = consolidation_window
        self.consolidation_threshold = consolidation_threshold
        self.displacement_multiplier = displacement_multiplier
        self.fvg_lookback = fvg_lookback
        self.min_gap_atr = min_gap_atr

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
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # Body size
        body = (close - open_).abs()
        avg_body = body.rolling(window=self.consolidation_window, min_periods=1).mean()
        avg_body_safe = avg_body.replace(0, np.nan).fillna(body.mean() or 1.0)

        # Consolidation: rolling range / ATR
        roll_hi = high.rolling(self.consolidation_window).max()
        roll_lo = low.rolling(self.consolidation_window).min()
        rolling_range = roll_hi - roll_lo
        consolidation = rolling_range / atr_safe
        is_consolidated = consolidation < self.consolidation_threshold

        # Displacement: large body relative to average
        is_displacement = body > self.displacement_multiplier * avg_body_safe

        # FVG detection
        fvgs = detect_fvgs(df, min_gap_atr=self.min_gap_atr)
        fvg_bars: set[int] = set()
        for fvg in fvgs:
            fvg_bars.add(fvg["idx"])

        vol_avg = volume.rolling(window=20, min_periods=1).mean().replace(0, 1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        body_vals = body.values
        atr_vals = atr_safe.values
        vol_vals = volume.values
        vol_avg_vals = vol_avg.values
        close_vals = close.values
        open_vals = open_.values
        cons_vals = consolidation.values

        for i in range(self.consolidation_window, len(df)):
            if not is_displacement.iloc[i]:
                continue
            if not is_consolidated.iloc[i - 1]:
                continue

            # Direction from candle color
            d = 1 if close_vals[i] > open_vals[i] else -1

            # Strength components
            body_rank = min(body_vals[i] / atr_vals[i] / 3.0, 1.0) if atr_vals[i] > 0 else 0.0
            vol_score = min(vol_vals[i] / vol_avg_vals[i] / 3.0, 1.0)

            # FVG bonus: check if FVG formed near this bar
            has_fvg = any(abs(i - fb) <= self.fvg_lookback for fb in fvg_bars)
            fvg_bonus = 1.0 if has_fvg else 0.0

            # Consolidation tightness (lower = tighter)
            cons_score = max(1.0 - cons_vals[i - 1] / (self.consolidation_threshold * 2), 0.0)

            s = 0.35 * body_rank + 0.30 * vol_score + 0.20 * fvg_bonus + 0.15 * cons_score

            direction.iloc[i] = d
            strength.iloc[i] = round(min(s, 1.0), 6)
            formation_idx.iloc[i] = i

        # Forward-fill from displacement events with exponential decay
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        dir_filled = direction.replace(0, np.nan).ffill().fillna(0).astype(int)
        str_filled = strength.replace(0.0, np.nan).ffill().fillna(0.0)

        has_signal = formation_idx > 0
        direction = dir_filled.where(has_signal, 0)
        strength = str_filled.where(has_signal, 0.0).clip(0.0, 1.0)

        # Decay strength: halve every 20 bars from formation
        bars_since = pd.Series(np.arange(len(df)), index=df.index) - formation_idx
        decay = np.power(0.5, bars_since.clip(lower=0) / 20.0)
        strength = (strength * decay).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_DISPLACEMENT_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "consolidation_window": self.consolidation_window,
                "consolidation_threshold": self.consolidation_threshold,
                "displacement_multiplier": self.displacement_multiplier,
                "fvg_lookback": self.fvg_lookback,
                "min_gap_atr": self.min_gap_atr,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Large body candle breaking out of consolidation, confirmed by FVG",
                "bars_processed": len(df),
            },
        )
