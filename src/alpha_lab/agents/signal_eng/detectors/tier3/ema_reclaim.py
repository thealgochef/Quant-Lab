"""
Category 18: EMA Reclaim Detector.

Detects price reclaiming the 13 EMA after sweeping below the 48 EMA.
A bullish reclaim pattern indicating institutional accumulation.

Signal composition:
- direction: +1 (bullish reclaim), -1 (bearish reclaim), 0 (no pattern)
- strength: combines sweep depth, reclaim speed, and volume confirmation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr, compute_ema
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 60


class EmaReclaimDetector(SignalDetector):
    """EMA Reclaim: reclaiming 13 EMA after sweeping below 48 EMA."""

    detector_id = "ema_reclaim"
    category = "ema_reclaim"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1]]

    def __init__(
        self,
        ema_fast: int = 13,
        ema_slow: int = 48,
        max_sweep_bars: int = 20,
        min_sweep_depth_atr: float = 0.3,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.max_sweep_bars = max_sweep_bars
        self.min_sweep_depth_atr = min_sweep_depth_atr

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
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

        ema_f = compute_ema(close, self.ema_fast)
        ema_s = compute_ema(close, self.ema_slow)
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        vol_avg = volume.rolling(window=20, min_periods=1).mean().replace(0, 1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        close_vals = close.values
        ema_f_vals = ema_f.values
        ema_s_vals = ema_s.values
        atr_vals = atr_safe.values
        vol_vals = volume.values
        vol_avg_vals = vol_avg.values

        # State machine: track reclaim patterns
        # States: 0=idle, 1=was_above_fast, 2=swept_below_slow
        bull_state = 0
        bull_sweep_start = 0
        bull_max_depth = 0.0

        bear_state = 0
        bear_sweep_start = 0
        bear_max_depth = 0.0

        for i in range(1, len(df)):
            c = close_vals[i]
            ef = ema_f_vals[i]
            es = ema_s_vals[i]
            a = atr_vals[i]

            if np.isnan(ef) or np.isnan(es) or np.isnan(a) or a <= 0:
                continue

            # --- Bullish reclaim: above fast -> below slow -> reclaim fast ---
            if bull_state == 0:
                if c > ef:
                    bull_state = 1
            elif bull_state == 1:
                if c < es:
                    bull_state = 2
                    bull_sweep_start = i
                    bull_max_depth = abs(c - es)
                elif c < ef:
                    bull_state = 0
            elif bull_state == 2:
                depth = abs(min(c, close_vals[i - 1]) - es)
                bull_max_depth = max(bull_max_depth, depth)
                bars_in = i - bull_sweep_start

                if bars_in > self.max_sweep_bars:
                    bull_state = 0
                elif c > ef:
                    # Reclaim! Signal fires
                    if bull_max_depth / a >= self.min_sweep_depth_atr:
                        depth_score = min(bull_max_depth / a / 2.0, 1.0)
                        speed_score = max(1.0 - bars_in / self.max_sweep_bars, 0.0)
                        vol_score = min(vol_vals[i] / vol_avg_vals[i] / 3.0, 1.0)
                        s = (0.40 * depth_score + 0.30 * speed_score + 0.30 * vol_score)
                        direction.iloc[i] = 1
                        strength.iloc[i] = round(min(s, 1.0), 6)
                        formation_idx.iloc[i] = i
                    bull_state = 1
                    bull_max_depth = 0.0

            # --- Bearish reclaim: below fast -> above slow -> reclaim below fast ---
            if bear_state == 0:
                if c < ef:
                    bear_state = 1
            elif bear_state == 1:
                if c > es:
                    bear_state = 2
                    bear_sweep_start = i
                    bear_max_depth = abs(c - es)
                elif c > ef:
                    bear_state = 0
            elif bear_state == 2:
                depth = abs(max(c, close_vals[i - 1]) - es)
                bear_max_depth = max(bear_max_depth, depth)
                bars_in = i - bear_sweep_start

                if bars_in > self.max_sweep_bars:
                    bear_state = 0
                elif c < ef:
                    if bear_max_depth / a >= self.min_sweep_depth_atr:
                        depth_score = min(bear_max_depth / a / 2.0, 1.0)
                        speed_score = max(1.0 - bars_in / self.max_sweep_bars, 0.0)
                        vol_score = min(vol_vals[i] / vol_avg_vals[i] / 3.0, 1.0)
                        s = (0.40 * depth_score + 0.30 * speed_score + 0.30 * vol_score)
                        direction.iloc[i] = -1
                        strength.iloc[i] = round(min(s, 1.0), 6)
                        formation_idx.iloc[i] = i
                    bear_state = 1
                    bear_max_depth = 0.0

        # Forward-fill formation index
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        # Forward-fill direction and strength from reclaim events
        dir_filled = direction.replace(0, np.nan).ffill().fillna(0).astype(int)
        str_filled = strength.replace(0.0, np.nan).ffill().fillna(0.0)
        # Only keep filled values where there has been a signal
        has_signal = formation_idx > 0
        direction = dir_filled.where(has_signal, 0)
        strength = str_filled.where(has_signal, 0.0).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_EMA_RECLAIM_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "ema_fast": self.ema_fast,
                "ema_slow": self.ema_slow,
                "max_sweep_bars": self.max_sweep_bars,
                "min_sweep_depth_atr": self.min_sweep_depth_atr,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Price reclaiming fast EMA after sweeping slow EMA",
                "bars_processed": len(df),
            },
        )
