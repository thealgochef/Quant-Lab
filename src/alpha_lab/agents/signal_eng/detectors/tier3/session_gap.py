"""
Category 19: Session Gap Detector.

Analyzes overnight gaps: gap probability, fill rate, and directional bias.
Uses gap size relative to ATR for signal strength.

Signal composition:
- direction: +1 (gap up / bullish continuation), -1 (gap down / bearish), 0 (no gap)
- strength: combines gap size/ATR, fill status, and time decay
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class SessionGapDetector(SignalDetector):
    """Session Gap: overnight gap probability, fill rate, bias."""

    detector_id = "session_gap"
    category = "session_gap"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.D1]]

    def __init__(
        self,
        min_gap_atr: float = 0.5,
        fill_threshold_pct: float = 0.8,
        max_gap_age_bars: int = 78,
    ) -> None:
        self.min_gap_atr = min_gap_atr
        self.fill_threshold_pct = fill_threshold_pct
        self.max_gap_age_bars = max_gap_age_bars

    def validate_inputs(self, data: DataBundle) -> bool:
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if (isinstance(df, pd.DataFrame)
                    and len(df) > _MIN_BARS
                    and hasattr(df.index, 'date')):
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            if not hasattr(df.index, 'date'):
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
        high = df["high"]
        low = df["low"]

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        open_vals = df["open"].values
        close_vals = close.values
        high_vals = high.values
        low_vals = low.values
        atr_vals = atr_safe.values

        # Detect session boundaries via date changes
        dates = df.index.date
        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        # Track active gap
        gap_active = False
        gap_dir = 0
        gap_size = 0.0
        gap_start_bar = 0
        prev_close = close_vals[0]
        gap_fill_level = 0.0

        for i in range(1, len(df)):
            a = atr_vals[i]
            if np.isnan(a) or a <= 0:
                prev_close = close_vals[i]
                continue

            # Check for new session (date change)
            is_new_session = dates[i] != dates[i - 1]

            if is_new_session:
                # Gap = open of new session - close of previous session
                gap = open_vals[i] - prev_close
                gap_atr = abs(gap) / a

                if gap_atr >= self.min_gap_atr:
                    gap_active = True
                    gap_dir = 1 if gap > 0 else -1
                    gap_size = abs(gap)
                    gap_start_bar = i
                    gap_fill_level = prev_close
                else:
                    gap_active = False
                    gap_dir = 0

            if gap_active:
                bars_since = i - gap_start_bar

                # Check gap fill
                if gap_dir == 1:
                    # Gap up: filled when price returns to prev close
                    fill_pct = max(1.0 - (low_vals[i] - gap_fill_level) / max(gap_size, 0.01), 0.0)
                else:
                    # Gap down: filled when price returns up to prev close
                    fill_pct = max(1.0 - (gap_fill_level - high_vals[i]) / max(gap_size, 0.01), 0.0)

                fill_pct = min(fill_pct, 1.0)
                is_filled = fill_pct >= self.fill_threshold_pct

                if is_filled or bars_since >= self.max_gap_age_bars:
                    gap_active = False
                    gap_dir = 0
                    direction.iloc[i] = 0
                    strength.iloc[i] = 0.0
                else:
                    # Signal: gap continuation bias
                    gap_size_score = min(gap_size / a / 2.0, 1.0)
                    fill_status = max(1.0 - fill_pct, 0.0)
                    time_decay = max(1.0 - bars_since / max(self.max_gap_age_bars, 1), 0.0)

                    s = 0.40 * gap_size_score + 0.30 * fill_status + 0.30 * time_decay

                    direction.iloc[i] = gap_dir
                    strength.iloc[i] = round(min(s, 1.0), 6)
                    formation_idx.iloc[i] = gap_start_bar

            prev_close = close_vals[i]

        # Forward-fill formation index for bars with active gap
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_SESSION_GAP_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "min_gap_atr": self.min_gap_atr,
                "fill_threshold_pct": self.fill_threshold_pct,
                "max_gap_age_bars": self.max_gap_age_bars,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Overnight gap continuation bias, strength decays as gap fills",
                "bars_processed": len(df),
            },
        )
