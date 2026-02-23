"""
Category 7: Market Structure Detector.

Identifies Break of Structure (BOS), Change of Character (CHOCH),
and Higher-High/Lower-Low sequences for trend analysis.

Signal composition:
- direction: +1 on bullish BOS, -1 on bearish BOS, 0 on CHOCH/neutral
- strength: combines break distance, sequence consistency, and volume
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_swing_highs,
    compute_swing_lows,
)
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class MarketStructureDetector(SignalDetector):
    """Market Structure: BOS, CHOCH, HH/LL sequences."""

    detector_id = "market_structure"
    category = "market_structure"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [
        tf.value for tf in [
            Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
            Timeframe.H1, Timeframe.H4, Timeframe.D1,
        ]
    ]

    def __init__(
        self,
        pivot_left: int = 3,
        pivot_right: int = 3,
        min_break_atr: float = 0.3,
        structure_memory: int = 5,
    ) -> None:
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.min_break_atr = min_break_atr
        self.structure_memory = structure_memory

    def validate_inputs(self, data: DataBundle) -> bool:
        min_needed = self.pivot_left + self.pivot_right + _MIN_BARS
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > min_needed:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        min_needed = self.pivot_left + self.pivot_right + _MIN_BARS
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= min_needed:
                continue
            sv = self._compute_timeframe(df, tf, data.instrument)
            if sv is not None:
                signals.append(sv)
        return signals

    def _compute_timeframe(
        self, df: pd.DataFrame, timeframe: str, instrument: str,
    ) -> SignalVector | None:
        swing_hi = compute_swing_highs(
            df["high"], self.pivot_left, self.pivot_right,
        )
        swing_lo = compute_swing_lows(
            df["low"], self.pivot_left, self.pivot_right,
        )

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index, dtype=float)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        volumes = df["volume"].values
        vol_avg = pd.Series(volumes).rolling(20, min_periods=1).mean().values
        closes = df["close"].values

        # Track recent swing points
        recent_highs: list[float] = []
        recent_lows: list[float] = []
        trend = 0  # +1 uptrend, -1 downtrend, 0 undefined
        consecutive = 0  # consecutive matching structure points

        for i in range(len(df)):
            sh = swing_hi.iloc[i]
            sl = swing_lo.iloc[i]

            sig_dir = 0
            sig_score = 0.0

            # Process new swing high
            if not np.isnan(sh):
                if recent_highs:
                    prev_sh = recent_highs[-1]
                    break_dist = (sh - prev_sh) / atr_safe.iloc[i]

                    if sh > prev_sh:  # Higher High
                        if trend == 1:
                            # BOS: continuation in uptrend
                            if break_dist >= self.min_break_atr:
                                sig_dir = 1
                                consecutive += 1
                        elif trend == -1:
                            # CHOCH: reversal from downtrend
                            consecutive = 1
                        trend = 1

                    elif sh < prev_sh:  # Lower High
                        if trend == 1:
                            # CHOCH: potential reversal
                            consecutive = 0
                        elif trend == -1:
                            consecutive += 1
                        if trend != -1:
                            trend = -1

                    # Break distance for strength
                    if sig_dir != 0:
                        dist_score = min(
                            abs(break_dist) / 2.0, 1.0,
                        )
                        seq_score = min(
                            consecutive / self.structure_memory, 1.0,
                        )
                        vol_ratio = (
                            volumes[i] / vol_avg[i]
                            if vol_avg[i] > 0 else 1.0
                        )
                        vol_score = min(vol_ratio / 3.0, 1.0)
                        sig_score = (
                            0.40 * dist_score
                            + 0.30 * seq_score
                            + 0.30 * vol_score
                        )
                else:
                    trend = 0

                recent_highs.append(sh)
                if len(recent_highs) > self.structure_memory:
                    recent_highs.pop(0)

            # Process new swing low
            if not np.isnan(sl):
                if recent_lows:
                    prev_sl = recent_lows[-1]
                    break_dist = (prev_sl - sl) / atr_safe.iloc[i]

                    if sl < prev_sl:  # Lower Low
                        if trend == -1:
                            # BOS: continuation in downtrend
                            if (break_dist >= self.min_break_atr
                                    and sig_dir == 0):
                                sig_dir = -1
                                consecutive += 1
                        elif trend == 1:
                            # CHOCH: reversal
                            consecutive = 1
                        trend = -1

                    elif sl > prev_sl:  # Higher Low
                        if trend == -1:
                            consecutive = 0
                        elif trend == 1:
                            consecutive += 1
                        if trend != 1:
                            trend = 1

                    if sig_dir == -1 and sig_score == 0.0:
                        dist_score = min(
                            abs(break_dist) / 2.0, 1.0,
                        )
                        seq_score = min(
                            consecutive / self.structure_memory, 1.0,
                        )
                        vol_ratio = (
                            volumes[i] / vol_avg[i]
                            if vol_avg[i] > 0 else 1.0
                        )
                        vol_score = min(vol_ratio / 3.0, 1.0)
                        sig_score = (
                            0.40 * dist_score
                            + 0.30 * seq_score
                            + 0.30 * vol_score
                        )

                recent_lows.append(sl)
                if len(recent_lows) > self.structure_memory:
                    recent_lows.pop(0)

            # Also propagate trend direction on non-swing bars
            if sig_dir == 0 and trend != 0:
                # Carry trend direction with reduced strength
                if trend == 1 and closes[i] > closes[i - 1] if i > 0 else False:
                    sig_dir = 1
                    sig_score = 0.15
                elif trend == -1 and closes[i] < closes[i - 1] if i > 0 else False:
                    sig_dir = -1
                    sig_score = 0.15

            direction.iloc[i] = sig_dir
            strength.iloc[i] = round(min(sig_score, 1.0), 6)

            # Formation index: set at pivot confirmation point
            if not np.isnan(sh) or not np.isnan(sl):
                formation_idx.iloc[i] = i

        # Forward-fill formation_idx
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        # Zero strength when neutral
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_MARKET_STRUCTURE_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "pivot_left": self.pivot_left,
                "pivot_right": self.pivot_right,
                "min_break_atr": self.min_break_atr,
                "structure_memory": self.structure_memory,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Swing structure BOS/CHOCH detection",
                "bars_processed": len(df),
            },
        )
