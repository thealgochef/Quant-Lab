"""
Category 5: Fair Value Gaps Detector.

Identifies FVG formation (3-candle pattern with gap), tracks fill probability,
and uses FVGs as potential entry zones.

Signal composition:
- direction: +1 near unfilled bullish FVG, -1 near bearish FVG, 0 otherwise
- strength: combines gap size, proximity to gap, and time decay
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.detectors.tier2._fvg_helpers import (
    detect_fvgs,
    track_fvg_fills,
)
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class FairValueGapsDetector(SignalDetector):
    """Fair Value Gaps: formation, fill probability, FVG as entry zone."""

    detector_id = "fair_value_gaps"
    category = "fair_value_gaps"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [
        tf.value for tf in [
            Timeframe.M1, Timeframe.M5, Timeframe.M15,
            Timeframe.M30, Timeframe.H1, Timeframe.H4,
        ]
    ]

    def __init__(
        self,
        min_gap_atr: float = 0.5,
        max_fvg_age: int = 200,
        approach_distance_atr: float = 1.0,
        decay_half_life: int = 50,
    ) -> None:
        self.min_gap_atr = min_gap_atr
        self.max_fvg_age = max_fvg_age
        self.approach_distance_atr = approach_distance_atr
        self.decay_half_life = decay_half_life

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
        self, df: pd.DataFrame, timeframe: str, instrument: str,
    ) -> SignalVector | None:
        # Detect and track FVGs
        fvgs = detect_fvgs(df, self.min_gap_atr)
        fvgs = track_fvg_fills(fvgs, df, self.max_fvg_age)

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index, dtype=float)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        closes = df["close"].values
        atr_vals = atr_safe.values

        for i in range(len(df)):
            best_score = 0.0
            best_dir = 0
            best_form = 0

            for fvg in fvgs:
                # Only consider unfilled FVGs that formed before bar i
                if fvg["filled"] and fvg["fill_idx"] <= i:
                    continue
                if fvg["idx"] >= i:
                    continue

                age = i - fvg["idx"]
                if age > self.max_fvg_age:
                    continue

                # Distance from close to nearest edge of FVG zone
                zone_mid = (fvg["zone_low"] + fvg["zone_high"]) / 2
                dist = abs(closes[i] - zone_mid)
                dist_atr = dist / atr_vals[i] if atr_vals[i] > 0 else 999

                if dist_atr > self.approach_distance_atr:
                    continue

                # Strength components
                gap_score = min(fvg["size_atr"] / 2.0, 1.0)
                prox_score = max(
                    1.0 - dist_atr / self.approach_distance_atr, 0.0,
                )
                decay = 0.5 ** (age / self.decay_half_life)

                score = (
                    0.35 * gap_score
                    + 0.35 * prox_score
                    + 0.30 * decay
                )

                if score > best_score:
                    best_score = score
                    best_dir = 1 if fvg["type"] == "bullish" else -1
                    best_form = fvg["idx"]

            direction.iloc[i] = best_dir
            strength.iloc[i] = round(min(best_score, 1.0), 6)
            formation_idx.iloc[i] = best_form

        # Zero strength when neutral
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_FAIR_VALUE_GAPS_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "min_gap_atr": self.min_gap_atr,
                "max_fvg_age": self.max_fvg_age,
                "approach_distance_atr": self.approach_distance_atr,
                "decay_half_life": self.decay_half_life,
            },
            metadata={
                "instrument": instrument,
                "intuition": "3-candle imbalance zones as S/R",
                "bars_processed": len(df),
                "fvgs_detected": len(fvgs),
            },
        )
