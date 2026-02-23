"""
Category 6: Inverse Fair Value Gap (IFVG) Detector.

Detects inverse FVG continuation patterns — filled FVGs that then act
as support/resistance zones when price returns and rejects.

Signal composition:
- direction: +1 on bullish rejection from filled FVG zone, -1 on bearish
- strength: combines rejection quality, zone touch count, and volume
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


class IFVGDetector(SignalDetector):
    """IFVG: filled FVG zones acting as S/R with rejection signals."""

    detector_id = "ifvg"
    category = "ifvg"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [
        tf.value for tf in [
            Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4,
        ]
    ]

    def __init__(
        self,
        min_gap_atr: float = 0.5,
        max_touches: int = 3,
        rejection_body_ratio: float = 0.6,
    ) -> None:
        self.min_gap_atr = min_gap_atr
        self.max_touches = max_touches
        self.rejection_body_ratio = rejection_body_ratio

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
        fvgs = detect_fvgs(df, self.min_gap_atr)
        fvgs = track_fvg_fills(fvgs, df, max_age=200)

        # Only keep filled FVGs — these become potential IFVG zones
        filled_fvgs = [f for f in fvgs if f["filled"]]

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index, dtype=float)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        highs = df["high"].values
        lows = df["low"].values
        opens = df["open"].values
        closes = df["close"].values
        volumes = df["volume"].values
        vol_avg = pd.Series(volumes).rolling(20, min_periods=1).mean().values

        # Suppress unused variable warning — atr_safe reserved for future use
        _ = atr_safe

        # Track touches per IFVG zone
        touch_counts: dict[int, int] = {}

        for i in range(len(df)):
            best_score = 0.0
            best_dir = 0
            best_form = 0

            for fvg in filled_fvgs:
                fid = fvg["idx"]
                # Zone only active after fill
                if i <= fvg["fill_idx"]:
                    continue
                # Expire after too many touches
                if touch_counts.get(fid, 0) >= self.max_touches:
                    continue

                zone_lo = fvg["zone_low"]
                zone_hi = fvg["zone_high"]

                # Check if price wicks into zone
                wick_into = lows[i] <= zone_hi and highs[i] >= zone_lo
                if not wick_into:
                    continue

                # Determine rejection direction
                body_range = abs(closes[i] - opens[i])
                candle_range = highs[i] - lows[i]
                if candle_range <= 0:
                    continue

                body_ratio = body_range / candle_range

                if fvg["type"] == "bullish":
                    # Bullish IFVG: price dips into zone, closes above
                    if (closes[i] > zone_hi
                            and body_ratio >= self.rejection_body_ratio):
                        sig_dir = 1
                    else:
                        continue
                else:
                    # Bearish IFVG: price pokes into zone, closes below
                    if (closes[i] < zone_lo
                            and body_ratio >= self.rejection_body_ratio):
                        sig_dir = -1
                    else:
                        continue

                touch_counts[fid] = touch_counts.get(fid, 0) + 1
                touches = touch_counts[fid]

                # Strength components
                rejection_score = min(body_ratio, 1.0)
                touch_score = max(
                    1.0 - (touches - 1) / self.max_touches, 0.0,
                )
                vol_ratio = (
                    volumes[i] / vol_avg[i]
                    if vol_avg[i] > 0 else 1.0
                )
                vol_score = min(vol_ratio / 3.0, 1.0)

                score = (
                    0.40 * rejection_score
                    + 0.30 * touch_score
                    + 0.30 * vol_score
                )

                if score > best_score:
                    best_score = score
                    best_dir = sig_dir
                    best_form = i

            direction.iloc[i] = best_dir
            strength.iloc[i] = round(min(best_score, 1.0), 6)
            if best_form > 0:
                formation_idx.iloc[i] = best_form

        # Forward-fill formation_idx
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        # Zero strength when neutral
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_IFVG_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "min_gap_atr": self.min_gap_atr,
                "max_touches": self.max_touches,
                "rejection_body_ratio": self.rejection_body_ratio,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Filled FVG zones acting as S/R",
                "bars_processed": len(df),
                "filled_fvgs": len(filled_fvgs),
            },
        )
