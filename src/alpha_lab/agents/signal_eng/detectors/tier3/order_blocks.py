"""
Category 14: Order Block Signals.

Identifies order blocks (OB) and detects OB+FVG overlap zones
for high-probability entry points.

Signal composition:
- direction: +1 (bullish OB / demand zone), -1 (bearish OB / supply zone), 0 (none)
- strength: combines break distance, OB freshness, FVG overlap, and volume
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.detectors.tier2._fvg_helpers import detect_fvgs
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_swing_highs,
    compute_swing_lows,
)
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 40


class OrderBlocksDetector(SignalDetector):
    """Order Blocks: OB identification, OB+FVG overlap."""

    detector_id = "order_blocks"
    category = "order_blocks"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]]

    def __init__(
        self,
        pivot_left: int = 3,
        pivot_right: int = 3,
        max_active_obs: int = 10,
        ob_max_age: int = 100,
        ob_proximity_atr: float = 0.5,
        min_gap_atr: float = 0.3,
    ) -> None:
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.max_active_obs = max_active_obs
        self.ob_max_age = ob_max_age
        self.ob_proximity_atr = ob_proximity_atr
        self.min_gap_atr = min_gap_atr

    def validate_inputs(self, data: DataBundle) -> bool:
        min_needed = self.pivot_left + self.pivot_right + 30
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > min_needed:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        min_needed = self.pivot_left + self.pivot_right + 30
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= min_needed:
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
        vol_avg = volume.rolling(window=20, min_periods=1).mean().replace(0, 1.0)

        swing_hi = compute_swing_highs(high, self.pivot_left, self.pivot_right)
        swing_lo = compute_swing_lows(low, self.pivot_left, self.pivot_right)

        # FVGs for overlap bonus
        fvgs = detect_fvgs(df, min_gap_atr=self.min_gap_atr)

        close_vals = close.values
        open_vals = open_.values
        high_vals = high.values
        low_vals = low.values
        atr_vals = atr_safe.values
        vol_vals = volume.values
        vol_avg_vals = vol_avg.values

        # Track active OB zones
        active_obs: list[dict] = []
        last_swing_hi = np.nan
        last_swing_lo = np.nan

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        for i in range(self.pivot_left + self.pivot_right, len(df)):
            # Update swing levels
            if not np.isnan(swing_hi.iloc[i]):
                last_swing_hi = swing_hi.iloc[i]
            if not np.isnan(swing_lo.iloc[i]):
                last_swing_lo = swing_lo.iloc[i]

            a = atr_vals[i]
            if np.isnan(a) or a <= 0 or np.isnan(last_swing_hi) or np.isnan(last_swing_lo):
                continue

            # Detect BOS (Break of Structure)
            bullish_bos = close_vals[i] > last_swing_hi
            bearish_bos = close_vals[i] < last_swing_lo

            if bullish_bos:
                break_dist = close_vals[i] - last_swing_hi
                # Find last bearish candle before BOS = bullish OB (demand zone)
                for j in range(i - 1, max(i - 10, 0), -1):
                    if close_vals[j] < open_vals[j]:
                        active_obs.append({
                            "type": "bullish",
                            "zone_low": low_vals[j],
                            "zone_high": high_vals[j],
                            "formed_at": i,
                            "break_dist": break_dist,
                        })
                        break
                last_swing_hi = close_vals[i]

            if bearish_bos:
                break_dist = last_swing_lo - close_vals[i]
                for j in range(i - 1, max(i - 10, 0), -1):
                    if close_vals[j] > open_vals[j]:
                        active_obs.append({
                            "type": "bearish",
                            "zone_low": low_vals[j],
                            "zone_high": high_vals[j],
                            "formed_at": i,
                            "break_dist": break_dist,
                        })
                        break
                last_swing_lo = close_vals[i]

            # Expire old OBs and limit count
            active_obs = [
                ob for ob in active_obs if (i - ob["formed_at"]) <= self.ob_max_age
            ]
            if len(active_obs) > self.max_active_obs:
                active_obs = active_obs[-self.max_active_obs:]

            # Check if price returns to any active OB zone
            best_score = 0.0
            best_dir = 0

            for ob in active_obs:
                age = i - ob["formed_at"]
                if age < 1:
                    continue

                zone_mid = (ob["zone_low"] + ob["zone_high"]) / 2.0

                # Check proximity to zone
                if ob["type"] == "bullish":
                    if low_vals[i] <= ob["zone_high"] + self.ob_proximity_atr * a:
                        if close_vals[i] > zone_mid:
                            d = 1
                        else:
                            continue
                    else:
                        continue
                else:
                    if high_vals[i] >= ob["zone_low"] - self.ob_proximity_atr * a:
                        if close_vals[i] < zone_mid:
                            d = -1
                        else:
                            continue
                    else:
                        continue

                # Strength components
                dist_score = min(ob["break_dist"] / a / 2.0, 1.0)
                freshness = 0.5 ** (age / max(self.ob_max_age, 1))

                # FVG overlap bonus
                fvg_overlap = 0.0
                for fvg in fvgs:
                    if (fvg["zone_low"] <= ob["zone_high"]
                            and fvg["zone_high"] >= ob["zone_low"]):
                        fvg_overlap = 1.0
                        break

                vol_score = min(vol_vals[i] / vol_avg_vals[i] / 3.0, 1.0)

                score = (0.30 * dist_score + 0.25 * freshness
                         + 0.20 * fvg_overlap + 0.25 * vol_score)

                if score > best_score:
                    best_score = score
                    best_dir = d

            if best_dir != 0:
                direction.iloc[i] = best_dir
                strength.iloc[i] = round(min(best_score, 1.0), 6)
                formation_idx.iloc[i] = i

        # Forward-fill
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        dir_filled = direction.replace(0, np.nan).ffill().fillna(0).astype(int)
        str_filled = strength.replace(0.0, np.nan).ffill().fillna(0.0)
        has_signal = formation_idx > 0
        direction = dir_filled.where(has_signal, 0)
        strength = str_filled.where(has_signal, 0.0).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_ORDER_BLOCKS_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "pivot_left": self.pivot_left,
                "pivot_right": self.pivot_right,
                "max_active_obs": self.max_active_obs,
                "ob_max_age": self.ob_max_age,
                "ob_proximity_atr": self.ob_proximity_atr,
                "min_gap_atr": self.min_gap_atr,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Order block zones from BOS with FVG overlap confluence",
                "bars_processed": len(df),
            },
        )
