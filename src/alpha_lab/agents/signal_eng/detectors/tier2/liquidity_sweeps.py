"""
Category 4: Liquidity Sweeps Detector.

Detects PD H/L sweeps, session sweeps, and stop-hunts.
A sweep occurs when price wicks through a known level then reverses.

Signal composition:
- direction: +1 bullish sweep (wicks below, closes above), -1 bearish
- strength: combines sweep magnitude, reversal quality, and volume
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class LiquiditySweepsDetector(SignalDetector):
    """Liquidity Sweeps: PD H/L sweeps, session sweeps, stop-hunts."""

    detector_id = "liquidity_sweeps"
    category = "liquidity_sweeps"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [
        tf.value for tf in [
            Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1,
        ]
    ]

    def __init__(
        self,
        sweep_buffer_atr: float = 0.1,
        min_reversal_ratio: float = 0.3,
    ) -> None:
        self.sweep_buffer_atr = sweep_buffer_atr
        self.min_reversal_ratio = min_reversal_ratio

    def validate_inputs(self, data: DataBundle) -> bool:
        if not data.pd_levels:
            return False
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > _MIN_BARS:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        if not data.pd_levels:
            return signals
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            sv = self._compute_timeframe(df, tf, data)
            if sv is not None:
                signals.append(sv)
        return signals

    def _get_sweep_levels(
        self, date_str: str, data: DataBundle,
    ) -> list[tuple[str, float, str]]:
        """Get (name, price, type) for all sweepable levels.

        type is 'support' (price should be below) or 'resistance' (above).
        """
        pdl = data.pd_levels.get(date_str)
        if pdl is None:
            return []
        levels = []
        # Support levels (sweep = wick below then close above)
        for attr in ["pd_low", "pw_low", "overnight_low"]:
            val = getattr(pdl, attr, None)
            if val is not None and val > 0:
                levels.append((attr, float(val), "support"))
        # Resistance levels (sweep = wick above then close below)
        for attr in ["pd_high", "pw_high", "overnight_high"]:
            val = getattr(pdl, attr, None)
            if val is not None and val > 0:
                levels.append((attr, float(val), "resistance"))
        return levels

    def _compute_timeframe(
        self, df: pd.DataFrame, timeframe: str, data: DataBundle,
    ) -> SignalVector | None:
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index, dtype=float)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values
        vol_avg = pd.Series(volumes).rolling(20, min_periods=1).mean().values
        atr_vals = atr_safe.values

        # Also track rolling session high/low for intraday sweeps
        session_high = pd.Series(highs).rolling(78, min_periods=1).max().values
        session_low = pd.Series(lows).rolling(78, min_periods=1).min().values

        level_cache: dict[str, list[tuple[str, float, str]]] = {}

        for i in range(1, len(df)):
            try:
                date_str = str(df.index[i].date())
            except (AttributeError, TypeError):
                continue

            if date_str not in level_cache:
                level_cache[date_str] = self._get_sweep_levels(
                    date_str, data,
                )
            levels = list(level_cache[date_str])

            # Add rolling session levels
            if i > 10:
                levels.append(
                    ("session_high", session_high[i - 1], "resistance"),
                )
                levels.append(
                    ("session_low", session_low[i - 1], "support"),
                )

            best_score = 0.0
            best_dir = 0

            for _name, level_price, level_type in levels:
                buffer = atr_vals[i] * self.sweep_buffer_atr

                if level_type == "support":
                    # Bullish sweep: wick below support, close above
                    swept = lows[i] < level_price - buffer
                    closed_above = closes[i] > level_price
                    if not (swept and closed_above):
                        continue

                    penetration = level_price - lows[i]
                    wick_below = level_price - lows[i]
                    recovery = closes[i] - level_price

                    if wick_below <= 0:
                        continue
                    reversal = recovery / (recovery + wick_below)
                    if reversal < self.min_reversal_ratio:
                        continue

                    sig_dir = 1

                else:
                    # Bearish sweep: wick above resistance, close below
                    swept = highs[i] > level_price + buffer
                    closed_below = closes[i] < level_price
                    if not (swept and closed_below):
                        continue

                    penetration = highs[i] - level_price
                    wick_above = highs[i] - level_price
                    recovery = level_price - closes[i]

                    if wick_above <= 0:
                        continue
                    reversal = recovery / (recovery + wick_above)
                    if reversal < self.min_reversal_ratio:
                        continue

                    sig_dir = -1

                # Strength components
                sweep_mag = min(
                    penetration / atr_vals[i], 1.0,
                ) if atr_vals[i] > 0 else 0
                reversal_score = min(reversal, 1.0)
                vol_ratio = (
                    volumes[i] / vol_avg[i]
                    if vol_avg[i] > 0 else 1.0
                )
                vol_score = min(vol_ratio / 3.0, 1.0)

                score = (
                    0.40 * sweep_mag
                    + 0.30 * reversal_score
                    + 0.30 * vol_score
                )

                if score > best_score:
                    best_score = score
                    best_dir = sig_dir

            direction.iloc[i] = best_dir
            strength.iloc[i] = round(min(best_score, 1.0), 6)
            if best_dir != 0:
                formation_idx.iloc[i] = i

        # Forward-fill formation_idx
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        # Zero strength when neutral
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_LIQUIDITY_SWEEPS_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "sweep_buffer_atr": self.sweep_buffer_atr,
                "min_reversal_ratio": self.min_reversal_ratio,
            },
            metadata={
                "instrument": data.instrument,
                "intuition": "Stop-hunt / liquidity grab reversal",
                "bars_processed": len(df),
            },
        )
