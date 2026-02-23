"""
Category 9: Previous Day Levels Point of Interest Detector.

Uses previous day high/low/mid/close as support/resistance.
Detects bounces and rejections at PD levels.

Signal composition:
- direction: +1 on support bounce, -1 on resistance rejection
- strength: combines level importance, proximity, reaction, and volume
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30

_DEFAULT_LEVEL_WEIGHTS = {
    "pd_high": 1.0,
    "pd_low": 1.0,
    "pd_mid": 0.8,
    "pd_close": 0.75,
    "pw_high": 0.9,
    "pw_low": 0.9,
    "overnight_high": 0.7,
    "overnight_low": 0.7,
}


class PDLevelsPOIDetector(SignalDetector):
    """PD Levels POI: PD levels as sweep targets, reactions."""

    detector_id = "pd_levels_poi"
    category = "pd_levels_poi"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [
        tf.value for tf in [
            Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1,
        ]
    ]

    def __init__(
        self,
        level_proximity_atr: float = 0.5,
        min_reaction_body_ratio: float = 0.5,
        level_weights: dict[str, float] | None = None,
    ) -> None:
        self.level_proximity_atr = level_proximity_atr
        self.min_reaction_body_ratio = min_reaction_body_ratio
        self.level_weights = level_weights or dict(_DEFAULT_LEVEL_WEIGHTS)

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

    def _get_levels_for_date(
        self, date_str: str, data: DataBundle,
    ) -> list[tuple[str, float, float]]:
        """Get (name, price, importance_weight) for all PD levels."""
        pdl = data.pd_levels.get(date_str)
        if pdl is None:
            return []
        levels = []
        for attr in [
            "pd_high", "pd_low", "pd_mid", "pd_close",
            "pw_high", "pw_low", "overnight_high", "overnight_low",
        ]:
            val = getattr(pdl, attr, None)
            if val is not None and val > 0:
                weight = self.level_weights.get(attr, 0.5)
                levels.append((attr, float(val), weight))
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
        opens = df["open"].values
        closes = df["close"].values
        volumes = df["volume"].values
        vol_avg = pd.Series(volumes).rolling(20, min_periods=1).mean().values
        atr_vals = atr_safe.values

        # Cache levels per date
        level_cache: dict[str, list[tuple[str, float, float]]] = {}

        for i in range(len(df)):
            # Get date string
            try:
                date_str = str(df.index[i].date())
            except (AttributeError, TypeError):
                continue

            if date_str not in level_cache:
                level_cache[date_str] = self._get_levels_for_date(
                    date_str, data,
                )
            levels = level_cache[date_str]
            if not levels:
                continue

            best_score = 0.0
            best_dir = 0

            for _name, level_price, importance in levels:
                dist = abs(closes[i] - level_price)
                dist_atr = dist / atr_vals[i] if atr_vals[i] > 0 else 999

                if dist_atr > self.level_proximity_atr:
                    continue

                # Check for bounce/rejection
                body = abs(closes[i] - opens[i])
                candle_range = highs[i] - lows[i]
                if candle_range <= 0:
                    continue

                body_ratio = body / candle_range

                # Bullish bounce: low near/below level, close above
                is_bounce = (
                    lows[i] <= level_price + atr_vals[i] * 0.1
                    and closes[i] > level_price
                    and body_ratio >= self.min_reaction_body_ratio
                )
                # Bearish rejection: high near/above level, close below
                is_reject = (
                    highs[i] >= level_price - atr_vals[i] * 0.1
                    and closes[i] < level_price
                    and body_ratio >= self.min_reaction_body_ratio
                )

                if not is_bounce and not is_reject:
                    continue

                sig_dir = 1 if is_bounce else -1

                # Strength components
                imp_score = importance
                prox_score = max(
                    1.0 - dist_atr / self.level_proximity_atr, 0.0,
                )
                react_score = min(body_ratio, 1.0)
                vol_ratio = (
                    volumes[i] / vol_avg[i]
                    if vol_avg[i] > 0 else 1.0
                )
                vol_score = min(vol_ratio / 3.0, 1.0)

                score = (
                    0.35 * imp_score
                    + 0.30 * prox_score
                    + 0.20 * react_score
                    + 0.15 * vol_score
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
            signal_id=f"SIG_PD_LEVELS_POI_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "level_proximity_atr": self.level_proximity_atr,
                "min_reaction_body_ratio": self.min_reaction_body_ratio,
            },
            metadata={
                "instrument": data.instrument,
                "intuition": "PD high/low/mid/close as S/R",
                "bars_processed": len(df),
            },
        )
