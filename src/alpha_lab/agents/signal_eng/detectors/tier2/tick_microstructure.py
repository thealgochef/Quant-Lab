"""
Category 10: Tick Microstructure Detector (adapted to 1m bars).

Originally designed for 987-tick and 2000-tick chart patterns, but
adapted to use 1m OHLCV bars as a velocity/momentum proxy since
tick data is not available from the Polygon.io provider.

Detects velocity bursts, momentum streaks, and volume thrusts.

Signal composition:
- direction: +1 bullish velocity/momentum, -1 bearish, 0 no signal
- strength: combines velocity magnitude, streak consistency, volume spike
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class TickMicrostructureDetector(SignalDetector):
    """Tick Microstructure: velocity bursts, momentum streaks (1m proxy)."""

    detector_id = "tick_microstructure"
    category = "tick_microstructure"
    tier = SignalTier.ICT_STRUCTURAL
    # Adapted: use 1m bars instead of tick-based timeframes
    timeframes = [Timeframe.M1.value]

    def __init__(
        self,
        min_velocity_atr: float = 0.3,
        streak_length: int = 5,
        volume_spike_ratio: float = 3.0,
        lookback_window: int = 20,
    ) -> None:
        self.min_velocity_atr = min_velocity_atr
        self.streak_length = streak_length
        self.volume_spike_ratio = volume_spike_ratio
        self.lookback_window = lookback_window

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
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index, dtype=float)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        closes = df["close"].values
        opens = df["open"].values
        volumes = df["volume"].values
        # Bar-level velocity: (close - open) / ATR
        velocity = (
            pd.Series(closes - opens) / atr_safe.values
        ).fillna(0).values

        # Rolling volume average
        vol_avg = pd.Series(volumes).rolling(
            self.lookback_window, min_periods=1,
        ).mean().values

        # Momentum streak: count consecutive same-direction closes
        bar_dir = np.sign(closes[1:] - closes[:-1])
        bar_dir = np.concatenate([[0], bar_dir])
        streak = np.zeros(len(df))
        for i in range(1, len(df)):
            if bar_dir[i] == bar_dir[i - 1] and bar_dir[i] != 0:
                streak[i] = streak[i - 1] + 1
            elif bar_dir[i] != 0:
                streak[i] = 1
            else:
                streak[i] = 0

        for i in range(self.lookback_window, len(df)):
            vel = velocity[i]
            vel_abs = abs(vel)

            # Volume spike
            vol_ratio = (
                volumes[i] / vol_avg[i] if vol_avg[i] > 0 else 1.0
            )
            has_vol_spike = vol_ratio >= self.volume_spike_ratio

            # Velocity burst
            has_velocity = vel_abs >= self.min_velocity_atr

            # Momentum streak
            has_streak = streak[i] >= self.streak_length

            # Need at least one trigger
            if not (has_velocity or has_streak or has_vol_spike):
                continue

            # Direction from velocity or streak
            if has_velocity:
                sig_dir = 1 if vel > 0 else -1
            elif has_streak:
                sig_dir = int(bar_dir[i])
            else:
                # Volume spike: use bar direction
                sig_dir = 1 if closes[i] > opens[i] else -1

            if sig_dir == 0:
                continue

            # Strength components
            vel_score = min(vel_abs / 1.0, 1.0) if has_velocity else 0.0
            streak_score = min(
                streak[i] / (self.streak_length * 2), 1.0,
            ) if has_streak else 0.0
            vol_score = min(
                vol_ratio / (self.volume_spike_ratio * 2), 1.0,
            ) if has_vol_spike else 0.0

            # Weight by which triggers fired
            active = [
                (0.40, vel_score),
                (0.30, streak_score),
                (0.30, vol_score),
            ]
            score = sum(w * s for w, s in active)

            direction.iloc[i] = sig_dir
            strength.iloc[i] = round(min(score, 1.0), 6)

            # Track formation
            if i > 0 and direction.iloc[i - 1] != sig_dir:
                formation_idx.iloc[i] = i

        # Forward-fill formation_idx
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        # Zero strength when neutral
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_TICK_MICROSTRUCTURE_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "min_velocity_atr": self.min_velocity_atr,
                "streak_length": self.streak_length,
                "volume_spike_ratio": self.volume_spike_ratio,
                "lookback_window": self.lookback_window,
            },
            metadata={
                "instrument": instrument,
                "intuition": "1m velocity bursts and momentum streaks",
                "bars_processed": len(df),
                "adapted_from": "tick-based to 1m bars (no tick data)",
            },
        )
