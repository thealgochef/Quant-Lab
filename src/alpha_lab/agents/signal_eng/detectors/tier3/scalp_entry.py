"""
Category 16: Scalp Entry Timing.

Uses lower-timeframe signals to time entries for higher-timeframe setups.
Combines micro-structure with macro direction.

Signal composition:
- direction: +1/-1 when micro momentum confirms macro direction, 0 otherwise
- strength: combines momentum magnitude, macro alignment score, and volume
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr, compute_ema
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class ScalpEntryDetector(SignalDetector):
    """Scalp Entry: lower-TF entry timing for higher-TF setups."""

    detector_id = "scalp_entry"
    category = "scalp_entry"
    tier = SignalTier.COMPOSITE
    timeframes = [
        Timeframe.TICK_987.value, Timeframe.TICK_2000.value,
        Timeframe.M1.value, Timeframe.M3.value,
    ]

    def __init__(
        self,
        macro_ema_fast: int = 13,
        macro_ema_slow: int = 48,
        macro_timeframes: list[str] | None = None,
        momentum_lookback: int = 5,
        min_velocity_atr: float = 0.3,
    ) -> None:
        self.macro_ema_fast = macro_ema_fast
        self.macro_ema_slow = macro_ema_slow
        self.macro_timeframes = macro_timeframes or ["15m", "1H"]
        self.momentum_lookback = momentum_lookback
        self.min_velocity_atr = min_velocity_atr

    def validate_inputs(self, data: DataBundle) -> bool:
        # Need at least one micro TF and one macro TF
        has_micro = False
        has_macro = False
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > _MIN_BARS:
                has_micro = True
        for tf in self.macro_timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > _MIN_BARS:
                has_macro = True
        return has_micro and has_macro

    def compute(self, data: DataBundle) -> list[SignalVector]:
        # Determine macro direction from higher TFs
        macro_dirs: list[int] = []
        for tf in self.macro_timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            close = df["close"]
            ema_f = compute_ema(close, self.macro_ema_fast)
            ema_s = compute_ema(close, self.macro_ema_slow)
            if ema_f.iloc[-1] > ema_s.iloc[-1]:
                macro_dirs.append(1)
            elif ema_f.iloc[-1] < ema_s.iloc[-1]:
                macro_dirs.append(-1)

        if not macro_dirs:
            return []

        n_bull = sum(1 for d in macro_dirs if d > 0)
        n_bear = sum(1 for d in macro_dirs if d < 0)
        if n_bull > n_bear:
            macro_dir = 1
        elif n_bear > n_bull:
            macro_dir = -1
        else:
            macro_dir = 0

        macro_alignment = max(n_bull, n_bear) / len(macro_dirs) if macro_dirs else 0.0

        # Generate signals on micro TFs
        signals: list[SignalVector] = []
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            sv = self._compute_timeframe(df, tf, data.instrument, macro_dir, macro_alignment)
            if sv is not None:
                signals.append(sv)

        return signals

    def _compute_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        instrument: str,
        macro_dir: int,
        macro_alignment: float,
    ) -> SignalVector | None:
        if macro_dir == 0:
            return None

        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # Micro momentum: velocity of price change / ATR
        velocity = close.diff(self.momentum_lookback) / atr_safe
        velocity = velocity.fillna(0.0)

        # Direction: micro momentum matches macro direction
        micro_bull = velocity > self.min_velocity_atr
        micro_bear = velocity < -self.min_velocity_atr

        direction = pd.Series(0, index=df.index, dtype=int)
        if macro_dir == 1:
            direction = direction.where(~micro_bull, 1)
        else:
            direction = direction.where(~micro_bear, -1)

        # Strength components
        momentum_mag = velocity.abs().clip(0.0, 3.0) / 3.0
        vol_avg = volume.rolling(window=20, min_periods=1).mean().replace(0, 1.0)
        vol_score = (volume / vol_avg).clip(0.0, 3.0) / 3.0

        strength = (
            0.40 * momentum_mag
            + 0.35 * macro_alignment
            + 0.25 * vol_score
        ).clip(0.0, 1.0).fillna(0.0)
        strength = strength.where(direction != 0, 0.0)

        # Formation index
        dir_change = direction.diff().fillna(0) != 0
        formation_idx = pd.Series(np.nan, index=df.index)
        formation_idx[dir_change] = np.arange(len(df))[dir_change.values]
        formation_idx = formation_idx.ffill().fillna(0).astype(int)

        return SignalVector(
            signal_id=f"SIG_SCALP_ENTRY_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "macro_ema_fast": self.macro_ema_fast,
                "macro_ema_slow": self.macro_ema_slow,
                "macro_timeframes": self.macro_timeframes,
                "momentum_lookback": self.momentum_lookback,
                "min_velocity_atr": self.min_velocity_atr,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Lower-TF momentum burst confirming higher-TF direction",
                "bars_processed": len(df),
                "macro_direction": macro_dir,
                "macro_alignment": macro_alignment,
            },
        )
