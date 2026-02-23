"""
Category 8: Killzone Timing Detector.

Generates timing signals based on killzone activity.
London open (2-5am ET), NY AM (8-11am ET), Asia (7-10pm ET).

Signal composition:
- direction: +1 bullish momentum in killzone, -1 bearish, 0 outside
- strength: combines activity vs average, directional consistency,
  and proximity to killzone start
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

_MIN_BARS = 30


class KillzoneTimingDetector(SignalDetector):
    """Killzone Timing: London/NY/Asia open, session overlap."""

    detector_id = "killzone_timing"
    category = "killzone_timing"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [
        tf.value for tf in [
            Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1,
        ]
    ]

    def __init__(
        self,
        london_hours: tuple[int, int] = (2, 5),
        ny_hours: tuple[int, int] = (8, 11),
        asia_hours: tuple[int, int] = (19, 22),
        min_activity_ratio: float = 1.2,
        direction_window: int = 12,
    ) -> None:
        self.london_hours = london_hours
        self.ny_hours = ny_hours
        self.asia_hours = asia_hours
        self.min_activity_ratio = min_activity_ratio
        self.direction_window = direction_window

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

    def _in_killzone(self, hour: float) -> tuple[bool, str, float]:
        """Check if fractional hour falls in a killzone.

        Args:
            hour: Fractional hour (e.g. 8.5 = 8:30am)

        Returns (in_kz, kz_name, proximity_to_start).
        proximity_to_start is 1.0 at start, decaying toward 0 at end.
        """
        for name, (start, end) in [
            ("LONDON", self.london_hours),
            ("NEW_YORK", self.ny_hours),
            ("ASIA", self.asia_hours),
        ]:
            if start <= hour < end:
                span = end - start
                elapsed = hour - start
                proximity = max(1.0 - elapsed / span, 0.0)
                return True, name, proximity
        return False, "NONE", 0.0

    def _compute_timeframe(
        self, df: pd.DataFrame, timeframe: str, instrument: str,
    ) -> SignalVector | None:
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index, dtype=float)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        closes = df["close"].values
        volumes = df["volume"].values
        vol_avg = pd.Series(volumes).rolling(
            min(100, len(df)), min_periods=1,
        ).mean().values
        atr_avg = atr_safe.rolling(
            min(100, len(df)), min_periods=1,
        ).mean().values

        # Bar-level returns for momentum
        returns = pd.Series(closes).diff().fillna(0.0).values

        # Extract fractional hours for sub-hourly precision
        try:
            hours = df.index.hour + df.index.minute / 60.0
        except AttributeError:
            return None

        last_kz_start = 0

        for i in range(self.direction_window, len(df)):
            hour = float(hours[i])
            in_kz, kz_name, proximity = self._in_killzone(hour)

            if not in_kz:
                continue

            # Activity ratio: current ATR vs rolling average
            bar_range = df["high"].iloc[i] - df["low"].iloc[i]
            activity = bar_range / atr_avg[i] if atr_avg[i] > 0 else 0
            vol_ratio = (
                volumes[i] / vol_avg[i] if vol_avg[i] > 0 else 1.0
            )
            activity_score = (activity + vol_ratio) / 2.0

            if activity_score < self.min_activity_ratio * 0.5:
                continue

            # Directional consistency: % of recent bars trending same way
            window = returns[max(0, i - self.direction_window + 1): i + 1]
            n_up = np.sum(window > 0)
            n_down = np.sum(window < 0)
            total = len(window)

            if n_up > n_down:
                sig_dir = 1
                consistency = n_up / total if total > 0 else 0
            elif n_down > n_up:
                sig_dir = -1
                consistency = n_down / total if total > 0 else 0
            else:
                continue

            # Strength components
            act_score = min(activity_score / 2.0, 1.0)
            cons_score = consistency
            prox_score = proximity

            score = (
                0.50 * act_score
                + 0.30 * cons_score
                + 0.20 * prox_score
            )

            direction.iloc[i] = sig_dir
            strength.iloc[i] = round(min(score, 1.0), 6)

            # Track killzone start
            if i > 0 and not self._in_killzone(float(hours[i - 1]))[0]:
                last_kz_start = i
            formation_idx.iloc[i] = last_kz_start

        # Zero strength when neutral
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_KILLZONE_TIMING_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "london_hours": list(self.london_hours),
                "ny_hours": list(self.ny_hours),
                "asia_hours": list(self.asia_hours),
                "min_activity_ratio": self.min_activity_ratio,
                "direction_window": self.direction_window,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Momentum within killzone windows",
                "bars_processed": len(df),
            },
        )
