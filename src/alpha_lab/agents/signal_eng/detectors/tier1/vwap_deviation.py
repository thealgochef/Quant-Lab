"""
Category 3: VWAP Deviation Detector.

Measures price deviation from session-anchored VWAP.
Signals based on standard deviation bands, z-score, and volume confirmation.

Signal composition:
- direction: +1 (price extended above VWAP, potential mean-reversion short or
  momentum long), -1 (extended below), 0 (near VWAP / no signal)
- strength: z-score magnitude + volume confirmation + VWAP slope
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_session_vwap_bands,
)
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

# Defaults
_BAND_STD = 2.0
_ZSCORE_THRESHOLD = 1.0


class VwapDeviationDetector(SignalDetector):
    """VWAP Deviation: std dev bands, slope, session-anchored VWAP."""

    detector_id = "vwap_deviation"
    category = "vwap_deviation"
    tier = SignalTier.CORE
    timeframes = [
        tf.value
        for tf in [
            Timeframe.M1,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
        ]
    ]

    def __init__(
        self,
        band_std: float = _BAND_STD,
        zscore_threshold: float = _ZSCORE_THRESHOLD,
    ) -> None:
        self.band_std = band_std
        self.zscore_threshold = zscore_threshold

    def validate_inputs(self, data: DataBundle) -> bool:
        """Need at least one intraday timeframe with volume data."""
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if (
                isinstance(df, pd.DataFrame)
                and len(df) > 20
                and "volume" in df.columns
            ):
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        """Compute VWAP deviation signals across intraday timeframes."""
        signals: list[SignalVector] = []

        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= 20:
                continue
            if "volume" not in df.columns:
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
        """Compute VWAP deviation for a single timeframe."""
        close = df["close"]
        volume = df["volume"]

        vwap, upper, lower = compute_session_vwap_bands(df, self.band_std)
        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        # --- Z-score: price deviation from VWAP normalized by band width ---
        band_width = (upper - lower).replace(0, np.nan)
        deviation = close - vwap
        zscore = (deviation / band_width).fillna(0.0)

        # --- Direction ---
        # Above upper band (+1) or below lower band (-1)
        # This signals price extension from fair value
        above = zscore > self.zscore_threshold
        below = zscore < -self.zscore_threshold

        direction = pd.Series(0, index=df.index, dtype=int)
        direction = direction.where(~above, 1)
        direction = direction.where(~below, -1)

        # --- Volume confirmation ---
        vol_sma = volume.rolling(window=20, min_periods=1).mean()
        vol_ratio = (volume / vol_sma.replace(0, np.nan)).fillna(1.0)
        vol_confirmation = vol_ratio.clip(0.0, 3.0) / 3.0  # Normalize to [0, 1]

        # --- VWAP slope (rising/falling fair value) ---
        vwap_slope = vwap.diff(5) / atr_safe
        slope_rank = vwap_slope.abs().rolling(window=min(50, len(df))).rank(pct=True)
        slope_rank = slope_rank.fillna(0.0)

        # --- Strength: z-score magnitude + volume + slope ---
        zscore_magnitude = zscore.abs().clip(0.0, 3.0) / 3.0
        strength = (
            0.5 * zscore_magnitude + 0.3 * vol_confirmation + 0.2 * slope_rank
        ).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        # Formation index
        direction_change = direction.diff().fillna(0) != 0
        formation_idx = pd.Series(np.nan, index=df.index)
        formation_idx[direction_change] = np.arange(len(df))[direction_change.values]
        formation_idx = formation_idx.ffill().fillna(0).astype(int)

        return SignalVector(
            signal_id=f"SIG_VWAP_DEVIATION_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "band_std": self.band_std,
                "zscore_threshold": self.zscore_threshold,
            },
            metadata={
                "instrument": instrument,
                "intuition": "Price extension from session VWAP with volume confirmation",
                "bars_processed": len(df),
            },
        )
