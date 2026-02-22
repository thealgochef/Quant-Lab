"""
Category 18: EMA Reclaim Detector.

Detects price reclaiming the 13 EMA after sweeping below the 48 EMA.
A bullish reclaim pattern indicating institutional accumulation.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class EmaReclaimDetector(SignalDetector):
    """EMA Reclaim: reclaiming 13 EMA after sweeping below 48 EMA."""
    detector_id = "ema_reclaim"
    category = "ema_reclaim"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
