"""
Category 5: Fair Value Gaps Detector.

Identifies FVG formation (3-candle pattern with gap), tracks fill probability,
and uses FVGs as potential entry zones.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


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

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError("FairValueGapsDetector.compute not yet implemented")
