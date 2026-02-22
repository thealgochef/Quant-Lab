"""
Category 15: Volume Profile Detector.

Identifies Point of Control (POC), Value Area High/Low,
and volume nodes as support/resistance levels.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class VolumeProfileDetector(SignalDetector):
    """Volume Profile: POC, value area, volume nodes as S/R."""
    detector_id = "volume_profile"
    category = "volume_profile"
    tier = SignalTier.COMPOSITE
    timeframes = [
        tf.value for tf in [
            Timeframe.M15, Timeframe.M30, Timeframe.H1,
            Timeframe.H4, Timeframe.D1,
        ]
    ]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
