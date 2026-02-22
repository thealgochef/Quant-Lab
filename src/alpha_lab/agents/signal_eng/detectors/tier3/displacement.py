"""
Category 13: Displacement Signals.

Detects large body candles after consolidation, especially when
accompanied by a Fair Value Gap formation.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class DisplacementDetector(SignalDetector):
    """Displacement: large body after consolidation + FVG."""
    detector_id = "displacement"
    category = "displacement"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
