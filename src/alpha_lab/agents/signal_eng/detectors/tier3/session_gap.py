"""
Category 19: Session Gap Detector.

Analyzes overnight gaps: gap probability, fill rate, and directional bias.
Uses gap size relative to ATR for signal strength.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class SessionGapDetector(SignalDetector):
    """Session Gap: overnight gap probability, fill rate, bias."""
    detector_id = "session_gap"
    category = "session_gap"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.D1]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
