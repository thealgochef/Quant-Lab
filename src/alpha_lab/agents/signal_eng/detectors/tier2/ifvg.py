"""
Category 6: Inverse Fair Value Gap (IFVG) Detector.

Detects inverse FVG continuation patterns and rejection probability.
IFVGs form when a FVG is filled and then acts as a continuation signal.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class IFVGDetector(SignalDetector):
    """IFVG: inverse FVG continuation, rejection probability."""
    detector_id = "ifvg"
    category = "ifvg"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError("IFVGDetector.compute not yet implemented")
