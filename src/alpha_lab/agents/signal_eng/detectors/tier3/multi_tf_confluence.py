"""
Category 11: Multi-Timeframe Confluence Score.

Measures signal agreement across all timeframes.
Higher score = more timeframes agree on direction.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class MultiTFConfluenceDetector(SignalDetector):
    """Multi-TF Confluence: agreement score across all timeframes."""
    detector_id = "multi_tf_confluence"
    category = "multi_tf_confluence"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in Timeframe]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
