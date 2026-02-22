"""
Category 9: Previous Day Levels Point of Interest Detector.

Uses previous day high/low/mid/close as points of interest.
Detects sweeps of PD levels and price reactions at these levels.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class PDLevelsPOIDetector(SignalDetector):
    """PD Levels POI: PD levels as sweep targets, reactions."""
    detector_id = "pd_levels_poi"
    category = "pd_levels_poi"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [tf.value for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError("PDLevelsPOIDetector.compute not yet implemented")
