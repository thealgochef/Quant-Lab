"""
Category 17: Sweep + FVG Combo Detector.

High conviction signal: liquidity sweep followed by FVG formation.
Combines Categories 4 and 5 for confluence.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class SweepFVGComboDetector(SignalDetector):
    """Sweep + FVG Combo: sweep followed by FVG, high conviction."""
    detector_id = "sweep_fvg_combo"
    category = "sweep_fvg_combo"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
