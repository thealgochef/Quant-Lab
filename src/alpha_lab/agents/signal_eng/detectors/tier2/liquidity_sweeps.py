"""
Category 4: Liquidity Sweeps Detector.

Detects PD H/L sweeps, session sweeps, and stop-hunts.
A sweep occurs when price takes out a known level then reverses.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class LiquiditySweepsDetector(SignalDetector):
    """Liquidity Sweeps: PD H/L sweeps, session sweeps, stop-hunts."""
    detector_id = "liquidity_sweeps"
    category = "liquidity_sweeps"
    tier = SignalTier.ICT_STRUCTURAL
    timeframes = [tf.value for tf in [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError("LiquiditySweepsDetector.compute not yet implemented")
