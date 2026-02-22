"""
Category 14: Order Block Signals.

Identifies order blocks (OB) and detects OB+FVG overlap zones
for high-probability entry points.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class OrderBlocksDetector(SignalDetector):
    """Order Blocks: OB identification, OB+FVG overlap."""
    detector_id = "order_blocks"
    category = "order_blocks"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
