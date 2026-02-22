"""
Category 16: Scalp Entry Timing.

Uses lower-timeframe signals to time entries for higher-timeframe setups.
Combines micro-structure with macro direction.
"""
from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe


class ScalpEntryDetector(SignalDetector):
    """Scalp Entry: lower-TF entry timing for higher-TF setups."""
    detector_id = "scalp_entry"
    category = "scalp_entry"
    tier = SignalTier.COMPOSITE
    timeframes = [
        Timeframe.TICK_987.value, Timeframe.TICK_2000.value,
        Timeframe.M1.value, Timeframe.M3.value,
    ]

    def compute(self, data: DataBundle) -> list[SignalVector]:
        raise NotImplementedError
