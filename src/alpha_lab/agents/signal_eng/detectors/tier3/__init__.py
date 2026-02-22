"""Tier 3 — Composite signals combining multiple indicators and timeframes."""
from alpha_lab.agents.signal_eng.detectors.tier3.adaptive_regime import AdaptiveRegimeDetector
from alpha_lab.agents.signal_eng.detectors.tier3.displacement import DisplacementDetector
from alpha_lab.agents.signal_eng.detectors.tier3.ema_reclaim import EmaReclaimDetector
from alpha_lab.agents.signal_eng.detectors.tier3.ema_vwap_interaction import (
    EmaVwapInteractionDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.multi_tf_confluence import (
    MultiTFConfluenceDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.order_blocks import OrderBlocksDetector
from alpha_lab.agents.signal_eng.detectors.tier3.scalp_entry import ScalpEntryDetector
from alpha_lab.agents.signal_eng.detectors.tier3.session_gap import SessionGapDetector
from alpha_lab.agents.signal_eng.detectors.tier3.sweep_fvg_combo import SweepFVGComboDetector
from alpha_lab.agents.signal_eng.detectors.tier3.volume_profile import VolumeProfileDetector

__all__ = [
    "MultiTFConfluenceDetector", "EmaVwapInteractionDetector", "DisplacementDetector",
    "OrderBlocksDetector", "VolumeProfileDetector", "ScalpEntryDetector",
    "SweepFVGComboDetector", "EmaReclaimDetector", "SessionGapDetector",
    "AdaptiveRegimeDetector",
]
