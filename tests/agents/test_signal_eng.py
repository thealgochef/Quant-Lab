"""Tests for Signal Engineering agent and detector registry."""

# Import detectors to trigger auto-registration
import alpha_lab.agents.signal_eng.detectors  # noqa: F401
from alpha_lab.agents.signal_eng.agent import SignalEngineeringAgent
from alpha_lab.agents.signal_eng.detector_base import SignalDetector, SignalDetectorRegistry
from alpha_lab.core.enums import AgentID, SignalTier


class TestSignalEngineeringAgent:
    def test_create(self, message_bus):
        agent = SignalEngineeringAgent(message_bus)
        assert agent.agent_id == AgentID.SIGNAL_ENG


class TestDetectorRegistry:
    def test_all_20_detectors_registered(self):
        assert SignalDetectorRegistry.count() == 20

    def test_tier1_has_3(self):
        tier1 = SignalDetectorRegistry.get_by_tier(SignalTier.CORE)
        assert len(tier1) == 3

    def test_tier2_has_7(self):
        tier2 = SignalDetectorRegistry.get_by_tier(SignalTier.ICT_STRUCTURAL)
        assert len(tier2) == 7

    def test_tier3_has_10(self):
        tier3 = SignalDetectorRegistry.get_by_tier(SignalTier.COMPOSITE)
        assert len(tier3) == 10

    def test_get_by_id(self):
        cls = SignalDetectorRegistry.get("ema_confluence")
        assert cls.detector_id == "ema_confluence"
        assert cls.tier == SignalTier.CORE

    def test_all_detector_ids(self):
        ids = SignalDetectorRegistry.list_ids()
        expected = [
            "adaptive_regime", "displacement", "ema_confluence",
            "ema_reclaim", "ema_vwap_interaction", "fair_value_gaps",
            "ifvg", "kama_regime", "killzone_timing", "liquidity_sweeps",
            "market_structure", "multi_tf_confluence", "order_blocks",
            "pd_levels_poi", "scalp_entry", "session_gap",
            "sweep_fvg_combo", "tick_microstructure", "volume_profile",
            "vwap_deviation",
        ]
        assert ids == expected

    def test_all_detectors_are_signal_detector_subclass(self):
        for det_id, cls in SignalDetectorRegistry.get_all().items():
            assert issubclass(cls, SignalDetector), f"{det_id} is not a SignalDetector subclass"

    def test_all_detectors_have_required_class_vars(self):
        for det_id, cls in SignalDetectorRegistry.get_all().items():
            assert hasattr(cls, "detector_id"), f"{det_id} missing detector_id"
            assert hasattr(cls, "category"), f"{det_id} missing category"
            assert hasattr(cls, "tier"), f"{det_id} missing tier"
            assert hasattr(cls, "timeframes"), f"{det_id} missing timeframes"
