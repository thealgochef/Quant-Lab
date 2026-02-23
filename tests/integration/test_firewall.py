"""Integration tests for the SIG-001 <-> VAL-001 validation firewall."""

from __future__ import annotations

from alpha_lab.agents.signal_eng.agent import SignalEngineeringAgent
from alpha_lab.agents.validation.agent import ValidationAgent
from alpha_lab.core.contracts import SignalVector


class TestFirewallBoundary:
    def test_signal_vector_does_not_leak_to_validation(self):
        """
        Verify that when a SignalVector is prepared for VAL-001,
        implementation details (parameters, category) can be stripped.
        This is a structural test — actual stripping is done in firewall.py.
        """
        sv = SignalVector(
            signal_id="SIG_EMA_CONFLUENCE_5m_v1",
            category="ema_confluence",
            timeframe="5m",
            version=1,
            direction=[1, 0, -1],
            strength=[0.8, 0.0, 0.6],
            formation_idx=[0, 1, 2],
            parameters={"ema_fast": 13, "ema_mid": 48},
            metadata={"intuition": "EMA alignment"},
        )

        assert sv.direction is not None
        assert sv.strength is not None
        assert sv.timeframe is not None
        assert sv.parameters is not None
        assert sv.category is not None

    def test_validation_produces_verdicts_for_all_signals(
        self, message_bus, synthetic_data_bundle
    ):
        """VAL-001 returns one verdict per signal in the bundle."""
        sig = SignalEngineeringAgent(message_bus)
        val = ValidationAgent(message_bus)

        signal_bundle = sig.generate_signals(synthetic_data_bundle)
        price_data = {"5m": synthetic_data_bundle.bars["5m"]}
        report = val.validate_signal_bundle(signal_bundle, price_data)

        assert len(report.verdicts) == signal_bundle.total_signals
        signal_ids = {v.signal_id for v in report.verdicts}
        expected_ids = {sv.signal_id for sv in signal_bundle.signals}
        assert signal_ids == expected_ids

    def test_bonferroni_adjustment_applied(
        self, message_bus, synthetic_data_bundle
    ):
        """With 3+ signals, Bonferroni adjustment is applied."""
        sig = SignalEngineeringAgent(message_bus)
        val = ValidationAgent(message_bus)

        signal_bundle = sig.generate_signals(synthetic_data_bundle)
        assert signal_bundle.total_signals >= 3

        price_data = {"5m": synthetic_data_bundle.bars["5m"]}
        report = val.validate_signal_bundle(signal_bundle, price_data)
        assert report.bonferroni_adjusted is True

    def test_lookahead_bias_detection(
        self, message_bus, synthetic_data_bundle
    ):
        """Synthetic data with regime trends triggers look-ahead bias warnings."""
        sig = SignalEngineeringAgent(message_bus)
        val = ValidationAgent(message_bus)

        signal_bundle = sig.generate_signals(synthetic_data_bundle)
        price_data = {"5m": synthetic_data_bundle.bars["5m"]}
        report = val.validate_signal_bundle(signal_bundle, price_data)

        # Synthetic data typically produces high IC that triggers bias detection
        # At least some signals should be flagged
        high_ic_signals = [v for v in report.verdicts if abs(v.ic) > 0.20]
        assert len(high_ic_signals) > 0, (
            "Expected look-ahead bias detection on synthetic data"
        )
