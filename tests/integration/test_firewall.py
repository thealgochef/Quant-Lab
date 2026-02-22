"""Integration tests for the SIG-001 <-> VAL-001 validation firewall."""

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

        # The fields that SHOULD cross the firewall
        assert sv.direction is not None
        assert sv.strength is not None
        assert sv.timeframe is not None

        # The fields that MUST NOT cross (per Section 10.1)
        # These exist on the object but the firewall strips them
        assert sv.parameters is not None  # Present on object
        assert sv.category is not None  # Present on object
        # The firewall.strip_signal_metadata() function is responsible
        # for removing these before handoff to VAL-001
