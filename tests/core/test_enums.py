"""Tests for core enumerations."""

from alpha_lab.core.enums import (
    AgentID,
    AlertLevel,
    MessageType,
    PipelineState,
    SignalTier,
    Timeframe,
    Verdict,
)


class TestAgentID:
    def test_all_six_agents_defined(self):
        assert len(AgentID) == 6

    def test_agent_ids_match_spec(self):
        assert AgentID.ORCHESTRATOR.value == "ORCH-001"
        assert AgentID.DATA_INFRA.value == "DATA-001"
        assert AgentID.SIGNAL_ENG.value == "SIG-001"
        assert AgentID.VALIDATION.value == "VAL-001"
        assert AgentID.EXECUTION.value == "EXEC-001"
        assert AgentID.MONITORING.value == "MON-001"


class TestMessageType:
    def test_all_message_types_defined(self):
        expected = {
            "DATA_REQUEST", "DATA_BUNDLE", "SIGNAL_BUNDLE",
            "VALIDATION_REPORT", "REFINE_REQUEST", "EXECUTION_REQUEST",
            "EXECUTION_REPORT", "DEPLOY_COMMAND", "ALERT",
            "REGIME_SHIFT", "RISK_VETO", "DAILY_REPORT",
            "HALT_COMMAND", "RESUME_COMMAND", "ACK", "NACK",
        }
        actual = {mt.value for mt in MessageType}
        assert actual == expected


class TestTimeframe:
    def test_tick_timeframes(self):
        assert Timeframe.TICK_987.value == "987t"
        assert Timeframe.TICK_2000.value == "2000t"

    def test_all_timeframes_present(self):
        assert len(Timeframe) == 11


class TestSignalTier:
    def test_tier_ordering(self):
        assert SignalTier.CORE < SignalTier.ICT_STRUCTURAL < SignalTier.COMPOSITE


class TestPipelineState:
    def test_all_states_defined(self):
        expected = {
            "INIT", "PHASE_1_2", "PHASE_3_4", "PHASE_5_6",
            "PHASE_7", "PHASE_8_9", "DEPLOYED", "HALT",
        }
        actual = {ps.value for ps in PipelineState}
        assert actual == expected


class TestVerdict:
    def test_three_verdicts(self):
        assert len(Verdict) == 3
        assert Verdict.DEPLOY.value == "DEPLOY"
        assert Verdict.REFINE.value == "REFINE"
        assert Verdict.REJECT.value == "REJECT"


class TestAlertLevel:
    def test_four_levels(self):
        assert len(AlertLevel) == 4
        assert AlertLevel.HALT.value == "HALT"
