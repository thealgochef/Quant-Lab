"""Tests for the Orchestrator agent (ORCH-001)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.orchestrator.agent import OrchestratorAgent
from alpha_lab.agents.orchestrator.pipeline import (
    PHASE_AGENTS,
    PHASE_CRITERIA,
    PHASE_TRANSITIONS,
    PipelineManager,
)
from alpha_lab.core.enums import AgentID, AgentState, MessageType, PipelineState, Priority
from alpha_lab.core.message import MessageEnvelope

# ─── Pipeline Manager Tests ────────────────────────────────────


class TestPipelineManager:
    def test_initial_state(self):
        pm = PipelineManager()
        assert pm.current_state == PipelineState.INIT

    def test_active_agents_per_phase(self):
        assert AgentID.DATA_INFRA in PHASE_AGENTS[PipelineState.INIT]
        assert AgentID.DATA_INFRA in PHASE_AGENTS[PipelineState.PHASE_1_2]
        assert AgentID.SIGNAL_ENG in PHASE_AGENTS[PipelineState.PHASE_1_2]
        assert AgentID.VALIDATION in PHASE_AGENTS[PipelineState.PHASE_3_4]
        assert AgentID.MONITORING in PHASE_AGENTS[PipelineState.DEPLOYED]

    def test_all_phases_have_transitions(self):
        """Every phase except DEPLOYED and HALT has a transition."""
        for state in PipelineState:
            if state not in (PipelineState.DEPLOYED, PipelineState.HALT):
                assert state in PHASE_TRANSITIONS, (
                    f"Missing transition for {state.value}"
                )

    def test_all_phases_have_criteria(self):
        """Every phase with a transition has criteria defined."""
        for state in PHASE_TRANSITIONS:
            assert state in PHASE_CRITERIA, (
                f"Missing criteria for {state.value}"
            )

    def test_transition_description(self):
        pm = PipelineManager()
        assert pm.transition_description is not None
        assert "Data quality" in pm.transition_description

    def test_next_state(self):
        pm = PipelineManager()
        assert pm.next_state == PipelineState.PHASE_1_2

    def test_evaluate_go_no_go_pass(self):
        pm = PipelineManager()
        result = pm.evaluate_go_no_go({"data_quality_passed": (True, {})})
        assert result.passed is True
        assert result.phase == PipelineState.INIT

    def test_evaluate_go_no_go_fail(self):
        pm = PipelineManager()
        result = pm.evaluate_go_no_go({"data_quality_passed": (False, {})})
        assert result.passed is False

    def test_evaluate_go_no_go_missing_criteria(self):
        pm = PipelineManager()
        result = pm.evaluate_go_no_go({})
        assert result.passed is False
        assert "data_quality_passed" in result.details
        assert result.details["data_quality_passed"]["passed"] is False

    def test_evaluate_go_no_go_bool_values(self):
        pm = PipelineManager()
        result = pm.evaluate_go_no_go({"data_quality_passed": True})
        assert result.passed is True

    def test_advance_success(self):
        pm = PipelineManager()
        go = pm.evaluate_go_no_go({"data_quality_passed": True})
        new_state = pm.advance(go)
        assert new_state == PipelineState.PHASE_1_2
        assert pm.current_state == PipelineState.PHASE_1_2
        assert len(pm.history) == 1

    def test_advance_fails_when_not_passed(self):
        pm = PipelineManager()
        go = pm.evaluate_go_no_go({"data_quality_passed": False})
        with pytest.raises(ValueError, match="go/no-go criteria not met"):
            pm.advance(go)

    def test_full_pipeline_progression(self):
        pm = PipelineManager()

        # INIT -> PHASE_1_2
        go = pm.evaluate_go_no_go({"data_quality_passed": True})
        pm.advance(go)
        assert pm.current_state == PipelineState.PHASE_1_2

        # PHASE_1_2 -> PHASE_3_4
        go = pm.evaluate_go_no_go({
            "signals_implemented": True,
            "unit_tests_pass": True,
        })
        pm.advance(go)
        assert pm.current_state == PipelineState.PHASE_3_4

        # PHASE_3_4 -> PHASE_5_6
        go = pm.evaluate_go_no_go({
            "deploy_count": (True, 10),
            "min_ic_tstat": (True, 2.5),
            "min_hit_rate": (True, 0.55),
            "min_sharpe": (True, 1.2),
        })
        pm.advance(go)
        assert pm.current_state == PipelineState.PHASE_5_6

        assert len(pm.history) == 3

    def test_halt(self):
        pm = PipelineManager()
        pm.halt("Test emergency")
        assert pm.current_state == PipelineState.HALT
        assert len(pm.history) == 1
        assert pm.history[0].passed is False

    def test_resume_from_halt(self):
        pm = PipelineManager()
        # Advance to PHASE_1_2, then halt, then resume
        go = pm.evaluate_go_no_go({"data_quality_passed": True})
        pm.advance(go)
        pm.halt("Test halt")
        assert pm.current_state == PipelineState.HALT

        new_state = pm.resume_from_halt()
        assert new_state == PipelineState.PHASE_1_2  # Resumed to last phase

    def test_resume_to_specific_state(self):
        pm = PipelineManager()
        pm.halt("Test halt")
        new_state = pm.resume_from_halt(PipelineState.PHASE_3_4)
        assert new_state == PipelineState.PHASE_3_4

    def test_resume_not_halted_raises(self):
        pm = PipelineManager()
        with pytest.raises(ValueError, match="not in HALT state"):
            pm.resume_from_halt()

    def test_store_and_get_phase_data(self):
        pm = PipelineManager()
        pm.store_phase_data("quality_report", {"passed": True})
        assert pm.get_phase_data("quality_report") == {"passed": True}
        assert pm.get_phase_data("nonexistent") is None

    def test_phase_data_resets_on_advance(self):
        pm = PipelineManager()
        pm.store_phase_data("test_key", "test_value")
        go = pm.evaluate_go_no_go({"data_quality_passed": True})
        pm.advance(go)
        assert pm.get_phase_data("test_key") is None

    def test_no_transition_from_deployed(self):
        pm = PipelineManager()
        assert pm.next_state is not None
        # Fast-forward to DEPLOYED
        pm.current_state = PipelineState.DEPLOYED
        assert pm.next_state is None
        assert pm.transition_description is None


# ─── Orchestrator Agent Tests ──────────────────────────────────


class TestOrchestratorAgent:
    def test_create(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        assert agent.agent_id == AgentID.ORCHESTRATOR
        assert agent.name == "Orchestrator"

    def test_registered_on_bus(self, message_bus):
        OrchestratorAgent(message_bus)
        assert AgentID.ORCHESTRATOR in message_bus._handlers

    def test_pipeline_initial_state(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        assert agent.pipeline.current_state == PipelineState.INIT

    def test_get_pipeline_status(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        status = agent.get_pipeline_status()
        assert status["current_phase"] == "INIT"
        assert "DATA-001" in status["active_agents"]
        assert status["pending_requests"] == 0
        assert status["decisions_made"] == 0

    def test_decision_log_starts_empty(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        assert agent.decision_log == []


class TestOrchestratorRouting:
    """Test message routing through the orchestrator."""

    def test_data_bundle_forwarded_to_sig(self, message_bus):
        """DATA_BUNDLE from DATA-001 should be forwarded to SIG-001."""
        agent = OrchestratorAgent(message_bus)
        received_by_sig = []
        message_bus.register_agent(
            AgentID.SIGNAL_ENG, lambda env: received_by_sig.append(env)
        )

        envelope = MessageEnvelope(
            request_id="test-001",
            sender=AgentID.DATA_INFRA,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.DATA_BUNDLE,
            priority=Priority.NORMAL,
            payload={"bars": {"5m": "test_data"}, "instrument": "NQ"},
        )
        agent.handle_message(envelope)

        assert len(received_by_sig) == 1
        assert received_by_sig[0].message_type == MessageType.DATA_BUNDLE
        assert received_by_sig[0].request_id == "test-001"
        assert len(agent.decision_log) >= 1

    def test_signal_bundle_forwarded_to_val(self, message_bus):
        """SIGNAL_BUNDLE from SIG-001 should be forwarded to VAL-001."""
        agent = OrchestratorAgent(message_bus)
        received_by_val = []
        message_bus.register_agent(
            AgentID.VALIDATION, lambda env: received_by_val.append(env)
        )

        # First store data bundle so ORCH has price data
        agent._pending_requests["test-002"] = {
            "data_bundle": {"bars": {"5m": "price_data"}},
            "stage": "signal_generation",
        }

        envelope = MessageEnvelope(
            request_id="test-002",
            sender=AgentID.SIGNAL_ENG,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.SIGNAL_BUNDLE,
            priority=Priority.NORMAL,
            payload={"bundle": {"signals": []}},
        )
        agent.handle_message(envelope)

        assert len(received_by_val) == 1
        assert received_by_val[0].message_type == MessageType.SIGNAL_BUNDLE
        # Should include price data
        assert "price_data" in received_by_val[0].payload

    def test_validation_report_deploy_to_exec(self, message_bus):
        """VALIDATION_REPORT with DEPLOY signals -> EXECUTION_REQUEST."""
        agent = OrchestratorAgent(message_bus)
        received_by_exec = []
        message_bus.register_agent(
            AgentID.EXECUTION, lambda env: received_by_exec.append(env)
        )

        agent._pending_requests["test-003"] = {
            "data_bundle": {"bars": {}},
            "signal_bundle": {},
        }

        envelope = MessageEnvelope(
            request_id="test-003",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.VALIDATION_REPORT,
            priority=Priority.NORMAL,
            payload={
                "report": {
                    "deploy_count": 2,
                    "refine_count": 0,
                    "reject_count": 1,
                    "verdicts": [
                        {"signal_id": "SIG_1", "verdict": "DEPLOY"},
                        {"signal_id": "SIG_2", "verdict": "DEPLOY"},
                        {"signal_id": "SIG_3", "verdict": "REJECT"},
                    ],
                }
            },
        )
        agent.handle_message(envelope)

        exec_msgs = [
            e for e in received_by_exec
            if e.message_type == MessageType.EXECUTION_REQUEST
        ]
        assert len(exec_msgs) == 1
        assert "validation_report" in exec_msgs[0].payload

    def test_validation_report_refine_to_sig(self, message_bus):
        """VALIDATION_REPORT with REFINE signals -> REFINE_REQUEST to SIG-001."""
        agent = OrchestratorAgent(message_bus)
        received_by_sig = []
        message_bus.register_agent(
            AgentID.SIGNAL_ENG, lambda env: received_by_sig.append(env)
        )

        envelope = MessageEnvelope(
            request_id="test-004",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.VALIDATION_REPORT,
            priority=Priority.NORMAL,
            payload={
                "report": {
                    "deploy_count": 0,
                    "refine_count": 2,
                    "reject_count": 0,
                    "verdicts": [
                        {
                            "signal_id": "SIG_A",
                            "verdict": "REFINE",
                            "failed_metrics": [],
                        },
                        {
                            "signal_id": "SIG_B",
                            "verdict": "REFINE",
                            "failed_metrics": [],
                        },
                    ],
                }
            },
        )
        agent.handle_message(envelope)

        refine_msgs = [
            e for e in received_by_sig
            if e.message_type == MessageType.REFINE_REQUEST
        ]
        assert len(refine_msgs) == 1
        assert refine_msgs[0].payload["signal_ids"] == ["SIG_A", "SIG_B"]

    def test_validation_all_reject_logs_stall(self, message_bus):
        """All REJECT should log stall, not forward anywhere."""
        agent = OrchestratorAgent(message_bus)
        received_by_exec = []
        received_by_sig = []
        message_bus.register_agent(
            AgentID.EXECUTION, lambda env: received_by_exec.append(env)
        )
        message_bus.register_agent(
            AgentID.SIGNAL_ENG, lambda env: received_by_sig.append(env)
        )

        envelope = MessageEnvelope(
            request_id="test-005",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.VALIDATION_REPORT,
            priority=Priority.NORMAL,
            payload={
                "report": {
                    "deploy_count": 0,
                    "refine_count": 0,
                    "reject_count": 3,
                    "verdicts": [],
                }
            },
        )
        agent.handle_message(envelope)

        # No messages to EXEC or SIG
        exec_msgs = [
            e for e in received_by_exec
            if e.message_type == MessageType.EXECUTION_REQUEST
        ]
        refine_msgs = [
            e for e in received_by_sig
            if e.message_type == MessageType.REFINE_REQUEST
        ]
        assert len(exec_msgs) == 0
        assert len(refine_msgs) == 0

        # Should have "all_rejected" in decision log
        actions = [d["action"] for d in agent.decision_log]
        assert "all_rejected" in actions

    def test_execution_report_logged(self, message_bus):
        """EXECUTION_REPORT from EXEC-001 should be logged."""
        agent = OrchestratorAgent(message_bus)

        envelope = MessageEnvelope(
            request_id="test-006",
            sender=AgentID.EXECUTION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.EXECUTION_REPORT,
            priority=Priority.NORMAL,
            payload={
                "report": {
                    "approved_signals": [
                        {"signal_id": "SIG_1", "verdict": "APPROVED"},
                    ],
                    "vetoed_signals": [],
                }
            },
        )
        agent.handle_message(envelope)

        actions = [d["action"] for d in agent.decision_log]
        assert "execution_result" in actions
        assert "deploy_ready" in actions

    def test_halt_command(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        envelope = MessageEnvelope(
            request_id="halt-001",
            sender=AgentID.MONITORING,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.HALT_COMMAND,
            priority=Priority.CRITICAL,
            payload={"reason": "Trailing DD exceeded"},
        )
        agent.handle_message(envelope)
        assert agent.pipeline.current_state == PipelineState.HALT

    def test_resume_command(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        agent.pipeline.halt("Test halt")

        envelope = MessageEnvelope(
            request_id="resume-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.RESUME_COMMAND,
            priority=Priority.NORMAL,
            payload={},
        )
        agent.handle_message(envelope)
        assert agent.pipeline.current_state != PipelineState.HALT

    def test_nack_logged(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        envelope = MessageEnvelope(
            request_id="nack-001",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.NACK,
            priority=Priority.NORMAL,
            payload={"reason": "Invalid bundle format"},
        )
        agent.handle_message(envelope)

        actions = [d["action"] for d in agent.decision_log]
        assert "nack_received" in actions

    def test_unknown_message_type_no_crash(self, message_bus):
        """Unhandled message types should be logged, not crash."""
        agent = OrchestratorAgent(message_bus)
        envelope = MessageEnvelope(
            request_id="test-unknown",
            sender=AgentID.DATA_INFRA,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.DAILY_REPORT,
            priority=Priority.NORMAL,
            payload={},
        )
        agent.handle_message(envelope)
        # Should not crash, agent returns to IDLE
        assert agent.state == AgentState.IDLE


class TestOrchestratorPipeline:
    """Test pipeline control via the orchestrator."""

    def test_advance_phase(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        result = agent.advance_phase({"data_quality_passed": True})
        assert result is True
        assert agent.pipeline.current_state == PipelineState.PHASE_1_2

    def test_advance_phase_blocked(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        result = agent.advance_phase({"data_quality_passed": False})
        assert result is False
        assert agent.pipeline.current_state == PipelineState.INIT

    def test_run_pipeline(self, message_bus):
        """run_pipeline sends DATA_REQUEST to DATA-001."""
        agent = OrchestratorAgent(message_bus)
        received = []
        message_bus.register_agent(
            AgentID.DATA_INFRA, lambda env: received.append(env)
        )

        rid = agent.run_pipeline(
            {"instrument": "NQ", "date_range": ("2026-01-01", "2026-02-01")}
        )
        assert rid is not None
        assert len(received) == 1
        assert received[0].message_type == MessageType.DATA_REQUEST

    def test_conflict_sig_vs_val(self, message_bus):
        """SIG vs VAL: VAL wins."""
        agent = OrchestratorAgent(message_bus)
        result = agent.handle_conflict(
            AgentID.SIGNAL_ENG, AgentID.VALIDATION,
            "Signal shows edge, VAL says noise",
        )
        assert "VAL-001" in result

    def test_conflict_exec_vs_val(self, message_bus):
        """EXEC vs VAL: EXEC wins."""
        agent = OrchestratorAgent(message_bus)
        result = agent.handle_conflict(
            AgentID.EXECUTION, AgentID.VALIDATION,
            "Costs destroy alpha",
        )
        assert "EXEC-001" in result

    def test_conflict_with_monitoring(self, message_bus):
        """MON involved: pause for consensus."""
        agent = OrchestratorAgent(message_bus)
        result = agent.handle_conflict(
            AgentID.MONITORING, AgentID.SIGNAL_ENG,
            "Regime shift detected",
        )
        assert "PAUSE" in result

    def test_conflict_unknown(self, message_bus):
        """Unknown conflict: ORCH decides."""
        agent = OrchestratorAgent(message_bus)
        result = agent.handle_conflict(
            AgentID.DATA_INFRA, AgentID.SIGNAL_ENG,
            "Data format dispute",
        )
        assert "ORCH-001" in result


class TestEndToEndFlow:
    """Integration test: full message flow through orchestrator."""

    def test_data_to_signal_to_validation_flow(self, message_bus):
        """Test the DATA -> SIG -> VAL routing chain."""
        agent = OrchestratorAgent(message_bus)

        sig_received = []
        val_received = []
        exec_received = []

        message_bus.register_agent(
            AgentID.SIGNAL_ENG, lambda env: sig_received.append(env)
        )
        message_bus.register_agent(
            AgentID.VALIDATION, lambda env: val_received.append(env)
        )
        message_bus.register_agent(
            AgentID.EXECUTION, lambda env: exec_received.append(env)
        )

        # Step 1: DATA-001 sends DATA_BUNDLE to ORCH
        data_env = MessageEnvelope(
            request_id="e2e-001",
            sender=AgentID.DATA_INFRA,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.DATA_BUNDLE,
            priority=Priority.NORMAL,
            payload={"bars": {"5m": "price_data"}, "instrument": "NQ"},
        )
        agent.handle_message(data_env)
        assert len(sig_received) == 1

        # Step 2: SIG-001 sends SIGNAL_BUNDLE to ORCH
        sig_env = MessageEnvelope(
            request_id="e2e-001",
            sender=AgentID.SIGNAL_ENG,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.SIGNAL_BUNDLE,
            priority=Priority.NORMAL,
            payload={"bundle": {"signals": ["sig1"]}},
        )
        agent.handle_message(sig_env)
        assert len(val_received) == 1

        # Step 3: VAL-001 sends VALIDATION_REPORT to ORCH
        val_env = MessageEnvelope(
            request_id="e2e-001",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.VALIDATION_REPORT,
            priority=Priority.NORMAL,
            payload={
                "report": {
                    "deploy_count": 1,
                    "refine_count": 0,
                    "reject_count": 0,
                    "verdicts": [
                        {"signal_id": "SIG_1", "verdict": "DEPLOY"},
                    ],
                }
            },
        )
        agent.handle_message(val_env)

        # Should have forwarded to EXEC
        exec_msgs = [
            e for e in exec_received
            if e.message_type == MessageType.EXECUTION_REQUEST
        ]
        assert len(exec_msgs) == 1

        # Check decision log covers the full flow
        actions = [d["action"] for d in agent.decision_log]
        assert "forward_data_to_sig" in actions
        assert "forward_signals_to_val" in actions
        assert "validation_triage" in actions
