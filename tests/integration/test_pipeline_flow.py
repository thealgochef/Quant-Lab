"""Integration tests for end-to-end message flow through all agents."""

from __future__ import annotations

from alpha_lab.agents.data_infra.agent import DataInfraAgent
from alpha_lab.agents.execution.agent import ExecutionAgent
from alpha_lab.agents.monitoring.agent import MonitoringAgent
from alpha_lab.agents.orchestrator.agent import OrchestratorAgent
from alpha_lab.agents.signal_eng.agent import SignalEngineeringAgent
from alpha_lab.agents.validation.agent import ValidationAgent
from alpha_lab.core.contracts import SignalVerdict, ValidationReport
from alpha_lab.core.enums import AgentID, MessageType, PipelineState
from alpha_lab.core.message import MessageBus, MessageEnvelope

# ═══════════════════════════════════════════════════════════════
#  AGENT REGISTRATION
# ═══════════════════════════════════════════════════════════════


class TestAllAgentsOnBus:
    def test_all_six_agents_register(self):
        bus = MessageBus()
        OrchestratorAgent(bus)
        DataInfraAgent(bus)
        SignalEngineeringAgent(bus)
        ValidationAgent(bus)
        ExecutionAgent(bus)
        MonitoringAgent(bus)

        assert AgentID.ORCHESTRATOR in bus._handlers
        assert AgentID.DATA_INFRA in bus._handlers
        assert AgentID.SIGNAL_ENG in bus._handlers
        assert AgentID.VALIDATION in bus._handlers
        assert AgentID.EXECUTION in bus._handlers
        assert AgentID.MONITORING in bus._handlers

    def test_orch_to_data_message_delivered(self):
        bus = MessageBus()
        OrchestratorAgent(bus)
        DataInfraAgent(bus)

        env = MessageEnvelope(
            request_id="pipeline-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
            payload={"instrument": "NQ", "date_range": ("2026-01-01", "2026-02-21")},
        )
        bus.send(env)
        log = bus.get_audit_log()
        assert len(log) >= 1
        assert log[0].request_id == "pipeline-001"


# ═══════════════════════════════════════════════════════════════
#  FULL PIPELINE (direct calls)
# ═══════════════════════════════════════════════════════════════


class TestFullPipeline:
    def test_sig_generates_signals_from_bundle(
        self, message_bus, synthetic_data_bundle
    ):
        """SIG-001 produces signals from a real DataBundle."""
        sig = SignalEngineeringAgent(message_bus)
        bundle = sig.generate_signals(synthetic_data_bundle)
        assert bundle.total_signals == 3
        assert "5m" in bundle.timeframes_covered
        for sv in bundle.signals:
            assert len(sv.direction) == len(synthetic_data_bundle.bars["5m"])

    def test_val_validates_signal_bundle(
        self, message_bus, synthetic_data_bundle
    ):
        """VAL-001 evaluates all signals and returns verdicts."""
        sig = SignalEngineeringAgent(message_bus)
        signal_bundle = sig.generate_signals(synthetic_data_bundle)
        val = ValidationAgent(message_bus)
        price_data = {"5m": synthetic_data_bundle.bars["5m"]}

        report = val.validate_signal_bundle(signal_bundle, price_data)
        assert len(report.verdicts) == signal_bundle.total_signals
        assert report.deploy_count + report.refine_count + report.reject_count == 3
        for v in report.verdicts:
            assert v.verdict in ("DEPLOY", "REFINE", "REJECT")

    def test_exec_analyzes_deploy_verdicts(
        self, message_bus, synthetic_data_bundle, nq_instrument, apex_50k_profile
    ):
        """EXEC-001 processes DEPLOY verdicts with cost/risk analysis."""
        sig = SignalEngineeringAgent(message_bus)
        val = ValidationAgent(message_bus)
        exec_agent = ExecutionAgent(
            message_bus, instrument=nq_instrument, prop_firm=apex_50k_profile
        )

        signal_bundle = sig.generate_signals(synthetic_data_bundle)
        price_data = {"5m": synthetic_data_bundle.bars["5m"]}
        val_report = val.validate_signal_bundle(signal_bundle, price_data)

        # Synthetic data typically gets all REJECT; create a DEPLOY override
        if val_report.deploy_count == 0 and val_report.verdicts:
            v = val_report.verdicts[0]
            val_report = ValidationReport(
                request_id=val_report.request_id,
                signal_bundle_id=val_report.signal_bundle_id,
                verdicts=[
                    SignalVerdict(
                        **{**v.model_dump(), "verdict": "DEPLOY", "max_factor_corr": 0.1}
                    )
                ],
                deploy_count=1,
                refine_count=0,
                reject_count=0,
                bonferroni_adjusted=False,
                overall_assessment="1 DEPLOY (test override)",
                timestamp=val_report.timestamp,
            )

        exec_report = exec_agent.analyze_signals(val_report, price_data)
        total = len(exec_report.approved_signals) + len(exec_report.vetoed_signals)
        assert total == val_report.deploy_count
        assert "total_signals" in exec_report.portfolio_risk

    def test_full_data_to_exec_flow(
        self, message_bus, synthetic_data_bundle, nq_instrument, apex_50k_profile
    ):
        """End-to-end: generate -> validate -> execute with valid contracts."""
        sig = SignalEngineeringAgent(message_bus)
        val = ValidationAgent(message_bus)
        exec_agent = ExecutionAgent(
            message_bus, instrument=nq_instrument, prop_firm=apex_50k_profile
        )

        # Step 1: Generate signals
        signal_bundle = sig.generate_signals(synthetic_data_bundle)
        assert signal_bundle.total_signals > 0

        # Step 2: Validate
        price_data = {"5m": synthetic_data_bundle.bars["5m"]}
        val_report = val.validate_signal_bundle(signal_bundle, price_data)
        assert len(val_report.verdicts) == signal_bundle.total_signals

        # Step 3: Execute (with DEPLOY override if needed)
        if val_report.deploy_count == 0 and val_report.verdicts:
            v = val_report.verdicts[0]
            val_report = ValidationReport(
                request_id=val_report.request_id,
                signal_bundle_id=val_report.signal_bundle_id,
                verdicts=[
                    SignalVerdict(
                        **{**v.model_dump(), "verdict": "DEPLOY", "max_factor_corr": 0.1}
                    )
                ],
                deploy_count=1,
                refine_count=0,
                reject_count=0,
                bonferroni_adjusted=False,
                overall_assessment="test",
                timestamp=val_report.timestamp,
            )

        exec_report = exec_agent.analyze_signals(val_report, price_data)
        assert exec_report.request_id is not None
        assert exec_report.timestamp is not None


# ═══════════════════════════════════════════════════════════════
#  HANDOFF PROTOCOLS (message-bus-driven)
# ═══════════════════════════════════════════════════════════════


class TestHandoffProtocols:
    def test_h002_orch_forwards_data_to_sig(self, full_pipeline_bus):
        """H-002: DATA_BUNDLE to ORCH -> forwarded to SIG-001."""
        bus, agents = full_pipeline_bus

        env = MessageEnvelope(
            request_id="h002-test",
            sender=AgentID.DATA_INFRA,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.DATA_BUNDLE,
            payload={"bars": {}, "quality": {"passed": True}},
        )
        bus.send(env)

        audit = bus.get_audit_log()
        sig_msgs = [
            e for e in audit
            if e.receiver == AgentID.SIGNAL_ENG
            and e.message_type == MessageType.DATA_BUNDLE
        ]
        assert len(sig_msgs) >= 1
        assert sig_msgs[0].request_id == "h002-test"

    def test_h003_orch_forwards_signals_to_val(self, full_pipeline_bus):
        """H-003: SIGNAL_BUNDLE to ORCH -> forwarded to VAL-001 with price data."""
        bus, agents = full_pipeline_bus

        # First store a data bundle so ORCH has price data
        agents["orch"]._pending_requests["h003-test"] = {
            "data_bundle": {"bars": {"5m": "mock_data"}},
            "stage": "signal_generation",
        }

        env = MessageEnvelope(
            request_id="h003-test",
            sender=AgentID.SIGNAL_ENG,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.SIGNAL_BUNDLE,
            payload={"bundle": {"signals": []}, "signal_data": {}},
        )
        bus.send(env)

        audit = bus.get_audit_log()
        val_msgs = [
            e for e in audit
            if e.receiver == AgentID.VALIDATION
            and e.message_type == MessageType.SIGNAL_BUNDLE
        ]
        assert len(val_msgs) >= 1
        # Verify price_data was attached
        assert "price_data" in val_msgs[0].payload

    def test_h005_orch_routes_deploy_to_exec(self, full_pipeline_bus):
        """H-005: VALIDATION_REPORT with DEPLOY -> EXEC-001 gets EXECUTION_REQUEST."""
        bus, agents = full_pipeline_bus

        agents["orch"]._pending_requests["h005-test"] = {
            "data_bundle": {"bars": {}},
            "signal_bundle": {"signal_data": {}},
            "stage": "validation",
        }

        env = MessageEnvelope(
            request_id="h005-test",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.VALIDATION_REPORT,
            payload={
                "report": {
                    "deploy_count": 1,
                    "refine_count": 0,
                    "reject_count": 0,
                    "verdicts": [{"signal_id": "SIG_TEST", "verdict": "DEPLOY"}],
                }
            },
        )
        bus.send(env)

        audit = bus.get_audit_log()
        exec_msgs = [
            e for e in audit
            if e.receiver == AgentID.EXECUTION
            and e.message_type == MessageType.EXECUTION_REQUEST
        ]
        assert len(exec_msgs) >= 1

    def test_h006_orch_routes_refine_to_sig(self, full_pipeline_bus):
        """H-006: VALIDATION_REPORT with REFINE -> SIG-001 gets REFINE_REQUEST."""
        bus, agents = full_pipeline_bus

        env = MessageEnvelope(
            request_id="h006-test",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.VALIDATION_REPORT,
            payload={
                "report": {
                    "deploy_count": 0,
                    "refine_count": 1,
                    "reject_count": 0,
                    "verdicts": [{"signal_id": "SIG_TEST", "verdict": "REFINE"}],
                }
            },
        )
        bus.send(env)

        audit = bus.get_audit_log()
        refine_msgs = [
            e for e in audit
            if e.receiver == AgentID.SIGNAL_ENG
            and e.message_type == MessageType.REFINE_REQUEST
        ]
        assert len(refine_msgs) >= 1
        assert "SIG_TEST" in refine_msgs[0].payload.get("signal_ids", [])

    def test_all_reject_no_exec_message(self, full_pipeline_bus):
        """All REJECT -> no EXECUTION_REQUEST sent to EXEC-001."""
        bus, agents = full_pipeline_bus
        initial_audit_len = len(bus.get_audit_log())

        env = MessageEnvelope(
            request_id="reject-test",
            sender=AgentID.VALIDATION,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.VALIDATION_REPORT,
            payload={
                "report": {
                    "deploy_count": 0,
                    "refine_count": 0,
                    "reject_count": 3,
                    "verdicts": [],
                }
            },
        )
        bus.send(env)

        audit = bus.get_audit_log()
        new_msgs = audit[initial_audit_len:]
        exec_msgs = [
            e for e in new_msgs
            if e.receiver == AgentID.EXECUTION
            and e.message_type == MessageType.EXECUTION_REQUEST
        ]
        assert len(exec_msgs) == 0


# ═══════════════════════════════════════════════════════════════
#  PIPELINE CONTROL
# ═══════════════════════════════════════════════════════════════


class TestPipelineControl:
    def test_halt_stops_pipeline(self, full_pipeline_bus):
        """HALT_COMMAND transitions pipeline to HALT state."""
        bus, agents = full_pipeline_bus
        orch = agents["orch"]

        env = MessageEnvelope(
            request_id="halt-test",
            sender=AgentID.MONITORING,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.HALT_COMMAND,
            payload={"reason": "Integration test halt"},
        )
        bus.send(env)

        assert orch.pipeline.current_state == PipelineState.HALT

    def test_resume_from_halt(self, full_pipeline_bus):
        """HALT then RESUME restores pipeline state."""
        bus, agents = full_pipeline_bus
        orch = agents["orch"]

        # Halt first
        orch.pipeline.halt("test halt")
        assert orch.pipeline.current_state == PipelineState.HALT

        # Resume
        env = MessageEnvelope(
            request_id="resume-test",
            sender=AgentID.MONITORING,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.RESUME_COMMAND,
            payload={"target_state": "INIT"},
        )
        bus.send(env)
        assert orch.pipeline.current_state == PipelineState.INIT

    def test_conflict_resolution(self, full_pipeline_bus):
        """SIG vs VAL -> VAL wins; EXEC vs VAL -> EXEC wins."""
        _, agents = full_pipeline_bus
        orch = agents["orch"]

        result1 = orch.handle_conflict(
            AgentID.SIGNAL_ENG, AgentID.VALIDATION, "IC threshold dispute"
        )
        assert "VAL-001" in result1

        result2 = orch.handle_conflict(
            AgentID.EXECUTION, AgentID.VALIDATION, "Cost model disagreement"
        )
        assert "EXEC-001" in result2
