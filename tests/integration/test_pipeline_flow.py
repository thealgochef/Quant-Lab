"""Integration tests for end-to-end message flow through all agents."""

from alpha_lab.agents.data_infra.agent import DataInfraAgent
from alpha_lab.agents.execution.agent import ExecutionAgent
from alpha_lab.agents.monitoring.agent import MonitoringAgent
from alpha_lab.agents.orchestrator.agent import OrchestratorAgent
from alpha_lab.agents.signal_eng.agent import SignalEngineeringAgent
from alpha_lab.agents.validation.agent import ValidationAgent
from alpha_lab.core.enums import AgentID, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope


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
        # If we got here without error, the message was delivered and handled
        log = bus.get_audit_log()
        assert len(log) >= 1
        assert log[0].request_id == "pipeline-001"
