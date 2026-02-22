"""Tests for BaseAgent lifecycle and convenience methods."""

from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.enums import AgentID, AgentState, MessageType, Priority
from alpha_lab.core.message import MessageBus, MessageEnvelope


class ConcreteAgent(BaseAgent):
    """Minimal concrete agent for testing BaseAgent functionality."""

    def __init__(self, bus: MessageBus):
        super().__init__(AgentID.DATA_INFRA, "Test Agent", bus)
        self.received_messages: list[MessageEnvelope] = []

    def handle_message(self, envelope: MessageEnvelope) -> None:
        self.received_messages.append(envelope)


class TestBaseAgent:
    def test_initial_state_is_idle(self, message_bus):
        agent = ConcreteAgent(message_bus)
        assert agent.state == AgentState.IDLE

    def test_auto_registers_with_bus(self, message_bus):
        agent = ConcreteAgent(message_bus)
        env = MessageEnvelope(
            request_id="test-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
        )
        message_bus.send(env)
        assert len(agent.received_messages) == 1

    def test_transition_state(self, message_bus):
        agent = ConcreteAgent(message_bus)
        agent.transition_state(AgentState.PROCESSING)
        assert agent.state == AgentState.PROCESSING
        agent.transition_state(AgentState.IDLE)
        assert agent.state == AgentState.IDLE

    def test_send_message(self, message_bus):
        agent = ConcreteAgent(message_bus)
        received = []
        message_bus.register_agent(AgentID.ORCHESTRATOR, lambda env: received.append(env))

        agent.send_message(
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.ACK,
            payload={"status": "ok"},
            request_id="ack-001",
        )
        assert len(received) == 1
        assert received[0].payload["status"] == "ok"

    def test_send_ack(self, message_bus, sample_envelope):
        agent = ConcreteAgent(message_bus)
        ack_received = []
        message_bus.register_agent(AgentID.ORCHESTRATOR, lambda env: ack_received.append(env))

        agent.send_ack(sample_envelope)
        assert len(ack_received) == 1
        assert ack_received[0].message_type == MessageType.ACK
        assert ack_received[0].payload["original_request_id"] == sample_envelope.request_id

    def test_send_nack(self, message_bus, sample_envelope):
        agent = ConcreteAgent(message_bus)
        nack_received = []
        message_bus.register_agent(AgentID.ORCHESTRATOR, lambda env: nack_received.append(env))

        agent.send_nack(sample_envelope, "bad schema")
        assert len(nack_received) == 1
        assert nack_received[0].message_type == MessageType.NACK
        assert nack_received[0].payload["reason"] == "bad schema"

    def test_escalate(self, message_bus):
        agent = ConcreteAgent(message_bus)
        escalations = []
        message_bus.register_agent(AgentID.ORCHESTRATOR, lambda env: escalations.append(env))

        agent.escalate("data gap detected")
        assert len(escalations) == 1
        assert escalations[0].priority == Priority.HIGH
        assert escalations[0].payload["issue"] == "data gap detected"
