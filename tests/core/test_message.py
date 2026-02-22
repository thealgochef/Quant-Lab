"""Tests for MessageEnvelope and MessageBus."""

from alpha_lab.core.enums import AgentID, MessageType, Priority
from alpha_lab.core.message import MessageEnvelope


class TestMessageEnvelope:
    def test_create_with_defaults(self):
        env = MessageEnvelope(
            request_id="req-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
        )
        assert env.priority == Priority.NORMAL
        assert env.message_id is not None
        assert env.timestamp is not None

    def test_roundtrip_json(self, sample_envelope):
        data = sample_envelope.model_dump()
        restored = MessageEnvelope.model_validate(data)
        assert restored.request_id == sample_envelope.request_id
        assert restored.sender == sample_envelope.sender
        assert restored.receiver == sample_envelope.receiver

    def test_priority_levels(self):
        env = MessageEnvelope(
            request_id="req-crit",
            sender=AgentID.MONITORING,
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.ALERT,
            priority=Priority.CRITICAL,
        )
        assert env.priority == Priority.CRITICAL


class TestMessageBus:
    def test_register_and_send(self, message_bus):
        received = []
        message_bus.register_agent(AgentID.DATA_INFRA, lambda env: received.append(env))

        env = MessageEnvelope(
            request_id="req-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
        )
        message_bus.send(env)
        assert len(received) == 1
        assert received[0].request_id == "req-001"

    def test_dedup_drops_duplicate(self, message_bus):
        received = []
        message_bus.register_agent(AgentID.DATA_INFRA, lambda env: received.append(env))

        env = MessageEnvelope(
            request_id="req-dup",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
        )
        message_bus.send(env)
        message_bus.send(env)  # Duplicate — should be dropped
        assert len(received) == 1

    def test_audit_log(self, message_bus):
        message_bus.register_agent(AgentID.DATA_INFRA, lambda env: None)

        env = MessageEnvelope(
            request_id="req-audit",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
        )
        message_bus.send(env)
        log = message_bus.get_audit_log()
        assert len(log) == 1
        assert log[0].request_id == "req-audit"

    def test_unregistered_receiver_does_not_crash(self, message_bus):
        env = MessageEnvelope(
            request_id="req-no-handler",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.SIGNAL_ENG,
            message_type=MessageType.DATA_BUNDLE,
        )
        message_bus.send(env)  # Should not raise
        assert len(message_bus.get_audit_log()) == 1

    def test_clear_dedup_cache(self, message_bus):
        received = []
        message_bus.register_agent(AgentID.DATA_INFRA, lambda env: received.append(env))

        env = MessageEnvelope(
            request_id="req-clear",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
        )
        message_bus.send(env)
        message_bus.clear_dedup_cache()
        message_bus.send(env)  # Should be delivered again after cache clear
        assert len(received) == 2

    def test_get_messages_for(self, message_bus):
        message_bus.register_agent(AgentID.DATA_INFRA, lambda env: None)
        message_bus.register_agent(AgentID.SIGNAL_ENG, lambda env: None)

        env1 = MessageEnvelope(
            request_id="req-a",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
        )
        env2 = MessageEnvelope(
            request_id="req-b",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.SIGNAL_ENG,
            message_type=MessageType.REFINE_REQUEST,
        )
        message_bus.send(env1)
        message_bus.send(env2)

        data_msgs = message_bus.get_messages_for(AgentID.DATA_INFRA)
        assert len(data_msgs) == 1
        assert data_msgs[0].request_id == "req-a"

    def test_reset(self, message_bus):
        message_bus.register_agent(AgentID.DATA_INFRA, lambda env: None)
        message_bus.reset()
        assert len(message_bus.get_audit_log()) == 0
