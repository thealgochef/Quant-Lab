"""
Abstract base class for all agents in the Alpha Signal Research Lab.

Every agent subclasses BaseAgent and gets:
- Identity (AgentID, name)
- State machine (idle -> processing -> idle, or -> error)
- Message send/receive via shared MessageBus
- Convenience methods for ACK, NACK, and escalation
- Structured logging bound to agent identity
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any

from alpha_lab.core.enums import AgentID, AgentState, MessageType, Priority
from alpha_lab.core.message import MessageBus, MessageEnvelope


class BaseAgent(ABC):
    """Abstract base for all 6 agents in the system."""

    def __init__(self, agent_id: AgentID, name: str, bus: MessageBus) -> None:
        self.agent_id = agent_id
        self.name = name
        self.bus = bus
        self.state = AgentState.IDLE
        self.logger = logging.getLogger(f"alpha_lab.agent.{agent_id.value}")
        self.bus.register_agent(self.agent_id, self.handle_message)

    @abstractmethod
    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Process an incoming message. Subclasses MUST implement."""

    def send_message(
        self,
        receiver: AgentID,
        message_type: MessageType,
        payload: dict[str, Any],
        request_id: str | None = None,
        priority: Priority = Priority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Build and send a MessageEnvelope via the bus. Returns the request_id."""
        rid = request_id or str(uuid.uuid4())
        envelope = MessageEnvelope(
            request_id=rid,
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            priority=priority,
            payload=payload,
            metadata=metadata or {"phase": None, "retry_count": 0, "parent_request_id": None},
        )
        self.bus.send(envelope)
        return rid

    def send_ack(self, original: MessageEnvelope, status: str = "received") -> None:
        """Send an ACK for a received message."""
        self.send_message(
            receiver=original.sender,
            message_type=MessageType.ACK,
            payload={
                "original_request_id": original.request_id,
                "status": status,
            },
            request_id=f"ack-{original.request_id}",
        )

    def send_nack(self, original: MessageEnvelope, reason: str) -> None:
        """Send a NACK with specific failure reason."""
        self.send_message(
            receiver=original.sender,
            message_type=MessageType.NACK,
            payload={
                "original_request_id": original.request_id,
                "reason": reason,
            },
            request_id=f"nack-{original.request_id}",
        )

    def escalate(
        self,
        issue: str,
        request_id: str | None = None,
        priority: Priority = Priority.HIGH,
    ) -> None:
        """Escalate an issue to the Orchestrator (Rule 4)."""
        self.logger.warning("Escalating to ORCH-001: %s", issue)
        self.send_message(
            receiver=AgentID.ORCHESTRATOR,
            message_type=MessageType.ALERT,
            payload={"source": self.agent_id.value, "issue": issue},
            request_id=request_id,
            priority=priority,
        )

    def transition_state(self, new_state: AgentState) -> None:
        """Transition agent state with logging."""
        old = self.state
        self.state = new_state
        self.logger.info("State: %s -> %s", old.value, new_state.value)
