"""
Message envelope and in-process message bus.

All inter-agent communication flows through the MessageBus.
Every message uses a typed MessageEnvelope for traceability,
idempotency, and orchestrator visibility.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from alpha_lab.core.enums import AgentID, MessageType, Priority


class MessageEnvelope(BaseModel):
    """Universal typed JSON message envelope per architecture spec Section 9."""

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = Field(description="Unique idempotent key for this request")
    sender: AgentID
    receiver: AgentID
    message_type: MessageType
    priority: Priority = Priority.NORMAL
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {
            "phase": None,
            "retry_count": 0,
            "parent_request_id": None,
        },
    )


MessageHandler = Callable[[MessageEnvelope], None]


class MessageBus:
    """
    In-process message router.

    - Routes messages by receiver AgentID
    - Logs all messages for orchestrator visibility (Rule 3)
    - Deduplicates by request_id + message_type + sender (Rule 5)
    - Supports priority levels
    """

    def __init__(self) -> None:
        self._handlers: dict[AgentID, MessageHandler] = {}
        self._seen: set[str] = set()
        self._audit_log: list[MessageEnvelope] = []
        self._logger = logging.getLogger("alpha_lab.bus")

    def register_agent(self, agent_id: AgentID, handler: MessageHandler) -> None:
        """Register an agent's message handler."""
        if agent_id in self._handlers:
            self._logger.warning("Overwriting handler for %s", agent_id.value)
        self._handlers[agent_id] = handler
        self._logger.info("Registered handler for %s", agent_id.value)

    def unregister_agent(self, agent_id: AgentID) -> None:
        """Remove an agent's handler."""
        self._handlers.pop(agent_id, None)
        self._logger.info("Unregistered handler for %s", agent_id.value)

    def send(self, envelope: MessageEnvelope) -> None:
        """
        Route a message to the receiver agent.

        1. Compute dedup key
        2. If duplicate, silently drop (idempotent handoffs, Rule 5)
        3. Append to audit log (orchestrator visibility, Rule 3)
        4. Dispatch to registered handler
        """
        dedup_key = f"{envelope.request_id}:{envelope.message_type.value}:{envelope.sender.value}"
        if dedup_key in self._seen:
            self._logger.debug("Dropping duplicate message: %s", dedup_key)
            return

        self._seen.add(dedup_key)
        self._audit_log.append(envelope)

        self._logger.info(
            "%s -> %s [%s] priority=%s request_id=%s",
            envelope.sender.value,
            envelope.receiver.value,
            envelope.message_type.value,
            envelope.priority.name,
            envelope.request_id,
        )

        handler = self._handlers.get(envelope.receiver)
        if handler is None:
            self._logger.error(
                "No handler registered for %s, message dropped", envelope.receiver.value
            )
            return

        handler(envelope)

    def get_audit_log(self) -> list[MessageEnvelope]:
        """Return full message audit trail (orchestrator visibility)."""
        return list(self._audit_log)

    def get_messages_for(
        self, agent_id: AgentID, message_type: MessageType | None = None
    ) -> list[MessageEnvelope]:
        """Filter audit log for messages sent to a specific agent."""
        msgs = [m for m in self._audit_log if m.receiver == agent_id]
        if message_type is not None:
            msgs = [m for m in msgs if m.message_type == message_type]
        return msgs

    def clear_dedup_cache(self) -> None:
        """Reset dedup set (e.g., between pipeline runs)."""
        self._seen.clear()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()

    def reset(self) -> None:
        """Full reset: clear handlers, dedup cache, and audit log."""
        self._handlers.clear()
        self._seen.clear()
        self._audit_log.clear()
