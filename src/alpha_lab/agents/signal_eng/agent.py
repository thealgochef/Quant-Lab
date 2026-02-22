"""
Signal Engineering Agent (SIG-001) — Alpha Researcher.

Builds, tests, and iterates on trading signal detectors for NQ/ES futures.
Takes clean DataBundle objects from DATA-001 and produces SignalBundle objects.

See architecture spec Section 4 for full system prompt.
"""

from __future__ import annotations

from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import DataBundle, SignalBundle
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope


class SignalEngineeringAgent(BaseAgent):
    """
    SIG-001: Alpha Researcher.

    Responsibilities:
    - Implement all 20 signal category detectors
    - Compute signals across ALL supported timeframes
    - Normalize signals to [-1, +1] range
    - Handle REFINE feedback (max 3 iterations per signal)
    - Build composite signals from individual detectors
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.SIGNAL_ENG, "Signal Engineering", bus)

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle DATA_BUNDLE and REFINE_REQUEST messages."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type in (MessageType.DATA_BUNDLE, MessageType.REFINE_REQUEST):
            self.send_ack(envelope)
            # TODO: Refine specific signal based on failed metrics
        else:
            self.send_nack(envelope, f"Unexpected message type: {envelope.message_type.value}")

        self.transition_state(AgentState.IDLE)

    def generate_signals(self, data_bundle: DataBundle) -> SignalBundle:
        """
        Run all registered signal detectors on the DataBundle.

        Returns a SignalBundle containing all signal vectors.
        """
        raise NotImplementedError

    def refine_signal(
        self, signal_id: str, failed_metric: str, failed_value: float, threshold: float
    ) -> SignalBundle:
        """
        Refine a specific signal based on VAL-001 feedback.

        The feedback is intentionally limited (firewall rule):
        only metric name, value, and threshold are provided.
        """
        raise NotImplementedError
