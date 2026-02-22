"""
Statistical Validation Agent (VAL-001) — Quant Reviewer.

The firewall against overfitting. Receives signal vectors as OPAQUE
NUMERICAL ARRAYS. Does not know how they were constructed.

See architecture spec Section 5 for full system prompt.
"""

from __future__ import annotations

from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import SignalBundle, ValidationReport
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope


class ValidationAgent(BaseAgent):
    """
    VAL-001: Quant Reviewer.

    Responsibilities:
    - Run full test battery on every signal (IC, hit rate, Sharpe, decay, orthogonality, robustness)
    - Issue DEPLOY/REFINE/REJECT verdicts per signal
    - Apply Bonferroni correction for multiple testing
    - Flag suspected look-ahead bias
    - Never share test methodology with SIG-001 (firewall)
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.VALIDATION, "Statistical Validation", bus)

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle SIGNAL_BUNDLE messages (opaque vectors only)."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.SIGNAL_BUNDLE:
            self.send_ack(envelope)
            # TODO: Run test battery, produce ValidationReport
        else:
            self.send_nack(envelope, f"Unexpected message type: {envelope.message_type.value}")

        self.transition_state(AgentState.IDLE)

    def validate_signal_bundle(self, bundle: SignalBundle) -> ValidationReport:
        """
        Run the full test battery on a SignalBundle.

        Tests: IC, hit rate, risk-adjusted returns, decay analysis,
        factor orthogonality, robustness checks.
        """
        raise NotImplementedError
