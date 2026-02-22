"""
Execution & Risk Agent (EXEC-001) — Risk Manager.

Determines whether validated signals can be profitably and safely
executed in prop firm accounts (Apex Trader Funding, Topstep).

See architecture spec Section 6 for full system prompt.
"""

from __future__ import annotations

from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import ExecutionReport
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope


class ExecutionAgent(BaseAgent):
    """
    EXEC-001: Risk Manager.

    Responsibilities:
    - Transaction cost modeling (slippage, commissions)
    - Turnover analysis (trades/day, holding period, flip rate)
    - Net-of-cost alpha computation
    - Prop firm feasibility (daily loss, trailing DD, consistency, MC ruin)
    - Position sizing (Kelly, half-Kelly)
    - APPROVED/VETOED verdicts
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.EXECUTION, "Execution & Risk", bus)

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle EXECUTION_REQUEST messages from Orchestrator."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.EXECUTION_REQUEST:
            self.send_ack(envelope)
            # TODO: Run cost analysis and prop firm checks
        else:
            self.send_nack(envelope, f"Unexpected message type: {envelope.message_type.value}")

        self.transition_state(AgentState.IDLE)

    def analyze_signals(self, signal_ids: list[str]) -> ExecutionReport:
        """Run full cost and risk analysis on DEPLOY-grade signals."""
        raise NotImplementedError
