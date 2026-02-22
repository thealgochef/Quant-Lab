"""
Live Monitoring Agent (MON-001) — Production Ops.

Tracks every deployed signal in real-time and detects degradation
before it impacts P&L. Early warning system for the entire lab.

See architecture spec Section 7 for full system prompt.
"""

from __future__ import annotations

from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import MonitoringReport
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope


class MonitoringAgent(BaseAgent):
    """
    MON-001: Production Ops.

    Responsibilities:
    - Continuous monitoring of deployed signal performance
    - Rolling IC, hit rate, Sharpe tracking
    - Alert generation (INFO/WARNING/CRITICAL/HALT)
    - Regime classification and transition detection
    - Prop firm buffer tracking (DD and daily limit headroom)
    - Daily report generation (mandatory, even on no-trade days)
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.MONITORING, "Live Monitoring", bus)

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle DEPLOY_COMMAND and RESUME_COMMAND messages."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type in (MessageType.DEPLOY_COMMAND, MessageType.RESUME_COMMAND):
            self.send_ack(envelope)
            # TODO: Resume monitoring with confirmed signals and risk params
        else:
            self.send_nack(envelope, f"Unexpected message type: {envelope.message_type.value}")

        self.transition_state(AgentState.IDLE)

    def generate_daily_report(self) -> MonitoringReport:
        """Generate end-of-session daily report (mandatory)."""
        raise NotImplementedError

    def check_signal_health(self, signal_id: str) -> dict:
        """Check current health metrics for a deployed signal."""
        raise NotImplementedError
