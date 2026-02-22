"""
Orchestrator Agent (ORCH-001) — Research Director.

Sequences phases, manages handoffs, resolves conflicts, and makes go/no-go decisions.
Holds the thesis: 'Discover multi-timeframe confluence signals that survive
prop firm transaction costs and risk constraints.'

See architecture spec Section 2 for full system prompt and decision framework.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from alpha_lab.agents.orchestrator.pipeline import PipelineManager
from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope

logger = logging.getLogger(__name__)

# Conflict resolution: which agent wins in a dispute
_CONFLICT_RESOLUTION: dict[tuple[AgentID, AgentID], AgentID] = {
    (AgentID.SIGNAL_ENG, AgentID.VALIDATION): AgentID.VALIDATION,
    (AgentID.VALIDATION, AgentID.SIGNAL_ENG): AgentID.VALIDATION,
    (AgentID.EXECUTION, AgentID.VALIDATION): AgentID.EXECUTION,
    (AgentID.VALIDATION, AgentID.EXECUTION): AgentID.EXECUTION,
}


class OrchestratorAgent(BaseAgent):
    """
    ORCH-001: The Research Director.

    Responsibilities:
    - Sequence work across 5 specialist agents
    - Manage handoff protocols (H-001 through H-012)
    - Resolve conflicts per decision framework
    - Make go/no-go decisions at phase boundaries
    - Log every decision with rationale for audit trail
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.ORCHESTRATOR, "Orchestrator", bus)
        self.pipeline = PipelineManager()
        self._decision_log: list[dict[str, Any]] = []
        self._pending_requests: dict[str, dict[str, Any]] = {}

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Route incoming messages based on type and current pipeline state."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        handler = self._get_handler(envelope.message_type)
        if handler is not None:
            handler(envelope)
        else:
            self.logger.warning(
                "No handler for %s in state %s",
                envelope.message_type.value,
                self.pipeline.current_state.value,
            )

        self.transition_state(AgentState.IDLE)

    def _get_handler(self, msg_type: MessageType):
        """Look up the handler for a message type."""
        handlers = {
            MessageType.DATA_BUNDLE: self._on_data_bundle,
            MessageType.SIGNAL_BUNDLE: self._on_signal_bundle,
            MessageType.VALIDATION_REPORT: self._on_validation_report,
            MessageType.EXECUTION_REPORT: self._on_execution_report,
            MessageType.ALERT: self._on_alert,
            MessageType.HALT_COMMAND: self._on_halt,
            MessageType.RESUME_COMMAND: self._on_resume,
            MessageType.ACK: self._on_ack,
            MessageType.NACK: self._on_nack,
        }
        return handlers.get(msg_type)

    # ─── Handoff Handlers ──────────────────────────────────────

    def _on_data_bundle(self, envelope: MessageEnvelope) -> None:
        """H-002: DATA-001 -> ORCH-001 -> SIG-001.

        Receives validated data, forwards to signal engineering.
        """
        self._log_decision(
            "forward_data_to_sig",
            f"Forwarding DATA_BUNDLE to SIG-001 (request_id={envelope.request_id})",
        )

        # Store the data bundle for later use (validation needs price data)
        self._pending_requests[envelope.request_id] = {
            "data_bundle": envelope.payload,
            "stage": "signal_generation",
            "started_at": datetime.now(UTC).isoformat(),
        }

        self.send_message(
            receiver=AgentID.SIGNAL_ENG,
            message_type=MessageType.DATA_BUNDLE,
            payload=envelope.payload,
            request_id=envelope.request_id,
        )

    def _on_signal_bundle(self, envelope: MessageEnvelope) -> None:
        """H-003: SIG-001 -> ORCH-001 -> VAL-001.

        Receives signal bundle, forwards to validation with price data.
        """
        self._log_decision(
            "forward_signals_to_val",
            f"Forwarding SIGNAL_BUNDLE to VAL-001 (request_id={envelope.request_id})",
        )

        # Retrieve stored data bundle for price data
        pending = self._pending_requests.get(envelope.request_id, {})
        data_payload = pending.get("data_bundle", {})
        price_data = data_payload.get("bars", {})

        # Update stage
        if envelope.request_id in self._pending_requests:
            self._pending_requests[envelope.request_id]["stage"] = "validation"
            self._pending_requests[envelope.request_id]["signal_bundle"] = (
                envelope.payload
            )

        self.send_message(
            receiver=AgentID.VALIDATION,
            message_type=MessageType.SIGNAL_BUNDLE,
            payload={
                "bundle": envelope.payload.get("bundle", envelope.payload),
                "price_data": price_data,
            },
            request_id=envelope.request_id,
        )

    def _on_validation_report(self, envelope: MessageEnvelope) -> None:
        """H-005/H-006: VAL-001 -> ORCH-001.

        Receives validation report, decides:
        - If DEPLOY signals exist: forward to EXEC-001
        - If REFINE signals: send back to SIG-001
        - If all REJECT: log and stop
        """
        report = envelope.payload.get("report", {})
        deploy_count = report.get("deploy_count", 0)
        refine_count = report.get("refine_count", 0)
        reject_count = report.get("reject_count", 0)

        self._log_decision(
            "validation_triage",
            f"Validation: {deploy_count} DEPLOY, {refine_count} REFINE, "
            f"{reject_count} REJECT",
        )

        # Update stage
        if envelope.request_id in self._pending_requests:
            self._pending_requests[envelope.request_id]["stage"] = "execution"
            self._pending_requests[envelope.request_id]["validation_report"] = (
                report
            )

        if deploy_count > 0:
            # Forward DEPLOY signals to execution
            pending = self._pending_requests.get(envelope.request_id, {})
            data_payload = pending.get("data_bundle", {})
            signal_payload = pending.get("signal_bundle", {})

            self.send_message(
                receiver=AgentID.EXECUTION,
                message_type=MessageType.EXECUTION_REQUEST,
                payload={
                    "validation_report": report,
                    "price_data": data_payload.get("bars", {}),
                    "signal_data": signal_payload.get("signal_data", {}),
                },
                request_id=envelope.request_id,
            )

        if refine_count > 0:
            # Send REFINE signals back to SIG-001
            refine_ids = [
                v["signal_id"]
                for v in report.get("verdicts", [])
                if v.get("verdict") == "REFINE"
            ]
            self.send_message(
                receiver=AgentID.SIGNAL_ENG,
                message_type=MessageType.REFINE_REQUEST,
                payload={
                    "signal_ids": refine_ids,
                    "verdicts": [
                        v for v in report.get("verdicts", [])
                        if v.get("verdict") == "REFINE"
                    ],
                },
                request_id=envelope.request_id,
            )

        if deploy_count == 0 and refine_count == 0:
            self._log_decision(
                "all_rejected",
                f"All {reject_count} signals rejected. Pipeline stalled.",
            )

    def _on_execution_report(self, envelope: MessageEnvelope) -> None:
        """H-008: EXEC-001 -> ORCH-001.

        Receives execution report with APPROVED/VETOED verdicts.
        """
        report = envelope.payload.get("report", {})
        approved = report.get("approved_signals", [])
        vetoed = report.get("vetoed_signals", [])

        self._log_decision(
            "execution_result",
            f"Execution: {len(approved)} APPROVED, {len(vetoed)} VETOED",
        )

        if envelope.request_id in self._pending_requests:
            self._pending_requests[envelope.request_id]["stage"] = "complete"
            self._pending_requests[envelope.request_id]["execution_report"] = (
                report
            )

        if approved:
            self._log_decision(
                "deploy_ready",
                f"{len(approved)} signals ready for deployment: "
                + ", ".join(v.get("signal_id", "?") for v in approved),
            )

    def _on_alert(self, envelope: MessageEnvelope) -> None:
        """Handle alerts from any agent (escalation)."""
        level = envelope.payload.get("level", "INFO")
        source = envelope.payload.get("source", "unknown")
        issue = envelope.payload.get("issue", "")

        self._log_decision(
            "alert_received",
            f"Alert from {source}: [{level}] {issue}",
        )

        if level == "HALT":
            self.pipeline.halt(f"HALT alert from {source}: {issue}")
        elif level == "CRITICAL":
            self._log_decision(
                "critical_alert",
                f"CRITICAL alert from {source}: {issue} — monitoring",
            )

    def _on_halt(self, envelope: MessageEnvelope) -> None:
        """Handle HALT_COMMAND — emergency stop."""
        reason = envelope.payload.get("reason", "External halt command")
        self.pipeline.halt(reason)
        self._log_decision("halt", f"Pipeline halted: {reason}")

    def _on_resume(self, envelope: MessageEnvelope) -> None:
        """Handle RESUME_COMMAND — resume from halt."""
        target = envelope.payload.get("target_state")
        from alpha_lab.core.enums import PipelineState

        target_state = PipelineState(target) if target else None
        new_state = self.pipeline.resume_from_halt(target_state)
        self._log_decision("resume", f"Pipeline resumed to {new_state.value}")

    def _on_ack(self, envelope: MessageEnvelope) -> None:
        """Log ACKs for audit trail."""
        self.logger.debug(
            "ACK from %s for %s",
            envelope.sender.value,
            envelope.payload.get("original_request_id", "?"),
        )

    def _on_nack(self, envelope: MessageEnvelope) -> None:
        """Handle NACKs — log and potentially retry."""
        reason = envelope.payload.get("reason", "unknown")
        self._log_decision(
            "nack_received",
            f"NACK from {envelope.sender.value}: {reason}",
        )

    # ─── Pipeline Control ──────────────────────────────────────

    def run_pipeline(
        self,
        data_payload: dict[str, Any],
        request_id: str | None = None,
    ) -> str:
        """Kick off the end-to-end pipeline by sending a DATA_REQUEST.

        Args:
            data_payload: Payload for DATA-001 (instrument, date_range, etc.)
            request_id: Optional request ID to track through pipeline

        Returns:
            The request_id for this pipeline run
        """
        import uuid
        rid = request_id or str(uuid.uuid4())

        self._log_decision(
            "pipeline_start",
            f"Starting pipeline run {rid}",
        )

        self.send_message(
            receiver=AgentID.DATA_INFRA,
            message_type=MessageType.DATA_REQUEST,
            payload=data_payload,
            request_id=rid,
        )

        return rid

    def advance_phase(self, criteria_results: dict[str, Any] | None = None) -> bool:
        """
        Check go/no-go criteria and advance to next pipeline phase if met.

        Args:
            criteria_results: Dict of criterion -> (passed, value) tuples.
                              If None, uses stored phase data.

        Returns True if phase was advanced, False if criteria not met.
        """
        if criteria_results is None:
            criteria_results = {}

        go_no_go = self.pipeline.evaluate_go_no_go(criteria_results)

        if go_no_go.passed:
            new_state = self.pipeline.advance(go_no_go)
            self._log_decision(
                "phase_advanced",
                f"Advanced to {new_state.value}",
            )
            return True
        else:
            self._log_decision(
                "phase_blocked",
                f"Cannot advance from {self.pipeline.current_state.value}: "
                f"criteria not met — {go_no_go.details}",
            )
            return False

    def handle_conflict(
        self, agent_a: AgentID, agent_b: AgentID, issue: str
    ) -> str:
        """
        Resolve inter-agent conflicts per the decision framework:
        - SIG vs VAL: VAL wins (statistical evidence overrides intuition)
        - EXEC vs VAL: EXEC wins (unprofitable execution = worthless)
        - MON regime shift: Pause and request consensus
        """
        winner = _CONFLICT_RESOLUTION.get((agent_a, agent_b))

        if winner is not None:
            resolution = f"{winner.value} wins: {issue}"
        elif AgentID.MONITORING in (agent_a, agent_b):
            resolution = f"PAUSE for consensus: {issue} (regime shift detected)"
        else:
            resolution = f"ORCH-001 decides: {issue} (no precedent rule)"

        self._log_decision(
            "conflict_resolved",
            f"Conflict {agent_a.value} vs {agent_b.value}: {resolution}",
        )

        return resolution

    # ─── Decision Logging ──────────────────────────────────────

    def _log_decision(self, action: str, rationale: str) -> None:
        """Log a decision with timestamp and rationale for audit trail."""
        entry = {
            "action": action,
            "rationale": rationale,
            "phase": self.pipeline.current_state.value,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._decision_log.append(entry)
        logger.info("[DECISION] %s: %s", action, rationale)

    @property
    def decision_log(self) -> list[dict[str, Any]]:
        """Access the decision audit trail."""
        return list(self._decision_log)

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get current pipeline status summary."""
        return {
            "current_phase": self.pipeline.current_state.value,
            "active_agents": [a.value for a in self.pipeline.active_agents],
            "next_transition": self.pipeline.transition_description,
            "pending_requests": len(self._pending_requests),
            "decisions_made": len(self._decision_log),
            "phase_history": [
                {"phase": h.phase.value, "passed": h.passed}
                for h in self.pipeline.history
            ],
        }
