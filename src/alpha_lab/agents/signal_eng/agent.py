"""
Signal Engineering Agent (SIG-001) — Alpha Researcher.

Builds, tests, and iterates on trading signal detectors for NQ/ES futures.
Takes clean DataBundle objects from DATA-001 and produces SignalBundle objects.

See docs/agent_prompts/SIG-001.md for full system prompt.
"""

from __future__ import annotations

from alpha_lab.agents.signal_eng.bundle_builder import build_signal_bundle
from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import DataBundle, SignalBundle
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope

_MAX_REFINE_ITERATIONS = 3


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
        self._refine_counts: dict[str, int] = {}

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle DATA_BUNDLE and REFINE_REQUEST messages."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.DATA_BUNDLE:
            self.send_ack(envelope)
            try:
                data = DataBundle.model_validate(envelope.payload["bundle"])
                bundle = self.generate_signals(data)
                self.send_message(
                    receiver=AgentID.ORCHESTRATOR,
                    message_type=MessageType.SIGNAL_BUNDLE,
                    payload={"bundle": bundle.model_dump()},
                    request_id=envelope.request_id,
                )
            except Exception:
                self.logger.exception("Failed to generate signals")
                self.send_nack(envelope, "Signal generation failed")

        elif envelope.message_type == MessageType.REFINE_REQUEST:
            self.send_ack(envelope)
            try:
                signal_id = envelope.payload["signal_id"]
                bundle = self.refine_signal(
                    signal_id=signal_id,
                    failed_metric=envelope.payload["failed_metric"],
                    failed_value=envelope.payload["failed_value"],
                    threshold=envelope.payload["threshold"],
                )
                self.send_message(
                    receiver=AgentID.ORCHESTRATOR,
                    message_type=MessageType.SIGNAL_BUNDLE,
                    payload={"bundle": bundle.model_dump()},
                    request_id=envelope.request_id,
                )
            except Exception:
                self.logger.exception("Failed to refine signal")
                self.send_nack(envelope, "Signal refinement failed")
        else:
            self.send_nack(
                envelope,
                f"Unexpected message type: {envelope.message_type.value}",
            )

        self.transition_state(AgentState.IDLE)

    def generate_signals(
        self,
        data_bundle: DataBundle,
        detector_ids: list[str] | None = None,
    ) -> SignalBundle:
        """
        Run all registered signal detectors on the DataBundle.

        Args:
            data_bundle: Clean DataBundle from DATA-001
            detector_ids: Optional subset of detectors to run

        Returns:
            SignalBundle containing all signal vectors
        """
        return build_signal_bundle(data_bundle, detector_ids)

    def refine_signal(
        self,
        signal_id: str,
        failed_metric: str,
        failed_value: float,
        threshold: float,
    ) -> SignalBundle:
        """
        Refine a specific signal based on VAL-001 feedback.

        The feedback is intentionally limited (firewall rule):
        only metric name, value, and threshold are provided.

        Max 3 refinement iterations per signal.
        """
        count = self._refine_counts.get(signal_id, 0)
        if count >= _MAX_REFINE_ITERATIONS:
            msg = (
                f"Signal {signal_id} has reached max refinement iterations "
                f"({_MAX_REFINE_ITERATIONS})"
            )
            raise ValueError(msg)

        self._refine_counts[signal_id] = count + 1
        self.logger.info(
            "Refining %s (iteration %d/%d): %s=%.4f (threshold=%.4f)",
            signal_id,
            count + 1,
            _MAX_REFINE_ITERATIONS,
            failed_metric,
            failed_value,
            threshold,
        )

        # TODO: Implement parameter adjustment based on failed_metric
        # For now, raise NotImplementedError for metrics we don't handle yet
        msg = (
            f"Refinement logic for metric '{failed_metric}' not yet implemented"
        )
        raise NotImplementedError(msg)
