"""
Data Infrastructure Agent (DATA-001) — Market Data Engineer.

Owns the entire data pipeline from raw exchange tick feeds through
clean, session-tagged OHLCV bars at all timeframes.

See architecture spec Section 3 for full system prompt.
"""

from __future__ import annotations

from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import DataBundle
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope


class DataInfraAgent(BaseAgent):
    """
    DATA-001: Market Data Engineer.

    Responsibilities:
    - Tick aggregation (987-tick, 2000-tick bars)
    - Time bar construction (1m through 1D)
    - Session tagging (RTH, GLOBEX, killzones)
    - Previous day/week level computation
    - Data quality validation
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.DATA_INFRA, "Data Infrastructure", bus)

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle DATA_REQUEST messages from Orchestrator."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s (request_id=%s)",
            envelope.message_type.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.DATA_REQUEST:
            self.send_ack(envelope)
            # TODO: Build DataBundle from data provider
        else:
            self.send_nack(envelope, f"Unexpected message type: {envelope.message_type.value}")

        self.transition_state(AgentState.IDLE)

    def build_data_bundle(
        self, instrument: str, date_range: tuple[str, str], timeframes: list[str]
    ) -> DataBundle:
        """
        Build a complete DataBundle for the given parameters.

        1. Fetch raw data from provider
        2. Aggregate ticks into bars at all timeframes
        3. Tag sessions and killzones
        4. Compute previous day/week levels
        5. Run quality validation
        6. Return DataBundle with quality report
        """
        raise NotImplementedError
