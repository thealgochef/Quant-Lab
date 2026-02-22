"""
Orchestrator Agent (ORCH-001) — Research Director.

Sequences phases, manages handoffs, resolves conflicts, and makes go/no-go decisions.
Holds the thesis: 'Discover multi-timeframe confluence signals that survive
prop firm transaction costs and risk constraints.'

See architecture spec Section 2 for full system prompt and decision framework.
"""

from __future__ import annotations

from alpha_lab.agents.orchestrator.pipeline import PipelineManager
from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.enums import AgentID, AgentState
from alpha_lab.core.message import MessageBus, MessageEnvelope


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

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Route incoming messages based on type and current pipeline state."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )
        # TODO: Implement message routing based on pipeline state
        self.transition_state(AgentState.IDLE)

    def advance_phase(self) -> bool:
        """
        Check go/no-go criteria and advance to next pipeline phase if met.

        Returns True if phase was advanced, False if criteria not met.
        """
        raise NotImplementedError

    def handle_conflict(self, agent_a: AgentID, agent_b: AgentID, issue: str) -> str:
        """
        Resolve inter-agent conflicts per the decision framework:
        - SIG vs VAL: VAL wins (statistical evidence overrides intuition)
        - EXEC vs VAL: EXEC wins (unprofitable execution = worthless)
        - MON regime shift: Pause and request consensus
        """
        raise NotImplementedError
