"""
Pipeline state machine for the Orchestrator.

Manages phase transitions per architecture spec Section 2.2:
INIT -> PHASE_1_2 -> PHASE_3_4 -> PHASE_5_6 -> PHASE_7 -> PHASE_8_9 -> DEPLOYED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_lab.core.enums import AgentID, PipelineState


@dataclass
class PhaseGoNoGo:
    """Go/no-go criteria for advancing a pipeline phase."""

    phase: PipelineState
    criteria: dict[str, Any]
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)


# Phase transition map: current_state -> (go_criteria_description, next_state)
PHASE_TRANSITIONS: dict[PipelineState, tuple[str, PipelineState]] = {
    PipelineState.INIT: (
        "Data quality report passes all checks",
        PipelineState.PHASE_1_2,
    ),
    PipelineState.PHASE_1_2: (
        "All 20 signal categories implemented, unit tests pass",
        PipelineState.PHASE_3_4,
    ),
    PipelineState.PHASE_3_4: (
        "8+ signals pass minimum IC/hit rate/Sharpe thresholds",
        PipelineState.PHASE_5_6,
    ),
    PipelineState.PHASE_5_6: (
        "Composite signal IC > 0.10, regime weights validated",
        PipelineState.PHASE_7,
    ),
    PipelineState.PHASE_7: (
        "Net-of-cost Sharpe > 1.0, prop firm feasibility = True",
        PipelineState.PHASE_8_9,
    ),
    PipelineState.PHASE_8_9: (
        "4+ weeks consistent performance, no CRITICAL alerts",
        PipelineState.DEPLOYED,
    ),
}

# Which agents are active in each phase
PHASE_AGENTS: dict[PipelineState, list[AgentID]] = {
    PipelineState.INIT: [AgentID.DATA_INFRA],
    PipelineState.PHASE_1_2: [AgentID.DATA_INFRA, AgentID.SIGNAL_ENG],
    PipelineState.PHASE_3_4: [AgentID.VALIDATION],
    PipelineState.PHASE_5_6: [AgentID.SIGNAL_ENG, AgentID.VALIDATION],
    PipelineState.PHASE_7: [AgentID.EXECUTION],
    PipelineState.PHASE_8_9: [AgentID.MONITORING, AgentID.EXECUTION],
    PipelineState.DEPLOYED: [AgentID.MONITORING],
}


class PipelineManager:
    """
    Manages the pipeline state machine.

    Tracks current phase, active agents, and go/no-go evaluations.
    """

    def __init__(self) -> None:
        self.current_state = PipelineState.INIT
        self.history: list[PhaseGoNoGo] = []

    @property
    def active_agents(self) -> list[AgentID]:
        """Return which agents should be active in the current phase."""
        return PHASE_AGENTS.get(self.current_state, [])

    def evaluate_go_no_go(self, criteria_results: dict[str, Any]) -> PhaseGoNoGo:
        """
        Evaluate whether the current phase's go/no-go criteria are met.

        Args:
            criteria_results: Dict of criterion_name -> (passed: bool, value: Any)

        Returns:
            PhaseGoNoGo with pass/fail status and details
        """
        raise NotImplementedError

    def advance(self, go_no_go: PhaseGoNoGo) -> PipelineState:
        """
        Advance to the next phase if go/no-go passed.

        Raises ValueError if criteria not met.
        """
        if not go_no_go.passed:
            raise ValueError(
                f"Cannot advance from {self.current_state.value}: "
                f"go/no-go criteria not met"
            )

        transition = PHASE_TRANSITIONS.get(self.current_state)
        if transition is None:
            raise ValueError(f"No transition defined from {self.current_state.value}")

        _, next_state = transition
        self.history.append(go_no_go)
        self.current_state = next_state
        return next_state
