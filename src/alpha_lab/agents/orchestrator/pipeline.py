"""
Pipeline state machine for the Orchestrator.

Manages phase transitions per architecture spec Section 2.2:
INIT -> PHASE_1_2 -> PHASE_3_4 -> PHASE_5_6 -> PHASE_7 -> PHASE_8_9 -> DEPLOYED

Go/no-go criteria per phase:
- INIT -> PHASE_1_2: Data quality passes all checks
- PHASE_1_2 -> PHASE_3_4: Signal categories implemented, unit tests pass
- PHASE_3_4 -> PHASE_5_6: 8+ signals pass IC/hit rate/Sharpe thresholds
- PHASE_5_6 -> PHASE_7: Composite IC > 0.10, regime weights validated
- PHASE_7 -> PHASE_8_9: Net-of-cost Sharpe > 1.0, prop firm feasibility
- PHASE_8_9 -> DEPLOYED: 4+ weeks consistent, no CRITICAL alerts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from alpha_lab.core.enums import AgentID, PipelineState

logger = logging.getLogger(__name__)


@dataclass
class PhaseGoNoGo:
    """Go/no-go criteria for advancing a pipeline phase."""

    phase: PipelineState
    criteria: dict[str, Any]
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


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

# Go/no-go criteria keys per phase
PHASE_CRITERIA: dict[PipelineState, list[str]] = {
    PipelineState.INIT: ["data_quality_passed"],
    PipelineState.PHASE_1_2: ["signals_implemented", "unit_tests_pass"],
    PipelineState.PHASE_3_4: [
        "deploy_count", "min_ic_tstat", "min_hit_rate", "min_sharpe",
    ],
    PipelineState.PHASE_5_6: ["composite_ic", "regime_weights_validated"],
    PipelineState.PHASE_7: ["net_sharpe", "prop_firm_feasible", "mc_ruin_ok"],
    PipelineState.PHASE_8_9: ["weeks_consistent", "no_critical_alerts"],
}


class PipelineManager:
    """
    Manages the pipeline state machine.

    Tracks current phase, active agents, and go/no-go evaluations.
    """

    def __init__(self) -> None:
        self.current_state = PipelineState.INIT
        self.history: list[PhaseGoNoGo] = []
        self._phase_data: dict[str, Any] = {}

    @property
    def active_agents(self) -> list[AgentID]:
        """Return which agents should be active in the current phase."""
        return PHASE_AGENTS.get(self.current_state, [])

    @property
    def transition_description(self) -> str | None:
        """Human-readable description of what's needed to advance."""
        t = PHASE_TRANSITIONS.get(self.current_state)
        return t[0] if t else None

    @property
    def next_state(self) -> PipelineState | None:
        """The state we'd transition to if go/no-go passes."""
        t = PHASE_TRANSITIONS.get(self.current_state)
        return t[1] if t else None

    def store_phase_data(self, key: str, value: Any) -> None:
        """Store intermediate data for the current phase."""
        self._phase_data[key] = value

    def get_phase_data(self, key: str, default: Any = None) -> Any:
        """Retrieve intermediate data stored during this phase."""
        return self._phase_data.get(key, default)

    def evaluate_go_no_go(self, criteria_results: dict[str, Any]) -> PhaseGoNoGo:
        """
        Evaluate whether the current phase's go/no-go criteria are met.

        Args:
            criteria_results: Dict of criterion_name -> (passed: bool, value: Any)
                              e.g. {"data_quality_passed": (True, {"total_bars": 5000})}

        Returns:
            PhaseGoNoGo with pass/fail status and details
        """
        expected = PHASE_CRITERIA.get(self.current_state, [])
        details: dict[str, Any] = {}
        all_pass = True

        for criterion in expected:
            result = criteria_results.get(criterion)
            if result is None:
                details[criterion] = {"passed": False, "reason": "not evaluated"}
                all_pass = False
            elif isinstance(result, tuple) and len(result) == 2:
                passed, value = result
                details[criterion] = {"passed": passed, "value": value}
                if not passed:
                    all_pass = False
            elif isinstance(result, bool):
                details[criterion] = {"passed": result}
                if not result:
                    all_pass = False
            else:
                details[criterion] = {"passed": bool(result), "value": result}
                if not result:
                    all_pass = False

        go_no_go = PhaseGoNoGo(
            phase=self.current_state,
            criteria=criteria_results,
            passed=all_pass,
            details=details,
        )

        logger.info(
            "Go/no-go for %s: %s (details: %s)",
            self.current_state.value,
            "PASS" if all_pass else "FAIL",
            details,
        )

        return go_no_go

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
            raise ValueError(
                f"No transition defined from {self.current_state.value}"
            )

        _, next_state = transition
        self.history.append(go_no_go)
        old_state = self.current_state
        self.current_state = next_state
        self._phase_data = {}  # Reset for new phase

        logger.info(
            "Pipeline advanced: %s -> %s",
            old_state.value, next_state.value,
        )

        return next_state

    def halt(self, reason: str) -> None:
        """Emergency halt — move to HALT state."""
        logger.critical("PIPELINE HALT: %s", reason)
        self.history.append(
            PhaseGoNoGo(
                phase=self.current_state,
                criteria={"halt_reason": reason},
                passed=False,
                details={"reason": reason},
            )
        )
        self.current_state = PipelineState.HALT

    def resume_from_halt(self, target_state: PipelineState | None = None) -> PipelineState:
        """Resume from HALT to the last active state or a specified state."""
        if self.current_state != PipelineState.HALT:
            raise ValueError("Cannot resume: not in HALT state")

        if target_state is not None:
            self.current_state = target_state
        elif self.history:
            self.current_state = self.history[-1].phase
        else:
            self.current_state = PipelineState.INIT

        logger.info("Pipeline resumed to %s", self.current_state.value)
        return self.current_state
