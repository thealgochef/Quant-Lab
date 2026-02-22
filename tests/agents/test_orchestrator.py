"""Tests for the Orchestrator agent."""

from alpha_lab.agents.orchestrator.agent import OrchestratorAgent
from alpha_lab.agents.orchestrator.pipeline import PHASE_AGENTS, PHASE_TRANSITIONS, PipelineManager
from alpha_lab.core.enums import AgentID, PipelineState


class TestOrchestratorAgent:
    def test_create(self, message_bus):
        agent = OrchestratorAgent(message_bus)
        assert agent.agent_id == AgentID.ORCHESTRATOR
        assert agent.name == "Orchestrator"

    def test_registered_on_bus(self, message_bus):
        OrchestratorAgent(message_bus)
        assert AgentID.ORCHESTRATOR in message_bus._handlers


class TestPipelineManager:
    def test_initial_state(self):
        pm = PipelineManager()
        assert pm.current_state == PipelineState.INIT

    def test_active_agents_per_phase(self):
        assert AgentID.DATA_INFRA in PHASE_AGENTS[PipelineState.INIT]
        assert AgentID.DATA_INFRA in PHASE_AGENTS[PipelineState.PHASE_1_2]
        assert AgentID.SIGNAL_ENG in PHASE_AGENTS[PipelineState.PHASE_1_2]
        assert AgentID.VALIDATION in PHASE_AGENTS[PipelineState.PHASE_3_4]
        assert AgentID.MONITORING in PHASE_AGENTS[PipelineState.DEPLOYED]

    def test_all_phases_have_transitions(self):
        """Every phase except DEPLOYED has a defined transition."""
        for state in PipelineState:
            if state not in (PipelineState.DEPLOYED, PipelineState.HALT):
                assert state in PHASE_TRANSITIONS, f"Missing transition for {state.value}"
