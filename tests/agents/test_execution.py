"""Tests for the Execution & Risk agent."""

from alpha_lab.agents.execution.agent import ExecutionAgent
from alpha_lab.core.enums import AgentID


class TestExecutionAgent:
    def test_create(self, message_bus):
        agent = ExecutionAgent(message_bus)
        assert agent.agent_id == AgentID.EXECUTION
        assert agent.name == "Execution & Risk"
