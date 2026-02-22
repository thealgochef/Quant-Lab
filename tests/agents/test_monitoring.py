"""Tests for the Live Monitoring agent."""

from alpha_lab.agents.monitoring.agent import MonitoringAgent
from alpha_lab.core.enums import AgentID


class TestMonitoringAgent:
    def test_create(self, message_bus):
        agent = MonitoringAgent(message_bus)
        assert agent.agent_id == AgentID.MONITORING
        assert agent.name == "Live Monitoring"
