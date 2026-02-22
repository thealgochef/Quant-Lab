"""Tests for the Statistical Validation agent."""

from alpha_lab.agents.validation.agent import ValidationAgent
from alpha_lab.agents.validation.firewall import ValidationTest
from alpha_lab.core.enums import AgentID


class TestValidationAgent:
    def test_create(self, message_bus):
        agent = ValidationAgent(message_bus)
        assert agent.agent_id == AgentID.VALIDATION
        assert agent.name == "Statistical Validation"


class TestValidationTestBase:
    def test_validation_test_is_abstract(self):
        """ValidationTest cannot be instantiated directly."""
        import pytest
        with pytest.raises(TypeError):
            ValidationTest()  # type: ignore[abstract]
