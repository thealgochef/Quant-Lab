"""Tests for the Data Infrastructure agent."""

from alpha_lab.agents.data_infra.agent import DataInfraAgent
from alpha_lab.agents.data_infra.providers.base import DataProvider
from alpha_lab.agents.data_infra.providers.stub import StubDataProvider
from alpha_lab.core.enums import AgentID


class TestDataInfraAgent:
    def test_create(self, message_bus):
        agent = DataInfraAgent(message_bus)
        assert agent.agent_id == AgentID.DATA_INFRA
        assert agent.name == "Data Infrastructure"


class TestStubProvider:
    def test_provider_name(self):
        provider = StubDataProvider()
        assert provider.provider_name == "stub"

    def test_supported_symbols(self):
        provider = StubDataProvider()
        assert "NQ" in provider.supported_symbols
        assert "ES" in provider.supported_symbols

    def test_connect_disconnect(self):
        provider = StubDataProvider()
        provider.connect()  # Should not raise
        provider.disconnect()  # Should not raise

    def test_is_data_provider(self):
        assert issubclass(StubDataProvider, DataProvider)
