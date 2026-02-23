"""Pluggable data provider adapters."""

from __future__ import annotations

from alpha_lab.agents.data_infra.providers.base import DataProvider


def create_provider(name: str, **kwargs) -> DataProvider:
    """Factory function to create a DataProvider by config name.

    Args:
        name: Provider identifier ("databento", "polygon", or "stub")
        **kwargs: Passed to provider constructor (e.g. api_key=...)

    Returns:
        Configured DataProvider instance
    """
    if name == "databento":
        from alpha_lab.agents.data_infra.providers.databento import (
            DatabentDataProvider,
        )

        return DatabentDataProvider(**kwargs)
    if name == "polygon":
        from alpha_lab.agents.data_infra.providers.polygon import (
            PolygonDataProvider,
        )

        return PolygonDataProvider(**kwargs)
    if name == "stub":
        from alpha_lab.agents.data_infra.providers.stub import StubDataProvider

        return StubDataProvider()
    msg = f"Unknown provider: {name!r}. Valid: databento, polygon, stub"
    raise ValueError(msg)
