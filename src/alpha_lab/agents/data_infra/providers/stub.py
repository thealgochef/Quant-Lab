"""
Stub data provider — returns synthetic data for testing.

This provider generates deterministic synthetic OHLCV bars
so that the pipeline can be tested end-to-end without a real data feed.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from alpha_lab.agents.data_infra.providers.base import DataProvider
from alpha_lab.core.enums import Timeframe


class StubDataProvider(DataProvider):
    """Generates synthetic market data for pipeline testing."""

    def connect(self) -> None:
        """No-op for stub provider."""

    def disconnect(self) -> None:
        """No-op for stub provider."""

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Generate synthetic tick data."""
        raise NotImplementedError("Stub tick generation not yet implemented")

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV bars with random walk prices."""
        raise NotImplementedError("Stub OHLCV generation not yet implemented")

    def get_daily_settlement(self, symbol: str, date: datetime) -> float:
        """Return a synthetic settlement price."""
        raise NotImplementedError("Stub settlement not yet implemented")

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def supported_symbols(self) -> list[str]:
        return ["NQ", "ES"]
