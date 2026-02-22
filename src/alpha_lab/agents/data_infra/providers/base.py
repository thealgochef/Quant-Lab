"""
Abstract data provider interface.

To add a new data vendor:
1. Subclass DataProvider
2. Implement all abstract methods
3. Set provider name in config/settings.yaml
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from alpha_lab.core.enums import Timeframe


class DataProvider(ABC):
    """Pluggable data vendor adapter interface."""

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to data source."""

    @abstractmethod
    def disconnect(self) -> None:
        """Cleanly close connection."""

    @abstractmethod
    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch raw tick data.

        Returns DataFrame with columns: [price, size, timestamp]
        """

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars.

        Returns DataFrame with columns: [open, high, low, close, volume]
        Index: DatetimeIndex
        """

    @abstractmethod
    def get_daily_settlement(self, symbol: str, date: datetime) -> float:
        """Get settlement price for a given date."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable name of this data provider."""

    @property
    @abstractmethod
    def supported_symbols(self) -> list[str]:
        """List of symbols this provider can serve."""
