"""Tests for the Databento data provider and provider factory."""

from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.providers import create_provider
from alpha_lab.agents.data_infra.providers.base import DataProvider
from alpha_lab.agents.data_infra.providers.databento import DatabentDataProvider
from alpha_lab.agents.data_infra.providers.polygon import PolygonDataProvider
from alpha_lab.agents.data_infra.providers.stub import StubDataProvider

# ────────────────────────────────────────────────────────────────
# Provider factory
# ────────────────────────────────────────────────────────────────


class TestProviderFactory:
    def test_create_stub(self):
        provider = create_provider("stub")
        assert isinstance(provider, StubDataProvider)

    def test_create_polygon(self):
        provider = create_provider("polygon", api_key="test")
        assert isinstance(provider, PolygonDataProvider)

    def test_create_databento(self):
        provider = create_provider("databento", api_key="test")
        assert isinstance(provider, DatabentDataProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent")

    def test_all_providers_are_data_provider(self):
        for name in ("stub", "polygon", "databento"):
            kwargs = {"api_key": "test"} if name != "stub" else {}
            provider = create_provider(name, **kwargs)
            assert isinstance(provider, DataProvider)


# ────────────────────────────────────────────────────────────────
# DatabentDataProvider
# ────────────────────────────────────────────────────────────────


class TestDatabentDataProvider:
    def test_is_data_provider_subclass(self):
        assert issubclass(DatabentDataProvider, DataProvider)

    def test_provider_name(self):
        provider = DatabentDataProvider(api_key="test-key")
        assert provider.provider_name == "databento"

    def test_supported_symbols(self):
        provider = DatabentDataProvider(api_key="test-key")
        assert "NQ" in provider.supported_symbols
        assert "ES" in provider.supported_symbols

    def test_connect_without_key_raises(self):
        provider = DatabentDataProvider(api_key="")
        with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
            provider.connect()

    def test_connect_and_disconnect(self):
        with patch.dict("sys.modules", {"databento": MagicMock()}):
            provider = DatabentDataProvider(api_key="test-key")
            provider.connect()
            assert provider._client is not None
            provider.disconnect()
            assert provider._client is None

    def test_parquet_path_format(self, tmp_path):
        provider = DatabentDataProvider(api_key="test", data_dir=tmp_path)
        path = provider._parquet_path("NQ", dt.date(2026, 2, 20), "mbp10")
        assert path == tmp_path / "NQ" / "2026-02-20" / "mbp10.parquet"

    def test_get_ticks_not_connected_raises(self, tmp_path):
        provider = DatabentDataProvider(api_key="test", data_dir=tmp_path)
        # _fetch_and_cache_ticks raises RuntimeError when not connected
        with pytest.raises(RuntimeError, match="Not connected"):
            provider._fetch_and_cache_ticks("NQ", dt.date(2026, 2, 20))

    def test_get_ohlcv_tick_timeframe_raises(self, tmp_path):
        provider = DatabentDataProvider(api_key="test", data_dir=tmp_path)
        from alpha_lab.core.enums import Timeframe

        with pytest.raises(NotImplementedError):
            provider.get_ohlcv(
                "NQ",
                Timeframe.TICK_987,
                dt.datetime(2026, 2, 20),
                dt.datetime(2026, 2, 21),
            )

    def test_get_ticks_cache_hit(self, tmp_path):
        """Pre-create Parquet file — verify no API call needed."""
        provider = DatabentDataProvider(api_key="test", data_dir=tmp_path)
        # Create a fake cached file
        date_dir = tmp_path / "NQ" / "2026-02-20"
        date_dir.mkdir(parents=True)
        fake_df = pd.DataFrame({
            "ts_event": pd.date_range("2026-02-20", periods=5, freq="1s", tz="UTC"),
            "price": [22000.0, 22001.0, 22002.0, 22001.5, 22003.0],
            "size": [10, 20, 15, 5, 30],
        })
        fake_df.to_parquet(date_dir / "mbp10.parquet")

        # Should read from cache without needing a client
        result = provider._fetch_and_cache_ticks("NQ", dt.date(2026, 2, 20))
        assert len(result) == 5
        assert "price" in result.columns

    def test_get_ticks_caches_parquet(self, tmp_path):
        """Mock API call, verify Parquet file is written."""
        provider = DatabentDataProvider(api_key="test", data_dir=tmp_path)

        mock_df = pd.DataFrame({
            "ts_event": pd.date_range("2026-02-20", periods=3, freq="1s", tz="UTC"),
            "price": [22000.0, 22001.0, 22002.0],
            "size": [10, 20, 15],
        })
        mock_data = MagicMock()
        mock_data.to_df.return_value = mock_df

        mock_client = MagicMock()
        mock_client.timeseries.get_range.return_value = mock_data
        provider._client = mock_client

        result = provider._fetch_and_cache_ticks("NQ", dt.date(2026, 2, 20))
        assert len(result) == 3

        # Verify Parquet was written
        cache_path = tmp_path / "NQ" / "2026-02-20" / "mbp10.parquet"
        assert cache_path.exists()

    def test_get_ohlcv_returns_dataframe(self, tmp_path):
        """Mock API, verify OHLCV DataFrame returned."""
        from alpha_lab.core.enums import Timeframe

        provider = DatabentDataProvider(api_key="test", data_dir=tmp_path)

        mock_df = pd.DataFrame({
            "ts_event": pd.date_range("2026-02-20 09:30", periods=5, freq="1min", tz="UTC"),
            "open": [22000.0, 22001.0, 22002.0, 22001.5, 22003.0],
            "high": [22005.0, 22006.0, 22007.0, 22006.5, 22008.0],
            "low": [21999.0, 22000.0, 22001.0, 22000.5, 22002.0],
            "close": [22001.0, 22002.0, 22003.0, 22002.5, 22004.0],
            "volume": [100, 200, 150, 80, 300],
        })
        mock_data = MagicMock()
        mock_data.to_df.return_value = mock_df

        mock_client = MagicMock()
        mock_client.timeseries.get_range.return_value = mock_data
        provider._client = mock_client

        result = provider.get_ohlcv(
            "NQ",
            Timeframe.M1,
            dt.datetime(2026, 2, 20, 9, 30),
            dt.datetime(2026, 2, 20, 9, 35),
        )
        assert len(result) == 5
        assert all(c in result.columns for c in ["open", "high", "low", "close", "volume"])


# ────────────────────────────────────────────────────────────────
# Front-month ticker resolution
# ────────────────────────────────────────────────────────────────


class TestDatabentFrontMonth:
    """Verify Databento provider uses same CME quarterly logic."""

    def test_january_resolves_to_march(self):
        assert DatabentDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 1, 15)
        ) == "NQH6"

    def test_february_resolves_to_march(self):
        assert DatabentDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 2, 10)
        ) == "NQH6"

    def test_march_late_resolves_to_june(self):
        assert DatabentDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 3, 20)
        ) == "NQM6"

    def test_december_late_rolls_to_next_year(self):
        assert DatabentDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 12, 20)
        ) == "NQH7"

    def test_es_ticker_format(self):
        assert DatabentDataProvider.resolve_front_month_ticker(
            "ES", dt.datetime(2026, 1, 15)
        ) == "ESH6"

    def test_matches_polygon_provider(self):
        """Databento and Polygon should produce identical tickers."""
        for month in range(1, 13):
            for day in (1, 14, 16, 28):
                try:
                    ref_date = dt.datetime(2026, month, day)
                except ValueError:
                    continue
                db_ticker = DatabentDataProvider.resolve_front_month_ticker("NQ", ref_date)
                pg_ticker = PolygonDataProvider.resolve_front_month_ticker("NQ", ref_date)
                assert db_ticker == pg_ticker, f"Mismatch at {ref_date}: {db_ticker} != {pg_ticker}"
