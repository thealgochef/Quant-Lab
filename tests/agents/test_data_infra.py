"""Tests for the Data Infrastructure agent and its sub-modules."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.agent import (
    DataInfraAgent,
    _build_session_metadata,
    _compute_pd_levels,
)
from alpha_lab.agents.data_infra.aggregation import aggregate_tick_bars, aggregate_time_bars
from alpha_lab.agents.data_infra.providers.base import DataProvider
from alpha_lab.agents.data_infra.providers.polygon import PolygonDataProvider
from alpha_lab.agents.data_infra.providers.stub import StubDataProvider
from alpha_lab.agents.data_infra.quality import (
    run_quality_checks,
    validate_cross_timeframe,
    validate_no_gaps,
    validate_ohlc,
    validate_timestamps,
    validate_volume,
)
from alpha_lab.agents.data_infra.sessions import (
    classify_killzone,
    tag_killzones,
    tag_sessions,
)
from alpha_lab.core.enums import AgentID, Killzone, Timeframe

# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_1m_bars() -> pd.DataFrame:
    """Synthetic 1m bars for a single RTH session (09:30-16:14)."""
    index = pd.date_range(
        "2026-02-20 09:30", periods=405, freq="1min", tz="US/Eastern"
    )
    rng = np.random.default_rng(42)
    base = 22000.0
    closes = base + rng.standard_normal(405).cumsum() * 5
    opens = closes + rng.uniform(-3, 3, 405)
    # Enforce OHLC consistency
    highs = np.maximum(opens, closes) + rng.uniform(0.25, 10, 405)
    lows = np.minimum(opens, closes) - rng.uniform(0.25, 10, 405)

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": rng.integers(100, 5000, 405),
        },
        index=index,
    )


@pytest.fixture
def sample_1m_with_sessions(sample_1m_bars: pd.DataFrame) -> pd.DataFrame:
    """1m bars with session and killzone tags applied."""
    df = tag_sessions(sample_1m_bars, "NQ")
    return tag_killzones(df)


@pytest.fixture
def sample_multiday_bars() -> pd.DataFrame:
    """Two days of RTH 1m bars for PD-levels / cross-TF tests."""
    rng = np.random.default_rng(99)
    frames = []
    for date_str in ("2026-02-19", "2026-02-20"):
        idx = pd.date_range(f"{date_str} 09:30", periods=405, freq="1min", tz="US/Eastern")
        base = 22000.0
        closes = base + rng.standard_normal(405).cumsum() * 5
        opens = closes + rng.uniform(-3, 3, 405)
        highs = np.maximum(opens, closes) + rng.uniform(0.25, 10, 405)
        lows = np.minimum(opens, closes) - rng.uniform(0.25, 10, 405)
        frames.append(
            pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": rng.integers(100, 5000, 405),
                },
                index=idx,
            )
        )
    return pd.concat(frames)


# ────────────────────────────────────────────────────────────────
# Agent
# ────────────────────────────────────────────────────────────────


class TestDataInfraAgent:
    def test_create(self, message_bus):
        agent = DataInfraAgent(message_bus)
        assert agent.agent_id == AgentID.DATA_INFRA
        assert agent.name == "Data Infrastructure"

    def test_registered_on_bus(self, message_bus):
        DataInfraAgent(message_bus)
        assert AgentID.DATA_INFRA in message_bus._handlers


# ────────────────────────────────────────────────────────────────
# Providers
# ────────────────────────────────────────────────────────────────


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
        provider.connect()
        provider.disconnect()

    def test_is_data_provider(self):
        assert issubclass(StubDataProvider, DataProvider)


class TestPolygonDataProvider:
    def test_is_data_provider_subclass(self):
        assert issubclass(PolygonDataProvider, DataProvider)

    def test_provider_name(self):
        provider = PolygonDataProvider(api_key="test-key")
        assert provider.provider_name == "polygon"

    def test_supported_symbols(self):
        provider = PolygonDataProvider(api_key="test-key")
        assert "NQ" in provider.supported_symbols
        assert "ES" in provider.supported_symbols

    def test_connect_without_api_key_raises(self):
        provider = PolygonDataProvider(api_key="")
        with pytest.raises(ValueError, match="POLYGON_API_KEY"):
            provider.connect()

    def test_connect_and_disconnect(self):
        provider = PolygonDataProvider(api_key="test-key")
        provider.connect()
        assert provider._client is not None
        provider.disconnect()
        assert provider._client is None

    def test_get_ticks_raises_not_implemented(self):
        provider = PolygonDataProvider(api_key="test-key")
        with pytest.raises(NotImplementedError):
            provider.get_ticks("NQ", dt.datetime.now(), dt.datetime.now())

    def test_get_ohlcv_tick_timeframe_raises(self):
        provider = PolygonDataProvider(api_key="test-key")
        provider.connect()
        with pytest.raises(NotImplementedError):
            provider.get_ohlcv(
                "NQ", Timeframe.TICK_987, dt.datetime.now(), dt.datetime.now()
            )
        provider.disconnect()


# ────────────────────────────────────────────────────────────────
# Front-month ticker resolution
# ────────────────────────────────────────────────────────────────


class TestFrontMonthTicker:
    def test_january_resolves_to_march(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 1, 15)
        ) == "NQH6"

    def test_february_resolves_to_march(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 2, 10)
        ) == "NQH6"

    def test_march_early_resolves_to_march(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 3, 10)
        ) == "NQH6"

    def test_march_late_resolves_to_june(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 3, 20)
        ) == "NQM6"

    def test_april_resolves_to_june(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 4, 1)
        ) == "NQM6"

    def test_june_early_resolves_to_june(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 6, 1)
        ) == "NQM6"

    def test_july_resolves_to_september(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 7, 15)
        ) == "NQU6"

    def test_december_late_rolls_to_next_year(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "NQ", dt.datetime(2026, 12, 20)
        ) == "NQH7"

    def test_es_ticker_format(self):
        assert PolygonDataProvider.resolve_front_month_ticker(
            "ES", dt.datetime(2026, 1, 15)
        ) == "ESH6"


# ────────────────────────────────────────────────────────────────
# Sessions
# ────────────────────────────────────────────────────────────────


class TestSessions:
    def test_tag_sessions_rth(self, sample_1m_bars):
        result = tag_sessions(sample_1m_bars, "NQ")
        assert "session_id" in result.columns
        assert "session_type" in result.columns
        assert (result["session_type"] == "RTH").all()

    def test_session_id_format(self, sample_1m_bars):
        result = tag_sessions(sample_1m_bars, "NQ")
        sid = result["session_id"].iloc[0]
        assert sid.startswith("NQ_")
        assert sid.endswith("_RTH")

    def test_globex_evening_trading_date(self):
        """Evening GLOBEX bars (>= 18:00) get next calendar date."""
        idx = pd.date_range("2026-02-20 18:00", periods=5, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame(
            {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100},
            index=idx,
        )
        result = tag_sessions(bars, "NQ")
        assert (result["session_type"] == "GLOBEX").all()
        assert "2026-02-21" in result["session_id"].iloc[0]

    def test_globex_morning_trading_date(self):
        """Morning GLOBEX bars (< 09:30) keep same calendar date."""
        idx = pd.date_range("2026-02-20 07:00", periods=5, freq="1min", tz="US/Eastern")
        bars = pd.DataFrame(
            {"open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 100},
            index=idx,
        )
        result = tag_sessions(bars, "NQ")
        assert (result["session_type"] == "GLOBEX").all()
        assert "2026-02-20" in result["session_id"].iloc[0]


# ────────────────────────────────────────────────────────────────
# Killzones
# ────────────────────────────────────────────────────────────────


class TestKillzones:
    def test_london(self):
        ts = pd.Timestamp("2026-02-20 03:00", tz="US/Eastern")
        assert classify_killzone(ts) == Killzone.LONDON

    def test_new_york(self):
        ts = pd.Timestamp("2026-02-20 10:00", tz="US/Eastern")
        assert classify_killzone(ts) == Killzone.NEW_YORK

    def test_asia(self):
        ts = pd.Timestamp("2026-02-20 20:00", tz="US/Eastern")
        assert classify_killzone(ts) == Killzone.ASIA

    def test_overlap(self):
        ts = pd.Timestamp("2026-02-20 08:30", tz="US/Eastern")
        assert classify_killzone(ts) == Killzone.OVERLAP

    def test_overlap_takes_precedence_over_ny(self):
        ts = pd.Timestamp("2026-02-20 08:00", tz="US/Eastern")
        assert classify_killzone(ts) == Killzone.OVERLAP

    def test_after_overlap_is_ny(self):
        ts = pd.Timestamp("2026-02-20 09:30", tz="US/Eastern")
        assert classify_killzone(ts) == Killzone.NEW_YORK

    def test_none(self):
        ts = pd.Timestamp("2026-02-20 12:00", tz="US/Eastern")
        assert classify_killzone(ts) == Killzone.NONE

    def test_tag_killzones_adds_column(self, sample_1m_bars):
        result = tag_killzones(sample_1m_bars)
        assert "killzone" in result.columns
        # RTH session 09:30-16:15 spans NY killzone (09:30-11:00) and NONE
        assert Killzone.NEW_YORK.value in result["killzone"].values
        assert Killzone.NONE.value in result["killzone"].values


# ────────────────────────────────────────────────────────────────
# Aggregation
# ────────────────────────────────────────────────────────────────


class TestAggregation:
    BOUNDARIES = {
        "rth_open": "09:30",
        "rth_close": "16:15",
        "globex_open": "18:00",
        "globex_close": "17:00",
    }

    def test_tick_bars_empty_input(self):
        result = aggregate_tick_bars(pd.DataFrame(columns=["price", "size"]), 987)
        assert result.empty

    def test_1m_passthrough(self, sample_1m_bars):
        result = aggregate_time_bars(sample_1m_bars, Timeframe.M1, self.BOUNDARIES)
        assert len(result) == len(sample_1m_bars)

    def test_5m_resampling(self, sample_1m_bars):
        result = aggregate_time_bars(sample_1m_bars, Timeframe.M5, self.BOUNDARIES)
        assert len(result) == 81  # 405 / 5 = 81
        assert all(c in result.columns for c in ["open", "high", "low", "close", "volume"])

    def test_15m_resampling(self, sample_1m_bars):
        result = aggregate_time_bars(sample_1m_bars, Timeframe.M15, self.BOUNDARIES)
        assert len(result) == 27  # 405 / 15 = 27

    def test_1h_resampling(self, sample_1m_bars):
        result = aggregate_time_bars(sample_1m_bars, Timeframe.H1, self.BOUNDARIES)
        # 09:30-16:14 spans parts of 7 clock hours
        assert 6 <= len(result) <= 8

    def test_ohlcv_integrity(self, sample_1m_bars):
        """Resampled high must match global 1m high, etc."""
        result = aggregate_time_bars(sample_1m_bars, Timeframe.H1, self.BOUNDARIES)
        assert abs(result["high"].max() - sample_1m_bars["high"].max()) < 0.01
        assert abs(result["low"].min() - sample_1m_bars["low"].min()) < 0.01

    def test_daily_aggregation(self, sample_1m_with_sessions):
        result = aggregate_time_bars(
            sample_1m_with_sessions, Timeframe.D1, self.BOUNDARIES
        )
        assert len(result) == 1  # One trading day
        assert abs(result["volume"].iloc[0] - sample_1m_with_sessions["volume"].sum()) < 1

    def test_empty_input(self):
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="US/Eastern"),
        )
        result = aggregate_time_bars(empty, Timeframe.M5, self.BOUNDARIES)
        assert result.empty

    def test_tick_timeframe_raises(self, sample_1m_bars):
        with pytest.raises(NotImplementedError):
            aggregate_time_bars(sample_1m_bars, Timeframe.TICK_987, self.BOUNDARIES)


# ────────────────────────────────────────────────────────────────
# Quality validation
# ────────────────────────────────────────────────────────────────


class TestQuality:
    def test_no_gaps_clean(self, sample_1m_with_sessions):
        assert validate_no_gaps(sample_1m_with_sessions) == []

    def test_no_gaps_with_gap(self, sample_1m_with_sessions):
        bars = sample_1m_with_sessions.drop(sample_1m_with_sessions.index[10:15])
        gaps = validate_no_gaps(bars)
        assert len(gaps) > 0
        assert gaps[0]["duration_sec"] > 120

    def test_volume_all_positive(self, sample_1m_with_sessions):
        assert validate_volume(sample_1m_with_sessions) == 0

    def test_volume_with_zeros(self, sample_1m_with_sessions):
        bars = sample_1m_with_sessions.copy()
        bars.iloc[5:8, bars.columns.get_loc("volume")] = 0
        assert validate_volume(bars) == 3

    def test_ohlc_clean(self, sample_1m_bars):
        assert validate_ohlc(sample_1m_bars) == 0

    def test_ohlc_violation(self, sample_1m_bars):
        bars = sample_1m_bars.copy()
        bars.iloc[0, bars.columns.get_loc("high")] = bars.iloc[0]["low"] - 10
        assert validate_ohlc(bars) >= 1

    def test_timestamps_monotonic(self, sample_1m_bars):
        assert validate_timestamps(sample_1m_bars) is True

    def test_timestamps_not_monotonic(self, sample_1m_bars):
        assert validate_timestamps(sample_1m_bars.iloc[::-1]) is False

    def test_timestamps_empty(self):
        empty = pd.DataFrame()
        assert validate_timestamps(empty) is True

    def test_cross_timeframe_no_data(self):
        assert validate_cross_timeframe({}) == 0
        assert validate_cross_timeframe({"1m": pd.DataFrame()}) == 0

    def test_run_quality_checks(self, sample_1m_with_sessions):
        report = run_quality_checks({"1m": sample_1m_with_sessions}, "NQ")
        assert report.passed is True
        assert report.total_bars > 0
        assert report.gaps_found == 0
        assert report.volume_zeros == 0
        assert report.ohlc_violations == 0
        assert 0.0 <= report.timestamp_coverage <= 1.0

    def test_run_quality_checks_fails_on_gaps(self, sample_1m_with_sessions):
        bars = sample_1m_with_sessions.drop(sample_1m_with_sessions.index[10:15])
        report = run_quality_checks({"1m": bars}, "NQ")
        assert report.passed is False
        assert report.gaps_found > 0


# ────────────────────────────────────────────────────────────────
# PD levels + session metadata helpers
# ────────────────────────────────────────────────────────────────


class TestPDLevels:
    def test_pd_levels_multiday(self, sample_multiday_bars):
        bars = tag_sessions(sample_multiday_bars, "NQ")
        bars = tag_killzones(bars)
        daily = aggregate_time_bars(
            bars,
            Timeframe.D1,
            {"rth_open": "09:30", "rth_close": "16:15",
             "globex_open": "18:00", "globex_close": "17:00"},
        )
        levels = _compute_pd_levels(daily, bars, "NQ")
        # First day has no previous, so only second day should have levels
        assert "2026-02-20" in levels
        assert "2026-02-19" not in levels
        lvl = levels["2026-02-20"]
        assert lvl.pd_high > lvl.pd_low
        assert lvl.pd_mid == (lvl.pd_high + lvl.pd_low) / 2

    def test_pd_levels_empty(self):
        assert _compute_pd_levels(pd.DataFrame(), pd.DataFrame(), "NQ") == {}


class TestSessionMetadata:
    def test_build_session_metadata(self, sample_1m_with_sessions):
        sessions = _build_session_metadata(sample_1m_with_sessions)
        assert len(sessions) >= 1
        s = sessions[0]
        assert s.session_id.startswith("NQ_")
        assert s.session_type in ("RTH", "GLOBEX", "POST_MARKET")
        assert "T09:30:00" in s.rth_open
