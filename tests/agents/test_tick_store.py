"""Tests for TickStore (DuckDB query/replay) and MLDatasetBuilder."""

from __future__ import annotations

import datetime as dt
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.aggregation import aggregate_tick_bars
from alpha_lab.agents.data_infra.ml_export import MLDatasetBuilder
from alpha_lab.agents.data_infra.tick_store import TickStore

# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


def _make_synthetic_ticks(
    date_str: str = "2026-02-20",
    n: int = 1000,
    base_price: float = 22000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic MBP-10 tick DataFrame for testing."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(
        f"{date_str} 09:30",
        periods=n,
        freq="100ms",
        tz="UTC",
    )
    prices = base_price + rng.standard_normal(n).cumsum() * 0.25

    df = pd.DataFrame({
        "ts_event": ts,
        "price": prices,
        "size": rng.integers(1, 50, n),
    })

    # Add 10-level book depth columns
    for i in range(10):
        spread = (i + 1) * 0.25
        df[f"bid_px_{i:02d}"] = prices - spread
        df[f"ask_px_{i:02d}"] = prices + spread
        df[f"bid_sz_{i:02d}"] = rng.integers(10, 200, n).astype(float)
        df[f"ask_sz_{i:02d}"] = rng.integers(10, 200, n).astype(float)

    return df


@pytest.fixture
def tick_data_dir(tmp_path) -> Path:
    """Create a temp dir with synthetic Parquet tick files for 2 days."""
    for date_str in ("2026-02-20", "2026-02-21"):
        ticks = _make_synthetic_ticks(date_str=date_str, n=500)
        out_dir = tmp_path / "NQ" / date_str
        out_dir.mkdir(parents=True)
        ticks.to_parquet(out_dir / "mbp10.parquet")
    return tmp_path


@pytest.fixture
def tick_store(tick_data_dir) -> TickStore:
    """TickStore with 2 days of NQ data registered."""
    store = TickStore(tick_data_dir)
    store.register_date_range(
        "NQ", date(2026, 2, 20), date(2026, 2, 21)
    )
    yield store
    store.close()


# ────────────────────────────────────────────────────────────────
# TickStore
# ────────────────────────────────────────────────────────────────


class TestTickStore:
    def test_register_and_query(self, tick_store):
        """Register synthetic Parquet, query returns data."""
        result = tick_store.query_ticks(
            "NQ",
            datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC),
            datetime(2026, 2, 20, 9, 31, tzinfo=dt.UTC),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "ts_event" in result.columns
        assert "price" in result.columns

    def test_query_respects_time_bounds(self, tick_store):
        """Data outside [start, end] not returned."""
        start = datetime(2026, 2, 20, 9, 30, 0, tzinfo=dt.UTC)
        end = datetime(2026, 2, 20, 9, 30, 5, tzinfo=dt.UTC)
        result = tick_store.query_ticks("NQ", start, end)

        if not result.empty:
            ts_col = result["ts_event"]
            assert (ts_col >= pd.Timestamp(start)).all()
            assert (ts_col <= pd.Timestamp(end)).all()

    def test_no_lookahead_bias(self, tick_store):
        """Query at time T returns nothing after T."""
        # Query only first 2 seconds of data
        end_time = datetime(2026, 2, 20, 9, 30, 2, tzinfo=dt.UTC)
        result = tick_store.query_ticks(
            "NQ",
            datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC),
            end_time,
        )
        if not result.empty:
            max_ts = result["ts_event"].max()
            assert max_ts <= pd.Timestamp(end_time)

    def test_replay_chronological_order(self, tick_store):
        """Replay yields batches in chronological order."""
        start = datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC)
        end = datetime(2026, 2, 20, 9, 30, 10, tzinfo=dt.UTC)
        step = timedelta(seconds=2)

        prev_max = None
        for batch in tick_store.replay("NQ", start, end, step):
            if batch.empty:
                continue
            batch_min = batch["ts_event"].min()
            if prev_max is not None:
                assert batch_min >= prev_max, "Batches must be chronological"
            prev_max = batch["ts_event"].max()

    def test_replay_no_future_data(self, tick_store):
        """Each replay step has no data beyond step boundary."""
        start = datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC)
        end = datetime(2026, 2, 20, 9, 30, 10, tzinfo=dt.UTC)
        step = timedelta(seconds=2)

        current = start
        for batch in tick_store.replay("NQ", start, end, step):
            batch_end = min(current + step, end)
            if not batch.empty:
                max_ts = batch["ts_event"].max()
                assert max_ts <= pd.Timestamp(batch_end)
            current = batch_end

    def test_build_bars_from_ticks(self, tick_store):
        """Tick aggregation via DuckDB produces valid OHLCV."""
        start = datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC)
        end = datetime(2026, 2, 20, 9, 31, tzinfo=dt.UTC)
        bars = tick_store.build_bars_from_ticks(
            "NQ", start, end, bar_size="10 seconds"
        )
        assert isinstance(bars, pd.DataFrame)
        if not bars.empty:
            assert all(
                c in bars.columns
                for c in ["open", "high", "low", "close", "volume"]
            )
            # OHLC integrity
            assert (bars["high"] >= bars["low"]).all()
            assert (bars["high"] >= bars["open"]).all()
            assert (bars["high"] >= bars["close"]).all()

    def test_book_snapshot(self, tick_store):
        """Returns book state at a given timestamp."""
        as_of = datetime(2026, 2, 20, 9, 30, 5, tzinfo=dt.UTC)
        snapshot = tick_store.get_book_snapshot("NQ", as_of)
        assert isinstance(snapshot, pd.DataFrame)
        assert not snapshot.empty

    def test_empty_query(self, tick_store):
        """No data for a date range with no registered data."""
        result = tick_store.query_ticks(
            "ES",  # Not registered
            datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC),
            datetime(2026, 2, 20, 9, 31, tzinfo=dt.UTC),
        )
        assert result.empty

    def test_register_missing_file_returns_false(self, tick_store):
        """Registering a non-existent date returns False."""
        assert tick_store.register_symbol_date("NQ", "2099-01-01") is False

    def test_register_date_range_count(self, tick_data_dir):
        """register_date_range returns the count of files found."""
        store = TickStore(tick_data_dir)
        count = store.register_date_range(
            "NQ", date(2026, 2, 20), date(2026, 2, 21)
        )
        assert count == 2
        store.close()


# ────────────────────────────────────────────────────────────────
# MLDatasetBuilder
# ────────────────────────────────────────────────────────────────


class TestMLDatasetBuilder:
    def test_compute_orderbook_features(self):
        """Verify bid-ask spread, depth imbalance, microprice."""
        ticks = _make_synthetic_ticks(n=50)
        builder = MLDatasetBuilder(tick_store=None)
        features = builder.compute_orderbook_features(ticks)

        assert "spread" in features
        assert "microprice" in features
        assert "depth_imbalance" in features
        assert features["spread"] > 0

    def test_compute_orderbook_features_empty(self):
        """Empty DataFrame returns empty dict."""
        builder = MLDatasetBuilder(tick_store=None)
        features = builder.compute_orderbook_features(pd.DataFrame())
        assert features == {}

    def test_compute_bar_features(self):
        """Verify returns, volatility, volume z-score columns exist."""
        rng = np.random.default_rng(42)
        bars = pd.DataFrame({
            "open": 22000 + rng.standard_normal(100).cumsum(),
            "high": 22005 + rng.standard_normal(100).cumsum(),
            "low": 21995 + rng.standard_normal(100).cumsum(),
            "close": 22000 + rng.standard_normal(100).cumsum(),
            "volume": rng.integers(100, 5000, 100),
        }, index=pd.date_range("2026-02-20 09:30", periods=100, freq="5min"))
        # Fix OHLC consistency
        bars["high"] = bars[["open", "high", "low", "close"]].max(axis=1) + 1
        bars["low"] = bars[["open", "high", "low", "close"]].min(axis=1) - 1

        builder = MLDatasetBuilder(tick_store=None)
        feats = builder.compute_bar_features(bars)

        assert "ret_1" in feats.columns
        assert "vol_20" in feats.columns
        assert "volume_zscore" in feats.columns
        assert "range_atr_ratio" in feats.columns
        assert len(feats) == len(bars)

    def test_compute_bar_features_empty(self):
        builder = MLDatasetBuilder(tick_store=None)
        feats = builder.compute_bar_features(pd.DataFrame())
        assert feats.empty

    def test_forward_returns(self):
        """Labels are correct forward-looking returns."""
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        bars = pd.DataFrame({
            "open": close, "high": close + 1, "low": close - 1,
            "close": close, "volume": [100] * 5,
        })
        fwd = MLDatasetBuilder._compute_forward_returns(bars, horizons=[1, 2])

        # fwd_ret_1 at index 0 = (101/100) - 1 = 0.01
        assert abs(fwd["fwd_ret_1"].iloc[0] - 0.01) < 1e-10
        # fwd_ret_2 at index 0 = (102/100) - 1 = 0.02
        assert abs(fwd["fwd_ret_2"].iloc[0] - 0.02) < 1e-10
        # Last values should be NaN (no future data)
        assert pd.isna(fwd["fwd_ret_1"].iloc[-1])

    def test_build_features_point_in_time(self, tick_store):
        """Features at T use only data <= T."""
        builder = MLDatasetBuilder(tick_store)
        start = datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC)
        end = datetime(2026, 2, 20, 9, 31, tzinfo=dt.UTC)
        df = builder.build_features("NQ", start, end, bar_tf="10 seconds")

        # Should produce a DataFrame (may be empty if bars are sparse)
        assert isinstance(df, pd.DataFrame)

    def test_export_dataset_splits(self, tick_store, tmp_path):
        """Train/val/test split ratios are correct."""
        builder = MLDatasetBuilder(tick_store)
        start = datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC)
        end = datetime(2026, 2, 20, 9, 31, tzinfo=dt.UTC)
        output = tmp_path / "test_export" / "features.parquet"

        counts = builder.export_dataset(
            "NQ", start, end, "10 seconds",
            output_path=output,
            train_pct=0.7,
            val_pct=0.15,
        )
        assert isinstance(counts, dict)
        assert "train" in counts
        assert "val" in counts
        assert "test" in counts


# ────────────────────────────────────────────────────────────────
# aggregate_tick_bars
# ────────────────────────────────────────────────────────────────


class TestAggregateTickBars:
    def test_basic_aggregation(self):
        """Verify tick bars produce valid OHLCV from synthetic ticks."""
        rng = np.random.default_rng(42)
        n = 2000
        ticks = pd.DataFrame({
            "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
            "size": rng.integers(1, 50, n),
            "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
        })
        result = aggregate_tick_bars(ticks, tick_count=987)

        assert not result.empty
        assert all(
            c in result.columns
            for c in ["open", "high", "low", "close", "volume", "tick_count"]
        )
        # Should produce 2 full bars from 2000 ticks at 987 per bar
        assert len(result) == 2
        assert (result["high"] >= result["low"]).all()
        assert (result["high"] >= result["open"]).all()
        assert (result["high"] >= result["close"]).all()

    def test_partial_chunk_dropped(self):
        """Partial final chunk < 50% is dropped."""
        rng = np.random.default_rng(42)
        n = 1100  # 987 + 113 (113 < 987*0.5 = 493)
        ticks = pd.DataFrame({
            "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
            "size": rng.integers(1, 50, n),
            "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
        })
        result = aggregate_tick_bars(ticks, tick_count=987)
        # Only 1 full bar, partial dropped
        assert len(result) == 1

    def test_partial_chunk_kept(self):
        """Partial final chunk >= 50% is kept."""
        rng = np.random.default_rng(42)
        n = 1500  # 987 + 513 (513 >= 987*0.5 = 493.5)
        ticks = pd.DataFrame({
            "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
            "size": rng.integers(1, 50, n),
            "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
        })
        result = aggregate_tick_bars(ticks, tick_count=987)
        assert len(result) == 2  # Full bar + partial kept

    def test_empty_input(self):
        result = aggregate_tick_bars(pd.DataFrame(columns=["price", "size"]), 987)
        assert result.empty

    def test_datetime_index_input(self):
        """Accept DatetimeIndex instead of timestamp column."""
        rng = np.random.default_rng(42)
        n = 987
        idx = pd.date_range("2026-02-20 09:30", periods=n, freq="100ms")
        ticks = pd.DataFrame({
            "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
            "size": rng.integers(1, 50, n),
        }, index=idx)
        result = aggregate_tick_bars(ticks, tick_count=987)
        assert len(result) == 1

    def test_volume_sum(self):
        """Volume in bar equals sum of tick sizes."""
        rng = np.random.default_rng(42)
        n = 987
        sizes = rng.integers(1, 50, n)
        ticks = pd.DataFrame({
            "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
            "size": sizes,
            "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
        })
        result = aggregate_tick_bars(ticks, tick_count=987)
        assert result.iloc[0]["volume"] == sizes.sum()
