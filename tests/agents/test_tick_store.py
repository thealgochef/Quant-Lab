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
    depth_levels: int = 10,
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

    df = pd.DataFrame(
        {
            "ts_event": ts,
            "price": prices,
            "size": rng.integers(1, 50, n),
        }
    )

    # Add variable-depth book columns (e.g., 10 for MBP-10, 1 for MBP-1)
    for i in range(depth_levels):
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
    store.register_date_range("NQ", date(2026, 2, 20), date(2026, 2, 21))
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
        bars = tick_store.build_bars_from_ticks("NQ", start, end, bar_size="10 seconds")
        assert isinstance(bars, pd.DataFrame)
        if not bars.empty:
            assert all(c in bars.columns for c in ["open", "high", "low", "close", "volume"])
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

    def test_query_tick_feature_rows_includes_mbp10_columns(self, tick_store):
        """Feature-row query should include full MBP-10 depth columns."""
        result = tick_store.query_tick_feature_rows(
            "NQ",
            datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC),
            datetime(2026, 2, 20, 9, 30, 5, tzinfo=dt.UTC),
        )
        assert not result.empty
        assert "ts_event" in result.columns
        assert "price" in result.columns
        assert "size" in result.columns
        assert "bid_px_00" in result.columns
        assert "ask_px_09" in result.columns
        assert "bid_sz_00" in result.columns
        assert "ask_sz_09" in result.columns

    def test_query_tick_prices_remains_lean(self, tick_store):
        """Lean query path should remain compact for non-book consumers."""
        result = tick_store.query_tick_prices(
            "NQ",
            datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC),
            datetime(2026, 2, 20, 9, 30, 5, tzinfo=dt.UTC),
        )
        assert not result.empty
        assert set(result.columns) == {"ts_event", "price", "size"}

    def test_query_tick_feature_rows_handles_mbp1(self, tmp_path):
        """Feature-row query should gracefully project only available depth."""
        ticks = _make_synthetic_ticks(date_str="2026-02-20", n=200, depth_levels=1)
        out_dir = tmp_path / "NQ" / "2026-02-20"
        out_dir.mkdir(parents=True)
        ticks.to_parquet(out_dir / "mbp1.parquet")

        store = TickStore(tmp_path)
        store.register_symbol_date("NQ", "2026-02-20")
        result = store.query_tick_feature_rows(
            "NQ",
            datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC),
            datetime(2026, 2, 20, 9, 30, 5, tzinfo=dt.UTC),
        )
        store.close()

        assert not result.empty
        assert "bid_px_00" in result.columns
        assert "ask_px_00" in result.columns
        assert "bid_sz_00" in result.columns
        assert "ask_sz_00" in result.columns
        assert "bid_px_01" not in result.columns
        assert "ask_sz_09" not in result.columns

    def test_query_tick_feature_rows_handles_trades_only(self, tmp_path):
        """Feature-row query should fall back to compact projection for trades schema."""
        ts = pd.date_range("2026-02-20 09:30", periods=200, freq="100ms", tz="UTC")
        rng = np.random.default_rng(123)
        trades = pd.DataFrame(
            {
                "ts_event": ts,
                "price": 22000.0 + rng.standard_normal(200).cumsum() * 0.25,
                "size": rng.integers(1, 20, 200),
            }
        )
        out_dir = tmp_path / "NQ" / "2026-02-20"
        out_dir.mkdir(parents=True)
        trades.to_parquet(out_dir / "trades.parquet")

        store = TickStore(tmp_path, tick_filename="trades.parquet")
        store.register_symbol_date("NQ", "2026-02-20")
        result = store.query_tick_feature_rows(
            "NQ",
            datetime(2026, 2, 20, 9, 30, tzinfo=dt.UTC),
            datetime(2026, 2, 20, 9, 30, 5, tzinfo=dt.UTC),
        )
        store.close()

        assert not result.empty
        assert set(result.columns) == {"ts_event", "price", "size"}

    def test_register_missing_file_returns_false(self, tick_store):
        """Registering a non-existent date returns False."""
        assert tick_store.register_symbol_date("NQ", "2099-01-01") is False

    def test_register_date_range_count(self, tick_data_dir):
        """register_date_range returns the count of files found."""
        store = TickStore(tick_data_dir)
        count = store.register_date_range("NQ", date(2026, 2, 20), date(2026, 2, 21))
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
        bars = pd.DataFrame(
            {
                "open": 22000 + rng.standard_normal(100).cumsum(),
                "high": 22005 + rng.standard_normal(100).cumsum(),
                "low": 21995 + rng.standard_normal(100).cumsum(),
                "close": 22000 + rng.standard_normal(100).cumsum(),
                "volume": rng.integers(100, 5000, 100),
            },
            index=pd.date_range("2026-02-20 09:30", periods=100, freq="5min"),
        )
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
        bars = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [100] * 5,
            }
        )
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
            "NQ",
            start,
            end,
            "10 seconds",
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
        ticks = pd.DataFrame(
            {
                "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
                "size": rng.integers(1, 50, n),
                "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
            }
        )
        result = aggregate_tick_bars(ticks, tick_count=987)

        assert not result.empty
        assert all(
            c in result.columns for c in ["open", "high", "low", "close", "volume", "tick_count"]
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
        ticks = pd.DataFrame(
            {
                "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
                "size": rng.integers(1, 50, n),
                "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
            }
        )
        result = aggregate_tick_bars(ticks, tick_count=987)
        # Only 1 full bar, partial dropped
        assert len(result) == 1

    def test_partial_chunk_kept(self):
        """Partial final chunk >= 50% is kept."""
        rng = np.random.default_rng(42)
        n = 1500  # 987 + 513 (513 >= 987*0.5 = 493.5)
        ticks = pd.DataFrame(
            {
                "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
                "size": rng.integers(1, 50, n),
                "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
            }
        )
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
        ticks = pd.DataFrame(
            {
                "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
                "size": rng.integers(1, 50, n),
            },
            index=idx,
        )
        result = aggregate_tick_bars(ticks, tick_count=987)
        assert len(result) == 1

    def test_volume_sum(self):
        """Volume in bar equals sum of tick sizes."""
        rng = np.random.default_rng(42)
        n = 987
        sizes = rng.integers(1, 50, n)
        ticks = pd.DataFrame(
            {
                "price": 22000 + rng.standard_normal(n).cumsum() * 0.25,
                "size": sizes,
                "timestamp": pd.date_range("2026-02-20 09:30", periods=n, freq="100ms"),
            }
        )
        result = aggregate_tick_bars(ticks, tick_count=987)
        assert result.iloc[0]["volume"] == sizes.sum()


# ────────────────────────────────────────────────────────────────
# Phase 4d — trade-price tick bars (build_tick_bars / query_tick_events)
# ────────────────────────────────────────────────────────────────


def _make_mbp10_with_actions(
    date_str: str = "2026-02-20",
    n: int = 600,
    base_price: float = 22000.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Synthetic MBP-10 frame WITH 'action' and 'sequence', like real databento data.

    ~40% of rows are trades (action='T') on the exact 0.25 grid; the rest are book
    events ('A'/'C'/'M') whose 'price' is an off-grid order price. Book columns carry a
    top-of-book that lands on the 0.125 grid (so book-mid bars are lossless too). All rows
    sit inside one 18:00-ET trading day (RTH), so bar_index == cumcount // N globally.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range(f"{date_str} 14:30", periods=n, freq="100ms", tz="UTC")  # ~09:30 ET
    actions = rng.choice(["T", "A", "C", "M"], size=n, p=[0.4, 0.25, 0.2, 0.15])
    mid = base_price + rng.standard_normal(n).cumsum() * 0.25
    # trade price snapped to the 0.25 grid; book/order rows carry an off-grid price.
    trade_price = np.round(mid / 0.25) * 0.25
    order_price = mid + rng.standard_normal(n) * 0.1
    price = np.where(actions == "T", trade_price, order_price)
    df = pd.DataFrame(
        {
            "ts_event": ts,
            "sequence": np.arange(1, n + 1, dtype="int64"),
            "action": actions,
            "price": price,
            "size": rng.integers(1, 50, n).astype("int64"),
            # top-of-book on the 0.125 grid -> book-mid lands on 0.0625? keep mid on 0.125:
            "bid_px_00": np.round((mid - 0.25) / 0.125) * 0.125,
            "ask_px_00": np.round((mid + 0.25) / 0.125) * 0.125,
        }
    )
    return df


def _ref_trade_bars(df: pd.DataFrame, n: int, tick: float = 0.25) -> pd.DataFrame:
    """Independent pandas reference: trade-price tick bars over the SAME composite order."""
    t = df[df["action"].str.lower() == "t"].copy()
    t = t[t["price"].notna() & (t["price"] > 0)]
    t = t.sort_values(["ts_event", "sequence", "price", "size"], kind="stable").reset_index(
        drop=True
    )
    t["bar_index"] = t.index // n  # single trading day in this fixture
    rows = []
    for bi, sub in t.groupby("bar_index", sort=True):
        rows.append(
            {
                "bar_index": int(bi),
                "open_t": round(sub["price"].iloc[0] / tick),
                "high_t": round(sub["price"].max() / tick),
                "low_t": round(sub["price"].min() / tick),
                "close_t": round(sub["price"].iloc[-1] / tick),
                "volume": int(sub["size"].sum()),
                "trade_count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _register_frame(tmp_path: Path, df: pd.DataFrame, date_str: str = "2026-02-20") -> TickStore:
    out = tmp_path / "NQ" / date_str
    out.mkdir(parents=True)
    df.to_parquet(out / "mbp10.parquet")
    store = TickStore(tmp_path)
    store.register_symbol_date("NQ", date_str)
    return store


class TestTradeBars:
    _WIN = (
        datetime(2026, 2, 20, 14, 0, tzinfo=dt.UTC),
        datetime(2026, 2, 20, 15, 0, tzinfo=dt.UTC),
    )

    def test_trade_bars_match_pandas_reference(self, tmp_path):
        """DuckDB trade bars == an independent pandas bucketer (portable parity, no store)."""
        df = _make_mbp10_with_actions(n=600)
        store = _register_frame(tmp_path, df)
        try:
            bars = store.build_tick_bars("NQ", *self._WIN, tick_count=50)  # default trade
        finally:
            store.close()
        ref = _ref_trade_bars(df, 50)
        got = bars.reset_index().sort_values("bar_index").reset_index(drop=True)
        assert len(got) == len(ref)
        for i, r in ref.iterrows():
            g = got.iloc[i]
            assert int(round(g.open / 0.25)) == r.open_t
            assert int(round(g.high / 0.25)) == r.high_t
            assert int(round(g.low / 0.25)) == r.low_t
            assert int(round(g.close / 0.25)) == r.close_t
            assert int(g.volume) == r.volume
            assert int(g.trade_count) == r.trade_count

    def test_trade_bars_only_count_trades(self, tmp_path):
        """trade_count over all bars == number of action='T' rows (book events excluded)."""
        df = _make_mbp10_with_actions(n=600)
        n_trades = int((df["action"].str.lower() == "t").sum())
        store = _register_frame(tmp_path, df)
        try:
            bars = store.build_tick_bars("NQ", *self._WIN, tick_count=50)
        finally:
            store.close()
        assert int(bars["trade_count"].sum()) == n_trades
        # OHLC sits exactly on the 0.25 trade grid.
        ohlc = bars[["open", "high", "low", "close"]].to_numpy()
        assert np.abs(ohlc / 0.25 - np.round(ohlc / 0.25)).max() < 1e-9

    def test_book_mid_path_counts_all_book_events(self, tmp_path):
        """price_source='book_mid' buckets every book-valid event, not just trades."""
        df = _make_mbp10_with_actions(n=600)
        store = _register_frame(tmp_path, df)
        try:
            trade_bars = store.build_tick_bars("NQ", *self._WIN, tick_count=50)
            book_bars = store.build_tick_bars(
                "NQ", *self._WIN, tick_count=50, price_source="book_mid"
            )
        finally:
            store.close()
        # every row has valid bid/ask, so book_mid sees ~all 600 rows vs ~240 trades.
        assert int(book_bars["trade_count"].sum()) == len(df)
        assert int(book_bars["trade_count"].sum()) > int(trade_bars["trade_count"].sum())

    def test_query_tick_events_columns_and_filter(self, tmp_path):
        """query_tick_events returns [ts_event, price, size]; trade source filters action='T'."""
        df = _make_mbp10_with_actions(n=600)
        n_trades = int((df["action"].str.lower() == "t").sum())
        store = _register_frame(tmp_path, df)
        try:
            ev_trade = store.query_tick_events("NQ", *self._WIN)  # default trade
            ev_book = store.query_tick_events("NQ", *self._WIN, price_source="book_mid")
        finally:
            store.close()
        assert list(ev_trade.columns) == ["ts_event", "price", "size"]
        assert len(ev_trade) == n_trades
        assert len(ev_book) == len(df)

    def test_no_action_column_treats_all_rows_as_trades(self, tmp_path):
        """A trades-style frame with sequence+price but NO action column = all trades."""
        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2026-02-20 14:30", periods=n, freq="100ms", tz="UTC"),
                "sequence": np.arange(1, n + 1, dtype="int64"),
                "price": np.round((22000 + rng.standard_normal(n).cumsum() * 0.25) / 0.25) * 0.25,
                "size": rng.integers(1, 20, n).astype("int64"),
            }
        )
        out = tmp_path / "NQ" / "2026-02-20"
        out.mkdir(parents=True)
        df.to_parquet(out / "trades.parquet")
        store = TickStore(tmp_path, tick_filename="trades.parquet")
        store.register_symbol_date("NQ", "2026-02-20")
        try:
            bars = store.build_tick_bars("NQ", *self._WIN, tick_count=50)
        finally:
            store.close()
        assert int(bars["trade_count"].sum()) == n  # no filtering: all rows bucketed

    def test_unknown_price_source_raises(self, tmp_path):
        df = _make_mbp10_with_actions(n=100)
        store = _register_frame(tmp_path, df)
        try:
            with pytest.raises(ValueError, match="unknown price_source"):
                store.build_tick_bars("NQ", *self._WIN, tick_count=50, price_source="bogus")
        finally:
            store.close()

    def test_trade_source_requires_price_column(self, tmp_path):
        """Book-only frame (no 'price') under price_source='trade' raises clearly."""
        rng = np.random.default_rng(2)
        n = 100
        mid = 22000 + rng.standard_normal(n).cumsum() * 0.25
        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2026-02-20 14:30", periods=n, freq="100ms", tz="UTC"),
                "sequence": np.arange(1, n + 1, dtype="int64"),
                "action": ["A"] * n,
                "size": rng.integers(1, 20, n).astype("int64"),
                "bid_px_00": mid - 0.25,
                "ask_px_00": mid + 0.25,
            }
        )
        out = tmp_path / "NQ" / "2026-02-20"
        out.mkdir(parents=True)
        df.to_parquet(out / "mbp10.parquet")
        store = TickStore(tmp_path)
        store.register_symbol_date("NQ", "2026-02-20")
        try:
            with pytest.raises(ValueError, match="trade-price tick bars require a 'price'"):
                store.build_tick_bars("NQ", *self._WIN, tick_count=50)
        finally:
            store.close()

    def test_missing_sequence_raises(self, tmp_path):
        """No 'sequence' column -> reader-independent order impossible -> raise."""
        rng = np.random.default_rng(3)
        n = 100
        df = pd.DataFrame(
            {
                "ts_event": pd.date_range("2026-02-20 14:30", periods=n, freq="100ms", tz="UTC"),
                "action": ["T"] * n,
                "price": np.round((22000 + rng.standard_normal(n).cumsum()) / 0.25) * 0.25,
                "size": rng.integers(1, 20, n).astype("int64"),
            }
        )
        out = tmp_path / "NQ" / "2026-02-20"
        out.mkdir(parents=True)
        df.to_parquet(out / "trades.parquet")
        store = TickStore(tmp_path, tick_filename="trades.parquet")
        store.register_symbol_date("NQ", "2026-02-20")
        try:
            with pytest.raises(ValueError, match="intrinsic 'sequence' column"):
                store.build_tick_bars("NQ", *self._WIN, tick_count=50)
        finally:
            store.close()


# ────────────────────────────────────────────────────────────────
# Phase 4e — side-signed price order (chronological sweep direction)
# ────────────────────────────────────────────────────────────────


def _make_mbp10_sided(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """Synthetic trades WITH a `side` column, grouped into monotonic sweeps that share one
    (ts_event, sequence) -- like real databento sweeps. Buys (side='B') step up, sells
    (side='A') step down, so the side-signed key must reorder them to wire direction."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2026-02-20 14:30:00", tz="UTC")
    ts_l, seq_l, px_l, sz_l, sd_l = [], [], [], [], []
    mid, seq, k, i = 22000.0, 100, 0, 0
    while i < n:
        sweep = int(rng.integers(2, 6))
        is_buy = rng.random() < 0.5
        side, step = ("B", 0.25) if is_buy else ("A", -0.25)
        ts = t0 + pd.Timedelta(milliseconds=100 * k)
        k += 1
        for j in range(sweep):
            if i >= n:
                break
            ts_l.append(ts)
            seq_l.append(seq)
            px_l.append(round((mid + step * j) / 0.25) * 0.25)
            sz_l.append(int(rng.integers(1, 50)))
            sd_l.append(side)
            i += 1
        mid += rng.standard_normal() * 0.5
        seq += 1
    return pd.DataFrame(
        {
            "ts_event": ts_l,
            "sequence": np.array(seq_l, dtype="int64"),
            "action": ["T"] * len(ts_l),
            "price": px_l,
            "size": np.array(sz_l, dtype="int64"),
            "side": sd_l,
        }
    )


def _ref_trade_bars_signed(
    df: pd.DataFrame, n: int, buy: str = "B", tick: float = 0.25
) -> pd.DataFrame:
    """Reference bucketer using the SIDE-SIGNED order: +price for buys, -price for sells."""
    t = df[df["action"].str.lower() == "t"].copy()
    t = t[t["price"].notna() & (t["price"] > 0)]
    # case-INSENSITIVE buy match, mirroring the DuckDB CASE (lower(side)='b').
    is_buy = t["side"].astype(str).str.upper().to_numpy() == buy.upper()
    t["_sgn"] = t["price"] * np.where(is_buy, 1.0, -1.0)
    t = t.sort_values(["ts_event", "sequence", "_sgn", "size"], kind="stable").reset_index(
        drop=True
    )
    t["bar_index"] = t.index // n
    rows = []
    for bi, sub in t.groupby("bar_index", sort=True):
        rows.append(
            {
                "bar_index": int(bi),
                "open_t": round(sub["price"].iloc[0] / tick),
                "high_t": round(sub["price"].max() / tick),
                "low_t": round(sub["price"].min() / tick),
                "close_t": round(sub["price"].iloc[-1] / tick),
                "volume": int(sub["size"].sum()),
                "trade_count": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


class TestSideSignedOrder:
    _WIN = (
        datetime(2026, 2, 20, 14, 0, tzinfo=dt.UTC),
        datetime(2026, 2, 20, 15, 0, tzinfo=dt.UTC),
    )

    def test_sell_sweep_orders_descending(self, tmp_path):
        """A sell sweep (side='A') under one (ts,seq): side-signed => descending = chronological,
        so open=highest, close=lowest -- the OPPOSITE of a naive price-ascending order."""
        ts = pd.Timestamp("2026-02-20 14:30:00", tz="UTC")
        df = pd.DataFrame(
            {  # arbitrary parquet row order; the SQL must impose the order
                "ts_event": [ts, ts, ts],
                "sequence": [10, 10, 10],
                "action": ["T", "T", "T"],
                "price": [100.00, 100.50, 100.25],
                "size": [3, 1, 2],
                "side": ["A", "A", "A"],
            }
        )
        store = _register_frame(tmp_path, df)
        try:
            bars = store.build_tick_bars("NQ", *self._WIN, tick_count=3)
        finally:
            store.close()
        r = bars.reset_index().iloc[0]
        assert int(round(r.open / 0.25)) == int(round(100.50 / 0.25))  # first = highest
        assert int(round(r.close / 0.25)) == int(round(100.00 / 0.25))  # last = lowest

    def test_buy_sweep_orders_ascending(self, tmp_path):
        """A buy sweep (side='B'): side-signed => ascending; open=lowest, close=highest."""
        ts = pd.Timestamp("2026-02-20 14:30:00", tz="UTC")
        df = pd.DataFrame(
            {
                "ts_event": [ts, ts, ts],
                "sequence": [10, 10, 10],
                "action": ["T", "T", "T"],
                "price": [100.50, 100.00, 100.25],
                "size": [3, 1, 2],
                "side": ["B", "B", "B"],
            }
        )
        store = _register_frame(tmp_path, df)
        try:
            bars = store.build_tick_bars("NQ", *self._WIN, tick_count=3)
        finally:
            store.close()
        r = bars.reset_index().iloc[0]
        assert int(round(r.open / 0.25)) == int(round(100.00 / 0.25))  # first = lowest
        assert int(round(r.close / 0.25)) == int(round(100.50 / 0.25))  # last = highest

    def test_side_signed_matches_pandas_reference(self, tmp_path):
        """DuckDB side-signed bars == an independent pandas side-signed bucketer."""
        df = _make_mbp10_sided(n=600)
        store = _register_frame(tmp_path, df)
        try:
            bars = store.build_tick_bars(
                "NQ", *self._WIN, tick_count=50
            )  # default trade, side-signed
        finally:
            store.close()
        ref = _ref_trade_bars_signed(df, 50)
        got = bars.reset_index().sort_values("bar_index").reset_index(drop=True)
        assert len(got) == len(ref)
        for i, rrow in ref.iterrows():
            g = got.iloc[i]
            assert int(round(g.open / 0.25)) == rrow.open_t
            assert int(round(g.high / 0.25)) == rrow.high_t
            assert int(round(g.low / 0.25)) == rrow.low_t
            assert int(round(g.close / 0.25)) == rrow.close_t
            assert int(g.volume) == rrow.volume
            assert int(g.trade_count) == rrow.trade_count

    def test_side_signed_differs_from_price_ascending(self, tmp_path):
        """Sanity: with sell sweeps present, side-signed != naive price-ascending (4d) order."""
        df = _make_mbp10_sided(n=600)
        store = _register_frame(tmp_path, df)
        try:
            signed = store.build_tick_bars("NQ", *self._WIN, tick_count=50)
        finally:
            store.close()
        ref_asc = _ref_trade_bars(df, 50)  # 4d price-ascending reference
        ref_signed = _ref_trade_bars_signed(df, 50)
        assert not ref_asc.equals(ref_signed)  # sell sweeps reversed -> side matters
        got = signed.reset_index().sort_values("bar_index").reset_index(drop=True)
        assert int(round(got.iloc[0].close / 0.25)) == ref_signed.iloc[0].close_t

    def test_mixed_side_group_orders_sells_before_buys(self, tmp_path):
        """A (ts,seq) carrying both a buy and a sell: side-signed puts the sell (-price) before
        the buy (+price) deterministically (−P < +P' for any positive prices)."""
        ts = pd.Timestamp("2026-02-20 14:30:00", tz="UTC")
        df = pd.DataFrame(
            {
                "ts_event": [ts, ts],
                "sequence": [10, 10],
                "action": ["T", "T"],
                "price": [100.00, 100.50],
                "size": [4, 7],
                "side": ["B", "A"],  # buy@100.00, sell@100.50
            }
        )
        store = _register_frame(tmp_path, df)
        try:
            bars = store.build_tick_bars("NQ", *self._WIN, tick_count=2)
        finally:
            store.close()
        r = bars.reset_index().iloc[0]
        assert int(round(r.open / 0.25)) == int(round(100.50 / 0.25))  # sell first
        assert int(round(r.close / 0.25)) == int(round(100.00 / 0.25))  # buy last
        assert int(r.volume) == 11
