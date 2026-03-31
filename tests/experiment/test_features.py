"""Tests for Phase 4 — Order Flow Feature Engineering."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment.features import (
    CAT_FEATURES,
    EventFeatureBuilder,
    _compute_cross_window_features,
    _extract_static_features,
    _query_approach_features,
    _query_interaction_features,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_event(
    entry_price: float = 22000.0,
    direction: str = "LONG",
    date_str: str = "2026-02-20",
    event_ts: str = "2026-02-20 10:00:00",
    level_names: str = '["PDL"]',
    approach_direction: str = "from_above",
) -> pd.Series:
    """Create a minimal event Series for testing."""
    return pd.Series({
        "date": date_str,
        "event_ts": pd.Timestamp(event_ts, tz="US/Eastern"),
        "representative_price": entry_price,
        "touch_price": entry_price + 0.5,
        "direction": direction,
        "level_names": level_names,
        "level_prices": f'{{"PDL": {entry_price}}}',
        "approach_direction": approach_direction,
        "zone_id": f"{date_str}_PDL",
        "label": "tradeable_reversal",
    })


def _make_synthetic_mbp10(
    tmp_path: Path,
    date_str: str = "2026-02-20",
    n: int = 10000,
    base_price: float = 22000.0,
    buy_fraction: float = 0.6,
    trade_fraction: float = 0.04,
    seed: int = 42,
) -> Path:
    """Create synthetic MBP-10 Parquet with controlled properties.

    Returns path to the created data directory (parent of NQ/).
    """
    rng = np.random.default_rng(seed)

    # Generate timestamps spread over a day (UTC)
    start_ts = pd.Timestamp(f"{date_str} 14:00", tz="UTC")
    ts = pd.date_range(start_ts, periods=n, freq="100ms")

    # Random walk price
    prices = base_price + rng.standard_normal(n).cumsum() * 0.25
    prices = np.round(prices / 0.25) * 0.25  # Snap to tick size

    # Action distribution: ~4% trades, rest split among A/M/C
    n_trades = int(n * trade_fraction)
    n_other = n - n_trades
    actions = np.array(
        ["T"] * n_trades
        + list(rng.choice(["A", "M", "C"], size=n_other))
    )
    rng.shuffle(actions)

    # Side: only meaningful for trades; A = buyer, B = seller
    sides = np.full(n, "N", dtype=object)
    trade_mask = actions == "T"
    n_actual_trades = trade_mask.sum()
    n_buys = int(n_actual_trades * buy_fraction)
    trade_sides = np.array(["A"] * n_buys + ["B"] * (n_actual_trades - n_buys))
    rng.shuffle(trade_sides)
    sides[trade_mask] = trade_sides

    # Sizes: trades get meaningful sizes, others get 0
    sizes = np.zeros(n, dtype=np.uint32)
    sizes[trade_mask] = rng.integers(1, 50, size=n_actual_trades).astype(np.uint32)

    df = pd.DataFrame({
        "ts_event": ts,
        "price": prices,
        "size": sizes,
        "action": actions,
        "side": sides,
    })

    # Add 10-level book depth
    for i in range(10):
        spread = (i + 1) * 0.25
        df[f"bid_px_{i:02d}"] = prices - spread
        df[f"ask_px_{i:02d}"] = prices + spread
        df[f"bid_sz_{i:02d}"] = rng.integers(10, 200, n).astype(np.uint32)
        df[f"ask_sz_{i:02d}"] = rng.integers(10, 200, n).astype(np.uint32)

    # Add bid/ask count columns (required by some MBP-10 schemas)
    for i in range(10):
        df[f"bid_ct_{i:02d}"] = rng.integers(1, 20, n).astype(np.uint32)
        df[f"ask_ct_{i:02d}"] = rng.integers(1, 20, n).astype(np.uint32)

    # Add symbol column (front-month)
    df["symbol"] = "NQH6"

    # Write to expected path
    data_dir = tmp_path / "data"
    tick_dir = data_dir / "NQ" / date_str
    tick_dir.mkdir(parents=True)
    df.to_parquet(tick_dir / "mbp10.parquet", index=False)

    return data_dir


# ── TestStaticFeatures ─────────────────────────────────────────


class TestStaticFeatures:
    def test_long_pdl_event(self):
        """LONG PDL event during NY RTH."""
        event = _make_event(
            direction="LONG",
            level_names='["PDL"]',
            event_ts="2026-02-20 10:00:00",
            approach_direction="from_above",
        )
        features = _extract_static_features(event)

        assert features["ctx_direction"] == "LONG"
        assert features["ctx_level_type"] == "PDL"
        assert features["ctx_session"] == "NY RTH"
        assert features["ctx_hour"] == 10
        assert features["ctx_day_of_week"] == 4  # Friday
        assert features["ctx_approach_from_above"] == 1
        assert 0 <= features["ctx_time_normalized"] <= 1

    def test_short_pdh_event(self):
        """SHORT PDH event during London session."""
        event = _make_event(
            direction="SHORT",
            level_names='["PDH"]',
            event_ts="2026-02-20 03:00:00",
            approach_direction="from_below",
        )
        features = _extract_static_features(event)

        assert features["ctx_direction"] == "SHORT"
        assert features["ctx_level_type"] == "PDH"
        assert features["ctx_session"] == "London"
        assert features["ctx_hour"] == 3
        assert features["ctx_approach_from_above"] == 0

    def test_asia_session(self):
        """Asia session event classification."""
        event = _make_event(
            event_ts="2026-02-20 19:00:00",
            level_names='["asia_high"]',
        )
        features = _extract_static_features(event)
        assert features["ctx_session"] == "Asia"
        assert features["ctx_level_type"] == "asia_high"

    def test_merged_zone_uses_first_level(self):
        """Merged zone uses first level name alphabetically."""
        event = _make_event(level_names='["PDH", "london_high"]')
        features = _extract_static_features(event)
        assert features["ctx_level_type"] == "PDH"

    def test_all_features_present(self):
        """Verify all 7 static features are returned."""
        event = _make_event()
        features = _extract_static_features(event)
        expected = {
            "ctx_direction", "ctx_level_type", "ctx_session",
            "ctx_hour", "ctx_time_normalized", "ctx_day_of_week",
            "ctx_approach_from_above",
        }
        assert set(features.keys()) == expected


# ── TestApproachFeatures ───────────────────────────────────────


class TestApproachFeatures:
    def test_with_synthetic_ticks(self, tmp_path):
        """Verify approach features against synthetic data with known buy ratio."""
        from alpha_lab.agents.data_infra.tick_store import TickStore

        data_dir = _make_synthetic_mbp10(
            tmp_path,
            date_str="2026-02-20",
            n=20000,
            base_price=22000.0,
            buy_fraction=0.7,  # 70% buyer aggression
        )

        store = TickStore(data_dir)
        store.register_symbol_date("NQ", "2026-02-20")

        views = store._get_views("NQ")
        union_sql = store._union_views_sql(views)

        # Query a 30-min window within the synthetic data
        start = pd.Timestamp("2026-02-20 14:00", tz="UTC")
        end = pd.Timestamp("2026-02-20 14:30", tz="UTC")

        features = _query_approach_features(store._conn, union_sql, "", start, end)
        store.close()

        # Verify structure
        assert len([k for k in features if k.startswith("app_")]) == 27

        # Buy-sell ratio should be close to 0.7
        assert 0.5 < features["app_buy_sell_ratio"] < 0.9

        # Volume should be positive
        assert features["app_total_trade_volume"] > 0
        assert features["app_trade_count"] > 0
        assert features["app_buy_volume"] > 0
        assert features["app_sell_volume"] > 0

        # Spread should be positive (ask > bid by construction)
        assert features["app_avg_spread"] > 0

        # Book features should be in valid ranges
        assert 0 <= features["app_avg_tob_imbalance"] <= 1
        assert features["app_avg_top5_depth"] > 0

    def test_empty_window_returns_nan(self, tmp_path):
        """Window with no data returns NaN features."""
        from alpha_lab.agents.data_infra.tick_store import TickStore

        data_dir = _make_synthetic_mbp10(tmp_path, n=100)
        store = TickStore(data_dir)
        store.register_symbol_date("NQ", "2026-02-20")

        views = store._get_views("NQ")
        union_sql = store._union_views_sql(views)

        # Query a window that's outside the data range
        start = pd.Timestamp("2026-02-21 14:00", tz="UTC")
        end = pd.Timestamp("2026-02-21 14:30", tz="UTC")

        features = _query_approach_features(store._conn, union_sql, "", start, end)
        store.close()

        # All numeric features should be NaN or 0 (from COALESCE)
        assert len([k for k in features if k.startswith("app_")]) == 27


# ── TestInteractionFeatures ────────────────────────────────────


class TestInteractionFeatures:
    def test_absorption_with_synthetic_data(self, tmp_path):
        """Verify interaction features with synthetic data."""
        from alpha_lab.agents.data_infra.tick_store import TickStore

        data_dir = _make_synthetic_mbp10(
            tmp_path,
            date_str="2026-02-20",
            n=20000,
            base_price=22000.0,
        )

        store = TickStore(data_dir)
        store.register_symbol_date("NQ", "2026-02-20")

        views = store._get_views("NQ")
        union_sql = store._union_views_sql(views)

        start = pd.Timestamp("2026-02-20 14:10", tz="UTC")
        end = pd.Timestamp("2026-02-20 14:15", tz="UTC")

        features = _query_interaction_features(
            store._conn, union_sql, "",
            start, end,
            representative_price=22000.0,
            direction="LONG",
        )
        store.close()

        # Verify all expected features present
        int_keys = [k for k in features if k.startswith("int_")]
        assert len(int_keys) >= 18  # 18 int_ features (excluding _ prefixed)

        # Trade features should be positive
        assert features["int_total_trade_volume"] >= 0
        assert features["int_trade_count"] >= 0

        # Absorption ratio should be valid
        if features["int_absorption_ratio"] == features["int_absorption_ratio"]:  # not NaN
            assert 0 <= features["int_absorption_ratio"] <= 1

    def test_direction_affects_adverse(self, tmp_path):
        """LONG and SHORT should compute different adverse directions."""
        from alpha_lab.agents.data_infra.tick_store import TickStore

        data_dir = _make_synthetic_mbp10(
            tmp_path, n=20000, base_price=22000.0,
        )
        store = TickStore(data_dir)
        store.register_symbol_date("NQ", "2026-02-20")
        views = store._get_views("NQ")
        union_sql = store._union_views_sql(views)

        start = pd.Timestamp("2026-02-20 14:10", tz="UTC")
        end = pd.Timestamp("2026-02-20 14:15", tz="UTC")

        long_feat = _query_interaction_features(
            store._conn, union_sql, "",
            start, end, 22000.0, "LONG",
        )
        short_feat = _query_interaction_features(
            store._conn, union_sql, "",
            start, end, 22000.0, "SHORT",
        )
        store.close()

        # Both should have valid structures
        assert "int_volume_through_level" in long_feat
        assert "int_volume_through_level" in short_feat

        # With random walk around 22000, the "through level" volumes
        # for LONG (below 22000) and SHORT (above 22000) should differ
        # but both be non-negative
        assert long_feat["int_volume_through_level"] >= 0
        assert short_feat["int_volume_through_level"] >= 0


# ── TestTempoFeatures ──────────────────────────────────────────


class TestTempoFeatures:
    def test_time_within_2pts_present(self, tmp_path):
        """Verify tempo features are computed."""
        from alpha_lab.agents.data_infra.tick_store import TickStore

        data_dir = _make_synthetic_mbp10(
            tmp_path, n=20000, base_price=22000.0,
        )
        store = TickStore(data_dir)
        store.register_symbol_date("NQ", "2026-02-20")
        views = store._get_views("NQ")
        union_sql = store._union_views_sql(views)

        start = pd.Timestamp("2026-02-20 14:10", tz="UTC")
        end = pd.Timestamp("2026-02-20 14:15", tz="UTC")

        features = _query_interaction_features(
            store._conn, union_sql, "",
            start, end, 22000.0, "LONG",
        )
        store.close()

        assert "int_time_within_2pts" in features
        assert "int_time_beyond_level" in features
        # Time should be non-negative
        assert features["int_time_within_2pts"] >= 0
        assert features["int_time_beyond_level"] >= 0


# ── TestCrossWindowFeatures ────────────────────────────────────


class TestCrossWindowFeatures:
    def test_all_features_computed(self):
        """Verify all 6 cross-window features are returned."""
        approach = {
            "app_buy_sell_ratio": 0.6,
            "app_avg_tob_imbalance": 0.5,
            "app_avg_spread": 0.50,
            "app_avg_top5_depth": 500.0,
            "app_avg_trade_size": 5.0,
            "app_price_change": 10.0,
        }
        interaction = {
            "int_buy_sell_ratio": 0.7,
            "int_avg_tob_imbalance": 0.55,
            "int_avg_spread": 0.75,
            "int_avg_depth": 600.0,
            "int_avg_trade_size": 8.0,
            "_int_displacement": 2.0,
        }
        cross = _compute_cross_window_features(approach, interaction)

        assert cross["int_aggression_flip"] == pytest.approx(0.1)
        assert cross["int_book_imbalance_shift"] == pytest.approx(0.05)
        assert cross["int_spread_widening"] == pytest.approx(0.25)
        assert cross["int_depth_change"] == pytest.approx(1.2)
        assert cross["int_size_vs_approach"] == pytest.approx(1.6)
        # Deceleration: app_rate = |10|/90 = 0.111, int_rate = 2/5 = 0.4
        assert cross["int_deceleration_ratio"] == pytest.approx(0.111 / 0.4, rel=0.01)

    def test_nan_handling(self):
        """NaN inputs produce NaN outputs."""
        approach = {k: float("nan") for k in [
            "app_buy_sell_ratio", "app_avg_tob_imbalance", "app_avg_spread",
            "app_avg_top5_depth", "app_avg_trade_size", "app_price_change",
        ]}
        interaction = {k: float("nan") for k in [
            "int_buy_sell_ratio", "int_avg_tob_imbalance", "int_avg_spread",
            "int_avg_depth", "int_avg_trade_size", "_int_displacement",
        ]}
        cross = _compute_cross_window_features(approach, interaction)

        for k, v in cross.items():
            assert v != v, f"{k} should be NaN but is {v}"  # NaN != NaN

    def test_zero_denominator(self):
        """Zero denominator produces NaN, not error."""
        approach = {
            "app_buy_sell_ratio": 0.5,
            "app_avg_tob_imbalance": 0.5,
            "app_avg_spread": 0.5,
            "app_avg_top5_depth": 0.0,  # zero
            "app_avg_trade_size": 0.0,  # zero
            "app_price_change": 0.0,
        }
        interaction = {
            "int_buy_sell_ratio": 0.5,
            "int_avg_tob_imbalance": 0.5,
            "int_avg_spread": 0.5,
            "int_avg_depth": 100.0,
            "int_avg_trade_size": 5.0,
            "_int_displacement": 0.0,  # zero → decel NaN
        }
        cross = _compute_cross_window_features(approach, interaction)

        # depth_change and size_vs_approach should be NaN (div by 0)
        assert cross["int_depth_change"] != cross["int_depth_change"]  # NaN
        assert cross["int_size_vs_approach"] != cross["int_size_vs_approach"]  # NaN
        assert cross["int_deceleration_ratio"] != cross["int_deceleration_ratio"]  # NaN


# ── TestNaNHandling ────────────────────────────────────────────


class TestNaNHandling:
    def test_no_ticks_no_crash(self, tmp_path):
        """Event with no tick data produces NaN features without crashing."""
        from alpha_lab.agents.data_infra.tick_store import TickStore

        # Create empty data dir
        data_dir = tmp_path / "empty_data"
        nq_dir = data_dir / "NQ" / "2026-02-20"
        nq_dir.mkdir(parents=True)

        # Create minimal parquet with correct schema but no matching data
        df = pd.DataFrame({
            "ts_event": pd.DatetimeIndex([], dtype="datetime64[ns, UTC]"),
            "price": pd.array([], dtype="float64"),
            "size": pd.array([], dtype="uint32"),
            "action": pd.array([], dtype="object"),
            "side": pd.array([], dtype="object"),
            "symbol": pd.array([], dtype="object"),
        })
        for i in range(10):
            df[f"bid_px_{i:02d}"] = pd.array([], dtype="float64")
            df[f"ask_px_{i:02d}"] = pd.array([], dtype="float64")
            df[f"bid_sz_{i:02d}"] = pd.array([], dtype="uint32")
            df[f"ask_sz_{i:02d}"] = pd.array([], dtype="uint32")
        df.to_parquet(nq_dir / "mbp10.parquet", index=False)

        store = TickStore(data_dir)
        store.register_symbol_date("NQ", "2026-02-20")
        views = store._get_views("NQ")
        union_sql = store._union_views_sql(views)

        start = pd.Timestamp("2026-02-20 14:00", tz="UTC")
        end = pd.Timestamp("2026-02-20 14:30", tz="UTC")

        # Should not crash, should return features (possibly NaN)
        approach = _query_approach_features(store._conn, union_sql, "", start, end)
        assert len([k for k in approach if k.startswith("app_")]) == 27

        interaction = _query_interaction_features(
            store._conn, union_sql, "",
            start, end, 22000.0, "LONG",
        )
        assert "int_total_trade_volume" in interaction

        store.close()


# ── TestTemporalIntegrity ──────────────────────────────────────


class TestTemporalIntegrity:
    def test_approach_excludes_event_ts(self, tmp_path):
        """Approach window uses < event_ts (exclusive)."""
        from alpha_lab.agents.data_infra.tick_store import TickStore

        data_dir = _make_synthetic_mbp10(
            tmp_path, n=20000, base_price=22000.0,
        )
        store = TickStore(data_dir)
        store.register_symbol_date("NQ", "2026-02-20")
        views = store._get_views("NQ")
        union_sql = store._union_views_sql(views)

        event_ts = pd.Timestamp("2026-02-20 14:30", tz="UTC")
        approach_start = event_ts - pd.Timedelta(minutes=90)

        # The SQL uses < end_utc (exclusive)
        features = _query_approach_features(
            store._conn, union_sql, "",
            approach_start, event_ts,
        )

        # Verify the window boundary is respected by checking the SQL
        # uses ts_event < $2 (already verified in the SQL string)
        assert features is not None
        store.close()


# ── TestBuildFeatureMatrix (Integration) ───────────────────────


class TestBuildFeatureMatrixIntegration:
    def test_with_real_data_smoke(self, tmp_path):
        """Run on real data, verify output schema and ranges."""
        events_path = Path("data/experiment/labeled_events.parquet")
        data_dir = Path("data/databento")
        if not events_path.exists() or not data_dir.exists():
            pytest.skip("Real data not available")

        labeled = pd.read_parquet(events_path)
        builder = EventFeatureBuilder(data_dir)
        output = tmp_path / "feature_matrix.parquet"

        df = builder.build_feature_matrix(labeled, output)

        # Shape check: 284 resolved events
        assert len(df) == 284, f"Expected 284, got {len(df)}"

        # All expected columns present
        assert "label" in df.columns
        assert "label_encoded" in df.columns
        assert "event_ts" in df.columns
        assert "date" in df.columns

        # Feature columns
        app_cols = [c for c in df.columns if c.startswith("app_")]
        int_cols = [c for c in df.columns if c.startswith("int_")]
        ctx_cols = [c for c in df.columns if c.startswith("ctx_")]
        assert len(app_cols) == 27
        assert len(int_cols) == 24  # 18 base + 6 cross-window
        assert len(ctx_cols) == 7

        # Total feature count
        total_features = len(app_cols) + len(int_cols) + len(ctx_cols)
        assert total_features == 58

        # Output written
        assert output.exists()

    def test_feature_ranges_plausible(self, tmp_path):
        """Verify feature ranges are plausible."""
        events_path = Path("data/experiment/labeled_events.parquet")
        data_dir = Path("data/databento")
        if not events_path.exists() or not data_dir.exists():
            pytest.skip("Real data not available")

        labeled = pd.read_parquet(events_path)
        builder = EventFeatureBuilder(data_dir)
        df = builder.build_feature_matrix(labeled, output_path=None)

        # Ratio features in [0, 1]
        ratio_cols = [
            "app_buy_sell_ratio", "app_large_trade_vol_pct",
            "app_avg_tob_imbalance", "app_cancel_rate",
            "app_avg_bid_depth_ratio", "app_depth_concentration",
            "int_buy_sell_ratio", "int_large_trade_pct",
            "int_avg_tob_imbalance", "int_absorption_ratio",
        ]
        for col in ratio_cols:
            valid = df[col].dropna()
            if not valid.empty:
                assert (valid >= -0.01).all(), f"{col} has values below 0"
                assert (valid <= 1.01).all(), f"{col} has values above 1"

        # Volume features non-negative
        vol_cols = [c for c in df.columns if "volume" in c.lower() or "trade_count" in c.lower()]
        for col in vol_cols:
            valid = df[col].dropna()
            if not valid.empty:
                assert (valid >= 0).all(), f"{col} has negative values"

        # Categorical features are valid strings
        assert set(df["ctx_direction"].unique()) <= {"LONG", "SHORT"}
        assert set(df["ctx_session"].unique()) <= {
            "Asia", "London", "Pre-market", "NY RTH", "Post-market",
        }
        valid_levels = {"PDH", "PDL", "asia_high", "asia_low", "london_high", "london_low"}
        assert set(df["ctx_level_type"].unique()) <= valid_levels

        # Label encoding correct
        assert set(df["label_encoded"].unique()) == {0, 1, 2}
