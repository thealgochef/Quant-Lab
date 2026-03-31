"""Tests for ML pipeline Phase 2: feature extraction and dataset builder."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ml.config import (
    ExtremaConfig,
    MLPipelineConfig,
)
from alpha_lab.agents.data_infra.ml.dataset_builder import ExtremaDatasetBuilder
from alpha_lab.agents.data_infra.ml.extrema_detection import Extremum
from alpha_lab.agents.data_infra.ml.features_microstructure import (
    extract_pl_features,
)
from alpha_lab.agents.data_infra.ml.features_momentum import (
    _compute_tick_macd,
    _compute_tick_rsi,
    extract_ms_features,
)
from alpha_lab.agents.data_infra.ml.features_signals import (
    _find_bar_index,
    extract_signal_features,
    extract_signal_features_batch,
)
from alpha_lab.agents.data_infra.tick_store import TickStore
from alpha_lab.core.contracts import SignalBundle, SignalVector

# ────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────


def _make_synthetic_ticks(
    date_str: str = "2026-02-20",
    n: int = 1000,
    base_price: float = 22000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic MBP-10 tick DataFrame (matches test_tick_store)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(
        f"{date_str} 09:30", periods=n, freq="100ms", tz="UTC",
    )
    prices = base_price + rng.standard_normal(n).cumsum() * 0.25

    df = pd.DataFrame({
        "ts_event": ts,
        "price": prices,
        "size": rng.integers(1, 50, n),
    })

    for i in range(10):
        spread = (i + 1) * 0.25
        df[f"bid_px_{i:02d}"] = prices - spread
        df[f"ask_px_{i:02d}"] = prices + spread
        df[f"bid_sz_{i:02d}"] = rng.integers(10, 200, n).astype(float)
        df[f"ask_sz_{i:02d}"] = rng.integers(10, 200, n).astype(float)

    return df


def _make_test_extremum(
    index: int = 100,
    price: float = 22010.0,
    etype: str = "peak",
) -> Extremum:
    return Extremum(
        index=index,
        timestamp=pd.Timestamp("2026-02-20 09:30:10", tz="UTC"),
        price=price,
        extremum_type=etype,
        prominence=5.0,
        width=100.0,
    )


def _make_test_signal_bundle() -> SignalBundle:
    """Create a minimal SignalBundle with one signal for testing."""
    idx = pd.date_range("2026-02-20 09:30", periods=100, freq="5min", tz="UTC")
    direction = pd.Series(
        np.random.default_rng(42).choice([-1, 0, 1], size=100), index=idx,
    )
    strength = pd.Series(
        np.random.default_rng(42).random(100), index=idx,
    )
    formation_idx = pd.Series(range(100), index=idx)

    sv = SignalVector(
        signal_id="SIG_EMA_CONFLUENCE_5m_v1",
        category="ema_confluence",
        timeframe="5m",
        version=1,
        direction=direction,
        strength=strength,
        formation_idx=formation_idx,
    )

    return SignalBundle(
        instrument="NQ",
        signals=[sv],
        timeframes_covered=["5m"],
        total_signals=1,
        generation_timestamp="2026-02-20T10:00:00",
    )


@pytest.fixture
def tick_data_dir(tmp_path) -> Path:
    """Create temp dir with synthetic Parquet tick files."""
    for date_str in ("2026-02-20",):
        ticks = _make_synthetic_ticks(date_str=date_str, n=2000)
        out_dir = tmp_path / "NQ" / date_str
        out_dir.mkdir(parents=True)
        ticks.to_parquet(out_dir / "mbp10.parquet")
    return tmp_path


@pytest.fixture
def tick_store(tick_data_dir) -> TickStore:
    store = TickStore(tick_data_dir)
    store.register_symbol_date("NQ", date(2026, 2, 20))
    yield store
    store.close()


# ────────────────────────────────────────────────────────────────
# PL Microstructure Features
# ────────────────────────────────────────────────────────────────


class TestPLFeatures:
    """Test PL (Price Level) microstructure feature extraction."""

    def test_basic_extraction(self):
        ticks = _make_synthetic_ticks(n=200)
        ext = _make_test_extremum(index=100)
        features = extract_pl_features(ext, ticks.iloc[50:101])

        assert "pl_total_bid_vol" in features
        assert "pl_total_ask_vol" in features
        assert "pl_total_vol" in features
        assert features["pl_total_vol"] > 0

    def test_trade_features(self):
        ticks = _make_synthetic_ticks(n=200)
        ext = _make_test_extremum(index=100)
        features = extract_pl_features(ext, ticks.iloc[50:101])

        assert "pl_trade_count" in features
        assert "pl_total_trade_vol" in features
        assert "pl_max_trade_size" in features
        assert features["pl_trade_count"] > 0

    def test_book_ratio_features(self):
        ticks = _make_synthetic_ticks(n=200)
        ext = _make_test_extremum(index=100)
        features = extract_pl_features(ext, ticks.iloc[50:101])

        assert "pl_bid_ask_ratio" in features
        assert 0.0 <= features["pl_bid_ask_ratio"] <= 1.0
        assert "pl_spread" in features
        assert features["pl_spread"] > 0

    def test_depth_shape_features(self):
        ticks = _make_synthetic_ticks(n=200)
        ext = _make_test_extremum(index=100)
        features = extract_pl_features(ext, ticks.iloc[50:101])

        assert "pl_bid_top3_concentration" in features
        assert "pl_ask_top3_concentration" in features
        assert 0.0 <= features["pl_bid_top3_concentration"] <= 1.0

    def test_morphology_features(self):
        ext_peak = _make_test_extremum(etype="peak")
        ext_trough = _make_test_extremum(etype="trough")
        ticks = _make_synthetic_ticks(n=200)

        f_peak = extract_pl_features(ext_peak, ticks.iloc[:101])
        f_trough = extract_pl_features(ext_trough, ticks.iloc[:101])

        assert f_peak["pl_extremum_type"] == 1.0
        assert f_trough["pl_extremum_type"] == 0.0
        assert f_peak["pl_prominence"] == 5.0

    def test_empty_ticks(self):
        ext = _make_test_extremum()
        features = extract_pl_features(ext, pd.DataFrame())
        assert features == {}

    def test_feature_count(self):
        """Should produce ~24 features from full MBP-10 data."""
        ticks = _make_synthetic_ticks(n=200)
        ext = _make_test_extremum(index=100)
        features = extract_pl_features(ext, ticks.iloc[50:101])
        assert len(features) >= 15


# ────────────────────────────────────────────────────────────────
# MS Momentum Features
# ────────────────────────────────────────────────────────────────


class TestMSFeatures:
    """Test MS (Market Shift) momentum feature extraction."""

    def test_basic_extraction(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(22000.0 + rng.standard_normal(500).cumsum() * 0.25)
        ext = _make_test_extremum(index=300)

        features = extract_ms_features(ext, prices)
        assert len(features) > 0

    def test_rsi_features(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(22000.0 + rng.standard_normal(500).cumsum() * 0.25)
        ext = _make_test_extremum(index=300)

        features = extract_ms_features(ext, prices)
        rsi_keys = [k for k in features if k.startswith("ms_rsi_")]
        assert len(rsi_keys) > 0
        for k in rsi_keys:
            assert 0.0 <= features[k] <= 100.0

    def test_macd_features(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(22000.0 + rng.standard_normal(500).cumsum() * 0.25)
        ext = _make_test_extremum(index=300)

        features = extract_ms_features(ext, prices)
        macd_keys = [k for k in features if k.startswith("ms_macd_")]
        assert len(macd_keys) > 0

    def test_price_velocity(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(22000.0 + rng.standard_normal(500).cumsum() * 0.25)
        ext = _make_test_extremum(index=300)

        features = extract_ms_features(ext, prices)
        vel_keys = [k for k in features if k.startswith("ms_price_velocity_")]
        assert len(vel_keys) > 0

    def test_volume_momentum(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(22000.0 + rng.standard_normal(500).cumsum() * 0.25)
        volumes = pd.Series(rng.integers(1, 50, 500).astype(float))
        ext = _make_test_extremum(index=300)

        features = extract_ms_features(ext, prices, volumes)
        assert "ms_volume_momentum" in features
        assert features["ms_volume_momentum"] > 0

    def test_short_data(self):
        """Extremum near start of series should still produce some features."""
        prices = pd.Series([22000.0] * 5)
        ext = _make_test_extremum(index=2)
        features = extract_ms_features(ext, prices)
        assert isinstance(features, dict)

    def test_uptick_downtick(self):
        rng = np.random.default_rng(42)
        prices = pd.Series(22000.0 + rng.standard_normal(500).cumsum() * 0.25)
        ext = _make_test_extremum(index=300)

        features = extract_ms_features(ext, prices)
        assert "ms_uptick_fraction" in features
        assert "ms_downtick_fraction" in features
        # Fractions should sum to ~1.0
        total = features["ms_uptick_fraction"] + features["ms_downtick_fraction"]
        assert abs(total - 1.0) < 0.01


class TestRSI:
    """Test the tick-level RSI computation."""

    def test_rsi_uptrend(self):
        """Strong uptrend → RSI near 100."""
        prices = np.linspace(100, 200, 100)
        rsi = _compute_tick_rsi(prices, 14)
        assert rsi > 70

    def test_rsi_downtrend(self):
        """Strong downtrend → RSI near 0."""
        prices = np.linspace(200, 100, 100)
        rsi = _compute_tick_rsi(prices, 14)
        assert rsi < 30

    def test_rsi_flat(self):
        """Flat prices → RSI near 50."""
        prices = np.full(100, 22000.0)
        rsi = _compute_tick_rsi(prices, 14)
        assert 40 <= rsi <= 60

    def test_rsi_insufficient_data(self):
        prices = np.array([100.0, 101.0])
        rsi = _compute_tick_rsi(prices, 14)
        assert np.isnan(rsi)


class TestMACD:
    """Test the tick-level MACD computation."""

    def test_macd_uptrend(self):
        """Uptrend → positive MACD."""
        prices = np.linspace(100, 200, 200)
        macd = _compute_tick_macd(prices, 12, 26)
        assert macd > 0

    def test_macd_downtrend(self):
        """Downtrend → negative MACD."""
        prices = np.linspace(200, 100, 200)
        macd = _compute_tick_macd(prices, 12, 26)
        assert macd < 0

    def test_macd_insufficient_data(self):
        prices = np.array([100.0, 101.0])
        macd = _compute_tick_macd(prices, 12, 26)
        assert np.isnan(macd)


# ────────────────────────────────────────────────────────────────
# Signal Features
# ────────────────────────────────────────────────────────────────


class TestSignalFeatures:
    """Test signal detector feature extraction."""

    def test_basic_extraction(self):
        bundle = _make_test_signal_bundle()
        ext = _make_test_extremum()
        features = extract_signal_features(ext, bundle)

        assert "sig_ema_confluence_5m_direction" in features
        assert "sig_ema_confluence_5m_strength" in features

    def test_direction_in_valid_range(self):
        bundle = _make_test_signal_bundle()
        ext = _make_test_extremum()
        features = extract_signal_features(ext, bundle)

        d = features["sig_ema_confluence_5m_direction"]
        assert d in (-1.0, 0.0, 1.0)

    def test_strength_in_valid_range(self):
        bundle = _make_test_signal_bundle()
        ext = _make_test_extremum()
        features = extract_signal_features(ext, bundle)

        s = features["sig_ema_confluence_5m_strength"]
        assert 0.0 <= s <= 1.0

    def test_batch_extraction(self):
        bundle = _make_test_signal_bundle()
        extrema = [_make_test_extremum(index=i * 10) for i in range(5)]
        results = extract_signal_features_batch(extrema, bundle)

        assert len(results) == 5
        for features in results:
            assert "sig_ema_confluence_5m_direction" in features

    def test_empty_bundle(self):
        bundle = SignalBundle(
            instrument="NQ",
            signals=[],
            timeframes_covered=[],
            total_signals=0,
            generation_timestamp="2026-02-20T10:00:00",
        )
        ext = _make_test_extremum()
        features = extract_signal_features(ext, bundle)
        assert features == {}


class TestFindBarIndex:
    """Test the bar index lookup utility."""

    def test_exact_match(self):
        idx = pd.DatetimeIndex(
            pd.date_range("2026-02-20 09:30", periods=10, freq="5min", tz="UTC")
        )
        ts = pd.Timestamp("2026-02-20 09:35", tz="UTC")
        pos = _find_bar_index(idx, ts)
        assert pos == 1

    def test_between_bars(self):
        """Should return the bar just before the timestamp."""
        idx = pd.DatetimeIndex(
            pd.date_range("2026-02-20 09:30", periods=10, freq="5min", tz="UTC")
        )
        ts = pd.Timestamp("2026-02-20 09:37", tz="UTC")
        pos = _find_bar_index(idx, ts)
        assert pos == 1  # 09:35 is the latest bar before 09:37

    def test_before_first_bar(self):
        idx = pd.DatetimeIndex(
            pd.date_range("2026-02-20 09:30", periods=10, freq="5min", tz="UTC")
        )
        ts = pd.Timestamp("2026-02-20 09:00", tz="UTC")
        pos = _find_bar_index(idx, ts)
        assert pos is None

    def test_empty_index(self):
        idx = pd.DatetimeIndex([])
        ts = pd.Timestamp("2026-02-20 09:30")
        pos = _find_bar_index(idx, ts)
        assert pos is None

    def test_timezone_alignment(self):
        """Tz-naive timestamp against tz-aware index."""
        idx = pd.DatetimeIndex(
            pd.date_range("2026-02-20 09:30", periods=10, freq="5min", tz="UTC")
        )
        ts = pd.Timestamp("2026-02-20 09:35")  # No tz
        pos = _find_bar_index(idx, ts)
        assert pos == 1


# ────────────────────────────────────────────────────────────────
# Dataset Builder Integration
# ────────────────────────────────────────────────────────────────


class TestDatasetBuilder:
    """Test the ExtremaDatasetBuilder end-to-end."""

    def test_build_dataset(self, tick_store):
        config = MLPipelineConfig(
            extrema=ExtremaConfig(
                window_size=500,
                min_peak_width=20,
                max_peak_width=400,
                min_prominence_ticks=0.5,
            ),
        )
        builder = ExtremaDatasetBuilder(tick_store, config)
        df = builder.build_dataset(
            "NQ",
            datetime(2026, 2, 20, 9, 30),
            datetime(2026, 2, 20, 10, 0),
        )

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "tick_index" in df.columns
            assert "price" in df.columns
            assert "extremum_type" in df.columns
            assert "symbol" in df.columns
            assert df["symbol"].iloc[0] == "NQ"

            # Check for PL features
            pl_cols = [c for c in df.columns if c.startswith("pl_")]
            assert len(pl_cols) > 0

            # Check for MS features
            ms_cols = [c for c in df.columns if c.startswith("ms_")]
            assert len(ms_cols) > 0

    def test_build_dataset_daily(self, tick_store):
        config = MLPipelineConfig(
            extrema=ExtremaConfig(
                window_size=500,
                min_peak_width=20,
                max_peak_width=400,
                min_prominence_ticks=0.5,
            ),
        )
        builder = ExtremaDatasetBuilder(tick_store, config)
        df = builder.build_dataset_daily("NQ", ["2026-02-20"])

        assert isinstance(df, pd.DataFrame)

    def test_no_data(self, tick_store):
        builder = ExtremaDatasetBuilder(tick_store)
        df = builder.build_dataset(
            "ES",  # No ES data registered
            datetime(2026, 2, 20, 9, 30),
            datetime(2026, 2, 20, 10, 0),
        )
        assert df.empty

    def test_export_dataset(self, tick_store, tmp_path):
        config = MLPipelineConfig(
            extrema=ExtremaConfig(
                window_size=500,
                min_peak_width=20,
                max_peak_width=400,
                min_prominence_ticks=0.5,
            ),
        )
        builder = ExtremaDatasetBuilder(tick_store, config)
        df = builder.build_dataset(
            "NQ",
            datetime(2026, 2, 20, 9, 30),
            datetime(2026, 2, 20, 10, 0),
        )

        if not df.empty:
            out_path = tmp_path / "test_output" / "features.parquet"
            builder.export_dataset(df, out_path)
            assert out_path.exists()
            loaded = pd.read_parquet(out_path)
            assert len(loaded) == len(df)

    def test_with_signal_bundle(self, tick_store):
        config = MLPipelineConfig(
            extrema=ExtremaConfig(
                window_size=500,
                min_peak_width=20,
                max_peak_width=400,
                min_prominence_ticks=0.5,
            ),
        )
        bundle = _make_test_signal_bundle()
        builder = ExtremaDatasetBuilder(tick_store, config, signal_bundle=bundle)
        df = builder.build_dataset(
            "NQ",
            datetime(2026, 2, 20, 9, 30),
            datetime(2026, 2, 20, 10, 0),
        )

        if not df.empty:
            sig_cols = [c for c in df.columns if c.startswith("sig_")]
            assert len(sig_cols) > 0
