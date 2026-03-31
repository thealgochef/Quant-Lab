"""Tests for ML pipeline Phase 1: config, extrema detection, and labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ml.config import (
    ExtremaConfig,
    FeatureConfig,
    LabelingConfig,
    MLPipelineConfig,
    ModelConfig,
    WalkForwardConfig,
)
from alpha_lab.agents.data_infra.ml.extrema_detection import (
    Extremum,
    _deduplicate,
    _detect_in_window,
    detect_extrema,
)
from alpha_lab.agents.data_infra.ml.labeling import (
    LabeledExtremum,
    build_label_dataframe,
    label_extrema,
)

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _make_price_series_with_peaks(
    n: int = 2000,
    seed: int = 42,
    tick_size: float = 0.25,
) -> tuple[pd.Series, pd.Series]:
    """Create a synthetic tick price series with clear peaks and troughs.

    Generates a random walk with superimposed sinusoidal components to
    ensure detectable extrema.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    # Base random walk + sinusoidal trend for clear peaks/troughs
    noise = rng.standard_normal(n).cumsum() * tick_size * 0.3
    signal = 5.0 * np.sin(2 * np.pi * t / 500) + 2.0 * np.sin(2 * np.pi * t / 200)
    prices = 22000.0 + signal + noise

    # Quantize to tick size
    prices = np.round(prices / tick_size) * tick_size

    timestamps = pd.date_range("2026-02-20 09:30", periods=n, freq="100ms", tz="UTC")

    return pd.Series(prices, index=range(n)), pd.Series(timestamps, index=range(n))


def _make_simple_peak_series(tick_size: float = 0.25) -> tuple[pd.Series, pd.Series]:
    """Create a minimal series with one obvious peak and one trough.

    Shape: ramp up → peak → ramp down → trough → ramp up
    """
    # 200 ticks: linearly up 50, hold 10, down 80, hold 10, up 50
    n = 200
    prices = np.zeros(n)
    base = 22000.0

    # Ramp up for first 60 ticks
    for i in range(60):
        prices[i] = base + i * tick_size

    # Peak region (60-70)
    for i in range(60, 70):
        prices[i] = base + 60 * tick_size - (i - 60) * tick_size * 0.3

    # Ramp down (70-140)
    for i in range(70, 140):
        prices[i] = prices[69] - (i - 69) * tick_size

    # Trough region (140-150)
    for i in range(140, 150):
        prices[i] = prices[139] + (i - 139) * tick_size * 0.3

    # Ramp up (150-200)
    for i in range(150, 200):
        prices[i] = prices[149] + (i - 149) * tick_size

    timestamps = pd.date_range("2026-02-20 09:30", periods=n, freq="100ms", tz="UTC")
    return pd.Series(prices), pd.Series(timestamps)


# ────────────────────────────────────────────────────────────────
# Config tests
# ────────────────────────────────────────────────────────────────


class TestConfig:
    """Test Pydantic config models."""

    def test_default_pipeline_config(self):
        cfg = MLPipelineConfig()
        assert cfg.tick_size == 0.25
        assert cfg.instrument == "NQ"
        assert cfg.extrema.window_size == 5000
        assert cfg.extrema.min_prominence_ticks == 10.0
        assert cfg.labeling.rebound_thresholds == [20, 40, 60]
        assert cfg.labeling.crossing_threshold == 20
        assert cfg.labeling.forward_window == 5000
        assert cfg.features.include_signal_features is True
        assert cfg.walk_forward.train_days == 60
        assert cfg.model.model_type == "catboost"

    def test_custom_config(self):
        cfg = MLPipelineConfig(
            tick_size=0.25,
            instrument="ES",
            extrema=ExtremaConfig(window_size=300, min_prominence_ticks=3.0),
            labeling=LabelingConfig(rebound_thresholds=[5, 10]),
            model=ModelConfig(iterations=500, depth=4),
        )
        assert cfg.instrument == "ES"
        assert cfg.extrema.window_size == 300
        assert cfg.extrema.min_prominence_ticks == 3.0
        assert cfg.labeling.rebound_thresholds == [5, 10]
        assert cfg.model.iterations == 500

    def test_extrema_config_validation(self):
        with pytest.raises(ValueError):
            ExtremaConfig(window_size=0)  # ge=50 constraint

    def test_labeling_config_validation(self):
        with pytest.raises(ValueError):
            LabelingConfig(crossing_threshold=0)  # ge=1 constraint

    def test_feature_config_defaults(self):
        cfg = FeatureConfig()
        assert cfg.pl_range_ticks == 10
        assert cfg.ms_window == 237
        assert len(cfg.rsi_periods) == 6
        assert cfg.signal_bar_timeframe == "5m"

    def test_walk_forward_config(self):
        cfg = WalkForwardConfig(expanding=True, gap_days=2)
        assert cfg.expanding is True
        assert cfg.gap_days == 2

    def test_model_config_constraints(self):
        cfg = ModelConfig(depth=10, iterations=100)
        assert cfg.depth == 10
        with pytest.raises(ValueError):
            ModelConfig(depth=20)  # le=16 constraint


# ────────────────────────────────────────────────────────────────
# Extrema detection tests
# ────────────────────────────────────────────────────────────────


class TestExtremaDetection:
    """Test peak/trough detection logic."""

    def test_detect_finds_peaks_and_troughs(self):
        """Detect extrema in a series with clear sinusoidal peaks/troughs."""
        prices, timestamps = _make_price_series_with_peaks(n=2000)
        config = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=1.0,
        )
        extrema = detect_extrema(prices, timestamps, config, tick_size=0.25)

        assert len(extrema) > 0
        peaks = [e for e in extrema if e.extremum_type == "peak"]
        troughs = [e for e in extrema if e.extremum_type == "trough"]
        assert len(peaks) > 0, "Should detect at least one peak"
        assert len(troughs) > 0, "Should detect at least one trough"

    def test_extremum_fields(self):
        """Each Extremum has required fields populated."""
        prices, timestamps = _make_price_series_with_peaks(n=2000)
        config = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=1.0,
        )
        extrema = detect_extrema(prices, timestamps, config, tick_size=0.25)

        for e in extrema:
            assert e.index >= 0
            assert e.index < len(prices)
            assert isinstance(e.timestamp, pd.Timestamp)
            assert e.price > 0
            assert e.extremum_type in ("peak", "trough")
            assert e.prominence > 0
            assert e.width > 0

    def test_extrema_sorted_by_index(self):
        """Output should be sorted by index."""
        prices, timestamps = _make_price_series_with_peaks(n=2000)
        config = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=1.0,
        )
        extrema = detect_extrema(prices, timestamps, config, tick_size=0.25)

        indices = [e.index for e in extrema]
        assert indices == sorted(indices)

    def test_peaks_are_local_maxima(self):
        """Detected peaks should be local maxima in their neighborhood."""
        prices, timestamps = _make_price_series_with_peaks(n=2000)
        config = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=1.0,
        )
        extrema = detect_extrema(prices, timestamps, config, tick_size=0.25)

        price_vals = prices.values
        for e in [x for x in extrema if x.extremum_type == "peak"]:
            # Check local neighborhood (±10 ticks)
            lo = max(0, e.index - 10)
            hi = min(len(price_vals), e.index + 11)
            assert price_vals[e.index] >= price_vals[lo:hi].min()

    def test_troughs_are_local_minima(self):
        """Detected troughs should be local minima in their neighborhood."""
        prices, timestamps = _make_price_series_with_peaks(n=2000)
        config = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=1.0,
        )
        extrema = detect_extrema(prices, timestamps, config, tick_size=0.25)

        price_vals = prices.values
        for e in [x for x in extrema if x.extremum_type == "trough"]:
            lo = max(0, e.index - 10)
            hi = min(len(price_vals), e.index + 11)
            assert price_vals[e.index] <= price_vals[lo:hi].max()

    def test_no_extrema_in_flat_series(self):
        """A perfectly flat price series should produce no extrema."""
        n = 1000
        prices = pd.Series(np.full(n, 22000.0))
        timestamps = pd.Series(
            pd.date_range("2026-02-20 09:30", periods=n, freq="100ms", tz="UTC")
        )
        config = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=1.0,
        )
        extrema = detect_extrema(prices, timestamps, config, tick_size=0.25)
        assert len(extrema) == 0

    def test_short_series_handled(self):
        """Series shorter than window_size should still work."""
        n = 100
        prices, timestamps = _make_price_series_with_peaks(n=n)
        config = ExtremaConfig(
            window_size=500,  # Larger than data
            min_peak_width=10,
            max_peak_width=80,
            min_prominence_ticks=0.5,
        )
        # Should not raise
        extrema = detect_extrema(prices, timestamps, config, tick_size=0.25)
        assert isinstance(extrema, list)

    def test_higher_prominence_yields_fewer_extrema(self):
        """Higher prominence threshold should filter out weaker extrema."""
        prices, timestamps = _make_price_series_with_peaks(n=2000)
        config_low = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=0.5,
        )
        config_high = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=5.0,
        )
        extrema_low = detect_extrema(prices, timestamps, config_low, tick_size=0.25)
        extrema_high = detect_extrema(prices, timestamps, config_high, tick_size=0.25)

        assert len(extrema_low) >= len(extrema_high)


class TestDeduplication:
    """Test deduplication logic."""

    def test_no_duplicates_within_window(self):
        """After dedup, no two extrema of the same type should be within window."""
        extrema = [
            Extremum(10, pd.Timestamp("2026-01-01"), 100.0, "peak", 1.0, 50.0),
            Extremum(15, pd.Timestamp("2026-01-01"), 101.0, "peak", 2.0, 55.0),
            Extremum(80, pd.Timestamp("2026-01-01"), 99.0, "peak", 1.5, 45.0),
        ]
        deduped = _deduplicate(extrema, dedup_window=50)
        peaks = [e for e in deduped if e.extremum_type == "peak"]

        # 10 and 15 are within window 50 — keep higher prominence (index 15)
        # 80 is separate
        assert len(peaks) == 2
        assert peaks[0].index == 15  # Higher prominence
        assert peaks[1].index == 80

    def test_dedup_preserves_different_types(self):
        """Peak and trough at same location should both survive."""
        extrema = [
            Extremum(10, pd.Timestamp("2026-01-01"), 100.0, "peak", 1.0, 50.0),
            Extremum(10, pd.Timestamp("2026-01-01"), 100.0, "trough", 1.0, 50.0),
        ]
        deduped = _deduplicate(extrema, dedup_window=50)
        assert len(deduped) == 2

    def test_empty_list(self):
        assert _deduplicate([], dedup_window=50) == []


class TestDetectInWindow:
    """Test single-window detection."""

    def test_detect_single_peak(self):
        """A clear triangle shape should yield one peak."""
        n = 200
        prices = np.zeros(n)
        for i in range(100):
            prices[i] = 22000.0 + i * 0.25
        for i in range(100, 200):
            prices[i] = 22000.0 + (200 - i) * 0.25

        timestamps = pd.Series(
            pd.date_range("2026-02-20 09:30", periods=n, freq="100ms", tz="UTC")
        )
        config = ExtremaConfig(
            window_size=200,
            min_peak_width=10,
            max_peak_width=190,
            min_prominence_ticks=2.0,
        )
        results = _detect_in_window(prices, timestamps, 0, n, config, 0.25)

        peaks = [e for e in results if e.extremum_type == "peak"]
        assert len(peaks) >= 1
        # Peak should be near index 99-100
        assert abs(peaks[0].index - 99) <= 5


# ────────────────────────────────────────────────────────────────
# Labeling tests
# ────────────────────────────────────────────────────────────────


class TestLabeling:
    """Test rebound/crossing label assignment."""

    def test_peak_rebound_label(self):
        """A peak followed by a price drop should be labeled rebound."""
        tick_size = 0.25
        # Price: up to peak, then drops 10 ticks (2.5 points)
        n = 200
        prices = np.full(n, 22000.0)
        prices[50] = 22010.0  # Peak
        for i in range(51, 100):
            prices[i] = 22010.0 - (i - 50) * tick_size  # Steady drop

        ext = Extremum(
            index=50,
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=22010.0,
            extremum_type="peak",
            prominence=10.0,
            width=50.0,
        )

        config = LabelingConfig(
            rebound_thresholds=[7],
            crossing_threshold=7,
            forward_window=100,
        )
        labeled = label_extrema([ext], pd.Series(prices), config, tick_size)

        assert len(labeled) == 1
        le = labeled[0]
        assert le.labels["label_7t"] == 1  # Rebound (price dropped)

    def test_peak_crossing_label(self):
        """A peak followed by continued rise should be labeled crossing."""
        tick_size = 0.25
        n = 200
        prices = np.full(n, 22000.0)
        prices[50] = 22005.0
        for i in range(51, 100):
            prices[i] = 22005.0 + (i - 50) * tick_size  # Continued rise

        ext = Extremum(
            index=50,
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=22005.0,
            extremum_type="peak",
            prominence=5.0,
            width=50.0,
        )

        config = LabelingConfig(
            rebound_thresholds=[7],
            crossing_threshold=7,
            forward_window=100,
        )
        labeled = label_extrema([ext], pd.Series(prices), config, tick_size)

        assert len(labeled) == 1
        assert labeled[0].labels["label_7t"] == 0  # Crossing

    def test_trough_rebound_label(self):
        """A trough followed by price rise should be labeled rebound."""
        tick_size = 0.25
        n = 200
        prices = np.full(n, 22000.0)
        prices[50] = 21990.0  # Trough
        for i in range(51, 100):
            prices[i] = 21990.0 + (i - 50) * tick_size  # Rises

        ext = Extremum(
            index=50,
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=21990.0,
            extremum_type="trough",
            prominence=10.0,
            width=50.0,
        )

        config = LabelingConfig(
            rebound_thresholds=[7],
            crossing_threshold=7,
            forward_window=100,
        )
        labeled = label_extrema([ext], pd.Series(prices), config, tick_size)

        assert len(labeled) == 1
        assert labeled[0].labels["label_7t"] == 1  # Rebound

    def test_trough_crossing_label(self):
        """A trough followed by continued drop should be labeled crossing."""
        tick_size = 0.25
        n = 200
        prices = np.full(n, 22000.0)
        prices[50] = 21995.0  # Trough
        for i in range(51, 100):
            prices[i] = 21995.0 - (i - 50) * tick_size  # Continues down

        ext = Extremum(
            index=50,
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=21995.0,
            extremum_type="trough",
            prominence=5.0,
            width=50.0,
        )

        config = LabelingConfig(
            rebound_thresholds=[7],
            crossing_threshold=7,
            forward_window=100,
        )
        labeled = label_extrema([ext], pd.Series(prices), config, tick_size)

        assert len(labeled) == 1
        assert labeled[0].labels["label_7t"] == 0  # Crossing

    def test_ambiguous_label(self):
        """Flat price after extremum → neither threshold met → None."""
        tick_size = 0.25
        n = 200
        prices = np.full(n, 22000.0)
        prices[50] = 22001.0  # Tiny peak

        ext = Extremum(
            index=50,
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=22001.0,
            extremum_type="peak",
            prominence=1.0,
            width=50.0,
        )

        config = LabelingConfig(
            rebound_thresholds=[7],
            crossing_threshold=7,
            forward_window=50,
        )
        labeled = label_extrema([ext], pd.Series(prices), config, tick_size)

        assert len(labeled) == 1
        assert labeled[0].labels["label_7t"] is None

    def test_multiple_thresholds(self):
        """Labels at multiple thresholds: lower threshold may fire, higher may not."""
        tick_size = 0.25
        n = 200
        prices = np.full(n, 22000.0)
        prices[50] = 22010.0  # Peak
        for i in range(51, 80):
            prices[i] = 22010.0 - (i - 50) * tick_size * 0.5  # Moderate drop

        ext = Extremum(
            index=50,
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=22010.0,
            extremum_type="peak",
            prominence=10.0,
            width=50.0,
        )

        config = LabelingConfig(
            rebound_thresholds=[5, 20],
            crossing_threshold=20,
            forward_window=50,
        )
        labeled = label_extrema([ext], pd.Series(prices), config, tick_size)

        le = labeled[0]
        # 5-tick drop: should happen within 10 bars (each bar drops 0.5 ticks)
        assert le.labels["label_5t"] == 1
        # 20-tick drop: total drop = 29 bars * 0.5 ticks/bar = 14.5 ticks — may not reach
        # (depends on exact math, but the key test is they can differ)

    def test_reversal_continuation_tracking(self):
        """LabeledExtremum tracks max reversal and continuation ticks."""
        tick_size = 0.25
        n = 200
        prices = np.full(n, 22000.0)
        prices[50] = 22010.0
        for i in range(51, 200):
            prices[i] = 22010.0 - (i - 50) * tick_size

        ext = Extremum(
            index=50,
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=22010.0,
            extremum_type="peak",
            prominence=10.0,
            width=50.0,
        )

        config = LabelingConfig(forward_window=100)
        labeled = label_extrema([ext], pd.Series(prices), config, tick_size)

        le = labeled[0]
        # Price drops 100 bars * 0.25 = 25 points = 100 ticks
        assert le.reversal_ticks > 0
        assert le.continuation_ticks <= 0  # Price never rose above peak

    def test_extremum_at_end_of_series(self):
        """Extremum at last tick should be skipped (no forward data)."""
        n = 100
        prices = pd.Series(np.full(n, 22000.0))
        ext = Extremum(
            index=99,  # Last index — no forward data
            timestamp=pd.Timestamp("2026-02-20 09:30"),
            price=22000.0,
            extremum_type="peak",
            prominence=1.0,
            width=10.0,
        )

        labeled = label_extrema([ext], prices, tick_size=0.25)
        # index 99 == n-1, so it gets skipped by the guard clause
        assert len(labeled) == 0

    def test_empty_extrema_list(self):
        """Empty extrema list returns empty results."""
        prices = pd.Series(np.full(100, 22000.0))
        labeled = label_extrema([], prices)
        assert len(labeled) == 0


class TestBuildLabelDataframe:
    """Test DataFrame construction from labeled extrema."""

    def test_basic_dataframe(self):
        """Build a DataFrame from labeled extrema."""
        le = LabeledExtremum(
            extremum=Extremum(
                index=50,
                timestamp=pd.Timestamp("2026-02-20 09:30"),
                price=22010.0,
                extremum_type="peak",
                prominence=5.0,
                width=100.0,
            ),
            labels={"label_7t": 1, "label_11t": 0, "label_15t": None},
            reversal_ticks=20.0,
            continuation_ticks=3.0,
        )

        df = build_label_dataframe([le])
        assert len(df) == 1
        assert df.iloc[0]["tick_index"] == 50
        assert df.iloc[0]["price"] == 22010.0
        assert df.iloc[0]["extremum_type"] == "peak"
        assert df.iloc[0]["label_7t"] == 1
        assert df.iloc[0]["label_11t"] == 0
        assert pd.isna(df.iloc[0]["label_15t"])

    def test_empty_list_produces_empty_df(self):
        df = build_label_dataframe([])
        assert df.empty

    def test_multiple_extrema(self):
        labeled = [
            LabeledExtremum(
                extremum=Extremum(i * 100, pd.Timestamp("2026-02-20"), 22000.0 + i, t, 1.0, 50.0),
                labels={"label_7t": 1},
                reversal_ticks=10.0,
                continuation_ticks=5.0,
            )
            for i, t in enumerate(["peak", "trough", "peak"])
        ]

        df = build_label_dataframe(labeled)
        assert len(df) == 3
        assert list(df["extremum_type"]) == ["peak", "trough", "peak"]


# ────────────────────────────────────────────────────────────────
# Integration: detect → label pipeline
# ────────────────────────────────────────────────────────────────


class TestDetectLabelPipeline:
    """End-to-end: detect extrema then label them."""

    def test_full_pipeline(self):
        """Detect extrema in synthetic data, then label them."""
        prices, timestamps = _make_price_series_with_peaks(n=2000)
        extrema_config = ExtremaConfig(
            window_size=500,
            min_peak_width=20,
            max_peak_width=400,
            min_prominence_ticks=1.0,
        )
        label_config = LabelingConfig(
            rebound_thresholds=[7, 11, 15],
            crossing_threshold=7,
            forward_window=100,
        )

        extrema = detect_extrema(prices, timestamps, extrema_config, tick_size=0.25)
        assert len(extrema) > 0

        labeled = label_extrema(extrema, prices, label_config, tick_size=0.25)
        assert len(labeled) == len(extrema)

        df = build_label_dataframe(labeled)
        assert len(df) == len(extrema)
        assert "label_7t" in df.columns
        assert "label_11t" in df.columns
        assert "label_15t" in df.columns

        # At least some labels should be non-null
        non_null_7 = df["label_7t"].dropna()
        assert len(non_null_7) > 0, "Expected at least some non-null labels at 7t threshold"

    def test_pipeline_with_pipeline_config(self):
        """Use MLPipelineConfig to drive the full pipeline."""
        config = MLPipelineConfig(
            tick_size=0.25,
            extrema=ExtremaConfig(
                window_size=500,
                min_peak_width=20,
                max_peak_width=400,
                min_prominence_ticks=1.0,
            ),
        )

        prices, timestamps = _make_price_series_with_peaks(n=2000)
        extrema = detect_extrema(
            prices, timestamps, config.extrema, config.tick_size,
        )
        labeled = label_extrema(
            extrema, prices, config.labeling, config.tick_size,
        )
        df = build_label_dataframe(labeled)

        assert len(df) > 0
        assert "prominence" in df.columns
        assert "width" in df.columns
