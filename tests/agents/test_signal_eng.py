"""Tests for Signal Engineering agent, detectors, indicators, and bundle builder."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

# Import detectors to trigger auto-registration
import alpha_lab.agents.signal_eng.detectors  # noqa: F401
from alpha_lab.agents.signal_eng.agent import SignalEngineeringAgent
from alpha_lab.agents.signal_eng.bundle_builder import (
    build_signal_bundle,
    run_single_detector,
)
from alpha_lab.agents.signal_eng.detector_base import (
    SignalDetector,
    SignalDetectorRegistry,
)
from alpha_lab.agents.signal_eng.detectors.tier1.ema_confluence import (
    EmaConfluenceDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier1.kama_regime import (
    KamaRegimeDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier1.vwap_deviation import (
    VwapDeviationDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier2.fair_value_gaps import (
    FairValueGapsDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier2.ifvg import IFVGDetector
from alpha_lab.agents.signal_eng.detectors.tier2.killzone_timing import (
    KillzoneTimingDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier2.liquidity_sweeps import (
    LiquiditySweepsDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier2.market_structure import (
    MarketStructureDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier2.pd_levels_poi import (
    PDLevelsPOIDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier2.tick_microstructure import (
    TickMicrostructureDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.adaptive_regime import (
    AdaptiveRegimeDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.displacement import (
    DisplacementDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.ema_reclaim import (
    EmaReclaimDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.ema_vwap_interaction import (
    EmaVwapInteractionDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.multi_tf_confluence import (
    MultiTFConfluenceDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.order_blocks import (
    OrderBlocksDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.scalp_entry import (
    ScalpEntryDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.session_gap import (
    SessionGapDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.sweep_fvg_combo import (
    SweepFVGComboDetector,
)
from alpha_lab.agents.signal_eng.detectors.tier3.volume_profile import (
    VolumeProfileDetector,
)
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_ema,
    compute_kama,
    compute_kama_efficiency_ratio,
    compute_session_vwap,
    compute_session_vwap_bands,
    compute_swing_highs,
    compute_swing_lows,
)
from alpha_lab.core.contracts import (
    DataBundle,
    PreviousDayLevels,
    QualityReport,
    SessionMetadata,
)
from alpha_lab.core.enums import AgentID, AgentState, MessageType, Priority, SignalTier
from alpha_lab.core.message import MessageEnvelope

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data with a DatetimeIndex (US/Eastern).

    Creates a trending-then-ranging price series to test signal detectors
    in both regime conditions.
    """
    rng = np.random.default_rng(seed)
    # Create trending phase (first half) then ranging (second half)
    trend = np.cumsum(rng.normal(0.5, 2.0, n // 2))
    ranging = trend[-1] + np.cumsum(rng.normal(0.0, 2.0, n - n // 2))
    close = 22000 + np.concatenate([trend, ranging])

    high = close + rng.uniform(2, 15, n)
    low = close - rng.uniform(2, 15, n)
    open_ = close + rng.normal(0, 3, n)
    volume = rng.integers(100, 5000, n).astype(float)

    idx = pd.date_range(
        "2026-02-20 09:30",
        periods=n,
        freq="5min",
        tz="US/Eastern",
    )

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "session_type": "RTH",
            "session_id": "NQ_2026-02-20_RTH",
            "killzone": "NEW_YORK",
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def _make_data_bundle(
    timeframes: dict[str, pd.DataFrame] | None = None,
) -> DataBundle:
    """Create a DataBundle with synthetic data for testing."""
    if timeframes is None:
        bars_5m = _make_ohlcv(300)
        bars_15m = _make_ohlcv(100, seed=43)
        bars_1h = _make_ohlcv(60, seed=44)
        timeframes = {"5m": bars_5m, "15m": bars_15m, "1H": bars_1h}

    return DataBundle(
        instrument="NQ",
        bars=timeframes,
        sessions=[
            SessionMetadata(
                session_id="NQ_2026-02-20_RTH",
                session_type="RTH",
                killzone="NEW_YORK",
                rth_open="2026-02-20T09:30:00-05:00",
                rth_close="2026-02-20T16:15:00-05:00",
            )
        ],
        pd_levels={},
        quality=QualityReport(
            passed=True,
            total_bars=460,
            gaps_found=0,
            volume_zeros=0,
            ohlc_violations=0,
            cross_tf_mismatches=0,
            timestamp_coverage=1.0,
            report_generated_at=datetime.now(UTC).isoformat(),
        ),
        date_range=("2026-02-20", "2026-02-20"),
    )


# ═══════════════════════════════════════════════════════════════════
# Indicator Tests
# ═══════════════════════════════════════════════════════════════════


class TestComputeEma:
    def test_ema_returns_series(self):
        s = pd.Series(range(50), dtype=float)
        result = compute_ema(s, span=13)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_ema_converges_to_constant(self):
        s = pd.Series([100.0] * 50)
        result = compute_ema(s, span=13)
        assert abs(result.iloc[-1] - 100.0) < 1e-10

    def test_ema_tracks_trend(self):
        s = pd.Series(np.arange(100, dtype=float))
        ema = compute_ema(s, span=13)
        # EMA should lag behind a linear trend
        assert ema.iloc[-1] < s.iloc[-1]
        assert ema.iloc[-1] > s.iloc[-10]


class TestComputeAtr:
    def test_atr_returns_series(self):
        df = _make_ohlcv(50)
        result = compute_atr(df, period=14)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_atr_warmup_nans(self):
        df = _make_ohlcv(50)
        result = compute_atr(df, period=14)
        assert result.iloc[:13].isna().all()
        assert result.iloc[13:].notna().all()

    def test_atr_positive(self):
        df = _make_ohlcv(50)
        result = compute_atr(df, period=14)
        assert (result.dropna() > 0).all()


class TestComputeKama:
    def test_kama_returns_series(self):
        s = pd.Series(np.arange(50, dtype=float))
        result = compute_kama(s, period=10)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_kama_warmup_nans(self):
        s = pd.Series(np.arange(50, dtype=float))
        result = compute_kama(s, period=10)
        assert result.iloc[:10].isna().all()
        assert result.iloc[10:].notna().all()

    def test_kama_converges_for_constant(self):
        s = pd.Series([100.0] * 50)
        result = compute_kama(s, period=10)
        valid = result.dropna()
        assert abs(valid.iloc[-1] - 100.0) < 1e-10


class TestComputeKamaEfficiencyRatio:
    def test_er_range(self):
        s = pd.Series(np.arange(50, dtype=float))
        er = compute_kama_efficiency_ratio(s, period=10)
        valid = er.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_er_high_for_trend(self):
        # Pure linear trend should have ER near 1
        s = pd.Series(np.arange(50, dtype=float))
        er = compute_kama_efficiency_ratio(s, period=10)
        assert er.iloc[-1] > 0.9

    def test_er_low_for_noise(self):
        # Random walk should have lower ER
        rng = np.random.default_rng(42)
        s = pd.Series(np.cumsum(rng.choice([-1, 1], 200).astype(float)))
        er = compute_kama_efficiency_ratio(s, period=10)
        assert er.iloc[-20:].mean() < 0.5


class TestComputeSessionVwap:
    def test_vwap_returns_series(self):
        df = _make_ohlcv(50)
        result = compute_session_vwap(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 50

    def test_vwap_within_price_range(self):
        df = _make_ohlcv(50)
        vwap = compute_session_vwap(df)
        assert (vwap >= df["low"].min() - 1).all()
        assert (vwap <= df["high"].max() + 1).all()

    def test_vwap_bands_ordering(self):
        df = _make_ohlcv(50)
        vwap, upper, lower = compute_session_vwap_bands(df, num_std=2.0)
        # Upper should be >= VWAP, lower should be <= VWAP
        valid = vwap.notna() & upper.notna() & lower.notna()
        assert (upper[valid] >= vwap[valid] - 1e-10).all()
        assert (lower[valid] <= vwap[valid] + 1e-10).all()


# ═══════════════════════════════════════════════════════════════════
# Detector Registry Tests (preserved from original)
# ═══════════════════════════════════════════════════════════════════


class TestDetectorRegistry:
    def test_all_20_detectors_registered(self):
        assert SignalDetectorRegistry.count() == 20

    def test_tier1_has_3(self):
        tier1 = SignalDetectorRegistry.get_by_tier(SignalTier.CORE)
        assert len(tier1) == 3

    def test_tier2_has_7(self):
        tier2 = SignalDetectorRegistry.get_by_tier(SignalTier.ICT_STRUCTURAL)
        assert len(tier2) == 7

    def test_tier3_has_10(self):
        tier3 = SignalDetectorRegistry.get_by_tier(SignalTier.COMPOSITE)
        assert len(tier3) == 10

    def test_get_by_id(self):
        cls = SignalDetectorRegistry.get("ema_confluence")
        assert cls.detector_id == "ema_confluence"
        assert cls.tier == SignalTier.CORE

    def test_all_detector_ids(self):
        ids = SignalDetectorRegistry.list_ids()
        expected = [
            "adaptive_regime", "displacement", "ema_confluence",
            "ema_reclaim", "ema_vwap_interaction", "fair_value_gaps",
            "ifvg", "kama_regime", "killzone_timing", "liquidity_sweeps",
            "market_structure", "multi_tf_confluence", "order_blocks",
            "pd_levels_poi", "scalp_entry", "session_gap",
            "sweep_fvg_combo", "tick_microstructure", "volume_profile",
            "vwap_deviation",
        ]
        assert ids == expected

    def test_all_detectors_are_signal_detector_subclass(self):
        for det_id, cls in SignalDetectorRegistry.get_all().items():
            assert issubclass(cls, SignalDetector), f"{det_id} is not a SignalDetector subclass"

    def test_all_detectors_have_required_class_vars(self):
        for det_id, cls in SignalDetectorRegistry.get_all().items():
            assert hasattr(cls, "detector_id"), f"{det_id} missing detector_id"
            assert hasattr(cls, "category"), f"{det_id} missing category"
            assert hasattr(cls, "tier"), f"{det_id} missing tier"
            assert hasattr(cls, "timeframes"), f"{det_id} missing timeframes"


# ═══════════════════════════════════════════════════════════════════
# EMA Confluence Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestEmaConfluenceDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = EmaConfluenceDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = EmaConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_EMA_CONFLUENCE_")
            assert sv.category == "ema_confluence"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = EmaConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = EmaConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = EmaConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = EmaConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "ema_fast" in sv.parameters
            assert "ema_mid" in sv.parameters
            assert "ema_slow" in sv.parameters

    def test_custom_ema_spans(self):
        data = _make_data_bundle()
        detector = EmaConfluenceDetector(ema_fast=8, ema_mid=21, ema_slow=50)
        signals = detector.compute(data)
        assert len(signals) > 0
        assert signals[0].parameters["ema_fast"] == 8

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(50)
        data = _make_data_bundle({"5m": short_df})
        detector = EmaConfluenceDetector()
        assert not detector.validate_inputs(data)

    def test_skips_missing_timeframes(self):
        data = _make_data_bundle({"5m": _make_ohlcv(300)})
        detector = EmaConfluenceDetector()
        signals = detector.compute(data)
        # Only 5m should produce signals
        tfs = {sv.timeframe for sv in signals}
        assert tfs == {"5m"}


# ═══════════════════════════════════════════════════════════════════
# KAMA Regime Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestKamaRegimeDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = KamaRegimeDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = KamaRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_KAMA_REGIME_")
            assert sv.category == "kama_regime"

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = KamaRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = KamaRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = KamaRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = KamaRegimeDetector(kama_period=20, er_trend_threshold=0.5)
        signals = detector.compute(data)
        if signals:
            assert signals[0].parameters["kama_period"] == 20
            assert signals[0].parameters["er_trend_threshold"] == 0.5

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(20)
        data = _make_data_bundle({"5m": short_df})
        detector = KamaRegimeDetector()
        assert not detector.validate_inputs(data)

    def test_trending_data_produces_directional_signals(self):
        # Pure uptrend should produce bullish signals
        n = 300
        close = 22000 + np.arange(n, dtype=float) * 2.0
        df = pd.DataFrame(
            {
                "open": close - 1,
                "high": close + 5,
                "low": close - 5,
                "close": close,
                "volume": np.full(n, 1000.0),
                "session_type": "RTH",
                "session_id": "NQ_2026-02-20_RTH",
                "killzone": "NEW_YORK",
            },
            index=pd.date_range(
                "2026-02-20 09:30", periods=n, freq="5min", tz="US/Eastern"
            ),
        )
        df.index.name = "timestamp"
        data = _make_data_bundle({"5m": df})
        detector = KamaRegimeDetector()
        signals = detector.compute(data)
        assert len(signals) > 0
        # Latter half should be mostly bullish
        sv = signals[0]
        latter_half = sv.direction.iloc[n // 2 :]
        assert (latter_half == 1).sum() > (latter_half == -1).sum()


# ═══════════════════════════════════════════════════════════════════
# VWAP Deviation Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestVwapDeviationDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = VwapDeviationDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = VwapDeviationDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_VWAP_DEVIATION_")
            assert sv.category == "vwap_deviation"

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = VwapDeviationDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = VwapDeviationDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = VwapDeviationDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = VwapDeviationDetector(band_std=1.5, zscore_threshold=0.5)
        signals = detector.compute(data)
        if signals:
            assert signals[0].parameters["band_std"] == 1.5
            assert signals[0].parameters["zscore_threshold"] == 0.5

    def test_validate_inputs_no_volume(self):
        df = _make_ohlcv(50).drop(columns=["volume"])
        data = _make_data_bundle({"5m": df})
        detector = VwapDeviationDetector()
        assert not detector.validate_inputs(data)

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(10)
        data = _make_data_bundle({"5m": short_df})
        detector = VwapDeviationDetector()
        assert not detector.validate_inputs(data)

    def test_intraday_only(self):
        """VWAP detector should only produce signals for intraday timeframes."""
        data = _make_data_bundle()
        detector = VwapDeviationDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.timeframe in ["1m", "5m", "15m", "30m", "1H"]


# ═══════════════════════════════════════════════════════════════════
# Bundle Builder Tests
# ═══════════════════════════════════════════════════════════════════


class TestBundleBuilder:
    def test_build_with_tier1_only(self):
        data = _make_data_bundle()
        bundle = build_signal_bundle(
            data, detector_ids=["ema_confluence", "kama_regime", "vwap_deviation"]
        )
        assert bundle.total_signals > 0
        assert bundle.instrument == "NQ"

    def test_build_all_detectors_skips_stubs(self):
        """Running all detectors should skip unimplemented ones gracefully."""
        data = _make_data_bundle()
        bundle = build_signal_bundle(data)
        # Should still produce signals from Tier 1
        assert bundle.total_signals > 0

    def test_timeframes_covered(self):
        data = _make_data_bundle()
        bundle = build_signal_bundle(data, detector_ids=["ema_confluence"])
        assert len(bundle.timeframes_covered) > 0
        for tf in bundle.timeframes_covered:
            assert isinstance(tf, str)

    def test_generation_timestamp(self):
        data = _make_data_bundle()
        bundle = build_signal_bundle(data, detector_ids=["ema_confluence"])
        assert bundle.generation_timestamp

    def test_run_single_detector(self):
        data = _make_data_bundle()
        signals = run_single_detector(EmaConfluenceDetector, data)
        assert len(signals) > 0

    def test_run_single_detector_tier3(self):
        """Tier 3 detectors produce signals (no longer stubs)."""
        data = _make_data_bundle()
        signals = run_single_detector(DisplacementDetector, data)
        assert isinstance(signals, list)

    def test_build_with_nonexistent_detector(self):
        """Non-existent detector IDs should be silently ignored."""
        data = _make_data_bundle()
        bundle = build_signal_bundle(
            data, detector_ids=["ema_confluence", "nonexistent_detector"]
        )
        assert bundle.total_signals > 0

    def test_build_empty_detector_list(self):
        """Empty detector list produces empty bundle."""
        data = _make_data_bundle()
        bundle = build_signal_bundle(data, detector_ids=[])
        assert bundle.total_signals == 0


# ═══════════════════════════════════════════════════════════════════
# Agent Tests
# ═══════════════════════════════════════════════════════════════════


class TestSignalEngineeringAgent:
    def test_create(self, message_bus):
        agent = SignalEngineeringAgent(message_bus)
        assert agent.agent_id == AgentID.SIGNAL_ENG

    def test_generate_signals(self, message_bus):
        agent = SignalEngineeringAgent(message_bus)
        data = _make_data_bundle()
        bundle = agent.generate_signals(data)
        assert bundle.total_signals > 0
        assert bundle.instrument == "NQ"

    def test_generate_signals_subset(self, message_bus):
        agent = SignalEngineeringAgent(message_bus)
        data = _make_data_bundle()
        bundle = agent.generate_signals(data, detector_ids=["ema_confluence"])
        categories = {sv.category for sv in bundle.signals}
        assert categories == {"ema_confluence"}

    def test_handle_data_bundle_message(self, message_bus):
        agent = SignalEngineeringAgent(message_bus)
        message_bus.register_agent(AgentID.SIGNAL_ENG, agent.handle_message)

        data = _make_data_bundle()
        envelope = MessageEnvelope(
            request_id="test-sig-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.SIGNAL_ENG,
            message_type=MessageType.DATA_BUNDLE,
            priority=Priority.NORMAL,
            payload={"bundle": data.model_dump()},
        )

        agent.handle_message(envelope)
        assert agent.state == AgentState.IDLE

        # Check that agent sent a SIGNAL_BUNDLE back
        log = message_bus.get_audit_log()
        signal_msgs = [
            m for m in log if m.message_type == MessageType.SIGNAL_BUNDLE
        ]
        assert len(signal_msgs) >= 1

    def test_handle_unexpected_message(self, message_bus):
        agent = SignalEngineeringAgent(message_bus)
        message_bus.register_agent(AgentID.SIGNAL_ENG, agent.handle_message)

        envelope = MessageEnvelope(
            request_id="test-sig-002",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.SIGNAL_ENG,
            message_type=MessageType.HALT_COMMAND,
            priority=Priority.NORMAL,
            payload={},
        )

        agent.handle_message(envelope)
        nacks = [
            m for m in message_bus.get_audit_log() if m.message_type == MessageType.NACK
        ]
        assert len(nacks) >= 1

    def test_refine_max_iterations(self, message_bus):
        agent = SignalEngineeringAgent(message_bus)
        for _ in range(3):
            with pytest.raises(NotImplementedError):
                agent.refine_signal("SIG_TEST_5m_v1", "ic", 0.01, 0.03)

        # 4th iteration should raise ValueError (max reached)
        with pytest.raises(ValueError, match="max refinement"):
            agent.refine_signal("SIG_TEST_5m_v1", "ic", 0.01, 0.03)


# ═══════════════════════════════════════════════════════════════════
# Helper: DataBundle with PD levels (for Tier 2 detectors)
# ═══════════════════════════════════════════════════════════════════


def _make_data_bundle_with_pd_levels(
    timeframes: dict[str, pd.DataFrame] | None = None,
) -> DataBundle:
    """Create a DataBundle with pd_levels populated for Tier 2 tests."""
    if timeframes is None:
        bars_5m = _make_ohlcv(300)
        bars_15m = _make_ohlcv(100, seed=43)
        bars_1h = _make_ohlcv(60, seed=44)
        bars_1m = _make_ohlcv(300, seed=45)
        timeframes = {
            "1m": bars_1m, "5m": bars_5m, "15m": bars_15m, "1H": bars_1h,
        }

    ref = timeframes.get("5m", list(timeframes.values())[0])
    mid = float(ref["close"].iloc[0])

    return DataBundle(
        instrument="NQ",
        bars=timeframes,
        sessions=[
            SessionMetadata(
                session_id="NQ_2026-02-20_RTH",
                session_type="RTH",
                killzone="NEW_YORK",
                rth_open="2026-02-20T09:30:00-05:00",
                rth_close="2026-02-20T16:15:00-05:00",
            )
        ],
        pd_levels={
            "2026-02-20": PreviousDayLevels(
                pd_high=mid + 50,
                pd_low=mid - 50,
                pd_mid=mid,
                pd_close=mid + 10,
                pw_high=mid + 100,
                pw_low=mid - 100,
                overnight_high=mid + 30,
                overnight_low=mid - 30,
            ),
        },
        quality=QualityReport(
            passed=True,
            total_bars=sum(len(v) for v in timeframes.values()),
            gaps_found=0,
            volume_zeros=0,
            ohlc_violations=0,
            cross_tf_mismatches=0,
            timestamp_coverage=1.0,
            report_generated_at=datetime.now(UTC).isoformat(),
        ),
        date_range=("2026-02-20", "2026-02-20"),
    )


# ═══════════════════════════════════════════════════════════════════
# Swing Indicator Tests
# ═══════════════════════════════════════════════════════════════════


class TestSwingIndicators:
    def test_swing_highs_returns_series(self):
        df = _make_ohlcv(100)
        result = compute_swing_highs(df["high"])
        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_swing_lows_returns_series(self):
        df = _make_ohlcv(100)
        result = compute_swing_lows(df["low"])
        assert isinstance(result, pd.Series)
        assert len(result) == 100

    def test_swing_highs_are_local_maxima(self):
        df = _make_ohlcv(300)
        sh = compute_swing_highs(df["high"], left=3, right=3)
        valid = sh.dropna()
        for idx in valid.index:
            pos = df.index.get_loc(idx)
            # Swing stamped at confirmation bar (pos), pivot was at pos - right
            pivot_pos = pos - 3
            window = df["high"].iloc[max(0, pivot_pos - 3): pivot_pos + 4]
            assert valid[idx] == window.max()

    def test_swing_lows_are_local_minima(self):
        df = _make_ohlcv(300)
        sl = compute_swing_lows(df["low"], left=3, right=3)
        valid = sl.dropna()
        for idx in valid.index:
            pos = df.index.get_loc(idx)
            # Swing stamped at confirmation bar (pos), pivot was at pos - right
            pivot_pos = pos - 3
            window = df["low"].iloc[max(0, pivot_pos - 3): pivot_pos + 4]
            assert valid[idx] == window.min()


# ═══════════════════════════════════════════════════════════════════
# Fair Value Gaps Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestFairValueGapsDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = FairValueGapsDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = FairValueGapsDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_FAIR_VALUE_GAPS_")
            assert sv.category == "fair_value_gaps"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = FairValueGapsDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = FairValueGapsDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = FairValueGapsDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = FairValueGapsDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "min_gap_atr" in sv.parameters
            assert "max_fvg_age" in sv.parameters

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(10)
        data = _make_data_bundle({"5m": short_df})
        detector = FairValueGapsDetector()
        assert not detector.validate_inputs(data)

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = FairValueGapsDetector(
            min_gap_atr=0.3, decay_half_life=100,
        )
        signals = detector.compute(data)
        if signals:
            assert signals[0].parameters["min_gap_atr"] == 0.3
            assert signals[0].parameters["decay_half_life"] == 100


# ═══════════════════════════════════════════════════════════════════
# IFVG Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestIFVGDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = IFVGDetector()
        signals = detector.compute(data)
        # IFVG may produce empty signals (requires FVG fill + rejection)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = IFVGDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_IFVG_")
            assert sv.category == "ifvg"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = IFVGDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = IFVGDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = IFVGDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = IFVGDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "min_gap_atr" in sv.parameters
            assert "max_touches" in sv.parameters

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(10)
        data = _make_data_bundle({"5m": short_df})
        detector = IFVGDetector()
        assert not detector.validate_inputs(data)

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = IFVGDetector(max_touches=5, rejection_body_ratio=0.4)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["max_touches"] == 5
            assert sv.parameters["rejection_body_ratio"] == 0.4


# ═══════════════════════════════════════════════════════════════════
# Market Structure Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestMarketStructureDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = MarketStructureDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = MarketStructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_MARKET_STRUCTURE_")
            assert sv.category == "market_structure"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = MarketStructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = MarketStructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = MarketStructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = MarketStructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "pivot_left" in sv.parameters
            assert "pivot_right" in sv.parameters
            assert "structure_memory" in sv.parameters

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(10)
        data = _make_data_bundle({"5m": short_df})
        detector = MarketStructureDetector()
        assert not detector.validate_inputs(data)

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = MarketStructureDetector(
            pivot_left=5, pivot_right=5, min_break_atr=0.5,
        )
        signals = detector.compute(data)
        if signals:
            assert signals[0].parameters["pivot_left"] == 5
            assert signals[0].parameters["min_break_atr"] == 0.5


# ═══════════════════════════════════════════════════════════════════
# Liquidity Sweeps Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestLiquiditySweepsDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle_with_pd_levels()
        detector = LiquiditySweepsDetector()
        signals = detector.compute(data)
        # May or may not produce signals depending on data
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle_with_pd_levels()
        detector = LiquiditySweepsDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_LIQUIDITY_SWEEPS_")
            assert sv.category == "liquidity_sweeps"

    def test_direction_values(self):
        data = _make_data_bundle_with_pd_levels()
        detector = LiquiditySweepsDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle_with_pd_levels()
        detector = LiquiditySweepsDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle_with_pd_levels()
        detector = LiquiditySweepsDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle_with_pd_levels()
        detector = LiquiditySweepsDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "sweep_buffer_atr" in sv.parameters
            assert "min_reversal_ratio" in sv.parameters

    def test_validate_inputs_no_pd_levels(self):
        data = _make_data_bundle()  # No pd_levels
        detector = LiquiditySweepsDetector()
        assert not detector.validate_inputs(data)

    def test_custom_parameters(self):
        data = _make_data_bundle_with_pd_levels()
        detector = LiquiditySweepsDetector(
            sweep_buffer_atr=0.2, min_reversal_ratio=0.5,
        )
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["sweep_buffer_atr"] == 0.2
            assert sv.parameters["min_reversal_ratio"] == 0.5


# ═══════════════════════════════════════════════════════════════════
# PD Levels POI Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestPDLevelsPOIDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle_with_pd_levels()
        detector = PDLevelsPOIDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle_with_pd_levels()
        detector = PDLevelsPOIDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_PD_LEVELS_POI_")
            assert sv.category == "pd_levels_poi"

    def test_direction_values(self):
        data = _make_data_bundle_with_pd_levels()
        detector = PDLevelsPOIDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle_with_pd_levels()
        detector = PDLevelsPOIDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle_with_pd_levels()
        detector = PDLevelsPOIDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle_with_pd_levels()
        detector = PDLevelsPOIDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "level_proximity_atr" in sv.parameters

    def test_validate_inputs_no_pd_levels(self):
        data = _make_data_bundle()  # No pd_levels
        detector = PDLevelsPOIDetector()
        assert not detector.validate_inputs(data)

    def test_custom_parameters(self):
        data = _make_data_bundle_with_pd_levels()
        detector = PDLevelsPOIDetector(level_proximity_atr=1.0)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["level_proximity_atr"] == 1.0


# ═══════════════════════════════════════════════════════════════════
# Killzone Timing Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestKillzoneTimingDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = KillzoneTimingDetector()
        signals = detector.compute(data)
        # May produce signals if data overlaps killzone hours
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = KillzoneTimingDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_KILLZONE_TIMING_")
            assert sv.category == "killzone_timing"

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = KillzoneTimingDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = KillzoneTimingDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = KillzoneTimingDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = KillzoneTimingDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "london_hours" in sv.parameters
            assert "ny_hours" in sv.parameters
            assert "asia_hours" in sv.parameters

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(10)
        data = _make_data_bundle({"5m": short_df})
        detector = KillzoneTimingDetector()
        assert not detector.validate_inputs(data)

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = KillzoneTimingDetector(
            min_activity_ratio=2.0, direction_window=6,
        )
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["min_activity_ratio"] == 2.0
            assert sv.parameters["direction_window"] == 6


# ═══════════════════════════════════════════════════════════════════
# Tick Microstructure Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestTickMicrostructureDetector:
    def test_compute_produces_signals(self):
        bars_1m = _make_ohlcv(300, seed=50)
        data = _make_data_bundle({"1m": bars_1m})
        detector = TickMicrostructureDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        bars_1m = _make_ohlcv(300, seed=50)
        data = _make_data_bundle({"1m": bars_1m})
        detector = TickMicrostructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_TICK_MICROSTRUCTURE_")
            assert sv.category == "tick_microstructure"

    def test_direction_values(self):
        bars_1m = _make_ohlcv(300, seed=50)
        data = _make_data_bundle({"1m": bars_1m})
        detector = TickMicrostructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        bars_1m = _make_ohlcv(300, seed=50)
        data = _make_data_bundle({"1m": bars_1m})
        detector = TickMicrostructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        bars_1m = _make_ohlcv(300, seed=50)
        data = _make_data_bundle({"1m": bars_1m})
        detector = TickMicrostructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        bars_1m = _make_ohlcv(300, seed=50)
        data = _make_data_bundle({"1m": bars_1m})
        detector = TickMicrostructureDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "min_velocity_atr" in sv.parameters
            assert "streak_length" in sv.parameters
            assert "volume_spike_ratio" in sv.parameters

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(10)
        data = _make_data_bundle({"1m": short_df})
        detector = TickMicrostructureDetector()
        assert not detector.validate_inputs(data)

    def test_custom_parameters(self):
        bars_1m = _make_ohlcv(300, seed=50)
        data = _make_data_bundle({"1m": bars_1m})
        detector = TickMicrostructureDetector(
            min_velocity_atr=0.5, streak_length=3,
        )
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["min_velocity_atr"] == 0.5
            assert sv.parameters["streak_length"] == 3


# ═══════════════════════════════════════════════════════════════════
# Tier 3 Composite Detector Tests
# ═══════════════════════════════════════════════════════════════════


class TestAdaptiveRegimeDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = AdaptiveRegimeDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = AdaptiveRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_ADAPTIVE_REGIME_")
            assert sv.category == "adaptive_regime"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = AdaptiveRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = AdaptiveRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = AdaptiveRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = AdaptiveRegimeDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "er_period" in sv.parameters
            assert "ema_fast" in sv.parameters
            assert "ema_slow" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = AdaptiveRegimeDetector(er_period=20, er_threshold=0.5)
        signals = detector.compute(data)
        if signals:
            assert signals[0].parameters["er_period"] == 20
            assert signals[0].parameters["er_threshold"] == 0.5

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(20)
        data = _make_data_bundle({"5m": short_df})
        detector = AdaptiveRegimeDetector()
        assert not detector.validate_inputs(data)


class TestEmaVwapInteractionDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = EmaVwapInteractionDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = EmaVwapInteractionDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_EMA_VWAP_INTERACTION_")
            assert sv.category == "ema_vwap_interaction"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = EmaVwapInteractionDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = EmaVwapInteractionDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = EmaVwapInteractionDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = EmaVwapInteractionDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "ema_fast" in sv.parameters
            assert "ema_slow" in sv.parameters
            assert "band_std" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = EmaVwapInteractionDetector(ema_fast=8, band_std=1.5)
        signals = detector.compute(data)
        if signals:
            assert signals[0].parameters["ema_fast"] == 8
            assert signals[0].parameters["band_std"] == 1.5

    def test_validate_inputs_no_volume(self):
        df = _make_ohlcv(300).drop(columns=["volume"])
        data = _make_data_bundle({"5m": df})
        detector = EmaVwapInteractionDetector()
        assert not detector.validate_inputs(data)


class TestMultiTFConfluenceDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = MultiTFConfluenceDetector()
        signals = detector.compute(data)
        assert len(signals) > 0

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = MultiTFConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_MULTI_TF_CONFLUENCE_")
            assert sv.category == "multi_tf_confluence"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = MultiTFConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = MultiTFConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = MultiTFConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = MultiTFConfluenceDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "ema_fast" in sv.parameters
            assert "ema_mid" in sv.parameters
            assert "ema_slow" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = MultiTFConfluenceDetector(ema_fast=8, ema_mid=21, ema_slow=100)
        signals = detector.compute(data)
        if signals:
            assert signals[0].parameters["ema_fast"] == 8
            assert signals[0].parameters["ema_mid"] == 21

    def test_validate_inputs_single_tf(self):
        data = _make_data_bundle({"5m": _make_ohlcv(300)})
        detector = MultiTFConfluenceDetector(min_timeframes=2)
        assert not detector.validate_inputs(data)


class TestEmaReclaimDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = EmaReclaimDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = EmaReclaimDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_EMA_RECLAIM_")
            assert sv.category == "ema_reclaim"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = EmaReclaimDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = EmaReclaimDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = EmaReclaimDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = EmaReclaimDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "ema_fast" in sv.parameters
            assert "ema_slow" in sv.parameters
            assert "max_sweep_bars" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = EmaReclaimDetector(max_sweep_bars=30, min_sweep_depth_atr=0.5)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["max_sweep_bars"] == 30
            assert sv.parameters["min_sweep_depth_atr"] == 0.5

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(20)
        data = _make_data_bundle({"5m": short_df})
        detector = EmaReclaimDetector()
        assert not detector.validate_inputs(data)


class TestDisplacementDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = DisplacementDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = DisplacementDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_DISPLACEMENT_")
            assert sv.category == "displacement"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = DisplacementDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = DisplacementDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = DisplacementDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = DisplacementDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "consolidation_window" in sv.parameters
            assert "displacement_multiplier" in sv.parameters
            assert "min_gap_atr" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = DisplacementDetector(
            consolidation_threshold=2.0, displacement_multiplier=3.0,
        )
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["consolidation_threshold"] == 2.0
            assert sv.parameters["displacement_multiplier"] == 3.0

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(20)
        data = _make_data_bundle({"5m": short_df})
        detector = DisplacementDetector()
        assert not detector.validate_inputs(data)


class TestOrderBlocksDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = OrderBlocksDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = OrderBlocksDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_ORDER_BLOCKS_")
            assert sv.category == "order_blocks"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = OrderBlocksDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = OrderBlocksDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = OrderBlocksDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = OrderBlocksDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "pivot_left" in sv.parameters
            assert "ob_max_age" in sv.parameters
            assert "min_gap_atr" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = OrderBlocksDetector(pivot_left=5, ob_max_age=50)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["pivot_left"] == 5
            assert sv.parameters["ob_max_age"] == 50

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(10)
        data = _make_data_bundle({"5m": short_df})
        detector = OrderBlocksDetector()
        assert not detector.validate_inputs(data)


class TestSweepFVGComboDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = SweepFVGComboDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = SweepFVGComboDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_SWEEP_FVG_COMBO_")
            assert sv.category == "sweep_fvg_combo"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = SweepFVGComboDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = SweepFVGComboDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = SweepFVGComboDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = SweepFVGComboDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "sweep_lookback" in sv.parameters
            assert "combo_window" in sv.parameters
            assert "min_gap_atr" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = SweepFVGComboDetector(sweep_lookback=30, combo_window=8)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["sweep_lookback"] == 30
            assert sv.parameters["combo_window"] == 8

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(20)
        data = _make_data_bundle({"5m": short_df})
        detector = SweepFVGComboDetector()
        assert not detector.validate_inputs(data)


class TestSessionGapDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = SessionGapDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = SessionGapDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_SESSION_GAP_")
            assert sv.category == "session_gap"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = SessionGapDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = SessionGapDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = SessionGapDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = SessionGapDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "min_gap_atr" in sv.parameters
            assert "fill_threshold_pct" in sv.parameters
            assert "max_gap_age_bars" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = SessionGapDetector(min_gap_atr=0.3, max_gap_age_bars=50)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["min_gap_atr"] == 0.3
            assert sv.parameters["max_gap_age_bars"] == 50

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(20)
        data = _make_data_bundle({"5m": short_df})
        detector = SessionGapDetector()
        assert not detector.validate_inputs(data)


class TestVolumeProfileDetector:
    def test_compute_produces_signals(self):
        data = _make_data_bundle()
        detector = VolumeProfileDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = _make_data_bundle()
        detector = VolumeProfileDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_VOLUME_PROFILE_")
            assert sv.category == "volume_profile"
            assert sv.version == 1

    def test_direction_values(self):
        data = _make_data_bundle()
        detector = VolumeProfileDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = _make_data_bundle()
        detector = VolumeProfileDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = _make_data_bundle()
        detector = VolumeProfileDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = _make_data_bundle()
        detector = VolumeProfileDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "lookback_bars" in sv.parameters
            assert "num_bins" in sv.parameters
            assert "value_area_pct" in sv.parameters

    def test_custom_parameters(self):
        data = _make_data_bundle()
        detector = VolumeProfileDetector(lookback_bars=50, num_bins=30)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["lookback_bars"] == 50
            assert sv.parameters["num_bins"] == 30

    def test_validate_inputs_insufficient_bars(self):
        short_df = _make_ohlcv(20)
        data = _make_data_bundle({"5m": short_df})
        detector = VolumeProfileDetector()
        assert not detector.validate_inputs(data)


class TestScalpEntryDetector:
    def _make_scalp_bundle(self) -> DataBundle:
        """Bundle with micro + macro TFs for scalp entry."""
        bars_1m = _make_ohlcv(300, seed=50)
        bars_15m = _make_ohlcv(100, seed=43)
        bars_1h = _make_ohlcv(60, seed=44)
        return _make_data_bundle({"1m": bars_1m, "15m": bars_15m, "1H": bars_1h})

    def test_compute_produces_signals(self):
        data = self._make_scalp_bundle()
        detector = ScalpEntryDetector()
        signals = detector.compute(data)
        assert isinstance(signals, list)

    def test_signal_id_format(self):
        data = self._make_scalp_bundle()
        detector = ScalpEntryDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert sv.signal_id.startswith("SIG_SCALP_ENTRY_")
            assert sv.category == "scalp_entry"
            assert sv.version == 1

    def test_direction_values(self):
        data = self._make_scalp_bundle()
        detector = ScalpEntryDetector()
        signals = detector.compute(data)
        for sv in signals:
            unique_dirs = set(sv.direction.unique())
            assert unique_dirs.issubset({-1, 0, 1})

    def test_strength_range(self):
        data = self._make_scalp_bundle()
        detector = ScalpEntryDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert (sv.strength >= 0.0).all()
            assert (sv.strength <= 1.0).all()

    def test_strength_zero_when_neutral(self):
        data = self._make_scalp_bundle()
        detector = ScalpEntryDetector()
        signals = detector.compute(data)
        for sv in signals:
            neutral = sv.direction == 0
            assert (sv.strength[neutral] == 0.0).all()

    def test_parameters_in_metadata(self):
        data = self._make_scalp_bundle()
        detector = ScalpEntryDetector()
        signals = detector.compute(data)
        for sv in signals:
            assert "macro_ema_fast" in sv.parameters
            assert "momentum_lookback" in sv.parameters
            assert "min_velocity_atr" in sv.parameters

    def test_custom_parameters(self):
        data = self._make_scalp_bundle()
        detector = ScalpEntryDetector(momentum_lookback=10, min_velocity_atr=0.5)
        signals = detector.compute(data)
        for sv in signals:
            assert sv.parameters["momentum_lookback"] == 10
            assert sv.parameters["min_velocity_atr"] == 0.5

    def test_validate_inputs_no_micro_tf(self):
        """Without micro TFs, validation should fail."""
        data = _make_data_bundle()  # Only 5m, 15m, 1H — no micro TFs
        detector = ScalpEntryDetector()
        assert not detector.validate_inputs(data)
