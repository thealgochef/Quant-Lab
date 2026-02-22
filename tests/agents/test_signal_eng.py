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
from alpha_lab.agents.signal_eng.indicators import (
    compute_atr,
    compute_ema,
    compute_kama,
    compute_kama_efficiency_ratio,
    compute_session_vwap,
    compute_session_vwap_bands,
)
from alpha_lab.core.contracts import DataBundle, QualityReport, SessionMetadata
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

    def test_run_single_detector_stub_raises(self):
        """Stub detectors should raise NotImplementedError."""
        from alpha_lab.agents.signal_eng.detectors.tier2.liquidity_sweeps import (
            LiquiditySweepsDetector,
        )

        data = _make_data_bundle()
        with pytest.raises(NotImplementedError):
            run_single_detector(LiquiditySweepsDetector, data)

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
