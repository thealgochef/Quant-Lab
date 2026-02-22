"""Tests for Statistical Validation agent, firewall, and test battery."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.validation.agent import ValidationAgent
from alpha_lab.agents.validation.firewall import (
    ValidationTest,
    assemble_verdict,
    compute_forward_returns,
    run_test_battery,
    strip_signal_metadata,
)
from alpha_lab.agents.validation.tests.decay_analysis import DecayAnalysisTest
from alpha_lab.agents.validation.tests.hit_rate import HitRateTest
from alpha_lab.agents.validation.tests.ic_testing import ICTest
from alpha_lab.agents.validation.tests.orthogonality import OrthogonalityTest
from alpha_lab.agents.validation.tests.risk_adjusted import RiskAdjustedTest
from alpha_lab.agents.validation.tests.robustness import RobustnessTest
from alpha_lab.core.contracts import (
    SignalBundle,
    SignalVector,
)
from alpha_lab.core.enums import AgentID, AgentState, MessageType, Priority
from alpha_lab.core.message import MessageEnvelope

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


def _make_price_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV with a slight upward trend."""
    rng = np.random.default_rng(seed)
    close = 22000 + np.cumsum(rng.normal(0.3, 2.0, n))
    high = close + rng.uniform(2, 10, n)
    low = close - rng.uniform(2, 10, n)
    volume = rng.integers(500, 3000, n).astype(float)

    idx = pd.date_range("2026-02-20 09:30", periods=n, freq="5min", tz="US/Eastern")
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 1, n),
            "high": high, "low": low, "close": close, "volume": volume,
        },
        index=idx,
    )


def _make_good_signal(price_data: pd.DataFrame) -> SignalVector:
    """Create a signal that has real predictive power (correlated with future returns)."""
    close = price_data["close"]
    # Cheat slightly: use a smoothed version of future returns as signal
    fwd = close.shift(-5) / close - 1
    direction = pd.Series(np.sign(fwd.fillna(0)), index=close.index).astype(int)
    strength = fwd.abs().fillna(0).clip(0, 1)
    # Add some noise so it's not perfect
    rng = np.random.default_rng(99)
    noise = rng.choice([-1, 0, 1], len(close))
    flip_mask = rng.random(len(close)) < 0.15
    direction_noisy = direction.copy()
    direction_noisy[flip_mask] = noise[flip_mask]

    return SignalVector(
        signal_id="SIG_TEST_5m_v1",
        category="test",
        timeframe="5m",
        version=1,
        direction=direction_noisy,
        strength=strength.clip(0.1, 0.9),
        formation_idx=pd.Series(range(len(close)), index=close.index),
        parameters={"lookback": 5},
        metadata={"test": True},
    )


def _make_random_signal(price_data: pd.DataFrame) -> SignalVector:
    """Create a random signal with no predictive power."""
    rng = np.random.default_rng(123)
    n = len(price_data)
    direction = pd.Series(rng.choice([-1, 0, 1], n), index=price_data.index)
    strength = pd.Series(rng.uniform(0.1, 0.9, n), index=price_data.index)

    return SignalVector(
        signal_id="SIG_RANDOM_5m_v1",
        category="random",
        timeframe="5m",
        version=1,
        direction=direction,
        strength=strength,
        formation_idx=pd.Series(range(n), index=price_data.index),
        parameters={},
        metadata={},
    )


# ═══════════════════════════════════════════════════════════════════
# Firewall Tests
# ═══════════════════════════════════════════════════════════════════


class TestFirewall:
    def test_strip_signal_metadata(self):
        price = _make_price_data(50)
        sig = _make_good_signal(price)
        stripped = strip_signal_metadata(sig)

        assert "signal_id" in stripped
        assert "direction" in stripped
        assert "strength" in stripped
        assert "timeframe" in stripped
        # Blocked fields
        assert "parameters" not in stripped
        assert "category" not in stripped
        assert "metadata" not in stripped

    def test_compute_forward_returns(self):
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        fwd = compute_forward_returns(close, horizon=1)
        assert abs(fwd.iloc[0] - 0.01) < 1e-10
        assert pd.isna(fwd.iloc[-1])

    def test_compute_forward_returns_horizon_5(self):
        close = pd.Series([100.0] * 5 + [110.0])
        fwd = compute_forward_returns(close, horizon=5)
        assert abs(fwd.iloc[0] - 0.10) < 1e-10

    def test_validation_test_is_abstract(self):
        with pytest.raises(TypeError):
            ValidationTest()  # type: ignore[abstract]


class TestAssembleVerdict:
    def test_deploy_verdict(self):
        results = {
            "information_coefficient": {"ic_mean": 0.05, "ic_tstat": 3.0},
            "hit_rate": {
                "hit_rate_overall": 0.55, "hit_rate_long": 0.56,
                "hit_rate_short": 0.54,
            },
            "risk_adjusted": {
                "sharpe": 1.5, "sortino": 2.0,
                "max_drawdown": 0.08, "profit_factor": 1.5,
            },
            "decay_analysis": {"half_life": 30.0, "decay_class": "medium"},
            "orthogonality": {
                "max_factor_corr": 0.15, "incremental_r2": 0.01,
                "is_orthogonal": True,
            },
            "robustness": {"subsample_stable": True},
        }
        verdict = assemble_verdict("SIG_TEST_5m_v1", results)
        assert verdict.verdict == "DEPLOY"
        assert len(verdict.failed_metrics) == 0

    def test_refine_verdict_low_sharpe(self):
        results = {
            "information_coefficient": {"ic_mean": 0.05, "ic_tstat": 2.5},
            "hit_rate": {
                "hit_rate_overall": 0.53, "hit_rate_long": 0.53,
                "hit_rate_short": 0.53,
            },
            "risk_adjusted": {
                "sharpe": 0.7, "sortino": 0.9,
                "max_drawdown": 0.10, "profit_factor": 1.3,
            },
            "decay_analysis": {"half_life": 20.0, "decay_class": "medium"},
            "orthogonality": {
                "max_factor_corr": 0.20, "incremental_r2": 0.01,
                "is_orthogonal": True,
            },
            "robustness": {"subsample_stable": True},
        }
        verdict = assemble_verdict("SIG_TEST_5m_v1", results)
        assert verdict.verdict == "REFINE"
        failed_names = {m["metric"] for m in verdict.failed_metrics}
        assert "sharpe" in failed_names

    def test_reject_verdict_low_ic(self):
        results = {
            "information_coefficient": {"ic_mean": 0.005, "ic_tstat": 0.3},
            "hit_rate": {
                "hit_rate_overall": 0.50, "hit_rate_long": 0.50,
                "hit_rate_short": 0.50,
            },
            "risk_adjusted": {
                "sharpe": 0.2, "sortino": 0.1,
                "max_drawdown": 0.25, "profit_factor": 0.8,
            },
            "decay_analysis": {
                "half_life": 2.0, "decay_class": "ultra-fast",
            },
            "orthogonality": {
                "max_factor_corr": 0.60, "incremental_r2": 0.001,
                "is_orthogonal": False,
            },
            "robustness": {"subsample_stable": False},
        }
        verdict = assemble_verdict("SIG_RANDOM_5m_v1", results)
        assert verdict.verdict == "REJECT"


# ═══════════════════════════════════════════════════════════════════
# IC Testing
# ═══════════════════════════════════════════════════════════════════


class TestICTest:
    def test_ic_returns_metrics(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = ICTest(horizons=[1, 5, 10])
        result = test.evaluate(sig, price)

        assert "ic_mean" in result
        assert "ic_tstat" in result
        assert "ic_by_horizon" in result

    def test_ic_positive_for_good_signal(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = ICTest(horizons=[5])
        result = test.evaluate(sig, price)
        assert result["ic_mean"] > 0

    def test_ic_near_zero_for_random(self):
        price = _make_price_data(300)
        sig = _make_random_signal(price)
        test = ICTest(horizons=[5])
        result = test.evaluate(sig, price)
        assert abs(result["ic_mean"]) < 0.15

    def test_ic_insufficient_data(self):
        price = _make_price_data(20)
        sig = _make_random_signal(price)
        test = ICTest(horizons=[1, 5])
        result = test.evaluate(sig, price)
        assert result["ic_tstat"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Hit Rate Testing
# ═══════════════════════════════════════════════════════════════════


class TestHitRateTest:
    def test_hit_rate_returns_metrics(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = HitRateTest()
        result = test.evaluate(sig, price)

        assert "hit_rate_overall" in result
        assert "hit_rate_long" in result
        assert "hit_rate_short" in result
        assert "n_signals" in result

    def test_hit_rate_above_50_for_good_signal(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = HitRateTest(horizon=5)
        result = test.evaluate(sig, price)
        assert result["hit_rate_overall"] > 0.50

    def test_hit_rate_near_50_for_random(self):
        price = _make_price_data(300)
        sig = _make_random_signal(price)
        test = HitRateTest(horizon=5)
        result = test.evaluate(sig, price)
        assert 0.35 < result["hit_rate_overall"] < 0.65

    def test_hit_rate_insufficient_data(self):
        price = _make_price_data(10)
        sig = _make_random_signal(price)
        test = HitRateTest()
        result = test.evaluate(sig, price)
        assert result["n_signals"] == 0


# ═══════════════════════════════════════════════════════════════════
# Risk-Adjusted Return Testing
# ═══════════════════════════════════════════════════════════════════


class TestRiskAdjustedTest:
    def test_returns_metrics(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = RiskAdjustedTest()
        result = test.evaluate(sig, price)

        assert "sharpe" in result
        assert "sortino" in result
        assert "max_drawdown" in result
        assert "profit_factor" in result
        assert "calmar" in result

    def test_sharpe_positive_for_good_signal(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = RiskAdjustedTest(horizon=5)
        result = test.evaluate(sig, price)
        assert result["sharpe"] > 0

    def test_drawdown_bounded(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = RiskAdjustedTest()
        result = test.evaluate(sig, price)
        assert 0 <= result["max_drawdown"] <= 1.0

    def test_insufficient_data(self):
        price = _make_price_data(10)
        sig = _make_random_signal(price)
        test = RiskAdjustedTest()
        result = test.evaluate(sig, price)
        assert result["n_bars"] == 0


# ═══════════════════════════════════════════════════════════════════
# Decay Analysis Testing
# ═══════════════════════════════════════════════════════════════════


class TestDecayAnalysisTest:
    def test_returns_metrics(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = DecayAnalysisTest(horizons=[1, 5, 10, 30])
        result = test.evaluate(sig, price)

        assert "ic_curve" in result
        assert "half_life" in result
        assert "decay_class" in result

    def test_decay_class_valid(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = DecayAnalysisTest(horizons=[1, 5, 10, 30])
        result = test.evaluate(sig, price)
        assert result["decay_class"] in [
            "ultra-fast", "fast", "medium", "slow", "persistent"
        ]

    def test_half_life_positive(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = DecayAnalysisTest(horizons=[1, 5, 10, 30])
        result = test.evaluate(sig, price)
        assert result["half_life"] >= 0


# ═══════════════════════════════════════════════════════════════════
# Orthogonality Testing
# ═══════════════════════════════════════════════════════════════════


class TestOrthogonalityTest:
    def test_returns_metrics(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = OrthogonalityTest()
        result = test.evaluate(sig, price)

        assert "factor_correlations" in result
        assert "max_factor_corr" in result
        assert "incremental_r2" in result
        assert "is_orthogonal" in result

    def test_factor_corrs_bounded(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = OrthogonalityTest()
        result = test.evaluate(sig, price)
        for corr in result["factor_correlations"].values():
            assert -1.0 <= corr <= 1.0

    def test_random_signal_low_corr(self):
        price = _make_price_data(300)
        sig = _make_random_signal(price)
        test = OrthogonalityTest()
        result = test.evaluate(sig, price)
        # Random signal shouldn't correlate highly with factors
        assert result["max_factor_corr"] < 0.5


# ═══════════════════════════════════════════════════════════════════
# Robustness Testing
# ═══════════════════════════════════════════════════════════════════


class TestRobustnessTest:
    def test_returns_metrics(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = RobustnessTest()
        result = test.evaluate(sig, price)

        assert "subsample_ics" in result
        assert "subsample_stable" in result
        assert "regime_ics" in result

    def test_subsample_count(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        test = RobustnessTest(n_subsamples=4)
        result = test.evaluate(sig, price)
        assert result["n_subsamples_computed"] <= 4

    def test_insufficient_data(self):
        price = _make_price_data(30)
        sig = _make_random_signal(price)
        test = RobustnessTest(n_subsamples=4)
        result = test.evaluate(sig, price)
        assert not result["subsample_stable"]


# ═══════════════════════════════════════════════════════════════════
# Run Test Battery Integration
# ═══════════════════════════════════════════════════════════════════


class TestRunTestBattery:
    def test_runs_all_tests(self):
        price = _make_price_data(300)
        sig = _make_good_signal(price)
        tests = [ICTest(horizons=[5]), HitRateTest(), RiskAdjustedTest()]
        results = run_test_battery(sig, price, tests)
        assert "information_coefficient" in results
        assert "hit_rate" in results
        assert "risk_adjusted" in results

    def test_handles_test_error(self):
        """A failing test should not crash the battery."""
        price = _make_price_data(5)  # Very short, will cause issues
        sig = _make_random_signal(price)
        tests = [ICTest(), HitRateTest(), RiskAdjustedTest()]
        results = run_test_battery(sig, price, tests)
        # Should still return results (possibly with defaults)
        assert len(results) == 3


# ═══════════════════════════════════════════════════════════════════
# Agent Tests
# ═══════════════════════════════════════════════════════════════════


class TestValidationAgent:
    def test_create(self, message_bus):
        agent = ValidationAgent(message_bus)
        assert agent.agent_id == AgentID.VALIDATION
        assert agent.name == "Statistical Validation"

    def test_validate_signal_bundle(self, message_bus):
        agent = ValidationAgent(message_bus)
        price = _make_price_data(300)
        sig = _make_good_signal(price)

        bundle = SignalBundle(
            instrument="NQ",
            signals=[sig],
            composite_scores={},
            timeframes_covered=["5m"],
            total_signals=1,
            generation_timestamp=datetime.now(UTC).isoformat(),
        )

        report = agent.validate_signal_bundle(bundle, {"5m": price})
        assert report.request_id == bundle.request_id
        assert len(report.verdicts) == 1
        assert report.verdicts[0].signal_id == "SIG_TEST_5m_v1"
        assert report.verdicts[0].verdict in ("DEPLOY", "REFINE", "REJECT")

    def test_validate_no_price_data(self, message_bus):
        """Missing price data should skip signals gracefully."""
        agent = ValidationAgent(message_bus)
        price = _make_price_data(300)
        sig = _make_good_signal(price)

        bundle = SignalBundle(
            instrument="NQ",
            signals=[sig],
            composite_scores={},
            timeframes_covered=["5m"],
            total_signals=1,
            generation_timestamp=datetime.now(UTC).isoformat(),
        )

        report = agent.validate_signal_bundle(bundle, {})
        assert len(report.verdicts) == 0
        assert "missing price data" in report.overall_assessment.lower()

    def test_handle_signal_bundle_message(self, message_bus):
        agent = ValidationAgent(message_bus)
        message_bus.register_agent(AgentID.VALIDATION, agent.handle_message)

        price = _make_price_data(300)
        sig = _make_good_signal(price)
        bundle = SignalBundle(
            instrument="NQ",
            signals=[sig],
            composite_scores={},
            timeframes_covered=["5m"],
            total_signals=1,
            generation_timestamp=datetime.now(UTC).isoformat(),
        )

        envelope = MessageEnvelope(
            request_id="test-val-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.VALIDATION,
            message_type=MessageType.SIGNAL_BUNDLE,
            priority=Priority.NORMAL,
            payload={"bundle": bundle.model_dump(), "price_data": {"5m": price}},
        )

        agent.handle_message(envelope)
        assert agent.state == AgentState.IDLE

        log = message_bus.get_audit_log()
        val_msgs = [m for m in log if m.message_type == MessageType.VALIDATION_REPORT]
        assert len(val_msgs) >= 1

    def test_handle_unexpected_message(self, message_bus):
        agent = ValidationAgent(message_bus)
        message_bus.register_agent(AgentID.VALIDATION, agent.handle_message)

        envelope = MessageEnvelope(
            request_id="test-val-002",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.VALIDATION,
            message_type=MessageType.HALT_COMMAND,
            priority=Priority.NORMAL,
            payload={},
        )

        agent.handle_message(envelope)
        nacks = [m for m in message_bus.get_audit_log() if m.message_type == MessageType.NACK]
        assert len(nacks) >= 1

    def test_bonferroni_raises_bar(self, message_bus):
        """Multiple signals should raise the IC t-stat threshold."""
        agent = ValidationAgent(message_bus)
        thresholds_1 = agent._apply_bonferroni(1)
        thresholds_10 = agent._apply_bonferroni(10)
        assert thresholds_10["ic_tstat_min"] > thresholds_1["ic_tstat_min"]
