"""Tests for the Live Monitoring agent (MON-001)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alpha_lab.agents.monitoring.agent import MonitoringAgent, SignalHealthState
from alpha_lab.agents.monitoring.alerts import (
    check_hit_rate_degradation,
    check_ic_degradation,
    check_prop_firm_buffer,
    check_sharpe_degradation,
    evaluate_all_alerts,
)
from alpha_lab.agents.monitoring.dashboard import (
    assemble_daily_report,
    assemble_realtime_update,
)
from alpha_lab.agents.monitoring.metrics import (
    compute_decay_velocity,
    compute_rolling_hit_rate,
    compute_rolling_ic,
    compute_rolling_sharpe,
    compute_slippage_tracking,
)
from alpha_lab.agents.monitoring.regime import (
    classify_regime,
    compute_regime_signal_weights,
    detect_regime_transition,
)
from alpha_lab.core.contracts import Alert, SignalHealthReport
from alpha_lab.core.enums import AgentID, MessageType, Regime
from alpha_lab.core.message import MessageEnvelope

# ─── Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def sample_signal_health():
    return SignalHealthReport(
        signal_id="SIG_TEST_v1",
        live_ic=0.04,
        backtest_ic=0.05,
        ic_ratio=0.8,
        live_hit_rate=0.56,
        live_sharpe=1.2,
        gross_pnl_today=150.0,
        net_pnl_today=120.0,
        trades_today=8,
        realized_slippage=2.5,
        status="HEALTHY",
    )


@pytest.fixture
def sample_alert():
    return Alert(
        level="WARNING",
        metric="ic_degradation",
        current_value=0.02,
        threshold=0.025,
        backtest_value=0.05,
        message="IC degraded",
        recommended_action="Review signal",
        timestamp="2026-01-15T12:00:00+00:00",
    )


@pytest.fixture
def trending_market_data():
    return {
        "ema_values": [22200.0, 22100.0, 22000.0],  # fast > mid > slow
        "kama_slope": 0.5,
        "atr_current": 50.0,
        "atr_avg": 50.0,
        "adx": 30.0,
    }


@pytest.fixture
def ranging_market_data():
    return {
        "ema_values": [22050.0, 22050.5, 22049.5],  # tightly clustered
        "kama_slope": 0.02,
        "atr_current": 30.0,
        "atr_avg": 30.0,
        "adx": 15.0,
    }


# ═══════════════════════════════════════════════════════════════
#  METRICS TESTS
# ═══════════════════════════════════════════════════════════════


class TestRollingIC:
    def test_perfect_correlation(self):
        signal = list(range(50))
        returns = [x * 2.0 for x in signal]
        result = compute_rolling_ic(signal, returns, window=20)
        assert len(result) > 0
        # Perfect linear correlation
        assert all(abs(v - 1.0) < 0.01 for v in result)

    def test_zero_correlation(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 100).tolist()
        returns = rng.normal(0, 1, 100).tolist()
        result = compute_rolling_ic(signal, returns, window=20)
        assert len(result) > 0
        # Mean IC should be near zero for independent noise
        assert abs(np.mean(result)) < 0.3

    def test_insufficient_data(self):
        result = compute_rolling_ic([1, 2, 3], [0.1, 0.2, 0.3], window=20)
        assert result == []

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_rolling_ic([1, 2], [1, 2, 3])


class TestRollingHitRate:
    def test_all_correct(self):
        preds = [1] * 60
        actuals = [1] * 60
        result = compute_rolling_hit_rate(preds, actuals, window=50)
        assert len(result) > 0
        assert all(v == 1.0 for v in result)

    def test_all_wrong(self):
        preds = [1] * 60
        actuals = [-1] * 60
        result = compute_rolling_hit_rate(preds, actuals, window=50)
        assert len(result) > 0
        assert all(v == 0.0 for v in result)

    def test_neutral_ignored(self):
        # 50 predictions: 25 correct, 25 neutral
        preds = [1] * 25 + [0] * 25 + [1] * 10
        actuals = [1] * 25 + [1] * 25 + [1] * 10
        result = compute_rolling_hit_rate(preds, actuals, window=50)
        assert len(result) > 0
        # Only non-zero predictions counted → all correct
        assert result[0] == 1.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_rolling_hit_rate([1, 2], [1])


class TestRollingSharpe:
    def test_positive_returns(self):
        # Slightly varying positive returns (constant returns → zero std → NaN)
        returns = [0.01 + 0.001 * (i % 3) for i in range(30)]
        result = compute_rolling_sharpe(returns, window=20)
        assert len(result) > 0
        assert all(v > 0 for v in result)

    def test_negative_returns(self):
        returns = [-0.01 + 0.001 * (i % 3) for i in range(30)]
        result = compute_rolling_sharpe(returns, window=20)
        assert len(result) > 0
        assert all(v < 0 for v in result)

    def test_insufficient_data(self):
        result = compute_rolling_sharpe([0.01] * 5, window=20)
        assert result == []


class TestDecayVelocity:
    def test_exponential_decay(self):
        # Create exponential decay: IC(t) = 0.1 * exp(-t/30)
        ic = [0.1 * math.exp(-t / 30) for t in range(60)]
        result = compute_decay_velocity(ic, expected_half_life=20.0)
        assert result["classification"] in (
            "as_expected",
            "faster_than_expected",
            "slower_than_expected",
        )
        assert result["actual_half_life"] > 0
        assert result["expected_half_life"] == 20.0

    def test_insufficient_data(self):
        result = compute_decay_velocity([0.1, 0.09, 0.08], expected_half_life=10.0)
        assert result["classification"] == "insufficient_data"
        assert math.isnan(result["actual_half_life"])


class TestSlippageTracking:
    def test_basic(self):
        expected = [22000.0, 22050.0, 22100.0]
        realized = [22001.0, 22051.5, 22102.0]
        result = compute_slippage_tracking(expected, realized)
        assert result["num_trades"] == 3
        assert result["avg_slippage"] > 0
        assert result["worst_slippage"] >= result["avg_slippage"]

    def test_perfect_fills(self):
        prices = [22000.0, 22050.0]
        result = compute_slippage_tracking(prices, prices)
        assert result["avg_slippage"] == 0.0
        assert result["worst_slippage"] == 0.0

    def test_empty(self):
        result = compute_slippage_tracking([], [])
        assert result["num_trades"] == 0
        assert result["avg_slippage"] == 0.0

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_slippage_tracking([1.0], [1.0, 2.0])


# ═══════════════════════════════════════════════════════════════
#  ALERTS TESTS
# ═══════════════════════════════════════════════════════════════


class TestICDegradation:
    def test_triggered(self):
        alert = check_ic_degradation(live_ic=0.02, backtest_ic=0.05, bars_below=25)
        assert alert is not None
        assert alert.level == "WARNING"
        assert alert.metric == "ic_degradation"

    def test_not_triggered_bars(self):
        alert = check_ic_degradation(live_ic=0.02, backtest_ic=0.05, bars_below=19)
        assert alert is None

    def test_not_triggered_ratio(self):
        alert = check_ic_degradation(live_ic=0.04, backtest_ic=0.05, bars_below=25)
        assert alert is None  # 0.04/0.05 = 0.80 > 0.50

    def test_zero_backtest_ic(self):
        alert = check_ic_degradation(live_ic=0.01, backtest_ic=0.0, bars_below=30)
        assert alert is None


class TestHitRateDegradation:
    def test_triggered(self):
        alert = check_hit_rate_degradation(
            live_hit_rate=0.46, consecutive_windows_below=3
        )
        assert alert is not None
        assert alert.level == "WARNING"

    def test_not_triggered_windows(self):
        alert = check_hit_rate_degradation(
            live_hit_rate=0.46, consecutive_windows_below=1
        )
        assert alert is None

    def test_not_triggered_rate(self):
        alert = check_hit_rate_degradation(
            live_hit_rate=0.49, consecutive_windows_below=3
        )
        assert alert is None


class TestSharpeDegradation:
    def test_triggered(self):
        alert = check_sharpe_degradation(live_sharpe=0.3, days_below=25)
        assert alert is not None
        assert alert.level == "CRITICAL"

    def test_not_triggered(self):
        alert = check_sharpe_degradation(live_sharpe=0.3, days_below=19)
        assert alert is None

    def test_not_triggered_sharpe_ok(self):
        alert = check_sharpe_degradation(live_sharpe=0.6, days_below=25)
        assert alert is None


class TestPropFirmBuffer:
    def test_halt_daily_buffer(self):
        alert = check_prop_firm_buffer(dd_buffer_pct=0.50, daily_buffer_pct=0.15)
        assert alert is not None
        assert alert.level == "HALT"

    def test_critical_dd_buffer(self):
        alert = check_prop_firm_buffer(dd_buffer_pct=0.25, daily_buffer_pct=0.90)
        assert alert is not None
        assert alert.level == "CRITICAL"

    def test_warning_daily_buffer(self):
        alert = check_prop_firm_buffer(dd_buffer_pct=0.50, daily_buffer_pct=0.75)
        assert alert is not None
        assert alert.level == "WARNING"

    def test_healthy(self):
        alert = check_prop_firm_buffer(dd_buffer_pct=0.80, daily_buffer_pct=0.90)
        assert alert is None


class TestEvaluateAllAlerts:
    def test_multiple_alerts(self):
        metrics = {
            "live_ic": 0.01,
            "backtest_ic": 0.05,
            "bars_below_ic": 25,
            "live_sharpe": 0.3,
            "days_below_sharpe": 25,
        }
        alerts = evaluate_all_alerts(metrics)
        assert len(alerts) >= 2  # IC warning + Sharpe critical

    def test_sorted_by_severity(self):
        metrics = {
            "live_ic": 0.01,
            "backtest_ic": 0.05,
            "bars_below_ic": 25,
            "live_sharpe": 0.3,
            "days_below_sharpe": 25,
        }
        alerts = evaluate_all_alerts(metrics)
        levels = [a.level for a in alerts]
        # CRITICAL should come before WARNING
        if "CRITICAL" in levels and "WARNING" in levels:
            assert levels.index("CRITICAL") < levels.index("WARNING")

    def test_no_alerts(self):
        metrics = {
            "live_ic": 0.04,
            "backtest_ic": 0.05,
            "bars_below_ic": 0,
            "live_hit_rate": 0.55,
            "live_sharpe": 1.5,
            "dd_buffer_pct": 0.80,
            "daily_buffer_pct": 0.90,
        }
        alerts = evaluate_all_alerts(metrics)
        assert alerts == []


# ═══════════════════════════════════════════════════════════════
#  REGIME TESTS
# ═══════════════════════════════════════════════════════════════


class TestClassifyRegime:
    def test_trending(self, trending_market_data):
        regime, confidence = classify_regime(trending_market_data)
        assert regime == Regime.TRENDING
        assert confidence > 0.4

    def test_ranging(self, ranging_market_data):
        regime, confidence = classify_regime(ranging_market_data)
        assert regime == Regime.RANGING
        assert confidence > 0.4

    def test_volatile(self):
        # Non-aligned EMAs with wide spread (cv > 0.005) avoids RANGING/TRENDING
        # Moderate slope/ADX avoids TRENDING. ATR spike drives VOLATILE.
        data = {
            "ema_values": [22500.0, 22000.0, 22200.0],  # not aligned, wide spread
            "kama_slope": 0.2,  # moderate: no TRENDING (+0), no RANGING (+0)
            "atr_current": 120.0,
            "atr_avg": 50.0,  # 2.4x spike → VOLATILE +5
            "adx": 22.0,  # moderate: no TRENDING (+0), no RANGING (+0)
        }
        regime, confidence = classify_regime(data)
        assert regime == Regime.VOLATILE

    def test_transitional(self):
        # Non-aligned EMAs with wide spread + moderate slope/ADX + normal ATR
        data = {
            "ema_values": [22500.0, 22000.0, 22200.0],  # not aligned, wide spread
            "kama_slope": 0.2,  # moderate (0.1-0.3 → TRANSITIONAL +2)
            "atr_current": 50.0,
            "atr_avg": 50.0,  # normal ratio → no VOLATILE
            "adx": 22.0,  # moderate (15-25 → TRANSITIONAL +2)
        }
        regime, confidence = classify_regime(data)
        assert regime == Regime.TRANSITIONAL

    def test_empty_data_defaults_ranging(self):
        regime, confidence = classify_regime({})
        assert regime == Regime.RANGING


class TestRegimeTransition:
    def test_triggered(self):
        result = detect_regime_transition(
            Regime.RANGING, Regime.TRENDING, confidence=0.75
        )
        assert result is not None
        assert result["from"] == "RANGING"
        assert result["to"] == "TRENDING"

    def test_same_regime_no_transition(self):
        result = detect_regime_transition(
            Regime.TRENDING, Regime.TRENDING, confidence=0.9
        )
        assert result is None

    def test_low_confidence_no_transition(self):
        result = detect_regime_transition(
            Regime.RANGING, Regime.TRENDING, confidence=0.4
        )
        assert result is None


class TestRegimeWeights:
    def test_trending_weights(self):
        weights = compute_regime_signal_weights(Regime.TRENDING)
        assert weights["trend_following"] > 1.0
        assert weights["mean_reversion"] < 1.0

    def test_ranging_weights(self):
        weights = compute_regime_signal_weights(Regime.RANGING)
        assert weights["mean_reversion"] > 1.0
        assert weights["trend_following"] < 1.0

    def test_volatile_reduces_all(self):
        weights = compute_regime_signal_weights(Regime.VOLATILE)
        assert all(v <= 1.0 for v in weights.values())

    def test_transitional_reduces_all(self):
        weights = compute_regime_signal_weights(Regime.TRANSITIONAL)
        assert all(v <= 1.0 for v in weights.values())


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD TESTS
# ═══════════════════════════════════════════════════════════════


class TestDashboard:
    def test_daily_report(self, sample_signal_health, sample_alert):
        report = assemble_daily_report(
            signal_health=[sample_signal_health],
            alerts=[sample_alert],
            regime={"current": "TRENDING", "confidence": 0.8},
            prop_firm_status={"dd_buffer": 0.7},
        )
        assert report.report_type == "DAILY_SUMMARY"
        assert len(report.signals) == 1
        assert len(report.alerts) == 1
        assert report.request_id  # auto-generated

    def test_realtime_update(self, sample_signal_health):
        report = assemble_realtime_update(
            signal_health=[sample_signal_health],
            active_alerts=[],
        )
        assert report.report_type == "REALTIME"
        assert report.regime == {}
        assert report.prop_firm_status == {}

    def test_recommendations_with_halt_alert(self, sample_signal_health):
        halt_alert = Alert(
            level="HALT",
            metric="daily_loss_buffer",
            current_value=0.15,
            threshold=0.20,
            backtest_value=0.0,
            message="Daily loss limit approaching",
            recommended_action="HALT",
            timestamp="2026-01-15T12:00:00+00:00",
        )
        report = assemble_daily_report(
            signal_health=[sample_signal_health],
            alerts=[halt_alert],
            regime={},
            prop_firm_status={},
        )
        assert any("HALT" in r for r in report.recommendations)

    def test_recommendations_healthy(self, sample_signal_health):
        report = assemble_daily_report(
            signal_health=[sample_signal_health],
            alerts=[],
            regime={},
            prop_firm_status={},
        )
        assert any("normal operations" in r for r in report.recommendations)


# ═══════════════════════════════════════════════════════════════
#  AGENT TESTS
# ═══════════════════════════════════════════════════════════════


class TestMonitoringAgent:
    def test_create(self, message_bus):
        agent = MonitoringAgent(message_bus)
        assert agent.agent_id == AgentID.MONITORING
        assert agent.name == "Live Monitoring"

    def test_initial_state(self, message_bus):
        agent = MonitoringAgent(message_bus)
        assert agent.active_signals == {}
        assert agent.current_regime == Regime.RANGING

    def test_deploy_command(self, message_bus):
        agent = MonitoringAgent(message_bus)
        envelope = MessageEnvelope(
            request_id="deploy-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.MONITORING,
            message_type=MessageType.DEPLOY_COMMAND,
            payload={
                "approved_signals": [
                    {
                        "signal_id": "SIG_TEST_v1",
                        "ic": 0.06,
                        "hit_rate": 0.58,
                        "sharpe": 1.8,
                        "risk_parameters": {"max_contracts": 4},
                    }
                ]
            },
        )
        agent.handle_message(envelope)
        assert "SIG_TEST_v1" in agent.active_signals
        state = agent.active_signals["SIG_TEST_v1"]
        assert state.backtest_ic == 0.06
        assert state.backtest_sharpe == 1.8

    def test_unexpected_message_nack(self, message_bus):
        agent = MonitoringAgent(message_bus)
        envelope = MessageEnvelope(
            request_id="test-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.MONITORING,
            message_type=MessageType.DATA_BUNDLE,
            payload={},
        )
        agent.handle_message(envelope)
        # Should have sent NACK
        audit = message_bus.get_audit_log()
        nack_msgs = [
            e for e in audit if e.message_type == MessageType.NACK
        ]
        assert len(nack_msgs) >= 1

    def test_update_metrics(self, message_bus):
        agent = MonitoringAgent(message_bus)
        agent._active_signals["SIG_TEST"] = SignalHealthState(
            signal_id="SIG_TEST", backtest_ic=0.05
        )
        agent.update_metrics(
            "SIG_TEST", live_ic=0.02, live_hit_rate=0.46, trades_today=5
        )
        state = agent._active_signals["SIG_TEST"]
        assert state.live_ic == 0.02
        assert state.live_hit_rate == 0.46
        assert state.trades_today == 5
        # IC below threshold → counter incremented
        assert state.bars_below_ic_threshold == 1

    def test_check_signal_health(self, message_bus):
        agent = MonitoringAgent(message_bus)
        agent._active_signals["SIG_TEST"] = SignalHealthState(
            signal_id="SIG_TEST",
            backtest_ic=0.05,
            live_ic=0.04,
            live_hit_rate=0.56,
            live_sharpe=1.2,
        )
        report = agent.check_signal_health("SIG_TEST")
        assert report.signal_id == "SIG_TEST"
        assert report.status == "HEALTHY"
        assert report.ic_ratio == pytest.approx(0.8)

    def test_check_signal_health_degrading(self, message_bus):
        agent = MonitoringAgent(message_bus)
        agent._active_signals["SIG_TEST"] = SignalHealthState(
            signal_id="SIG_TEST",
            backtest_ic=0.05,
            live_ic=0.02,
            bars_below_ic_threshold=10,  # >= 5 → DEGRADING
        )
        report = agent.check_signal_health("SIG_TEST")
        assert report.status == "DEGRADING"

    def test_check_signal_health_failing(self, message_bus):
        agent = MonitoringAgent(message_bus)
        agent._active_signals["SIG_TEST"] = SignalHealthState(
            signal_id="SIG_TEST",
            backtest_ic=0.05,
            bars_below_ic_threshold=25,  # >= 20 → FAILING
        )
        report = agent.check_signal_health("SIG_TEST")
        assert report.status == "FAILING"

    def test_check_unknown_signal_raises(self, message_bus):
        agent = MonitoringAgent(message_bus)
        with pytest.raises(KeyError, match="not tracked"):
            agent.check_signal_health("NONEXISTENT")

    def test_generate_daily_report(self, message_bus):
        agent = MonitoringAgent(message_bus)
        agent._active_signals["SIG_TEST"] = SignalHealthState(
            signal_id="SIG_TEST",
            backtest_ic=0.05,
            live_ic=0.04,
            live_hit_rate=0.55,
            live_sharpe=1.2,
        )
        report = agent.generate_daily_report()
        assert report.report_type == "DAILY_SUMMARY"
        assert len(report.signals) == 1
        assert report.regime["current"] == "RANGING"

    def test_update_regime_with_transition(self, message_bus):
        agent = MonitoringAgent(message_bus)
        assert agent.current_regime == Regime.RANGING

        trending_data = {
            "ema_values": [22200.0, 22100.0, 22000.0],
            "kama_slope": 0.5,
            "atr_current": 50.0,
            "atr_avg": 50.0,
            "adx": 30.0,
        }
        transition = agent.update_regime(trending_data)
        assert agent.current_regime == Regime.TRENDING
        assert transition is not None
        assert transition["from"] == "RANGING"
        assert transition["to"] == "TRENDING"

    def test_evaluate_alerts_escalates(self, message_bus):
        agent = MonitoringAgent(message_bus)
        agent._active_signals["SIG_TEST"] = SignalHealthState(
            signal_id="SIG_TEST",
            backtest_ic=0.05,
            live_ic=0.01,
            bars_below_ic_threshold=25,
            live_sharpe=0.3,
            days_below_sharpe_threshold=25,
        )
        alerts = agent.evaluate_alerts()
        assert len(alerts) >= 1
        # CRITICAL sharpe alert should have been escalated
        audit = message_bus.get_audit_log()
        alert_msgs = [e for e in audit if e.message_type == MessageType.ALERT]
        assert len(alert_msgs) >= 1
