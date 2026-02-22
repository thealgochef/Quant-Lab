"""Tests for the Execution & Risk agent (EXEC-001)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.execution.agent import ExecutionAgent
from alpha_lab.agents.execution.cost_model import (
    compute_cost_analysis,
    compute_round_turn_cost,
    compute_turnover_metrics,
    estimate_trade_stats,
)
from alpha_lab.agents.execution.monte_carlo import (
    compute_expected_value,
    simulate_ruin_probability,
)
from alpha_lab.agents.execution.position_sizing import (
    half_kelly_contracts,
    kelly_fraction,
    max_contracts_from_daily_limit,
    recommended_contracts,
)
from alpha_lab.agents.execution.prop_constraints import (
    validate_prop_firm_constraints,
)
from alpha_lab.core.config import InstrumentSpec, PropFirmProfile
from alpha_lab.core.contracts import (
    ExecutionReport,
    SignalVerdict,
    ValidationReport,
)
from alpha_lab.core.enums import AgentID, MessageType, Priority
from alpha_lab.core.message import MessageEnvelope

# ─── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def nq_spec() -> InstrumentSpec:
    """NQ instrument specification."""
    return InstrumentSpec(
        full_name="E-mini Nasdaq-100 Futures",
        exchange="CME",
        tick_size=0.25,
        tick_value=5.00,
        point_value=20.00,
        exchange_nfa_per_side=2.14,
        broker_commission_per_side=0.50,
        avg_slippage_ticks=0.5,
        avg_slippage_per_side=2.50,
        total_round_turn=7.78,
        session_open="18:00",
        session_close="17:00",
        rth_open="09:30",
        rth_close="16:15",
    )


@pytest.fixture
def es_spec() -> InstrumentSpec:
    """ES instrument specification."""
    return InstrumentSpec(
        full_name="E-mini S&P 500 Futures",
        exchange="CME",
        tick_size=0.25,
        tick_value=12.50,
        point_value=50.00,
        exchange_nfa_per_side=2.14,
        broker_commission_per_side=0.50,
        avg_slippage_ticks=0.25,
        avg_slippage_per_side=3.13,
        total_round_turn=8.41,
        session_open="18:00",
        session_close="17:00",
        rth_open="09:30",
        rth_close="16:15",
    )


@pytest.fixture
def apex_50k() -> PropFirmProfile:
    """Apex Trader Funding 50K profile."""
    return PropFirmProfile(
        name="Apex Trader Funding 50K",
        account_size=50000,
        daily_loss_limit=None,
        trailing_max_drawdown=2500,
        drawdown_type="real_time",
        max_contracts=4,
        consistency_rule_pct=30,
        profit_target=3000,
    )


@pytest.fixture
def topstep_50k() -> PropFirmProfile:
    """Topstep 50K profile."""
    return PropFirmProfile(
        name="Topstep 50K",
        account_size=50000,
        daily_loss_limit=1000,
        trailing_max_drawdown=2000,
        drawdown_type="end_of_day",
        max_contracts=5,
        consistency_rule_pct=50,
        profit_target=3000,
    )


@pytest.fixture
def sample_deploy_verdict() -> SignalVerdict:
    """A DEPLOY-grade signal verdict."""
    return SignalVerdict(
        signal_id="SIG_EMA_CONFLUENCE_5m_v1",
        verdict="DEPLOY",
        ic=0.05,
        ic_tstat=2.5,
        hit_rate=0.55,
        hit_rate_long=0.57,
        hit_rate_short=0.53,
        sharpe=1.5,
        sortino=2.0,
        max_drawdown=0.08,
        profit_factor=1.8,
        decay_half_life=30.0,
        decay_class="medium",
        max_factor_corr=0.15,
        incremental_r2=0.02,
        is_orthogonal=True,
        subsample_stable=True,
    )


@pytest.fixture
def sample_validation_report(sample_deploy_verdict) -> ValidationReport:
    """Validation report with one DEPLOY signal."""
    return ValidationReport(
        request_id="test-req-001",
        signal_bundle_id="test-req-001",
        verdicts=[sample_deploy_verdict],
        deploy_count=1,
        refine_count=0,
        reject_count=0,
        bonferroni_adjusted=False,
        overall_assessment="1 signals evaluated: 1 DEPLOY",
        recommended_composites=["SIG_EMA_CONFLUENCE_5m_v1"],
        correlation_matrix={},
        timestamp=datetime.now(UTC).isoformat(),
    )


# ─── Cost Model Tests ─────────────────────────────────────────


class TestCostModel:
    def test_round_turn_cost_nq(self, nq_spec):
        cost = compute_round_turn_cost(nq_spec)
        # (2.14 + 0.50 + 2.50) * 2 = 10.28
        assert abs(cost - 10.28) < 0.01

    def test_round_turn_cost_es(self, es_spec):
        cost = compute_round_turn_cost(es_spec)
        # (2.14 + 0.50 + 3.13) * 2 = 11.54
        assert abs(cost - 11.54) < 0.01

    def test_cost_analysis_basic(self, nq_spec):
        analysis = compute_cost_analysis(
            gross_pnl=5000.0,
            num_trades=100,
            instrument=nq_spec,
            gross_sharpe=2.0,
        )
        assert analysis.gross_pnl == 5000.0
        assert analysis.total_costs == pytest.approx(1028.0, abs=1)
        assert analysis.net_pnl == pytest.approx(3972.0, abs=1)
        assert 0 < analysis.cost_drag_pct < 1
        assert analysis.net_sharpe < analysis.gross_sharpe
        assert 0 < analysis.breakeven_hit_rate < 1

    def test_cost_analysis_zero_gross(self, nq_spec):
        analysis = compute_cost_analysis(
            gross_pnl=0.0,
            num_trades=10,
            instrument=nq_spec,
        )
        assert analysis.cost_drag_pct == float("inf")
        assert analysis.net_pnl < 0

    def test_turnover_metrics_basic(self):
        direction = pd.Series([1, 1, 1, 0, -1, -1, 0, 1, 1, 0])
        metrics = compute_turnover_metrics(direction, bars_per_day=10)
        assert metrics["trades_per_day"] > 0
        assert metrics["avg_holding_bars"] > 0
        assert 0 <= metrics["flip_rate"] <= 1

    def test_turnover_metrics_empty(self):
        direction = pd.Series([], dtype=float)
        metrics = compute_turnover_metrics(direction)
        assert metrics["trades_per_day"] == 0.0

    def test_turnover_no_flips(self):
        direction = pd.Series([1, 1, 1, 0, 0, 1, 1, 1])
        metrics = compute_turnover_metrics(direction, bars_per_day=8)
        assert metrics["flip_rate"] == 0.0

    def test_turnover_with_flips(self):
        direction = pd.Series([1, 1, -1, -1, 1, 1])
        metrics = compute_turnover_metrics(direction, bars_per_day=6)
        assert metrics["flip_rate"] > 0

    def test_estimate_trade_stats(self):
        np.random.seed(42)
        n = 500
        close = pd.Series(np.cumsum(np.random.randn(n) * 0.5) + 100)
        direction = pd.Series(np.random.choice([-1, 0, 1], n))
        strength = pd.Series(np.random.uniform(0.3, 1.0, n))
        strength[direction == 0] = 0.0

        stats = estimate_trade_stats(direction, strength, close, horizon=5)
        assert 0 < stats["win_rate"] < 1
        assert stats["avg_win"] > 0
        assert stats["avg_loss"] > 0
        assert stats["num_trades"] > 0


# ─── Position Sizing Tests ─────────────────────────────────────


class TestPositionSizing:
    def test_kelly_positive_edge(self):
        # 55% win rate, 1:1 R/R -> f* = (0.55*1 - 0.45)/1 = 0.10
        f = kelly_fraction(0.55, 100, 100)
        assert abs(f - 0.10) < 0.01

    def test_kelly_high_rr(self):
        # 40% win rate, 3:1 R/R -> f* = (0.40*3 - 0.60)/3 = 0.20
        f = kelly_fraction(0.40, 300, 100)
        assert abs(f - 0.20) < 0.01

    def test_kelly_no_edge(self):
        # 50% win rate, 1:1 -> f* = 0
        f = kelly_fraction(0.50, 100, 100)
        assert abs(f) < 0.01

    def test_kelly_negative_edge(self):
        f = kelly_fraction(0.30, 100, 100)
        assert f < 0

    def test_kelly_edge_cases(self):
        assert kelly_fraction(0.0, 100, 100) == 0.0
        assert kelly_fraction(1.0, 100, 100) == 0.0
        assert kelly_fraction(0.5, 0, 100) == 0.0
        assert kelly_fraction(0.5, 100, 0) == 0.0

    def test_half_kelly_contracts(self, nq_spec):
        f = 0.10
        contracts = half_kelly_contracts(f, 50000, nq_spec)
        # half_kelly = 0.05 * 50000 / 20 = 125 contracts
        assert contracts == 125

    def test_half_kelly_zero_edge(self, nq_spec):
        assert half_kelly_contracts(0.0, 50000, nq_spec) == 0
        assert half_kelly_contracts(-0.1, 50000, nq_spec) == 0

    def test_max_contracts_from_limit(self, nq_spec):
        # daily limit $1000, stop 8 ticks
        # loss_per_contract = 8 * 5.0 = 40
        # rt_cost = 10.28
        # total = 50.28
        # max = floor(1000 / 50.28) = 19
        contracts = max_contracts_from_daily_limit(1000, 8, nq_spec)
        assert contracts == 19

    def test_max_contracts_zero_limit(self, nq_spec):
        assert max_contracts_from_daily_limit(0, 8, nq_spec) == 0

    def test_max_contracts_zero_stop(self, nq_spec):
        assert max_contracts_from_daily_limit(1000, 0, nq_spec) == 0

    def test_recommended_contracts(self):
        rec = recommended_contracts(
            signal_strength=0.8,
            kelly_f=0.1,
            max_from_limit=10,
            max_from_firm=4,
        )
        # base = min(10, 4) = 4, scaled = 4 * 0.8 = 3.2 -> 3
        assert rec == 3

    def test_recommended_zero_strength(self):
        assert recommended_contracts(0.0, 0.1, 10, 4) == 0

    def test_recommended_negative_kelly(self):
        assert recommended_contracts(0.8, -0.1, 10, 4) == 0


# ─── Prop Constraints Tests ────────────────────────────────────


class TestPropConstraints:
    def test_passes_apex(self, apex_50k):
        # Moderate daily P&L, well within drawdown limits
        daily_pnl = [100, -50, 200, -30, 150, 80, -100, 200, 50, -20]
        result = validate_prop_firm_constraints(
            daily_pnl, apex_50k,
            kelly_f=0.05, half_kelly_contracts=2, mc_ruin_prob=0.02,
        )
        assert result.passes_trailing_dd
        assert result.passes_daily_limit  # Apex has no daily limit
        assert result.passes_mc_check
        assert result.worst_day_pnl == -100

    def test_fails_trailing_dd(self, apex_50k):
        # Large losses that exceed trailing DD
        daily_pnl = [100, -1000, -1000, -600, 50]
        result = validate_prop_firm_constraints(
            daily_pnl, apex_50k,
            kelly_f=0.05, half_kelly_contracts=2, mc_ruin_prob=0.02,
        )
        assert not result.passes_trailing_dd
        assert result.max_trailing_dd > apex_50k.trailing_max_drawdown

    def test_fails_daily_limit_topstep(self, topstep_50k):
        # Day with >$1000 loss
        daily_pnl = [100, -1500, 200, 50]
        result = validate_prop_firm_constraints(
            daily_pnl, topstep_50k,
            kelly_f=0.05, half_kelly_contracts=2, mc_ruin_prob=0.02,
        )
        assert not result.passes_daily_limit

    def test_passes_daily_limit_topstep(self, topstep_50k):
        daily_pnl = [100, -800, 200, 50]
        result = validate_prop_firm_constraints(
            daily_pnl, topstep_50k,
            kelly_f=0.05, half_kelly_contracts=2, mc_ruin_prob=0.02,
        )
        assert result.passes_daily_limit

    def test_consistency_score_good(self, apex_50k):
        # Even distribution of wins
        daily_pnl = [100, 120, 90, 110, 80, 130, 100, 95, 105, 115]
        result = validate_prop_firm_constraints(daily_pnl, apex_50k)
        assert result.consistency_score > 0.5

    def test_consistency_score_bad(self, apex_50k):
        # One huge day dominates
        daily_pnl = [5000, 10, 20, 15, 10, 5, 10, 20, 15, 10]
        result = validate_prop_firm_constraints(daily_pnl, apex_50k)
        assert result.consistency_score < 0.5

    def test_empty_pnl(self, apex_50k):
        result = validate_prop_firm_constraints([], apex_50k)
        assert result.worst_day_pnl == 0.0
        assert result.passes_daily_limit
        assert result.passes_trailing_dd

    def test_recommended_contracts_capped(self, apex_50k):
        result = validate_prop_firm_constraints(
            [100, 200], apex_50k,
            kelly_f=0.2, half_kelly_contracts=100,
        )
        # Should be capped at firm max (4)
        assert result.recommended_contracts <= apex_50k.max_contracts

    def test_mc_ruin_check(self, apex_50k):
        result_pass = validate_prop_firm_constraints(
            [100], apex_50k, mc_ruin_prob=0.03,
        )
        assert result_pass.passes_mc_check

        result_fail = validate_prop_firm_constraints(
            [100], apex_50k, mc_ruin_prob=0.10,
        )
        assert not result_fail.passes_mc_check


# ─── Monte Carlo Tests ─────────────────────────────────────────


class TestMonteCarlo:
    def test_strong_edge_low_ruin(self, apex_50k):
        # Strong edge: 60% WR, 2:1 R/R
        results = simulate_ruin_probability(
            win_rate=0.60,
            avg_win=200,
            avg_loss=100,
            profile=apex_50k,
            num_simulations=1000,
            trade_sequences=[100],
            rng_seed=42,
        )
        assert 100 in results
        assert results[100] < 0.10  # Low ruin probability

    def test_no_edge_high_ruin(self, apex_50k):
        # No edge: 50% WR, 1:1 R/R
        results = simulate_ruin_probability(
            win_rate=0.50,
            avg_win=100,
            avg_loss=100,
            profile=apex_50k,
            num_simulations=1000,
            trade_sequences=[500],
            rng_seed=42,
        )
        assert results[500] > 0.05  # Higher ruin

    def test_multiple_sequences(self, apex_50k):
        results = simulate_ruin_probability(
            win_rate=0.55,
            avg_win=150,
            avg_loss=100,
            profile=apex_50k,
            num_simulations=500,
            trade_sequences=[100, 500, 1000],
            rng_seed=42,
        )
        assert len(results) == 3
        assert all(0 <= v <= 1 for v in results.values())

    def test_invalid_inputs(self, apex_50k):
        results = simulate_ruin_probability(0.0, 100, 100, apex_50k)
        assert all(v == 1.0 for v in results.values())

        results = simulate_ruin_probability(0.5, 0, 100, apex_50k)
        assert all(v == 1.0 for v in results.values())

    def test_reproducible_with_seed(self, apex_50k):
        r1 = simulate_ruin_probability(
            0.55, 150, 100, apex_50k,
            num_simulations=100, trade_sequences=[100], rng_seed=123,
        )
        r2 = simulate_ruin_probability(
            0.55, 150, 100, apex_50k,
            num_simulations=100, trade_sequences=[100], rng_seed=123,
        )
        assert r1 == r2

    def test_expected_value(self):
        ev = compute_expected_value(0.55, 100, 100)
        assert abs(ev - 10.0) < 0.01  # 0.55*100 - 0.45*100 = 10

        ev = compute_expected_value(0.50, 100, 100)
        assert abs(ev) < 0.01


# ─── Agent Wiring Tests ───────────────────────────────────────


class TestExecutionAgent:
    def test_create(self, message_bus):
        agent = ExecutionAgent(message_bus)
        assert agent.agent_id == AgentID.EXECUTION
        assert agent.name == "Execution & Risk"

    def test_create_with_config(self, message_bus, nq_spec, apex_50k):
        agent = ExecutionAgent(
            message_bus, instrument=nq_spec, prop_firm=apex_50k
        )
        assert agent._instrument == nq_spec
        assert agent._prop_firm == apex_50k

    def test_handle_execution_request(
        self, message_bus, nq_spec, apex_50k, sample_validation_report
    ):
        agent = ExecutionAgent(
            message_bus, instrument=nq_spec, prop_firm=apex_50k
        )
        received = []
        message_bus.register_agent(
            AgentID.EXECUTION, agent.handle_message
        )
        message_bus.register_agent(
            AgentID.ORCHESTRATOR, lambda env: received.append(env)
        )

        envelope = MessageEnvelope(
            request_id="test-exec-001",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.EXECUTION,
            message_type=MessageType.EXECUTION_REQUEST,
            priority=Priority.NORMAL,
            payload={
                "validation_report": sample_validation_report.model_dump(),
            },
        )
        agent.handle_message(envelope)

        # Should have sent ACK + EXECUTION_REPORT
        reports = [
            e for e in received
            if e.message_type == MessageType.EXECUTION_REPORT
        ]
        assert len(reports) == 1
        report_data = reports[0].payload["report"]
        report = ExecutionReport.model_validate(report_data)
        assert report.request_id == "test-req-001"

    def test_handle_unexpected_message(self, message_bus):
        agent = ExecutionAgent(message_bus)
        received = []
        message_bus.register_agent(
            AgentID.EXECUTION, agent.handle_message
        )
        message_bus.register_agent(
            AgentID.ORCHESTRATOR, lambda env: received.append(env)
        )

        envelope = MessageEnvelope(
            request_id="test-002",
            sender=AgentID.ORCHESTRATOR,
            receiver=AgentID.EXECUTION,
            message_type=MessageType.DATA_BUNDLE,
            priority=Priority.NORMAL,
            payload={},
        )
        agent.handle_message(envelope)

        nacks = [e for e in received if e.message_type == MessageType.NACK]
        assert len(nacks) == 1

    def test_analyze_signals_no_config(
        self, message_bus, sample_validation_report
    ):
        """Agent works with default fallbacks when no instrument/firm."""
        agent = ExecutionAgent(message_bus)
        report = agent.analyze_signals(sample_validation_report)
        assert isinstance(report, ExecutionReport)
        total = len(report.approved_signals) + len(report.vetoed_signals)
        assert total == 1  # One DEPLOY signal evaluated

    def test_analyze_signals_with_config(
        self, message_bus, nq_spec, apex_50k, sample_validation_report
    ):
        agent = ExecutionAgent(
            message_bus, instrument=nq_spec, prop_firm=apex_50k
        )
        report = agent.analyze_signals(sample_validation_report)
        assert isinstance(report, ExecutionReport)

        # Check that verdicts have complete data
        all_verdicts = report.approved_signals + report.vetoed_signals
        assert len(all_verdicts) == 1
        v = all_verdicts[0]
        assert v.signal_id == "SIG_EMA_CONFLUENCE_5m_v1"
        assert v.verdict in ("APPROVED", "VETOED")
        assert v.costs.total_costs >= 0
        assert v.prop_firm.kelly_fraction >= 0

    def test_skips_non_deploy(self, message_bus, nq_spec, apex_50k):
        """Only DEPLOY verdicts are evaluated."""
        report = ValidationReport(
            request_id="test-003",
            signal_bundle_id="test-003",
            verdicts=[
                SignalVerdict(
                    signal_id="SIG_REJECT",
                    verdict="REJECT",
                    ic=0.01,
                    ic_tstat=0.5,
                    hit_rate=0.48,
                    hit_rate_long=0.47,
                    hit_rate_short=0.49,
                    sharpe=0.3,
                    sortino=0.4,
                    max_drawdown=0.25,
                    profit_factor=0.8,
                    decay_half_life=5.0,
                    decay_class="ultra-fast",
                    max_factor_corr=0.5,
                    incremental_r2=0.001,
                    is_orthogonal=False,
                    subsample_stable=False,
                ),
            ],
            deploy_count=0,
            refine_count=0,
            reject_count=1,
            bonferroni_adjusted=False,
            overall_assessment="1 signals evaluated: 1 REJECT",
            timestamp=datetime.now(UTC).isoformat(),
        )
        agent = ExecutionAgent(
            message_bus, instrument=nq_spec, prop_firm=apex_50k
        )
        exec_report = agent.analyze_signals(report)
        assert len(exec_report.approved_signals) == 0
        assert len(exec_report.vetoed_signals) == 0
