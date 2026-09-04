"""UI-3 Phase 3 — the Results readings (pure): the selected configuration's
strategy metrics against the charter's resolved gates, the prop vector
(worst across the simulated firms, D-#18) against the prop gates, and the
explorer column guide from the registry."""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.presentation.results_presentation import (
    column_guide,
    prop_readings,
    strategy_readings,
    worst_vector_value,
)
from alpha_lab.agents.data_infra.ifvg.presentation.rollups import (
    RollupSection,
    rollup_section,
)
from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import UiStatus
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    ResolvedPropGateThresholds,
    ResolvedStrategyGateThresholds,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import StrategyMetrics


def _metrics(**overrides) -> StrategyMetrics:
    values = {
        "executed_trades": 40,
        "independent_days": 25,
        "gross_expectancy_r": 0.30,
        "net_expectancy_r": 0.20,
        "profit_factor": 1.4,
        "max_drawdown_r": 6.0,
        "time_under_water_days": 12,
        "time_block_sign_consistency": 0.7,
        "session_stability_score": 0.6,
        "top_day_pnl_share": 0.2,
        "top_setup_pnl_share": 0.1,
        "net_expectancy_bootstrap_ci95": (0.05, 0.35),
        "planned_vs_realized": None,
        "trade_stats": {},
    }
    values.update(overrides)
    return StrategyMetrics(**values)


_GATES = ResolvedStrategyGateThresholds(min_session_stability_score=0.5)


def test_strategy_readings_follow_the_charter_gates_and_flag_proposed_thresholds() -> None:
    readings = {r.technical_key: r for r in strategy_readings(_metrics(), _GATES)}
    assert readings["executed_trades"].status is UiStatus.PASS
    assert readings["executed_trades"].reference_value == 30
    assert readings["max_drawdown_r"].status is UiStatus.PASS  # 6 ≤ 15
    assert readings["profit_factor"].proposed is True
    assert "net_expectancy_bootstrap_ci95" not in readings  # the CI gate is not required
    failing = {
        r.technical_key: r for r in strategy_readings(_metrics(profit_factor=0.9), _GATES)
    }
    assert failing["profit_factor"].status is UiStatus.FAIL and failing["profit_factor"].blocking
    rollup = rollup_section(RollupSection.STRATEGY_QUALITY, list(failing.values()))
    assert rollup.status is UiStatus.FAIL
    cautious = rollup_section(RollupSection.STRATEGY_QUALITY, list(readings.values()))
    assert cautious.status is UiStatus.WARNING  # every gate passes under proposed defaults
    ratified = strategy_readings(_metrics(), _GATES, proposed=False)
    assert rollup_section(RollupSection.STRATEGY_QUALITY, ratified).status is UiStatus.PASS
    required_ci = ResolvedStrategyGateThresholds(
        min_session_stability_score=0.5, require_bootstrap_ci_excludes_zero=True
    )
    with_ci = {r.technical_key: r for r in strategy_readings(_metrics(), required_ci)}
    assert with_ci["net_expectancy_bootstrap_ci95"].status is UiStatus.PASS
    crossing = {
        r.technical_key: r
        for r in strategy_readings(
            _metrics(net_expectancy_bootstrap_ci95=(-0.1, 0.3)), required_ci
        )
    }
    assert crossing["net_expectancy_bootstrap_ci95"].status is UiStatus.INCONCLUSIVE
    absent = strategy_readings(None, _GATES)
    assert {r.status for r in absent} == {UiStatus.UNAVAILABLE}


def test_prop_readings_use_the_worst_firm_and_the_prop_gates() -> None:
    summaries = {
        "firm_a": {
            "payout_reliability_vector": {
                "first_payout_probability_60d": 0.7,
                "three_payout_probability": 0.4,
                "expected_net_payout_90d": 1200.0,
                "p10_net_payout_90d": 100.0,
                "p90_payout_drought_days": 20.0,
                "breach_probability_90d": 0.2,
                "payout_probability_per_rolling_30d": 0.6,
                "expected_replacement_cost": 50.0,
                "median_account_lifetime_days": 80.0,
            }
        },
        "firm_b": {
            "payout_reliability_vector": {
                "first_payout_probability_60d": 0.4,
                "three_payout_probability": 0.3,
                "expected_net_payout_90d": 900.0,
                "p10_net_payout_90d": -50.0,
                "p90_payout_drought_days": 35.0,
                "breach_probability_90d": 0.45,
                "payout_probability_per_rolling_30d": 0.5,
                "expected_replacement_cost": 80.0,
                "median_account_lifetime_days": 60.0,
            }
        },
    }
    assert worst_vector_value(summaries, "breach_probability_90d") == 0.45  # minimise → max
    assert worst_vector_value(summaries, "expected_net_payout_90d") == 900.0  # maximise → min
    assert worst_vector_value({}, "breach_probability_90d") is None
    gates = ResolvedPropGateThresholds(
        minimum_first_payout_probability_60d=0.5,
        maximum_breach_probability_90d=0.35,
        minimum_expected_net_payout_90d=None,
        minimum_p10_net_payout_90d=None,
        maximum_p90_payout_drought_days=None,
        minimum_three_payout_probability=None,
    )
    readings = {r.technical_key: r for r in prop_readings(summaries, gates)}
    assert readings["first_payout_probability_60d"].status is UiStatus.FAIL  # worst firm 0.4
    assert readings["breach_probability_90d"].status is UiStatus.FAIL  # worst firm 0.45
    assert readings["expected_net_payout_90d"].status is UiStatus.INFORMATIONAL  # no gate
    assert readings["payout_probability_per_rolling_30d"].status is UiStatus.INFORMATIONAL
    assert readings["survival_probability_90d"].display == "55.0%"
    rollup = rollup_section(RollupSection.PROP_FEASIBILITY, list(readings.values()))
    assert rollup.status is UiStatus.FAIL
    empty = prop_readings({}, gates)
    assert {r.status for r in empty} == {UiStatus.UNAVAILABLE}
    assert rollup_section(RollupSection.PROP_FEASIBILITY, empty).status is UiStatus.INCONCLUSIVE


def test_column_guide_describes_every_explorer_column() -> None:
    guide = column_guide(
        {
            "net E[R]": "net_expectancy_r",
            "max DD R": "max_drawdown_r",
            "outer folds passed": None,
        }
    )
    assert [row["column"] for row in guide] == ["net E[R]", "max DD R", "outer folds passed"]
    assert guide[0]["metric"] == "Net expectancy" and "higher is better" in guide[0]["direction"]
    assert guide[1]["direction"] == "lower is better"
    assert "not persisted" in guide[2]["definition"]
