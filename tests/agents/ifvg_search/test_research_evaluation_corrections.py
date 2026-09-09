"""Regressions found in the completed parent-staleness development study."""

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.charter import CostPolicy
from alpha_lab.agents.data_infra.ifvg.search.frontier import ObjectiveSpec, build_frontier
from alpha_lab.agents.data_infra.ifvg.search.identities import CostedEvaluationIdentity
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
    CostedEvaluationEnvelope,
    _child_evaluation_envelope,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import (
    compute_strategy_metrics,
    per_trade_net_r,
    research_trades,
)
from tests.agents.ifvg_search.conftest import SYNTHETIC_DAYS, make_resolved_trades_frame


def test_warmup_is_excluded_from_every_metric_and_trade_detail():
    trades = make_resolved_trades_frame(SYNTHETIC_DAYS)
    trades["is_warmup"] = trades.trading_day.astype(str) == SYNTHETIC_DAYS[0]
    args = dict(cost_points=0.514, evaluation_config_hash="1" * 64)
    actual = compute_strategy_metrics({RecordTable.EXECUTED_TRADE: trades}, **args)
    expected = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: trades.loc[~trades.is_warmup].drop(columns="is_warmup")},
        **args,
    )
    assert actual == expected
    assert actual.independent_days == 2
    assert actual.executed_trades == 8
    assert len(per_trade_net_r(trades, cost_points=0.514)) == 8
    assert len(trades) == 12  # raw history retained


def test_drawdown_counts_initial_losses_and_matches_detailed_report():
    trades = make_resolved_trades_frame(SYNTHETIC_DAYS).iloc[:3].copy()
    trades["resolution"] = "stop"
    trades["realized_ticks"] = -trades.risk_ticks
    metrics = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: trades}, cost_points=0, evaluation_config_hash="2" * 64
    )
    assert metrics.max_drawdown_r == 3
    assert (
        metrics.max_drawdown_r
        == metrics.trade_stats["equity"]["r"]["drawdown"]["max_drawdown_close"]
    )
    assert metrics.time_under_water_days == 1


@pytest.mark.parametrize("bad", [None, "false", "True", 2])
def test_ambiguous_warmup_flags_fail_closed(bad):
    with pytest.raises(ValueError, match="warmup"):
        research_trades(pd.DataFrame({"is_warmup": [False, bad]}))


def test_metric_correction_uses_new_identity_and_preserves_legacy_loads():
    current = _child_evaluation_envelope("a" * 64, CostPolicy())
    legacy = CostedEvaluationEnvelope.from_payload(
        CostedEvaluationIdentity(
            core_replay_id="a" * 64, cost_policy_sha256=current.payload.cost_policy_sha256
        )
    )
    assert current.costed_evaluation_id != legacy.costed_evaluation_id
    assert CostedEvaluationEnvelope.model_validate_json(legacy.model_dump_json()) == legacy
    assert CostedEvaluationEnvelope.model_validate_json(current.model_dump_json()) == current


def test_drawdown_tie_break_prefers_less_risk():
    result = build_frontier(
        {
            "a" * 64: {"net_expectancy_r": 0.2, "max_drawdown_r": 8},
            "b" * 64: {"net_expectancy_r": 0.2, "max_drawdown_r": 2},
        },
        objectives=(ObjectiveSpec(metric="net_expectancy_r", direction="maximize"),),
        lexicographic_tie_breaks=("max_drawdown_r", "core_replay_id"),
    )
    assert result.development_exploratory_representative_id == "b" * 64
