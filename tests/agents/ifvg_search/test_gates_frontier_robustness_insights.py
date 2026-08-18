"""Strategy gates, frontier, robustness, and deterministic insights (CS §3.4/§12)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    ResolvedStrategyGateThresholds,
)
from alpha_lab.agents.data_infra.ifvg.search.frontier import (
    ObjectiveSpec,
    build_frontier,
)
from alpha_lab.agents.data_infra.ifvg.search.gates import evaluate_strategy_gates
from alpha_lab.agents.data_infra.ifvg.search.insights import (
    FORBIDDEN_INSIGHT_WORDING,
    InsightCategory,
    render_insight_panel,
)
from alpha_lab.agents.data_infra.ifvg.search.robustness import evaluate_robustness
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import (
    compute_strategy_metrics,
)
from alpha_lab.agents.data_infra.ifvg.study.population_delta import (
    PopulationDeltaReport,
)
from tests.agents.ifvg_search.conftest import (
    SYNTHETIC_DAYS,
    make_resolved_trades_frame,
)

_THRESHOLDS = ResolvedStrategyGateThresholds(
    min_executed_trades=10,
    min_independent_days=3,
    min_session_stability_score=0.5,
)


def _metrics(**kwargs):
    tables = {RecordTable.EXECUTED_TRADE: make_resolved_trades_frame(SYNTHETIC_DAYS)}
    return compute_strategy_metrics(
        tables, cost_points=0.514, evaluation_config_hash="1" * 64, **kwargs
    )


def test_metrics_and_gates_pass_with_full_explanations() -> None:
    metrics = _metrics()
    assert metrics.executed_trades == 12
    assert metrics.independent_days == 3
    assert metrics.net_expectancy_r == pytest.approx(0.4486, abs=1e-3)
    assert metrics.profit_factor is not None and metrics.profit_factor > 1.1
    assert metrics.session_stability_score == 1.0
    report = evaluate_strategy_gates(metrics, _THRESHOLDS)
    assert report.passed is True
    assert report.failure_reason is None
    # every check carries a human explanation; the report covers all 11 gates
    assert len(report.checks) == 11
    assert all(check.explanation for check in report.checks)
    gate_ids = {check.gate_id for check in report.checks}
    assert "require_bootstrap_ci_excludes_zero" in gate_ids


def test_gate_failure_yields_typed_reason_and_first_failure_text() -> None:
    metrics = _metrics()
    strict = _THRESHOLDS.model_copy(update={"min_executed_trades": 100})
    report = evaluate_strategy_gates(metrics, strict)
    assert report.passed is False
    assert report.failure_reason is not None
    assert report.human_explanation.startswith("failed 1 of 11")
    assert "min_executed_trades" in report.human_explanation


def test_bootstrap_ci_gate_enforced_when_required() -> None:
    metrics = _metrics()
    required = _THRESHOLDS.model_copy(
        update={"require_bootstrap_ci_excludes_zero": True}
    )
    report = evaluate_strategy_gates(metrics, required)
    check = next(
        c for c in report.checks if c.gate_id == "require_bootstrap_ci_excludes_zero"
    )
    assert metrics.net_expectancy_bootstrap_ci95 is not None
    low, high = metrics.net_expectancy_bootstrap_ci95
    assert check.passed is (low > 0.0 or high < 0.0)
    assert "CI" in check.explanation


def test_frontier_dominance_ties_and_trace() -> None:
    feasible = {
        "a" * 64: {"net_expectancy_r": 0.5, "max_drawdown_r": 3.0},
        "b" * 64: {"net_expectancy_r": 0.4, "max_drawdown_r": 4.0},  # dominated by a
        "c" * 64: {"net_expectancy_r": 0.6, "max_drawdown_r": 5.0},  # trade-off vs a
    }
    objectives = (
        ObjectiveSpec(metric="net_expectancy_r", direction="maximize"),
        ObjectiveSpec(metric="max_drawdown_r", direction="minimize"),
    )
    result = build_frontier(
        feasible,
        objectives=objectives,
        lexicographic_tie_breaks=("net_expectancy_r", "core_replay_id"),
    )
    assert set(result.frontier_ids) == {"a" * 64, "c" * 64}
    assert ("a" * 64, "b" * 64) in result.dominance_edges
    assert result.per_objective_champions["net_expectancy_r"] == "c" * 64
    assert result.per_objective_champions["max_drawdown_r"] == "a" * 64
    # the tie-break trace is persisted and names the representative honestly
    assert result.development_exploratory_representative_id == "c" * 64
    assert any("Development Exploratory Representative" in line for line in result.tie_break_trace)
    assert any("development lane only" in line for line in result.tie_break_trace)
    # a missing objective refuses instead of guessing
    with pytest.raises(ValueError, match="lacks objective metrics"):
        build_frontier(
            {"a" * 64: {"net_expectancy_r": 0.5}},
            objectives=objectives,
            lexicographic_tie_breaks=(),
        )


def test_max_gates_fail_closed_when_unmeasurable_with_trades() -> None:
    """F9: a concentration cap that could not be measured while trades exist
    fails closed; with zero trades the cap is vacuous (min gates fail first)."""

    metrics = _metrics()
    unmeasured = metrics.model_copy(update={"top_setup_pnl_share": None})
    report = evaluate_strategy_gates(unmeasured, _THRESHOLDS)
    check = next(c for c in report.checks if c.gate_id == "max_top_setup_pnl_share")
    assert check.passed is False
    assert "unmeasurable" in check.explanation
    zero_trades = unmeasured.model_copy(
        update={"executed_trades": 0, "top_day_pnl_share": None}
    )
    report = evaluate_strategy_gates(zero_trades, _THRESHOLDS)
    day_check = next(
        c for c in report.checks if c.gate_id == "max_top_day_pnl_share"
    )
    assert day_check.passed is True  # cap over nothing holds
    assert report.passed is False  # the min gates fail closed instead


def test_robustness_neighbors_plateau_and_knife_edge() -> None:
    grid = ("v.none", "v.240", "v.360", "v.480")
    positions = {
        f"{index}" * 64: {"axis": value} for index, value in enumerate(grid)
    }
    metric = {
        "0" * 64: 0.50,
        "1" * 64: 0.48,
        "2" * 64: 0.10,  # cliff
        "3" * 64: 0.05,
    }
    report = evaluate_robustness(
        "1" * 64,
        axis_grids={"axis": grid},
        child_positions=positions,
        metric_by_child=metric,
        degradation_warning_r=0.2,
    )
    assert len(report.neighbor_checks) == 2
    assert report.worst_neighbor_degradation_r == pytest.approx(0.38)
    assert report.knife_edge is True
    assert any("degrades" in warning for warning in report.knife_edge_warnings)
    # the plateau ends where the metric falls out of the warning band
    assert report.per_axis_plateau["axis"] == 2
    assert report.outer_fold_recurrence is None  # schema-reserved (decision 15)


def test_insights_render_seven_categories_and_refuse_forbidden_wording() -> None:
    metrics = _metrics()
    delta = PopulationDeltaReport(
        entity_kind="setup",
        match_basis="not_comparable",
        match_basis_reason="collisions persisted in the LineageUniquenessReport",
        common_keys=(),
        added_keys=(),
        removed_keys=(),
        jaccard=None,
        first_divergence=None,
    )
    panel = render_insight_panel(
        changed_axis_labels=("parent retest timeout",),
        child_id="a" * 64,
        metrics=metrics,
        baseline_metrics=metrics,
        population_deltas=(delta,),
    )
    rendered = {insight.category for insight in panel.insights}
    assert rendered == set(InsightCategory)
    suppressed = [insight for insight in panel.insights if insight.suppressed]
    assert len(suppressed) == 1
    assert suppressed[0].match_basis == "not_comparable"
    assert "disabled" in suppressed[0].text
    # forbidden publishable/causal wording is structurally refused
    for token in ("publishable", "causes"):
        assert token in FORBIDDEN_INSIGHT_WORDING
    with pytest.raises(ValueError, match="forbidden insight wording"):
        render_insight_panel(
            changed_axis_labels=("the best axis",),
            child_id="a" * 64,
            metrics=metrics,
            baseline_metrics=None,
        )
