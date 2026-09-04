"""UI-3 §6.2 — the metric metadata registry (pure): one spec per displayed
technical key, directionality from ``OBJECTIVE_DIRECTIONS`` where registered,
references from existing code only, deterministic status rules, and no
invented threshold anywhere."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import (
    GATE_FIELD_METRICS,
    METRIC_SPECS,
    ReferenceKind,
    describe,
    evaluate_gate_flag,
    evaluate_interval,
    evaluate_metric,
    format_value,
    gate_metric_key,
    metric_keys,
    metric_keys_for_surface,
)
from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import UiStatus
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    OBJECTIVE_DIRECTIONS,
    ResolvedPropGateThresholds,
    ResolvedRobustnessGateThresholds,
    ResolvedStrategyGateThresholds,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    FIRM_MATRIX_METRICS,
    HEATMAP_METRICS,
)

_DIRECTION_OF = {"maximize": "higher_better", "minimize": "lower_better"}


def test_every_spec_is_complete_and_directions_follow_the_charter_registry() -> None:
    assert len(METRIC_SPECS) >= 80
    for key, spec in METRIC_SPECS.items():
        assert spec.technical_key == key
        assert spec.human_name.strip() and len(spec.definition) > 20, key
        assert spec.formula_or_source.strip(), key
        assert spec.unit, key
        assert spec.directionality in (
            "higher_better",
            "lower_better",
            "target",
            "descriptive",
        ), key
        assert spec.surfaces, key
        if spec.reference is not None:
            assert spec.reference.source.strip(), key
            assert spec.reference.kind in ReferenceKind
        assert describe(key) is spec
    # directionality comes from OBJECTIVE_DIRECTIONS where the metric is registered
    for key, direction in OBJECTIVE_DIRECTIONS.items():
        assert key in METRIC_SPECS, key  # every charter objective is describable
        assert METRIC_SPECS[key].directionality == _DIRECTION_OF[direction], key
    assert tuple(metric_keys()) == tuple(sorted(METRIC_SPECS))
    with pytest.raises(ValueError, match="unregistered metric"):
        describe("no_such_metric")


def test_registry_covers_every_displayed_key() -> None:
    """Plan §10 'registry coverage of displayed metric keys' — the Results
    pickers, the explorer columns, the wizard gate rows, the ladder columns,
    the Context Research metrics and the Data & Audit report fields."""

    for attribute in (*HEATMAP_METRICS.values(), *FIRM_MATRIX_METRICS.values()):
        assert attribute in METRIC_SPECS, attribute
    for attribute in (
        "executed_trades",
        "net_expectancy_r",
        "realized_payoff_ratio",
        "profit_factor",
        "max_drawdown_r",
        "trade_frequency",
        "setup_occupancy",
        "first_payout_probability_60d",
        "three_payout_probability",
        "expected_net_payout_90d",
        "p10_net_payout_90d",
        "expected_replacement_cost",
        "survival_probability_90d",
        "total_fees",
    ):
        assert attribute in METRIC_SPECS, attribute
    for field in (
        *ResolvedStrategyGateThresholds.model_fields,
        *ResolvedPropGateThresholds.model_fields,
        *ResolvedRobustnessGateThresholds.model_fields,
    ):
        assert field in GATE_FIELD_METRICS, field
        assert gate_metric_key(field) in METRIC_SPECS, field
    for key in (
        "brier_score",
        "brier_skill_score",
        "log_loss",
        "auc",
        "prevalence",
        "mean_probability",
        "reference_brier_score",
        "calibration_slope",
        "calibration_intercept",
        "oos_prediction_count",
        "threshold_coverage_fraction",
        "threshold_net_r_sum",
        "setup_cluster_net_r_mean",
        "trading_day_block_net_r_mean",
        "valid_fold_count",
        "bootstrap_cluster_count",
        "win_rate",
        "total_realized_r",
        "max_drawdown_dollars",
        "feature_coverage_fraction",
        "replay_slowdown_fraction",
        "repeated_run_p95_slowdown_fraction",
        "completed_1m_step_p99_ms",
        "multi_timeframe_callback_p99_ms",
        "terminal_seed_bytes",
        "max_transition_bytes",
        "protected_file_opens",
        "denied_attempt_count",
        "validity_report",
        "reconciliation_report",
        "capacity_report",
        "performance_report",
        "identity_report",
        "oos_row_count",
        "oos_assignment_coverage",
        "bootstrap_aligned_ami_mean",
        "regime_occupancy",
        "mbp1_day_coverage_fraction",
        "paired_brier_delta",
        "realized_r",
        "mfe_ticks",
        "mae_ticks",
    ):
        assert key in METRIC_SPECS, key
    assert set(metric_keys_for_surface("ladder")) >= {"brier_score", "auc", "oos_row_count"}
    with pytest.raises(ValueError, match="gate field"):
        gate_metric_key("min_something_else")


@pytest.mark.parametrize(
    ("key", "value", "kwargs", "expected"),
    [
        # the selected resolved gate (higher_better / lower_better)
        ("executed_trades", 42, {"gate": 30}, UiStatus.PASS),
        ("executed_trades", 12, {"gate": 30}, UiStatus.FAIL),
        ("max_drawdown_r", 9.0, {"gate": 15.0}, UiStatus.PASS),
        ("max_drawdown_r", 20.0, {"gate": 15.0}, UiStatus.FAIL),
        ("net_expectancy_r", 0.2, {}, UiStatus.INFORMATIONAL),  # no gate selected
        # Brier against the prevalence reference (boundary, strict)
        ("brier_score", 0.20, {"reference": 0.25}, UiStatus.PASS),
        ("brier_score", 0.30, {"reference": 0.25}, UiStatus.FAIL),
        ("brier_score", 0.25, {"reference": 0.25}, UiStatus.INCONCLUSIVE),
        ("brier_score", 0.20, {}, UiStatus.UNAVAILABLE),  # no reference persisted
        # Brier skill against the 0 boundary
        ("brier_skill_score", 0.08, {}, UiStatus.PASS),
        ("brier_skill_score", -0.02, {}, UiStatus.FAIL),
        ("brier_skill_score", 0.0, {}, UiStatus.INCONCLUSIVE),
        # AUC: the 0.5 chance line is a DIRECTION only — never a band
        ("auc", 0.67, {}, UiStatus.INFORMATIONAL),
        ("auc", 0.41, {}, UiStatus.INFORMATIONAL),
        # calibration targets: distance only
        ("calibration_slope", 1.3, {}, UiStatus.INFORMATIONAL),
        ("calibration_intercept", -0.2, {}, UiStatus.INFORMATIONAL),
        # sample adequacy against a REGISTERED minimum
        ("bootstrap_cluster_count", 1, {}, UiStatus.INCONCLUSIVE),
        ("bootstrap_cluster_count", 5, {}, UiStatus.PASS),
        ("valid_fold_count", 0, {}, UiStatus.INCONCLUSIVE),
        ("fold_train_candidate_count", 12, {}, UiStatus.INCONCLUSIVE),
        ("fold_train_candidate_count", 30, {}, UiStatus.PASS),
        # capacity / performance against report limits
        ("terminal_seed_bytes", 1000, {"reference": 838_860}, UiStatus.PASS),
        ("terminal_seed_bytes", 900_000, {"reference": 838_860}, UiStatus.FAIL),
        ("replay_slowdown_fraction", 0.05, {}, UiStatus.PASS),
        ("replay_slowdown_fraction", 0.35, {}, UiStatus.FAIL),
        # access counters: measured zeros PASS, nonzero FAIL; policy zeros informational
        ("denied_attempt_count", 0, {}, UiStatus.PASS),
        ("denied_attempt_count", 3, {}, UiStatus.FAIL),
        ("protected_file_opens", 0, {}, UiStatus.INFORMATIONAL),
        # descriptive values never pass or fail
        ("prevalence", 0.4, {}, UiStatus.INFORMATIONAL),
        ("candidate_count", 12, {}, UiStatus.INFORMATIONAL),
    ],
)
def test_metric_status_rules(key, value, kwargs, expected) -> None:
    reading = evaluate_metric(key, value, **kwargs)
    assert reading.status is expected, reading.interpretation
    assert reading.chip.startswith(
        {
            UiStatus.PASS: "✓",
            UiStatus.FAIL: "✕",
            UiStatus.INCONCLUSIVE: "?",
            UiStatus.INFORMATIONAL: "ℹ",
            UiStatus.UNAVAILABLE: "∅",
        }[expected]
    )
    assert reading.human_name == describe(key).human_name
    assert reading.interpretation.strip()
    if expected is UiStatus.FAIL:
        assert reading.blocking


def test_missing_or_unevaluated_evidence_is_never_pass() -> None:
    assert evaluate_metric("executed_trades", None, gate=30).status is UiStatus.UNAVAILABLE
    assert evaluate_metric("executed_trades", 99, gate=30, evaluated=False).status is (
        UiStatus.UNAVAILABLE
    )
    assert evaluate_metric("denied_attempt_count", None).status is UiStatus.UNAVAILABLE
    assert evaluate_metric("terminal_seed_bytes", 10).status is UiStatus.UNAVAILABLE  # no limit
    assert evaluate_metric("auc", None).status is UiStatus.UNAVAILABLE
    assert evaluate_metric("auc", True).status is UiStatus.UNAVAILABLE  # a flag is not a value
    assert evaluate_gate_flag("validity_report", True).status is UiStatus.PASS
    assert evaluate_gate_flag("validity_report", False).status is UiStatus.FAIL
    assert evaluate_gate_flag("validity_report", None).status is UiStatus.UNAVAILABLE
    assert evaluate_gate_flag("validity_report", True, evaluated=False).status is (
        UiStatus.UNAVAILABLE
    )


def test_interval_crossing_zero_is_inconclusive() -> None:
    crossing = evaluate_interval("setup_cluster_net_r_mean", lower=-0.1, upper=0.3)
    assert crossing.status is UiStatus.INCONCLUSIVE
    assert "crosses zero" in crossing.interpretation
    positive = evaluate_interval("setup_cluster_net_r_mean", lower=0.05, upper=0.3)
    assert positive.status is UiStatus.PASS
    negative = evaluate_interval("setup_cluster_net_r_mean", lower=-0.4, upper=-0.1)
    assert negative.status is UiStatus.FAIL
    # lower_better intervals favour the negative side (paired Brier delta: negative favours MBP-1)
    favourable = evaluate_interval("paired_brier_delta", lower=-0.02, upper=-0.005)
    assert favourable.status is UiStatus.PASS
    unavailable = evaluate_interval(
        "trading_day_block_net_r_mean",
        lower=None,
        upper=None,
        available=False,
        reason="fewer_than_two_usable_clusters",
    )
    assert unavailable.status is UiStatus.UNAVAILABLE
    assert "fewer_than_two_usable_clusters" in unavailable.interpretation


def test_proposed_thresholds_and_references_are_carried_as_caveats() -> None:
    reading = evaluate_metric("profit_factor", 1.4, gate=1.1, proposed=True)
    assert reading.status is UiStatus.PASS
    assert reading.proposed is True
    assert reading.caveat and "proposed" in reading.caveat.lower()
    assert reading.reference_value == pytest.approx(1.1)
    sampled = evaluate_metric("net_expectancy_r", 0.3, gate=0.0, sample=12)
    assert sampled.sample == 12
    assert "12" in sampled.interpretation


def test_display_formatting_follows_the_unit() -> None:
    assert format_value(describe("breach_probability_90d"), 0.2) == "20.0%"
    assert format_value(describe("net_expectancy_r"), 0.4213) == "0.42 R"
    assert format_value(describe("expected_net_payout_90d"), 1234.6) == "$1,235"
    assert format_value(describe("executed_trades"), 1234) == "1,234"
    assert format_value(describe("terminal_seed_bytes"), 4096) == "4,096 B"
    assert format_value(describe("completed_1m_step_p99_ms"), 1.234) == "1.23 ms"
    assert format_value(describe("executed_trades"), None) == "—"
    assert format_value(describe("time_under_water_days"), 12) == "12 days"
