"""UI-3 §6.3 — deterministic section roll-ups (pure)."""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import (
    evaluate_gate_flag,
    evaluate_interval,
    evaluate_metric,
)
from alpha_lab.agents.data_infra.ifvg.presentation.rollups import (
    SECTION_LABELS,
    RollupSection,
    rollup_section,
)
from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import UiStatus


def test_nine_sections_are_labelled() -> None:
    assert [section.value for section in RollupSection] == [
        "data_integrity",
        "strategy_quality",
        "probability_skill",
        "calibration",
        "stability",
        "prop_feasibility",
        "robustness",
        "authorization_readiness",
        "capacity_and_performance",
    ]
    assert set(SECTION_LABELS) == set(RollupSection)


def test_rollup_rule_order_is_fail_blocked_inconclusive_warning_pass_informational() -> None:
    passing = evaluate_metric("executed_trades", 40, gate=30)
    failing = evaluate_metric("profit_factor", 0.9, gate=1.1)
    inconclusive = evaluate_interval("setup_cluster_net_r_mean", lower=-0.1, upper=0.2)
    proposed = evaluate_metric("max_drawdown_r", 5.0, gate=15.0, proposed=True)
    informational = evaluate_metric("prevalence", 0.4)
    missing = evaluate_metric("brier_score", None)

    failed = rollup_section(RollupSection.STRATEGY_QUALITY, [passing, failing, inconclusive])
    assert failed.status is UiStatus.FAIL
    assert "profit_factor" in failed.failed_keys and "Profit factor" in failed.main_reason

    blocked = rollup_section(
        RollupSection.PROBABILITY_SKILL,
        [passing, missing],
        required_keys=("brier_score",),
    )
    assert blocked.status is UiStatus.BLOCKED
    assert "brier_score" in blocked.missing_keys

    authorization = rollup_section(
        RollupSection.AUTHORIZATION_READINESS,
        [passing],
        missing_authorization="owner authorization readiness: missing",
    )
    assert authorization.status is UiStatus.BLOCKED
    assert "missing" in authorization.main_reason

    undecided = rollup_section(RollupSection.STABILITY, [passing, inconclusive])
    assert undecided.status is UiStatus.INCONCLUSIVE
    # a NON-required missing reading is insufficient evidence, never ignored
    thin = rollup_section(RollupSection.PROBABILITY_SKILL, [passing, missing])
    assert thin.status is UiStatus.INCONCLUSIVE

    cautious = rollup_section(RollupSection.STRATEGY_QUALITY, [passing, proposed])
    assert cautious.status is UiStatus.WARNING
    assert "proposed" in cautious.main_reason.lower()

    clean = rollup_section(RollupSection.STRATEGY_QUALITY, [passing, informational])
    assert clean.status is UiStatus.PASS
    assert clean.passed_count == 1

    descriptive = rollup_section(RollupSection.CALIBRATION, [informational])
    assert descriptive.status is UiStatus.INFORMATIONAL

    empty = rollup_section(RollupSection.ROBUSTNESS, [])
    assert empty.status is UiStatus.UNAVAILABLE
    for rollup in (failed, blocked, undecided, cautious, clean, descriptive, empty):
        assert rollup.sentence.strip() and rollup.inspect_next.strip()
        assert rollup.chip.split(" ", 1)[1] in rollup.sentence or rollup.chip


def test_rollups_are_deterministic_and_carry_the_section_label() -> None:
    readings = [
        evaluate_gate_flag("validity_report", True),
        evaluate_gate_flag("reconciliation_report", True),
        evaluate_metric("denied_attempt_count", 0),
    ]
    first = rollup_section(RollupSection.DATA_INTEGRITY, readings, inspect_next="the gate cards")
    second = rollup_section(RollupSection.DATA_INTEGRITY, readings, inspect_next="the gate cards")
    assert first == second
    assert first.status is UiStatus.PASS
    assert SECTION_LABELS[RollupSection.DATA_INTEGRITY] in first.sentence
    assert first.inspect_next == "the gate cards"
    corrupt_like = rollup_section(
        RollupSection.DATA_INTEGRITY,
        [evaluate_gate_flag("validity_report", False), *readings[1:]],
    )
    assert corrupt_like.status is UiStatus.FAIL
