"""UI-1 §5.4 / F-04 — contradictory drafts fail BEFORE freeze (pure report)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.charter_satisfiability import (
    StudyGoal,
    evaluate_charter_satisfiability,
    goal_for_draft,
    summarize_challenger_differences,
)
from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    EvidenceClass,
    RunPurpose,
)

_AXIS = "parent_retest_timeout_1m_bars"
_OTHER = "entry_near_parent"


def _report(**overrides):
    base = dict(
        goal=StudyGoal.FSM_SEARCH,
        search_mode="fsm_config_search",
        axis_selections={_AXIS: (f"{_AXIS}.240",)},
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        pareto_objectives=("net_expectancy_r",),
        tie_breaks=("profit_factor", "core_replay_id"),
        selected_contract_ids=(),
        launchable_contract_ids=(),
        purpose=RunPurpose.IMPLEMENTATION_VERIFICATION,
        evidence_class=EvidenceClass.SYNTHETIC_FIXTURE,
        prop_gates_configured=False,
        robustness_gates_configured=False,
    )
    base.update(overrides)
    return evaluate_charter_satisfiability(**base)


def _failed_ids(report) -> set[str]:
    return {rule.rule_id for rule in report.failures}


def test_goal_derivation_from_mode_and_question() -> None:
    assert goal_for_draft("fsm_config_search", "find_robust_fsm") is StudyGoal.FSM_SEARCH
    assert (
        goal_for_draft("single_configuration", "compare_one_with_baseline")
        is StudyGoal.COMPARE_WITH_BASELINE
    )
    assert (
        goal_for_draft("single_configuration", "evaluate_one_configuration")
        is StudyGoal.EVALUATE_ONE
    )
    assert (
        goal_for_draft("prop_benchmark", "repeat_payout_feasibility")
        is StudyGoal.PROP_FEASIBILITY
    )
    assert (
        goal_for_draft("universal_prop_search", "one_config_across_firms")
        is StudyGoal.UNIVERSAL_PROP
    )
    assert (
        goal_for_draft("full_pipeline_run", "find_robust_fsm") is StudyGoal.ADVANCED_END_TO_END
    )
    with pytest.raises(ValueError, match="study mode"):
        goal_for_draft("nope", "find_robust_fsm")


def test_satisfiable_fsm_search_passes_every_rule() -> None:
    report = _report()
    assert report.passed is True
    assert report.failures == ()
    assert {rule.rule_id for rule in report.rules} >= {
        "baseline_present",
        "fsm_search_has_a_challenger",
        "prop_objective_requires_a_verified_contract",
        "synthetic_fixture_confined_to_verification",
        "objectives_registered",
    }
    assert report.resolved_profile_count == 2
    assert report.challenger_count == 1
    assert "baseline + 1 challenger" in report.configuration_sentence


@pytest.mark.parametrize(
    ("overrides", "expected_rule"),
    [
        ({"axis_selections": {}}, "fsm_search_has_a_challenger"),
        (
            {
                "goal": StudyGoal.COMPARE_WITH_BASELINE,
                "search_mode": "single_configuration",
                "axis_selections": {},
            },
            "compare_requires_exactly_one_challenger",
        ),
        (
            {
                "goal": StudyGoal.COMPARE_WITH_BASELINE,
                "search_mode": "single_configuration",
                "axis_selections": {
                    _AXIS: (f"{_AXIS}.240",),
                    _OTHER: (f"{_OTHER}.within_40",),
                },
            },
            "compare_requires_exactly_one_challenger",
        ),
        (
            {
                "goal": StudyGoal.EVALUATE_ONE,
                "search_mode": "single_configuration",
            },
            "evaluate_requires_exactly_one_profile",
        ),
        (
            {"pareto_objectives": ("expected_net_payout_90d",)},
            "prop_objective_requires_a_verified_contract",
        ),
        (
            {"tie_breaks": ("breach_probability_90d", "core_replay_id")},
            "prop_objective_requires_a_verified_contract",
        ),
        (
            {
                "goal": StudyGoal.UNIVERSAL_PROP,
                "search_mode": "universal_prop_search",
                "pareto_objectives": ("payout_probability_per_rolling_30d",),
                "selected_contract_ids": ("c" * 64,),
                "launchable_contract_ids": ("c" * 64,),
            },
            "universal_requires_two_firm_contracts",
        ),
        (
            {
                "goal": StudyGoal.PROP_FEASIBILITY,
                "search_mode": "prop_benchmark",
                "pareto_objectives": ("payout_probability_per_rolling_30d",),
            },
            "prop_benchmark_requires_a_firm_contract",
        ),
        (
            {
                "evidence_class": EvidenceClass.REAL,
                "prop_gates_configured": True,
            },
            "verification_excludes_research_gates",
        ),
        (
            {"evidence_class": EvidenceClass.REAL},
            "verification_is_the_exact_baseline",
        ),
        (
            {"purpose": RunPurpose.DEVELOPMENT_RESEARCH},
            "synthetic_fixture_confined_to_verification",
        ),
        ({"pareto_objectives": ("sharpe",)}, "objectives_registered"),
        ({"baseline_profile_name": ""}, "baseline_present"),
    ],
)
def test_contradictory_drafts_fail_before_freeze(overrides, expected_rule) -> None:
    report = _report(**overrides)
    assert report.passed is False
    assert expected_rule in _failed_ids(report)
    failure = next(rule for rule in report.failures if rule.rule_id == expected_rule)
    assert failure.detail.strip()  # every failure explains itself


def test_selected_prop_objective_is_never_rewritten_only_blocked() -> None:
    blocked = _report(pareto_objectives=("expected_net_payout_90d",))
    assert blocked.passed is False
    # the report echoes the SELECTED objectives unchanged
    assert blocked.pareto_objectives == ("expected_net_payout_90d",)
    allowed = _report(
        pareto_objectives=("expected_net_payout_90d",),
        selected_contract_ids=("c" * 64,),
        launchable_contract_ids=("c" * 64,),
    )
    assert allowed.passed is True
    unverified = _report(
        pareto_objectives=("expected_net_payout_90d",),
        selected_contract_ids=("c" * 64,),
        launchable_contract_ids=(),
    )
    assert "prop_objective_requires_a_verified_contract" in _failed_ids(unverified)


def test_one_challenger_may_carry_multiple_registered_differences() -> None:
    challenger = {_AXIS: f"{_AXIS}.240", _OTHER: f"{_OTHER}.within_40"}
    report = _report(
        goal=StudyGoal.COMPARE_WITH_BASELINE,
        search_mode="single_configuration",
        axis_selections={},
        challenger_configurations=(challenger,),
    )
    assert report.passed is True
    assert report.resolved_profile_count == 2
    assert report.challenger_count == 1
    differences = summarize_challenger_differences(
        {_AXIS: f"{_AXIS}.none", _OTHER: f"{_OTHER}.false"}, challenger
    )
    assert len(differences) == 2
    assert any(_AXIS in line and "none" in line and "240" in line for line in differences)
    two = _report(
        goal=StudyGoal.COMPARE_WITH_BASELINE,
        search_mode="single_configuration",
        axis_selections={},
        challenger_configurations=(challenger, {_AXIS: f"{_AXIS}.360"}),
    )
    assert "compare_requires_exactly_one_challenger" in _failed_ids(two)


def test_real_verification_over_the_exact_baseline_passes() -> None:
    report = _report(
        goal=StudyGoal.VERIFICATION,
        search_mode="single_configuration",
        axis_selections={},
        evidence_class=EvidenceClass.REAL,
    )
    assert report.passed is True, report.failures
    assert report.resolved_profile_count == 1
    assert report.configuration_sentence == "Baseline only: 1 configuration"
