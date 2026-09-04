"""UI-2 (plan §5.4 / §9 Phase 2): goal-derived conditional study flows.

Steps derive from the card's goal, the purpose and the evidence class;
skipped steps are LISTED with a reason (never rendered empty); a selected
prop objective keeps the prop steps in a strategy goal's flow so the
contradiction blocks instead of disappearing; the Validation step stays in
every research flow; a step skipped by the flow contributes nothing; exact
restore works under a conditional flow and for legacy eight-step indices.
"""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.charter_satisfiability import StudyGoal
from alpha_lab.agents.data_infra.ifvg.presentation.flows import (
    FLOW_STEP_TITLES,
    FlowStep,
    StudyFlow,
    flow_for_draft_fields,
    flow_for_goal,
    goal_for_flow,
    prop_objective_selected,
    restore_step_index,
    step_position,
)
from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import EvidenceClass, RunPurpose
from alpha_lab.agents.data_infra.ifvg.study_drafts import STEP_KEYS


def _keys(flow: StudyFlow) -> tuple[str, ...]:
    return tuple(step.key for step in flow.steps)


def _skipped(flow: StudyFlow) -> dict[str, str]:
    return {step.key: step.skip_reason or "" for step in flow.skipped}


def test_every_step_key_is_either_included_or_skipped_with_a_reason() -> None:
    for goal in StudyGoal:
        for purpose in RunPurpose:
            if goal is StudyGoal.VERIFICATION and purpose is not (
                RunPurpose.IMPLEMENTATION_VERIFICATION
            ):
                continue
            evidence = (
                EvidenceClass.SYNTHETIC_FIXTURE
                if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
                else EvidenceClass.REAL
            )
            flow = flow_for_goal(
                goal, purpose=purpose, evidence_class=evidence, prop_objective=False
            )
            included = _keys(flow)
            skipped = _skipped(flow)
            assert set(included) | set(skipped) == set(STEP_KEYS), (goal, purpose)
            assert not set(included) & set(skipped)
            assert included[0] == "objective" and included[-1] == "review"
            assert "validation" in included  # dates / seed / checklist live here
            for step in flow.steps:
                assert isinstance(step, FlowStep) and step.title and step.skip_reason is None
            for key, reason in skipped.items():
                assert reason, (goal, key)  # every skipped step explains itself
            # the flow keeps the canonical step order
            assert [STEP_KEYS.index(key) for key in included] == sorted(
                STEP_KEYS.index(key) for key in included
            )


def test_strategy_goals_skip_prop_steps_unless_a_prop_objective_is_selected() -> None:
    for goal in (StudyGoal.EVALUATE_ONE, StudyGoal.COMPARE_WITH_BASELINE, StudyGoal.FSM_SEARCH):
        without = flow_for_goal(
            goal,
            purpose=RunPurpose.DEVELOPMENT_RESEARCH,
            evidence_class=EvidenceClass.REAL,
            prop_objective=False,
        )
        assert "prop_contracts" not in _keys(without)
        assert "risk_policies" not in _keys(without)
        assert "no prop objective" in _skipped(without)["prop_contracts"]
        with_prop = flow_for_goal(
            goal,
            purpose=RunPurpose.DEVELOPMENT_RESEARCH,
            evidence_class=EvidenceClass.REAL,
            prop_objective=True,
        )
        # the selected prop objective keeps the contract step IN the flow so
        # the missing contract blocks there instead of disappearing (plan §7)
        assert "prop_contracts" in _keys(with_prop)
        assert "risk_policies" in _keys(with_prop)
        step = next(step for step in with_prop.steps if step.key == "prop_contracts")
        assert "prop objective" in step.reason


def test_goal_specific_steps_and_titles() -> None:
    research = dict(purpose=RunPurpose.DEVELOPMENT_RESEARCH, evidence_class=EvidenceClass.REAL)
    evaluate = flow_for_goal(StudyGoal.EVALUATE_ONE, prop_objective=False, **research)
    assert "search_space" not in _keys(evaluate)
    assert "no comparison" in _skipped(evaluate)["search_space"]
    assert next(s.title for s in evaluate.steps if s.key == "baseline") == "Configuration"
    compare = flow_for_goal(StudyGoal.COMPARE_WITH_BASELINE, prop_objective=False, **research)
    assert next(s.title for s in compare.steps if s.key == "search_space") == "Challenger"
    search = flow_for_goal(StudyGoal.FSM_SEARCH, prop_objective=False, **research)
    assert next(s.title for s in search.steps if s.key == "search_space") == "Search axes"
    prop = flow_for_goal(StudyGoal.PROP_FEASIBILITY, prop_objective=True, **research)
    assert _keys(prop) == (
        "objective",
        "baseline",
        "prop_contracts",
        "risk_policies",
        "benchmarks",
        "validation",
        "review",
    )
    assert next(s.title for s in prop.steps if s.key == "baseline") == "Strategy source"
    assert "≥ 1" in next(s.title for s in prop.steps if s.key == "prop_contracts")
    universal = flow_for_goal(StudyGoal.UNIVERSAL_PROP, prop_objective=True, **research)
    assert "search_space" in _keys(universal)
    assert "≥ 2" in next(s.title for s in universal.steps if s.key == "prop_contracts")
    feature = flow_for_goal(StudyGoal.FEATURE_MODEL, prop_objective=False, **research)
    assert "search_space" not in _keys(feature)
    assert next(s.title for s in feature.steps if s.key == "baseline") == "Cohort anchor"
    review = next(s for s in feature.steps if s.key == "review")
    assert "Feature bundles" in review.reason and "Model ladder" in review.reason
    advanced = flow_for_goal(
        StudyGoal.ADVANCED_END_TO_END,
        purpose=RunPurpose.FULL_AUTHORIZED_DEVELOPMENT,
        evidence_class=EvidenceClass.REAL,
        prop_objective=False,
    )
    assert _keys(advanced) == STEP_KEYS  # every applicable step, then Configure / Review / Launch
    assert not advanced.skipped


def test_real_verification_is_the_exact_baseline_flow_and_synthetic_keeps_the_family() -> None:
    real = flow_for_goal(
        StudyGoal.VERIFICATION,
        purpose=RunPurpose.IMPLEMENTATION_VERIFICATION,
        evidence_class=EvidenceClass.REAL,
        prop_objective=False,
    )
    assert _keys(real) == ("objective", "baseline", "validation", "review")
    skipped = _skipped(real)
    assert "exact baseline" in skipped["search_space"]
    assert "verification" in skipped["benchmarks"].lower()
    assert "verification" in skipped["prop_contracts"].lower()
    # a synthetic fixture under Implementation Verification keeps its study
    # family's flow (the R4-era synthetic search stays reachable)
    synthetic = flow_for_goal(
        StudyGoal.FSM_SEARCH,
        purpose=RunPurpose.IMPLEMENTATION_VERIFICATION,
        evidence_class=EvidenceClass.SYNTHETIC_FIXTURE,
        prop_objective=False,
    )
    assert "search_space" in _keys(synthetic)
    assert synthetic.flow_id != real.flow_id


def test_goal_for_flow_refines_the_pipeline_family_by_purpose() -> None:
    assert goal_for_flow(
        "fsm_config_search", "find_robust_fsm", RunPurpose.DEVELOPMENT_RESEARCH
    ) is (StudyGoal.FSM_SEARCH)
    assert (
        goal_for_flow("full_pipeline_run", "find_robust_fsm", RunPurpose.DEVELOPMENT_RESEARCH)
        is StudyGoal.FEATURE_MODEL
    )
    assert (
        goal_for_flow(
            "full_pipeline_run", "find_robust_fsm", RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
        )
        is StudyGoal.ADVANCED_END_TO_END
    )
    assert (
        goal_for_flow(
            "single_configuration", "evaluate_one_configuration", RunPurpose.DEVELOPMENT_RESEARCH
        )
        is StudyGoal.EVALUATE_ONE
    )
    with pytest.raises(ValueError, match="unknown study mode"):
        goal_for_flow("nope", None, RunPurpose.DEVELOPMENT_RESEARCH)


def test_flow_for_draft_fields_detects_the_prop_objective_and_the_real_verification_path():
    fields = {
        "mode_id": "fsm_config_search",
        "question_id": "find_robust_fsm",
        "template_id": "payout_reliability",
        "custom_objectives": (),
    }
    assert prop_objective_selected(fields) is True
    flow = flow_for_draft_fields(
        fields, purpose=RunPurpose.DEVELOPMENT_RESEARCH, evidence_class=EvidenceClass.REAL
    )
    assert "prop_contracts" in _keys(flow)
    strategy_only = {**fields, "template_id": "strategy_quality_only"}
    assert prop_objective_selected(strategy_only) is False
    custom = {**fields, "template_id": "custom", "custom_objectives": ("net_expectancy_r",)}
    assert prop_objective_selected(custom) is False
    custom_prop = {**custom, "custom_objectives": ("breach_probability_90d",)}
    assert prop_objective_selected(custom_prop) is True
    real = flow_for_draft_fields(
        {**fields, "mode_id": "single_configuration", "question_id": "evaluate_one_configuration"},
        purpose=RunPurpose.IMPLEMENTATION_VERIFICATION,
        evidence_class=EvidenceClass.REAL,
    )
    assert real.goal is StudyGoal.VERIFICATION
    assert _keys(real) == ("objective", "baseline", "validation", "review")


def test_exact_restore_under_a_conditional_flow_and_for_legacy_indices() -> None:
    flow = flow_for_goal(
        StudyGoal.FSM_SEARCH,
        purpose=RunPurpose.DEVELOPMENT_RESEARCH,
        evidence_class=EvidenceClass.REAL,
        prop_objective=False,
    )
    assert _keys(flow) == (
        "objective",
        "baseline",
        "search_space",
        "benchmarks",
        "validation",
        "review",
    )
    assert step_position(flow, "benchmarks") == 3
    assert step_position(flow, "prop_contracts") is None
    # a stored step key restores exactly
    assert restore_step_index(flow, stored_step_key="validation", legacy_step_index=0) == 4
    # a legacy eight-step index maps to the same key, or the next present step
    assert restore_step_index(flow, stored_step_key=None, legacy_step_index=2) == 2  # search_space
    assert (
        restore_step_index(flow, stored_step_key=None, legacy_step_index=3) == 3
    )  # prop → benchmarks
    assert restore_step_index(flow, stored_step_key=None, legacy_step_index=7) == 5  # review
    assert restore_step_index(flow, stored_step_key=None, legacy_step_index=99) == 5
    assert restore_step_index(flow, stored_step_key="nope", legacy_step_index=1) == 1
    assert set(FLOW_STEP_TITLES) == set(STEP_KEYS)
