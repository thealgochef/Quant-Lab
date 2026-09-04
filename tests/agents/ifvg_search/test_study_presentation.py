"""Pure presentation logic: grouping, validators, funnel/stage derivations,
naming, pagination, estimates (FUX §§7–16, 24, 33; FUX-MON-001/002)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    SEARCH_AXIS_REGISTRY_V1,
    AxisClassification,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    ACTIVE_RUNS_COLUMNS,
    AXIS_GROUP_ORDER,
    EXPLORER_PRESETS,
    EXPLORER_STICKY_COLUMNS,
    FRONTIER_EXCLUSION_SENTINELS,
    FUNNEL_STAGE_LABELS,
    OBJECTIVE_TEMPLATES,
    PROP_SIM_FAILED_SENTINEL,
    RESEARCH_QUESTIONS,
    REUSED_UNEVALUATED_SENTINEL,
    STUDY_MODES,
    WIZARD_STEP_TITLES,
    axis_renders_widget,
    child_row_presentation,
    children_for_stage,
    classification_label,
    computation_path_chip,
    enumerate_child_count,
    estimate_search_work,
    format_duration,
    format_storage,
    funnel_counts,
    grouped_axis_keys,
    heatmap_cell_class,
    human_config_name,
    mode_compatibility_error,
    page_slice,
    validate_objective_step,
    validate_review_step,
    validate_search_space_step,
    validate_validation_step,
    viewport_class,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (
    NOT_RUN_STRATEGY_GATE_TEXT,
    StudyStatusKey,
)

_ORCHESTRATOR_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "src/alpha_lab/agents/data_infra/ifvg/search/orchestrator.py"
).read_text(encoding="utf-8")


def _child(**overrides) -> dict:
    child = {
        "ordinal": 0,
        "core_replay_id": "a" * 64,
        "axis_value_ids": {},
        "comparison_role": "baseline",
        "state": "completed",
        "failure_reason": None,
        "explanation": "",
        "replay_invocations": 1,
    }
    child.update(overrides)
    return child


# ── wizard registries ───────────────────────────────────────────────────────


def test_wizard_shape_is_exact() -> None:
    """FUX §§6–8: 8 steps, 5 modes, 4 questions, 6 templates."""

    assert WIZARD_STEP_TITLES == (
        "Objective",
        "Baseline",
        "Strategy Search Space",
        "Prop Contracts",
        "Risk Policies",
        "Benchmarks",
        "Validation",
        "Review & Launch",
    )
    assert [mode.label for mode in STUDY_MODES] == [
        "Single Configuration",
        "FSM Configuration Search",
        "Prop Benchmark",
        "Universal Prop Search",
        "Full Pipeline Run",
    ]
    assert list(RESEARCH_QUESTIONS.values()) == [
        "Evaluate one configuration",
        "Compare one configuration with the baseline",
        "Find a robust FSM configuration",
        "Test repeat-payout feasibility",
        "Find one strategy configuration across multiple firms",
    ]
    assert [template.label for template in OBJECTIVE_TEMPLATES] == [
        "Payout Reliability",
        "Maximum Expected Payout",
        "Low Breach / Long Account Life",
        "Balanced Prop Performance",
        "Strategy Quality Only",
        "Custom",
    ]
    for template in OBJECTIVE_TEMPLATES:
        if template.template_id != "custom":
            assert template.status == "proposed_protocol_default"


def test_mode_compatibility_explains_before_next() -> None:
    assert mode_compatibility_error(
        "fsm_config_search", "find_robust_fsm", "payout_reliability"
    ) is None
    error = mode_compatibility_error(
        "prop_benchmark", "find_robust_fsm", "payout_reliability"
    )
    assert error is not None and "different study mode" in error
    error = mode_compatibility_error(
        "universal_prop_search", "one_config_across_firms", "strategy_quality_only"
    )
    assert error is not None and "prop" in error.lower()


# ── axis grouping / cards ───────────────────────────────────────────────────


def test_every_registered_axis_lands_in_a_fux_group() -> None:
    grouped = grouped_axis_keys()
    assert set(grouped) <= set(AXIS_GROUP_ORDER)  # no 'Other' bucket today
    covered = {key for keys in grouped.values() for key in keys}
    assert covered == set(SEARCH_AXIS_REGISTRY_V1)


def test_classification_labels_project_to_the_five_fux_labels() -> None:
    labels = {
        classification_label(spec.classification)
        for spec in SEARCH_AXIS_REGISTRY_V1.values()
    }
    assert labels <= {
        "Locked Invariant",
        "Search Axis",
        "Measured Only",
        "Blocked",
        "Experimental",
    }


def test_locked_and_blocked_axes_render_no_widget() -> None:
    """FUX §10.3 — via the single gating predicate every card uses."""

    for spec in SEARCH_AXIS_REGISTRY_V1.values():
        if spec.classification in (
            AxisClassification.LOCKED_INVARIANT,
            AxisClassification.THESIS_DEFINING,
            AxisClassification.MEASUREMENT_ONLY,
            AxisClassification.BLOCKED,
        ):
            assert not axis_renders_widget(spec), spec.technical_key
        chip = computation_path_chip(spec)
        assert chip in (
            "Requires New Sequential Replay",
            "Analysis Filter",
            "Prop Resimulation",
        )


# ── step validators ─────────────────────────────────────────────────────────


def test_objective_validator_rejects_unknowns_and_incompatibility() -> None:
    errors = validate_objective_step(
        {"mode_id": "nope", "question_id": "?", "template_id": "?"}
    )
    assert set(errors) == {"mode_id", "question_id", "template_id"}
    errors = validate_objective_step(
        {
            "mode_id": "prop_benchmark",
            "question_id": "find_robust_fsm",
            "template_id": "payout_reliability",
        }
    )
    assert "template_id" in errors
    errors = validate_objective_step(
        {
            "mode_id": "fsm_config_search",
            "question_id": "find_robust_fsm",
            "template_id": "custom",
            "custom_objectives": ("not_a_metric",),
        }
    )
    assert "unregistered" in errors["custom_objectives"]


def test_search_space_validator_enforces_registry_and_ceiling() -> None:
    errors = validate_search_space_step(
        {"axis_selections": {"parent_retest_timeout_1m_bars": ("bogus.value",)}}
    )
    assert "parent_retest_timeout_1m_bars" in errors
    errors = validate_search_space_step(
        {"axis_selections": {"break_even_enabled": ("break_even_enabled.baseline",)}}
    )
    assert "Blocked" in errors["break_even_enabled"]
    spec = SEARCH_AXIS_REGISTRY_V1["parent_retest_timeout_1m_bars"]
    values = tuple(spec.registered_values)
    errors = validate_search_space_step(
        {"axis_selections": {"parent_retest_timeout_1m_bars": values}},
        max_child_count=2,
    )
    assert "ceiling" in errors["axis_selections"]
    errors = validate_search_space_step(
        {
            "mode_id": "single_configuration",
            "axis_selections": {"parent_retest_timeout_1m_bars": values[:2]},
        }
    )
    assert "Single Configuration" in errors["axis_selections"]


def test_interpretation_selector_gates_enumeration() -> None:
    """FUX §10.5 (adversarial F2): only Sequential Strategy Profile creates
    an executable counterfactual — descriptive interpretations over changed
    session/direction axes refuse at validation, never silently replay."""

    spec = SEARCH_AXIS_REGISTRY_V1["enable_shorts"]
    selection = {"enable_shorts": tuple(spec.registered_values[:1])}
    for interpretation in ("Descriptive Slice", "Specialized Model"):
        errors = validate_search_space_step(
            {"axis_selections": selection, "interpretation": interpretation}
        )
        assert "never replays" in errors["interpretation"]
    assert (
        validate_search_space_step(
            {
                "axis_selections": selection,
                "interpretation": "Sequential Strategy Profile",
            }
        )
        == {}
    )
    # non-session axes are unaffected by the interpretation
    other = {
        "parent_retest_timeout_1m_bars": tuple(
            SEARCH_AXIS_REGISTRY_V1[
                "parent_retest_timeout_1m_bars"
            ].registered_values[:1]
        )
    }
    assert (
        validate_search_space_step(
            {"axis_selections": other, "interpretation": "Descriptive Slice"}
        )
        == {}
    )


def test_remaining_step_validators_units() -> None:
    """F18a: baseline / prop / risk / benchmarks validators."""

    from alpha_lab.agents.data_infra.ifvg.study_presentation import (
        validate_baseline_step,
        validate_prop_step,
        validate_risk_step,
    )
    from alpha_lab.agents.data_infra.ifvg.study_presentation import (
        validate_benchmarks_step as validate_benchmarks,
    )

    assert "baseline_profile_name" in validate_baseline_step({})
    assert "not runnable" in validate_baseline_step(
        {
            "baseline_profile_name": "x",
            "baseline_blocked_reason": "blocked: reason",
        }
    )["baseline_profile_name"]
    assert (
        validate_baseline_step({"baseline_profile_name": "ok"}) == {}
    )

    assert "at least one" in validate_prop_step(
        {"mode_id": "prop_benchmark", "selected_contract_ids": ()}
    )["selected_contract_ids"]
    assert "cannot launch" in validate_prop_step(
        {
            "mode_id": "prop_benchmark",
            "selected_contract_ids": ("c" * 64,),
            "launchable_contract_ids": (),
        }
    )["selected_contract_ids"]
    assert (
        validate_prop_step(
            {
                "mode_id": "fsm_config_search",
                "selected_contract_ids": (),
                "launchable_contract_ids": (),
            }
        )
        == {}
    )

    contract = "c" * 64
    assert "assign" in validate_risk_step(
        {"selected_contract_ids": (contract,), "per_firm_policies": {}}
    )[contract]
    assert "registered risk template" in validate_risk_step(
        {
            "selected_contract_ids": (contract,),
            "per_firm_policies": {contract: {"risk_template": "bogus"}},
        }
    )[contract]
    assert "owner decision 13" in validate_risk_step(
        {
            "selected_contract_ids": (contract,),
            "per_firm_policies": {
                contract: {"risk_template": "Fixed Dollar", "n_accounts": 3}
            },
        }
    )[contract]
    assert (
        validate_risk_step(
            {
                "selected_contract_ids": (contract,),
                "per_firm_policies": {
                    contract: {"risk_template": "Fixed Dollar", "n_accounts": 2}
                },
            }
        )
        == {}
    )

    assert "must be numeric" in validate_benchmarks(
        {"strategy_gates": {"min_profit_factor": "high"}}
    )["strategy_gates.min_profit_factor"]
    assert "[0, 1]" in validate_benchmarks(
        {"prop_gates": {"maximum_breach_probability_90d": 3.5}}
    )["prop_gates.maximum_breach_probability_90d"]
    assert validate_benchmarks({"strategy_gates": {"max_drawdown_r": 15.0}}) == {}


def test_validation_step_enforces_sequential_v1_and_development_dates() -> None:
    """UI-1: no worker value above one is accepted (HARDENING-BACKEND §4.6)
    and full-scope evidence dates follow the backend logical-day contract."""

    from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES

    base = {"run_scope": "verification_5d", "seed": 7, "worker_limit": 1}
    assert "worker_limit" in validate_validation_step({**base, "worker_limit": 4})
    assert "effective workers: 1" in validate_validation_step(
        {**base, "worker_limit": 2}
    )["worker_limit"]
    full = {
        "run_scope": "full_authorized_development",
        "seed": 7,
        "worker_limit": 1,
        "warmup_dates": FROZEN_WARMUP_DATES,
    }
    assert validate_validation_step({**full, "real_dates": ("2026-01-13", "2026-01-14")}) == {}
    assert "real_dates" in validate_validation_step({**full, "real_dates": ()})
    weekend = validate_validation_step({**full, "real_dates": ("2026-01-17",)})
    assert "not a logical trading day" in weekend["real_dates"]
    protected = validate_validation_step({**full, "real_dates": ("2026-06-11",)})
    assert "outside the development evidence window" in protected["real_dates"]
    unordered = validate_validation_step(
        {**full, "real_dates": ("2026-01-14", "2026-01-13")}
    )
    assert "chronological" in unordered["real_dates"]
    no_prefix = validate_validation_step(
        {**full, "real_dates": ("2026-01-13",), "warmup_dates": ()}
    )
    assert "frozen ten-date warmup prefix" in no_prefix["warmup_dates"]
    assert "evidence_class" in validate_validation_step({**base, "evidence_class": "maybe"})


def test_validation_step_enforces_the_five_day_budget() -> None:
    base = {"run_scope": "verification_5d", "seed": 7, "worker_limit": 1}
    ok = validate_validation_step(
        {**base, "real_dates": ("2026-06-04", "2026-06-05")}
    )
    assert ok == {}
    errors = validate_validation_step(
        {**base, "real_dates": tuple(f"2026-06-{day:02d}" for day in range(1, 7))}
    )
    assert "at most 5" in errors["real_dates"]
    errors = validate_validation_step(
        {
            **base,
            "real_dates": ("2026-06-04",) * 4,
            "warmup_dates": ("2026-06-01", "2026-06-02"),
        }
    )
    assert "5-day budget" in errors["warmup_dates"]


def test_review_step_requires_the_typed_full_scope_acknowledgement() -> None:
    errors = validate_review_step(
        {
            "run_scope": "full_authorized_development",
            "typed_acknowledgement": "nope",
            "required_acknowledgement": "run full authorized development",
        }
    )
    assert "typed_acknowledgement" in errors
    assert (
        validate_review_step(
            {
                "run_scope": "full_authorized_development",
                "typed_acknowledgement": "Run Full Authorized Development",
                "required_acknowledgement": "run full authorized development",
            }
        )
        == {}
    )
    assert validate_review_step({"run_scope": "verification_5d"}) == {}


def test_enumerate_child_count_is_the_cartesian_product() -> None:
    assert enumerate_child_count({}) == 1
    assert enumerate_child_count({"a": ("1", "2"), "b": ("x", "y", "z")}) == 6


# ── funnel counts and stage cells ───────────────────────────────────────────


def test_orchestrator_sentinels_are_pinned_to_source() -> None:
    """The state file carries no per-stage booleans; these exact strings are
    the contract between the orchestrator and the monitor derivations."""

    # join adjacent implicit string-literal fragments before scanning
    normalized = re.sub(r'"\s*\n\s*"', "", _ORCHESTRATOR_SOURCE)
    assert REUSED_UNEVALUATED_SENTINEL in normalized
    assert '"prop simulation failed: "' in normalized
    assert PROP_SIM_FAILED_SENTINEL == "prop simulation failed:"
    for sentinel in FRONTIER_EXCLUSION_SENTINELS:
        assert sentinel in normalized


def test_funnel_counts_full_mapping_table() -> None:
    children = [
        _child(core_replay_id="a" * 64),  # fully feasible
        _child(core_replay_id="b" * 64, state="failed", failure_reason="replay"),
        _child(
            core_replay_id="c" * 64,
            failure_reason="insufficient_trades",
            explanation="strategy gate failed",
        ),
        _child(
            core_replay_id="d" * 64,
            failure_reason="breach",
            explanation="prop gates failed on x",
        ),
        _child(
            core_replay_id="e" * 64,
            failure_reason="replay",
            explanation="prop simulation failed: boom",
        ),
        _child(
            core_replay_id="f" * 64,
            state="reused",
            explanation=(
                "reused replay has no published costed evaluation for this "
                "cost policy; gates were not evaluated this run"
            ),
        ),
        _child(core_replay_id="1" * 64, state="blocked"),
    ]
    counts = funnel_counts(children)
    assert counts.generated == 7
    assert counts.replay_valid == 5  # failed replay + blocked excluded
    assert counts.strategy_pass == 3  # gate fail + unevaluated reuse excluded
    assert counts.prop_feasible == 1  # prop fail + prop crash excluded
    assert counts.robust is None  # unknown before the frontier exists
    labels = counts.as_button_labels()
    assert labels[0] == "Generated · 7"
    assert labels[4] == "Robust · –"
    assert list(FUNNEL_STAGE_LABELS) == [
        "Generated",
        "Replay Valid",
        "Strategy Pass",
        "Prop Feasible",
        "Robust",
    ]
    # the persisted frontier is authoritative when present
    with_frontier = funnel_counts(
        children, feasible_ids=("a" * 64,), frontier_ids=("a" * 64,)
    )
    assert with_frontier.prop_feasible == 1
    assert with_frontier.robust == 1
    robust_children = children_for_stage(
        children, "Robust", frontier_ids=("a" * 64,)
    )
    assert [c["core_replay_id"] for c in robust_children] == ["a" * 64]
    with pytest.raises(ValueError, match="unknown funnel stage"):
        children_for_stage(children, "Nope")


def test_child_rows_carry_exact_skipped_stage_copy() -> None:
    """FUX §16.4: stages not run display the exact earlier-gate reason."""

    strategy_fail = child_row_presentation(
        _child(failure_reason="insufficient_trades", explanation="too few")
    )
    assert strategy_fail.prop_simulation == NOT_RUN_STRATEGY_GATE_TEXT
    assert strategy_fail.status_key is StudyStatusKey.STRATEGY_REJECTED
    assert strategy_fail.as_row()["Prop Simulation"] == (
        "Not run — strategy gate failed"
    )

    replay_fail = child_row_presentation(
        _child(state="failed", failure_reason="replay", explanation="boom")
    )
    assert replay_fail.status_key is StudyStatusKey.REPLAY_FAILED
    assert replay_fail.strategy_gate == "Not run — replay failed"

    prop_fail = child_row_presentation(
        _child(failure_reason="fees", explanation="prop gates failed on f")
    )
    assert prop_fail.status_key is StudyStatusKey.PROP_REJECTED
    assert prop_fail.strategy_gate == "✓ Passed"

    prop_crash = child_row_presentation(
        _child(failure_reason="replay", explanation="prop simulation failed: x")
    )
    assert prop_crash.status_key is StudyStatusKey.PROP_REJECTED
    assert prop_crash.prop_simulation == "✕ Simulation failed"

    reused_unevaluated = child_row_presentation(
        _child(
            state="reused",
            explanation=(
                "reused replay has no published costed evaluation for this "
                "cost policy; gates were not evaluated this run"
            ),
        )
    )
    assert reused_unevaluated.status_key is StudyStatusKey.BLOCKED
    assert "Not evaluated" in reused_unevaluated.strategy_gate

    representative = child_row_presentation(
        _child(),
        feasible_ids=("a" * 64,),
        frontier_ids=("a" * 64,),
        representative_id="a" * 64,
    )
    assert representative.status_key is StudyStatusKey.SELECTED_REPRESENTATIVE

    finalist = child_row_presentation(_child(), feasible_ids=("a" * 64,))
    assert finalist.status_key is StudyStatusKey.ROBUST_FINALIST
    assert list(ACTIVE_RUNS_COLUMNS) == [
        "Config",
        "Replay",
        "Strategy Gate",
        "Prop Simulation",
        "Robustness",
        "Status",
        "Human Explanation",
    ]


# ── naming, presets, pagination, estimates, responsiveness ─────────────────


def test_human_names_derive_from_the_baseline_diff() -> None:
    assert human_config_name({}) == "Baseline (doc-default)"
    spec = SEARCH_AXIS_REGISTRY_V1["parent_retest_timeout_1m_bars"]
    assert (
        human_config_name(
            {"parent_retest_timeout_1m_bars": spec.baseline_value_id}
        )
        == "Baseline (doc-default)"
    )
    challenger = next(
        value
        for value in spec.registered_values
        if value != spec.baseline_value_id
    )
    name = human_config_name({"parent_retest_timeout_1m_bars": challenger})
    assert name != "Baseline (doc-default)"
    assert len(name) > 3


def test_explorer_presets_are_exact() -> None:
    assert list(EXPLORER_PRESETS) == ["Strategy", "Prop", "Robustness"]
    assert EXPLORER_PRESETS["Strategy"] == (
        "trade count",
        "net E[R]",
        "realized payoff ratio",
        "profit factor",
        "max DD R",
        "trade frequency",
        "setup occupancy",
    )
    assert "Q10 payout" in EXPLORER_PRESETS["Prop"]
    assert "concentration warning" in EXPLORER_PRESETS["Robustness"]
    assert EXPLORER_STICKY_COLUMNS == (
        "rank",
        "config name",
        "status",
        "changed parameters",
    )


def test_page_slice_clamps_and_counts() -> None:
    assert page_slice(0, 0, 25) == (0, 0, 1)
    assert page_slice(100, 0, 25) == (0, 25, 4)
    assert page_slice(100, 99, 25) == (75, 100, 4)  # page clamped into range
    with pytest.raises(ValueError, match="positive"):
        page_slice(10, 0, 0)


def test_estimates_are_operational_annotations() -> None:
    estimate = estimate_search_work(
        n_children=4,
        n_replay_days=5,
        reuse_hits=1,
        n_firm_policy_combinations=2,
    )
    assert estimate.full_replay_count == 3
    assert estimate.verified_reuse_hits == 1
    assert estimate.historical_prop_replays == 8
    assert estimate.estimated_runtime_seconds > 0
    assert format_duration(30) == "30 s"
    assert format_duration(600) == "10 min"
    assert format_storage(100) == "100 MB"
    assert format_storage(4096) == "4.0 GB"


def test_viewport_classes_match_fux_breakpoints() -> None:
    assert viewport_class(None) == "desktop"
    assert viewport_class(1440) == "desktop"
    assert viewport_class(1024) == "desktop"
    assert viewport_class(768) == "tablet"
    assert viewport_class(390) == "mobile"


def test_heatmap_cell_classes() -> None:
    assert heatmap_cell_class(value=1.0, sample_count=5) == "stable_plateau"
    assert (
        heatmap_cell_class(value=1.0, sample_count=5, knife_edge=True)
        == "knife_edge_point"
    )
    assert heatmap_cell_class(value=None, sample_count=5) == "insufficient_data"
    assert heatmap_cell_class(value=1.0, sample_count=0) == "insufficient_data"
    assert (
        heatmap_cell_class(value=1.0, sample_count=5, failed=True)
        == "failed_region"
    )
    assert (
        heatmap_cell_class(value=None, sample_count=0, blocked=True)
        == "blocked_cell"
    )


def test_no_fuzzy_fallback_tokens_in_presentation_source() -> None:
    """FUX-DRILL-001 source scan: no nearest/keep-last/fuzzy matching."""

    source = (
        Path(__file__).resolve().parents[3]
        / "src/alpha_lab/agents/data_infra/ifvg/study_presentation.py"
    ).read_text(encoding="utf-8")
    for token in ("nearest", "keep_last", "keep-last", "fuzzy"):
        assert not re.search(token, source, re.IGNORECASE)
