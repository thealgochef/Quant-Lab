"""Pure presentation logic for the FSM/prop study workspace (R4).

Everything in this module is Streamlit-free and unit-testable: wizard step
registries and validators, mode/question/template compatibility, axis
grouping by market meaning, computation-path chips, human configuration
names derived from the baseline diff, Active Runs funnel/stage derivations,
explorer column presets, pagination, work estimates (operational
annotations, never identity), and responsive layout decisions
(``FRONTEND_UX_CONTRACT.md`` §4 module table; CS §13).

Nothing here produces or alters a research identity. Estimate output is an
operational annotation only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import prod
from types import MappingProxyType
from typing import Any, Literal

from .search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    AxisClassification,
    SearchAxisSpec,
)
from .search.charter import (
    MAX_CHILD_COUNT_CEILING,
    OBJECTIVE_DIRECTIONS,
)
from .search.failure import FailureReason
from .study_status import (
    NOT_RUN_STRATEGY_GATE_TEXT,
    StudyStatusKey,
)

__all__ = [
    "AXIS_GROUP_ORDER",
    "AXIS_GROUP_ASSIGNMENTS",
    "CLASSIFICATION_LABELS",
    "SESSION_DIRECTION_AXES",
    "INTERPRETATIONS",
    "axis_group_for",
    "grouped_axis_keys",
    "classification_label",
    "axis_renders_widget",
    "computation_path_chip",
    "WIZARD_STEP_TITLES",
    "STUDY_MODES",
    "StudyModeSpec",
    "RESEARCH_QUESTIONS",
    "OBJECTIVE_TEMPLATES",
    "ObjectiveTemplate",
    "mode_compatibility_error",
    "validate_objective_step",
    "validate_baseline_step",
    "validate_search_space_step",
    "validate_prop_step",
    "validate_risk_step",
    "validate_benchmarks_step",
    "validate_validation_step",
    "validate_review_step",
    "enumerate_child_count",
    "FunnelCounts",
    "funnel_counts",
    "children_for_stage",
    "ChildRowPresentation",
    "child_row_presentation",
    "STRATEGY_STAGE_REASONS",
    "PROP_STAGE_REASONS",
    "REPLAY_STAGE_REASONS",
    "REUSED_UNEVALUATED_SENTINEL",
    "FRONTIER_EXCLUSION_SENTINELS",
    "PROP_SIM_FAILED_SENTINEL",
    "human_config_name",
    "changed_axis_labels",
    "EXPLORER_PRESETS",
    "EXPLORER_STICKY_COLUMNS",
    "ACTIVE_RUNS_COLUMNS",
    "FUNNEL_STAGE_LABELS",
    "HEATMAP_METRICS",
    "FIRM_MATRIX_METRICS",
    "PAYOUT_HORIZONS",
    "heatmap_cell_class",
    "page_slice",
    "WorkEstimate",
    "estimate_search_work",
    "format_duration",
    "format_storage",
    "viewport_class",
    "RISK_POLICY_TEMPLATES",
]


# ─────────────────────────────────────────────────────────────────────────────
# Axis grouping by market meaning (FUX §10.1) and classification labels
# ─────────────────────────────────────────────────────────────────────────────

AXIS_GROUP_ORDER: tuple[str, ...] = (
    "Staleness",
    "Parent Handling",
    "HTF Selection",
    "Causality & Locality",
    "Entry Timing",
    "Session Policy",
    "Risk Admissibility",
)

#: Engineering-default assignment of every registered axis to one FUX §10.1
#: market-meaning group (recorded in DECISIONS_TAKEN; the registry stays the
#: identity authority — this mapping is presentation only).
AXIS_GROUP_ASSIGNMENTS: Mapping[str, str] = MappingProxyType(
    {
        # Staleness
        "parent_retest_timeout_1m_bars": "Staleness",
        "opposing_timeout_1m_bars": "Staleness",
        "inversion_timeout_1m_bars": "Staleness",
        "post_inversion_expiry_1m_bars_max": "Staleness",
        "htf_registry_max_age_days": "Staleness",
        # Parent Handling
        "parent_reaction_window_parent_bars": "Parent Handling",
        "parent_reaction_window_1m_bars_max": "Parent Handling",
        "retest_trigger": "Parent Handling",
        "parent_full_fill_invalidation": "Parent Handling",
        "parent_structural_invalidation": "Parent Handling",
        # HTF Selection
        "htf_timeframes": "HTF Selection",
        "parent_timeframes": "HTF Selection",
        "htf_selection_max_per_timeframe": "HTF Selection",
        "ltf_registry_max_live": "HTF Selection",
        "min_gap_ticks_capture": "HTF Selection",
        "swing_strength_bars": "HTF Selection",
        "swing_pool_max": "HTF Selection",
        "anchor_policy": "HTF Selection",
        # Causality & Locality
        "causality_parent": "Causality & Locality",
        "causality_opposing": "Causality & Locality",
        "causality_entry": "Causality & Locality",
        "parent_htf_distance_ticks_max": "Causality & Locality",
        "opposing_parent_distance_ticks_max": "Causality & Locality",
        "entry_parent_distance_ticks_max": "Causality & Locality",
        "entry_near_parent": "Causality & Locality",
        # Entry Timing
        "entry_family": "Entry Timing",
        "entry_families": "Entry Timing",
        "label_family": "Entry Timing",
        "qualification_mode": "Entry Timing",
        "resolver_policy": "Entry Timing",
        "legacy_candidate_row_limit": "Entry Timing",
        # Session Policy
        "session_scheme": "Session Policy",
        "doc_sessions": "Session Policy",
        "enabled_entry_sessions": "Session Policy",
        "outside_session_policy": "Session Policy",
        "enable_longs": "Session Policy",
        "enable_shorts": "Session Policy",
        # Risk Admissibility
        "sl_buffer_ticks": "Risk Admissibility",
        "tp_r_multiple": "Risk Admissibility",
        "max_executed_trades_per_day": "Risk Admissibility",
        "break_even_enabled": "Risk Admissibility",
        "runnable": "Risk Admissibility",
        "execution_enabled": "Risk Admissibility",
        "non_runnable_reason": "Risk Admissibility",
    }
)

#: FUX §10.3 visible classification labels (five, exact). The seven registry
#: classifications project onto them; the technical classification stays
#: visible in the card's audit fields.
CLASSIFICATION_LABELS: Mapping[AxisClassification, str] = MappingProxyType(
    {
        AxisClassification.LOCKED_INVARIANT: "Locked Invariant",
        AxisClassification.THESIS_DEFINING: "Locked Invariant",
        AxisClassification.APPROVED_SEARCH_AXIS: "Search Axis",
        AxisClassification.RISK_POLICY_AXIS: "Search Axis",
        AxisClassification.MEASUREMENT_ONLY: "Measured Only",
        AxisClassification.BLOCKED: "Blocked",
        AxisClassification.EXPERIMENTAL: "Experimental",
    }
)

#: Session/direction axes: changing one invokes the FUX §10.5 interpretation
#: selector, and only `Sequential Strategy Profile` creates an executable
#: counterfactual child.
SESSION_DIRECTION_AXES = frozenset(
    {
        "session_scheme",
        "doc_sessions",
        "enabled_entry_sessions",
        "outside_session_policy",
        "enable_longs",
        "enable_shorts",
    }
)

#: The FUX §10.5 interpretation options (visible labels).
INTERPRETATIONS: tuple[str, ...] = (
    "Descriptive Slice",
    "Specialized Model",
    "Sequential Strategy Profile",
)

#: Classifications that render an input widget (everything else renders the
#: value + reason with NO widget — FUX §10.3).
_WIDGET_CLASSIFICATIONS = frozenset(
    {
        AxisClassification.APPROVED_SEARCH_AXIS,
        AxisClassification.RISK_POLICY_AXIS,
        AxisClassification.EXPERIMENTAL,
    }
)


def axis_group_for(technical_key: str) -> str:
    """The market-meaning group for one axis (deterministic 'Other' fallback)."""

    return AXIS_GROUP_ASSIGNMENTS.get(technical_key, "Other")


def grouped_axis_keys(
    axes: Mapping[str, SearchAxisSpec] = SEARCH_AXIS_REGISTRY_V1,
) -> Mapping[str, tuple[str, ...]]:
    """Ordered group → axis technical keys (registry order inside a group)."""

    grouped: dict[str, list[str]] = {group: [] for group in AXIS_GROUP_ORDER}
    for key in axes:
        grouped.setdefault(axis_group_for(key), []).append(key)
    return {
        group: tuple(keys) for group, keys in grouped.items() if keys
    }


def classification_label(classification: AxisClassification) -> str:
    return CLASSIFICATION_LABELS[classification]


def axis_renders_widget(spec: SearchAxisSpec) -> bool:
    """Whether an input widget may exist for this axis (FUX §10.3)."""

    return spec.classification in _WIDGET_CLASSIFICATIONS


def computation_path_chip(spec: SearchAxisSpec) -> str:
    """The FUX §10.4 computation-path chip for one axis card."""

    if spec.classification is AxisClassification.RISK_POLICY_AXIS:
        return "Prop Resimulation"
    if spec.requires_full_sequential_replay:
        return "Requires New Sequential Replay"
    return "Analysis Filter"


# ─────────────────────────────────────────────────────────────────────────────
# Wizard registries: steps, modes, questions, objective templates (FUX §§6-8)
# ─────────────────────────────────────────────────────────────────────────────

WIZARD_STEP_TITLES: tuple[str, ...] = (
    "Objective",
    "Baseline",
    "Strategy Search Space",
    "Prop Contracts",
    "Risk Policies",
    "Benchmarks",
    "Validation",
    "Review & Launch",
)


@dataclass(frozen=True)
class StudyModeSpec:
    mode_id: str
    label: str
    description: str
    #: The backend SearchMode value, or None for the Full Pipeline Run mode
    #: (its operator workflow surface lands with R5; the mode itself renders
    #: and validates now, and its launch stays capability-blocked until R5).
    search_mode: str | None
    requires_prop_selection: bool


STUDY_MODES: tuple[StudyModeSpec, ...] = (
    StudyModeSpec(
        mode_id="single_configuration",
        label="Single Configuration",
        description="Evaluate one exact frozen strategy profile.",
        search_mode="single_configuration",
        requires_prop_selection=False,
    ),
    StudyModeSpec(
        mode_id="fsm_config_search",
        label="FSM Configuration Search",
        description=(
            "Search a small, registered, owner-authorized set of strategy "
            "configurations; every occupancy/order-changing child uses a "
            "full sequential replay or verified reuse."
        ),
        search_mode="fsm_config_search",
        requires_prop_selection=False,
    ),
    StudyModeSpec(
        mode_id="prop_benchmark",
        label="Prop Benchmark",
        description=(
            "Apply selected prop contracts, account/risk/withdrawal/"
            "replacement policies, and simulation protocols to one frozen "
            "executed-strategy stream."
        ),
        search_mode="prop_benchmark",
        requires_prop_selection=True,
    ),
    StudyModeSpec(
        mode_id="universal_prop_search",
        label="Universal Prop Search",
        description=(
            "Evaluate one strategy configuration across multiple firms with "
            "per-firm account policy sets while preserving one universal "
            "strategy profile."
        ),
        search_mode="universal_prop_search",
        requires_prop_selection=True,
    ),
    StudyModeSpec(
        mode_id="full_pipeline_run",
        label="Full Pipeline Run",
        description=(
            "Configure the standardized pipeline under Verification Fixture "
            "(maximum five authorized real trading days) or Full Authorized "
            "Development Data (explicit post-acceptance operator action). "
            "The Configure/Preview/Launch/Monitor workflow lands with R5; "
            "until then the launch stays capability-blocked with this "
            "reason."
        ),
        search_mode=None,
        requires_prop_selection=False,
    ),
)

_MODES_BY_ID = {mode.mode_id: mode for mode in STUDY_MODES}

RESEARCH_QUESTIONS: Mapping[str, str] = MappingProxyType(
    {
        "compare_one_with_baseline": (
            "Compare one configuration with the baseline"
        ),
        "find_robust_fsm": "Find a robust FSM configuration",
        "repeat_payout_feasibility": "Test repeat-payout feasibility",
        "one_config_across_firms": (
            "Find one strategy configuration across multiple firms"
        ),
    }
)

#: Engineering-default question↔mode compatibility (each search mode answers
#: its natural question; the pipeline mode standardizes any of them).
MODE_QUESTION_COMPATIBILITY: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "single_configuration": ("compare_one_with_baseline",),
        "fsm_config_search": ("find_robust_fsm",),
        "prop_benchmark": ("repeat_payout_feasibility",),
        "universal_prop_search": ("one_config_across_firms",),
        "full_pipeline_run": tuple(RESEARCH_QUESTIONS),
    }
)


@dataclass(frozen=True)
class ObjectiveTemplate:
    template_id: str
    label: str
    primary_objective: tuple[str, ...]
    #: Human-readable hard constraints (the resolved numeric thresholds come
    #: from the benchmark step's gate classes and are displayed there too —
    #: never hidden behind the template name, FUX §8.2).
    hard_constraints: tuple[str, ...]
    tie_breaks: tuple[str, ...]
    #: proposal status — nothing here is owner-ratified (OWNER_DECISIONS 7/8).
    status: Literal["proposed_protocol_default", "custom"] = (
        "proposed_protocol_default"
    )
    prop_bearing: bool = True


OBJECTIVE_TEMPLATES: tuple[ObjectiveTemplate, ...] = (
    ObjectiveTemplate(
        template_id="payout_reliability",
        label="Payout Reliability",
        primary_objective=("payout_probability_per_rolling_30d",),
        hard_constraints=(
            "strategy feasibility gates",
            "prop feasibility gates",
        ),
        tie_breaks=("expected_net_payout_90d", "p10_net_payout_90d"),
    ),
    ObjectiveTemplate(
        template_id="maximum_expected_payout",
        label="Maximum Expected Payout",
        primary_objective=("expected_net_payout_90d",),
        hard_constraints=(
            "strategy feasibility gates",
            "prop feasibility gates",
        ),
        tie_breaks=("payout_probability_per_rolling_30d", "p10_net_payout_90d"),
    ),
    ObjectiveTemplate(
        template_id="low_breach_long_life",
        label="Low Breach / Long Account Life",
        primary_objective=("breach_probability_90d",),
        hard_constraints=(
            "strategy feasibility gates",
            "prop feasibility gates",
        ),
        tie_breaks=(
            "payout_probability_per_rolling_30d",
            "expected_net_payout_90d",
        ),
    ),
    ObjectiveTemplate(
        template_id="balanced_prop_performance",
        label="Balanced Prop Performance",
        primary_objective=(
            "expected_net_payout_90d",
            "payout_probability_per_rolling_30d",
            "breach_probability_90d",
        ),
        hard_constraints=(
            "strategy feasibility gates",
            "prop feasibility gates",
        ),
        tie_breaks=("p10_net_payout_90d",),
    ),
    ObjectiveTemplate(
        template_id="strategy_quality_only",
        label="Strategy Quality Only",
        primary_objective=("net_expectancy_r",),
        hard_constraints=("strategy feasibility gates",),
        tie_breaks=("profit_factor", "max_drawdown_r"),
        prop_bearing=False,
    ),
    ObjectiveTemplate(
        template_id="custom",
        label="Custom",
        primary_objective=(),
        hard_constraints=("strategy feasibility gates",),
        tie_breaks=(),
        status="custom",
    ),
)

_TEMPLATES_BY_ID = {template.template_id: template for template in OBJECTIVE_TEMPLATES}


def mode_compatibility_error(
    mode_id: str, question_id: str, template_id: str
) -> str | None:
    """FUX §8.3 — an explanation when the combination is incompatible."""

    mode = _MODES_BY_ID.get(mode_id)
    if mode is None:
        return f"unknown study mode {mode_id!r}"
    if question_id not in RESEARCH_QUESTIONS:
        return f"unknown research question {question_id!r}"
    template = _TEMPLATES_BY_ID.get(template_id)
    if template is None:
        return f"unknown objective template {template_id!r}"
    allowed = MODE_QUESTION_COMPATIBILITY[mode_id]
    if question_id not in allowed:
        allowed_labels = " / ".join(RESEARCH_QUESTIONS[q] for q in allowed)
        return (
            f"'{RESEARCH_QUESTIONS[question_id]}' is answered by a different "
            f"study mode; '{mode.label}' answers: {allowed_labels}."
        )
    if mode.requires_prop_selection and not template.prop_bearing:
        return (
            f"'{template.label}' contains no prop objective, but "
            f"'{mode.label}' evaluates prop realization; choose a "
            "prop-bearing template."
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Wizard step validators (pure; sanitized, field-adjacent messages)
# ─────────────────────────────────────────────────────────────────────────────


def validate_objective_step(fields: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    mode_id = fields.get("mode_id") or ""
    question_id = fields.get("question_id") or ""
    template_id = fields.get("template_id") or ""
    if mode_id not in _MODES_BY_ID:
        errors["mode_id"] = "choose one of the five study modes"
    if question_id not in RESEARCH_QUESTIONS:
        errors["question_id"] = "choose one of the four research questions"
    if template_id not in _TEMPLATES_BY_ID:
        errors["template_id"] = "choose one of the six objective templates"
    if not errors:
        incompatibility = mode_compatibility_error(mode_id, question_id, template_id)
        if incompatibility:
            errors["template_id"] = incompatibility
    if template_id == "custom":
        custom = tuple(fields.get("custom_objectives") or ())
        if not custom:
            errors["custom_objectives"] = (
                "a Custom objective needs at least one registered metric"
            )
        else:
            unknown = [name for name in custom if name not in OBJECTIVE_DIRECTIONS]
            if unknown:
                errors["custom_objectives"] = (
                    f"unregistered objective metric(s): {sorted(unknown)}"
                )
    return errors


def validate_baseline_step(fields: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    profile = fields.get("baseline_profile_name") or ""
    if not profile:
        errors["baseline_profile_name"] = "select a capability-eligible baseline"
    elif fields.get("baseline_blocked_reason"):
        errors["baseline_profile_name"] = (
            "this baseline is not runnable: "
            + str(fields["baseline_blocked_reason"])
        )
    return errors


def validate_search_space_step(
    fields: Mapping[str, Any],
    *,
    axes: Mapping[str, SearchAxisSpec] = SEARCH_AXIS_REGISTRY_V1,
    max_child_count: int = MAX_CHILD_COUNT_CEILING,
) -> dict[str, str]:
    """Validates {axis_key: (value_id, ...)} selections against the registry."""

    errors: dict[str, str] = {}
    selections: Mapping[str, Sequence[str]] = fields.get("axis_selections") or {}
    mode_id = fields.get("mode_id") or ""
    for key, value_ids in selections.items():
        spec = axes.get(key)
        if spec is None:
            errors[key] = "unregistered axis"
            continue
        if not axis_renders_widget(spec):
            errors[key] = (
                f"axis is {classification_label(spec.classification)}; it "
                "cannot carry search values"
            )
            continue
        registered = set(spec.registered_values)
        unknown = [value for value in value_ids if value not in registered]
        if unknown:
            errors[key] = f"unregistered value id(s): {sorted(unknown)}"
    if mode_id == "single_configuration":
        multi = [
            key for key, values in selections.items() if len(tuple(values)) > 1
        ]
        if multi:
            errors["axis_selections"] = (
                "Single Configuration evaluates one exact profile; axes "
                f"with multiple values: {sorted(multi)}"
            )
    # FUX §10.5: only `Sequential Strategy Profile` creates an executable
    # counterfactual child. A descriptive/specialized interpretation over
    # changed session/direction axes is a cohort study — it never
    # enumerates replay children, so the charter path refuses it here
    # (the CohortSpec descriptive flow is a study-lane surface).
    session_changed = sorted(set(selections) & SESSION_DIRECTION_AXES)
    interpretation = fields.get("interpretation")
    if session_changed and interpretation not in (None, INTERPRETATIONS[2]):
        errors["interpretation"] = (
            f"'{interpretation}' is a descriptive interpretation and never "
            "replays; to enumerate executable counterfactual children for "
            f"{session_changed}, choose 'Sequential Strategy Profile' — or "
            "clear the session/direction selections"
        )
    if not errors:
        count = enumerate_child_count(selections)
        if count > max_child_count:
            errors["axis_selections"] = (
                f"{count} children exceed the ceiling of {max_child_count}; "
                "narrow the registered values"
            )
    return errors


def validate_prop_step(fields: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    mode = _MODES_BY_ID.get(fields.get("mode_id") or "")
    selected = tuple(fields.get("selected_contract_ids") or ())
    launchable = set(fields.get("launchable_contract_ids") or ())
    if mode is not None and mode.requires_prop_selection and not selected:
        errors["selected_contract_ids"] = (
            f"{mode.label} needs at least one selectable firm contract"
        )
    blocked = [contract for contract in selected if contract not in launchable]
    if blocked:
        errors["selected_contract_ids"] = (
            "unverified/blocked contract(s) cannot launch: "
            f"{sorted(blocked)}"
        )
    return errors


RISK_POLICY_TEMPLATES: tuple[str, ...] = (
    "Fixed Dollar",
    "Percent of Starting Buffer",
    "Percent of Current Buffer",
    "NQ/MNQ Adaptive",
    "Custom Contract Count",
)

#: Owner decision 13 — the UI caps accounts per firm at 2.
MAX_ACCOUNTS_PER_FIRM = 2


def validate_risk_step(fields: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    selected_contracts = tuple(fields.get("selected_contract_ids") or ())
    per_firm = fields.get("per_firm_policies") or {}
    for contract_id in selected_contracts:
        policy = per_firm.get(contract_id)
        if policy is None:
            errors[contract_id] = "assign an account/risk/withdrawal policy set"
            continue
        template = policy.get("risk_template")
        if template not in RISK_POLICY_TEMPLATES:
            errors[contract_id] = "choose a registered risk template"
        n_accounts = int(policy.get("n_accounts") or 1)
        if not 1 <= n_accounts <= MAX_ACCOUNTS_PER_FIRM:
            errors[contract_id] = (
                f"accounts per firm must be 1..{MAX_ACCOUNTS_PER_FIRM} "
                "(owner decision 13)"
            )
    return errors


def validate_benchmarks_step(fields: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for group in ("strategy_gates", "prop_gates", "robustness_gates"):
        thresholds = fields.get(group) or {}
        for name, value in thresholds.items():
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                errors[f"{group}.{name}"] = "threshold must be numeric or empty"
            elif name.startswith(("minimum_first_payout", "maximum_breach")) and not (
                0.0 <= float(value) <= 1.0
            ):
                errors[f"{group}.{name}"] = "probability thresholds live in [0, 1]"
    return errors


def validate_validation_step(fields: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    run_scope = fields.get("run_scope")
    if run_scope not in ("verification_5d", "full_authorized_development"):
        errors["run_scope"] = (
            "choose verification_5d or full_authorized_development"
        )
    real_dates = tuple(fields.get("real_dates") or ())
    warmup_dates = tuple(fields.get("warmup_dates") or ())
    if run_scope == "verification_5d":
        if len(real_dates) > 5:
            errors["real_dates"] = "verification scope allows at most 5 real days"
        if len(real_dates) + len(warmup_dates) > 5:
            errors["warmup_dates"] = (
                "warmup + evidence must stay within the 5-day budget"
            )
    seed = fields.get("seed")
    if seed is None or not isinstance(seed, int):
        errors["seed"] = "a deterministic integer seed is required"
    workers = fields.get("worker_limit")
    if workers is not None and not 1 <= int(workers) <= 4:
        errors["worker_limit"] = "worker limit is bounded to 1..4"
    return errors


def validate_review_step(fields: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    if fields.get("run_scope") == "full_authorized_development":
        typed = (fields.get("typed_acknowledgement") or "").strip().lower()
        expected = (fields.get("required_acknowledgement") or "").strip().lower()
        if not expected or typed != expected:
            errors["typed_acknowledgement"] = (
                "type the exact acknowledgement phrase to enable the "
                "full-scope launch control"
            )
    return errors


def enumerate_child_count(
    axis_selections: Mapping[str, Sequence[str]],
    *,
    axes: Mapping[str, SearchAxisSpec] = SEARCH_AXIS_REGISTRY_V1,
) -> int:
    """Launch-time cartesian child count (pure).

    Charter assembly merges each axis's BASELINE value id with the selected
    challengers (the baseline child must enumerate for comparison), so the
    preview counts the union — never fewer than the launch enumeration.
    Unregistered axes (synthetic tests) count their selected values only.
    """

    sizes: list[int] = []
    for axis_key, values in axis_selections.items():
        selected = set(tuple(values))
        if not selected:
            continue
        spec = axes.get(axis_key)
        if spec is not None:
            selected.add(spec.baseline_value_id)
        sizes.append(len(selected))
    return prod(sizes) if sizes else 1


# ─────────────────────────────────────────────────────────────────────────────
# Active Runs derivations (FUX §16) — from search_state.json children rows
# ─────────────────────────────────────────────────────────────────────────────

#: FailureReason values assigned by the replay stage (state == "failed").
REPLAY_STAGE_REASONS = frozenset({FailureReason.REPLAY, FailureReason.INVARIANT})

#: FailureReason values the strategy gate assigns (gates.py reason_by_gate).
STRATEGY_STAGE_REASONS = frozenset(
    {
        FailureReason.INSUFFICIENT_TRADES,
        FailureReason.INSUFFICIENT_DAYS,
        FailureReason.NEGATIVE_EXPECTANCY,
        FailureReason.DRAWDOWN,
        FailureReason.KNIFE_EDGE,
    }
)

#: FailureReason values the prop gate assigns (gates.py reason_by_gate).
PROP_STAGE_REASONS = frozenset(
    {
        FailureReason.FUNDED_SURVIVAL,
        FailureReason.BREACH,
        FailureReason.FEES,
        FailureReason.ONE_FIRM,
        FailureReason.STRESS,
        FailureReason.UNVERIFIED_CONTRACT,
    }
)

#: Exact orchestrator explanation sentinels (contract-tested against
#: ``search/orchestrator.py`` — the state file carries no per-stage booleans,
#: so these strings are the only signal for two corner states).
REUSED_UNEVALUATED_SENTINEL = "reused replay has no published costed evaluation"
PROP_SIM_FAILED_SENTINEL = "prop simulation failed:"
FRONTIER_EXCLUSION_SENTINELS: tuple[str, ...] = (
    "excluded from the frontier",
    "excluded from the prop-feasible set",
)

FUNNEL_STAGE_LABELS: tuple[str, ...] = (
    "Generated",
    "Replay Valid",
    "Strategy Pass",
    "Prop Feasible",
    "Robust",
)


@dataclass(frozen=True)
class FunnelCounts:
    generated: int
    replay_valid: int
    strategy_pass: int
    prop_feasible: int | None
    robust: int | None

    def as_button_labels(self) -> tuple[str, ...]:
        """FUX §16.3 — the five keyboard-operable funnel button labels."""

        values: tuple[int | None, ...] = (
            self.generated,
            self.replay_valid,
            self.strategy_pass,
            self.prop_feasible,
            self.robust,
        )
        return tuple(
            f"{label} · {value if value is not None else '–'}"
            for label, value in zip(FUNNEL_STAGE_LABELS, values, strict=True)
        )


def _reason(child: Mapping[str, Any]) -> FailureReason | None:
    raw = child.get("failure_reason")
    if raw in (None, ""):
        return None
    return FailureReason(raw)


def _replay_valid(child: Mapping[str, Any]) -> bool:
    return child.get("state") in ("completed", "reused")


def _strategy_pass(child: Mapping[str, Any]) -> bool:
    if not _replay_valid(child):
        return False
    reason = _reason(child)
    if reason in STRATEGY_STAGE_REASONS:
        return False
    explanation = str(child.get("explanation") or "")
    # a reused child whose evaluation was never published is annotated,
    # never silently gate-free (R2 scoping note)
    return REUSED_UNEVALUATED_SENTINEL not in explanation


def _prop_feasible_from_state(child: Mapping[str, Any]) -> bool:
    if not _strategy_pass(child):
        return False
    reason = _reason(child)
    if reason in PROP_STAGE_REASONS:
        return False
    explanation = str(child.get("explanation") or "")
    if explanation.startswith(PROP_SIM_FAILED_SENTINEL):
        return False
    if any(sentinel in explanation for sentinel in FRONTIER_EXCLUSION_SENTINELS):
        return False
    # a prop-sim crash keeps state "completed" but stamps reason REPLAY
    return reason is not FailureReason.REPLAY


def funnel_counts(
    children: Sequence[Mapping[str, Any]],
    *,
    feasible_ids: Sequence[str] | None = None,
    frontier_ids: Sequence[str] | None = None,
) -> FunnelCounts:
    """The five FUX §16.2 counts.

    ``feasible_ids``/``frontier_ids`` come from the persisted frontier
    envelope and are authoritative when present; before the frontier phase
    the prop/robust counts derive from the state rows (prop) or stay
    unknown (robust).
    """

    generated = len(children)
    replay_valid = sum(1 for child in children if _replay_valid(child))
    strategy_pass = sum(1 for child in children if _strategy_pass(child))
    if feasible_ids is not None:
        prop_feasible: int | None = len(tuple(feasible_ids))
    else:
        prop_feasible = sum(
            1 for child in children if _prop_feasible_from_state(child)
        )
    robust = len(tuple(frontier_ids)) if frontier_ids is not None else None
    return FunnelCounts(
        generated=generated,
        replay_valid=replay_valid,
        strategy_pass=strategy_pass,
        prop_feasible=prop_feasible,
        robust=robust,
    )


def children_for_stage(
    children: Sequence[Mapping[str, Any]],
    stage: str,
    *,
    feasible_ids: Sequence[str] = (),
    frontier_ids: Sequence[str] = (),
) -> tuple[Mapping[str, Any], ...]:
    """Filter the child table by a clicked funnel stage (FUX §16.3)."""

    if stage == "Generated":
        return tuple(children)
    if stage == "Replay Valid":
        return tuple(child for child in children if _replay_valid(child))
    if stage == "Strategy Pass":
        return tuple(child for child in children if _strategy_pass(child))
    if stage == "Prop Feasible":
        feasible = set(feasible_ids)
        if feasible:
            return tuple(
                child
                for child in children
                if child.get("core_replay_id") in feasible
            )
        return tuple(
            child for child in children if _prop_feasible_from_state(child)
        )
    if stage == "Robust":
        frontier = set(frontier_ids)
        return tuple(
            child for child in children if child.get("core_replay_id") in frontier
        )
    raise ValueError(f"unknown funnel stage {stage!r}")


ACTIVE_RUNS_COLUMNS: tuple[str, ...] = (
    "Config",
    "Replay",
    "Strategy Gate",
    "Prop Simulation",
    "Robustness",
    "Status",
    "Human Explanation",
)


@dataclass(frozen=True)
class ChildRowPresentation:
    config: str
    replay: str
    strategy_gate: str
    prop_simulation: str
    robustness: str
    status_key: StudyStatusKey
    human_explanation: str
    core_replay_id: str = ""
    ordinal: int = 0
    comparison_role: str = ""
    axis_value_ids: Mapping[str, str] = field(default_factory=dict)

    def as_row(self) -> dict[str, str]:
        from .study_status import status_presentation  # local: avoid cycle

        presentation = status_presentation(self.status_key)
        return {
            "Config": self.config,
            "Replay": self.replay,
            "Strategy Gate": self.strategy_gate,
            "Prop Simulation": self.prop_simulation,
            "Robustness": self.robustness,
            "Status": f"{presentation.glyph} {presentation.visible_label}",
            "Human Explanation": self.human_explanation,
        }


_PENDING = "Pending"
_PASSED = "✓ Passed"
_FAILED = "✕ Failed"
_NOT_RUN_REPLAY = "Not run — replay failed"
_NOT_RUN_BLOCKED = "Not run — blocked before replay"
_ROBUSTNESS_SURFACE = "Evaluated on the comparison surfaces"


def child_row_presentation(
    child: Mapping[str, Any],
    *,
    feasible_ids: Sequence[str] = (),
    frontier_ids: Sequence[str] = (),
    representative_id: str | None = None,
    prop_wired: bool = True,
    config_name: str | None = None,
) -> ChildRowPresentation:
    """Stage cells + status for one search_state.json child row (FUX §16.4).

    Skipped downstream stages always name the earlier gate that stopped
    them; a prop-simulation crash is distinguished from a replay failure by
    the orchestrator's exact explanation sentinel.
    """

    state = child.get("state")
    reason = _reason(child)
    explanation = str(child.get("explanation") or "")
    core_replay_id = str(child.get("core_replay_id") or "")
    name = config_name or (core_replay_id[:12] + "…" if core_replay_id else "?")
    base = {
        "config": name,
        "core_replay_id": core_replay_id,
        "ordinal": int(child.get("ordinal") or 0),
        "comparison_role": str(child.get("comparison_role") or ""),
        "axis_value_ids": dict(child.get("axis_value_ids") or {}),
        "human_explanation": explanation,
    }
    if state == "blocked":
        return ChildRowPresentation(
            replay=_NOT_RUN_BLOCKED,
            strategy_gate=_NOT_RUN_BLOCKED,
            prop_simulation=_NOT_RUN_BLOCKED,
            robustness=_NOT_RUN_BLOCKED,
            status_key=StudyStatusKey.BLOCKED,
            **base,
        )
    if state == "queued":
        return ChildRowPresentation(
            replay=_PENDING,
            strategy_gate=_PENDING,
            prop_simulation=_PENDING,
            robustness=_PENDING,
            status_key=StudyStatusKey.QUEUED,
            **base,
        )
    if state == "running":
        return ChildRowPresentation(
            replay="Running",
            strategy_gate=_PENDING,
            prop_simulation=_PENDING,
            robustness=_PENDING,
            status_key=StudyStatusKey.RUNNING,
            **base,
        )
    if state == "cancelled_at_safe_boundary":
        return ChildRowPresentation(
            replay="Cancelled at safe boundary",
            strategy_gate=_NOT_RUN_REPLAY,
            prop_simulation=_NOT_RUN_REPLAY,
            robustness=_NOT_RUN_REPLAY,
            status_key=StudyStatusKey.BLOCKED,
            **base,
        )
    if state == "failed":
        return ChildRowPresentation(
            replay=_FAILED,
            strategy_gate=_NOT_RUN_REPLAY,
            prop_simulation=_NOT_RUN_REPLAY,
            robustness=_NOT_RUN_REPLAY,
            status_key=StudyStatusKey.REPLAY_FAILED,
            **base,
        )
    # completed / reused
    replay_cell = "✓ Reused (verified)" if state == "reused" else _PASSED
    if REUSED_UNEVALUATED_SENTINEL in explanation:
        return ChildRowPresentation(
            replay=replay_cell,
            strategy_gate="Not evaluated this run",
            prop_simulation="Not run — gates not evaluated",
            robustness="Not run — gates not evaluated",
            status_key=StudyStatusKey.BLOCKED,
            **base,
        )
    if reason in STRATEGY_STAGE_REASONS:
        return ChildRowPresentation(
            replay=replay_cell,
            strategy_gate=_FAILED,
            prop_simulation=NOT_RUN_STRATEGY_GATE_TEXT,
            robustness=NOT_RUN_STRATEGY_GATE_TEXT,
            status_key=StudyStatusKey.STRATEGY_REJECTED,
            **base,
        )
    if reason in PROP_STAGE_REASONS:
        return ChildRowPresentation(
            replay=replay_cell,
            strategy_gate=_PASSED,
            prop_simulation=_FAILED,
            robustness="Not run — prop gate failed",
            status_key=StudyStatusKey.PROP_REJECTED,
            **base,
        )
    if reason is FailureReason.REPLAY and explanation.startswith(
        PROP_SIM_FAILED_SENTINEL
    ):
        return ChildRowPresentation(
            replay=replay_cell,
            strategy_gate=_PASSED,
            prop_simulation="✕ Simulation failed",
            robustness="Not run — prop simulation failed",
            status_key=StudyStatusKey.PROP_REJECTED,
            **base,
        )
    if any(sentinel in explanation for sentinel in FRONTIER_EXCLUSION_SENTINELS):
        return ChildRowPresentation(
            replay=replay_cell,
            strategy_gate=_PASSED,
            prop_simulation="Excluded (objective unavailable)",
            robustness=_ROBUSTNESS_SURFACE,
            status_key=StudyStatusKey.BLOCKED,
            **base,
        )
    prop_cell = _PASSED if prop_wired else "Not run — no prop simulator wired"
    status = StudyStatusKey.ROBUST_FINALIST
    if representative_id and core_replay_id == representative_id:
        status = StudyStatusKey.SELECTED_REPRESENTATIVE
    return ChildRowPresentation(
        replay=replay_cell,
        strategy_gate=_PASSED,
        prop_simulation=prop_cell,
        robustness=_ROBUSTNESS_SURFACE,
        status_key=status,
        **base,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Human configuration names from the baseline diff (FUX §24)
# ─────────────────────────────────────────────────────────────────────────────


def changed_axis_labels(
    axis_value_ids: Mapping[str, str],
    *,
    value_registry: Mapping[str, Any] = AXIS_VALUE_REGISTRY_V1,
    axis_specs: Mapping[str, SearchAxisSpec] = SEARCH_AXIS_REGISTRY_V1,
) -> tuple[str, ...]:
    """Human labels of every value that differs from the axis baseline."""

    labels: list[str] = []
    for axis_key in sorted(axis_value_ids):
        value_id = axis_value_ids[axis_key]
        spec = axis_specs.get(axis_key)
        if spec is not None and value_id == spec.baseline_value_id:
            continue
        value = value_registry.get(value_id)
        labels.append(value.human_label if value is not None else value_id)
    return tuple(labels)


def human_config_name(
    axis_value_ids: Mapping[str, str],
    *,
    comparison_role: str = "",
    value_registry: Mapping[str, Any] = AXIS_VALUE_REGISTRY_V1,
    axis_specs: Mapping[str, SearchAxisSpec] = SEARCH_AXIS_REGISTRY_V1,
) -> str:
    """Baseline-diff display name (e.g. ``60-Bar Parent Timeout``).

    Display names are catalog annotations; they never replace technical
    identity (FUX §5.5).
    """

    changed = changed_axis_labels(
        axis_value_ids, value_registry=value_registry, axis_specs=axis_specs
    )
    if not changed:
        return "Baseline (doc-default)"
    return " + ".join(changed)


# ─────────────────────────────────────────────────────────────────────────────
# Explorer presets, metric pickers, pagination (FUX §§20-24, 33)
# ─────────────────────────────────────────────────────────────────────────────

EXPLORER_PRESETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "Strategy": (
            "trade count",
            "net E[R]",
            "realized payoff ratio",
            "profit factor",
            "max DD R",
            "trade frequency",
            "setup occupancy",
        ),
        "Prop": (
            "firm",
            "risk policy",
            "first payout",
            "3 payouts",
            "90-day survival",
            "expected payout",
            "Q10 payout",
            "fees",
            "replacement cost",
        ),
        "Robustness": (
            "outer folds passed",
            "stress tests passed",
            "neighbor stability",
            "worst-firm result",
            "concentration warning",
        ),
    }
)

EXPLORER_STICKY_COLUMNS: tuple[str, ...] = (
    "rank",
    "config name",
    "status",
    "changed parameters",
)

#: FUX §20 metric families → StrategyMetrics / PayoutReliabilityVector attrs.
HEATMAP_METRICS: Mapping[str, str] = MappingProxyType(
    {
        "net E[R]": "net_expectancy_r",
        "trade count": "executed_trades",
        "maximum drawdown": "max_drawdown_r",
        "expected payout": "expected_net_payout_90d",
        "breach probability": "breach_probability_90d",
        "payout reliability": "payout_probability_per_rolling_30d",
    }
)

#: FUX §21 firm-matrix metric toggle → PayoutReliabilityVector attrs.
FIRM_MATRIX_METRICS: Mapping[str, str] = MappingProxyType(
    {
        "P(3 payouts before breach)": "three_payout_probability",
        "expected payout": "expected_net_payout_90d",
        "breach probability": "breach_probability_90d",
        "first-payout probability": "first_payout_probability_60d",
    }
)

PAYOUT_HORIZONS: tuple[str, ...] = ("30-day", "60-day", "90-day", "lifetime")


def heatmap_cell_class(
    *,
    value: float | None,
    sample_count: int,
    blocked: bool = False,
    failed: bool = False,
    knife_edge: bool = False,
    min_sample: int = 1,
) -> str:
    """FUX §20 glyph class for one sensitivity-heatmap cell."""

    if blocked:
        return "blocked_cell"
    if failed:
        return "failed_region"
    if value is None or sample_count < min_sample:
        return "insufficient_data"
    if knife_edge:
        return "knife_edge_point"
    return "stable_plateau"


def page_slice(total: int, page: int, page_size: int) -> tuple[int, int, int]:
    """(start, end, n_pages) for indexed pagination; page clamps into range."""

    if page_size <= 0:
        raise ValueError("page_size must be positive")
    n_pages = max(1, -(-total // page_size))
    page = min(max(page, 0), n_pages - 1)
    start = page * page_size
    return start, min(start + page_size, total), n_pages


# ─────────────────────────────────────────────────────────────────────────────
# Work estimates (operational annotations — never scientific identity)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorkEstimate:
    unique_profiles: int
    full_replay_count: int
    verified_reuse_hits: int
    firm_policy_combinations: int
    historical_prop_replays: int
    bootstrap_and_stress_simulations: int
    estimated_runtime_seconds: float
    estimated_storage_mb: float


def estimate_search_work(
    *,
    n_children: int,
    n_replay_days: int,
    reuse_hits: int = 0,
    n_firm_policy_combinations: int = 0,
    bootstrap_paths: int = 10_000,
    n_stress_scenarios: int = 0,
    replay_seconds_per_day: float = 15.0,
    storage_mb_per_child: float = 40.0,
) -> WorkEstimate:
    """Replay-vs-prop work preview (FUX §15). Estimates only.

    Basis: TEST_MATRIX §4 (~15 s/day sequential replay; prop sims are
    sub-second walks; bootstrap ≈ 60 s per 10k paths per firm×policy).
    """

    replays = max(0, n_children - reuse_hits)
    replay_seconds = replays * n_replay_days * replay_seconds_per_day
    prop_replays = n_children * n_firm_policy_combinations
    bootstrap_sims = prop_replays * (1 + n_stress_scenarios)
    bootstrap_seconds = (
        prop_replays * (bootstrap_paths / 10_000.0) * 60.0 * (1 + n_stress_scenarios)
    )
    return WorkEstimate(
        unique_profiles=n_children,
        full_replay_count=replays,
        verified_reuse_hits=reuse_hits,
        firm_policy_combinations=n_firm_policy_combinations,
        historical_prop_replays=prop_replays,
        bootstrap_and_stress_simulations=bootstrap_sims,
        estimated_runtime_seconds=replay_seconds + bootstrap_seconds,
        estimated_storage_mb=n_children * storage_mb_per_child,
    )


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60.0:.1f} h"


def format_storage(megabytes: float) -> str:
    megabytes = max(0.0, float(megabytes))
    if megabytes < 1024:
        return f"{megabytes:.0f} MB"
    return f"{megabytes / 1024.0:.1f} GB"


# ─────────────────────────────────────────────────────────────────────────────
# Responsive decisions (FUX §32.3)
# ─────────────────────────────────────────────────────────────────────────────


def viewport_class(width_px: int | None) -> Literal["desktop", "tablet", "mobile"]:
    """Layout class for a width hint (desktop ≥1024, tablet ≥768, else mobile).

    Streamlit cannot read the viewport server-side; callers pass a hint (the
    QA harness sets it explicitly) and default to desktop when unknown.
    """

    if width_px is None:
        return "desktop"
    if width_px >= 1024:
        return "desktop"
    if width_px >= 768:
        return "tablet"
    return "mobile"
