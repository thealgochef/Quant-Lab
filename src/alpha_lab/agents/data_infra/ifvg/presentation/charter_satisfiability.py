"""Charter satisfiability BEFORE freeze (UI-1; plan §5.4, §8, F-04, owner Q4).

A pure report over a draft's resolved intent. Every rule is named, every
failure explains itself, and a failing report disables Freeze. The service
validator (``search/charter.validate_charter``) remains the authority for
the identity-bearing subset of these rules; this report exists so a
contradiction is refused at the Review step with its reason, never after
eight steps as a generic "Freeze failed".

Rules (owner Q4 and the revision-2 clarifications):

* *Evaluate one configuration* — exactly one resolved profile, zero
  challengers.
* *Compare one configuration with the baseline* — exactly one baseline plus
  exactly ONE challenger configuration; the challenger may carry several
  registered differences (``challenger_configurations``), or, in the
  cartesian axis model, exactly one axis with one challenger value.
* *FSM Configuration Search* — at least one axis with a challenger (≥ 2
  profiles).
* A selected prop objective (or prop tie-break) is NEVER rewritten: it
  requires at least one selected, launchable (verified) firm contract, else
  the path blocks.
* *Prop Benchmark* needs ≥ 1 firm contract; *Universal Prop Search* ≥ 2.
* *Implementation Verification with real evidence* is the exact baseline
  with the verification gates only: no challengers, no prop / robustness
  research gates, no firm contracts.
* A synthetic fixture is confined to Implementation Verification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import prod

from ..search.charter import OBJECTIVE_DIRECTIONS
from ..search.strategy_metrics import StrategyMetrics
from .run_purpose import EvidenceClass, RunPurpose

__all__ = [
    "CharterSatisfiabilityReport",
    "SatisfiabilityRule",
    "StudyGoal",
    "evaluate_charter_satisfiability",
    "goal_for_draft",
    "is_prop_metric",
    "summarize_challenger_differences",
]


class StudyGoal(StrEnum):
    EVALUATE_ONE = "evaluate_one_configuration"
    COMPARE_WITH_BASELINE = "compare_with_baseline"
    FSM_SEARCH = "fsm_config_search"
    PROP_FEASIBILITY = "prop_benchmark"
    UNIVERSAL_PROP = "universal_prop_search"
    FEATURE_MODEL = "feature_model_evidence"
    ADVANCED_END_TO_END = "advanced_end_to_end"
    VERIFICATION = "implementation_verification"


_GOAL_BY_MODE_AND_QUESTION: Mapping[tuple[str, str], StudyGoal] = {
    ("single_configuration", "evaluate_one_configuration"): StudyGoal.EVALUATE_ONE,
    ("single_configuration", "compare_one_with_baseline"): StudyGoal.COMPARE_WITH_BASELINE,
    ("fsm_config_search", "find_robust_fsm"): StudyGoal.FSM_SEARCH,
    ("prop_benchmark", "repeat_payout_feasibility"): StudyGoal.PROP_FEASIBILITY,
    ("universal_prop_search", "one_config_across_firms"): StudyGoal.UNIVERSAL_PROP,
}

_MODE_GOALS: Mapping[str, StudyGoal] = {
    "single_configuration": StudyGoal.COMPARE_WITH_BASELINE,
    "fsm_config_search": StudyGoal.FSM_SEARCH,
    "prop_benchmark": StudyGoal.PROP_FEASIBILITY,
    "universal_prop_search": StudyGoal.UNIVERSAL_PROP,
    "full_pipeline_run": StudyGoal.ADVANCED_END_TO_END,
}


def goal_for_draft(mode_id: str, question_id: str | None) -> StudyGoal:
    """The study goal a mode + research question resolve to."""

    if mode_id not in _MODE_GOALS:
        raise ValueError(f"unknown study mode {mode_id!r}")
    if mode_id == "full_pipeline_run":
        return StudyGoal.ADVANCED_END_TO_END
    return _GOAL_BY_MODE_AND_QUESTION.get((mode_id, str(question_id)), _MODE_GOALS[mode_id])


#: The strategy-owned metric names (pydantic v2 keeps fields in
#: ``model_fields`` — ``hasattr(StrategyMetrics, name)`` is ALWAYS False for a
#: field, which is exactly how the R4 wizard silently rewrote every
#: strategy-only objective to ``net_expectancy_r`` (plan F-04)).
STRATEGY_METRIC_NAMES: frozenset[str] = frozenset(StrategyMetrics.model_fields)


def is_prop_metric(metric: str) -> bool:
    """A registered objective that is NOT a strategy-owned metric."""

    return metric != "core_replay_id" and metric not in STRATEGY_METRIC_NAMES


@dataclass(frozen=True)
class SatisfiabilityRule:
    rule_id: str
    label: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class CharterSatisfiabilityReport:
    goal: StudyGoal
    rules: tuple[SatisfiabilityRule, ...]
    passed: bool
    failures: tuple[SatisfiabilityRule, ...]
    resolved_profile_count: int
    challenger_count: int
    configuration_sentence: str
    pareto_objectives: tuple[str, ...]
    tie_breaks: tuple[str, ...]


def _configuration_sentence(profiles: int, challengers: int) -> str:
    if profiles <= 1:
        return "Baseline only: 1 configuration"
    return f"{profiles} configurations: baseline + {challengers} challenger" + (
        "s" if challengers != 1 else ""
    )


def summarize_challenger_differences(
    baseline_values: Mapping[str, str], challenger: Mapping[str, str]
) -> tuple[str, ...]:
    """One line per registered difference between the baseline and ONE
    challenger configuration (the Review page lists every difference)."""

    lines: list[str] = []
    for axis, value in sorted(challenger.items()):
        baseline = baseline_values.get(axis)
        if baseline == value:
            continue
        lines.append(f"{axis}: {baseline if baseline is not None else '(baseline)'} → {value}")
    return tuple(lines)


def evaluate_charter_satisfiability(
    *,
    goal: StudyGoal | str,
    search_mode: str,
    axis_selections: Mapping[str, Sequence[str]],
    baseline_profile_name: str,
    pareto_objectives: Sequence[str],
    tie_breaks: Sequence[str],
    selected_contract_ids: Sequence[str],
    launchable_contract_ids: Sequence[str],
    purpose: RunPurpose | str,
    evidence_class: EvidenceClass | str,
    prop_gates_configured: bool,
    robustness_gates_configured: bool,
    challenger_configurations: Sequence[Mapping[str, str]] | None = None,
) -> CharterSatisfiabilityReport:
    goal = StudyGoal(goal)
    purpose = RunPurpose(purpose)
    evidence = EvidenceClass(evidence_class)
    objectives = tuple(str(metric) for metric in pareto_objectives)
    ties = tuple(str(metric) for metric in tie_breaks)
    selections = {
        str(axis): tuple(str(value) for value in values)
        for axis, values in (axis_selections or {}).items()
        if tuple(values)
    }
    selected = tuple(str(contract) for contract in selected_contract_ids)
    launchable = set(str(contract) for contract in launchable_contract_ids)
    verified_selected = tuple(contract for contract in selected if contract in launchable)

    if challenger_configurations is not None:
        challengers = len(tuple(challenger_configurations))
        profiles = 1 + challengers
    else:
        # the cartesian axis model: every axis carries its baseline value plus
        # the selected challengers (charter assembly merges the baseline first)
        profiles = prod(len(values) + 1 for values in selections.values()) if selections else 1
        challengers = profiles - 1

    rules: list[SatisfiabilityRule] = []

    def _rule(rule_id: str, label: str, passed: bool, detail: str) -> None:
        rules.append(
            SatisfiabilityRule(rule_id=rule_id, label=label, passed=passed, detail=detail)
        )

    _rule(
        "baseline_present",
        "A capability-eligible baseline anchors the charter",
        bool(str(baseline_profile_name).strip()),
        f"baseline {baseline_profile_name!r}"
        if str(baseline_profile_name).strip()
        else "no baseline profile is selected",
    )

    if goal is StudyGoal.FSM_SEARCH:
        _rule(
            "fsm_search_has_a_challenger",
            "FSM search enumerates at least one challenger (≥ 2 profiles)",
            challengers >= 1,
            _configuration_sentence(profiles, challengers)
            if challengers >= 1
            else "no search axis carries a challenger value: a zero-axis search would freeze "
            "as a one-child study — select at least one registered challenger value",
        )
    if goal is StudyGoal.EVALUATE_ONE:
        _rule(
            "evaluate_requires_exactly_one_profile",
            "Evaluate one configuration resolves exactly one profile",
            profiles == 1 and challengers == 0,
            "one resolved profile, no comparison or delta claim"
            if profiles == 1
            else f"{challengers} challenger(s) selected — Evaluate makes no comparison; "
            "clear the challengers or choose Compare",
        )
    if goal is StudyGoal.COMPARE_WITH_BASELINE:
        _rule(
            "compare_requires_exactly_one_challenger",
            "Compare resolves exactly one baseline and one challenger configuration",
            profiles == 2 and challengers == 1,
            "baseline + 1 challenger configuration (every registered difference is listed)"
            if profiles == 2
            else (
                "no challenger is selected: the baseline would be compared with itself"
                if challengers == 0
                else f"{challengers} challenger configurations resolve — Compare takes exactly "
                "one (a challenger may carry several registered differences as ONE "
                "configuration; use FSM search for several challengers)"
            ),
        )

    prop_metrics = tuple(
        metric for metric in (*objectives, *ties) if metric in OBJECTIVE_DIRECTIONS
        and is_prop_metric(metric)
    )
    _rule(
        "prop_objective_requires_a_verified_contract",
        "A selected prop objective needs at least one verified firm contract",
        not prop_metrics or bool(verified_selected),
        (
            "no prop objective is selected"
            if not prop_metrics
            else f"prop objective(s) {', '.join(prop_metrics)} carried by "
            f"{len(verified_selected)} verified contract(s)"
            if verified_selected
            else f"prop objective(s) {', '.join(prop_metrics)} are selected but no verified "
            "(launchable) firm contract is — the objective is never rewritten; select a "
            "first_party_verified contract or a strategy-only objective"
        ),
    )
    if goal is StudyGoal.PROP_FEASIBILITY or search_mode == "prop_benchmark":
        _rule(
            "prop_benchmark_requires_a_firm_contract",
            "Prop Benchmark needs at least one verified firm contract",
            len(verified_selected) >= 1,
            f"{len(verified_selected)} verified contract(s) selected"
            if verified_selected
            else "no verified firm contract is selected",
        )
    if goal is StudyGoal.UNIVERSAL_PROP or search_mode == "universal_prop_search":
        _rule(
            "universal_requires_two_firm_contracts",
            "Universal Prop Search needs at least two verified firm contracts",
            len(verified_selected) >= 2,
            f"{len(verified_selected)} verified contract(s) selected"
            + ("" if len(verified_selected) >= 2 else " — one firm cannot be universal"),
        )

    real_verification = (
        purpose is RunPurpose.IMPLEMENTATION_VERIFICATION and evidence is EvidenceClass.REAL
    )
    if real_verification:
        _rule(
            "verification_excludes_research_gates",
            "Real verification applies the verification gates only",
            not (prop_gates_configured or robustness_gates_configured or selected),
            "verification_control_flow_gates_v1 only"
            if not (prop_gates_configured or robustness_gates_configured or selected)
            else "prop / robustness research gates or firm contracts entered a verification "
            "charter — the real ≤5-day slice never carries research gates",
        )
        _rule(
            "verification_is_the_exact_baseline",
            "Real verification replays the exact baseline only",
            profiles == 1,
            "exact baseline, one configuration"
            if profiles == 1
            else f"{challengers} challenger(s) selected — the real verification slice runs "
            "the exact baseline only",
        )
    _rule(
        "synthetic_fixture_confined_to_verification",
        "A synthetic fixture is confined to Implementation Verification",
        evidence is EvidenceClass.REAL or purpose is RunPurpose.IMPLEMENTATION_VERIFICATION,
        "real evidence"
        if evidence is EvidenceClass.REAL
        else "synthetic fixture under Implementation Verification"
        if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
        else f"a synthetic fixture cannot carry the {purpose.value} purpose; research "
        "purposes require real owner authorization",
    )
    unregistered = tuple(
        metric
        for metric in (*objectives, *ties)
        if metric != "core_replay_id" and metric not in OBJECTIVE_DIRECTIONS
    )
    _rule(
        "objectives_registered",
        "Every objective and tie-break has a registered direction",
        not unregistered and bool(objectives),
        "all registered"
        if not unregistered and objectives
        else f"unregistered metric(s): {', '.join(unregistered)}"
        if unregistered
        else "no objective is selected",
    )

    failures = tuple(rule for rule in rules if not rule.passed)
    return CharterSatisfiabilityReport(
        goal=goal,
        rules=tuple(rules),
        passed=not failures,
        failures=failures,
        resolved_profile_count=profiles,
        challenger_count=challengers,
        configuration_sentence=_configuration_sentence(profiles, challengers),
        pareto_objectives=objectives,
        tie_breaks=ties,
    )
