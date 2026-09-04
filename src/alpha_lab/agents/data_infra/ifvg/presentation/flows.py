"""Goal-derived conditional study flows (UI-2; plan §5.4, §7, §9 Phase 2).

The eight wizard step keys (``study_drafts.STEP_KEYS``) are the persisted
vocabulary; a *flow* is the ordered subset a goal actually needs, plus the
steps it SKIPS — each with a visible reason ("Prop Contracts — skipped: no
prop objective"). Nothing here renders; the wizard consumes the flow.

Rules (owner Q1 / Q4 and the revision-2 clarifications):

* the Goal (objective) step and Review are always present; the Validation
  step (evidence dates, seed, execution truth, the authorization checklist)
  stays in every research flow because those inputs have no other home;
* strategy goals (Evaluate / Compare / FSM search / feature-model) skip the
  prop-contract and risk-policy steps ONLY while no prop objective is
  selected — a selected prop objective keeps the contract step in the flow
  so a missing verified contract blocks there, instead of the objective
  silently disappearing (plan F-04 / §7);
* Evaluate skips the search space (no comparison); Prop Benchmark applies
  contracts to one frozen executed-strategy stream (no search space);
  feature / model evidence runs over the baseline cohort (feature bundles,
  the model ladder and the optional regime study are configured in the
  pipeline Configure phase reached from Review);
* the REAL ≤5-day verification slice is the exact baseline with the
  verification gates only: Goal → Baseline → Validation → Review; a
  synthetic fixture under Implementation Verification keeps its study
  family's flow (the R4-era synthetic search stays reachable);
* the advanced end-to-end study (Full Authorized Development) walks every
  applicable step, then Configure / Review / Launch.

Exact restore: a draft stores its ``current_step_key``; a legacy eight-step
``step_index`` maps to the same key, or to the next step present in the flow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..study_drafts import STEP_KEYS
from ..study_presentation import OBJECTIVE_TEMPLATES
from .charter_satisfiability import StudyGoal, goal_for_draft, is_prop_metric
from .run_purpose import EvidenceClass, RunPurpose

__all__ = [
    "FLOW_STEP_TITLES",
    "FlowStep",
    "StudyFlow",
    "flow_for_draft_fields",
    "flow_for_goal",
    "goal_for_flow",
    "prop_objective_selected",
    "restore_step_index",
    "step_position",
]

#: The default (R4) titles per step key; a flow may retitle a step for its goal.
FLOW_STEP_TITLES: Mapping[str, str] = MappingProxyType(
    {
        "objective": "Goal & objective",
        "baseline": "Baseline",
        "search_space": "Strategy Search Space",
        "prop_contracts": "Prop Contracts",
        "risk_policies": "Risk Policies",
        "benchmarks": "Benchmarks",
        "validation": "Validation",
        "review": "Review & Launch",
    }
)

_NO_PROP_OBJECTIVE = "skipped: no prop objective is selected for this goal"
_VERIFICATION_ONLY = (
    "skipped: the real verification slice applies verification_control_flow_gates_v1 "
    "only — research gates, prop contracts and risk policies are never shown on a "
    "verification path"
)


@dataclass(frozen=True)
class FlowStep:
    key: str
    title: str
    #: why the step is part of (or absent from) this flow — always visible
    reason: str
    skip_reason: str | None = None


@dataclass(frozen=True)
class StudyFlow:
    flow_id: str
    goal: StudyGoal
    purpose: RunPurpose
    evidence_class: EvidenceClass
    steps: tuple[FlowStep, ...]
    skipped: tuple[FlowStep, ...]

    @property
    def step_keys(self) -> tuple[str, ...]:
        return tuple(step.key for step in self.steps)

    def step(self, key: str) -> FlowStep | None:
        return next((step for step in self.steps if step.key == key), None)

    def includes(self, key: str) -> bool:
        return any(step.key == key for step in self.steps)


_TEMPLATES_BY_ID = {template.template_id: template for template in OBJECTIVE_TEMPLATES}


def prop_objective_selected(objective_fields: Mapping[str, Any] | None) -> bool:
    """Whether the objective step SELECTS a prop metric (a prop-bearing
    template, or a custom objective list carrying a non-strategy metric)."""

    fields = dict(objective_fields or {})
    template = _TEMPLATES_BY_ID.get(str(fields.get("template_id") or ""))
    if template is not None and template.template_id != "custom":
        return bool(template.prop_bearing)
    custom = tuple(str(metric) for metric in (fields.get("custom_objectives") or ()))
    return any(is_prop_metric(metric) for metric in custom)


def goal_for_flow(mode_id: str, question_id: str | None, purpose: RunPurpose | str) -> StudyGoal:
    """The study goal a mode + question + purpose resolve to. The pipeline
    family refines by purpose: Development Research → feature / model
    evidence (S05–S10 over the baseline cohort); Full Authorized Development
    → the advanced end-to-end study."""

    goal = goal_for_draft(mode_id, question_id)
    if mode_id == "full_pipeline_run":
        return (
            StudyGoal.ADVANCED_END_TO_END
            if RunPurpose(purpose) is RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
            else StudyGoal.FEATURE_MODEL
        )
    return goal


def _flow(
    flow_id: str,
    goal: StudyGoal,
    purpose: RunPurpose,
    evidence: EvidenceClass,
    included: Mapping[str, tuple[str, str]],
    skipped: Mapping[str, str],
) -> StudyFlow:
    steps = tuple(
        FlowStep(key=key, title=included[key][0], reason=included[key][1])
        for key in STEP_KEYS
        if key in included
    )
    absent = tuple(
        FlowStep(key=key, title=FLOW_STEP_TITLES[key], reason="", skip_reason=skipped[key])
        for key in STEP_KEYS
        if key in skipped
    )
    missing = set(STEP_KEYS) - set(included) - set(skipped)
    if missing or set(included) & set(skipped):
        raise ValueError(f"flow {flow_id!r} must include or skip every step exactly once")
    return StudyFlow(
        flow_id=flow_id,
        goal=goal,
        purpose=purpose,
        evidence_class=evidence,
        steps=steps,
        skipped=absent,
    )


def flow_for_goal(
    goal: StudyGoal | str,
    *,
    purpose: RunPurpose | str,
    evidence_class: EvidenceClass | str,
    prop_objective: bool,
) -> StudyFlow:
    goal = StudyGoal(goal)
    purpose = RunPurpose(purpose)
    evidence = EvidenceClass(evidence_class)
    real_verification = (
        purpose is RunPurpose.IMPLEMENTATION_VERIFICATION and evidence is EvidenceClass.REAL
    )
    if goal is StudyGoal.VERIFICATION or real_verification:
        return _flow(
            "verification_real_v1",
            StudyGoal.VERIFICATION,
            purpose,
            evidence,
            {
                "objective": ("Goal", "Implementation Verification over the exact baseline"),
                "baseline": ("Exact baseline", "the one profile the real slice replays"),
                "validation": (
                    "Validation",
                    "the canonical logical-day allowlist, the seed and the typed "
                    "VerificationAuthorizationRef readiness",
                ),
                "review": ("Review", "verification-only; never a research result"),
            },
            {
                "search_space": (
                    "skipped: the real verification slice replays the exact baseline only "
                    "(no challengers)"
                ),
                "prop_contracts": _VERIFICATION_ONLY,
                "risk_policies": _VERIFICATION_ONLY,
                "benchmarks": _VERIFICATION_ONLY,
            },
        )
    validation = (
        "Validation",
        "evidence dates, the deterministic seed, the sequential-V1 execution truth and "
        "the computation-path-scoped authorization checklist",
    )
    prop_steps = {
        "prop_contracts": (
            "Firm contracts",
            "the selected prop objective requires at least one first_party_verified "
            "contract — its absence blocks here, the objective is never rewritten",
        ),
        "risk_policies": (
            "Risk / payout policies",
            "per-firm account, risk, withdrawal and replacement policies",
        ),
    }
    prop_skipped = {"prop_contracts": _NO_PROP_OBJECTIVE, "risk_policies": _NO_PROP_OBJECTIVE}

    def _with_prop(included: dict, skipped: dict, suffix: str) -> tuple[dict, dict, str]:
        if prop_objective:
            return {**included, **prop_steps}, skipped, f"{suffix}_prop"
        return included, {**skipped, **prop_skipped}, suffix

    if goal is StudyGoal.EVALUATE_ONE:
        included = {
            "objective": ("Goal", "Evaluate one configuration — no comparison or delta claim"),
            "baseline": ("Configuration", "exactly one selected frozen profile as the anchor"),
            "benchmarks": ("Strategy gates", "the strategy feasibility gates, for display"),
            "validation": validation,
            "review": ("Review", "one resolved profile; challenger count 0"),
        }
        skipped = {
            "search_space": (
                "skipped: Evaluate makes no comparison — a challenger would turn this "
                "into Compare or FSM search"
            )
        }
        included, skipped, suffix = _with_prop(included, skipped, "evaluate_one_v1")
        return _flow(suffix, goal, purpose, evidence, included, skipped)
    if goal is StudyGoal.COMPARE_WITH_BASELINE:
        included = {
            "objective": ("Goal", "Compare one configuration with the baseline"),
            "baseline": ("Baseline", "the exact baseline profile"),
            "search_space": (
                "Challenger",
                "exactly one challenger configuration, which may carry several registered "
                "differences",
            ),
            "benchmarks": ("Strategy gates", "the strategy feasibility gates, for display"),
            "validation": validation,
            "review": (
                "Review",
                "every registered difference is listed; deltas only when compatible",
            ),
        }
        included, skipped, suffix = _with_prop(included, {}, "compare_with_baseline_v1")
        return _flow(suffix, goal, purpose, evidence, included, skipped)
    if goal is StudyGoal.FSM_SEARCH:
        included = {
            "objective": ("Goal", "Search FSM parameters"),
            "baseline": ("Baseline", "the exact baseline profile every axis starts from"),
            "search_space": (
                "Search axes",
                "≥ 1 registered axis with a challenger value (≥ 2 profiles); recommended "
                "preset: parent-retest timeout unbounded / 240 / 360 / 480",
            ),
            "benchmarks": ("Strategy gates", "the strategy and robustness gates, for display"),
            "validation": validation,
            "review": ("Review", "baseline + N challengers; one sequential replay per profile"),
        }
        included, skipped, suffix = _with_prop(included, {}, "fsm_search_v1")
        return _flow(suffix, goal, purpose, evidence, included, skipped)
    if goal is StudyGoal.PROP_FEASIBILITY:
        return _flow(
            "prop_feasibility_v1",
            goal,
            purpose,
            evidence,
            {
                "objective": ("Goal", "Test prop-firm feasibility"),
                "baseline": (
                    "Strategy source",
                    "the one frozen executed-strategy stream the contracts apply to",
                ),
                "prop_contracts": (
                    "Firm contracts (≥ 1 verified)",
                    "Prop Benchmark needs at least one first_party_verified contract",
                ),
                "risk_policies": prop_steps["risk_policies"],
                "benchmarks": ("Prop benchmarks", "the payout-reliability gates"),
                "validation": validation,
                "review": ("Review", "contracts × policies over one frozen stream"),
            },
            {
                "search_space": (
                    "skipped: Prop Benchmark applies contracts and policies to one frozen "
                    "executed-strategy stream — no strategy search"
                )
            },
        )
    if goal is StudyGoal.UNIVERSAL_PROP:
        return _flow(
            "universal_prop_v1",
            goal,
            purpose,
            evidence,
            {
                "objective": ("Goal", "Find one strategy configuration across multiple firms"),
                "baseline": ("Baseline", "the universal strategy profile's anchor"),
                "search_space": ("Strategy search", "one universal strategy profile per child"),
                "prop_contracts": (
                    "Firm contracts (≥ 2 verified)",
                    "Universal Prop Search needs at least two first_party_verified contracts "
                    "— one firm cannot be universal",
                ),
                "risk_policies": ("Policies", "per-firm account-policy sets"),
                "benchmarks": ("Universal gates", "the universal payout-reliability gates"),
                "validation": validation,
                "review": ("Review", "one strategy profile, every firm's policy set"),
            },
            {},
        )
    if goal is StudyGoal.FEATURE_MODEL:
        included = {
            "objective": ("Goal", "Evaluate feature and model evidence (S05–S10)"),
            "baseline": ("Cohort anchor", "the baseline cohort the feature views derive from"),
            "benchmarks": ("Strategy gates", "the strategy feasibility gates, for display"),
            "validation": validation,
            "review": (
                "Review & Configure",
                "Feature bundles · Model ladder · Optional regime analysis are configured "
                "in the pipeline Configure phase; S11 stays blocked by contract",
            ),
        }
        skipped = {
            "search_space": (
                "skipped: feature / model evidence runs over the baseline cohort — a strategy "
                "search is a different goal"
            )
        }
        included, skipped, suffix = _with_prop(included, skipped, "feature_model_v1")
        return _flow(suffix, goal, purpose, evidence, included, skipped)
    # the advanced end-to-end study: every applicable step, then Configure / Review / Launch
    return _flow(
        "advanced_end_to_end_v1",
        StudyGoal.ADVANCED_END_TO_END,
        purpose,
        evidence,
        {
            "objective": ("Goal", "Run an advanced end-to-end study (full stage plan)"),
            "baseline": ("Baseline", "the exact baseline profile"),
            "search_space": ("Strategy Search Space", "registered axes and challenger values"),
            "prop_contracts": ("Prop Contracts", "first_party_verified contracts, if any"),
            "risk_policies": ("Risk Policies", "per-firm policy sets, if any"),
            "benchmarks": ("Benchmarks", "strategy, prop and robustness gates"),
            "validation": validation,
            "review": (
                "Review & Launch",
                "Configure → Review → Launch with the typed acknowledgement, once",
            ),
        },
        {},
    )


def flow_for_draft_fields(
    objective_fields: Mapping[str, Any] | None,
    *,
    purpose: RunPurpose | str,
    evidence_class: EvidenceClass | str,
) -> StudyFlow:
    """The flow of a draft from its objective step (mode, question, template /
    custom objectives), its purpose and its evidence class."""

    fields = dict(objective_fields or {})
    mode_id = str(fields.get("mode_id") or "")
    goal = goal_for_flow(mode_id, fields.get("question_id"), purpose)
    return flow_for_goal(
        goal,
        purpose=purpose,
        evidence_class=evidence_class,
        prop_objective=prop_objective_selected(fields),
    )


def step_position(flow: StudyFlow, step_key: str) -> int | None:
    keys = flow.step_keys
    return keys.index(step_key) if step_key in keys else None


def restore_step_index(
    flow: StudyFlow, *, stored_step_key: str | None, legacy_step_index: int
) -> int:
    """Exact restore: the stored step key when the flow includes it; otherwise
    the legacy eight-step index maps to its key, or to the next step the flow
    includes (never to an absent step); out-of-range lands on Review."""

    keys = flow.step_keys
    if stored_step_key in keys:
        return keys.index(str(stored_step_key))
    index = max(int(legacy_step_index), 0)
    if index >= len(STEP_KEYS):
        return len(keys) - 1
    for candidate in STEP_KEYS[index:]:
        if candidate in keys:
            return keys.index(candidate)
    return len(keys) - 1
