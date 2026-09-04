"""New Study — the eight-step wizard (R4; FUX §§6–15; UI-1 amendments).

``render_new_study`` owns drafts (disk-persisted, autosaved on Next, Save
Draft always visible, exact-step restore, immutable after freeze, Clone as
New Search), the five study modes, the objective/baseline/search-space/
prop/risk/benchmark/validation/review steps, and the freeze-and-launch
handoff. Freezing builds a ``SearchCharterPayload``, runs the fail-closed
``validate_charter``, saves it immutably, marks the draft frozen, and — for
charters with a REGISTERED runner entry — spawns the detached job shim and
routes to Active Runs. Launching happens ONLY inside the explicit button
handler; render, import, AppTest, and page refresh launch nothing.

UI-1 (plan Phase 1): one presentation-only RUN PURPOSE replaces the
independent namespace / run-scope radios — the goal card derives the run
scope, the semantic namespace (verified id shown read-only), the evidence
class and the authorization class of the ACTUAL computation path (a
synthetic fixture → the typed marker; the real ≤5-day slice → the owner's
validated VerificationAuthorizationRef; research scopes → the
computation-path-scoped owner bundle); a ``CharterSatisfiabilityReport``
refuses contradictory drafts BEFORE freeze and the selected objective is
never rewritten; the launch resolves the registered runner BEFORE spawning
and reports success ONLY after the worker's state file exists; the V1
executor is sequential (no worker control); development dates follow the
backend logical-day contract with the frozen warmup prefix.

UI-2 (plan Phase 2; owner Q2): a new draft lives in the SESSION until the
first explicit Save Draft or the first valid Next (no file is written); a
persisted draft autosaves on every change with a visible Saved / Autosaved /
Not-saved chip; the draft name is required before the first persistence
(default proposal ``<goal> — <baseline> — <date>``) and an identical persisted
draft raises a warning; the eight fixed steps are replaced by the
GOAL-DERIVED flow (``presentation/flows.py``) — skipped steps are listed with
their reason on the goal card and contribute nothing to satisfiability or
charter assembly; a selected prop objective keeps the contract step in the
flow and blocks there instead of disappearing; an archived draft cannot be
edited until restored from History.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_ui_common import (
    SESSION_DRAFT_KEY,
    STATE_PREFIX,
    cli_escape_hatch,
    dev_only_badge,
    identity_block,
    render_empty_state,
    sanitize_error,
    sanitize_select,
    status_badge,
    verification_badge,
)

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    PROFILE_CAPABILITY_REGISTRY,
    ProfileCapabilityStatus,
)
from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
from alpha_lab.agents.data_infra.ifvg.presentation.charter_satisfiability import (
    CharterSatisfiabilityReport,
    StudyGoal,
    evaluate_charter_satisfiability,
    goal_for_draft,
    is_prop_metric,
    summarize_challenger_differences,
)
from alpha_lab.agents.data_infra.ifvg.presentation.flows import (
    FLOW_STEP_TITLES,
    StudyFlow,
    flow_for_draft_fields,
    prop_objective_selected,
    restore_step_index,
)
from alpha_lab.agents.data_infra.ifvg.presentation.help_registry import (
    help_for_metric,
    help_text,
)
from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import (
    gate_metric_key,
)
from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    PURPOSE_DESCRIPTIONS,
    PURPOSE_LABELS,
    PURPOSE_NAMESPACE_CLASS,
    PURPOSE_RUN_SCOPE,
    EvidenceClass,
    PurposeResolution,
    PurposeResolutionStatus,
    ResolvedPurpose,
    RunPurpose,
    RunPurposeAnnotation,
    resolve_draft_purpose,
    resolve_purpose,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    SyntheticAuthorizationMarker,
    derive_authorization_requirements,
)
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
)
from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    CharterValidationError,
    CostPolicy,
    DatePolicy,
    ObjectivePolicy,
    ResolvedPropGateThresholds,
    ResolvedRobustnessGateThresholds,
    ResolvedStrategyGateThresholds,
    SearchCharterEnvelope,
    SearchCharterPayload,
    SearchMode,
    SimulationProtocol,
    save_charter,
    validate_charter,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonicalize_section,
    name_free_section_hash,
)
from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (
    RunnerEntryError,
    resolve_registered_runner_entry,
    runner_entry_key_for_charter,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    PROPOSED_VERIFICATION_ALLOWLIST,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    STEP_KEYS,
    DraftError,
    StudyDraft,
    clone_draft,
    find_duplicate_drafts,
    list_drafts,
    load_draft,
    mark_frozen,
    new_draft,
    proposed_draft_name,
    save_draft,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    AXIS_GROUP_ORDER,
    DEVELOPMENT_EVIDENCE_FIRST_DAY,
    DEVELOPMENT_EVIDENCE_LAST_DAY,
    INTERPRETATIONS,
    MAX_ACCOUNTS_PER_FIRM,
    OBJECTIVE_TEMPLATES,
    RESEARCH_QUESTIONS,
    RISK_POLICY_TEMPLATES,
    SESSION_DIRECTION_AXES,
    STUDY_MODES,
    WIZARD_STEP_TITLES,
    axis_renders_widget,
    classification_label,
    computation_path_chip,
    enumerate_child_count,
    estimate_search_work,
    format_duration,
    format_storage,
    grouped_axis_keys,
    mode_compatibility_error,
    validate_baseline_step,
    validate_benchmarks_step,
    validate_objective_step,
    validate_prop_step,
    validate_review_step,
    validate_risk_step,
    validate_search_space_step,
    validate_validation_step,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    AuthorizationReadiness,
    owner_authorization_bundle_from_store,
    owner_authorization_readiness,
    resolve_store_namespace,
    verification_authorization_readiness,
    verification_owner_bundle,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (
    FULL_SCOPE_ACKNOWLEDGEMENT,
    FULL_SCOPE_WARNING_TEXT,
    StudyStatusKey,
)

__all__ = ["render_new_study"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DRAFT_KEY = f"{STATE_PREFIX}draft_id"
_ACTIVE_DRAFT_KEY = f"{STATE_PREFIX}wz_active_draft"
_W = f"{STATE_PREFIX}wz_"

_INTERPRETATIONS = INTERPRETATIONS
_SESSION_DIRECTION_AXES = SESSION_DIRECTION_AXES

#: Draft-header widget keys that are already instantiated when the shell
#: decides whether to purge stale step-widget state on a draft switch.
_HEADER_WIDGET_KEYS = frozenset(
    {
        f"{_W}new_purpose",
        f"{_W}new_mode",
        f"{_W}new_draft",
        f"{_W}open_draft",
        f"{_W}open_btn",
    }
)

#: UI-1 honest launch: after the detached spawn the handler waits for the
#: worker's persisted ``search_state.json`` up to this many seconds before
#: reporting the launch as started (tests shorten it); silence renders the
#: typed ``launch_not_started`` state, never a success.
LAUNCH_STATE_WAIT_SECONDS = 10.0
_LAUNCH_POLL_SECONDS = 0.2

_EVIDENCE_LABELS: Mapping[str, str] = {
    "synthetic_fixture": "Synthetic fixture — proves the machinery; never evidence",
    "real": (
        "Real ≤5-day verification slice — requires the owner's validated "
        "VerificationAuthorizationRef"
    ),
}

_PROP_GATE_DEFAULTS = ResolvedPropGateThresholds(
    minimum_first_payout_probability_60d=0.5,
    maximum_breach_probability_90d=0.35,
    minimum_expected_net_payout_90d=None,
    minimum_p10_net_payout_90d=None,
    maximum_p90_payout_drought_days=None,
    minimum_three_payout_probability=None,
).model_dump()
_ROBUSTNESS_GATE_DEFAULTS = ResolvedRobustnessGateThresholds(
    maximum_neighbor_expectancy_degradation_r=None,
    minimum_plateau_width=None,
    maximum_worst_firm_breach_probability_90d=None,
    minimum_time_block_sign_consistency=None,
).model_dump()


def _spawn_search_job(command: list[str]) -> int:
    """The ONLY DETACHED process launch (tests monkeypatch it).

    The wizard's one other subprocess use is the read-only
    ``git rev-parse HEAD`` provenance query in :func:`_commit_of`
    (DEV-R4-14); nothing else in the UI lane touches ``subprocess``.
    """

    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0
    )
    process = subprocess.Popen(  # noqa: S603 — exact interpreter + repo script
        command, cwd=_REPO_ROOT, creationflags=creation_flags
    )
    return process.pid


# ─────────────────────────────────────────────────────────────────────────────
# Draft header
# ─────────────────────────────────────────────────────────────────────────────


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _session_draft(st_module) -> StudyDraft | None:
    """The session-only draft (owner Q2), or ``None``."""

    payload = st_module.session_state.get(SESSION_DRAFT_KEY)
    if not isinstance(payload, dict):
        return None
    known = set(StudyDraft.__dataclass_fields__)
    try:
        return StudyDraft(**{key: value for key, value in payload.items() if key in known})
    except TypeError:
        return None


def _stash_session_draft(st_module, draft: StudyDraft) -> None:
    st_module.session_state[SESSION_DRAFT_KEY] = asdict(draft)


def _draft_is_persisted(draft_root: Path, draft: StudyDraft) -> bool:
    return (Path(draft_root) / draft.draft_id / "draft.json").exists()


def _persist_now(st_module, draft_root: Path, draft: StudyDraft) -> bool:
    """The ONE persistence seam of the wizard: the draft name is required
    before the first save; a saved draft leaves the session stash."""

    if not (draft.display_name or "").strip():
        st_module.error(
            "Draft name required — name the draft above before it is saved (the "
            "proposed default is '<goal> — <baseline> — <date>')."
        )
        return False
    save_draft(draft_root, draft)
    st_module.session_state.pop(SESSION_DRAFT_KEY, None)
    return True


def _saved_chip(st_module, state: str, draft: StudyDraft) -> None:
    """The visible persistence state (owner Q2): Not saved yet (session only)
    · Saved · Autosaved."""

    if state == "session":
        st_module.markdown(
            "**● Not saved yet** — session only, no file written; Save Draft or the "
            "first valid Next persists it, then every change autosaves."
        )
    elif state == "autosaved":
        st_module.markdown(f"**✓ Autosaved** {draft.updated_at_utc}")
    else:
        st_module.markdown(f"**✓ Saved** {draft.updated_at_utc}")


def _duplicate_warning(
    st_module,
    draft: StudyDraft,
    draft_root: Path,
    *,
    step_key: str,
    fields: Mapping[str, Any],
) -> None:
    """Owner Q2: an identical persisted goal / baseline draft is a warning
    on the unsaved draft — never a block. The visible fields of the current
    step count (the Goal step's question before it is ever saved)."""

    objective = dict(fields) if step_key == "objective" else draft.step_payload("objective")
    baseline = dict(fields) if step_key == "baseline" else draft.step_payload("baseline")
    try:
        duplicates = find_duplicate_drafts(
            list_drafts(draft_root),
            mode_id=draft.mode_id,
            question_id=objective.get("question_id"),
            purpose=(draft.purpose_annotation or {}).get("purpose"),
            baseline_profile_name=baseline.get("baseline_profile_name"),
            exclude_draft_id=draft.draft_id,
        )
    except Exception:  # noqa: BLE001 — a listing failure never blocks authoring
        return
    if duplicates:
        names = ", ".join(f"'{other.display_name}'" for other in duplicates[:3])
        st_module.warning(
            f"An identical unsaved goal / baseline draft already exists: {names}. "
            "Open it from History or keep this one — saving twice creates two drafts."
        )


def _draft_header(st_module, roots: Mapping[str, Any]) -> StudyDraft | None:
    draft_root = Path(roots["draft_root"])
    drafts = list_drafts(draft_root)
    open_labels = {
        f"{draft.display_name} · {draft.status} · {draft.draft_id[:8]}": draft
        for draft in drafts
    }
    columns = st_module.columns([2, 2, 1])
    with columns[0]:
        purpose_labels = [PURPOSE_LABELS[purpose] for purpose in RunPurpose]
        purpose_label = st_module.selectbox(
            "Purpose for a new draft",
            purpose_labels,
            key=f"{_W}new_purpose",
            help=(
                "The purpose derives the run scope, the semantic namespace, the "
                "evidence class and the authorization class; it is a presentation "
                "annotation and never changes a charter identity (owner Q1)."
            ),
        )
        mode_label = st_module.selectbox(
            "Study mode for a new draft",
            [mode.label for mode in STUDY_MODES],
            key=f"{_W}new_mode",
            help="The five study modes (FUX §6).",
        )
        if st_module.button(
            "Start new draft",
            key=f"{_W}new_draft",
            help=(
                "Creates a SESSION draft — nothing is written until the first Save "
                "Draft or the first valid Next (owner Q2)."
            ),
        ):
            mode = next(m for m in STUDY_MODES if m.label == mode_label)
            purpose = next(p for p in RunPurpose if PURPOSE_LABELS[p] == purpose_label)
            now = _utc_now()
            draft = new_draft(
                mode.mode_id,
                display_name=proposed_draft_name(
                    mode.label, baseline_profile_name=None, day=now[:10]
                ),
            )
            draft.purpose_annotation = RunPurposeAnnotation(
                purpose=purpose,
                derivation="card_selected",
                owner_confirmed=True,
                updated_at=now,
            ).to_dict()
            draft.steps["validation"] = {
                "run_scope": PURPOSE_RUN_SCOPE[purpose],
                "evidence_class": (
                    "synthetic_fixture"
                    if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
                    else "real"
                ),
                "worker_limit": 1,
            }
            draft.current_step_key = "objective"
            _stash_session_draft(st_module, draft)
            st_module.session_state[_DRAFT_KEY] = draft.draft_id
    with columns[1]:
        options = ["—", *open_labels]
        sanitize_select(st_module, f"{_W}open_draft", options)
        opened = st_module.selectbox(
            "Open a saved draft", options, key=f"{_W}open_draft",
            help=help_text("wizard.open_draft"),
        )
        if opened != "—" and st_module.button("Open", key=f"{_W}open_btn"):
            st_module.session_state.pop(SESSION_DRAFT_KEY, None)
            st_module.session_state[_DRAFT_KEY] = open_labels[opened].draft_id
    with columns[2]:
        st_module.caption(f"{len(drafts)} saved draft(s)")

    draft_id = st_module.session_state.get(_DRAFT_KEY)
    if not draft_id:
        st_module.info(
            "Start a new draft (or one from Start's task cards) or open a saved "
            "one. A new draft is session-only until its first Save Draft or first "
            "valid Next; afterwards it autosaves on every change."
        )
        return None
    session = _session_draft(st_module)
    if (
        session is not None
        and session.draft_id == draft_id
        and not _draft_is_persisted(draft_root, session)
    ):
        return session
    try:
        draft = load_draft(draft_root, str(draft_id))
    except DraftError as error:
        st_module.warning(sanitize_error(error))
        st_module.session_state.pop(_DRAFT_KEY, None)
        return None
    if draft.archived:
        render_empty_state(
            st_module,
            "draft_archived",
            detail=f"'{draft.display_name}' archived {draft.archived_at_utc or ''}",
        )
        return None
    if draft.status == "frozen":
        status_badge(st_module, StudyStatusKey.FROZEN)
        identity_block(st_module, "Frozen search charter id", draft.frozen_search_id or "")
        if st_module.button(
            "Clone as New Search",
            key=f"{_W}clone_frozen",
            help=help_text("wizard.clone_frozen"),
        ):
            clone = clone_draft(draft)
            save_draft(draft_root, clone)
            st_module.session_state[_DRAFT_KEY] = clone.draft_id
            st_module.rerun()
        return None
    return draft


# ─────────────────────────────────────────────────────────────────────────────
# UI-1 — purpose resolution, typed readiness and satisfiability (pure inputs
# from the presentation package and the providers; nothing here launches)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DraftResolution:
    """Everything the steps and the freeze handler need about a draft's
    purpose: the resolution, the resolved purpose (``None`` when
    ``purpose_unresolved``), the purpose's roots, the typed readiness of the
    real authorization the path requires and the requirement set."""

    resolution: PurposeResolution
    resolved: ResolvedPurpose | None
    roots: dict[str, Any]
    readiness: AuthorizationReadiness | None
    requirement_set: Any | None
    #: UI-2: the goal-derived flow (``None`` while the purpose is unresolved —
    #: every step then renders until the owner confirms the purpose)
    flow: StudyFlow | None = None


def _effective_payload(
    draft: StudyDraft, flow: StudyFlow | None, step_key: str
) -> dict[str, Any]:
    """A step the flow SKIPS contributes nothing (a stale payload from an
    earlier goal is ignored, never silently carried into a charter)."""

    if flow is not None and not flow.includes(step_key):
        return {}
    return dict(draft.steps.get(step_key) or {})


def _search_mode_for(draft: StudyDraft, axes_present: bool) -> SearchMode:
    if draft.mode_id in {mode.value for mode in SearchMode}:
        return SearchMode(draft.mode_id)
    # Full Pipeline Run drafts (mode 5) freeze the underlying SEARCH charter
    # their pipeline runs over (engineering default, DECISIONS_TAKEN R5)
    return SearchMode.FSM_CONFIG_SEARCH if axes_present else SearchMode.SINGLE_CONFIGURATION


def _requirement_set_for_draft(draft: StudyDraft, run_scope: str, flow: StudyFlow | None = None):
    """Mirror of ``validate_charter``'s computation path and dimensions —
    the requirement set a real bundle must reference."""

    from alpha_lab.agents.data_infra.ifvg.study.computation_path import (  # noqa: PLC0415
        ComputationPath,
    )

    prop_selected = tuple(
        _effective_payload(draft, flow, "prop_contracts").get("selected_contract_ids") or ()
    )
    selections = _effective_payload(draft, flow, "search_space").get("axis_selections") or {}
    axes = sorted(str(axis) for axis in selections if tuple(selections[axis] or ()))
    search_mode = _search_mode_for(draft, bool(axes))
    path = ComputationPath(
        full_strategy_replay=any(
            SEARCH_AXIS_REGISTRY_V1[axis].requires_full_sequential_replay
            for axis in axes
            if axis in SEARCH_AXIS_REGISTRY_V1
        )
        or search_mode in (SearchMode.FSM_CONFIG_SEARCH, SearchMode.SINGLE_CONFIGURATION),
        feature_materialization=False,
        label_recomputation=False,
        model_refit=False,
        model_gated_sequential_replay=False,
        cost_recomputation=True,
        prop_resimulation=bool(prop_selected),
        bootstrap_resimulation=False,
        reuse_trade_stream_hash=False,
    )
    dimensions = tuple(f"strategy_profile.{axis}" for axis in axes) + tuple(
        f"prop_contract.{contract}" for contract in prop_selected
    )
    return derive_authorization_requirements(run_scope, dimensions, path, (), prop_selected)


def _resolve_for_draft(draft: StudyDraft, roots: Mapping[str, Any]) -> DraftResolution:
    """Resolve the draft's purpose (stored annotation → unambiguous legacy
    derivation → ``purpose_unresolved``), the purpose's roots, the verified
    store namespace and the TYPED readiness of the authorization the actual
    path requires."""

    from ifvg_study_tab import roots_for_purpose  # noqa: PLC0415

    validation = draft.step_payload("validation")
    resolution = resolve_draft_purpose(
        draft.purpose_annotation, run_scope=validation.get("run_scope")
    )
    if resolution.status is not PurposeResolutionStatus.RESOLVED or resolution.purpose is None:
        return DraftResolution(resolution, None, dict(roots), None, None)
    purpose = resolution.purpose
    purpose_roots = roots_for_purpose(roots, purpose)
    store_root = Path(purpose_roots["store_root"])
    stored_evidence = str(validation.get("evidence_class") or "")
    if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION:
        evidence = (
            EvidenceClass.REAL if stored_evidence == "real" else EvidenceClass.SYNTHETIC_FIXTURE
        )
    else:
        evidence = EvidenceClass.REAL  # a synthetic fixture is confined to verification
    run_scope = PURPOSE_RUN_SCOPE[purpose]
    flow = flow_for_draft_fields(
        {**draft.step_payload("objective"), "mode_id": draft.mode_id},
        purpose=purpose,
        evidence_class=evidence,
    )
    requirement_set = _requirement_set_for_draft(draft, run_scope, flow)
    namespace = resolve_store_namespace(
        store_root, expected_class=PURPOSE_NAMESPACE_CLASS[purpose]
    )
    readiness: AuthorizationReadiness | None = None
    if evidence is EvidenceClass.REAL:
        baseline = draft.step_payload("baseline")
        try:
            if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION:
                readiness = verification_authorization_readiness(
                    store_root,
                    baseline_profile_name=baseline.get("baseline_profile_name") or None,
                    baseline_section_config_hash=(
                        baseline.get("baseline_section_config_hash") or None
                    ),
                    allowlist=tuple(validation.get("real_dates") or ()) or None,
                )
            else:
                readiness = owner_authorization_readiness(
                    store_root,
                    requirement_set,
                    expected_class=PURPOSE_NAMESPACE_CLASS[purpose],
                )
        except Exception as error:  # noqa: BLE001 — typed, sanitized at render
            readiness = AuthorizationReadiness(
                authorization_class=(
                    "verification_authorization_ref"
                    if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
                    else "owner_authorization_bundle"
                ),
                status="unavailable",
                detail=f"readiness lookup failed: {sanitize_error(error)}",
            )
    resolved = resolve_purpose(
        purpose,
        evidence_class=evidence,
        namespace=namespace,
        authorization_readiness=readiness.status if readiness is not None else None,
        authorization_detail=readiness.detail if readiness is not None else "",
    )
    return DraftResolution(
        resolution, resolved, purpose_roots, readiness, requirement_set, flow=flow
    )


def _purpose_card(
    st_module, draft: StudyDraft, draft_root: Path, resolution: DraftResolution
) -> None:
    """The goal card fixed at the top of every step (plan §7 New Study)."""

    st_module.markdown("**Goal card**")
    resolved = resolution.resolved
    if resolved is None:
        render_empty_state(
            st_module, "purpose_unresolved", detail=resolution.resolution.reason
        )
        labels = [PURPOSE_LABELS[purpose] for purpose in RunPurpose]
        chosen = st_module.selectbox(
            "Confirm the run purpose",
            labels,
            key=f"{_W}confirm_purpose",
            help=(
                "Confirming records a presentation annotation only; the run scope, "
                "namespace and authorization are still validated by the backend."
            ),
        )
        if st_module.button(
            "Confirm purpose",
            key=f"{_W}confirm_purpose_btn",
            help=help_text("wizard.confirm_purpose"),
        ):
            purpose = next(p for p in RunPurpose if PURPOSE_LABELS[p] == chosen)
            draft.purpose_annotation = RunPurposeAnnotation(
                purpose=purpose,
                derivation="owner_confirmed",
                owner_confirmed=True,
                updated_at=_utc_now(),
            ).to_dict()
            validation = draft.step_payload("validation")
            validation["run_scope"] = PURPOSE_RUN_SCOPE[purpose]
            validation.setdefault(
                "evidence_class",
                "synthetic_fixture"
                if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
                else "real",
            )
            save_draft(draft_root, draft)
            st_module.rerun()
        return
    if resolved.purpose is RunPurpose.IMPLEMENTATION_VERIFICATION:
        verification_badge(st_module)
    st_module.table(
        {
            "purpose": [resolved.label],
            "run scope": [resolved.run_scope],
            "namespace class": [resolved.namespace_class],
            "evidence class": [resolved.evidence_class.value],
            "authorization": [
                f"{resolved.authorization_class} · readiness: "
                f"{resolved.authorization_readiness}"
            ],
            "stage plan policy": [resolved.stage_plan_policy],
            "publication": [
                "separate eligibility gate"
                if resolved.publication_available
                else "not available for this purpose"
            ],
            "result label": [resolved.result_label],
        }
    )
    st_module.caption(PURPOSE_DESCRIPTIONS[resolved.purpose])
    st_module.caption(
        f"Store namespace: **{resolved.namespace_status}** — "
        f"{sanitize_error(resolved.namespace_detail)}"
    )
    if resolved.store_namespace_id:
        identity_block(st_module, "store_namespace_id", resolved.store_namespace_id)
    if not resolved.freeze_allowed and resolved.freeze_block_reason:
        st_module.warning(
            f"Freeze is disabled — {sanitize_error(resolved.freeze_block_reason)}"
        )
    flow = resolution.flow
    if flow is not None:
        st_module.markdown(
            "**Flow** (goal-derived; plan §5.4): "
            + " › ".join(step.title for step in flow.steps)
        )
        if flow.skipped:
            st_module.markdown(
                "**Skipped steps** (listed with their reason, never rendered empty):"
            )
            for step in flow.skipped:
                st_module.markdown(f"- {FLOW_STEP_TITLES[step.key]} — {step.skip_reason}")


def _gates_configured(stored: Mapping[str, Any] | None, defaults: Mapping[str, Any]) -> bool:
    return any(
        value is not None and value != defaults.get(name)
        for name, value in (stored or {}).items()
    )


def _resolved_objectives(draft: StudyDraft) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The SELECTED objectives and tie-breaks — never rewritten (plan F-04)."""

    objective_step = draft.step_payload("objective")
    template = next(
        (
            template
            for template in OBJECTIVE_TEMPLATES
            if template.template_id == objective_step.get("template_id")
        ),
        OBJECTIVE_TEMPLATES[-1],
    )
    if template.template_id == "custom":
        return tuple(objective_step.get("custom_objectives") or ()), ("core_replay_id",)
    return tuple(template.primary_objective), (*template.tie_breaks, "core_replay_id")


def _challenger_selections(
    draft: StudyDraft, flow: StudyFlow | None = None
) -> dict[str, tuple[str, ...]]:
    """Selected CHALLENGER value ids per axis (the baseline value never counts;
    a flow that skips the search space contributes none)."""

    selections = _effective_payload(draft, flow, "search_space").get("axis_selections") or {}
    out: dict[str, tuple[str, ...]] = {}
    for axis, values in selections.items():
        spec = SEARCH_AXIS_REGISTRY_V1.get(str(axis))
        baseline = spec.baseline_value_id if spec is not None else None
        chosen = tuple(str(value) for value in (values or ()) if value != baseline)
        if chosen:
            out[str(axis)] = chosen
    return out


def _satisfiability_for_draft(
    draft: StudyDraft, resolved: ResolvedPurpose, flow: StudyFlow | None = None
) -> CharterSatisfiabilityReport:
    objective_step = draft.step_payload("objective")
    prop = _effective_payload(draft, flow, "prop_contracts")
    benchmarks = _effective_payload(draft, flow, "benchmarks")
    pareto, ties = _resolved_objectives(draft)
    try:
        goal = goal_for_draft(draft.mode_id, objective_step.get("question_id"))
    except ValueError:
        goal = StudyGoal.ADVANCED_END_TO_END
    if (
        resolved.purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
        and resolved.evidence_class is EvidenceClass.REAL
    ):
        goal = StudyGoal.VERIFICATION
    return evaluate_charter_satisfiability(
        goal=goal,
        search_mode=draft.mode_id,
        axis_selections=_challenger_selections(draft, flow),
        baseline_profile_name=str(
            draft.step_payload("baseline").get("baseline_profile_name") or ""
        ),
        pareto_objectives=pareto,
        tie_breaks=ties,
        selected_contract_ids=tuple(prop.get("selected_contract_ids") or ()),
        launchable_contract_ids=tuple(prop.get("launchable_contract_ids") or ()),
        purpose=resolved.purpose,
        evidence_class=resolved.evidence_class,
        prop_gates_configured=_gates_configured(
            benchmarks.get("prop_gates"), _PROP_GATE_DEFAULTS
        ),
        robustness_gates_configured=_gates_configured(
            benchmarks.get("robustness_gates"), _ROBUSTNESS_GATE_DEFAULTS
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step bodies — each renders widgets seeded from the draft and returns the
# collected field mapping used for validation + persistence.
# ─────────────────────────────────────────────────────────────────────────────


def _step_objective(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("objective")
    mode = next((m for m in STUDY_MODES if m.mode_id == draft.mode_id), STUDY_MODES[0])
    st_module.markdown(f"**Study mode:** {mode.label}")
    st_module.caption(mode.description)
    question_labels = list(RESEARCH_QUESTIONS.values())
    stored_question = RESEARCH_QUESTIONS.get(payload.get("question_id", ""), None)
    question_label = st_module.radio(
        "Research question",
        question_labels,
        index=question_labels.index(stored_question)
        if stored_question in question_labels
        else 0,
        key=f"{_W}question",
        help=help_text("wizard.research_question"),
    )
    question_id = next(
        key for key, label in RESEARCH_QUESTIONS.items() if label == question_label
    )
    template_labels = [template.label for template in OBJECTIVE_TEMPLATES]
    stored_template = payload.get("template_id", "")
    template_label = st_module.selectbox(
        "Objective template",
        template_labels,
        index=next(
            (
                index
                for index, template in enumerate(OBJECTIVE_TEMPLATES)
                if template.template_id == stored_template
            ),
            0,
        ),
        key=f"{_W}template",
        help=help_text("wizard.objective_template"),
    )
    template = next(t for t in OBJECTIVE_TEMPLATES if t.label == template_label)
    st_module.markdown("**Resolved objective (nothing hidden behind the template):**")
    st_module.table(
        {
            "primary objective": [", ".join(template.primary_objective) or "—"],
            "hard constraints": [", ".join(template.hard_constraints)],
            "tie-breaks": [", ".join(template.tie_breaks) or "—"],
            "status": [template.status],
        }
    )
    st_module.caption(
        "The objective policy contains no hidden weighted score; resolved "
        "thresholds are shown field-by-field in the Benchmarks step."
    )
    custom_objectives: tuple[str, ...] = tuple(payload.get("custom_objectives") or ())
    if template.template_id == "custom":
        from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
            OBJECTIVE_DIRECTIONS,
        )

        custom_objectives = tuple(
            st_module.multiselect(
                "Custom registered objective metrics",
                sorted(OBJECTIVE_DIRECTIONS),
                default=list(custom_objectives),
                key=f"{_W}custom_objectives",
                help=help_text("wizard.custom_objectives"),
            )
        )
    incompatibility = mode_compatibility_error(
        draft.mode_id, question_id, template.template_id
    )
    if incompatibility:
        st_module.error(incompatibility)
    return {
        "mode_id": draft.mode_id,
        "question_id": question_id,
        "template_id": template.template_id,
        "custom_objectives": custom_objectives,
    }


def _step_baseline(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("baseline")
    names = list(PROFILE_CAPABILITY_REGISTRY)
    stored = payload.get("baseline_profile_name") or names[0]
    selected = st_module.selectbox(
        "Baseline profile",
        names,
        index=names.index(stored) if stored in names else 0,
        key=f"{_W}baseline",
        help=(
            "Only a capability-eligible baseline can anchor a charter. "
            "Generated children are baseline diffs gated by their own "
            "GeneratedProfileCapability — they are never presented as fixed "
            "M0–M3 profiles."
        ),
    )
    capability = PROFILE_CAPABILITY_REGISTRY[selected]
    blocked_reason: str | None = None
    if capability.status is not ProfileCapabilityStatus.RUNNABLE:
        blocked_reason = f"{capability.status.value}: {capability.reason}"
        st_module.error(
            f"This baseline is not runnable — {blocked_reason}. Blocked, "
            "experimental, superseded, or unavailable profiles render their "
            "reason and no launch path exists (FUX §9)."
        )
    card: dict[str, str] = {"Profile": selected, "Status": capability.status.value}
    section_hash = ""
    name_free_hash = ""
    try:
        resolved = resolve_profile_config({"profile_name": selected})
        section = canonicalize_section(resolved.section)
        from strategy_core.strategies.ifvg_smc.section import (  # noqa: PLC0415
            ifvg_profile_hash,
        )

        section_hash = ifvg_profile_hash(section)
        name_free_hash = name_free_section_hash(section)
        direction = []
        if getattr(section, "enable_longs", False):
            direction.append("long")
        if getattr(section, "enable_shorts", False):
            direction.append("short")
        card.update(
            {
                "Entry thesis": ", ".join(
                    getattr(section, "entry_families", ()) or ()
                )
                or str(getattr(section, "entry_family", "—")),
                "Direction": "/".join(direction) or "—",
                "Target / label family": str(getattr(section, "label_family", "—")),
                "FSM concurrency": "one active setup / one active trade",
                "Development date range": (
                    "through 2026-06-10 21:00Z (development cutoff)"
                ),
            }
        )
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.warning(f"Baseline resolution failed: {sanitize_error(error)}")
        blocked_reason = blocked_reason or "baseline section did not resolve"
    st_module.markdown("**Baseline card**")
    st_module.table({key: [value] for key, value in card.items()})
    with st_module.expander("View Technical Identity"):
        identity_block(st_module, "Profile hash (resolved section)", section_hash)
        identity_block(st_module, "Name-free section hash", name_free_hash)
        st_module.caption(
            "Core Replay ID / artifact ids appear per child after launch; "
            "resolved Strategy-Core and Quant-Lab source identities are "
            "computed at freeze time and ride the replay identity. Seed and "
            "date-policy references are shown in the Validation step."
        )
    return {
        "baseline_profile_name": selected,
        "baseline_section_config_hash": section_hash,
        "baseline_blocked_reason": blocked_reason,
    }


def _axis_card(st_module, axis_key: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    spec = SEARCH_AXIS_REGISTRY_V1[axis_key]
    label = classification_label(spec.classification)
    chip = computation_path_chip(spec)
    st_module.markdown(
        f"**{spec.human_label}**  \n"
        f"`{spec.technical_key}` · {label} · {chip}"
    )
    st_module.caption(spec.description)
    baseline_value = AXIS_VALUE_REGISTRY_V1.get(spec.baseline_value_id)
    baseline_label = (
        baseline_value.human_label if baseline_value is not None else spec.baseline_value_id
    )
    selected: tuple[str, ...] = ()
    if axis_renders_widget(spec):
        options = {
            AXIS_VALUE_REGISTRY_V1[value_id].human_label: value_id
            for value_id in spec.registered_values
            if value_id in AXIS_VALUE_REGISTRY_V1
        }
        stored = tuple(payload.get(axis_key) or ())
        default_labels = [
            label for label, value_id in options.items() if value_id in stored
        ]
        chosen = st_module.multiselect(
            f"Registered search values — baseline: {baseline_label}",
            list(options),
            default=default_labels,
            key=f"{_W}axis_{axis_key}",
            help=(
                "Values come from the typed axis/value registry only; no "
                "free-form overrides exist anywhere in this workspace."
            ),
        )
        selected = tuple(options[label] for label in chosen)
    else:
        st_module.caption(
            f"Value: **{baseline_label}** — no input widget "
            f"({label.lower()}; visible for audit)"
        )
    with st_module.expander("Evidence & authorization"):
        st_module.write(f"Baseline value: `{spec.baseline_value_id}`")
        st_module.write(
            "Registered values: "
            + (", ".join(f"`{value}`" for value in spec.registered_values) or "—")
        )
        rows = []
        for value_id in spec.registered_values:
            value = AXIS_VALUE_REGISTRY_V1.get(value_id)
            if value is None:
                continue
            rows.append(
                {
                    "value": value_id,
                    "capability": str(getattr(value, "capability_status", "—")),
                    "owner ratification": str(
                        getattr(value, "owner_ratification_status", "—")
                    ),
                    "evidence ref": str(
                        getattr(value, "ratification_evidence_ref", None) or "missing"
                    ),
                }
            )
        if rows:
            st_module.dataframe(rows, width="stretch", hide_index=True)
        st_module.write(
            f"Artifact / profile impact: "
            f"{getattr(spec, 'expected_artifact_effect', '—')}; "
            f"full sequential replay required: "
            f"{'yes' if spec.requires_full_sequential_replay else 'no'}"
        )
    return selected


def _step_search_space(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("search_space")
    stored_selections: dict[str, tuple[str, ...]] = {
        key: tuple(values)
        for key, values in (payload.get("axis_selections") or {}).items()
    }
    selections: dict[str, tuple[str, ...]] = {}
    grouped = grouped_axis_keys()
    for group in AXIS_GROUP_ORDER:
        keys = grouped.get(group, ())
        if not keys:
            continue
        with st_module.expander(
            group, expanded=group in ("Staleness", "Parent Handling")
        ):
            for axis_key in keys:
                chosen = _axis_card(st_module, axis_key, stored_selections)
                if chosen:
                    selections[axis_key] = chosen
    session_changed = sorted(set(selections) & _SESSION_DIRECTION_AXES)
    interpretation = payload.get("interpretation") or _INTERPRETATIONS[2]
    if session_changed:
        st_module.markdown("**Session/direction interpretation** (FUX §10.5)")
        interpretation = st_module.radio(
            f"Changed session/direction axes {session_changed} are interpreted as",
            _INTERPRETATIONS,
            index=_INTERPRETATIONS.index(interpretation)
            if interpretation in _INTERPRETATIONS
            else 2,
            key=f"{_W}interpretation",
            help=(
                "Only 'Sequential Strategy Profile' creates an executable "
                "counterfactual child; the other interpretations are "
                "descriptive and never replay."
            ),
        )
    count = enumerate_child_count(selections) if selections else 1
    st_module.caption(
        "Baseline only: 1 configuration"
        if count <= 1
        else f"{count} configurations: baseline + {count - 1} challenger"
        + ("s" if count - 1 != 1 else "")
        + " (the baseline value of every searched axis enumerates at launch)"
    )
    return {
        "mode_id": draft.mode_id,
        "axis_selections": selections,
        "interpretation": interpretation,
    }


def _contract_cards(
    st_module, roots: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        load_contract_summaries,
    )

    try:
        return load_contract_summaries(Path(roots["store_root"]))
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.warning(f"Contract listing failed: {sanitize_error(error)}")
        return ()


def _prop_objectives_of(draft: StudyDraft) -> tuple[str, ...]:
    pareto, ties = _resolved_objectives(draft)
    return tuple(
        metric
        for metric in (*pareto, *ties)
        if metric != "core_replay_id" and is_prop_metric(metric)
    )


def _step_prop(st_module, draft: StudyDraft, roots: Mapping[str, Any]) -> dict[str, Any]:
    payload = draft.step_payload("prop_contracts")
    mode = next((m for m in STUDY_MODES if m.mode_id == draft.mode_id), STUDY_MODES[0])
    prop_objective = prop_objective_selected(
        {**draft.step_payload("objective"), "mode_id": draft.mode_id}
    )
    prop_objectives = _prop_objectives_of(draft) if prop_objective else ()
    cards = _contract_cards(st_module, roots)
    if not cards:
        render_empty_state(st_module, "no_verified_firm_contract")
        if prop_objective:
            # UI-2 (plan §7 / F-04): the objective is echoed unchanged and the
            # path blocks here with the contract workflow — never rewritten
            st_module.error(
                f"The selected prop objective(s) {', '.join(prop_objectives)} require at "
                "least one first_party_verified firm contract; the objective is never "
                "rewritten. Compile first-party contract evidence and complete the owner "
                "review (owner decisions 5/6), or choose a strategy-only objective on the "
                "Goal step."
            )
        elif not mode.requires_prop_selection:
            st_module.caption(
                "This study mode does not require a prop contract; continue "
                "to Risk Policies."
            )
        return {
            "mode_id": draft.mode_id,
            "selected_contract_ids": (),
            "launchable_contract_ids": (),
            "prop_objective_selected": prop_objective,
            "prop_objectives": prop_objectives,
        }
    stored = set(payload.get("selected_contract_ids") or ())
    selected: list[str] = []
    launchable: list[str] = []
    for card in cards:
        contract_id = card["firm_contract_id"]
        title = (
            f"{card['firm']} · {card['account_type']} · "
            f"{card['account_size_label']}"
        )
        with st_module.expander(title, expanded=False):
            synthetic = card["verification_status"] == "synthetic_fixture_verified"
            if synthetic:
                st_module.warning(
                    "SYNTHETIC fixture contract — proves the machinery; it "
                    "can never enter a real study.",
                    icon="🧪",
                )
            st_module.table(
                {
                    "contract version": [card["contract_version"]],
                    "effective date": [card["effective_date"]],
                    "verification status": [card["verification_status"]],
                    "drawdown rule": [card["drawdown_rule"]],
                    "daily rule": [card["daily_rule"]],
                    "contract limits": [card["contract_limits"]],
                    "payout rules": [card["payout_rules"]],
                    "fees": [card["fees"]],
                    "post-payout behavior": [card["post_payout"]],
                    "source/evidence status": [card["source_status"]],
                    "min path capabilities": [card["path_capabilities"]],
                }
            )
            with st_module.expander("Audit: evidence bundle & supersession"):
                identity_block(st_module, "firm_contract_id", contract_id)
                identity_block(
                    st_module,
                    "contract_evidence_bundle_id",
                    card.get("contract_evidence_bundle_id", ""),
                )
                st_module.write(
                    f"Supersession: {card.get('supersession', 'not superseded')}"
                )
            st_module.caption(
                "Assumed intrabar path scenarios are NOT part of the firm "
                "contract; they are selected under the simulation policy "
                "(Risk Policies / Validation)."
            )
            eligible = card["launchable"]
            if eligible:
                launchable.append(contract_id)
                checked = st_module.checkbox(
                    "Select this contract",
                    value=contract_id in stored,
                    key=f"{_W}contract_{contract_id[:16]}",
                    help=help_text("wizard.select_contract"),
                )
                if checked:
                    selected.append(contract_id)
            else:
                st_module.caption(
                    f"Not launchable: {card.get('launch_block_reason', '—')} "
                    "(no selection control is rendered)"
                )
    return {
        "mode_id": draft.mode_id,
        "selected_contract_ids": tuple(selected),
        "launchable_contract_ids": tuple(launchable),
        "prop_objective_selected": prop_objective,
        "prop_objectives": prop_objectives,
    }


def _step_risk(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("risk_policies")
    selected_contracts = tuple(
        draft.step_payload("prop_contracts").get("selected_contract_ids") or ()
    )
    st_module.markdown("**Universal Strategy Profile**")
    if selected_contracts:
        tree = "\n".join(
            f"    ├── Firm {contract[:8]}… Account/Risk/Withdrawal/Replacement "
            "Policy Set"
            for contract in selected_contracts
        )
        st_module.code("Universal Strategy Profile\n" + tree, language="text")
    else:
        st_module.caption(
            "No firm contracts selected — strategy-only study; firm-specific "
            "policies never alter strategy identity."
        )
    per_firm: dict[str, dict[str, Any]] = dict(payload.get("per_firm_policies") or {})
    for contract in selected_contracts:
        with st_module.expander(f"Firm {contract[:12]}… policy set", expanded=True):
            stored = per_firm.get(contract, {})
            template = st_module.selectbox(
                "Risk template",
                RISK_POLICY_TEMPLATES,
                index=RISK_POLICY_TEMPLATES.index(stored.get("risk_template"))
                if stored.get("risk_template") in RISK_POLICY_TEMPLATES
                else 0,
                key=f"{_W}risk_{contract[:16]}",
                help=help_text("wizard.risk_template"),
            )
            risk_value = st_module.number_input(
                "Risk parameter (USD or % per the template)",
                min_value=0.0,
                value=float(stored.get("risk_value", 100.0)),
                key=f"{_W}riskv_{contract[:16]}",
                help=(
                    "Every result-changing account policy is visible here "
                    "and enters the resolved simulation identity. UI caps "
                    "follow OWNER_DECISIONS.md without implying research "
                    "ratification."
                ),
            )
            withdrawal = st_module.selectbox(
                "Withdrawal behavior (trader policy — separate from the "
                "firm's permitted payout contract)",
                (
                    "withdraw_at_every_eligibility_v1",
                    "withdraw_monthly_cadence_v1",
                    "accumulate_no_withdrawal_v1",
                ),
                index=0
                if stored.get("withdrawal_policy")
                not in (
                    "withdraw_monthly_cadence_v1",
                    "accumulate_no_withdrawal_v1",
                )
                else (
                    "withdraw_at_every_eligibility_v1",
                    "withdraw_monthly_cadence_v1",
                    "accumulate_no_withdrawal_v1",
                ).index(stored.get("withdrawal_policy")),
                key=f"{_W}wd_{contract[:16]}",
                help=help_text("wizard.withdrawal_behavior"),
            )
            replacement = st_module.selectbox(
                "Replacement policy (separate scenario; owner decision 12)",
                ("none", "replace_once", "replace_twice"),
                index=("none", "replace_once", "replace_twice").index(
                    stored.get("replacement_policy", "none")
                ),
                key=f"{_W}rep_{contract[:16]}",
                help=help_text("wizard.replacement_policy"),
            )
            n_accounts = st_module.number_input(
                f"Accounts per firm (max {MAX_ACCOUNTS_PER_FIRM}, owner decision 13)",
                min_value=1,
                max_value=MAX_ACCOUNTS_PER_FIRM,
                value=int(stored.get("n_accounts", 1)),
                key=f"{_W}acct_{contract[:16]}",
                help=help_text("wizard.accounts_per_firm"),
            )
            if int(n_accounts) > 1:
                st_module.info(
                    "Copied accounts visibly share the SAME resampled "
                    "market/trade path per simulation path — no per-account "
                    "resampling exists."
                )
            if risk_value <= 0:
                st_module.warning(
                    "A zero risk parameter means every trade is skipped "
                    "(typed skip reason), not silently resized."
                )
            per_firm[contract] = {
                "risk_template": template,
                "risk_value": float(risk_value),
                "withdrawal_policy": withdrawal,
                "replacement_policy": replacement,
                "n_accounts": int(n_accounts),
            }
    per_firm = {
        contract: policy
        for contract, policy in per_firm.items()
        if contract in selected_contracts
    }
    return {
        "selected_contract_ids": selected_contracts,
        "per_firm_policies": per_firm,
    }


_STRATEGY_GATE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("min_executed_trades", "Minimum executed trades", "trades"),
    ("min_independent_days", "Minimum independent trading days", "days"),
    ("min_net_expectancy_r", "Minimum net expectancy", "R"),
    ("min_profit_factor", "Minimum profit factor", "ratio"),
    ("max_drawdown_r", "Maximum drawdown", "R"),
    ("max_time_under_water_days", "Maximum time under water", "days"),
    ("min_session_stability_score", "Minimum session stability", "score 0–1"),
    ("min_time_block_sign_consistency", "Minimum time-block sign consistency", "share"),
    ("max_top_day_pnl_share", "Maximum single-day PnL share", "share"),
    ("max_top_setup_pnl_share", "Maximum single-setup PnL share", "share"),
    ("require_bootstrap_ci_excludes_zero", "Bootstrap CI must exclude zero", "flag"),
)

_PROP_GATE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "minimum_first_payout_probability_60d",
        "Minimum P(first payout within 60 days)",
        "probability",
    ),
    (
        "maximum_breach_probability_90d",
        "Maximum P(breach within 90 days)",
        "probability",
    ),
    (
        "minimum_expected_net_payout_90d",
        "Minimum expected 90-day net payout",
        "USD",
    ),
    ("minimum_p10_net_payout_90d", "Minimum P10 90-day net payout", "USD"),
    (
        "maximum_p90_payout_drought_days",
        "Maximum P90 payout drought",
        "days",
    ),
    (
        "minimum_three_payout_probability",
        "Minimum P(three payouts)",
        "probability",
    ),
)

_ROBUSTNESS_GATE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "maximum_neighbor_expectancy_degradation_r",
        "Maximum neighbor expectancy degradation",
        "R",
    ),
    ("minimum_plateau_width", "Minimum plateau width", "steps"),
    (
        "maximum_worst_firm_breach_probability_90d",
        "Maximum worst-firm breach probability",
        "probability",
    ),
    (
        "minimum_time_block_sign_consistency",
        "Minimum time-block sign consistency",
        "share",
    ),
    (
        "minimum_outer_fold_recurrence",
        "Minimum outer-fold recurrence (schema-reserved)",
        "share",
    ),
)


def _gate_group(
    st_module,
    *,
    title: str,
    rows: tuple[tuple[str, str, str], ...],
    defaults: Mapping[str, Any],
    stored: Mapping[str, Any],
    key_prefix: str,
) -> dict[str, Any]:
    st_module.markdown(f"**{title}**")
    values: dict[str, Any] = {}
    for name, human, unit in rows:
        default = stored.get(name, defaults.get(name))
        columns = st_module.columns([3, 2, 2])
        with columns[0]:
            st_module.write(f"{human}  \n`{name}` · {unit}")
        with columns[1]:
            if isinstance(default, bool):
                value: Any = st_module.checkbox(
                    "required",
                    value=bool(default),
                    key=f"{key_prefix}{name}",
                    label_visibility="visible",
                    help=help_text("wizard.gate_required"),
                )
            elif default is None:
                st_module.caption("not required (no threshold)")
                value = None
            else:
                value = st_module.number_input(
                    f"Resolved value for {human}",
                    value=float(default),
                    key=f"{key_prefix}{name}",
                    label_visibility="collapsed",
                    help=help_for_metric(gate_metric_key(name)),
                )
        with columns[2]:
            st_module.caption(
                "status: proposed_protocol_default"
                if default is not None
                else "status: not required"
            )
        values[name] = value
    return values


_ROBUSTNESS_GOALS = frozenset(
    {StudyGoal.FSM_SEARCH, StudyGoal.UNIVERSAL_PROP, StudyGoal.ADVANCED_END_TO_END}
)


def _step_benchmarks(
    st_module, draft: StudyDraft, flow: StudyFlow | None = None
) -> dict[str, Any]:
    payload = draft.step_payload("benchmarks")
    show_prop = flow is None or flow.includes("prop_contracts")
    show_robustness = flow is None or flow.goal in _ROBUSTNESS_GOALS
    strategy_defaults = ResolvedStrategyGateThresholds(
        min_session_stability_score=0.5
    ).model_dump()
    prop_defaults = ResolvedPropGateThresholds(
        minimum_first_payout_probability_60d=0.5,
        maximum_breach_probability_90d=0.35,
        minimum_expected_net_payout_90d=None,
        minimum_p10_net_payout_90d=None,
        maximum_p90_payout_drought_days=None,
        minimum_three_payout_probability=None,
    ).model_dump()
    robustness_defaults = ResolvedRobustnessGateThresholds(
        maximum_neighbor_expectancy_degradation_r=None,
        minimum_plateau_width=None,
        maximum_worst_firm_breach_probability_90d=None,
        minimum_time_block_sign_consistency=None,
    ).model_dump()
    strategy = _gate_group(
        st_module,
        title="1 · Underlying Strategy Gate",
        rows=_STRATEGY_GATE_ROWS,
        defaults=strategy_defaults,
        stored=payload.get("strategy_gates") or {},
        key_prefix=f"{_W}sg_",
    )
    if show_prop:
        prop = _gate_group(
            st_module,
            title="2 · Prop Feasibility Gate",
            rows=_PROP_GATE_ROWS,
            defaults=prop_defaults,
            stored=payload.get("prop_gates") or {},
            key_prefix=f"{_W}pg_",
        )
    else:
        prop = {}
        st_module.caption(
            "2 · Prop Feasibility Gate — skipped: no prop objective is selected for "
            "this goal (select a prop-bearing objective on the Goal step to add it)"
        )
    if show_robustness:
        robustness = _gate_group(
            st_module,
            title="3 · Robustness Gate",
            rows=_ROBUSTNESS_GATE_ROWS,
            defaults=robustness_defaults,
            stored=payload.get("robustness_gates") or {},
            key_prefix=f"{_W}rg_",
        )
    else:
        robustness = {}
        st_module.caption(
            "3 · Robustness Gate — not applicable: this goal enumerates no search "
            "axes (plateau / neighbour gates need a searched neighbourhood)"
        )
    st_module.info(
        "Why configurations fail here: each gate row explains its own "
        "failure; later gates that were not run display the exact earlier "
        "gate that stopped them (e.g. 'Not run — strategy gate failed'). "
        "Verification fixtures use verification_control_flow_gates_v1, "
        "NEVER these research thresholds. Frontier ranking happens only "
        "after the feasibility gates, and no hidden weighted score exists."
    )
    return {
        "strategy_gates": strategy,
        "prop_gates": prop,
        "robustness_gates": robustness,
    }


def _step_validation(
    st_module, draft: StudyDraft, roots: Mapping[str, Any], resolution: DraftResolution
) -> dict[str, Any]:
    payload = draft.step_payload("validation")
    resolved = resolution.resolved
    seed_default = int(payload.get("seed", 7))
    if resolved is None:
        render_empty_state(
            st_module, "purpose_unresolved", detail=resolution.resolution.reason
        )
        return {
            "run_scope": payload.get("run_scope"),
            "evidence_class": payload.get("evidence_class"),
            "real_dates": tuple(payload.get("real_dates") or ()),
            "warmup_dates": tuple(payload.get("warmup_dates") or ()),
            "seed": seed_default,
            "worker_limit": 1,
        }
    run_scope = resolved.run_scope
    store_root = Path(resolution.roots["store_root"])
    st_module.markdown(
        f"**Run scope:** `{run_scope}` — derived from the purpose "
        f"**{resolved.label}**; it is not a separate selector (owner Q1)."
    )
    baseline = draft.step_payload("baseline")
    if resolved.purpose is RunPurpose.IMPLEMENTATION_VERIFICATION:
        verification_badge(st_module)
        stored_evidence = str(payload.get("evidence_class") or "synthetic_fixture")
        evidence_label = st_module.radio(
            "Evidence class",
            list(_EVIDENCE_LABELS.values()),
            index=1 if stored_evidence == "real" else 0,
            key=f"{_W}evidence",
            help=(
                "A synthetic fixture carries the typed SyntheticAuthorizationMarker "
                "and proves the machinery only. The real ≤5-day slice needs the "
                "owner's VerificationAuthorizationRef bound to this store's verified "
                "namespace and current supersession head; its typed readiness is "
                "shown below."
            ),
        )
        evidence_class = next(
            key for key, label in _EVIDENCE_LABELS.items() if label == evidence_label
        )
        allowlist = tuple(PROPOSED_VERIFICATION_ALLOWLIST)
        st_module.write(
            "One canonical program-wide allowlist (read-only; owner decision "
            "21/R-5 — the proposed candidate pending coverage sign-off; the owner "
            "selects the logical trading-day window from the HARDENING-BACKEND "
            "shortlist):"
        )
        st_module.code("\n".join(allowlist), language="text")
        st_module.download_button(
            "Download resolved allowlist",
            data=json.dumps({"allowlist": allowlist}, indent=2),
            file_name="verification_allowlist_v1.json",
            key=f"{_W}dl_allowlist",
            help=help_text("wizard.download_allowlist"),
        )
        st_module.caption(
            "Research interpretation and catalog activation are prohibited "
            "for verification runs."
        )
        real_dates = allowlist
        warmup_dates: tuple[str, ...] = ()
        st_module.caption(
            f"Real-date count: {len(real_dates)} · warmup: 0 real days "
            "(the profile-matching precomputed seed snapshot replaces real "
            "warmup; warmup + evidence ≤ 5 holds)"
        )
        # FUX §14.2 as amended: the TYPED readiness of the exact
        # VerificationAuthorizationRef — derived from the repository, never
        # asserted, never collapsed into a Boolean
        try:
            readiness: AuthorizationReadiness | None = verification_authorization_readiness(
                store_root,
                baseline_profile_name=baseline.get("baseline_profile_name") or None,
                baseline_section_config_hash=(
                    baseline.get("baseline_section_config_hash") or None
                ),
                allowlist=real_dates,
            )
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            st_module.warning(f"Authorization lookup failed: {sanitize_error(error)}")
            readiness = None
        if readiness is not None:
            if readiness.status == "ready":
                st_module.success(
                    f"VerificationAuthorizationRef: READY — {sanitize_error(readiness.detail)}"
                )
                for run_id in readiness.evidence_ids:
                    identity_block(st_module, "verification_run_id", run_id)
            elif readiness.status in ("missing", "store_unmarked"):
                render_empty_state(
                    st_module,
                    "verification_authorization_missing",
                    detail=f"{readiness.status}: {readiness.detail}",
                )
            else:
                render_empty_state(
                    st_module,
                    "authorization_not_ready",
                    detail=f"{readiness.status}: {readiness.detail}",
                )
            if evidence_class == "synthetic_fixture":
                st_module.caption(
                    "Synthetic fixture: the typed marker satisfies the charter layer; "
                    "the real-ref readiness above is informational."
                )
            elif readiness.status != "ready":
                st_module.error(
                    "The real ≤5-day slice cannot freeze until the "
                    "VerificationAuthorizationRef readiness is ready."
                )
    else:
        evidence_class = "real"
        st_module.warning(
            "Owner-authorized development scope: no run starts during tests, "
            "import, startup, or page render. Launch requires a frozen charter or "
            "pipeline specification, a ready owner authorization bundle bound to "
            "this store, and — for Full Authorized Development — the typed second "
            "confirmation in Review & Launch."
        )
        st_module.write(
            "Frozen warmup prefix (read-only; the development date policy "
            "requires it before any evidence date):"
        )
        st_module.code("\n".join(FROZEN_WARMUP_DATES), language="text")
        raw = st_module.text_area(
            "Evidence dates — logical trading days between "
            f"{DEVELOPMENT_EVIDENCE_FIRST_DAY} and {DEVELOPMENT_EVIDENCE_LAST_DAY}, "
            "one ISO date per line",
            value="\n".join(payload.get("real_dates") or ()),
            key=f"{_W}full_dates",
            help=(
                "Weekends, registered full closures, the protected buffer "
                "2026-06-11 and the sealed range are refused field-by-field here, "
                "never as a generic freeze failure."
            ),
        )
        real_dates = tuple(line.strip() for line in raw.splitlines() if line.strip())
        warmup_dates = FROZEN_WARMUP_DATES
        st_module.download_button(
            "Download resolved date list",
            data=json.dumps(
                {"warmup": list(warmup_dates), "evidence": list(real_dates)}, indent=2
            ),
            file_name="authorized_development_dates.json",
            key=f"{_W}dl_dates",
            help=help_text("wizard.download_dates"),
        )
        st_module.caption(
            f"Warmup: {len(warmup_dates)} frozen days · evidence: {len(real_dates)}"
        )
        readiness = resolution.readiness
        if readiness is not None:
            if readiness.status == "ready":
                st_module.success(
                    "Owner authorization bundle: READY — "
                    f"{sanitize_error(readiness.detail)}"
                )
            else:
                render_empty_state(
                    st_module,
                    "authorization_not_ready",
                    detail=f"{readiness.status}: {readiness.detail}",
                )
    st_module.write("Protected buffer and sealed boundary (read-only):")
    st_module.code(
        "protected: 2026-06-11 (never constructed, listed, stat-ed, or read)\n"
        "sealed range: per development access policy — counters must stay 0\n"
        "development cutoff: 2026-06-10T21:00:00Z",
        language="text",
    )
    seed = int(
        st_module.number_input(
            "Deterministic seed",
            value=seed_default,
            step=1,
            key=f"{_W}seed",
            help=help_text("wizard.seed"),
        )
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        EXECUTION_MODE_V1,
        SUPPORTED_CHILD_WORKERS,
    )

    st_module.caption(
        f"Execution mode: {EXECUTION_MODE_V1} — sequential in V1; effective workers: "
        f"{SUPPORTED_CHILD_WORKERS}. No worker control exists; the backend refuses "
        "any other value before a job is created."
    )
    selections_for_estimate = _challenger_selections(draft, resolution.flow)
    n_children_estimate = enumerate_child_count(selections_for_estimate)
    estimate = estimate_search_work(
        n_children=n_children_estimate,
        n_replay_days=len(real_dates) + len(warmup_dates),
        n_firm_policy_combinations=sum(
            int(policy.get("n_accounts", 1))
            for policy in (
                _effective_payload(draft, resolution.flow, "risk_policies").get(
                    "per_firm_policies"
                )
                or {}
            ).values()
        ),
    )
    st_module.table(
        {
            "search algorithm": ["deterministic_exhaustive_v1"],
            "outer-fold protocol": [
                "schema-reserved (exploratory lane; owner decision 15)"
            ],
            "block-bootstrap protocol": ["single_day · 10,000 paths · seed 42"],
            "stress paths": ["9 registered scenarios × bootstrap"],
            "estimated runtime": [
                format_duration(estimate.estimated_runtime_seconds)
            ],
            "estimated storage": [
                format_storage(estimate.estimated_storage_mb)
            ],
        }
    )
    requirements = resolution.requirement_set
    st_module.markdown("**Authorization requirement checklist** (computation-path-scoped)")
    missing_keys = set(
        resolution.readiness.missing_decision_keys if resolution.readiness else ()
    )
    if requirements is not None:
        for requirement in requirements.payload.requirements:
            if evidence_class == "synthetic_fixture":
                state = "informational — a synthetic fixture needs the typed marker only"
            elif resolution.readiness is None:
                state = "readiness not evaluated"
            elif requirement.decision_key in missing_keys:
                state = "no registered evidence artifact — the freeze fails closed"
            else:
                state = f"readiness: {resolution.readiness.status}"
            st_module.write(f"- `{requirement.decision_key}` — {requirement.reason} ({state})")
        if not requirements.payload.requirements:
            st_module.caption(
                "No owner decisions are required for this computation path."
            )
    return {
        "run_scope": run_scope,
        "evidence_class": evidence_class,
        "real_dates": real_dates,
        "warmup_dates": warmup_dates,
        "seed": seed,
        "worker_limit": 1,
    }


def _step_review(
    st_module, draft: StudyDraft, roots: Mapping[str, Any], resolution: DraftResolution
) -> dict[str, Any]:
    objective = draft.step_payload("objective")
    risk = _effective_payload(draft, resolution.flow, "risk_policies")
    validation = draft.step_payload("validation")
    resolved = resolution.resolved
    selections = _challenger_selections(draft, resolution.flow)
    n_children = enumerate_child_count(selections) if selections else 1
    firm_combos = sum(
        int(policy.get("n_accounts", 1))
        for policy in (risk.get("per_firm_policies") or {}).values()
    )
    estimate = estimate_search_work(
        n_children=n_children,
        n_replay_days=len(tuple(validation.get("real_dates") or ()))
        + len(tuple(validation.get("warmup_dates") or ())),
        n_firm_policy_combinations=firm_combos,
        n_stress_scenarios=0,
    )
    baseline_step = draft.step_payload("baseline")
    frozen_axes = sorted(
        key for key in SEARCH_AXIS_REGISTRY_V1 if key not in selections
    )
    pareto, ties = _resolved_objectives(draft)
    st_module.markdown("**Resolved charter preview**")
    st_module.table(
        {
            "purpose": [resolved.label if resolved is not None else "unresolved"],
            "study mode": [draft.mode_id],
            "objective template": [objective.get("template_id", "—")],
            "objective (selected; never rewritten)": [", ".join(pareto) or "—"],
            "tie-breaks": [", ".join(ties)],
            "baseline": [baseline_step.get("baseline_profile_name", "—")],
            "changed axes": [", ".join(sorted(selections)) or "none (baseline only)"],
            "frozen dimensions": [
                f"{len(frozen_axes)} axes at their baseline values "
                f"(first: {', '.join(frozen_axes[:4])}, …)"
            ],
            "unique strategy profiles": [str(n_children)],
            "firm/risk/account-policy combinations": [str(firm_combos)],
            "run scope": [resolved.run_scope if resolved is not None else "—"],
            "evidence class": [
                resolved.evidence_class.value if resolved is not None else "—"
            ],
            "seed": [str(validation.get("seed", "—"))],
            "execution": ["sequential_children_v1 · effective workers 1"],
        }
    )
    identity_block(
        st_module,
        "Baseline resolved section hash",
        str(baseline_step.get("baseline_section_config_hash") or ""),
    )
    st_module.markdown("**Charter satisfiability** (refused before freeze, with the reason)")
    report: CharterSatisfiabilityReport | None = None
    if resolved is None:
        render_empty_state(
            st_module, "purpose_unresolved", detail=resolution.resolution.reason
        )
    else:
        report = _satisfiability_for_draft(draft, resolved, resolution.flow)
        st_module.caption(report.configuration_sentence)
        st_module.table(
            {
                "rule": [rule.label for rule in report.rules],
                "result": ["✓ PASS" if rule.passed else "✕ FAIL" for rule in report.rules],
                "detail": [rule.detail for rule in report.rules],
            }
        )
        if report.failures:
            st_module.error(
                "This draft cannot freeze: "
                + "; ".join(f"{rule.rule_id} — {rule.detail}" for rule in report.failures)
            )
        if report.goal is StudyGoal.COMPARE_WITH_BASELINE and selections:
            baseline_values = {
                axis: SEARCH_AXIS_REGISTRY_V1[axis].baseline_value_id
                for axis in selections
                if axis in SEARCH_AXIS_REGISTRY_V1
            }
            challenger = {axis: values[0] for axis, values in selections.items()}
            st_module.markdown("**Challenger differences (every registered difference)**")
            for line in summarize_challenger_differences(baseline_values, challenger):
                st_module.write(f"- {line}")
    st_module.markdown("**Owner-authorization readiness (resolved for this path)**")
    if resolved is not None:
        st_module.write(
            f"Authorization class `{resolved.authorization_class}` · readiness "
            f"**{resolved.authorization_readiness}**"
            + (
                f" — {sanitize_error(resolved.authorization_detail)}"
                if resolved.authorization_detail
                else ""
            )
        )
        st_module.caption(
            "The computation-path-scoped requirement list with its live evidence "
            "state is on the Validation step; every unmet requirement fails the "
            "freeze closed."
        )
    st_module.markdown("**Estimated work — expensive vs cheap, separated**")
    st_module.table(
        {
            "Expensive strategy replays": [
                f"{estimate.full_replay_count} full sequential replays "
                f"(~{format_duration(estimate.estimated_runtime_seconds)})"
            ],
            "Verified reuse hits": [
                "resolved at launch (identity-deduped; no source path is "
                "constructed at preview)"
            ],
            "Cheaper downstream prop simulations": [
                f"{estimate.historical_prop_replays} historical/scenario prop "
                f"replays + {estimate.bootstrap_and_stress_simulations} "
                "bootstrap/stress simulations"
            ],
            "Estimated storage": [format_storage(estimate.estimated_storage_mb)],
            "New artifacts expected": [
                "core replays · memberships · costed evaluations · frontier"
            ],
            "Blocked by contract": [
                "S11 frozen model-gated replays (owner decision R-1) · MBP-1 "
                "research_only_offline · no worker parallelism (sequential V1)"
            ],
        }
    )
    typed = ""
    requires_acknowledgement = (
        resolved is not None and resolved.purpose is RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
    )
    if requires_acknowledgement:
        st_module.error(FULL_SCOPE_WARNING_TEXT)
        typed = st_module.text_input(
            f"Type '{FULL_SCOPE_ACKNOWLEDGEMENT}' to enable the launch control",
            value="",
            key=f"{_W}ack",
            help=help_text("wizard.full_scope_acknowledgement"),
        )
    return {
        "run_scope": resolved.run_scope if resolved is not None else validation.get("run_scope"),
        "purpose": resolved.purpose.value if resolved is not None else None,
        "typed_acknowledgement": typed,
        "required_acknowledgement": FULL_SCOPE_ACKNOWLEDGEMENT,
        "requires_acknowledgement": requires_acknowledgement,
        "n_children": n_children,
        "satisfiable": bool(report.passed) if report is not None else False,
        "freeze_block_reason": (
            resolved.freeze_block_reason
            if resolved is not None
            else resolution.resolution.reason
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Charter assembly + freeze/launch handlers
# ─────────────────────────────────────────────────────────────────────────────


def _commit_of(path: Path) -> str:
    """Read-only ``git rev-parse HEAD`` provenance query (DEV-R4-14).

    Returns ``"unknown"`` on failure; :func:`_freeze_and_launch` REFUSES to
    freeze a charter whose provenance resolved to ``"unknown"`` — two
    different source states must never share a charter identity through a
    silent fallback.
    """

    try:
        result = subprocess.run(  # noqa: S603, S607 — read-only git query
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 — surfaced as a freeze refusal
        return "unknown"


def _strategy_core_root() -> Path:
    """The Strategy-Core working tree, resolved from the INSTALLED package.

    The editable install points at the actual checkout (case-exact on every
    filesystem); walking up from ``strategy_core.__file__`` to the first
    directory containing ``.git`` avoids hardcoding a sibling folder name.
    """

    try:
        import strategy_core  # noqa: PLC0415

        candidate = Path(strategy_core.__file__).resolve()
        for parent in candidate.parents:
            if (parent / ".git").exists():
                return parent
    except Exception:  # noqa: BLE001 — fall through to the sibling guess
        pass
    return _REPO_ROOT.parent / "strategy-core"


def _assemble_charter(
    draft: StudyDraft,
    roots: Mapping[str, Any],
    resolution: DraftResolution | None = None,
) -> SearchCharterPayload:
    """Assemble the charter for the draft's RESOLVED purpose: the objective
    is never rewritten, the date policy follows the backend contract for the
    run scope, and the authorization is the class the ACTUAL path requires
    (synthetic marker / validated verification ref / owner bundle)."""

    if resolution is None:
        resolution = _resolve_for_draft(draft, roots)
    resolved = resolution.resolved
    if resolved is None:
        raise CharterValidationError(
            f"run purpose unresolved: {resolution.resolution.reason}"
        )
    flow = resolution.flow
    baseline = draft.step_payload("baseline")
    prop = _effective_payload(draft, flow, "prop_contracts")
    risk = _effective_payload(draft, flow, "risk_policies")
    benchmarks = _effective_payload(draft, flow, "benchmarks")
    validation = draft.step_payload("validation")
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: PLC0415
        AxisClassification,
        registry_sha256,
    )

    selections = _challenger_selections(draft, flow)
    axes: dict[str, tuple[str, ...]] = {}
    for axis_key, values in selections.items():
        spec = SEARCH_AXIS_REGISTRY_V1[axis_key]
        merged = tuple(
            dict.fromkeys((spec.baseline_value_id, *values))
        )  # baseline first, deduped, order-stable
        axes[axis_key] = merged
    pareto, tie_breaks = _resolved_objectives(draft)
    prop_selected = tuple(prop.get("selected_contract_ids") or ())
    strategy_gates = {
        **(benchmarks.get("strategy_gates") or {}),
    }
    strategy_gates.setdefault("min_session_stability_score", 0.5)
    prop_gate_values = benchmarks.get("prop_gates") or {}
    robustness_values = benchmarks.get("robustness_gates") or {}
    run_scope = resolved.run_scope
    if run_scope == "verification_5d":
        date_policy = DatePolicy(
            replay_dates=tuple(validation.get("real_dates") or PROPOSED_VERIFICATION_ALLOWLIST),
            warmup_dates=(),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        )
    else:
        evidence_dates = tuple(validation.get("real_dates") or ())
        date_policy = DatePolicy(
            replay_dates=(*FROZEN_WARMUP_DATES, *evidence_dates),
            warmup_dates=FROZEN_WARMUP_DATES,
            access_policy_id="development_explicit_dates_before_path_v2",
        )
    withdrawal_ids = tuple(
        sorted(
            {
                str(policy.get("withdrawal_policy"))
                for policy in (risk.get("per_firm_policies") or {}).values()
                if policy.get("withdrawal_policy")
            }
        )
    )
    # Every result-changing account-policy parameter enters the charter
    # identity: the risk-policy ids are CONTENT hashes of the complete
    # per-firm policy dicts (template + value + accounts + replacement), so
    # two charters differing in any parameter can never share an identity.
    from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: PLC0415
        canonical_contract_sha256,
    )

    risk_ids = tuple(
        sorted(
            canonical_contract_sha256(
                {
                    "firm_contract_id": contract_id,
                    "risk_template": policy.get("risk_template"),
                    "risk_value": policy.get("risk_value"),
                    "n_accounts": policy.get("n_accounts"),
                    "replacement_policy": policy.get("replacement_policy"),
                }
            )
            for contract_id, policy in (
                risk.get("per_firm_policies") or {}
            ).items()
        )
    )
    # Authorization by the ACTUAL computation path (plan F-01 / F-13): the
    # typed marker for a fully synthetic fixture only; the real ≤5-day slice
    # carries the bundle assembled from the persisted, verified
    # VerificationAuthorizationRef; research scopes carry the
    # computation-path-scoped owner bundle — each bound to THIS store's
    # verified namespace and current supersession head, or the freeze refuses.
    store_root = Path(resolution.roots["store_root"])
    owner_authorization: Any
    if resolved.authorization_class == "synthetic_marker":
        owner_authorization = SyntheticAuthorizationMarker()
    elif resolved.authorization_class == "verification_authorization_ref":
        bundle = (
            verification_owner_bundle(
                store_root, resolution.readiness, resolution.requirement_set
            )
            if resolution.readiness is not None and resolution.requirement_set is not None
            else None
        )
        if bundle is None:
            raise CharterValidationError(
                "the real verification slice requires a ready "
                "VerificationAuthorizationRef (readiness: "
                f"{resolved.authorization_readiness}): "
                f"{resolved.authorization_detail or 'no ready evidence'}"
            )
        owner_authorization = bundle
    else:
        bundle = (
            owner_authorization_bundle_from_store(
                store_root,
                resolution.requirement_set,
                expected_class=PURPOSE_NAMESPACE_CLASS[resolved.purpose],
            )
            if resolution.requirement_set is not None
            else None
        )
        if bundle is None:
            raise CharterValidationError(
                "owner authorization is not ready for this computation path "
                f"(readiness: {resolved.authorization_readiness}): "
                f"{resolved.authorization_detail or 'no ready evidence'}"
            )
        owner_authorization = bundle
    measured_only = tuple(
        key
        for key, spec in SEARCH_AXIS_REGISTRY_V1.items()
        if spec.classification is AxisClassification.MEASUREMENT_ONLY
    )
    blocked = tuple(
        key
        for key, spec in SEARCH_AXIS_REGISTRY_V1.items()
        if spec.classification is AxisClassification.BLOCKED
    )
    search_mode = _search_mode_for(draft, bool(axes))
    return SearchCharterPayload(
        search_mode=search_mode,
        baseline_profile_name=str(baseline.get("baseline_profile_name")),
        baseline_section_config_hash=str(
            baseline.get("baseline_section_config_hash") or ""
        ),
        axes=axes,
        locked_invariants_registry_sha256=registry_sha256(),
        measured_only_fields=measured_only,
        blocked_capabilities=blocked,
        authorized_firm_contract_ids=prop_selected,
        authorized_risk_policy_ids=risk_ids,
        authorized_withdrawal_policy_ids=withdrawal_ids,
        objective_policy=ObjectivePolicy(
            feasibility_gates=ResolvedStrategyGateThresholds(**strategy_gates),
            prop_feasibility_gates=ResolvedPropGateThresholds(**prop_gate_values)
            if prop_gate_values
            else ResolvedPropGateThresholds(**_PROP_GATE_DEFAULTS),
            robustness_gates=ResolvedRobustnessGateThresholds(**robustness_values)
            if robustness_values
            else ResolvedRobustnessGateThresholds(**_ROBUSTNESS_GATE_DEFAULTS),
            pareto_objectives=pareto,
            lexicographic_tie_breaks=tie_breaks,
        ),
        date_policy=date_policy,
        simulation_protocol=SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="historical_calendar_clock_v1",
        ),
        max_child_count=max(
            int(draft.step_payload("review").get("n_children", 1)),
            enumerate_child_count(selections) if selections else 1,
            1,
        ),
        seed=int(validation.get("seed", 7)),
        cost_policy=CostPolicy(),
        strategy_core_commit=_commit_of(_strategy_core_root()),
        quant_lab_commit=_commit_of(_REPO_ROOT),
        source_artifact_ids=(),
        owner_authorization=owner_authorization,
    )


def _wait_for_search_state(state_root: Path, search_id: str) -> dict[str, Any] | None:
    """The worker's persisted state within the wait window, else ``None``
    (silence is never reported as a started launch)."""

    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (  # noqa: PLC0415
        read_search_state,
    )

    deadline = time.monotonic() + float(LAUNCH_STATE_WAIT_SECONDS)
    while True:
        try:
            state = read_search_state(Path(state_root), search_id)
        except Exception:  # noqa: BLE001 — a torn write reads again on the next poll
            state = None
        if state is not None:
            return state
        if time.monotonic() >= deadline:
            return None
        time.sleep(_LAUNCH_POLL_SECONDS)


def _annotate_purpose(store_root: Path, search_id: str, resolved: ResolvedPurpose) -> None:
    """The mutable catalog purpose annotation of the frozen charter (never
    part of its identity; best effort — a catalog lock timeout is reported,
    not fatal)."""

    append_catalog_event(
        store_root,
        kind="purpose",
        artifact_id=search_id,
        payload={
            "purpose": resolved.purpose.value,
            "evidence_class": resolved.evidence_class.value,
            "run_scope": resolved.run_scope,
            "namespace_class": resolved.namespace_class,
        },
    )


def _freeze_and_launch(st_module, draft: StudyDraft, roots: Mapping[str, Any]) -> None:
    """The explicit launch-button handler — the only place work starts."""

    from ifvg_study_tab import request_route  # noqa: PLC0415

    draft_root = Path(roots["draft_root"])
    state_root = Path(roots["state_root"])
    try:
        resolution = _resolve_for_draft(draft, roots)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Purpose could not be resolved: {sanitize_error(error)}")
        return
    resolved = resolution.resolved
    if resolved is None:
        render_empty_state(
            st_module, "purpose_unresolved", detail=resolution.resolution.reason
        )
        return
    store_root = Path(resolution.roots["store_root"])
    if not resolved.freeze_allowed:
        state_id = (
            "store_namespace_unverified"
            if resolved.namespace_status != "verified"
            and resolved.authorization_class != "synthetic_marker"
            else "authorization_not_ready"
        )
        render_empty_state(st_module, state_id, detail=resolved.freeze_block_reason)
        return
    report = _satisfiability_for_draft(draft, resolved, resolution.flow)
    if not report.passed:
        st_module.error(
            "Charter cannot freeze (fail-closed): "
            + "; ".join(f"{rule.rule_id} — {rule.detail}" for rule in report.failures)
        )
        return
    try:
        payload = _assemble_charter(draft, resolution.roots, resolution=resolution)
        if "unknown" in (payload.strategy_core_commit, payload.quant_lab_commit):
            # DEV-R4-14: two source states must never share a charter
            # identity through a silent provenance fallback.
            raise CharterValidationError(
                "source-commit provenance could not be resolved (git "
                "unavailable?); freezing is refused rather than stamping "
                "'unknown' into the immutable charter"
            )
        validate_charter(
            payload,
            as_of_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            store_root=(
                None if resolved.authorization_class == "synthetic_marker" else store_root
            ),
        )
        envelope = SearchCharterEnvelope.from_payload(payload)
        save_charter(store_root, envelope)
    except CharterValidationError as error:
        st_module.error(f"Charter cannot freeze (fail-closed): {sanitize_error(error)}")
        return
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Freeze failed: {sanitize_error(error)}")
        return
    mark_frozen(draft_root, draft, search_id=envelope.search_id)
    try:
        _annotate_purpose(store_root, envelope.search_id, resolved)
    except Exception as error:  # noqa: BLE001 — annotation is non-semantic
        st_module.caption(f"purpose annotation not recorded: {sanitize_error(error)}")
    identity_block(st_module, "Frozen search charter id", envelope.search_id)
    entry_key = runner_entry_key_for_charter(envelope)
    if entry_key is None:
        render_empty_state(
            st_module,
            "runner_executor_planned",
            detail=(
                "this charter carries real full-development authorization; the "
                "operator full run is a separate, explicitly authorized action"
            ),
        )
        cli_escape_hatch(
            st_module,
            (
                "python scripts/ifvg_search_job.py start --search-id "
                f"{envelope.search_id} --runner-entry-key <registered key>"
            ),
            reason="explicit owner action once an executor is registered",
        )
        return
    # UI-1 (plan F-02): the registered runner is resolved BEFORE any spawn —
    # an unregistered key (the synthetic fixture key outside the development
    # checkout) is the typed runner_unavailable state, never a spawned worker
    # that exits before writing state
    try:
        resolve_registered_runner_entry(entry_key)
    except RunnerEntryError as error:
        render_empty_state(
            st_module,
            "runner_unavailable",
            detail=f"runner-entry key {entry_key!r}: {sanitize_error(error)}",
        )
        cli_escape_hatch(
            st_module,
            (
                "python scripts/ifvg_search_job.py start --search-id "
                f"{envelope.search_id} --runner-entry-key {entry_key}"
            ),
            reason="launch from a process that registers this executor",
        )
        return
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "ifvg_search_job.py"),
        "start",
        "--search-id",
        envelope.search_id,
        "--store-root",
        str(store_root),
        "--state-root",
        str(state_root),
        "--runner-entry-key",
        entry_key,
    ]
    try:
        pid = _spawn_search_job(command)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Detached launch failed: {sanitize_error(error)}")
        cli_escape_hatch(
            st_module,
            " ".join(
                [
                    "python",
                    "scripts/ifvg_search_job.py",
                    "start",
                    "--search-id",
                    envelope.search_id,
                    "--runner-entry-key",
                    entry_key,
                ]
            ),
            reason="the detached spawn failed",
        )
        return
    st_module.session_state[f"{STATE_PREFIX}monitor_search_id"] = envelope.search_id
    state = _wait_for_search_state(state_root, envelope.search_id)
    if state is None:
        # honest launch outcome: no persisted state → NOT reported as started
        render_empty_state(
            st_module,
            "launch_not_started",
            detail=(
                f"pid {pid}; no search_state.json within {LAUNCH_STATE_WAIT_SECONDS:.0f} s; "
                f"job log: data/ifvg_search_jobs/{envelope.search_id[:12]}…/job.log"
            ),
        )
        cli_escape_hatch(
            st_module,
            f"python scripts/ifvg_search_job.py status --search-id {envelope.search_id}",
            reason="check whether the worker persisted state",
        )
        return
    st_module.success(
        f"Search launched detached (pid {pid}); state persisted "
        f"(phase {state.get('phase', 'unknown')}). Routing to Active Runs."
    )
    request_route(st_module, "active_runs")
    st_module.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Wizard shell
# ─────────────────────────────────────────────────────────────────────────────

_STEP_VALIDATORS_BY_KEY = {
    "objective": validate_objective_step,
    "baseline": validate_baseline_step,
    "search_space": validate_search_space_step,
    "prop_contracts": validate_prop_step,
    "risk_policies": validate_risk_step,
    "benchmarks": validate_benchmarks_step,
    "validation": validate_validation_step,
    "review": validate_review_step,
}


def _normalized(value: Any) -> Any:
    """Tuples become lists (the shape a draft takes after its JSON round trip)
    so a persisted payload compares equal to the freshly collected fields."""

    if isinstance(value, dict):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalized(item) for item in value]
    return value


def _fields_equal(left: Any, right: Any) -> bool:
    return _normalized(left) == _normalized(right)


def _render_step_body(
    st_module,
    step_key: str,
    draft: StudyDraft,
    purpose_roots: Mapping[str, Any],
    resolution: DraftResolution,
) -> dict[str, Any]:
    if step_key == "objective":
        return _step_objective(st_module, draft)
    if step_key == "baseline":
        return _step_baseline(st_module, draft)
    if step_key == "search_space":
        return _step_search_space(st_module, draft)
    if step_key == "prop_contracts":
        return _step_prop(st_module, draft, purpose_roots)
    if step_key == "risk_policies":
        return _step_risk(st_module, draft)
    if step_key == "benchmarks":
        return _step_benchmarks(st_module, draft, resolution.flow)
    if step_key == "validation":
        return _step_validation(st_module, draft, purpose_roots, resolution)
    return _step_review(st_module, draft, purpose_roots, resolution)


def render_new_study(st_module=st, *, roots: Mapping[str, Any]) -> None:
    dev_only_badge(st_module)
    draft = _draft_header(st_module, roots)
    if draft is None:
        return
    # Cross-draft widget hygiene (FUX-WIZ-002 exact restore): step widgets
    # use fixed keys, so on a draft SWITCH every not-yet-instantiated
    # step-widget key is purged — draft B can never inherit draft A's
    # mounted widget values (the header widgets already rendered this run
    # and keep their own state).
    if st_module.session_state.get(_ACTIVE_DRAFT_KEY) != draft.draft_id:
        for key in list(st_module.session_state.keys()):
            if key.startswith(_W) and key not in _HEADER_WIDGET_KEYS:
                del st_module.session_state[key]
        st_module.session_state[_ACTIVE_DRAFT_KEY] = draft.draft_id
    draft_root = Path(roots["draft_root"])
    persisted = _draft_is_persisted(draft_root, draft)
    try:
        resolution = _resolve_for_draft(draft, roots)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Purpose could not be resolved: {sanitize_error(error)}")
        return
    _purpose_card(st_module, draft, draft_root, resolution)
    purpose_roots = resolution.roots
    flow = resolution.flow
    if flow is not None:
        step_keys: tuple[str, ...] = flow.step_keys
        titles = [step.title for step in flow.steps]
        position = restore_step_index(
            flow, stored_step_key=draft.current_step_key, legacy_step_index=draft.step_index
        )
    else:
        step_keys = STEP_KEYS
        titles = list(WIZARD_STEP_TITLES)
        position = min(max(int(draft.step_index), 0), len(STEP_KEYS) - 1)
        st_module.caption(
            "Every step is shown until the run purpose is confirmed (the goal-derived "
            "flow needs the purpose)."
        )
    step_key = step_keys[position]
    n_steps = len(step_keys)
    st_module.progress(
        (position + 1) / n_steps,
        text=f"Step {position + 1} of {n_steps}: {titles[position]}",
    )
    st_module.caption(
        f"Step {position + 1} of {n_steps} · "
        + " › ".join(
            f"**{title}**" if index == position else title
            for index, title in enumerate(titles)
        )
    )
    stored_name = draft.display_name
    new_name = st_module.text_input(
        "Draft name",
        value=draft.display_name,
        key=f"{_W}name",
        help=(
            "Required before the first save. The proposed default is "
            "'<goal> — <baseline short name> — <date>'."
        ),
    )
    draft.display_name = (new_name or "").strip()
    if not persisted:
        _saved_chip(st_module, "session", draft)

    fields = _render_step_body(st_module, step_key, draft, purpose_roots, resolution)
    errors = _STEP_VALIDATORS_BY_KEY[step_key](fields)
    for field, message in errors.items():
        st_module.error(f"{field}: {message}")

    if persisted:
        # owner Q2: after the first persistence every change autosaves (a
        # legacy draft without a stored step key is not "changed" by itself)
        changed = (
            not _fields_equal(draft.steps.get(step_key), fields)
            or draft.display_name != stored_name
            or (draft.current_step_key is not None and draft.current_step_key != step_key)
        )
        if changed and draft.display_name:
            draft.steps[step_key] = dict(fields)
            draft.current_step_key = step_key
            draft.step_index = position
            try:
                save_draft(draft_root, draft)
            except DraftError as error:
                st_module.warning(f"Autosave failed: {sanitize_error(error)}")
            else:
                _saved_chip(st_module, "autosaved", draft)
        elif changed:
            st_module.error(
                "Draft name required — the change is not saved until the draft is named."
            )
        else:
            _saved_chip(st_module, "saved", draft)
    else:
        _duplicate_warning(st_module, draft, draft_root, step_key=step_key, fields=fields)
        _stash_session_draft(st_module, draft)

    columns = st_module.columns([1, 1, 1, 2])
    with columns[0]:
        if st_module.button("Save Draft", key=f"{_W}save"):
            draft.steps[step_key] = dict(fields)
            draft.current_step_key = step_key
            draft.step_index = position
            if _persist_now(st_module, draft_root, draft):
                st_module.success("Draft saved.")
                st_module.rerun()
    with columns[1]:
        if position > 0 and st_module.button("Back", key=f"{_W}back"):
            # Back never discards valid data: persist the visible fields.
            draft.steps[step_key] = dict(fields)
            draft.current_step_key = step_keys[position - 1]
            draft.step_index = position - 1
            if _persist_now(st_module, draft_root, draft):
                st_module.rerun()
    with columns[2]:
        if position < n_steps - 1 and st_module.button(
            "Next", key=f"{_W}next", disabled=bool(errors)
        ):
            draft.steps[step_key] = dict(fields)
            draft.current_step_key = step_keys[position + 1]
            draft.step_index = position + 1
            if _persist_now(st_module, draft_root, draft):  # the first valid Next persists
                st_module.rerun()
    with columns[3]:
        if step_key == "review":
            mode = next(
                (m for m in STUDY_MODES if m.mode_id == draft.mode_id),
                STUDY_MODES[0],
            )
            if mode.search_mode is None:
                st_module.caption(
                    "Full Pipeline Run drafts freeze and launch from the "
                    "operator workflow below (Configure / Preview / Launch / "
                    "Monitor / Resume-Retry / Publish)."
                )
            elif st_module.button(
                "Freeze Search Charter and Launch",
                key=f"{_W}freeze",
                disabled=bool(errors),
                type="primary",
                help=(
                    "Freezes the immutable charter into the purpose's store, "
                    "resolves the registered executor before any spawn, and "
                    "reports the launch only after the worker persisted state."
                ),
            ):
                draft.steps[step_key] = dict(fields)
                draft.current_step_key = step_key
                draft.step_index = position
                if _persist_now(st_module, draft_root, draft):
                    _freeze_and_launch(st_module, draft, roots)
    if step_key == "review":
        mode = next(
            (m for m in STUDY_MODES if m.mode_id == draft.mode_id),
            STUDY_MODES[0],
        )
        if mode.search_mode is None:
            # R5: the standardized §30 operator workflow replaces the R4-era
            # planned-capability state; the draft's visible fields persist
            # before the surface renders so Launch assembles the same state.
            draft.steps[step_key] = dict(fields)
            draft.current_step_key = step_key
            draft.step_index = position
            if _persist_now(st_module, draft_root, draft):
                from ifvg_pipeline_tab import render_pipeline_run  # noqa: PLC0415

                render_pipeline_run(st_module, roots=purpose_roots, draft=draft)
