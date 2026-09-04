"""Experiments sub-navigation router for the study workspace (R4; FUX §3 as
amended by UI-1).

``render_ifvg_study_tab`` owns the session-state-backed horizontal radio:

    Start | Verify Implementation | New Study | Active Runs | Results | History
    | Context Research

Only the selected sub-surface executes (hidden panels never poll jobs or
build charts — FUX §3.3); ``Context Research`` delegates to the existing
M0–M3 experiments panel verbatim (FUX-IA-003); programmatic navigation
(wizard freeze/launch → Active Runs; monitor row → Results; History clone →
New Study; a Start card → New Study / Verify Implementation) goes through
:func:`request_route`, applied before the radio is instantiated on the
next run.

UI-1 (plan F-01 / owner Q1): the mutable "Artifact namespace" radio is
GONE. The semantic namespace derives from the draft's run purpose
(Implementation Verification → the ``test`` store ``search_test/v1``;
Development Research / Full Authorized Development → the ``research`` store
``search/v1``) and is displayed read-only with the store's VERIFIED
``store_namespace_id``. The module-level roots below are operational
locations only — a local path never defines authority.

Import launches nothing; every root is a module attribute so tests inject
temporary directories.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_ui_common import (
    STATE_PREFIX,
    dev_only_badge,
    identity_block,
    render_empty_state,
    sanitize_error,
    sanitize_select,
    verification_badge,
)

from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    PURPOSE_DESCRIPTIONS,
    PURPOSE_LABELS,
    PURPOSE_NAMESPACE_CLASS,
    PURPOSE_RUN_SCOPE,
    RunPurpose,
    RunPurposeAnnotation,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SEARCH_STORE_ROOT,
    SEARCH_TEST_STORE_ROOT,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    STUDY_DRAFT_ROOT,
    new_draft,
    save_draft,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (
    ROUTE_LABELS,
    StudyWorkspaceRoute,
)

__all__ = [
    "render_ifvg_study_tab",
    "request_route",
    "ROUTE_KEY",
    "workspace_roots",
    "roots_for_purpose",
    "store_root_for_namespace_class",
    "TASK_CARDS",
    "TaskCard",
    "start_draft_from_card",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: Module-level roots — tests monkeypatch these with tmp directories. They
#: are OPERATIONAL locations; the semantic class of each store is verified
#: from its STORE_NAMESPACE.json envelope, never inferred from the path.
STORE_ROOT_RESEARCH = _REPO_ROOT / SEARCH_STORE_ROOT
STORE_ROOT_VERIFICATION = _REPO_ROOT / SEARCH_TEST_STORE_ROOT
STATE_ROOT = _REPO_ROOT / "data/ifvg_search_jobs"
DRAFT_ROOT = _REPO_ROOT / STUDY_DRAFT_ROOT

ROUTE_KEY = f"{STATE_PREFIX}route"
_PENDING_ROUTE_KEY = f"{STATE_PREFIX}pending_route"
_DRAFT_KEY = f"{STATE_PREFIX}draft_id"
_START = f"{STATE_PREFIX}start_"


def request_route(st_module, route: StudyWorkspaceRoute | str) -> None:
    """Queue a programmatic sub-route change (applied on the next run)."""

    st_module.session_state[_PENDING_ROUTE_KEY] = ROUTE_LABELS[
        StudyWorkspaceRoute(route)
    ]


def store_root_for_namespace_class(namespace_class: str) -> Path:
    """The operational root deployed for a semantic namespace CLASS."""

    return STORE_ROOT_RESEARCH if namespace_class == "research" else STORE_ROOT_VERIFICATION


def workspace_roots(st_module=None, *, namespace_class: str | None = None) -> dict[str, Any]:
    """The operational roots the routes consume.

    ``store_root`` is the DEFAULT read location for listings and catalog
    annotations (the research root unless a class is named); ``store_roots``
    carries both deployed stores by class so listings annotate every run with
    the store that actually holds its charter. Where a charter freezes or
    launches is decided ONLY by :func:`roots_for_purpose` over the draft's
    resolved purpose — never by these paths and never by a session selector.
    """

    default_class = namespace_class or "research"
    return {
        "store_root": store_root_for_namespace_class(default_class),
        "store_roots": {
            "research": STORE_ROOT_RESEARCH,
            "test": STORE_ROOT_VERIFICATION,
        },
        "state_root": STATE_ROOT,
        "draft_root": DRAFT_ROOT,
        "namespace_class": default_class,
    }


def roots_for_purpose(roots: Mapping[str, Any], purpose: RunPurpose | str) -> dict[str, Any]:
    """The roots a draft of ``purpose`` freezes / launches / reads under:
    the store of the purpose's semantic namespace CLASS (owner Q1)."""

    namespace_class = PURPOSE_NAMESPACE_CLASS[RunPurpose(purpose)]
    stores = dict(roots.get("store_roots") or {})
    store_root = stores.get(namespace_class) or store_root_for_namespace_class(namespace_class)
    return {**roots, "store_root": Path(store_root), "namespace_class": namespace_class}


# ─────────────────────────────────────────────────────────────────────────────
# Start — task-oriented entry (plan §5.2)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TaskCard:
    card_id: str
    title: str
    purpose: RunPurpose | None
    mode_id: str | None
    question_id: str | None
    template_id: str | None
    route: StudyWorkspaceRoute | None
    what_runs: str
    approvals: str
    #: selectable | owner_unratified | research_only_offline | blocked_dependency
    #: | planned_post_v1 | other_tab
    availability: str
    availability_reason: str


TASK_CARDS: tuple[TaskCard, ...] = (
    TaskCard(
        "verify_implementation",
        "Verify the current implementation",
        RunPurpose.IMPLEMENTATION_VERIFICATION,
        "single_configuration",
        "evaluate_one_configuration",
        "strategy_quality_only",
        StudyWorkspaceRoute.VERIFY_IMPLEMENTATION,
        "the exact baseline over a synthetic fixture (machinery proof) or, with the "
        "owner's VerificationAuthorizationRef, the real ≤5-day slice — verification "
        "gates only, never a research result",
        "none for a synthetic fixture; SeedProductionAuthorizationRef then "
        "VerificationAuthorizationRef for the real slice",
        "selectable",
        "implemented and selectable",
    ),
    TaskCard(
        "review_setups",
        "Review strategy setups and trades",
        None,
        None,
        None,
        None,
        None,
        "no run — exact setup / trade inspection of a verified pair",
        "none",
        "other_tab",
        "open the Replay / Verifier tab",
    ),
    TaskCard(
        "evaluate_one",
        "Evaluate one configuration",
        RunPurpose.DEVELOPMENT_RESEARCH,
        "single_configuration",
        "evaluate_one_configuration",
        "strategy_quality_only",
        StudyWorkspaceRoute.NEW_STUDY,
        "exactly one frozen profile (baseline or another eligible anchor); no "
        "comparison or delta claim",
        "computation-path-scoped owner authorization bundle",
        "selectable",
        "implemented; freezing waits for a ready owner authorization",
    ),
    TaskCard(
        "compare_with_baseline",
        "Compare one configuration with the baseline",
        RunPurpose.DEVELOPMENT_RESEARCH,
        "single_configuration",
        "compare_one_with_baseline",
        "strategy_quality_only",
        StudyWorkspaceRoute.NEW_STUDY,
        "exactly one baseline + exactly one challenger configuration; deltas only "
        "when populations, policies and evidence are compatible",
        "computation-path-scoped owner authorization bundle",
        "selectable",
        "implemented; freezing waits for a ready owner authorization",
    ),
    TaskCard(
        "fsm_search",
        "Search FSM parameters",
        RunPurpose.DEVELOPMENT_RESEARCH,
        "fsm_config_search",
        "find_robust_fsm",
        "strategy_quality_only",
        StudyWorkspaceRoute.NEW_STUDY,
        "≥ 1 registered axis, ≥ 2 profiles (recommended preset: parent-retest "
        "timeout unbounded / 240 / 360 / 480), one sequential replay per profile",
        "owner decisions 1 / 2 / 7 / R-2 (axes, values, thresholds, workflow)",
        "selectable",
        "implemented; freezing waits for a ready owner authorization",
    ),
    TaskCard(
        "feature_model_evidence",
        "Evaluate feature and model evidence",
        RunPurpose.DEVELOPMENT_RESEARCH,
        "full_pipeline_run",
        "find_robust_fsm",
        "strategy_quality_only",
        StudyWorkspaceRoute.NEW_STUDY,
        "the pipeline's feature / label / fold / model stages (S05–S10) over a "
        "bundle and the supervised ladder; S11 stays blocked by contract",
        "owner authorization bundle; regime decisions when a regime study is added",
        "selectable",
        "implemented; S11 model-gated replays remain blocked (owner decision R-1)",
    ),
    TaskCard(
        "prop_feasibility",
        "Test prop-firm feasibility",
        RunPurpose.DEVELOPMENT_RESEARCH,
        "prop_benchmark",
        "repeat_payout_feasibility",
        "payout_reliability",
        StudyWorkspaceRoute.NEW_STUDY,
        "Prop Benchmark (≥ 1 first_party_verified firm) or Universal Prop Search "
        "(≥ 2 firms) over the strategy's executed trades",
        "owner decisions 5 / 6 / 8 / R-3 / R-4 / 10 plus first-party-verified contracts",
        "blocked_dependency",
        "no first_party_verified firm contract exists yet (synthetic fixtures are "
        "visibly synthetic and confined to verification)",
    ),
    TaskCard(
        "advanced_end_to_end",
        "Run an advanced end-to-end study",
        RunPurpose.FULL_AUTHORIZED_DEVELOPMENT,
        "full_pipeline_run",
        "find_robust_fsm",
        "strategy_quality_only",
        StudyWorkspaceRoute.NEW_STUDY,
        "the full owner-authorized 16-stage plan (S11 blocked); publication is a "
        "separate eligibility gate",
        "the complete computation-path-scoped owner authorization bundle plus the "
        "typed full-scope acknowledgement",
        "owner_unratified",
        "visible-disabled until the typed owner authorization readiness is ready",
    ),
    TaskCard(
        "inspect_health",
        "Inspect system health and audit evidence",
        None,
        None,
        None,
        None,
        None,
        "no run — artifact integrity, pairing, access, capacity and performance "
        "evidence",
        "none",
        "other_tab",
        "open the Data & Audit tab",
    ),
)

_AVAILABILITY_CHIPS: Mapping[str, str] = {
    "selectable": "✓ Implemented and selectable",
    "owner_unratified": "▲ Implemented — owner authorization not ready",
    "research_only_offline": "ℹ Research-only offline",
    "blocked_dependency": "⛔ Blocked by a missing dependency",
    "planned_post_v1": "○ Planned post-V1",
    "other_tab": "ℹ Available on another tab",
}


def start_draft_from_card(card: TaskCard, roots: Mapping[str, Any], *, now: str | None = None):
    """Create the draft a Start card describes: the card-selected purpose is
    recorded immediately as the presentation annotation (owner Q1 / §5.6)
    and the objective step is preset from the card."""

    if card.purpose is None or card.mode_id is None:
        raise ValueError(f"task card {card.card_id!r} does not create a draft")
    stamp = now or datetime.now(UTC).isoformat(timespec="seconds")
    draft = new_draft(card.mode_id, display_name=f"{card.title} ({stamp[:10]})")
    draft.purpose_annotation = RunPurposeAnnotation(
        purpose=card.purpose,
        derivation="card_selected",
        owner_confirmed=True,
        updated_at=stamp,
    ).to_dict()
    draft.steps["objective"] = {
        "mode_id": card.mode_id,
        "question_id": card.question_id,
        "template_id": card.template_id,
        "custom_objectives": (),
    }
    draft.steps["validation"] = {
        "run_scope": PURPOSE_RUN_SCOPE[card.purpose],
        "evidence_class": (
            "synthetic_fixture"
            if card.purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
            else "real"
        ),
        "worker_limit": 1,
    }
    save_draft(Path(roots["draft_root"]), draft)
    return draft


def _card_availability(card: TaskCard, roots: Mapping[str, Any]) -> tuple[str, str]:
    """The live availability chip + reason (typed readiness for the full
    plan; verified-contract presence for prop cards; static otherwise)."""

    if card.card_id == "advanced_end_to_end":
        from alpha_lab.agents.data_infra.ifvg.search.authorization import (  # noqa: PLC0415
            derive_authorization_requirements,
        )
        from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
            owner_authorization_readiness,
        )

        try:
            requirement_set = derive_authorization_requirements(
                "full_authorized_development", (), None, (), ()
            )
            readiness = owner_authorization_readiness(
                Path(roots_for_purpose(roots, card.purpose)["store_root"]), requirement_set
            )
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            return "owner_unratified", f"readiness lookup failed: {sanitize_error(error)}"
        if readiness.status == "ready":
            return "selectable", "owner authorization ready"
        return "owner_unratified", f"authorization readiness: {readiness.status} — " + (
            sanitize_error(readiness.detail)
        )
    if card.card_id == "prop_feasibility":
        from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
            load_contract_summaries,
        )

        try:
            cards = load_contract_summaries(
                Path(roots_for_purpose(roots, card.purpose)["store_root"])
            )
        except Exception as error:  # noqa: BLE001
            return "blocked_dependency", f"contract lookup failed: {sanitize_error(error)}"
        verified = [
            contract
            for contract in cards
            if contract.get("launchable")
            and contract.get("verification_status") != "synthetic_fixture_verified"
        ]
        if verified:
            return "selectable", f"{len(verified)} first_party_verified contract(s) available"
        return "blocked_dependency", card.availability_reason
    return card.availability, card.availability_reason


def _render_start(st_module, roots: Mapping[str, Any]) -> None:
    st_module.subheader("What are you trying to do?")
    st_module.caption(
        "Each card derives the run purpose, the study family, the stage plan and the "
        "semantic namespace. Purpose is a presentation classification; the run scope, "
        "namespace, stage plan and owner authorization are validated by the backend "
        "contracts and never bypassed."
    )
    for card in TASK_CARDS:
        availability, reason = _card_availability(card, roots)
        with st_module.container(border=True):
            st_module.markdown(f"**{card.title}**")
            if card.purpose is not None:
                purpose_roots = roots_for_purpose(roots, card.purpose)
                st_module.caption(
                    f"Purpose: **{PURPOSE_LABELS[card.purpose]}** · run scope "
                    f"`{PURPOSE_RUN_SCOPE[card.purpose]}` · namespace class "
                    f"`{purpose_roots['namespace_class']}`"
                )
                _namespace_line(st_module, Path(purpose_roots["store_root"]), card.purpose)
            st_module.write(f"What will run: {card.what_runs}.")
            st_module.write(f"Owner approvals needed: {card.approvals}.")
            st_module.write(f"Availability: {_AVAILABILITY_CHIPS[availability]} — {reason}.")
            if card.route is StudyWorkspaceRoute.VERIFY_IMPLEMENTATION:
                if st_module.button("Open the Verification Center", key=f"{_START}{card.card_id}"):
                    request_route(st_module, StudyWorkspaceRoute.VERIFY_IMPLEMENTATION)
                    st_module.rerun()
            elif card.route is StudyWorkspaceRoute.NEW_STUDY and st_module.button(
                "Start this study",
                key=f"{_START}{card.card_id}",
                disabled=availability in ("blocked_dependency", "planned_post_v1"),
                help=(
                    "Creates a draft carrying this card's purpose and objective "
                    "preset; nothing runs until Review & Launch."
                ),
            ):
                try:
                    draft = start_draft_from_card(card, roots)
                except Exception as error:  # noqa: BLE001 — sanitized surface only
                    st_module.error(f"Draft could not be created: {sanitize_error(error)}")
                else:
                    st_module.session_state[_DRAFT_KEY] = draft.draft_id
                    request_route(st_module, StudyWorkspaceRoute.NEW_STUDY)
                    st_module.rerun()


def _namespace_line(st_module, store_root: Path, purpose: RunPurpose) -> None:
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        resolve_store_namespace,
    )

    expected = PURPOSE_NAMESPACE_CLASS[purpose]
    try:
        state = resolve_store_namespace(store_root, expected_class=expected)
    except Exception as error:  # noqa: BLE001
        st_module.caption(f"Store namespace: unavailable ({sanitize_error(error)})")
        return
    if state.status == "verified":
        st_module.caption(
            f"Store namespace: verified `{state.namespace_class}` · "
            f"id `{(state.store_namespace_id or '')[:16]}…`"
        )
    else:
        st_module.caption(
            f"Store namespace: **{state.status}** — {sanitize_error(state.detail)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verify Implementation — the Verification Center readiness surface (UI-1
# lands the purpose / namespace / typed-authorization readiness; the complete
# fixture → seed → final-authorization → run flow lands with UI-2)
# ─────────────────────────────────────────────────────────────────────────────


def _render_verify_implementation(st_module, roots: Mapping[str, Any]) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.authorization import (  # noqa: PLC0415
        derive_authorization_requirements,
    )
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        resolve_store_namespace,
        verification_authorization_readiness,
    )

    purpose = RunPurpose.IMPLEMENTATION_VERIFICATION
    purpose_roots = roots_for_purpose(roots, purpose)
    store_root = Path(purpose_roots["store_root"])
    st_module.subheader("Verification Center")
    verification_badge(st_module)
    st_module.caption(PURPOSE_DESCRIPTIONS[purpose])
    st_module.markdown("**Readiness**")
    try:
        namespace = resolve_store_namespace(store_root, expected_class="test")
    except Exception as error:  # noqa: BLE001
        st_module.error(f"Namespace lookup failed: {sanitize_error(error)}")
        return
    st_module.write(
        f"1. Semantic store namespace: **{namespace.status}** — "
        f"{sanitize_error(namespace.detail)}"
    )
    if namespace.store_namespace_id:
        identity_block(st_module, "store_namespace_id", namespace.store_namespace_id)
    try:
        readiness = verification_authorization_readiness(store_root)
    except Exception as error:  # noqa: BLE001
        st_module.error(f"Readiness lookup failed: {sanitize_error(error)}")
        return
    st_module.write(
        f"2. Final verification authorization (VerificationAuthorizationRef): "
        f"**{readiness.status}** — {sanitize_error(readiness.detail)}"
    )
    for run_id in readiness.evidence_ids:
        identity_block(st_module, "verification_run_id", run_id)
    st_module.write(
        "3. Logical trading-day fixture, seed-production authorization, seed job and "
        "verified seed: the complete state-driven flow lands with UI-2; until then the "
        "owner steps run through `scripts/ifvg_verification_window_shortlist.py`, "
        "`scripts/ifvg_seed_production.py` and `scripts/ifvg_bounded_verification.py` "
        "(exact commands and packets; nothing here signs or produces a seed)."
    )
    requirement_set = derive_authorization_requirements("verification_5d", (), None, (), ())
    st_module.markdown("**Authorization requirement (real slice)**")
    for requirement in requirement_set.payload.requirements:
        st_module.write(f"- `{requirement.decision_key}` — {requirement.reason}")
    if readiness.status in ("missing", "store_unmarked"):
        render_empty_state(
            st_module,
            "verification_authorization_missing",
            detail=f"{readiness.status}: {readiness.detail}",
        )
    elif readiness.status != "ready":
        render_empty_state(st_module, "authorization_not_ready", detail=readiness.detail)
    st_module.markdown("**Start a verification draft**")
    st_module.caption(
        "A synthetic fixture proves the machinery in the test namespace and needs "
        "no owner authorization; the real ≤5-day slice is selectable on the draft's "
        "Validation step only when the readiness above is ready."
    )
    card = next(card for card in TASK_CARDS if card.card_id == "verify_implementation")
    if st_module.button("Start a verification draft", key=f"{_START}verify_draft"):
        try:
            draft = start_draft_from_card(card, roots)
        except Exception as error:  # noqa: BLE001
            st_module.error(f"Draft could not be created: {sanitize_error(error)}")
        else:
            st_module.session_state[_DRAFT_KEY] = draft.draft_id
            request_route(st_module, StudyWorkspaceRoute.NEW_STUDY)
            st_module.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────


def render_ifvg_study_tab(st_module=st, context_research=None) -> None:
    """The Experiments surface: sub-nav radio + exactly one executed route."""

    route_labels = [ROUTE_LABELS[route] for route in StudyWorkspaceRoute]
    pending = st_module.session_state.pop(_PENDING_ROUTE_KEY, None)
    if pending in route_labels:
        st_module.session_state[ROUTE_KEY] = pending
    sanitize_select(st_module, ROUTE_KEY, route_labels)
    selected = st_module.radio(
        "Experiments workspace",
        route_labels,
        horizontal=True,
        key=ROUTE_KEY,
        help=(
            "Start derives a purpose from what you are trying to do; Verify "
            "Implementation is the Verification Center; the remaining routes keep "
            "their R4 meaning. Only the selected route executes."
        ),
    )
    if selected == ROUTE_LABELS[StudyWorkspaceRoute.CONTEXT_RESEARCH]:
        if context_research is None:
            from ifvg_lab_tab import render_ifvg_experiments_tab  # noqa: PLC0415

            context_research = render_ifvg_experiments_tab
        context_research(st_module)
        return

    roots = workspace_roots(st_module)
    if selected == ROUTE_LABELS[StudyWorkspaceRoute.START]:
        dev_only_badge(st_module)
        _render_start(st_module, roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.VERIFY_IMPLEMENTATION]:
        dev_only_badge(st_module)
        _render_verify_implementation(st_module, roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.NEW_STUDY]:
        from ifvg_study_wizard import render_new_study  # noqa: PLC0415

        render_new_study(st_module, roots=roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.ACTIVE_RUNS]:
        from ifvg_active_runs_tab import render_active_runs  # noqa: PLC0415

        render_active_runs(st_module, roots=roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.RESULTS]:
        from ifvg_results_tab import render_results  # noqa: PLC0415

        render_results(st_module, roots=roots)
    elif selected == ROUTE_LABELS[StudyWorkspaceRoute.HISTORY]:
        from ifvg_results_tab import render_history  # noqa: PLC0415

        render_history(st_module, roots=roots)
