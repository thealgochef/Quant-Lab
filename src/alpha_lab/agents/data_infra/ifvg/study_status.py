"""Presentation status contracts for the FSM/prop study workspace (R4).

Implements ``CONTRACTS_AND_SCHEMAS.md`` §13 of
``ifvg_prop_robust_config_search_v1``: the internal status keys, visible
labels, glyphs, accessible help text, result-scope captions, empty/blocked/
failure-state presentations, and the forbidden/required wording the
``FRONTEND_UX_CONTRACT.md`` scans enforce.

These contracts are non-research-bearing presentation data: they contain no
Streamlit objects, produce no identities, and can never change or replace an
artifact identity. Every status renders as glyph + visible word/phrase +
high-contrast semantic class — color is never the sole carrier of meaning
(FUX §5.4).
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from .search.identities import FrozenContract

__all__ = [
    "StudyWorkspaceRoute",
    "DisclosureLevel",
    "ResultScope",
    "StudyStatusKey",
    "StatusPresentation",
    "EmptyStateKey",
    "EmptyStatePresentation",
    "STATUS_PRESENTATIONS",
    "EMPTY_STATE_PRESENTATIONS",
    "RESULT_SCOPE_LABELS",
    "ROUTE_LABELS",
    "HEATMAP_GLYPHS",
    "TIMELINE_MARKERS",
    "DEV_BADGE_TEXT",
    "VERIFICATION_BADGE_TEXT",
    "NO_PASS_SENTENCE",
    "NOT_RUN_STRATEGY_GATE_TEXT",
    "FULL_SCOPE_WARNING_TEXT",
    "FULL_SCOPE_ACKNOWLEDGEMENT",
    "FORBIDDEN_DISPLAY_PHRASES",
    "status_presentation",
]


class StudyWorkspaceRoute(StrEnum):
    """The Experiments sub-navigation routes (FUX §3.2 as amended by UI-1).

    UI-1 adds the task-oriented ``Start`` entry and the dedicated
    ``Verify Implementation`` route (the Verification Center); the existing
    route ids are unchanged so programmatic ``request_route`` callers keep
    working. The mutable "Artifact namespace" selector is gone — the
    namespace derives from the run purpose (owner Q1).
    """

    START = "start"
    VERIFY_IMPLEMENTATION = "verify_implementation"
    NEW_STUDY = "new_study"
    ACTIVE_RUNS = "active_runs"
    RESULTS = "results"
    HISTORY = "history"
    CONTEXT_RESEARCH = "context_research"


class DisclosureLevel(StrEnum):
    """Progressive-disclosure levels every result page supports (FUX §5.1)."""

    SUMMARY = "summary"
    ANALYST = "analyst"
    AUDIT = "audit"


class ResultScope(StrEnum):
    """Mandatory scope label on every metric block and chart (FUX §5.2)."""

    CANDIDATE_RESEARCH = "candidate_research"
    ACTUAL_EXECUTED_STRATEGY = "actual_executed_strategy"
    PROP_HISTORICAL_CLOSED_TRADE = "prop_historical_closed_trade"
    PROP_1M_SCENARIO = "prop_1m_scenario"
    PROP_ORDERED_EVENT_REPLAY = "prop_ordered_event_replay"
    BOOTSTRAP_SIMULATION = "bootstrap_simulation"
    STRESS_SIMULATION = "stress_simulation"


class StudyStatusKey(StrEnum):
    """Internal status keys (CS §13). The visible text is a presentation."""

    DRAFT = "draft"
    FROZEN = "frozen"
    QUEUED = "queued"
    RUNNING = "running"
    REPLAY_FAILED = "replay_failed"
    STRATEGY_REJECTED = "strategy_rejected"
    PROP_REJECTED = "prop_rejected"
    ROBUST_FINALIST = "robust_finalist"
    SELECTED_REPRESENTATIVE = "selected_representative"
    SUPERSEDED = "superseded"
    BLOCKED = "blocked"


class StatusPresentation(FrozenContract):
    status_key: StudyStatusKey
    visible_label: str
    glyph: str
    semantic_class: str
    help_text: str


class EmptyStateKey(StrEnum):
    """Intentional empty/blocked/failure states (CS §13 + FUX §31).

    ``BROWSER_QA_UNAVAILABLE`` is an additive member beyond the CS §13
    enumeration: FUX §31 requires the UI to render that state intentionally
    (recorded as an R4 schema-additive deviation, same rule as DEV-R3-8).
    """

    NO_CONFIGURATIONS_PASS = "no_configurations_pass"
    NO_VERIFIED_FIRM_CONTRACT = "no_verified_firm_contract"
    BLOCKED_SEARCH_AXIS = "blocked_search_axis"
    INSUFFICIENT_SAMPLE = "insufficient_sample"
    CHILD_REPLAY_FAILED = "child_replay_failed"
    PROP_NOT_RUN_STRATEGY_GATE = "prop_not_run_strategy_gate"
    NO_MODEL_RESULT = "no_model_result"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    PROTECTED_RANGE_REFUSAL = "protected_range_refusal"
    VERIFICATION_AUTHORIZATION_MISSING = "verification_authorization_missing"
    CAPABILITY_PLANNED = "capability_planned"
    LINEAGE_NOT_COMPARABLE = "lineage_not_comparable"
    BROWSER_QA_UNAVAILABLE = "browser_qa_unavailable"
    #: R6.1 (plan §6.G "new §31 states"): a regime stratum below the stamped
    #: minimum, and a stratified report class refused by the regime status
    INSUFFICIENT_REGIME_PARTITION = "insufficient_regime_partition"
    REGIME_STATUS_BELOW_MINIMUM = "regime_status_below_minimum"
    #: UI-1 (plan §6.7): distinct states so a "no runs" / "not selected" /
    #: "not applicable" situation never renders as ARTIFACT_UNAVAILABLE, the
    #: missing/corrupt artifact split, and the typed purpose / runner /
    #: launch / authorization / namespace / seed states the redesign renders
    NO_RUNS = "no_runs"
    NOT_SELECTED = "not_selected"
    NOT_APPLICABLE = "not_applicable"
    NOT_CONFIGURED = "not_configured"
    RESUME_AVAILABLE = "resume_available"
    ARTIFACT_MISSING = "artifact_missing"
    ARTIFACT_CORRUPT = "artifact_corrupt"
    LEGACY_READ_ONLY = "legacy_read_only"
    SUPERSEDED = "superseded"
    PURPOSE_UNRESOLVED = "purpose_unresolved"
    RUNNER_UNAVAILABLE = "runner_unavailable"
    LAUNCH_NOT_STARTED = "launch_not_started"
    AUTHORIZATION_NOT_READY = "authorization_not_ready"
    STORE_NAMESPACE_UNVERIFIED = "store_namespace_unverified"
    SEED_PRODUCTION_NOT_AUTHORIZED = "seed_production_not_authorized"
    #: UI-2 (plan §6.7 / §9 Phase 2): the Verification Center's typed states
    #: (shortlist evidence, the owner's provisional window, the verified seed,
    #: the signed final authorization, the bounded-run preflight) and the
    #: draft lifecycle states (session-only, archived) — owner Q2.
    SHORTLIST_UNAVAILABLE = "shortlist_unavailable"
    WINDOW_NOT_SELECTED = "window_not_selected"
    SEED_MISSING = "seed_missing"
    FINAL_AUTHORIZATION_UNSIGNED = "final_authorization_unsigned"
    PREFLIGHT_REFUSED = "preflight_refused"
    DRAFT_ARCHIVED = "draft_archived"
    DRAFT_SESSION_ONLY = "draft_session_only"


class EmptyStatePresentation(FrozenContract):
    key: EmptyStateKey
    heading: str
    explanation: str
    owning_gate: str | None
    next_action: str | None


# ─────────────────────────────────────────────────────────────────────────────
# Exact required copy (FUX §§5.3, 14.2, 15, 16.4, 18, 31)
# ─────────────────────────────────────────────────────────────────────────────

#: FUX §5.3 — the only permitted development-selection title.
DEV_BADGE_TEXT = "Development Exploratory Representative"

#: FUX §14.2 — the non-dismissible verification-scope badge.
VERIFICATION_BADGE_TEXT = "VERIFICATION ONLY — not research evidence"

#: FUX §18/§31 — the exact required no-pass sentence.
NO_PASS_SENTENCE = "No configuration passed all benchmarks."

#: FUX §16.4 — skipped-stage copy when the strategy gate failed earlier.
NOT_RUN_STRATEGY_GATE_TEXT = "Not run — strategy gate failed"

#: FUX §15 — the exact full-authorized-scope second-confirmation warning.
FULL_SCOPE_WARNING_TEXT = (
    "This will run the full authorized development pipeline.\n"
    "It is not an implementation verification run."
)

#: The acknowledgement phrase the user must type before the full-scope launch
#: control is enabled (FUX §15: "the required study/pipeline name or
#: acknowledgement phrase").
FULL_SCOPE_ACKNOWLEDGEMENT = "run full authorized development"

#: Display phrases that may never appear as UI status/title wording without a
#: separately authorized status contract (FUX §5.3; TEST_MATRIX §3.7 P1-6;
#: assumed-path truthfulness per FUX §1). Scanned case-insensitively against
#: every visible-label registry in this module and against the UI sources.
FORBIDDEN_DISPLAY_PHRASES: tuple[str, ...] = (
    "production ready",
    "live ready",
    "robust representative",
    "publishable",
    "exact historical",
)


ROUTE_LABELS: Mapping[StudyWorkspaceRoute, str] = MappingProxyType(
    {
        StudyWorkspaceRoute.START: "Start",
        StudyWorkspaceRoute.VERIFY_IMPLEMENTATION: "Verify Implementation",
        StudyWorkspaceRoute.NEW_STUDY: "New Study",
        StudyWorkspaceRoute.ACTIVE_RUNS: "Active Runs",
        StudyWorkspaceRoute.RESULTS: "Results",
        StudyWorkspaceRoute.HISTORY: "History",
        StudyWorkspaceRoute.CONTEXT_RESEARCH: "Context Research",
    }
)

RESULT_SCOPE_LABELS: Mapping[ResultScope, str] = MappingProxyType(
    {
        ResultScope.CANDIDATE_RESEARCH: "Candidate Research",
        ResultScope.ACTUAL_EXECUTED_STRATEGY: "Actual Executed Strategy",
        ResultScope.PROP_HISTORICAL_CLOSED_TRADE: (
            "Prop Historical Closed-Trade Replay"
        ),
        ResultScope.PROP_1M_SCENARIO: "Prop 1m Scenario / Approximation",
        ResultScope.PROP_ORDERED_EVENT_REPLAY: "Prop Ordered-Event Replay",
        ResultScope.BOOTSTRAP_SIMULATION: "Bootstrap Simulation",
        ResultScope.STRESS_SIMULATION: "Stress Simulation",
    }
)

#: FUX §20 — the exact heatmap cell glyph classes (color never sole carrier).
HEATMAP_GLYPHS: Mapping[str, str] = MappingProxyType(
    {
        "stable_plateau": "◼",
        "knife_edge_point": "▲",
        "failed_region": "✕",
        "insufficient_data": "·",
        "blocked_cell": "⊘",
    }
)

#: FUX §30.4 — the 16 pipeline stages' short human titles (order-exact).
PIPELINE_STAGE_TITLES: Mapping[str, str] = MappingProxyType(
    {
        "00_validate_inputs": "Validate Inputs",
        "01_prepare_strategy_profiles": "Prepare Strategy Profiles",
        "02_run_or_reuse_sequential_replays": "Run / Reuse Sequential Replays",
        "03_build_or_reuse_fsm_audit": "Build / Reuse FSM Audit",
        "04_build_or_reuse_replay_charts": "Build / Reuse Replay Charts",
        "05_materialize_feature_views": "Materialize Feature Views",
        "06_validate_feature_coverage": "Validate Feature Coverage",
        "07_derive_labels": "Derive Labels",
        "08_build_folds": "Build Folds",
        "09_train_models": "Train Models",
        "10_generate_predictions_and_diagnostics": "Predictions & Diagnostics",
        "11_run_frozen_model_gated_replays": "Frozen Model-Gated Replays",
        "12_run_prop_historical_replays": "Prop Historical Replays",
        "13_run_bootstrap_and_stress": "Bootstrap & Stress",
        "14_build_frontier_and_insights": "Frontier & Insights",
        "15_verify_and_publish": "Verify & Publish",
    }
)

#: FUX §30.4 — glyph + word per pipeline stage status (color never sole
#: carrier; the "not required" presentation covers stages outside the plan).
PIPELINE_STAGE_STATUS_PRESENTATIONS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "pending": ("·", "Pending"),
        "queued": ("·", "Queued"),
        "running": ("▶", "Running"),
        "checkpointed": ("▶", "Checkpointed"),
        "completed": ("✓", "Completed"),
        "reused": ("↺", "Reused"),
        "failed": ("✕", "Failed"),
        "cancel_requested": ("⊘", "Cancel Requested"),
        "cancelled_at_safe_boundary": ("⊘", "Cancelled (safe boundary)"),
        "blocked": ("⛔", "Blocked"),
    }
)

#: The word rendered for a stage the selected plan does not include.
PIPELINE_STAGE_NOT_REQUIRED = ("—", "Not required")


#: FUX §28 — the exact account-timeline marker shapes.
TIMELINE_MARKERS: Mapping[str, str] = MappingProxyType(
    {
        "payout": "▽",
        "fee": "◇",
        "breach": "✕",
        "replacement": "□",
    }
)


STATUS_PRESENTATIONS: Mapping[StudyStatusKey, StatusPresentation] = MappingProxyType(
    {
        StudyStatusKey.DRAFT: StatusPresentation(
            status_key=StudyStatusKey.DRAFT,
            visible_label="Draft",
            glyph="✎",
            semantic_class="neutral",
            help_text=(
                "A mutable, disk-persisted wizard draft. Not frozen, not "
                "launched, carries no research weight."
            ),
        ),
        StudyStatusKey.FROZEN: StatusPresentation(
            status_key=StudyStatusKey.FROZEN,
            visible_label="Frozen",
            glyph="🔒",
            semantic_class="info",
            help_text=(
                "The search charter is immutably saved. Research-bearing "
                "changes require Clone as New Search."
            ),
        ),
        StudyStatusKey.QUEUED: StatusPresentation(
            status_key=StudyStatusKey.QUEUED,
            visible_label="Queued",
            glyph="⏳",
            semantic_class="info",
            help_text="Waiting for a worker slot; no replay has started.",
        ),
        StudyStatusKey.RUNNING: StatusPresentation(
            status_key=StudyStatusKey.RUNNING,
            visible_label="Running",
            glyph="▶",
            semantic_class="running",
            help_text="The detached search job is executing.",
        ),
        StudyStatusKey.REPLAY_FAILED: StatusPresentation(
            status_key=StudyStatusKey.REPLAY_FAILED,
            visible_label="Replay Failed",
            glyph="✕",
            semantic_class="failure",
            help_text=(
                "The sequential replay (or its audit-neutrality proof) "
                "failed; downstream stages did not run."
            ),
        ),
        StudyStatusKey.STRATEGY_REJECTED: StatusPresentation(
            status_key=StudyStatusKey.STRATEGY_REJECTED,
            visible_label="Strategy Rejected",
            glyph="⊖",
            semantic_class="failure",
            help_text=(
                "The underlying strategy gate failed; prop simulation was "
                "not run for this configuration."
            ),
        ),
        StudyStatusKey.PROP_REJECTED: StatusPresentation(
            status_key=StudyStatusKey.PROP_REJECTED,
            visible_label="Prop Rejected",
            glyph="⊘",
            semantic_class="failure",
            help_text=(
                "At least one prop feasibility gate failed (conservative "
                "ALL-legs rule)."
            ),
        ),
        StudyStatusKey.ROBUST_FINALIST: StatusPresentation(
            status_key=StudyStatusKey.ROBUST_FINALIST,
            visible_label="Robust Finalist",
            glyph="✓",
            semantic_class="success",
            help_text=(
                "Passed every applicable feasibility gate and survived the "
                "robustness checks on the comparison surfaces."
            ),
        ),
        StudyStatusKey.SELECTED_REPRESENTATIVE: StatusPresentation(
            status_key=StudyStatusKey.SELECTED_REPRESENTATIVE,
            visible_label=DEV_BADGE_TEXT,
            glyph="★",
            semantic_class="highlight",
            help_text=(
                "The interior-of-plateau development selection. NOT a "
                "research-authorized recommendation: a separately frozen, "
                "owner-approved outer evaluation protocol is required before "
                "any stronger status exists."
            ),
        ),
        StudyStatusKey.SUPERSEDED: StatusPresentation(
            status_key=StudyStatusKey.SUPERSEDED,
            visible_label="Superseded",
            glyph="↻",
            semantic_class="neutral",
            help_text=(
                "A newer immutable artifact supersedes this one; it remains "
                "readable for audit."
            ),
        ),
        StudyStatusKey.BLOCKED: StatusPresentation(
            status_key=StudyStatusKey.BLOCKED,
            visible_label="Blocked",
            glyph="⛔",
            semantic_class="blocked",
            help_text=(
                "A fail-closed gate (capability, authorization, or policy) "
                "prevents this action; the reason is shown where it blocks."
            ),
        ),
    }
)


def status_presentation(key: StudyStatusKey | str) -> StatusPresentation:
    """The registered presentation for ``key`` (KeyError for unknown keys)."""

    return STATUS_PRESENTATIONS[StudyStatusKey(key)]


EMPTY_STATE_PRESENTATIONS: Mapping[str, EmptyStatePresentation] = MappingProxyType(
    {
        "no_configurations_pass": EmptyStatePresentation(
            key=EmptyStateKey.NO_CONFIGURATIONS_PASS,
            heading=NO_PASS_SENTENCE,
            explanation=(
                "Every enumerated configuration stopped at a feasibility "
                "gate. The dominant failure reasons and the count stopped at "
                "each gate are listed below."
            ),
            owning_gate="strategy / prop / robustness feasibility gates",
            next_action=(
                "Inspect the dominant failure reasons, then Clone as New "
                "Search to adjust registered axes or thresholds."
            ),
        ),
        "no_verified_firm_contract": EmptyStatePresentation(
            key=EmptyStateKey.NO_VERIFIED_FIRM_CONTRACT,
            heading="No verified firm contract is available",
            explanation=(
                "A real prop study requires a first_party_verified contract "
                "compiled from first-party evidence and owner-reviewed. "
                "Synthetic fixture contracts are visibly synthetic and "
                "cannot enter a real study."
            ),
            owning_gate="contract-evidence compiler status ladder (CS §5.5)",
            next_action=(
                "Compile first-party contract evidence and complete the "
                "owner review (owner decisions 5/6)."
            ),
        ),
        "blocked_search_axis": EmptyStatePresentation(
            key=EmptyStateKey.BLOCKED_SEARCH_AXIS,
            heading="This axis is blocked",
            explanation=(
                "The axis is registered but not searchable: it is inert, "
                "reducer-hardcoded, legacy, or pending an owner policy "
                "review. Its value and reason remain visible for audit; no "
                "input widget exists."
            ),
            owning_gate="typed axis/value registry (fail-closed)",
            next_action=None,
        ),
        "insufficient_sample": EmptyStatePresentation(
            key=EmptyStateKey.INSUFFICIENT_SAMPLE,
            heading="Insufficient sample",
            explanation=(
                "The observed sample is below the applicable minimum; the "
                "value is withheld rather than rendered as if reliable."
            ),
            owning_gate="sample-adequacy thresholds (proposed_protocol_default)",
            next_action=None,
        ),
        "child_replay_failed": EmptyStatePresentation(
            key=EmptyStateKey.CHILD_REPLAY_FAILED,
            heading="Child replay failed",
            explanation=(
                "The sequential replay for this configuration failed; the "
                "sanitized failure text is shown in the child detail. "
                "Downstream gates and simulations did not run."
            ),
            owning_gate="child replay worker",
            next_action="Open the child detail for the sanitized evidence.",
        ),
        "prop_not_run_strategy_gate": EmptyStatePresentation(
            key=EmptyStateKey.PROP_NOT_RUN_STRATEGY_GATE,
            heading=NOT_RUN_STRATEGY_GATE_TEXT,
            explanation=(
                "Prop simulation only runs for configurations that pass the "
                "underlying strategy gate; this configuration stopped there."
            ),
            owning_gate="underlying strategy gate",
            next_action="Open the strategy-gate evidence in the child detail.",
        ),
        "no_model_result": EmptyStatePresentation(
            key=EmptyStateKey.NO_MODEL_RESULT,
            heading="No model result",
            explanation=(
                "No model fit exists for this selection. The supervised "
                "ladder and its result surfaces land with R5; nothing is "
                "silently substituted."
            ),
            owning_gate="R5 pipeline / supervised model ladder",
            next_action=None,
        ),
        "artifact_unavailable": EmptyStatePresentation(
            key=EmptyStateKey.ARTIFACT_UNAVAILABLE,
            heading="Artifact unavailable",
            explanation=(
                "The referenced immutable artifact is not present in this "
                "store (or failed manifest verification). Nothing was "
                "fabricated in its place."
            ),
            owning_gate="immutable store manifest verification",
            next_action=(
                "Verify the store root and the exact artifact identity in "
                "Audit disclosure."
            ),
        ),
        "protected_range_refusal": EmptyStatePresentation(
            key=EmptyStateKey.PROTECTED_RANGE_REFUSAL,
            heading="Protected range refused",
            explanation=(
                "The request touches a protected or sealed date range and "
                "was refused before any source path was constructed."
            ),
            owning_gate="development/verification access policy",
            next_action=None,
        ),
        "verification_authorization_missing": EmptyStatePresentation(
            key=EmptyStateKey.VERIFICATION_AUTHORIZATION_MISSING,
            heading="Verification authorization missing",
            explanation=(
                "The real five-day verification slice requires the "
                "owner-approved VerificationAuthorizationRef (approved "
                "allowlist hash + coverage-matrix artifact + seed snapshot "
                "+ approver). It does not exist yet, so the real run is "
                "blocked before source-path construction. Code authoring "
                "and synthetic states remain available."
            ),
            owning_gate="owner decisions 21 / R-5 (BLOCKING-VERIFICATION)",
            next_action=(
                "Owner: choose the canonical allowlist, sign off the "
                "coverage matrix, and approve the authorization reference."
            ),
        ),
        "feature_block_planned": EmptyStatePresentation(
            key=EmptyStateKey.CAPABILITY_PLANNED,
            heading="Feature block planned / unavailable",
            explanation=(
                "This feature block is registered as planned and is not "
                "resolvable into a bundle yet. MBP-1 activation is R5B and "
                "remains research_only_offline after activation."
            ),
            owning_gate="feature-block registry (planned state, fail-closed)",
            next_action=None,
        ),
        "runner_executor_planned": EmptyStatePresentation(
            key=EmptyStateKey.CAPABILITY_PLANNED,
            heading="No registered executor for this charter",
            explanation=(
                "Real full-development charters have no registered runner "
                "executor: the operator full run is a separate, explicitly "
                "authorized owner action, never an implementation launch. "
                "The frozen charter stays immutable and reusable."
            ),
            owning_gate=(
                "runner-entry registry (fail-closed; the operator full run "
                "is a separate authorized action)"
            ),
            next_action=(
                "Run verification-fixture or synthetic charters, or wait for "
                "the owner-authorized operator run."
            ),
        ),
        "pipeline_no_runs": EmptyStatePresentation(
            key=EmptyStateKey.NO_RUNS,
            heading="No pipeline runs exist yet",
            explanation=(
                "No 16-stage pipeline has been launched from this workspace. "
                "Nothing is missing or corrupt — configure and freeze a "
                "pipeline specification from the Full Pipeline Run wizard "
                "mode to create one."
            ),
            owning_gate=None,
            next_action=(
                "Open New Study → Full Pipeline Run, complete the wizard, "
                "and launch from the Review step."
            ),
        ),
        "no_runs": EmptyStatePresentation(
            key=EmptyStateKey.NO_RUNS,
            heading="No search runs exist yet",
            explanation=(
                "No frozen search charter has been launched from this "
                "workspace. Nothing is missing or corrupt — this is the "
                "empty starting state, not an artifact failure."
            ),
            owning_gate=None,
            next_action=(
                "Open Start, choose what you are trying to do, and freeze "
                "a satisfiable draft from Review & Launch."
            ),
        ),
        "not_selected": EmptyStatePresentation(
            key=EmptyStateKey.NOT_SELECTED,
            heading="Nothing selected",
            explanation=(
                "No item is selected for this view yet; the panel waits "
                "for a selection and fabricates nothing in the meantime."
            ),
            owning_gate=None,
            next_action="Select a run, configuration, or case above.",
        ),
        "not_applicable": EmptyStatePresentation(
            key=EmptyStateKey.NOT_APPLICABLE,
            heading="Not applicable to this selection",
            explanation=(
                "This block does not apply to the selected purpose, study "
                "family, or evidence class; it is intentionally absent "
                "rather than rendered empty."
            ),
            owning_gate="purpose / study-family flow rules",
            next_action=None,
        ),
        "not_configured": EmptyStatePresentation(
            key=EmptyStateKey.NOT_CONFIGURED,
            heading="Not configured yet",
            explanation=(
                "The required configuration step has not been completed "
                "for this draft, so nothing downstream can be previewed "
                "or launched."
            ),
            owning_gate="wizard step validation",
            next_action="Complete the earlier step named in the goal card.",
        ),
        "resume_available": EmptyStatePresentation(
            key=EmptyStateKey.RESUME_AVAILABLE,
            heading="Interrupted — resume available",
            explanation=(
                "The run stopped before its terminal state. Completed "
                "stages and children keep their verified identities; a "
                "new operational attempt continues from the last atomic "
                "checkpoint under the same semantic id."
            ),
            owning_gate="execution-attempt identity (P0-3)",
            next_action="Use Resume / Retry to start a new attempt.",
        ),
        "artifact_missing": EmptyStatePresentation(
            key=EmptyStateKey.ARTIFACT_MISSING,
            heading="Artifact missing",
            explanation=(
                "The referenced immutable artifact is not present in the "
                "store the run belongs to. Nothing was fabricated in its "
                "place; the exact identity is shown under Audit."
            ),
            owning_gate="immutable store exact-id load",
            next_action=(
                "Verify the exact artifact identity under Technical "
                "identity & audit; a missing artifact is never guessed."
            ),
        ),
        "artifact_corrupt": EmptyStatePresentation(
            key=EmptyStateKey.ARTIFACT_CORRUPT,
            heading="Artifact failed verification",
            explanation=(
                "The referenced artifact exists but its manifest, hash, or "
                "identity did not verify. Nothing derived from it is shown "
                "as trusted evidence."
            ),
            owning_gate="immutable store manifest verification",
            next_action=(
                "Inspect the sanitized verification failure under Audit; "
                "the store never overwrites a corrupt entry."
            ),
        ),
        "legacy_read_only": EmptyStatePresentation(
            key=EmptyStateKey.LEGACY_READ_ONLY,
            heading="Legacy read-only result",
            explanation=(
                "This result was produced by an earlier lane and is kept "
                "with its original reports and caveats. No rerun, "
                "modification, or promotion control exists for it."
            ),
            owning_gate="fixed M0–M3 lane (unchanged)",
            next_action=None,
        ),
        "superseded": EmptyStatePresentation(
            key=EmptyStateKey.SUPERSEDED,
            heading="Superseded",
            explanation=(
                "A newer artifact or decision replaced this one. It stays "
                "read-only for audit under its catalog archive flag; "
                "nothing is deleted."
            ),
            owning_gate="catalog archive flag / supersession chain",
            next_action="Open the replacement named in the audit record.",
        ),
        "purpose_unresolved": EmptyStatePresentation(
            key=EmptyStateKey.PURPOSE_UNRESOLVED,
            heading="Run purpose unresolved",
            explanation=(
                "This draft carries no purpose annotation and its legacy "
                "run scope does not derive one unambiguous purpose "
                "(Development Research and Full Authorized Development "
                "share the full_authorized_development scope). It cannot "
                "freeze or launch until the owner confirms the purpose; "
                "the confirmation is a presentation annotation and never "
                "changes a charter identity."
            ),
            owning_gate="run-purpose resolution (plan §5.6)",
            next_action="Confirm the purpose in the goal card above.",
        ),
        "runner_unavailable": EmptyStatePresentation(
            key=EmptyStateKey.RUNNER_UNAVAILABLE,
            heading="No registered executor is available in this process",
            explanation=(
                "The frozen charter names a runner-entry key that this "
                "process has not registered, so no worker was spawned. "
                "The charter stays immutable and reusable; nothing was "
                "reported as launched."
            ),
            owning_gate="runner-entry registry (fail-closed before spawn)",
            next_action=(
                "Launch from a process that registers the executor, or "
                "use the CLI fallback with a registered runner-entry key."
            ),
        ),
        "launch_not_started": EmptyStatePresentation(
            key=EmptyStateKey.LAUNCH_NOT_STARTED,
            heading="Launch requested — no persisted state yet",
            explanation=(
                "The detached worker was spawned but no job state appeared "
                "within the wait window, so the launch is NOT reported as "
                "started. The worker may still be initializing, or it may "
                "have exited before writing state; the job log is the "
                "evidence."
            ),
            owning_gate="honest launch outcome (state must exist)",
            next_action=(
                "Refresh Active Runs; if no state appears, read the job log "
                "named under Audit and relaunch with the CLI fallback."
            ),
        ),
        "authorization_not_ready": EmptyStatePresentation(
            key=EmptyStateKey.AUTHORIZATION_NOT_READY,
            heading="Owner authorization is not ready",
            explanation=(
                "The actual computation path requires real owner "
                "authorization bound to this store's verified namespace and "
                "current supersession head. Its typed readiness state is "
                "shown with the reason; freezing and launching stay "
                "disabled until it is ready. Synthetic fixtures remain "
                "available under Implementation Verification."
            ),
            owning_gate="computation-path-scoped owner authorization",
            next_action=(
                "Resolve the named readiness reason (missing, stale head, "
                "superseded, wrong namespace, wrong profile or source) "
                "through the owner's separate authorization workflow."
            ),
        ),
        "store_namespace_unverified": EmptyStatePresentation(
            key=EmptyStateKey.STORE_NAMESPACE_UNVERIFIED,
            heading="Store namespace not verified",
            explanation=(
                "The store this purpose resolves to carries no verified "
                "semantic namespace (unmarked, corrupt, incoherently "
                "deployed, or of the wrong class). A local path never "
                "defines authority, so real charters cannot freeze here."
            ),
            owning_gate="semantic store namespace (HARDENING-BACKEND §4.1)",
            next_action=(
                "Initialize the store explicitly with its intended class "
                "and a recorded store instance id (ifvg_store_namespace.py)."
            ),
        ),
        "seed_production_not_authorized": EmptyStatePresentation(
            key=EmptyStateKey.SEED_PRODUCTION_NOT_AUTHORIZED,
            heading="Seed production not authorized",
            explanation=(
                "Producing the verification seed requires the owner's "
                "signed SeedProductionAuthorizationRef bound to this store; "
                "none is verified yet, so no seed job can start and no "
                "final verification authorization can exist."
            ),
            owning_gate="SeedProductionAuthorizationRef (HARDENING-BACKEND §5.3)",
            next_action=(
                "Prepare the seed-production packet, obtain the owner's "
                "signature, and refresh."
            ),
        ),
        "shortlist_unavailable": EmptyStatePresentation(
            key=EmptyStateKey.SHORTLIST_UNAVAILABLE,
            heading="Verification window shortlist unavailable",
            explanation=(
                "The ranked logical trading-day shortlist (rebuilt from "
                "already-authorized evidence only) is not present or failed "
                "verification, so no window can be reviewed or selected. "
                "Nothing here reads raw source data or registers an allowlist."
            ),
            owning_gate="logical_window_shortlist_v1 (HARDENING-BACKEND §5.1)",
            next_action=(
                "Rebuild the shortlist with scripts/ifvg_verification_window_shortlist.py "
                "and refresh."
            ),
        ),
        "window_not_selected": EmptyStatePresentation(
            key=EmptyStateKey.WINDOW_NOT_SELECTED,
            heading="No provisional verification window selected",
            explanation=(
                "The owner selects one eligible window of consecutive logical "
                "trading days from the shortlist. Until then no seed-production "
                "packet, seed job or final authorization packet can be prepared. "
                "The selection is provisional and registers nothing."
            ),
            owning_gate="owner window selection (HARDENING-BACKEND §5.4 step A)",
            next_action=(
                "Select an eligible window in the Fixture step and record it as the "
                "provisional window."
            ),
        ),
        "seed_missing": EmptyStatePresentation(
            key=EmptyStateKey.SEED_MISSING,
            heading="No verified profile-matching seed",
            explanation=(
                "The real verification slice starts from a verified seed snapshot "
                "produced by the separately authorized seed-production chain "
                "(a preparation action with zero verification-evidence footprint). "
                "No verified seed continuous with the selected window exists, so the "
                "final verification packet cannot be built."
            ),
            owning_gate="seed snapshot store (exact-id, profile-bound load)",
            next_action=(
                "Run the authorized seed-production job from its exact command, then "
                "refresh so the receipt is picked up and the seed verified."
            ),
        ),
        "final_authorization_unsigned": EmptyStatePresentation(
            key=EmptyStateKey.FINAL_AUTHORIZATION_UNSIGNED,
            heading="Final verification authorization not signed",
            explanation=(
                "The unsigned packet carries owner placeholders that fail "
                "validation; only the owner completes the VerificationAuthorizationRef "
                "outside this workspace. Until a completed reference validates against "
                "this store's namespace, current supersession head, seed and window, "
                "the verification charter cannot freeze and no run can be registered."
            ),
            owning_gate="VerificationAuthorizationRef (owner decisions 21 / R-5)",
            next_action=(
                "Complete the packet's owner fields in the named file, then validate it "
                "here."
            ),
        ),
        "preflight_refused": EmptyStatePresentation(
            key=EmptyStateKey.PREFLIGHT_REFUSED,
            heading="Bounded-run preflight refused",
            explanation=(
                "One of the typed §6.1 checks refused before any source path was "
                "constructed (namespace, head witness, real authorization, logical "
                "window, partition mapping, canonical allowlist, seed). The exact "
                "reason is shown; no run command is offered while it holds."
            ),
            owning_gate="bounded_verification_preflight_v1 (HARDENING-BACKEND §6.1)",
            next_action="Resolve the named refusal reason, then run the preflight again.",
        ),
        "draft_archived": EmptyStatePresentation(
            key=EmptyStateKey.DRAFT_ARCHIVED,
            heading="This draft is archived",
            explanation=(
                "Archived drafts are hidden from the default History listing and "
                "cannot be edited or frozen; nothing was deleted and every field is "
                "kept."
            ),
            owning_gate="draft lifecycle (owner Q2)",
            next_action="Restore it from History's archived view to continue editing.",
        ),
        "draft_session_only": EmptyStatePresentation(
            key=EmptyStateKey.DRAFT_SESSION_ONLY,
            heading="Draft not saved yet",
            explanation=(
                "This draft exists only in the current session — no file has been "
                "written. It is persisted on the first explicit Save Draft or the "
                "first valid Next, and autosaved afterwards."
            ),
            owning_gate="draft lifecycle (owner Q2)",
            next_action="Name the draft and Save Draft (or complete the step and press Next).",
        ),
        "regime_algorithm_planned": EmptyStatePresentation(
            key=EmptyStateKey.CAPABILITY_PLANNED,
            heading="Regime algorithm planned / unavailable",
            explanation=(
                "This regime algorithm is registered as planned; no fit "
                "implementation is callable in V1. KMeans is the only V1 "
                "regime lane (R6)."
            ),
            owning_gate="regime algorithm registry (planned state, fail-closed)",
            next_action=None,
        ),
        "lineage_not_comparable": EmptyStatePresentation(
            key=EmptyStateKey.LINEAGE_NOT_COMPARABLE,
            heading="Populations not comparable",
            explanation=(
                "No exact profile-independent lineage basis exists for this "
                "pair (or a lineage collision was detected), so common/"
                "added/removed membership claims are disabled rather than "
                "approximated. The parameter diff remains available."
            ),
            owning_gate="opportunity lineage match_basis (no fuzzy fallback)",
            next_action="Review the lineage reason in Audit disclosure.",
        ),
        "browser_qa_unavailable": EmptyStatePresentation(
            key=EmptyStateKey.BROWSER_QA_UNAVAILABLE,
            heading="Browser QA unavailable",
            explanation=(
                "Interactive browser/viewport evidence could not be "
                "captured in this environment. The corresponding acceptance "
                "gate remains OPEN; AppTest success is not substituted for "
                "it."
            ),
            owning_gate="hardening viewport/keyboard evidence gate",
            next_action="Re-run the QA pass where a browser backend exists.",
        ),
        "insufficient_regime_partition": EmptyStatePresentation(
            key=EmptyStateKey.INSUFFICIENT_REGIME_PARTITION,
            heading="Insufficient regime partition",
            explanation=(
                "This regime stratum holds fewer observations than the "
                "stamped minimum per regime stratum (a "
                "proposed_protocol_default); its metrics are withheld and "
                "the row stays typed rather than rendered as if reliable. "
                "Regime ids are nominal and no ranking is implied."
            ),
            owning_gate=(
                "minimum_trades_per_regime_stratum / "
                "minimum_training_rows_per_regime_stratum (proposed_protocol_default)"
            ),
            next_action=None,
        ),
        "regime_status_below_minimum": EmptyStatePresentation(
            key=EmptyStateKey.REGIME_STATUS_BELOW_MINIMUM,
            heading="Regime status below the class minimum",
            explanation=(
                "The requested stratified report class requires a regime "
                "status this run's exact decision does not reach "
                "(STRATIFICATION_READY for descriptive classes; "
                "FEATURE_ELIGIBLE with verified owner evidence for modeled "
                "classes). The report is unconstructible and the refusal is "
                "recorded; nothing here promotes."
            ),
            owning_gate="regime promotion ladder + verified owner-decision evidence (P1-3)",
            next_action=(
                "Descriptive classes: a passing capability assessment derives "
                "STRATIFICATION_READY at S10. Modeled classes: persist the "
                "owner decision artifact, promote to FEATURE_ELIGIBLE, and "
                "freeze the exact ids into a new model-bearing study."
            ),
        ),
    }
)
