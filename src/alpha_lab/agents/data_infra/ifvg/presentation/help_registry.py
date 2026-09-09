"""Helper text, the glossary and the explicit help exemptions (UI-3; plan §6.4) — pure.

Every non-obvious control of the IFVG Lab UI has ONE ``HelpEntry`` keyed by
its control id (``<surface>.<control>``): what the control changes, why it
exists, its default, whether changing it mints a new scientific identity,
which expensive work it requires (replay / refit / resimulation), whether an
owner approval is involved and when the control is available.
``help_text`` renders the entry as the ``help=`` tooltip of the widget;
``help_for_metric`` renders a metric's registry row the same way. The
source scan (``test_ifvg_study_scans``) requires every widget call to carry
``help=`` or to be a registered exemption below — an exemption records the
control and its rationale, and a stale exemption fails the scan.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal

from ..search.identities import FrozenContract
from .metric_registry import describe

__all__ = [
    "GLOSSARY",
    "GLOSSARY_TERMS",
    "HELP_EXEMPTIONS",
    "HELP_REGISTRY",
    "GlossaryEntry",
    "HelpEntry",
    "glossary_markdown",
    "help_for_metric",
    "help_text",
]

Requirement = Literal["replay", "refit", "resimulation"]


class HelpEntry(FrozenContract):
    control_id: str
    what_it_changes: str
    why: str
    default: str
    changes_identity: bool
    requires: tuple[Requirement, ...]
    owner_approval: bool
    availability: str


def _h(
    control_id: str,
    what: str,
    why: str,
    *,
    default: str = "none (an action)",
    identity: bool = False,
    requires: tuple[Requirement, ...] = (),
    owner: bool = False,
    availability: str = "always on this screen",
) -> HelpEntry:
    return HelpEntry(
        control_id=control_id,
        what_it_changes=what,
        why=why,
        default=default,
        changes_identity=identity,
        requires=requires,
        owner_approval=owner,
        availability=availability,
    )


_EXACT_ID = (
    "only the exact 64-hex identity resolves — there is no fuzzy, prefix or nearest-time "
    "lookup"
)
_STORES_NEVER_LISTED = (
    "stores are never listed; an id is either auto-resolved from the selected run or pasted"
)

_ENTRIES: tuple[HelpEntry, ...] = (
    # ── shared primitives ──
    _h(
        "common.rows_per_page",
        "How many rows one page of the table shows",
        "indexed pagination keeps large tables readable without loading every artifact",
        default="25",
        availability="every paginated table",
    ),
    _h(
        "common.page",
        "Which page of the table is shown",
        "the page index is server-side; nothing beyond the page is loaded",
        default="1",
        availability="every paginated table",
    ),
    _h(
        "common.detail_level",
        "Which tier of the screen renders: Summary (plain-language answers and roll-ups), "
        "Research details (charts, tables and their twins) or Technical identity & audit "
        "(exact ids, manifests, raw JSON)",
        "progressive disclosure keeps the decision visible first while the exact evidence "
        "stays one step away",
        default="Summary",
        availability="every major results screen",
    ),
    # ── Start ──
    _h(
        "start.open_verification_center",
        "Opens the Verify Implementation route (the Verification Center)",
        "the verification workflow lives on its own route with no research controls",
    ),
    # ── New Study wizard ──
    _h(
        "wizard.open_draft",
        "Which saved draft the wizard loads",
        "a draft restores at its exact stored step; the session draft is stashed first",
        default="the most recent draft",
        availability="when at least one draft is saved",
    ),
    _h(
        "wizard.clone_frozen",
        "Creates a new mutable draft from this frozen study's steps",
        "frozen charters are immutable; a clone is the only way to vary a frozen study",
        availability="when the open draft is frozen",
    ),
    _h(
        "wizard.confirm_purpose",
        "Records the owner-confirmed presentation purpose on this legacy draft",
        "a draft whose purpose cannot be derived unambiguously is purpose_unresolved and cannot "
        "freeze until the owner confirms it",
        owner=True,
        availability="only for legacy drafts with an unresolved purpose",
    ),
    _h(
        "wizard.research_question",
        "The research question the study answers; it selects the study family and the flow",
        "Evaluate, Compare, FSM search and the prop questions produce different steps and "
        "satisfiability rules (owner Q4)",
        default="from the Start card",
        identity=True,
    ),
    _h(
        "wizard.objective_template",
        "The objective template — the primary objectives, hard constraints and tie-breaks that "
        "enter the charter",
        "no hidden weighted score exists; the template is the declared objective and a selected "
        "prop objective is never rewritten",
        default="from the Start card",
        identity=True,
    ),
    _h(
        "wizard.custom_objectives",
        "The registered objective metrics of a Custom template",
        "only metrics with a registered optimisation direction may enter the charter",
        default="none",
        identity=True,
        availability="Custom template only",
    ),
    _h(
        "wizard.select_contract",
        "Adds this verified firm contract to the charter",
        "a prop objective needs at least one first_party_verified contract (Universal Prop "
        "Search needs two); synthetic fixtures are confined to verification",
        default="not selected",
        identity=True,
        requires=("resimulation",),
        availability="launchable contracts only",
    ),
    _h(
        "wizard.risk_template",
        "The account risk template applied to this firm's accounts",
        "the risk policy shapes every simulated account path",
        default="the first registered template",
        identity=True,
        requires=("resimulation",),
    ),
    _h(
        "wizard.withdrawal_behavior",
        "The trader's withdrawal behaviour in the simulation (separate from the firm's "
        "permitted payout schedule)",
        "withdrawals change the account equity path and therefore every payout metric",
        default="withdraw at every eligible payout",
        identity=True,
        requires=("resimulation",),
    ),
    _h(
        "wizard.replacement_policy",
        "Whether a breached account is replaced — a separate scenario (owner decision 12)",
        "replacement adds fees and a new account path; it is never mixed into the base scenario",
        default="none",
        identity=True,
        requires=("resimulation",),
        owner=True,
    ),
    _h(
        "wizard.accounts_per_firm",
        "How many simultaneous accounts per firm the portfolio simulation runs (owner decision 13)",
        "portfolio metrics aggregate across the accounts of one firm",
        default="1",
        identity=True,
        requires=("resimulation",),
        owner=True,
    ),
    _h(
        "wizard.gate_required",
        "Whether this Boolean gate is required",
        "a required gate that fails stops the configuration at its stage; later gates show "
        "which earlier gate stopped them",
        default="from the proposed protocol default",
        identity=True,
    ),
    _h(
        "wizard.download_allowlist",
        "Downloads the resolved date allowlist as JSON",
        "the external seed and verification steps consume the exact allowlist",
        availability="verification scope",
    ),
    _h(
        "wizard.download_dates",
        "Downloads the resolved warmup and evidence date lists as JSON",
        "the frozen warmup prefix and the evidence days are read-only derivations of the "
        "backend logical-day contract",
        availability="development scopes",
    ),
    _h(
        "wizard.seed",
        "The deterministic seed of the bootstrap and simulation protocols",
        "the same seed reproduces the same bootstrap draws; it enters the charter identity",
        default="7",
        identity=True,
        requires=("resimulation",),
    ),
    _h(
        "wizard.full_scope_acknowledgement",
        "Typing the exact phrase enables the full-scope launch control",
        "the full authorized development run is not an implementation verification; the typed "
        "acknowledgement is the second confirmation",
        default="empty",
        owner=True,
        availability="Full Authorized Development only",
    ),
    # ── Active Runs ──
    _h(
        "active_runs.search",
        "Which search job the monitor polls",
        "each run reads its own store, located by the exact charter id",
        default="the most recent run",
    ),
    _h(
        "active_runs.funnel_button",
        "Filters the child table to the children at this funnel stage",
        "the buttons are the keyboard-operable twin of the funnel chart",
        default="Generated",
    ),
    _h(
        "active_runs.child_detail",
        "Which child's sanitized detail, identities and actions are shown",
        "the detail repeats the exact identities the table truncates",
        default="the first child on the page",
    ),
    _h(
        "active_runs.open_results",
        "Opens Results for this search with the child preselected",
        "programmatic navigation keeps the selected configuration synchronised",
    ),
    _h(
        "active_runs.jump_kind",
        "The identity kind of the exact verifier jump",
        _EXACT_ID,
        default="candidate_id",
    ),
    _h(
        "active_runs.jump_value",
        "The exact identifier the Replay / Verifier loads",
        _EXACT_ID,
        default="empty",
    ),
    _h(
        "active_runs.queue_jump",
        "Queues the exact jump; the verifier resolves it or reports it unresolved",
        "an unresolved id terminates in a sanitized warning inside the verifier — never a fallback",
    ),
    _h(
        "active_runs.cancel_confirm",
        "Confirms that the run stops at the NEXT safe child boundary",
        "completed children stay immutable and reusable; nothing in flight is torn down",
        default="unchecked",
        availability="while the run is not terminal",
    ),
    _h(
        "active_runs.request_safe_cancel",
        "Writes the safe-cancel sentinel the orchestrator honours at the next child boundary",
        "cancellation is a persisted request, never a kill",
        availability="after the confirmation is checked",
    ),
    # ── Results ──
    _h(
        "results.run",
        "Which completed or running search the Results screen reads",
        "each run reads its own store, located by the exact charter id; the verification badge "
        "derives from the artifact",
        default="the most recent run",
    ),
    _h(
        "results.selected_configuration",
        "The configuration every block synchronises on — the accessible twin of the chart "
        "selection",
        "charts never provide the only route to a selection",
        default="the Development Exploratory Representative",
    ),
    _h(
        "results.heat_row",
        "The searched axis on the rows of the sensitivity heatmap",
        "cells aggregate the children sharing the row and column values (reported, never silent)",
        default="the first searched axis",
    ),
    _h(
        "results.heat_col",
        "The searched axis on the columns of the sensitivity heatmap",
        "a single-axis view collapses the columns into one reported aggregate",
        default="(single axis)",
    ),
    _h(
        "results.heat_metric",
        "The metric the heatmap colours; its colorscale follows the registered direction",
        "minimise metrics use the reversed scale so a worse value never reads green",
        default="net E[R]",
    ),
    _h(
        "results.firm_metric",
        "The prop metric of the firm compatibility matrix",
        "cells show missing / blocked reasons as text, never a fake zero",
        default="P(3 payouts before breach)",
    ),
    _h(
        "results.payout_horizon",
        "The payout horizon of the distribution and its lower-tail summary",
        "P10 leads the summary because the lower tail decides feasibility",
        default="30-day",
    ),
    _h(
        "results.preset",
        "Which column preset the explorer shows (Strategy / Prop / Robustness)",
        "the identity columns lead every preset; gross vs net is explicit per column",
        default="Strategy",
    ),
    _h(
        "results.row_detail",
        "The row whose full identity repeats above the table",
        "the pinned-column fallback keeps the exact identity copyable next to a wide table",
        default="the first row on the page",
    ),
    # ── History ──
    _h(
        "history.clone_draft",
        "Creates a new mutable draft with this draft's steps",
        "the original draft is unchanged",
    ),
    _h(
        "history.delete_permanently",
        "Permanently deletes a never-frozen archived draft after its exact name is typed",
        "owner Q2: archive is the reversible action; permanent deletion exists only in the "
        "archived view, only for drafts never frozen or launched, and only with the exact typed "
        "name",
        owner=True,
        availability="archived view; never-frozen drafts; exact typed name",
    ),
    _h(
        "history.clone_run",
        "Creates a new mutable draft from the frozen study's charter",
        "frozen evidence is never edited; a clone carries the steps into a new draft",
    ),
    _h(
        "history.display_name",
        "A new display name for the run — a mutable catalog annotation",
        "a display name never replaces the technical identity",
        default="the current display name",
    ),
    # ── Comparison / timeline ──
    _h(
        "compare.pick_child",
        "Which configuration this comparison panel uses",
        "deltas render only when populations, policies and evidence are compatible",
        default="the first configuration",
    ),
    _h(
        "compare.account_simulation",
        "Which catalogued account simulation the timeline renders",
        "the timeline follows the persisted event order by ordinal, never chart timestamps",
        default="(enter an exact id)",
    ),
    _h(
        "compare.event_ordinal",
        "Which event's linked evidence opens",
        "the total event order is the envelope order; exact ids only",
        default="the first event",
    ),
    # ── Full Pipeline Run ──
    _h(
        "pipeline.phase",
        "Which operator phase renders (Configure, Preview, Launch, Monitor, Resume / Retry, "
        "Publish)",
        "the phase radio remains until UI-6 lands the state-driven lifecycle",
        default="Configure",
    ),
    _h(
        "pipeline.bundle",
        "The feature bundle the supervised stages materialise",
        "planned / blocked bundles stay visible with their reason; the panel bundle is never a "
        "candidate view",
        default="the first available bundle",
        identity=True,
        requires=("refit",),
        availability="full 16-stage plan",
    ),
    _h(
        "pipeline.model_protocol",
        "The headline model protocol of the ladder (the prevalence reference is always included)",
        "an MBP-1-bearing bundle pins the bundle-parametrised protocols",
        default="the first available protocol",
        identity=True,
        requires=("refit",),
        availability="full 16-stage plan",
    ),
    _h(
        "pipeline.regime_algorithm",
        "The regime algorithm; V1 executes kmeans_v1 only",
        "planned post-V1 entries are visible but nothing fits them",
        default="kmeans_v1",
        identity=True,
        requires=("refit",),
        availability="regime study enabled",
    ),
    _h(
        "pipeline.regime_inputs",
        "The numeric bundle members the regime fit consumes",
        "the panel grain uses the stamped panel feature set",
        default="the stamped candidate-grain inputs",
        identity=True,
        requires=("refit",),
        availability="regime study enabled",
    ),
    _h(
        "pipeline.regime_refits",
        "Bootstrap refits per fold for the stability report",
        "the protocol-wide minimum fold AMI mean is the stability gate",
        default="50 (proposed_protocol_default; total cap 400)",
        identity=True,
        requires=("refit",),
        availability="regime study enabled",
    ),
    _h(
        "pipeline.regime_stratified",
        "Requests the S14 stratified reports (zero fitting)",
        "stratification describes results by nominal regime; it never selects",
        default="checked",
        identity=True,
        availability="regime study enabled",
    ),
    _h(
        "pipeline.regime_classes",
        "The ML §5.5 comparison classes; supervised classes require the frozen FEATURE_ELIGIBLE "
        "authority",
        "descriptive classes need no owner evidence; supervised classes fit on regime features",
        default="the descriptive classes",
        identity=True,
        owner=True,
        availability="regime study enabled",
    ),
    _h(
        "pipeline.regime_decision_id",
        "The exact frozen FEATURE_ELIGIBLE promotion decision id the supervised classes verify",
        _EXACT_ID,
        default="empty",
        identity=True,
        owner=True,
        availability="supervised classes selected",
    ),
    _h(
        "pipeline.regime_owner_id",
        "The exact frozen owner-decision artifact id the supervised classes verify",
        _EXACT_ID,
        default="empty",
        identity=True,
        owner=True,
        availability="supervised classes selected",
    ),
    _h(
        "pipeline.regime_assessment_id",
        "The exact frozen capability assessment id the supervised classes verify",
        _EXACT_ID,
        default="empty",
        identity=True,
        owner=True,
        availability="supervised classes selected",
    ),
    _h(
        "pipeline.full_scope_acknowledgement",
        "Typing the exact phrase enables the full-scope launch control",
        "the full authorized development pipeline is not an implementation verification",
        default="empty",
        owner=True,
        availability="Full Authorized Development only",
    ),
    _h(
        "pipeline.launch",
        "Freezes the pipeline specification and launches the detached worker",
        "the launch is reported as started only after the worker's persisted state exists; an "
        "unregistered executor is refused before any spawn",
        identity=True,
        owner=True,
        availability="after the satisfiability report passes and the readiness is ready",
    ),
    _h(
        "pipeline.run",
        "Which pipeline run the Monitor, Resume / Retry and Publish phases act on",
        "each run reads its own store, located by the exact charter id",
        default="the most recent run",
    ),
    _h(
        "pipeline.cancel_confirm",
        "Confirms that the run stops at the NEXT stage boundary",
        "completed semantic results stay immutable and reusable",
        default="unchecked",
        availability="while the run is not terminal",
    ),
    _h(
        "pipeline.request_safe_cancel",
        "Writes the safe-cancel sentinel the runner honours at the next stage boundary",
        "cancellation is a persisted request, never a kill",
        availability="after the confirmation is checked",
    ),
    _h(
        "pipeline.retry_reason",
        "The operational reason recorded on the new execution attempt",
        "a retry is an operational identity; a research change creates a new semantic id instead",
        default="empty",
    ),
    _h(
        "pipeline.retry",
        "Starts a new execution attempt reusing the checkpointed stages",
        "the semantic identity is unchanged; only the attempt history grows",
        availability="failed or interrupted runs",
    ),
    _h(
        "pipeline.run_gates",
        "Runs the publication gates for the selected run",
        "gates are bound to the run's verified namespace and state digest; they never activate",
        availability="completed research runs",
    ),
    _h(
        "pipeline.activate",
        "Appends the catalog activation event after passed gates",
        "verification results and cross-namespace or stale-state activations are refused",
        owner=True,
        availability="after the gates pass under the same namespace and state",
    ),
    # ── MBP-1 panel ──
    _h(
        "mbp1.coverage_report_id",
        "The exact mbp1_coverage_report_id to load",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="auto-resolved from the selected run's S05 evidence when present",
        availability="Advanced diagnostics",
    ),
    _h(
        "mbp1.feature_artifact_id",
        "The exact MBP-1 feature artifact id of the stage-window drill-down",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="auto-resolved from the selected run's S05 evidence when present",
        availability="Advanced diagnostics",
    ),
    _h(
        "mbp1.candidate_id",
        "The exact candidate id whose stage windows the drill-down shows",
        _EXACT_ID,
        default="empty",
        availability="Advanced diagnostics",
    ),
    _h(
        "mbp1.controlled_study_id",
        "The exact controlled_feature_study_id of the Baseline vs Baseline+MBP-1 comparison",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="auto-resolved from the selected run's S09 outputs when present",
        availability="Advanced diagnostics",
    ),
    # ── Regime lane ──
    _h(
        "regime.protocol_id",
        "The exact resolved regime protocol id the model card loads",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="auto-resolved from the selected run's S10 diagnostics when present",
        availability="Advanced diagnostics",
    ),
    _h(
        "regime.assessment_id",
        "The exact capability assessment id (coverage, occupancy, stability, gates)",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="auto-resolved from the selected run's S10 diagnostics when present",
        availability="Advanced diagnostics",
    ),
    _h(
        "regime.fit_id",
        "The exact regime fit id of the assignment / stratification view",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="the first fit of the selected run when present",
        availability="Advanced diagnostics",
    ),
    _h(
        "regime.decision_id",
        "The exact promotion decision id of the role / status view",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="the final decision of the selected run when present",
        availability="Advanced diagnostics",
    ),
    _h(
        "regime.report_id",
        "The exact stratified report id of the stratified result view",
        f"{_STORES_NEVER_LISTED}; {_EXACT_ID}",
        default="the first stratified report of the selected run when present",
        availability="Advanced diagnostics",
    ),
    # ── Verification Center ──
    _h(
        "center.start_verification_draft",
        "Creates a session-only verification draft (synthetic fixture) carrying the "
        "Implementation Verification purpose",
        "a synthetic fixture proves the machinery in the test namespace and never becomes "
        "verification evidence",
        availability="always on this screen",
    ),
    _h(
        "center.refresh_seed",
        "Re-reads the seed receipts and the persisted authorization state",
        "the seed steps run outside Streamlit; refresh picks their receipts up by exact id",
    ),
    _h(
        "center.validate_reference",
        "Re-reads and validates the owner's completed VerificationAuthorizationRef",
        "the center validates the reference typed and never persists it",
        availability="after a verified seed",
    ),
    _h(
        "center.preflight_again",
        "Re-runs the bounded-verification preflight over the persisted state",
        "the exact bounded-run command appears only after the preflight passes",
        availability="after the run is registered",
    ),
    # ── Context Research ──
    _h(
        "context.artifact_pair",
        "The verified v2 / v3 artifact pair the experiment reads",
        "pairs are loaded by exact ids and manifest-verified on every load",
        default="the first context-ready pair",
    ),
    _h(
        "context.saved_run",
        "Which immutable context run renders below",
        "every run is verified against its manifest before it renders",
        default="the first catalogued run",
    ),
    _h(
        "context.compare_with",
        "The second immutable run of the comparison",
        "quantitative deltas render only when the two runs reconcile; the reasons are listed",
        default="the next catalogued run",
        availability="two or more catalogued runs",
    ),
    _h(
        "context.feature_tier",
        "The registered feature tier (M0, M1, M2, M3 or the experimental M1 + 240m)",
        "the tier fixes the feature set and enters the run identity",
        default="M2",
        identity=True,
        requires=("refit",),
    ),
    _h(
        "context.target",
        "The counterfactual label target (an R multiple or fixed SL / TP in ticks)",
        "the target fixes the label derivation and enters the run identity",
        default="R1.0",
        identity=True,
        requires=("refit",),
    ),
    _h(
        "context.cohort",
        "The observation cohort by trading-day range",
        "the cohort fixes the candidate population and enters the run identity",
        default="prior research",
        identity=True,
        requires=("refit",),
    ),
    _h(
        "context.fixed_stop",
        "The fixed stop distance of the fixed SL / TP label",
        "the stop distance defines 1R for the fixed label family",
        default="16 ticks",
        identity=True,
        requires=("refit",),
        availability="Fixed SL / TP target",
    ),
    _h(
        "context.fixed_target",
        "The fixed target distance of the fixed SL / TP label",
        "the target distance defines the counterfactual win for the fixed label family",
        default="16 ticks",
        identity=True,
        requires=("refit",),
        availability="Fixed SL / TP target",
    ),
    _h(
        "context.run_experiment",
        "Runs the fixed walk-forward protocol and catalogs the immutable run",
        "an identical run is verified and reused; nothing here searches, selects or promotes",
        requires=("refit",),
    ),
    _h(
        "verifier.outcome_filter",
        "Keeps only the candidates in the selected outcome states",
        "executed, blocked and censored candidates are different evidence classes",
        default="all",
    ),
    _h(
        "verifier.m3_only",
        "Keeps only the M3-qualifying candidates",
        "M3 qualification is the equal-level pool and sweep context",
        default="unchecked",
    ),
    _h(
        "verifier.session_filter",
        "Keeps only the candidates entered in the selected sessions",
        "sessions are the engine's ET windows",
        default="all",
    ),
    _h(
        "verifier.include_warmup",
        "Includes candidates from the warmup days",
        "warmup evidence is excluded from research by default",
        default="unchecked",
    ),
    _h(
        "verifier.day_range",
        "The trading-day range of the candidate list",
        "the range narrows the list; it never changes the evidence",
        default="the full range",
    ),
    _h(
        "verifier.detail_verdict",
        "An optional per-aspect verdict recorded with the review",
        "detail verdicts are stored beside the overall verdict; nothing is written until Save "
        "Review",
        default="Unreviewed",
    ),
    _h(
        "verifier.review_tags",
        "Tags recorded with the review",
        "tags are review annotations, never evidence",
        default="none",
    ),
    _h(
        "verifier.review_notes",
        "Free-text notes recorded with the review",
        "notes are review annotations, visibly separated from the immutable evidence",
        default="empty",
    ),
    _h(
        "verifier.download_ledger",
        "Downloads the append-only review ledger as CSV",
        "the ledger is exported, never edited",
    ),
    _h(
        "verifier.setup_candidate_less",
        "Keeps setups with or without a candidate",
        "candidate-less setups died before an entry candidate formed",
        default="all",
    ),
    _h(
        "verifier.setup_terminal_reason",
        "Keeps only setups that ended for the selected reasons",
        "the terminal reason is the FSM's persisted exit",
        default="all",
    ),
    _h(
        "verifier.setup_phase",
        "Keeps only setups that died in the selected phases",
        "the phase at death locates where the lifecycle stopped",
        default="all",
    ),
    _h(
        "verifier.setup_conflict",
        "Keeps only conflict-flagged setups",
        "the count shows how many setups carry the flag on this artifact",
        default="unchecked",
    ),
    _h(
        "verifier.setup_suppression",
        "Keeps only setups suppressed by structure",
        "the count shows how many setups carry the flag on this artifact",
        default="unchecked",
    ),
    _h(
        "verifier.setup_htf_tf",
        "Keeps only setups whose tapped zone is on the selected higher timeframes",
        "the HTF timeframe starts a setup",
        default="all",
    ),
    _h(
        "verifier.setup_parent_tf",
        "Keeps only setups whose parent zone is on the selected timeframes",
        "the parent timeframe is the locked structure",
        default="all",
    ),
    _h(
        "verifier.setup_session",
        "Keeps only setups activated in the selected documented sessions",
        "the session at activation is the persisted document session",
        default="all",
    ),
    _h(
        "verifier.setup_q40",
        "Keeps setups by their Q-40 exposure",
        "Q-40 exposure marks the experimental 240m anchor (open owner question)",
        default="all",
    ),
    _h(
        "verifier.setup_parentless",
        "Keeps setups that were ever parentless (had at least one interval without a parent)",
        "parentless intervals are a lifecycle fact the FSM persists",
        default="all",
    ),
    _h(
        "verifier.setup_warmup",
        "Includes setups from the warmup days",
        "warmup evidence is excluded from research by default",
        default="unchecked",
    ),
    _h(
        "verifier.setup_activation_from",
        "The earliest activation time kept (UTC ISO)",
        "an empty bound means no lower limit",
        default="empty",
    ),
    _h(
        "verifier.setup_activation_to",
        "The latest activation time kept (UTC ISO)",
        "an empty bound means no upper limit",
        default="empty",
    ),
    _h(
        "verifier.setup_display_end_from",
        "The earliest display-end time kept (UTC ISO)",
        "an empty bound means no lower limit",
        default="empty",
    ),
    _h(
        "verifier.setup_display_end_to",
        "The latest display-end time kept (UTC ISO)",
        "an empty bound means no upper limit",
        default="empty",
    ),
    _h(
        "verifier.setup_jump",
        "Jumps to a setup by its exact id or a unique prefix",
        "a non-unique prefix or an unknown id is reported, never guessed",
        default="empty",
    ),
    _h(
        "verifier.exact_setup_id",
        "The exact setup rendered",
        "the option label carries the setup facts; the value is the exact id",
        default="the first setup in the filtered list",
    ),
    _h(
        "verifier.setup_stage",
        "The lifecycle stage up to which setup evidence is shown",
        "evidence after the chosen stage is hidden; terminal shows the full life",
        default="terminal",
    ),
    _h(
        "verifier.exact_candidate_id",
        "The exact candidate rendered",
        "the option label carries the case facts; the value is the exact id",
        default="the first candidate in the filtered list",
    ),
    _h(
        "verifier.mode",
        "Full audit shows every persisted fact including the outcome; Point-in-time hides "
        "everything after the chosen stage",
        "point-in-time review prevents outcome knowledge from colouring the judgement",
        default="Full audit",
    ),
    _h(
        "verifier.stage_scrubber",
        "The lifecycle stage the point-in-time evidence is shown as of",
        "later stages and the outcome stay hidden",
        default="the last gateable stage",
        availability="Point-in-time mode",
    ),
    _h(
        "verifier.range",
        "The visible time range of the chart (setup, trade, formation or custom)",
        "the range changes what is drawn, never what is persisted",
        default="Setup",
    ),
    _h(
        "verifier.layer_structure",
        "Draws the structure overlays (swings and structure levels)",
        "overlays are display layers over the same persisted evidence",
        default="unchecked",
    ),
    _h(
        "verifier.layer_displacement",
        "Draws the displacement windows",
        "overlays are display layers over the same persisted evidence",
        default="unchecked",
    ),
    _h(
        "verifier.layer_pools",
        "Draws the equal-high / equal-low pools and their sweeps",
        "overlays are display layers over the same persisted evidence",
        default="unchecked",
    ),
    _h(
        "verifier.layer_sessions",
        "Shades the session windows",
        "sessions are the engine's ET windows",
        default="checked",
    ),
    _h(
        "verifier.layer_projection",
        "Projects the higher-timeframe zones onto the 1m execution pane",
        "projection keeps the zones visible when the parent panes are hidden",
        default="checked",
    ),
    _h(
        "verifier.custom_start",
        "The custom range start (UTC ISO)",
        "both custom bounds are required",
        default="empty",
        availability="Custom range",
    ),
    _h(
        "verifier.custom_end",
        "The custom range end (UTC ISO)",
        "both custom bounds are required",
        default="empty",
        availability="Custom range",
    ),
)

HELP_REGISTRY: Mapping[str, HelpEntry] = MappingProxyType(
    {entry.control_id: entry for entry in _ENTRIES}
)
if len(HELP_REGISTRY) != len(_ENTRIES):  # pragma: no cover — registry integrity
    raise RuntimeError("duplicate help control ids")


def help_text(control_id: str) -> str:
    """The tooltip text of a registered control (every field rendered)."""

    try:
        entry = HELP_REGISTRY[str(control_id)]
    except KeyError:
        raise ValueError(f"unregistered help control {control_id!r}") from None
    requires = ", ".join(entry.requires) if entry.requires else "none"
    return (
        f"What it changes: {entry.what_it_changes}. "
        f"Why: {entry.why}. "
        f"Default: {entry.default}. "
        f"Identity: {'new identity' if entry.changes_identity else 'unchanged'} · "
        f"Requires: {requires} · "
        f"Owner approval: {'required' if entry.owner_approval else 'not required'} · "
        f"Availability: {entry.availability}."
    )


_DIRECTION_WORDS = {
    "higher_better": "higher is better",
    "lower_better": "lower is better",
    "target": "closer to the target is better",
    "descriptive": "descriptive — no direction",
}


def help_for_metric(technical_key: str) -> str:
    """The tooltip text of a registry metric (definition, source, direction, reference)."""

    spec = describe(technical_key)
    reference = (
        f" Reference: {spec.reference.source}."
        if spec.reference is not None
        else " Reference: none — no gate applies."
    )
    return (
        f"{spec.human_name} ({spec.technical_key}): {spec.definition} "
        f"Source: {spec.formula_or_source}. Unit: {spec.unit}; "
        f"{_DIRECTION_WORDS[spec.directionality]}.{reference}"
    )


# ── glossary (plan §6.4) ────────────────────────────────────────────────────


class GlossaryEntry(FrozenContract):
    term: str
    expansion: str
    definition: str


GLOSSARY_TERMS: tuple[str, ...] = (
    "FVG",
    "IFVG",
    "FSM",
    "HTF",
    "LTF",
    "OOS",
    "PIT",
    "MBP-1",
    "R multiple",
    "Brier",
    "Brier skill",
    "AUC",
    "EQH/EQL",
    "MFE/MAE",
    "AMI",
    "Q-40",
)


def _g(term: str, expansion: str, definition: str) -> GlossaryEntry:
    return GlossaryEntry(term=term, expansion=expansion, definition=definition)


GLOSSARY: Mapping[str, GlossaryEntry] = MappingProxyType(
    {
        entry.term: entry
        for entry in (
            _g(
                "FVG",
                "Fair value gap",
                "A three-bar price imbalance that leaves an unfilled gap between the first "
                "bar's extreme and the third bar's extreme; the zones the setup engine tracks.",
            ),
            _g(
                "IFVG",
                "Inversion fair value gap",
                "A fair value gap that price closed through, inverting its role from support "
                "to resistance or the reverse; the strategy's entry-zone family.",
            ),
            _g(
                "FSM",
                "Finite-state machine",
                "The setup lifecycle engine (tap → parent → lock → armed → inversion → entry → "
                "resolution) whose registered parameters a configuration search varies.",
            ),
            _g(
                "HTF",
                "Higher timeframe",
                "The timeframe of the tapped zone that starts a setup; larger than the "
                "execution timeframe.",
            ),
            _g(
                "LTF",
                "Lower timeframe",
                "The execution timeframe (one-minute bars) on which entries and resolutions "
                "are evaluated.",
            ),
            _g(
                "OOS",
                "Out-of-sample",
                "Predictions or assignments made on rows outside the fold they were fitted on "
                "— the walk-forward test days — never on the training rows.",
            ),
            _g(
                "PIT",
                "Point-in-time",
                "Evidence exactly as it existed at a chosen lifecycle stage; later evidence, "
                "including the outcome, is hidden from the reviewer.",
            ),
            _g(
                "MBP-1",
                "Market-by-price, one level",
                "Top-of-book order-flow data; the activated MBP-1 feature block is "
                "research-only offline and never a serving or execution feature.",
            ),
            _g(
                "R multiple",
                "Risk multiple",
                "The profit or loss of a trade expressed as a multiple of its initial risk — "
                "the entry-to-stop distance is 1R.",
            ),
            _g(
                "Brier",
                "Brier score",
                "The mean squared error between the predicted probability and the 0/1 "
                "outcome; lower is better; read against the prevalence-reference Brier.",
            ),
            _g(
                "Brier skill",
                "Brier skill score",
                "One minus the Brier score over the reference Brier score; positive means "
                "the model beats the training-prevalence reference, zero means no skill.",
            ),
            _g(
                "AUC",
                "Area under the ROC curve",
                "The ranking quality of the predicted probabilities; 0.5 is the chance line; "
                "no good / bad band is applied and it is undefined for a single-class set.",
            ),
            _g(
                "EQH/EQL",
                "Equal highs / equal lows",
                "Pools of equal-level liquidity the context features track through their "
                "lifecycle and the sweeps that qualify against them.",
            ),
            _g(
                "MFE/MAE",
                "Maximum favourable / adverse excursion",
                "The furthest price moved for (MFE) and against (MAE) a trade between entry "
                "and resolution, in ticks.",
            ),
            _g(
                "AMI",
                "Adjusted mutual information",
                "The agreement between two clusterings adjusted for chance; the regime "
                "lane's bootstrap stability measure after label alignment.",
            ),
            _g(
                "Q-40",
                "Open owner question 40 (the 240-minute anchor)",
                "The frozen flag marking setups exposed to the experimental 240m structure "
                "context; the 240m block stays outside every primary tier until the owner "
                "decides.",
            ),
        )
    }
)


def glossary_markdown() -> str:
    return "\n".join(
        f"- **{term}** — {GLOSSARY[term].expansion}: {GLOSSARY[term].definition}"
        for term in GLOSSARY_TERMS
    )


# ── explicit exemptions (plan §6.4): self-explanatory navigation only ──────

_NAV = "self-explanatory navigation control; the destination is the label"
_SAVE = "self-explanatory: persists the draft in place; the Saved / Autosaved chip shows the state"
_STEP = "self-explanatory linear step control; the breadcrumb names the steps"

HELP_EXEMPTIONS: Mapping[tuple[str, str], str] = MappingProxyType(
    {
        ("ifvg_study_wizard.py", "Open"): _NAV,
        ("ifvg_study_wizard.py", "Back"): _STEP,
        ("ifvg_study_wizard.py", "Next"): _STEP,
        ("ifvg_study_wizard.py", "Save Draft"): _SAVE,
        ("ifvg_results_tab.py", "Open"): _NAV,
        ("ifvg_results_tab.py", "Restore"): "self-explanatory: un-archives the draft (owner Q2)",
        ("ifvg_results_tab.py", "Results"): _NAV,
        ("ifvg_results_tab.py", "Rename"): (
            "self-explanatory: applies the typed display name (a catalog annotation)"
        ),
        ("ifvg_results_compare.py", "Open exact"): _NAV,
        ("ifvg_results_compare.py", "Open timeline"): _NAV,
        ("ifvg_verifier_tab.py", "◀ Prev setup"): _NAV,
        ("ifvg_verifier_tab.py", "Next setup ▶"): _NAV,
        ("ifvg_verifier_tab.py", "◀ Prev candidate"): _NAV,
        ("ifvg_verifier_tab.py", "Next candidate ▶"): _NAV,
        ("ifvg_verifier_tab.py", "◀ Prev trade"): (
            "navigation over the executed subset; the caption explains trades are executed "
            "candidates"
        ),
        ("ifvg_verifier_tab.py", "Next trade ▶"): (
            "navigation over the executed subset; the caption explains trades are executed "
            "candidates"
        ),
    }
)
