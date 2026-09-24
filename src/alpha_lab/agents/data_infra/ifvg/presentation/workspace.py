"""Read-only study view models. Scientific records and exact keys stay intact."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..study_drafts import StudyDraft, list_drafts
from ..study_providers import (
    artifact_scope_for_charter,
    catalog_annotations,
    list_pipeline_runs,
    list_search_runs,
    load_charter,
    load_search_state,
)

QUESTIONS = {
    "evaluate_one_configuration": "How did this configuration perform?",
    "compare_one_with_baseline": "Does this configuration improve on the baseline?",
    "find_robust_fsm": "Which settings perform consistently?",
    "repeat_payout_feasibility": "How reliably could this strategy reach payouts?",
    "universal_prop": "How does this strategy compare across firms?",
    "one_config_across_firms": "How does this strategy compare across firms?",
}
PRIMARY_METRICS = ("net_expectancy_r", "profit_factor", "max_drawdown_r", "executed_trades")
_TECHNICAL = re.compile(r"[0-9a-f]{12,}|[A-Za-z]:[\\/]|\b\w+_\w+\b", re.I)


def failure_explanation(reason: str) -> str:
    return {
        "replay": "A configuration could not be evaluated. Restore its data or runner and resume.",
        "invariant": (
            "Strategy evidence failed a correctness check. "
            "Restore verified evidence before resuming."
        ),
        "insufficient_days": (
            "Too few independent days support this estimate. Clone with a longer permitted period."
        ),
        "insufficient_trades": (
            "Too few trades support this estimate. Clone with a longer permitted period."
        ),
        "negative_expectancy": "Net expectancy did not meet the study's required threshold.",
        "drawdown": "Drawdown exceeded the study's limit.",
        "funded_survival": "Simulated funded-account survival did not meet the required threshold.",
        "breach": "Simulated account breach risk exceeded the study's limit.",
        "fees": "Simulated fees prevented the required result.",
        "one_firm": "The result did not meet the required criteria across firms.",
        "stress": "The result did not hold under the study's stress scenarios.",
        "knife_edge": "Nearby settings did not provide the required consistency.",
        "unverified_contract": (
            "A firm contract is not verified. Restore its verified terms before resuming."
        ),
        "blocked_axis": (
            "A selected setting is unavailable. Clone the study to select supported settings."
        ),
        "cancelled": "Work stopped at a safe boundary. Resume to continue unfinished work.",
    }.get(
        reason, "A required check could not be resolved. Restore the study's evidence and refresh."
    )


def human_name(value: Any, fallback: str = "Untitled study") -> str:
    text = str(value or "").strip()
    return fallback if not text or _TECHNICAL.search(text) else text


def profile_name(value: str) -> str:
    from .labels import PROFILE_LABELS

    if value in PROFILE_LABELS:
        return PROFILE_LABELS[value].replace("Doc-default", "Baseline")
    return human_name(value, "Saved configuration")


def axis_value_name(value_id: str) -> str:
    from ..search.axis_registry import AXIS_VALUE_REGISTRY_V1, SEARCH_AXIS_REGISTRY_V1
    from .axis_values import format_axis_value

    value = AXIS_VALUE_REGISTRY_V1.get(value_id)
    if value is None:
        return "Unavailable setting"
    axis_key = getattr(value, "axis_technical_key", "")
    spec = SEARCH_AXIS_REGISTRY_V1.get(axis_key)
    label = value.human_label
    if spec:
        # Derive display text from the typed payload, including registries loaded
        # before clearer human labels were added. Saved drafts keep value IDs.
        label = f"{spec.human_label} — {format_axis_value(axis_key, value.payload)}"
        if value_id == spec.baseline_value_id:
            label += " (default)"
    return label.replace("_", " ")


def configuration_name(values: Mapping[str, str]) -> str:
    from ..search.axis_registry import SEARCH_AXIS_REGISTRY_V1
    from ..study_presentation import human_config_name

    label = human_config_name(values)
    for key, spec in SEARCH_AXIS_REGISTRY_V1.items():
        label = label.replace(key, spec.human_label)
    return human_name(label.replace("_", " "), "Saved configuration")


def date_scope(days: Any) -> str:
    dates = sorted(
        str(day)[:10] for day in (days or ()) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(day)[:10])
    )
    if not dates:
        return "Dates not selected"
    return dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]} · {len(dates)} days"


def search_status(state: Mapping[str, Any] | None) -> str:
    if state is None:
        return "Evidence unavailable"
    phase = state.get("phase")
    if phase == "search_complete":
        return "Completed"
    if phase == "failed":
        return "Failed"
    if phase == "cancelled":
        return "Interrupted"
    if phase in ("created", "queued"):
        return "Ready"
    # Only known persisted running phases are interpreted as activity.
    from ..search.orchestrator import SEARCH_PHASES

    return "Running" if phase in SEARCH_PHASES else "Status unresolved"


def pipeline_status(state: Mapping[str, Any] | None) -> str:
    if state is None:
        return "Evidence unavailable"
    stages = [
        row.get("status")
        for row in (state.get("stages") or {}).values()
        if isinstance(row, Mapping) and row.get("in_plan", True)
    ]
    if "running" in stages:
        return "Running"
    if "failed" in stages:
        return "Failed"
    if "cancelled_at_safe_boundary" in stages:
        return "Interrupted"
    if "blocked" in stages:
        return "Blocked"
    if stages and all(value in ("completed", "reused", "not_applicable") for value in stages):
        return "Completed"
    return (
        "Ready" if stages and all(value == "pending" for value in stages) else "Status unresolved"
    )


@dataclass(frozen=True)
class SearchProgress:
    """Exclusive saved child states; failures never count as queued work."""

    total: int
    completed: int = 0
    reused: int = 0
    failed: int = 0
    running: int = 0
    queued: int = 0
    stopped: int = 0
    blocked: int = 0
    unresolved: int = 0

    @property
    def attempted(self) -> int:
        return self.completed + self.reused + self.failed + self.running


def search_progress(state: Mapping[str, Any]) -> SearchProgress:
    rows = list(state.get("children") or ())
    counts = {
        key: sum(isinstance(row, Mapping) and row.get("state") == key for row in rows)
        for key in (
            "completed", "reused", "failed", "running", "queued",
            "cancelled_at_safe_boundary", "blocked",
        )
    }
    return SearchProgress(
        total=len(rows),
        completed=counts["completed"],
        reused=counts["reused"],
        failed=counts["failed"],
        running=counts["running"],
        queued=counts["queued"],
        stopped=counts["cancelled_at_safe_boundary"],
        blocked=counts["blocked"],
        unresolved=len(rows) - sum(counts.values()),
    )


def search_activity(state: Mapping[str, Any]) -> str:
    phase = state.get("phase")
    if phase == "replays":
        progress = search_progress(state)
        if progress.total and not (progress.queued or progress.running or progress.unresolved):
            return "Finalizing configuration results"
        return "Evaluating configurations"
    return {
        "charter_frozen": "Verifying study inputs",
        "children_enumerated": "Preparing input artifacts",
        "artifacts_prewarmed": "Preparing configuration replays",
        "underlying_edge_passed": "Preparing account simulations",
        "prop_simulations": "Evaluating account simulations",
        "prop_feasible": "Checking robustness",
        "robustness_passed": "Comparing qualifying configurations",
        "frontier_complete": "Saving study results",
    }.get(phase, "Working on the study")


def elapsed(state: Mapping[str, Any]) -> str:
    try:
        attempts = state.get("attempts") or ()
        attempt = attempts[-1] if attempts else {}
        start = datetime.fromisoformat(
            str(
                attempt.get("started_at")
                or state.get("attempt_started_at_utc")
                or state.get("started_at_utc")
                or state.get("created_at_utc")
            )
        )
        end_raw = (
            attempt.get("ended_at")
            or state.get("attempt_finished_at_utc")
            or state.get("finished_at_utc")
        )
        if "phase" in state:
            if search_status(state) != "Running":
                end_raw = end_raw or state.get("paused_at_utc") or state.get("updated_at_utc")
                if not end_raw:
                    return "Not recorded"
        else:
            # Preserve the existing pipeline attempt timing contract.
            end_raw = end_raw or state.get("updated_at_utc")
        end = datetime.fromisoformat(str(end_raw)) if end_raw else datetime.now(start.tzinfo)
        minutes = max(0, int((end - start).total_seconds() // 60))
        return f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes}m"
    except (ValueError, TypeError):
        return "Not recorded"


@dataclass(frozen=True)
class StudySummary:
    key: str
    kind: str
    name: str
    question: str
    dates: str
    status: str
    scope: str
    updated: str = ""
    archived: bool = False
    store_root: Path | None = None
    draft: StudyDraft | None = field(default=None, compare=False, repr=False)
    state: Mapping[str, Any] | None = field(default=None, compare=False, repr=False)
    charter_id: str | None = None

    @property
    def next_action(self) -> str:
        if self.scope == "unresolved":
            return "Inspect scope"
        return {
            "Draft": "Continue",
            "Ready": "Run",
            "Running": "View progress",
            "Completed": "View results",
            "Failed": "Review recovery",
            "Interrupted": "Review recovery",
            "Blocked": "Review issue",
        }.get(self.status, "Inspect study")


def _draft_scope(draft: StudyDraft) -> str:
    from .run_purpose import resolve_draft_purpose

    # Use the existing legacy derivation rules, never guess from a directory.
    validation = draft.steps.get("validation", {})
    resolution = resolve_draft_purpose(
        draft.purpose_annotation,
        run_scope=validation.get("run_scope"),
        stage_plan_kind=validation.get("stage_plan_kind"),
        authorization_kind=validation.get("authorization_kind"),
    )
    if resolution.purpose is None:
        return "unresolved"
    return (
        "verification" if resolution.purpose.value == "implementation_verification" else "research"
    )


def load_studies(
    roots: Mapping[str, Any], *, include_verification: bool = False
) -> tuple[list[StudySummary], list[str]]:
    """Unify mutable listings; every linked scientific record uses its exact ID."""
    result: list[StudySummary] = []
    issues: list[str] = []
    stores = list(
        dict.fromkeys(
            [Path(roots["store_root"]), *(Path(p) for p in roots.get("store_roots", {}).values())]
        )
    )
    try:
        drafts = list_drafts(Path(roots["draft_root"]), include_archived=True)
        saved = set(Path(roots["draft_root"]).glob("*/draft.json"))
        if len(saved) > len(drafts):
            issues.append(
                "Some saved drafts are unreadable. Other studies remain available; "
                "restore the damaged drafts from a saved copy to resume them."
            )
    except Exception:
        drafts = []
        issues.append(
            "Saved drafts could not be read. Refresh after the storage issue is resolved."
        )
    linked = {draft.frozen_search_id: draft for draft in drafts if draft.frozen_search_id}
    try:
        searches = list_search_runs(Path(roots["state_root"]), stores[0], store_roots=stores)
    except Exception:
        searches = ()
        issues.append(
            "Study progress could not be read. Some studies may be missing from this list."
        )
    try:
        pipelines = list_pipeline_runs(Path(roots["pipeline_state_root"]), store_roots=stores)
    except Exception:
        pipelines = ()
        issues.append(
            "Workflow progress could not be read. Some studies may be missing from this list."
        )
    seen_charters: set[str] = set()
    for run in [*pipelines, *searches]:
        pipeline = hasattr(run, "pipeline_semantic_id")
        charter_id = run.search_charter_id if pipeline else run.search_id
        if not pipeline and charter_id in seen_charters:
            continue
        key = run.pipeline_semantic_id if pipeline else run.search_id
        draft = linked.get(charter_id)
        store = Path(run.store_root) if run.store_root else None
        try:
            charter = load_charter(store, charter_id) if store and charter_id else None
            scope = (
                "unresolved"
                if charter is None
                else "verification"
                if artifact_scope_for_charter(charter).verification_only
                else "research"
            )
        except Exception:
            charter, scope = None, "unresolved"
        try:
            if pipeline:
                from ..search.pipeline import read_pipeline_state

                state = read_pipeline_state(Path(roots["pipeline_state_root"]), key)
            else:
                state = load_search_state(Path(roots["state_root"]), key)
            status = pipeline_status(state) if pipeline else search_status(state)
        except Exception:
            state, status = None, "Evidence unavailable"
        if scope == "verification" and not include_verification:
            continue
        if charter_id:
            seen_charters.add(charter_id)
        payload = charter.payload if charter else None
        dates = getattr(getattr(payload, "date_policy", None), "replay_dates", ())
        objective = draft.steps.get("objective", {}) if draft else {}
        if draft:
            dates = draft.steps.get("validation", {}).get("real_dates", dates)
        annotation_name = None
        annotation_archived = False
        if store and charter_id:
            try:
                saved_annotation = catalog_annotations(store).get(charter_id, {})
                annotation = saved_annotation.get("display_name")
                annotation_name = (
                    annotation.get("display_name")
                    if isinstance(annotation, Mapping)
                    else annotation
                )
                annotation_archived = bool(saved_annotation.get("archived"))
            except Exception:
                issues.append(
                    "Study names and archive status could not be read. "
                    "Restore the research catalog and refresh."
                )
        result.append(
            StudySummary(
                key=key,
                kind="pipeline" if pipeline else "search",
                name=human_name(
                    annotation_name
                    or (draft.display_name if draft else getattr(run, "display_name", "")),
                    "Workflow study" if pipeline else "Strategy study",
                ),
                question=QUESTIONS.get(
                    objective.get("question_id"), "How did this research study perform?"
                ),
                dates=date_scope(dates),
                status=status,
                scope=scope,
                updated=str(
                    (state or {}).get("updated_at_utc") or (draft.updated_at_utc if draft else "")
                ),
                archived=bool(
                    annotation_archived
                    or getattr(run, "archived", False)
                    or (draft and draft.archived)
                ),
                store_root=store,
                draft=draft,
                state=state,
                charter_id=charter_id,
            )
        )
    funded_mode = "funded_payout_simulation"
    try:
        from ifvg_funded_study import list_funded_runs

        funded_runs = list_funded_runs(roots, drafts)
    except ImportError:  # the funded screen lives in scripts/
        funded_runs = []
    except Exception:
        funded_runs = []
        issues.append("Funded payout simulations could not be listed. Refresh to retry.")
    for run in funded_runs:
        name = run.draft.display_name if run.draft else "Funded payout simulation"
        result.append(
            StudySummary(
                key=run.plan_id,
                kind="funded",
                name=human_name(name, "Funded payout simulation"),
                question="Earlier five-account check (shared signal): which funded operation "
                         "earned the most cash after account costs?",
                dates=run.dates,
                status=str(run.state.get("status", "Evidence unavailable")),
                scope="research",
                updated=str(run.state.get("updated_at_utc", "")),
                archived=bool(run.draft and run.draft.archived),
                draft=run.draft,
                state=run.state,
            )
        )
    comparison_mode = "funded_configuration_comparison"
    try:
        from ifvg_funded_comparison_study import list_comparison_runs

        comparison_runs = list_comparison_runs(roots, drafts)
    except ImportError:  # the comparison screen lives in scripts/
        comparison_runs = []
    except Exception:
        comparison_runs = []
        issues.append("Funded configuration comparisons could not be listed. Refresh to retry.")
    for run in comparison_runs:
        title = getattr(run, "title", "Funded configuration comparison")
        name = human_name(run.draft.display_name, title) if run.draft else title
        result.append(
            StudySummary(
                key=run.plan_id,
                kind="funded_comparison",
                name=human_name(name, "Funded configuration comparison"),
                question="Which configuration earned the most cash after every account cost?",
                dates=run.dates,
                status=str(run.state.get("status", "Evidence unavailable")),
                scope="research",
                updated=str(run.state.get("updated_at_utc", "")),
                archived=bool(run.draft and run.draft.archived),
                draft=run.draft,
                state=run.state,
            )
        )
    for draft in drafts:
        if draft.frozen_search_id in seen_charters:
            continue
        if draft.mode_id == comparison_mode:
            if draft.frozen_search_id:
                continue  # listed through its run above
            result.append(
                StudySummary(
                    key=draft.draft_id, kind="draft", name=human_name(draft.display_name),
                    question="Which configuration earned the most cash after every account "
                             "cost?",
                    dates="Dates of the chosen completed study", status="Draft",
                    scope="research",
                    updated=draft.updated_at_utc, archived=draft.archived, draft=draft,
                )
            )
            continue
        if draft.mode_id == funded_mode:
            if draft.frozen_search_id:
                continue  # listed through its run above
            result.append(
                StudySummary(
                    key=draft.draft_id, kind="draft", name=human_name(draft.display_name),
                    question="Which funded operation earned the most cash after account costs?",
                    dates="Dates of the chosen completed study", status="Draft",
                    scope="research",
                    updated=draft.updated_at_utc, archived=draft.archived, draft=draft,
                )
            )
            continue
        scope = _draft_scope(draft)
        store = None
        if draft.frozen_search_id:
            from ..study_providers import locate_charter_store

            try:
                store = locate_charter_store(draft.frozen_search_id, stores)
                charter = load_charter(store, draft.frozen_search_id) if store else None
                scope = (
                    "unresolved"
                    if charter is None
                    else "verification"
                    if artifact_scope_for_charter(charter).verification_only
                    else "research"
                )
            except Exception:
                scope = "unresolved"
        if scope == "verification" and not include_verification:
            continue
        result.append(
            StudySummary(
                key=draft.draft_id,
                kind="draft",
                name=human_name(draft.display_name),
                question=QUESTIONS.get(
                    draft.steps.get("objective", {}).get("question_id"),
                    "Choose a research question",
                ),
                dates=date_scope(draft.steps.get("validation", {}).get("real_dates")),
                status=("Evidence unavailable" if draft.mode_id == "full_pipeline_run" else "Ready")
                if draft.frozen_search_id
                else "Draft",
                scope=scope,
                updated=draft.updated_at_utc,
                archived=draft.archived,
                draft=draft,
                charter_id=draft.frozen_search_id,
                store_root=store,
            )
        )
    if roots.get("context_catalog"):
        from ..context_run_store import list_context_run_catalog, load_context_experiment_run

        try:
            entries = list_context_run_catalog(catalog_path=Path(roots["context_catalog"]))
        except Exception:
            entries = []
            issues.append(
                "Saved model studies could not be listed. Restore the research catalog and refresh."
            )
        for index, entry in enumerate(entries):
            try:
                saved = load_context_experiment_run(
                    entry["run_id"], base_dir=Path(roots["context_run_root"])
                )
                days = saved.predictions.get("trading_day", ())
                status = "Completed" if saved.result.status == "complete" else "Inconclusive"
                scope = "research"
            except Exception:
                days, status, scope = (), "Evidence unavailable", "unresolved"
            result.append(
                StudySummary(
                    key=entry["run_id"],
                    kind="context",
                    name=human_name(
                        entry.get("display_name"), f"Context feature study {index + 1}"
                    ),
                    question="Do context features improve hypothetical outcome predictions?",
                    dates=date_scope(sorted(set(days))),
                    status=status,
                    scope=scope,
                )
            )
    return sorted(result, key=lambda study: study.updated, reverse=True), issues
