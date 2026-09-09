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

    value = AXIS_VALUE_REGISTRY_V1.get(value_id)
    if value is None:
        return "Unavailable setting"
    axis_key = getattr(value, "axis_technical_key", "")
    spec = SEARCH_AXIS_REGISTRY_V1.get(axis_key)
    label = value.human_label
    if spec:
        label = label.replace(axis_key, spec.human_label)
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


def elapsed(state: Mapping[str, Any]) -> str:
    try:
        attempts = state.get("attempts") or ()
        attempt = attempts[-1] if attempts else {}
        start = datetime.fromisoformat(
            str(
                attempt.get("started_at")
                or state.get("started_at_utc")
                or state.get("created_at_utc")
            )
        )
        end_raw = (
            attempt.get("ended_at") or state.get("finished_at_utc") or state.get("updated_at_utc")
        )
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
    for draft in drafts:
        if draft.frozen_search_id in seen_charters:
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
