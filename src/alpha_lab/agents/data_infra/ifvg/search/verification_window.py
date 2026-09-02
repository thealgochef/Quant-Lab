"""The logical-window coverage shortlist for the owner's window selection
(§5.1; F-22). Rebuilt from ALREADY-AUTHORIZED evidence only (the accepted
v2 dataset's typed tables and permitted inventory, the FSM-audit day
funnel's day set); ranked LEXICOGRAPHICALLY in the exact §5.1 order with no
hidden score; it never hashes or registers a permanent allowlist and never
selects for the owner (``owner_selection = "NOT PERFORMED"``).

Per logical day the coverage row records: both physical partitions present
and hash-addressable, the funnel counts by ``envelope_trading_day``, the
exact verifier targets (resolved executed trades whose setup id resolves in
the setup-lifecycle table — the setup-verifier deep-link target), the
audit-day coverage (the funnel's ``source_date`` set, which also proves
bars were processed — the measurable control-flow proxy), MBP-1 scope
evidence (``not_evaluated`` unless evidence exists) and the distance from
the protected boundary (days before 2026-06-11).

Hard constraints (§5.1): 1–5 consecutive logical trading days; every
source partition present and hash-addressable; a profile-compatible seed
producible through the prior STORE day (the canonical store-day chain from
2026-01-01 through the day before the window is complete in the
inventory); at least one exact verifier target; June 11 and the sealed
range excluded. "One common window for the whole program" is an owner
commitment enforced at registration time (V3 P0-7), not a scoring input.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd
from pydantic import Field, model_validator

from ..context_experiment_contracts import canonical_contract_sha256
from ..contracts import RecordTable
from .identities import SHA256_PATTERN, FrozenContract, ImmutableMap
from .trading_calendar import (
    CANONICAL_CHAIN_START_DAY,
    CME_GLOBEX_18ET_WEEKDAY_V1,
    PERMITTED_WINDOW_LAST_DAY,
    PROTECTED_BUFFER_DAY,
    TRADING_CALENDAR_POLICY_ID,
    TradingCalendarPolicy,
    VerificationTradingDayRef,
    consecutive_logical_windows,
    is_logical_trading_day,
    logical_trading_days,
    store_day_chain,
    trading_day_ref_from_inventory,
)

__all__ = [
    "SHORTLIST_POLICY_ID",
    "LOGICAL_WINDOW_RANKING_ORDER",
    "LIFECYCLE_PATH_CLASSES",
    "HARD_CONSTRAINT_IDS",
    "JUNE_PROPOSAL_WINDOW",
    "R1_STORE_DAY_CANDIDATE",
    "LogicalDayCoverage",
    "LogicalWindowScore",
    "ShortlistEntry",
    "R1CandidateAssessment",
    "VerificationWindowShortlist",
    "build_logical_day_coverage",
    "rank_logical_windows",
    "build_verification_window_shortlist",
    "render_shortlist_markdown",
]

SHORTLIST_POLICY_ID = "logical_window_shortlist_v1"
#: The §5.1 lexicographic ranking — in this order, nothing else.
LOGICAL_WINDOW_RANKING_ORDER: tuple[str, ...] = (
    "required_source_and_replay_coverage",
    "lifecycle_path_classes_represented",
    "audit_day_coverage",
    "executed_trades",
    "decisions",
    "candidates",
    "exact_verifier_targets",
    "mbp1_scope_evidence",
    "panel_control_flow_coverage",
    "distance_from_protected_boundary",
)
LIFECYCLE_PATH_CLASSES: tuple[str, ...] = (
    "setup_activation",
    "entry_candidate",
    "eligible_decision",
    "execution_resolution",
)
HARD_CONSTRAINT_IDS: tuple[str, ...] = (
    "consecutive_logical_days_1_to_5",
    "all_source_partitions_present_and_hash_addressable",
    "seed_chain_producible_through_prior_store_day",
    "at_least_one_exact_verifier_target",
    "june_11_and_sealed_excluded",
)
JUNE_PROPOSAL_WINDOW: tuple[str, ...] = (
    "2026-06-04",
    "2026-06-05",
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
)
#: The R1 "store-day" candidate as listed in ``../R1/WINDOW_COVERAGE_SCAN.json``.
R1_STORE_DAY_CANDIDATE: tuple[str, ...] = (
    "2026-02-06",
    "2026-02-08",
    "2026-02-09",
    "2026-02-10",
    "2026-02-11",
)


class LogicalDayCoverage(FrozenContract):
    logical_trading_day: str
    source_partitions_present: bool
    source_partition_refs: tuple[VerificationTradingDayRef, ...] = ()
    setup_lifecycle_rows: int = Field(ge=0)
    entry_candidate_rows: int = Field(ge=0)
    eligible_decision_rows: int = Field(ge=0)
    executed_trade_rows: int = Field(ge=0)
    candidate_label_rows: int = Field(ge=0)
    exact_verifier_targets: int = Field(ge=0)
    audit_day_covered: bool
    control_flow_covered: bool
    mbp1_scope_evidence: Literal["not_evaluated"] = "not_evaluated"
    panel_coverage: Literal["not_evaluated"] = "not_evaluated"
    days_before_protected_boundary: int

    @model_validator(mode="after")
    def _coherent(self):
        if not is_logical_trading_day(self.logical_trading_day):
            raise ValueError(f"{self.logical_trading_day} is not a logical trading day")
        if self.source_partitions_present != bool(self.source_partition_refs):
            raise ValueError("source_partitions_present must reflect the partition refs")
        if self.exact_verifier_targets > self.executed_trade_rows:
            raise ValueError("exact verifier targets cannot exceed the executed trades")
        return self

    @property
    def source_partition_refs_flat(self) -> tuple:
        return tuple(
            ref
            for day_ref in self.source_partition_refs
            for ref in day_ref.ordered_source_partition_refs
        )


class LogicalWindowScore(FrozenContract):
    days: tuple[str, ...]
    trading_day_refs: tuple[VerificationTradingDayRef, ...]
    coverage: tuple[LogicalDayCoverage, ...]
    hard_constraints: ImmutableMap[str, bool]
    ineligibility_reasons: tuple[str, ...]
    eligible: bool
    rank_components: ImmutableMap[str, int]
    rank_key: tuple[int, ...]
    seed_chain_replay_days: tuple[str, ...]
    seed_chain_replay_day_count: int = Field(ge=0)
    seed_chain_missing_partitions: tuple[str, ...]
    schema_era_boundary_inside_window: bool

    @model_validator(mode="after")
    def _coherent(self):
        if tuple(row.logical_trading_day for row in self.coverage) != self.days:
            raise ValueError("coverage rows must cover exactly the window days")
        if sorted(self.hard_constraints) != sorted(HARD_CONSTRAINT_IDS):
            raise ValueError("hard_constraints must cover exactly the registered constraints")
        if self.eligible != all(self.hard_constraints.values()):
            raise ValueError("eligible must equal the conjunction of the hard constraints")
        if set(self.rank_components) != set(LOGICAL_WINDOW_RANKING_ORDER):
            raise ValueError("rank_components must cover exactly the registered ranking order")
        if self.rank_key != tuple(
            self.rank_components[name] for name in LOGICAL_WINDOW_RANKING_ORDER
        ):
            raise ValueError(
                "rank_key must be the projection of rank_components in the registered order"
            )
        if self.seed_chain_replay_day_count != len(self.seed_chain_replay_days):
            raise ValueError("seed_chain_replay_day_count must count the chain days")
        return self


class ShortlistEntry(FrozenContract):
    label: str
    rank: int = Field(ge=1)
    window: LogicalWindowScore


class R1CandidateAssessment(FrozenContract):
    store_days: tuple[str, ...]
    status: Literal["provisional_ineligible_as_stated"]
    non_trading_day_ids: tuple[str, ...]
    corrected_logical_window: tuple[str, ...]
    note: str


class VerificationWindowShortlist(FrozenContract):
    shortlist_policy_id: Literal["logical_window_shortlist_v1"] = SHORTLIST_POLICY_ID
    calendar_policy_id: Literal["cme_globex_18et_weekday_v1"] = TRADING_CALENDAR_POLICY_ID
    ranking_order: tuple[str, ...] = LOGICAL_WINDOW_RANKING_ORDER
    hard_constraint_ids: tuple[str, ...] = HARD_CONSTRAINT_IDS
    evidence_source_dataset_id: str = Field(pattern=SHA256_PATTERN)
    evidence_source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    audit_artifact_id: str = Field(pattern=SHA256_PATTERN)
    window_length: int = Field(ge=1, le=5)
    candidate_window_count: int = Field(ge=0)
    eligible_window_count: int = Field(ge=0)
    ranked_windows: tuple[LogicalWindowScore, ...]
    entries: tuple[ShortlistEntry, ...]
    r1_store_day_candidate: R1CandidateAssessment
    owner_selection: Literal["NOT PERFORMED"] = "NOT PERFORMED"
    register_program_allowlist_called: Literal[False] = False
    no_raw_source_reads: Literal[True] = True

    @property
    def shortlist_id(self) -> str:
        return canonical_contract_sha256(self)


def _counts_by_day(frame: pd.DataFrame | None) -> dict[str, int]:
    if frame is None or frame.empty or "envelope_trading_day" not in frame:
        return {}
    return {
        str(day): int(count)
        for day, count in frame.groupby(frame["envelope_trading_day"].astype(str)).size().items()
    }


def _exact_verifier_targets_by_day(
    trades: pd.DataFrame | None, lifecycle: pd.DataFrame | None
) -> dict[str, int]:
    """Resolved executed trades whose setup id resolves in the setup-lifecycle
    table (the exact setup-verifier deep-link target)."""

    if trades is None or trades.empty or "envelope_trading_day" not in trades:
        return {}
    setup_column = "setup_id" if "setup_id" in trades else "envelope_setup_id"
    if setup_column not in trades:
        return {}
    known: set[str] = set()
    if lifecycle is not None and not lifecycle.empty:
        for column in ("envelope_setup_id", "setup_id"):
            if column in lifecycle:
                known |= set(lifecycle[column].dropna().astype(str))
    frame = trades
    if "status" in frame:
        frame = frame.loc[frame["status"].astype(str) == "resolved"]
    resolvable = frame.loc[frame[setup_column].astype(str).isin(known)]
    return _counts_by_day(resolvable)


def build_logical_day_coverage(
    *,
    logical_days: Iterable[str],
    inventory: Mapping[str, tuple[str, str]],
    tables: Mapping[RecordTable, pd.DataFrame],
    funnel_days: Iterable[str],
    policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1,
) -> tuple[LogicalDayCoverage, ...]:
    """One coverage row per LOGICAL trading day from already-authorized
    evidence (typed tables keyed by ``envelope_trading_day``; the accepted
    inventory; the audit funnel's day set). No raw source is read."""

    lifecycle = _counts_by_day(tables.get(RecordTable.SETUP_LIFECYCLE))
    candidates = _counts_by_day(tables.get(RecordTable.ENTRY_CANDIDATE))
    decisions = _counts_by_day(tables.get(RecordTable.ELIGIBLE_DECISION))
    trades = _counts_by_day(tables.get(RecordTable.EXECUTED_TRADE))
    labels = _counts_by_day(tables.get(RecordTable.CANDIDATE_LABEL))
    targets = _exact_verifier_targets_by_day(
        tables.get(RecordTable.EXECUTED_TRADE), tables.get(RecordTable.SETUP_LIFECYCLE)
    )
    audit = {str(day) for day in funnel_days}
    boundary = date.fromisoformat(PROTECTED_BUFFER_DAY)
    rows: list[LogicalDayCoverage] = []
    for day in logical_days:
        ref = trading_day_ref_from_inventory(day, inventory, policy=policy)
        rows.append(
            LogicalDayCoverage(
                logical_trading_day=day,
                source_partitions_present=ref is not None,
                source_partition_refs=() if ref is None else (ref,),
                setup_lifecycle_rows=lifecycle.get(day, 0),
                entry_candidate_rows=candidates.get(day, 0),
                eligible_decision_rows=decisions.get(day, 0),
                executed_trade_rows=trades.get(day, 0),
                candidate_label_rows=labels.get(day, 0),
                exact_verifier_targets=min(targets.get(day, 0), trades.get(day, 0)),
                audit_day_covered=day in audit,
                control_flow_covered=day in audit,
                days_before_protected_boundary=(boundary - date.fromisoformat(day)).days,
            )
        )
    return tuple(rows)


def _schema_era_boundary(inventory: Mapping[str, tuple[str, str]]) -> str | None:
    """The first physical date whose source kind differs from its predecessor
    (the mbp10 → mbp1 era boundary of the accepted inventory), if any."""

    previous_kind: str | None = None
    for day in sorted(inventory):
        kind = inventory[day][0]
        if previous_kind is not None and kind != previous_kind:
            return day
        previous_kind = kind
    return None


def _score_window(
    days: tuple[str, ...],
    coverage_by_day: Mapping[str, LogicalDayCoverage],
    *,
    inventory: Mapping[str, tuple[str, str]],
    era_boundary: str | None,
) -> LogicalWindowScore:
    rows = tuple(coverage_by_day[day] for day in days)
    refs = tuple(ref for row in rows for ref in row.source_partition_refs)
    source_days = sum(1 for row in rows if row.source_partitions_present)
    lifecycle = sum(row.setup_lifecycle_rows for row in rows)
    candidates = sum(row.entry_candidate_rows for row in rows)
    decisions = sum(row.eligible_decision_rows for row in rows)
    trades = sum(row.executed_trade_rows for row in rows)
    targets = sum(row.exact_verifier_targets for row in rows)
    audit_days = sum(1 for row in rows if row.audit_day_covered)
    control_days = sum(1 for row in rows if row.control_flow_covered)
    classes = sum(
        1 for present in (lifecycle > 0, candidates > 0, decisions > 0, trades > 0) if present
    )
    prior_store_day = (date.fromisoformat(days[0]) - timedelta(days=1)).isoformat()
    chain = store_day_chain(CANONICAL_CHAIN_START_DAY, prior_store_day)
    missing_chain = tuple(day for day in chain if day not in inventory)
    constraints = {
        "consecutive_logical_days_1_to_5": 1 <= len(days) <= 5,
        "all_source_partitions_present_and_hash_addressable": source_days == len(days),
        "seed_chain_producible_through_prior_store_day": (bool(chain) and not missing_chain),
        "at_least_one_exact_verifier_target": targets >= 1,
        "june_11_and_sealed_excluded": days[-1] <= PERMITTED_WINDOW_LAST_DAY,
    }
    reasons = tuple(name for name, passed in constraints.items() if not passed)
    components = {
        "required_source_and_replay_coverage": source_days,
        "lifecycle_path_classes_represented": classes,
        "audit_day_coverage": audit_days,
        "executed_trades": trades,
        "decisions": decisions,
        "candidates": candidates,
        "exact_verifier_targets": targets,
        "mbp1_scope_evidence": 0,
        "panel_control_flow_coverage": control_days,
        "distance_from_protected_boundary": rows[-1].days_before_protected_boundary,
    }
    ordered_components = {name: int(components[name]) for name in LOGICAL_WINDOW_RANKING_ORDER}
    inside_era = era_boundary is not None and any(
        ref.physical_utc_date == era_boundary
        for day_ref in refs
        for ref in day_ref.ordered_source_partition_refs
    )
    return LogicalWindowScore(
        days=days,
        trading_day_refs=refs,
        coverage=rows,
        hard_constraints=constraints,
        ineligibility_reasons=reasons,
        eligible=all(constraints.values()),
        rank_components=ordered_components,
        rank_key=tuple(ordered_components.values()),
        seed_chain_replay_days=chain,
        seed_chain_replay_day_count=len(chain),
        seed_chain_missing_partitions=missing_chain,
        schema_era_boundary_inside_window=inside_era,
    )


def rank_logical_windows(
    coverage: Iterable[LogicalDayCoverage],
    *,
    window_length: int = 5,
    inventory: Mapping[str, tuple[str, str]],
    policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1,
) -> tuple[LogicalWindowScore, ...]:
    """Every ``window_length``-day consecutive logical window, scored and
    ordered by the descending lexicographic rank key (ties: earlier start)."""

    rows = tuple(coverage)
    by_day = {row.logical_trading_day: row for row in rows}
    days = tuple(sorted(by_day))
    era_boundary = _schema_era_boundary(inventory)
    scored = [
        _score_window(window, by_day, inventory=inventory, era_boundary=era_boundary)
        for window in consecutive_logical_windows(days, window_length, policy=policy)
    ]
    scored.sort(key=lambda score: (tuple(-value for value in score.rank_key), score.days[0]))
    return tuple(scored)


def _find(ranked: Iterable[LogicalWindowScore], days: tuple[str, ...]) -> LogicalWindowScore | None:
    for score in ranked:
        if score.days == days:
            return score
    return None


def build_verification_window_shortlist(
    coverage: Iterable[LogicalDayCoverage],
    *,
    inventory: Mapping[str, tuple[str, str]],
    evidence_source_dataset_id: str,
    evidence_source_manifest_sha256: str,
    audit_artifact_id: str,
    window_length: int = 5,
    policy: TradingCalendarPolicy = CME_GLOBEX_18ET_WEEKDAY_V1,
) -> VerificationWindowShortlist:
    """The ranked shortlist with the three REQUIRED entries of §5.1 (the June
    proposal; the highest lifecycle/candidate/decision/trade-coverage window
    — the §5.1 order over executed trades, decisions, candidates, lifecycle
    rows; the highest complete-audit-day-coverage window) plus the top-ranked
    eligible window. Nothing here selects for the owner."""

    rows = tuple(coverage)
    ranked = rank_logical_windows(
        rows, window_length=window_length, inventory=inventory, policy=policy
    )
    rank_of = {score.days: index + 1 for index, score in enumerate(ranked)}
    eligible = [score for score in ranked if score.eligible]
    pool = eligible or list(ranked)
    entries: list[ShortlistEntry] = []
    june = _find(ranked, JUNE_PROPOSAL_WINDOW)
    if june is None:
        by_day = {row.logical_trading_day: row for row in rows}
        if all(day in by_day for day in JUNE_PROPOSAL_WINDOW):
            june = _score_window(
                JUNE_PROPOSAL_WINDOW,
                by_day,
                inventory=inventory,
                era_boundary=_schema_era_boundary(inventory),
            )
    if june is not None:
        entries.append(
            ShortlistEntry(
                label="june_proposal", rank=rank_of.get(june.days, len(ranked) + 1), window=june
            )
        )
    if pool:
        richest = max(
            pool,
            key=lambda score: (
                score.rank_components["executed_trades"],
                score.rank_components["decisions"],
                score.rank_components["candidates"],
                sum(row.setup_lifecycle_rows for row in score.coverage),
                -rank_of[score.days],
            ),
        )
        entries.append(
            ShortlistEntry(
                label="highest_lifecycle_candidate_decision_trade_coverage",
                rank=rank_of[richest.days],
                window=richest,
            )
        )
        audited = max(
            pool,
            key=lambda score: (score.rank_components["audit_day_coverage"], -rank_of[score.days]),
        )
        entries.append(
            ShortlistEntry(
                label="highest_complete_audit_day_coverage",
                rank=rank_of[audited.days],
                window=audited,
            )
        )
        entries.append(
            ShortlistEntry(label="top_ranked", rank=rank_of[pool[0].days], window=pool[0])
        )
    non_trading = tuple(
        day for day in R1_STORE_DAY_CANDIDATE if not is_logical_trading_day(day, policy=policy)
    )
    corrected = logical_trading_days(
        R1_STORE_DAY_CANDIDATE[0],
        (date.fromisoformat(R1_STORE_DAY_CANDIDATE[-1]) + timedelta(days=7)).isoformat(),
        policy=policy,
    )[:window_length]
    assessment = R1CandidateAssessment(
        store_days=R1_STORE_DAY_CANDIDATE,
        status="provisional_ineligible_as_stated",
        non_trading_day_ids=non_trading,
        corrected_logical_window=corrected,
        note=(
            "the R1 scan scored consecutive STORE days (physical partition dates); "
            f"{', '.join(non_trading)} is a Sunday partition holding the Sunday 18:00 ET open "
            "of the following Monday's trading day and is not a trading-day id; the corrected "
            "logical form starts at the same first day and takes the next consecutive logical "
            "trading days"
        ),
    )
    return VerificationWindowShortlist(
        evidence_source_dataset_id=evidence_source_dataset_id,
        evidence_source_manifest_sha256=evidence_source_manifest_sha256,
        audit_artifact_id=audit_artifact_id,
        window_length=window_length,
        candidate_window_count=len(ranked),
        eligible_window_count=len(eligible),
        ranked_windows=ranked,
        entries=tuple(entries),
        r1_store_day_candidate=assessment,
    )


def _refs_table(score: LogicalWindowScore) -> list[str]:
    lines = [
        "| Logical day | Session (UTC) | Partition | Key | Kind | Content sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for day_ref in score.trading_day_refs:
        for ref in day_ref.ordered_source_partition_refs:
            lines.append(
                f"| {day_ref.logical_trading_day} | {day_ref.session_open_ts_utc} → "
                f"{day_ref.session_close_ts_utc} | {ref.physical_utc_date} | "
                f"{ref.relative_logical_partition_key} | {ref.source_kind} | "
                f"`{ref.content_sha256}` |"
            )
    return lines


def render_shortlist_markdown(shortlist: VerificationWindowShortlist, *, top: int = 12) -> str:
    lines = [
        "# Verification window shortlist — logical trading days (§5.1; F-22)",
        "",
        f"**Owner selection: {shortlist.owner_selection}.** This document ranks candidate",
        "windows from already-authorized evidence only; it registers nothing:",
        f"`register_program_allowlist` called = `{shortlist.register_program_allowlist_called}`;",
        f"raw source reads = `{not shortlist.no_raw_source_reads}`. No permanent allowlist",
        "was hashed or selected; the owner selects one corrected logical-day window.",
        "",
        f"- shortlist id (content hash): `{shortlist.shortlist_id}`",
        f"- calendar policy: `{shortlist.calendar_policy_id}`; window length: "
        f"{shortlist.window_length}",
        f"- evidence: accepted v2 dataset `{shortlist.evidence_source_dataset_id}` "
        f"(manifest `{shortlist.evidence_source_manifest_sha256}`), FSM-audit artifact "
        f"`{shortlist.audit_artifact_id}`",
        f"- candidate windows: {shortlist.candidate_window_count}; eligible: "
        f"{shortlist.eligible_window_count}",
        "",
        "## Ranking order (lexicographic, no hidden score)",
        "",
        "```text",
        "\n→ ".join(shortlist.ranking_order),
        "```",
        "",
        "## Required shortlist entries",
        "",
        "| Label | Rank | Window | Eligible | Rank key | Seed chain (store days) |",
        "|---|---|---|---|---|---|",
    ]
    for entry in shortlist.entries:
        window = entry.window
        lines.append(
            f"| {entry.label} | {entry.rank} | {' '.join(window.days)} | {window.eligible} | "
            f"{list(window.rank_key)} | {window.seed_chain_replay_day_count} "
            f"({window.seed_chain_replay_days[0] if window.seed_chain_replay_days else '—'} … "
            f"{window.seed_chain_replay_days[-1] if window.seed_chain_replay_days else '—'}) |"
        )
    lines += [
        "",
        "## R1 store-day candidate",
        "",
        f"- as listed: {' '.join(shortlist.r1_store_day_candidate.store_days)} — "
        f"**{shortlist.r1_store_day_candidate.status}**",
        f"- non-trading-day ids: {', '.join(shortlist.r1_store_day_candidate.non_trading_day_ids)}",
        "- corrected logical form: "
        f"{' '.join(shortlist.r1_store_day_candidate.corrected_logical_window)}",
        f"- {shortlist.r1_store_day_candidate.note}",
        "",
        f"## Ranking trace (top {top})",
        "",
        "| # | Window | Eligible | "
        + " | ".join(shortlist.ranking_order)
        + " | Failed constraints |",
        "|---|---|---|" + "---|" * len(shortlist.ranking_order) + "---|",
    ]
    for index, window in enumerate(shortlist.ranked_windows[:top], start=1):
        lines.append(
            f"| {index} | {' '.join(window.days)} | {window.eligible} | "
            + " | ".join(str(value) for value in window.rank_key)
            + f" | {', '.join(window.ineligibility_reasons) or '—'} |"
        )
    lines += ["", "## Hard constraints of the shortlisted windows", ""]
    lines.append("| Window | " + " | ".join(shortlist.hard_constraint_ids) + " |")
    lines.append("|---|" + "---|" * len(shortlist.hard_constraint_ids))
    seen: set[tuple[str, ...]] = set()
    for entry in shortlist.entries:
        if entry.window.days in seen:
            continue
        seen.add(entry.window.days)
        lines.append(
            f"| {' '.join(entry.window.days)} | "
            + " | ".join(
                str(entry.window.hard_constraints[name]) for name in shortlist.hard_constraint_ids
            )
            + " |"
        )
    lines += ["", "## Exact trading-day mapping of the shortlisted windows", ""]
    seen.clear()
    for entry in shortlist.entries:
        if entry.window.days in seen:
            continue
        seen.add(entry.window.days)
        lines.append(f"### {entry.label}: {' '.join(entry.window.days)}")
        lines.append("")
        lines += _refs_table(entry.window)
        lines.append("")
    lines.append(
        "Every window above is at most five CONSECUTIVE logical trading days; June 11 "
        "and the sealed range are excluded; the seed chain is the canonical STORE-DAY "
        "chain from 2026-01-01 through the day before the window (a separately "
        "authorized preparation action, never part of the ≤5-day verification "
        "evidence footprint)."
    )
    lines.append("")
    return "\n".join(lines)


def shortlist_document(shortlist: VerificationWindowShortlist, **extra: Any) -> dict[str, Any]:
    """The JSON document persisted as evidence (the content id first)."""

    return {
        "shortlist_id": shortlist.shortlist_id,
        **extra,
        "shortlist": shortlist.model_dump(mode="json"),
    }
