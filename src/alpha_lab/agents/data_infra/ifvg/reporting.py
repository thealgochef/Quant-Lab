"""Separated IFVG v2 candidate, decision, and execution reports."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping

import pandas as pd

from .context_contracts import (
    ContextRecordTable,
    context_count_reconciliation,
    validate_context_foreign_keys,
)
from .contracts import (
    IFVG_REPORT_SCHEMA_VERSION,
    RecordTable,
    count_reconciliation,
    validate_foreign_keys,
)
from .trade_stats import compute_trade_stats

__all__ = [
    "build_candidate_report",
    "build_decision_report",
    "build_executed_trade_report",
    "build_invariant_audit",
    "build_context_validity_report",
    "build_context_coverage_report",
    "build_context_reconciliation_report",
    "build_context_capacity_report",
    "build_context_identity_report",
    "build_context_performance_report",
]


def _parse_list(value) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(str(item) for item in value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return (value,) if value else ()
        if isinstance(decoded, list):
            return tuple(str(item) for item in decoded)
    return (str(value),)


def _counts_by(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("none").value_counts().sort_index().items()
    }


def _candidate_evaluation_view(
    candidates: pd.DataFrame,
    max_candidates_per_day: int | None,
) -> pd.DataFrame:
    if candidates.empty or max_candidates_per_day is None:
        return candidates
    order = [
        column
        for column in (
            "trading_day",
            "envelope_ts_utc",
            "trigger_cursor",
            "candidate_id",
        )
        if column in candidates
    ]
    ordered = candidates.sort_values(order, kind="mergesort")
    return ordered.groupby("trading_day", sort=False).head(max_candidates_per_day)


def build_candidate_report(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    evaluation_config_hash: str,
    max_candidates_per_day: int | None,
) -> dict:
    """Counts, ordered blocks, and counterfactual label distributions only."""
    view = _candidate_evaluation_view(candidates, max_candidates_per_day)
    block_counts: Counter[str] = Counter()
    blocked = 0
    if "block_reasons" in view:
        for value in view["block_reasons"]:
            reasons = _parse_list(value)
            blocked += int(bool(reasons))
            block_counts.update(reasons)

    label_view = labels
    if not labels.empty and max_candidates_per_day is not None:
        label_view = labels[
            labels["candidate_id"].astype(str).isin(
                set(view["candidate_id"].astype(str))
            )
        ]
    label_distributions: list[dict] = []
    if not label_view.empty:
        group_columns = [
            column
            for column in ("entry_family", "label_family", "r_multiple")
            if column in label_view
        ]
        for keys, group in label_view.groupby(group_columns, dropna=False, sort=True):
            values = keys if isinstance(keys, tuple) else (keys,)
            row = {
                column: value
                for column, value in zip(group_columns, values, strict=True)
            }
            row.update(
                {
                    "n": int(len(group)),
                    "labels": _counts_by(group, "label"),
                    "censored": int(group["censored"].fillna(False).sum()),
                }
            )
            label_distributions.append(row)

    return {
        "report_schema_version": IFVG_REPORT_SCHEMA_VERSION,
        "report_type": "candidate",
        "evaluation_config_hash": evaluation_config_hash,
        "candidate_row_limit": max_candidates_per_day,
        "raw_candidates": int(len(candidates)),
        "reported_candidates": int(len(view)),
        "blocked_candidates": blocked,
        "unblocked_counterfactual_candidates": int(len(view) - blocked),
        "by_entry_family": _counts_by(view, "entry_family"),
        "by_entry_session": _counts_by(view, "entry_session"),
        "ordered_block_reason_counts": dict(sorted(block_counts.items())),
        "counterfactual_label_distributions": label_distributions,
        "performance_metrics_present": False,
    }


def build_decision_report(
    candidates: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    evaluation_config_hash: str,
) -> dict:
    unblocked_ids: set[str] = set()
    if not candidates.empty:
        unblocked_ids = {
            str(row["candidate_id"])
            for row in candidates.to_dict("records")
            if not _parse_list(row.get("block_reasons"))
        }
    decision_candidate_ids = (
        set(decisions["candidate_id"].astype(str))
        if not decisions.empty
        else set()
    )
    guard_counts: Counter[str] = Counter()
    if "passed_guards" in decisions:
        for value in decisions["passed_guards"]:
            guard_counts.update(_parse_list(value))
    return {
        "report_schema_version": IFVG_REPORT_SCHEMA_VERSION,
        "report_type": "eligible_decision",
        "evaluation_config_hash": evaluation_config_hash,
        "decisions": int(len(decisions)),
        "unblocked_candidates": len(unblocked_ids),
        "decision_candidate_fk_matches": len(
            decision_candidate_ids & unblocked_ids
        ),
        "unblocked_candidates_without_decision": sorted(
            unblocked_ids - decision_candidate_ids
        ),
        "decisions_from_blocked_candidates": sorted(
            decision_candidate_ids - unblocked_ids
        ),
        "by_entry_family": _counts_by(decisions, "entry_family"),
        "by_entry_session": _counts_by(decisions, "entry_session"),
        "passed_guard_counts": dict(sorted(guard_counts.items())),
        "performance_metrics_present": False,
    }


def build_executed_trade_report(
    trades: pd.DataFrame,
    *,
    cost_points: float,
    evaluation_config_hash: str,
    tick_size: float,
) -> dict:
    """Only this report exposes trade count, P&L, equity, or performance."""
    resolved = (
        trades.loc[trades["status"] == "resolved"].copy()
        if not trades.empty
        else trades.copy()
    )
    open_unresolved = (
        trades.loc[trades["status"] == "open_unresolved"].copy()
        if not trades.empty
        else trades.copy()
    )
    performance = compute_trade_stats(
        resolved,
        cost_points=cost_points,
        evaluation_config_hash=evaluation_config_hash,
        tick_size=tick_size,
    )
    return {
        "report_schema_version": IFVG_REPORT_SCHEMA_VERSION,
        "report_type": "executed_trade",
        "evaluation_config_hash": evaluation_config_hash,
        "trade_count": int(len(trades)),
        "resolved_trade_count": int(len(resolved)),
        "open_unresolved_trade_count": int(len(open_unresolved)),
        "open_unresolved_trade_ids": (
            sorted(open_unresolved["trade_id"].astype(str))
            if not open_unresolved.empty
            else []
        ),
        "performance": performance,
    }


def _id_set(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame:
        return set()
    return set(frame[column].dropna().astype(str))


def _overlap_count(trades: pd.DataFrame) -> int:
    if len(trades) < 2:
        return 0
    ordered = trades.copy()
    ordered["_entry"] = pd.to_datetime(
        ordered["entry_ts_utc"], utc=True, errors="coerce"
    )
    ordered["_resolution"] = pd.to_datetime(
        ordered["resolution_ts_utc"], utc=True, errors="coerce"
    )
    ordered = ordered.sort_values("_entry", kind="mergesort")
    overlaps = 0
    previous_resolution = None
    previous_open = False
    for row in ordered.to_dict("records"):
        entry = row["_entry"]
        if previous_open or (
            previous_resolution is not None
            and pd.notna(previous_resolution)
            and entry <= previous_resolution
        ):
            overlaps += 1
        previous_resolution = row["_resolution"]
        previous_open = pd.isna(previous_resolution)
    return overlaps


def _simultaneous_phase_counts(lifecycle: pd.DataFrame) -> tuple[int, int]:
    if lifecycle.empty:
        return 0, 0
    order = [
        column
        for column in ("trace_ordinal", "envelope_ts_utc", "lifecycle_event_id")
        if column in lifecycle
    ]
    active_setups: set[str] = set()
    active_trades: set[str] = set()
    setup_violations = 0
    trade_violations = 0
    for row in lifecycle.sort_values(order, kind="mergesort").to_dict("records"):
        setup_id = str(row.get("setup_id") or row.get("envelope_setup_id"))
        to_phase = row.get("to_phase")
        if to_phase == "S0":
            active_setups.discard(setup_id)
            active_trades.discard(setup_id)
        else:
            active_setups.add(setup_id)
            if to_phase == "S5":
                active_trades.add(setup_id)
        setup_violations += int(len(active_setups) > 1)
        trade_violations += int(len(active_trades) > 1)
    return setup_violations, trade_violations


def build_invariant_audit(
    tables: Mapping[RecordTable, pd.DataFrame],
    *,
    data_access_audit: dict,
    old_artifact_mutations: int = 0,
    prohibited_action_counts: Mapping[str, int] | None = None,
) -> dict:
    """Compute the acceptance-gate counters; every violation must be zero."""
    validate_foreign_keys(tables)
    lifecycle = tables[RecordTable.SETUP_LIFECYCLE]
    candidates = tables[RecordTable.ENTRY_CANDIDATE]
    decisions = tables[RecordTable.ELIGIBLE_DECISION]
    trades = tables[RecordTable.EXECUTED_TRADE]
    geometry = tables[RecordTable.GEOMETRY_DOSSIER]
    quarantine = tables[RecordTable.QUARANTINE]

    blocked_ids = {
        str(row["candidate_id"])
        for row in candidates.to_dict("records")
        if _parse_list(row.get("block_reasons"))
    }
    already_in_trade_ids = {
        str(row["candidate_id"])
        for row in candidates.to_dict("records")
        if "already_in_trade" in _parse_list(row.get("block_reasons"))
    }
    quarantined_ids = _id_set(quarantine, "candidate_id")
    execution_candidate_ids = _id_set(trades, "candidate_id")
    decision_candidate_ids = _id_set(decisions, "candidate_id")
    geometry_trade_ids = _id_set(geometry, "trade_id")
    invalidated_setup_ids = {
        str(row.get("setup_id") or row.get("envelope_setup_id"))
        for row in lifecycle.to_dict("records")
        if row.get("to_phase") == "S0"
        and str(row.get("reason", "")).startswith("invalidated_")
    }
    execution_setup_ids = _id_set(trades, "setup_id") or _id_set(
        trades, "envelope_setup_id"
    )

    direction_disagreements = 0
    if not decisions.empty:
        candidate_direction = candidates.set_index("candidate_id")["direction"]
        for row in decisions.to_dict("records"):
            if str(candidate_direction.loc[row["candidate_id"]]) != str(row["direction"]):
                direction_disagreements += 1
    if not trades.empty:
        decision_direction = decisions.set_index("decision_id")["direction"]
        for row in trades.to_dict("records"):
            if str(decision_direction.loc[row["decision_id"]]) != str(row["direction"]):
                direction_disagreements += 1

    risk_or_stop_violations = 0
    entry_on_inversion = 0
    entry_bar_resolutions = 0
    for row in trades.to_dict("records"):
        entry = int(row["entry_ticks"])
        stop = int(row["stop_ticks"])
        risk = int(row["risk_ticks"])
        direction = str(row["direction"])
        risk_or_stop_violations += int(risk <= 0 or abs(entry - stop) != risk)
        risk_or_stop_violations += int(
            (direction == "LONG" and stop >= entry)
            or (direction == "SHORT" and stop <= entry)
        )
        inversion_cursor = row.get("geometry_inversion_bar_cursor")
        entry_cursor = row.get("entry_cursor")
        entry_on_inversion += int(
            inversion_cursor is not None and inversion_cursor == entry_cursor
        )
        bars_after = row.get("bars_after_entry_to_resolution")
        if row.get("status") == "resolved":
            entry_bar_resolutions += int(pd.isna(bars_after) or int(bars_after) < 1)

    setup_violations, trade_violations = _simultaneous_phase_counts(lifecycle)
    allowed_dates = set(data_access_audit.get("allowlist", ()))
    touched_dates = (
        set(data_access_audit.get("path_constructions_by_date", {}))
        | set(data_access_audit.get("metadata_accesses_by_date", {}))
        | set(data_access_audit.get("file_opens_by_date", {}))
        | set(data_access_audit.get("rows_read_by_date", {}))
    )
    # Missing allowlist metadata is fail-closed: every observed touch is then
    # unauthorized rather than silently accepted.
    forbidden_touches = touched_dates - allowed_dates

    open_trades = (
        trades.loc[trades["status"] == "open_unresolved"]
        if not trades.empty
        else trades
    )
    open_with_realized = 0
    if not open_trades.empty:
        for row in open_trades.to_dict("records"):
            open_with_realized += int(
                pd.notna(row.get("realized_ticks"))
                or pd.notna(row.get("realized_r"))
            )
    invalid_statuses = (
        int((~trades["status"].isin(("resolved", "open_unresolved"))).sum())
        if not trades.empty
        else 0
    )

    prohibited = {
        "model_invocations": 0,
        "search_runs": 0,
        "optimization_actions": 0,
        **dict(prohibited_action_counts or {}),
    }
    violations = {
        "duplicate_executed_trade_ids": int(
            trades["trade_id"].duplicated().sum()
        )
        if not trades.empty
        else 0,
        "overlapping_executed_trades": _overlap_count(trades),
        "simultaneous_setup_violations": setup_violations,
        "simultaneous_trade_violations": trade_violations,
        "executions_from_blocked_candidates": len(
            execution_candidate_ids & blocked_ids
        ),
        "executions_from_already_in_trade_candidates": len(
            execution_candidate_ids & already_in_trade_ids
        ),
        "executions_from_invalidated_setups": len(
            execution_setup_ids & invalidated_setup_ids
        ),
        "decisions_from_blocked_candidates": len(
            decision_candidate_ids & blocked_ids
        ),
        "executions_from_quarantined_candidates": len(
            execution_candidate_ids & quarantined_ids
        ),
        "missing_executed_geometry": len(
            _id_set(trades, "trade_id") - geometry_trade_ids
        ),
        "direction_disagreements": direction_disagreements,
        "risk_or_wrong_side_stop_violations": risk_or_stop_violations,
        "entry_on_inversion_candle_trades": entry_on_inversion,
        "entry_bar_resolutions": entry_bar_resolutions,
        "invalid_executed_trade_statuses": invalid_statuses,
        "multiple_open_unresolved_trades": max(0, len(open_trades) - 1),
        "open_unresolved_trades_with_realized_pnl": open_with_realized,
        "forbidden_source_path_or_io_dates": len(forbidden_touches),
        "forbidden_source_rows": sum(
            int(data_access_audit.get("rows_read_by_date", {}).get(day, 0))
            for day in (touched_dates - allowed_dates)
        ),
        "old_artifact_mutations": int(old_artifact_mutations),
        **{key: int(value) for key, value in prohibited.items()},
    }
    return {
        "report_schema_version": IFVG_REPORT_SCHEMA_VERSION,
        "violations": violations,
        "passed": all(value == 0 for value in violations.values()),
        "count_reconciliation": count_reconciliation(tables),
    }


def build_context_validity_report(
    tables: Mapping[ContextRecordTable, pd.DataFrame],
) -> dict:
    provenance = tables.get(
        ContextRecordTable.CONTEXT_VALIDITY_PROVENANCE,
        pd.DataFrame(),
    )
    violations = {
        "missing_as_of_ts": 0,
        "source_close_after_as_of": 0,
        "source_confirmation_after_as_of": 0,
        "invalid_without_missing_reason": 0,
        "unavailable_without_missing_reason": 0,
    }
    if not provenance.empty:
        as_of = pd.to_datetime(provenance["as_of_ts"], utc=True, errors="coerce")
        violations["missing_as_of_ts"] = int(as_of.isna().sum())
        for source, key in (
            ("source_close_ts", "source_close_after_as_of"),
            ("source_confirmed_ts", "source_confirmation_after_as_of"),
        ):
            values = pd.to_datetime(provenance[source], utc=True, errors="coerce")
            violations[key] = int(((values.notna()) & (values > as_of)).sum())
        missing = provenance["missing_reason"].isna() | (
            provenance["missing_reason"].astype(str) == ""
        )
        violations["invalid_without_missing_reason"] = int(
            ((~provenance["valid"].astype(bool)) & missing).sum()
        )
        violations["unavailable_without_missing_reason"] = int(
            ((~provenance["source_available"].astype(bool)) & missing).sum()
        )
    return {
        "schema_version": 3,
        "passed": not any(violations.values()),
        "records": int(len(provenance)),
        "valid_records": int(provenance["valid"].astype(bool).sum())
        if not provenance.empty
        else 0,
        "warmup_complete_records": int(
            provenance["warmup_complete"].astype(bool).sum()
        )
        if not provenance.empty
        else 0,
        "missing_reason_counts": _counts_by(provenance, "missing_reason"),
        "violations": violations,
    }


def build_context_coverage_report(
    tables: Mapping[ContextRecordTable, pd.DataFrame],
) -> dict:
    captures = tables.get(ContextRecordTable.CONTEXT_CAPTURE, pd.DataFrame())
    states = tables.get(ContextRecordTable.CONTEXT_STATE, pd.DataFrame())
    candidates = tables.get(ContextRecordTable.CANDIDATE_CONTEXT_LINK, pd.DataFrame())
    decisions = tables.get(ContextRecordTable.DECISION_CONTEXT_LINK, pd.DataFrame())
    trades = tables.get(ContextRecordTable.TRADE_CONTEXT_LINK, pd.DataFrame())
    return {
        "schema_version": 3,
        "capture_count": int(len(captures)),
        "state_count": int(len(states)),
        "capture_kind_counts": _counts_by(captures, "capture_kind"),
        "warmup_capture_count": int(captures["is_warmup"].astype(bool).sum())
        if not captures.empty and "is_warmup" in captures
        else 0,
        "evidence_capture_count": int((~captures["is_warmup"].astype(bool)).sum())
        if not captures.empty and "is_warmup" in captures
        else int(len(captures)),
        "candidate_link_count": int(len(candidates)),
        "decision_link_count": int(len(decisions)),
        "trade_link_count": int(len(trades)),
    }


def build_context_reconciliation_report(
    tables: Mapping[ContextRecordTable, pd.DataFrame],
    *,
    core_tables: Mapping[RecordTable, pd.DataFrame],
    baseline_reconciliation: dict,
) -> dict:
    violations: list[str] = []
    try:
        validate_context_foreign_keys(tables, core_tables=core_tables)
    except ValueError as exc:
        violations.append(str(exc))
    if baseline_reconciliation.get("passed") is not True:
        violations.append("transient core replay does not match accepted v2")
    return {
        "schema_version": 3,
        "passed": not violations,
        "accepted_v2_core_parity": bool(baseline_reconciliation.get("passed")),
        "exact_id_links": not violations,
        "counts": context_count_reconciliation(tables),
        "violations": violations,
    }


def build_context_capacity_report(
    tables: Mapping[ContextRecordTable, pd.DataFrame],
    *,
    terminal_state_bytes: int,
    terminal_seed_bytes: int,
    max_transition_bytes: int,
) -> dict:
    members = tables.get(ContextRecordTable.EQUAL_LEVEL_POOL_MEMBER, pd.DataFrame())
    max_members = 0
    if not members.empty:
        max_members = int(members.groupby("pool_id").size().max())
    observed = {
        "terminal_state_bytes": int(terminal_state_bytes),
        "terminal_seed_bytes": int(terminal_seed_bytes),
        "max_transition_bytes": int(max_transition_bytes),
        "max_members_per_pool": max_members,
    }
    limits = {
        "terminal_state_bytes": 4_194_304,
        "terminal_seed_bytes": 838_860,
        "max_transition_bytes": 26_214,
        "max_members_per_pool": 16,
    }
    violations = {
        key: {"observed": observed[key], "limit": limit}
        for key, limit in limits.items()
        if observed[key] > limit
    }
    return {
        "schema_version": 4,
        "passed": not violations,
        "observed": observed,
        "limits": limits,
        "violations": violations,
    }


def build_context_identity_report(
    tables: Mapping[ContextRecordTable, pd.DataFrame],
    *,
    expected: Mapping[str, str],
) -> dict:
    observed: dict[str, list[str]] = {}
    for column in (
        "feature_set_version",
        "feature_formula_version",
        "feature_schema_hash",
        "context_config_hash",
    ):
        values: set[str] = set()
        for frame in tables.values():
            if column in frame:
                values.update(frame[column].dropna().astype(str))
        observed[column] = sorted(values)
    violations = {
        column: {"expected": value, "observed": observed.get(column, [])}
        for column, value in expected.items()
        if observed.get(column) not in ([], [value])
    }
    return {
        "schema_version": 3,
        "passed": not violations,
        "expected": dict(expected),
        "observed": observed,
        "violations": violations,
    }


def build_context_performance_report(
    *,
    disabled_replay_seconds: float,
    enabled_replay_seconds: float,
    completed_1m_step_p99_ms: float,
    multi_timeframe_callback_p99_ms: float,
    repeated_run_p95_slowdown_fraction: float | None = None,
    **measurement_metadata,
) -> dict:
    slowdown = (
        (enabled_replay_seconds / disabled_replay_seconds) - 1.0
        if disabled_replay_seconds > 0
        else None
    )
    violations: dict[str, dict] = {}
    if slowdown is None or slowdown > 0.20:
        violations["replay_slowdown_fraction"] = {
            "observed": slowdown,
            "limit": 0.20,
        }
    if (
        repeated_run_p95_slowdown_fraction is not None
        and repeated_run_p95_slowdown_fraction > 0.25
    ):
        violations["repeated_run_p95_slowdown_fraction"] = {
            "observed": repeated_run_p95_slowdown_fraction,
            "limit": 0.25,
        }
    if completed_1m_step_p99_ms > 1.6:
        violations["completed_1m_step_p99_ms"] = {
            "observed": completed_1m_step_p99_ms,
            "limit": 1.6,
        }
    if multi_timeframe_callback_p99_ms > 8.0:
        violations["multi_timeframe_callback_p99_ms"] = {
            "observed": multi_timeframe_callback_p99_ms,
            "limit": 8.0,
        }
    return {
        "schema_version": 4,
        "passed": not violations,
        "disabled_replay_seconds": disabled_replay_seconds,
        "enabled_replay_seconds": enabled_replay_seconds,
        "replay_slowdown_fraction": slowdown,
        "completed_1m_step_p99_ms": completed_1m_step_p99_ms,
        "multi_timeframe_callback_p99_ms": multi_timeframe_callback_p99_ms,
        "repeated_run_p95_slowdown_fraction": (
            repeated_run_p95_slowdown_fraction
        ),
        **measurement_metadata,
        "violations": violations,
    }
