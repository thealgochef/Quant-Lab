"""Task B saved-run tables, with account cash kept at its event timestamp.

This module does not execute replays or funded accounts. Point metrics consume
the validated execution rows; cash consumes one verified account-driven result
per configuration and firm. Subgroup rows never allocate or resimulate cash.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from strategy_core.decisions.sessions import trading_day_for
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

from alpha_lab.propsim.funded.clock import CHICAGO, chicago_date, to_ns
from alpha_lab.propsim.funded.comparison_result import COMPARISON_MODE, COMPARISON_SCHEMA
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

from .contracts import RecordTable
from .experiment import _post_warmup_report_views
from .menthorq_reporting import GROUP_COLUMNS, _grouped, _links, _metrics, record_context
from .search.task_b import task_b_configurations
from .trade_stats import _validate_and_normalize_executed_trades

__all__ = ["StudyConfigurationInput", "assert_context_execution_parity",
           "build_study_reports", "write_study_reports"]

_RECORD_KINDS = (RecordTable.ENTRY_CANDIDATE, RecordTable.ELIGIBLE_DECISION,
                 RecordTable.EXECUTED_TRADE)
_METRIC_COLUMNS = ("trades", "wins", "win_rate", "gross_points", "net_points",
                   "expectancy_points")
_ADDITIVE_METRICS = ("trades", "wins", "gross_points", "net_points")
# Core setup IDs bind the profile hash; downstream linkage IDs bind the setup.
# All execution, timing, session, structural and outcome fields remain compared.
EXECUTION_PARITY_EXCLUSION_PROVENANCE = {
    "envelope_profile_name": "QL canonicalize_section derives the generated name from section hash",
    "profile_name": "QL contract alias of envelope_profile_name",
    "envelope_profile_hash": "Core envelope profile hash binds the section configuration",
    "envelope_section_config_hash": "Core envelope section hash binds the section configuration",
    "evaluation_config_hash": "QL profiles evaluation hash includes section_config_hash",
    "section_config_hash": "QL contract alias of envelope_section_config_hash",
    "profile_hash": "QL contract alias of envelope_profile_hash",
    "setup_id": "Core make_setup_id UUID5 includes profile_hash",
    "envelope_setup_id": "Core envelope foreign key to the profile-bound setup UUID",
    "candidate_id": "Core make_candidate_id UUID5 includes setup_id",
    "decision_id": "Core make_decision_id UUID5 includes candidate_id and execution_profile_hash",
    "trade_id": "Core make_trade_id UUID5 includes decision_id",
    "entering_seed_hash": (
        "QL capture_driver hashes the complete Core IfvgDaySeed; Core state.seed_hash "
        "includes seed/reducer profile_hash and profile-derived setup IDs"
    ),
}
EXECUTION_PARITY_EXCLUSIONS = frozenset(EXECUTION_PARITY_EXCLUSION_PROVENANCE)


@dataclass(frozen=True, slots=True)
class StudyConfigurationInput:
    configuration: str
    tables: Mapping[RecordTable, pd.DataFrame]
    provider: Any
    section: IfvgSmcSection
    evaluation_days: tuple[str, ...]
    funded_result: Mapping[str, Any]
    cost_points: float
    tick_size: float = .25
    run_identity: Mapping[str, Any] = field(default_factory=dict)


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("study reporting requires an aware authoritative timestamp")
    return stamp.tz_convert("UTC")


def _day(value: Any) -> str:
    return date.fromisoformat(str(value)[:10]).isoformat()


def _calendar_month(value: Any) -> str:
    return chicago_date(to_ns(_timestamp(value).isoformat())).strftime("%Y-%m")


def _entry_logical_day(value: Any) -> str:
    day = trading_day_for(_timestamp(value).to_pydatetime())
    if day is None:
        raise ValueError("study entry is inside the existing closed session window")
    return day.isoformat()


def _months(days: Sequence[str]) -> tuple[str, ...]:
    first, last = min(days)[:7], max(days)[:7]
    return tuple(str(period) for period in pd.period_range(first, last, freq="M"))


def assert_context_execution_parity(baseline: pd.DataFrame, context: pd.DataFrame) -> None:
    """Assert exact behavior after removing named profile-derived identities."""
    left_columns = set(baseline) - EXECUTION_PARITY_EXCLUSIONS
    right_columns = set(context) - EXECUTION_PARITY_EXCLUSIONS
    if left_columns != right_columns:
        raise ValueError("context_on executed-trade columns differ from baseline")
    columns = sorted(left_columns)
    order = [name for name in ("entry_ts_utc", "entry_cursor", "resolution_ts_utc")
             if name in columns]

    def project(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.loc[:, columns].copy()
        return (out.sort_values(order, kind="stable") if order else out).reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(project(baseline), project(context), check_dtype=False,
                                      check_exact=True)
    except AssertionError as error:
        raise ValueError("context_on executed trades differ from baseline") from error


def _reconcile(rows: Sequence[dict], total: Mapping[str, Any], label: str) -> None:
    for column in _ADDITIVE_METRICS:
        actual = sum(row[column] for row in rows)
        if not math.isclose(actual, total[column], rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"{label} {column} does not reconcile")


def _signed_cash(event: Mapping[str, Any]) -> int:
    amount = Decimal(str(event["amount_usd"])) * 100
    if not amount.is_finite() or amount != amount.to_integral_value() or amount < 0:
        raise ValueError("funded cash event is not exact nonnegative cents")
    return int(amount) * (1 if event["kind"] == "payout_received" else -1)


def _cash(config: StudyConfigurationInput) -> tuple[dict[str, int], list[dict]]:
    result = config.funded_result
    if result.get("validation", {}).get("passed") is not True:
        raise ValueError("study reporting requires a verified funded result")
    if (result.get("schema_version") != COMPARISON_SCHEMA
            or result.get("mode") != COMPARISON_MODE):
        raise ValueError("study reporting requires the current account-driven pair result")
    selected = [summary for summary in result.get("summaries_cents", {}).values()
                if summary.get("configuration") == config.configuration]
    summaries = {summary["firm_key"]: summary for summary in selected}
    if len(selected) != len(summaries):
        raise ValueError("duplicate funded pairs for configuration and firm")
    if set(summaries) != set(FIRM_PROFILES):
        raise ValueError("study reporting requires every existing funded firm configuration")
    if not summaries or any(summary.get("status") != "Completed" for summary in summaries.values()):
        raise ValueError(f"missing completed funded pairs for {config.configuration}")
    totals = {firm: 0 for firm in summaries}
    components = {firm: {"payouts_received_cents": 0, "account_costs_cents": 0,
                         "payouts_received_count": 0, "accounts_purchased": 0}
                  for firm in summaries}
    events = []
    for event in result.get("tables", {}).get("cash_ledger", []):
        if event.get("configuration") != config.configuration:
            continue
        if event["kind"] not in ("payout_received", "account_purchase"):
            raise ValueError("unsupported cash ledger event kind")
        firm = event["firm_key"]
        if firm not in totals:
            raise ValueError("cash ledger has an unverified firm configuration")
        stamp = _timestamp(event["ts_utc"])
        cents = _signed_cash(event)
        if "amount_cents" in event and (
            not isinstance(event["amount_cents"], int)
            or isinstance(event["amount_cents"], bool)
            or event["amount_cents"] != abs(cents)
        ):
            raise ValueError("cash ledger amount fields disagree")
        totals[firm] += cents
        if event["kind"] == "payout_received":
            components[firm]["payouts_received_cents"] += cents
            components[firm]["payouts_received_count"] += 1
        else:
            components[firm]["account_costs_cents"] -= cents
            components[firm]["accounts_purchased"] += 1
        events.append({"firm_key": firm, "ts_utc": stamp,
                       "cash_event_month": _calendar_month(stamp),
                       "cash_event_day": chicago_date(to_ns(stamp.isoformat())).isoformat(),
                       "net_cash_cents": cents})
    for firm, summary in summaries.items():
        expected = summary["net_cash_earned_cents"]
        if not isinstance(expected, int) or isinstance(expected, bool) or totals[firm] != expected:
            raise ValueError("cash events do not reconcile to verified funded totals")
        for column, actual in components[firm].items():
            if column in summary and (
                not isinstance(summary[column], int) or isinstance(summary[column], bool)
                or summary[column] != actual
            ):
                raise ValueError(f"cash events do not reconcile to verified funded {column}")
    return totals, events


def _point_views(config: StudyConfigurationInput) -> tuple[dict, dict, pd.DataFrame]:
    views, _ = _post_warmup_report_views(dict(config.tables))
    contexts = {kind: record_context(views[kind], provider=config.provider, section=config.section,
                                     tick_size=config.tick_size) for kind in _RECORD_KINDS}
    executions = views[RecordTable.EXECUTED_TRADE]
    resolved = executions.loc[executions.status.eq("resolved")] if len(executions) else executions
    if not math.isfinite(config.cost_points) or config.cost_points < 0:
        raise ValueError("study reporting requires finite nonnegative trade costs")
    if len(resolved):
        normalized = _validate_and_normalize_executed_trades(resolved, tick_size=config.tick_size)
        normalized["_net_points"] = normalized["_realized_pts"] - config.cost_points
        context = contexts[RecordTable.EXECUTED_TRADE].set_index("candidate_id")
        for column in (*GROUP_COLUMNS, "roll_flag", "trading_date"):
            normalized[column] = normalized.candidate_id.map(context[column])
        normalized["entry_calendar_month"] = normalized.entry_ts_utc.map(_calendar_month)
        normalized["_entry_logical_day"] = normalized.entry_ts_utc.map(_entry_logical_day)
    else:
        normalized = pd.DataFrame(columns=[*GROUP_COLUMNS, "roll_flag", "trading_date",
                                          "entry_calendar_month", "_label", "_realized_pts",
                                          "_net_points", "_entry_logical_day"])
    for kind, context in contexts.items():
        if len(context) and not set(context.availability_ts_utc.map(_entry_logical_day)) <= set(
            config.evaluation_days
        ):
            raise ValueError(f"{kind.value} contains entries outside the evaluation calendar")
    return views, contexts, normalized


def _block_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if frame.empty:
        return counts
    if "block_reasons" not in frame:
        raise ValueError("candidate coverage requires persisted block reasons")
    for raw in frame.block_reasons:
        reasons = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(reasons, (list, tuple)) or any(
            not isinstance(reason, str) for reason in reasons
        ):
            raise ValueError("candidate block reasons must be a list of strings")
        for reason in set(reasons):
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def build_study_reports(
    configurations: Sequence[StudyConfigurationInput], *, baseline_name: str = "baseline",
    context_name: str = "context_on",
) -> dict[str, pd.DataFrame]:
    """Reconcile saved configuration outputs, retaining every known roll day."""
    by_name = {item.configuration: item for item in configurations}
    if (len(by_name) != len(configurations) or baseline_name not in by_name
            or context_name not in by_name):
        raise ValueError("study reporting requires unique configurations and baseline/context_on")
    expected_names = tuple(row.name for row in task_b_configurations())
    if set(by_name) != set(expected_names):
        raise ValueError("study reporting requires the thirteen owner-specified configurations")
    configurations = tuple(by_name[name] for name in expected_names)
    evaluation_days = by_name[baseline_name].evaluation_days
    if (not evaluation_days or evaluation_days[0] != "2025-06-16"
            or evaluation_days[-1] != "2026-06-10"
            or any(config.evaluation_days != evaluation_days for config in configurations)):
        raise ValueError("configuration evaluation calendars differ from the Task B scope")
    assert_context_execution_parity(by_name[baseline_name].tables[RecordTable.EXECUTED_TRADE],
                                   by_name[context_name].tables[RecordTable.EXECUTED_TRADE])
    outputs: dict[str, list[dict]] = {name: [] for name in (
        "comparison", "comparison_by_session", "comparison_by_regime", "comparison_by_slot",
        "comparison_by_month", "cells", "coverage", "roll_days",
    )}
    per_configuration: dict[str, pd.DataFrame] = {}
    firm_keys: set[str] | None = None
    for config in configurations:
        days = config.evaluation_days
        if not days or tuple(sorted(set(days))) != days or any(_day(day) != day for day in days):
            raise ValueError("evaluation days must be ordered unique ISO dates")
        if days[0] < "2025-06-16" or days[-1] > "2026-06-10":
            raise ValueError("evaluation days are outside the Task B authorized interval")
        views, contexts, points = _point_views(config)
        total = _metrics(points)
        cash, cash_events = _cash(config)
        if firm_keys is None:
            firm_keys = set(cash)
        elif set(cash) != firm_keys:
            raise ValueError("configurations do not have the same verified funded firm set")
        snapshots = {day: config.provider.snapshot(datetime.combine(
            date.fromisoformat(day), time(6), tzinfo=CHICAGO)) for day in days}
        roll_days = {day for day, snapshot in snapshots.items() if snapshot.roll_flag is True}
        roll_points = (
            points.loc[points._entry_logical_day.isin(roll_days)] if len(points) else points
        )
        comparison = {"configuration": config.configuration, **total,
                      **{f"net_cash_cents.{firm}": cash[firm] for firm in sorted(cash)},
                      **{f"ny.{key}": value for key, value in _metrics(
                          points.loc[points.entry_session.eq("ny")]).items()},
                      "roll_day_trades": len(roll_points),
                      "roll_day_net_points": _metrics(roll_points)["net_points"],
                      "open_unresolved_trades": (
                          len(views[RecordTable.EXECUTED_TRADE]) - len(points))}
        outputs["comparison"].append(comparison)
        for name, columns in (("comparison_by_session", ("entry_session",)),
                              ("comparison_by_regime", ("regime",)),
                              ("comparison_by_slot", ("slot_chicago",)),
                              ("cells", GROUP_COLUMNS)):
            grouped = _grouped(points, columns, metrics=True)
            _reconcile(grouped, total, name)
            if name != "cells" or config.configuration != baseline_name:
                outputs[name].extend(
                    {"configuration": config.configuration, **row} for row in grouped)
        months = _months(days)
        if any(event["cash_event_month"] not in months for event in cash_events):
            raise ValueError("cash event month is outside the evaluation months")
        month_rows = []
        for month in months:
            row = {"configuration": config.configuration, "entry_calendar_month": month,
                   **_metrics(points.loc[points.entry_calendar_month.eq(month)]),
                   **{f"net_cash_by_cash_event_month_cents.{firm}": sum(
                       event["net_cash_cents"] for event in cash_events
                       if event["firm_key"] == firm and event["cash_event_month"] == month)
                      for firm in sorted(cash)}}
            month_rows.append(row)
        _reconcile(month_rows, total, "comparison_by_month")
        for firm, cents in cash.items():
            if sum(row[f"net_cash_by_cash_event_month_cents.{firm}"]
                   for row in month_rows) != cents:
                raise ValueError("cash event months do not reconcile")
        outputs["comparison_by_month"].extend(month_rows)
        roll_rows = []
        for day in sorted(roll_days):
            day_points = (
                points.loc[points._entry_logical_day.eq(day)] if len(points) else points
            )
            day_events = [event for event in cash_events if event["cash_event_day"] == day]
            roll_rows.append({
                "configuration": config.configuration, "trading_day": day,
                "selected_instrument_id": snapshots[day].selected_instrument_id,
                **_metrics(day_points), "cash_events": len(day_events),
                **{f"cash_events.{firm}": sum(event["firm_key"] == firm for event in day_events)
                   for firm in sorted(cash)},
                **{f"net_cash_by_cash_event_day_cents.{firm}": sum(
                    event["net_cash_cents"] for event in day_events if event["firm_key"] == firm)
                   for firm in sorted(cash)},
            })
        _reconcile(roll_rows, _metrics(roll_points), "roll_days")
        flagged_cash_events = [event for event in cash_events
                               if event["cash_event_day"] in roll_days]
        if sum(row["cash_events"] for row in roll_rows) != len(flagged_cash_events):
            raise ValueError("roll-day cash event counts do not reconcile")
        for firm in cash:
            events = [event for event in flagged_cash_events if event["firm_key"] == firm]
            if (sum(row[f"cash_events.{firm}"] for row in roll_rows) != len(events)
                    or sum(row[f"net_cash_by_cash_event_day_cents.{firm}"] for row in roll_rows)
                    != sum(event["net_cash_cents"] for event in events)):
                raise ValueError("roll-day firm cash events do not reconcile")
        outputs["roll_days"].extend(roll_rows)
        candidate_context = contexts[RecordTable.ENTRY_CANDIDATE]
        blocked = {"context_unavailable": 0, "regime_gate": 0, "nearest_support_gex1": 0,
                   **_block_counts(views[RecordTable.ENTRY_CANDIDATE])}
        coverage = {"configuration": config.configuration, "evaluation_days": len(days),
                    "days_with_context_available": sum(snapshot.context_available
                                                        for snapshot in snapshots.values()),
                    "candidates": len(candidate_context),
                    "candidates_with_context": int(candidate_context.context_available.sum()),
                    **{f"candidates_blocked.{reason}": count
                       for reason, count in sorted(blocked.items())}}
        outputs["coverage"].append(coverage)
        for kind, context in contexts.items():
            groups = _grouped(context, GROUP_COLUMNS, metrics=False)
            if sum(row["count"] for row in groups) != len(context):
                raise ValueError("grouped record counts do not reconcile")
            per_configuration[f"{config.configuration}/grouped_{kind.value}"] = pd.DataFrame(
                groups, columns=[*GROUP_COLUMNS, "count"])
        per_configuration[f"{config.configuration}/executed_trade_metrics"] = pd.DataFrame(
            _grouped(points, GROUP_COLUMNS, metrics=True),
            columns=[*GROUP_COLUMNS, *_METRIC_COLUMNS])
        # The review export intentionally keeps the full candidate chain,
        # including stamped warmup, as A1 did; it supplies no ML inputs.
        export = record_context(
            config.tables[RecordTable.ENTRY_CANDIDATE], provider=config.provider,
            section=config.section, tick_size=config.tick_size)
        for index, kind, key in ((1, RecordTable.ELIGIBLE_DECISION, "decision_id"),
                                 (2, RecordTable.EXECUTED_TRADE, "trade_id")):
            export.insert(index, key, export.candidate_id.astype(str).map(
                _links(config.tables[kind], key)))
        per_configuration[f"{config.configuration}/context_export"] = export
    tables = {name: pd.DataFrame(rows) for name, rows in outputs.items()}
    tables["comparison_by_month"] = tables["comparison_by_month"].rename(columns={
        "entry_calendar_month": "calendar_month", "net_points": "net_points_by_entry_month",
    })
    for name in ("comparison_by_session", "comparison_by_regime", "comparison_by_slot", "cells"):
        if tables[name].empty:
            dimensions = {"comparison_by_session": ("entry_session",),
                          "comparison_by_regime": ("regime",),
                          "comparison_by_slot": ("slot_chicago",),
                          "cells": GROUP_COLUMNS}[name]
            tables[name] = pd.DataFrame(columns=["configuration", *dimensions, *_METRIC_COLUMNS])
    tables["coverage"] = tables["coverage"].fillna(0)
    if tables["roll_days"].empty:
        tables["roll_days"] = pd.DataFrame(columns=["configuration", "trading_day",
                                                   "selected_instrument_id", *_METRIC_COLUMNS,
                                                   "cash_events"])
    return {**tables, **per_configuration}


def write_study_reports(
    root: Path, configurations: Sequence[StudyConfigurationInput], **kwargs,
) -> dict[str, Path]:
    """Write only reconciled CSVs; no report files are emitted after a failure."""
    tables = build_study_reports(configurations, **kwargs)
    by_name = {config.configuration: config for config in configurations}
    paths = {}
    for name, table in tables.items():
        path = Path(root) / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = table.copy()
        for column in table:
            if table[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
                table[column] = table[column].map(
                    lambda value: json.dumps(value, default=str)
                    if isinstance(value, (dict, list, tuple))
                    else value)
        if name.endswith("/context_export"):
            config = by_name[name.split("/", 1)[0]]
            metadata = {
                "configuration": config.configuration, "run_identity": dict(config.run_identity),
                "menthorq_context_version": config.section.menthorq_context_version,
                "report_only": True, "archival": False,
                "source_file_sha256": dict(config.provider.source_file_sha256),
                "schema_version": config.provider.schema_version,
                "formula_version": config.provider.formula_version,
            }
            with path.open("w", encoding="utf-8", newline="") as stream:
                stream.write("# " + json.dumps(metadata, sort_keys=True, default=str) + "\n")
                table.to_csv(stream, index=False)
        else:
            table.to_csv(path, index=False)
        paths[name] = path
    validation_path = Path(root) / "validation.json"
    validation_path.write_text(json.dumps({
        "passed": True, "context_on_equals_baseline": True,
        "execution_parity_exclusions": sorted(EXECUTION_PARITY_EXCLUSIONS),
        "execution_parity_exclusion_provenance": EXECUTION_PARITY_EXCLUSION_PROVENANCE,
        "configuration_count": len(configurations),
        "cash_scope": "configuration_and_firm",
        "monthly_points_timestamp": "entry_ts_utc_in_America_Chicago",
        "monthly_cash_timestamp": "cash_event_ts_utc_in_America_Chicago",
        "monthly_index_column": "calendar_month",
        "monthly_trade_metrics_scope": "entry_calendar_month_in_America_Chicago",
        "monthly_net_points_column": "net_points_by_entry_month",
        "monthly_net_cash_columns": "net_cash_by_cash_event_month_cents.<firm>",
        "roll_points_day": "entry_logical_trading_date",
        "evaluation_scope_timestamp": "entry_availability_ts_utc_with_Core_trading_day_for",
        "context_trading_date": "civil_date_in_America_Chicago",
        "roll_cash_day": "cash_event_calendar_date_in_America_Chicago",
        "cash_allocated_to_trades": False, "roll_days_excluded": False,
    }, indent=2) + "\n", encoding="utf-8")
    paths["validation"] = validation_path
    return paths
