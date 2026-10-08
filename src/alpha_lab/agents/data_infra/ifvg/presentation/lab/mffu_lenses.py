"""Transparent views of saved MFFU outcomes; never configuration changes.

All clocks retain the original operation opening. Money is posted cents and
sorting uses full precision. Numeric targets require an observed measurement.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from functools import cmp_to_key
from statistics import median
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import policy_label

VERSION = "ifsm_mffu_reporting_v6"
FIRM = "myfundedfutures"
LENSES = {
    "total_cash": (
        "Total net cash",
        (("net_received_cash_cents", "desc"), ("acquisition_spend_cents", "asc")),
    ),
    "payout_speed": (
        "Payout speed",
        (
            ("first_receipt_elapsed_seconds", "asc"),
            ("net_received_cash_cents", "desc"),
            ("acquisition_spend_cents", "asc"),
        ),
    ),
    "cushion_speed": (
        "Build the cushion",
        (
            ("first_flat_buffer_elapsed_seconds", "asc"),
            ("net_received_cash_cents", "desc"),
            ("acquisition_spend_cents", "asc"),
        ),
    ),
    "account_usage": (
        "Lower account usage",
        (
            ("accounts_lost", "asc"),
            ("acquisition_spend_cents", "asc"),
            ("net_received_cash_cents", "desc"),
        ),
    ),
    "activity": (
        "Regular activity",
        (
            ("entry_date_coverage", "desc"),
            ("longest_no_entry_evaluated_dates", "asc"),
            ("net_received_cash_cents", "desc"),
        ),
    ),
    "large_payouts": (
        "Larger payouts",
        (
            ("largest_received_payout_cents", "desc"),
            ("median_received_payout_cents", "desc"),
            ("net_received_cash_cents", "desc"),
        ),
    ),
}
VISIBLE = {
    "total_cash": (
        "received_payouts_cents",
        "acquisition_spend_cents",
        "accounts_bought",
        "accounts_lost",
        "first_receipt_calendar_days",
        "account_buffer_reached",
        "account_buffer_failed",
        "account_buffer_right_censored",
        "account_buffer_unavailable",
        "conditional_median_account_days_to_buffer",
        "average_entries_per_evaluated_date",
    ),
    "payout_speed": (
        "first_receipt_utc",
        "first_receipt_calendar_days",
        "first_receipt_evaluated_date_days",
        "purchases_before_first_receipt",
        "spend_before_first_receipt_cents",
        "longest_no_receipt_calendar_days",
    ),
    "cushion_speed": (
        "first_flat_buffer_utc",
        "first_flat_buffer_calendar_days",
        "first_flat_buffer_account",
        "accounts_failed_before_first_buffer",
        "first_request_utc",
        "first_eligibility_utc",
        "first_receipt_calendar_days",
    ),
    "account_usage": (
        "accounts_bought",
        "accounts_lost",
        "replacement_count",
        "replacement_spend_cents",
        "first_receipt_calendar_days",
        "payments_received",
    ),
    "activity": (
        "entry_date_coverage",
        "average_entries_per_evaluated_date",
        "zero_entry_dates",
        "one_entry_dates",
        "multiple_entry_dates",
        "longest_no_entry_evaluated_dates",
        "processing_duration_seconds",
        "protection_duration_seconds",
    ),
    "large_payouts": (
        "largest_received_payout_cents",
        "median_received_payout_cents",
        "payments_received",
        "first_receipt_calendar_days",
        "longest_no_receipt_calendar_days",
        "largest_account_receipt_share",
    ),
}
TARGETS = {
    "average_entries_per_evaluated_date": (
        "range",
        "Average entries per evaluated date",
        "trades/date",
    ),
    "entry_date_coverage": ("minimum", "Minimum dates with an entry", "fraction"),
    "first_flat_buffer_calendar_days": ("maximum", "Maximum days to flat +$2,000", "calendar days"),
    "first_receipt_calendar_days": ("maximum", "Maximum days to received payment", "calendar days"),
    "accounts_bought": ("maximum", "Maximum accounts bought", "accounts"),
    "accounts_lost": ("maximum", "Maximum accounts lost", "accounts"),
    "replacement_spend_cents": ("maximum", "Maximum replacement spending", "cents"),
    "net_received_cash_cents": ("minimum", "Minimum net received cash", "cents"),
    "payments_received": ("minimum", "Minimum received payment count", "payments"),
    "longest_no_receipt_calendar_days": (
        "maximum",
        "Maximum longest no-receipt interval",
        "calendar days",
    ),
    "longest_no_entry_evaluated_dates": ("maximum", "Maximum no-entry stretch", "evaluated dates"),
}
AXES = (
    "schedule",
    "exit",
    "sizing",
    "geometry",
    "entry_context",
    "daily_cap",
    "overhead",
    "review_status",
)
LABELS = {
    "configuration_id": "Configuration",
    "net_received_cash_cents": "Net received cash",
    "received_payouts_cents": "Received payouts",
    "acquisition_spend_cents": "Acquisition spend",
    "accounts_bought": "Accounts bought",
    "accounts_lost": "Accounts lost",
    "replacement_count": "Replacements",
    "replacement_spend_cents": "Replacement spend",
    "payments_received": "Payments received",
    "average_entries_per_evaluated_date": "Entries / evaluated date",
    "entry_date_coverage": "Dates with entry",
    "zero_entry_dates": "Zero-entry dates",
    "one_entry_dates": "One-entry dates",
    "multiple_entry_dates": "Multiple-entry dates",
    "first_receipt_calendar_days": "First receipt · calendar days",
    "first_receipt_evaluated_date_days": "First receipt · evaluated dates",
    "first_receipt_utc": "First receipt (UTC)",
    "first_flat_buffer_calendar_days": "Flat +$2,000 · calendar days",
    "first_flat_buffer_utc": "First flat +$2,000 (UTC)",
    "first_flat_buffer_account": "Account reaching cushion",
    "accounts_failed_before_first_buffer": "Failures before cushion",
    "first_request_utc": "First request (UTC)",
    "first_eligibility_utc": "First secured eligibility (UTC)",
    "purchases_before_first_receipt": "Bought before first receipt",
    "spend_before_first_receipt_cents": "Spend before first receipt",
    "longest_no_receipt_calendar_days": "Longest no receipt · calendar days",
    "longest_no_entry_evaluated_dates": "Longest no entry · evaluated dates",
    "processing_duration_seconds": "Processing · seconds",
    "protection_duration_seconds": "Protection · seconds",
    "largest_received_payout_cents": "Largest received payment",
    "median_received_payout_cents": "Median received payment",
    "largest_account_receipt_share": "Largest paying-account share",
    "overall_cash_rank": "Overall cash rank",
    "account_buffer_reached": "Accounts reaching cushion",
    "account_buffer_failed": "Accounts failing before cushion",
    "account_buffer_right_censored": "Accounts ongoing without cushion",
    "account_buffer_unavailable": "Accounts with unavailable cushion evidence",
    "conditional_median_account_days_to_buffer": "Median account days · reaching accounts only",
    "account_receipt_reached": "Accounts receiving payment",
    "account_receipt_failed": "Accounts failing before payment",
    "account_receipt_right_censored": "Accounts ongoing without payment",
    "account_receipt_unavailable": "Accounts with unavailable payment evidence",
}


def cents(row: Mapping[str, Any], name: str) -> int | None:
    if row.get(name + "_cents") is not None:
        return int(row[name + "_cents"])
    if row.get(name + "_usd") is not None:
        amount = Decimal(str(row[name + "_usd"])) * 100
        if not amount.is_finite() or amount != amount.to_integral_value():
            raise ValueError("saved money must be posted whole cents")
        return int(amount)
    return None


def stamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC")


def clock(start: Any, end: Any, calendar: Sequence[str]) -> dict[str, Any]:
    left, right = stamp(start), stamp(end)
    if right < left:
        raise ValueError("milestone precedes operation opening")
    ld, rd = left.tz_convert("America/Chicago").date(), right.tz_convert("America/Chicago").date()
    return {
        "elapsed_seconds": (right.value - left.value) / 1_000_000_000,
        "calendar_days": (rd - ld).days,
        "evaluated_date_days": sum(ld.isoformat() < str(day) <= rd.isoformat() for day in calendar),
    }


def _event_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return stamp(row["ts_utc"]).value, int(row.get("seq") or 0)


def _duration(events: Sequence[Mapping[str, Any]], cutoff: Any, state: str) -> float:
    total = 0
    grouped: dict[Any, list] = {}
    for row in events:
        grouped.setdefault(row.get("account_number"), []).append(row)
    for rows in grouped.values():
        ordered = sorted(rows, key=_event_key)
        for i, row in enumerate(ordered):
            if row.get("status_after") != state:
                continue
            end = ordered[i + 1]["ts_utc"] if i + 1 < len(ordered) else cutoff
            total += max(0, stamp(end).value - stamp(row["ts_utc"]).value)
    return total / 1_000_000_000


def build_rows(study: Any) -> list[dict[str, Any]]:
    """Project all frozen intents from actual trades and ordered saved ledgers."""
    output = []
    cutoff = study.plan.source.cutoff_utc
    for variant in study.plan.configurations:
        key = variant.name
        summary = study.summary(key, FIRM) or {}
        intent = json.loads(variant.intent_json)
        disposition = next(
            (
                r
                for r in study.result.get("mffu_batch", {}).get("dispositions", [])
                if r.get("variant_id") == key
            ),
            {},
        )
        row: dict[str, Any] = {
            "configuration_id": key,
            "economic_result_id": study.result_id,
            "reporting_version": VERSION,
            "firm": FIRM,
            "population": "actual_funded",
            "scope": "full_study",
            **intent,
            "review_status": disposition.get("status", summary.get("status", "unavailable")),
            "measurement_status": {},
            "evidence_status": summary.get("status", "unavailable"),
        }
        statuses = row["measurement_status"]
        if summary.get("status") != "Completed":
            row["unavailable_reason"] = summary.get("reason", "economic result unavailable")
            output.append(row)
            continue
        trades = list(study.trades_by_pair.get(f"{key}|{FIRM}", ()))
        journeys = study.rows("account_journeys", key, FIRM)
        cash = sorted(study.rows("cash_ledger", key, FIRM), key=_event_key)
        events = study.rows("account_events", key, FIRM)
        purchases = [r for r in cash if r.get("kind") == "account_purchase"]
        receipts = [r for r in cash if r.get("kind") == "payout_received"]
        start = min((r["created_utc"] for r in journeys), key=stamp) if journeys else None
        row["operation_start_utc"] = start
        row["cutoff_utc"] = cutoff
        row["acquisition_spend_cents"] = summary.get("account_costs_cents")
        row["received_payouts_cents"] = summary.get("payouts_received_cents")
        row["net_received_cash_cents"] = summary.get("net_cash_earned_cents")
        row["accounts_bought"] = len(purchases) if purchases else summary.get("accounts_purchased")
        failures = [r for r in journeys if r.get("failed_utc")]
        row["accounts_lost"] = len(failures) if journeys else None
        row["replacement_count"] = max(0, len(purchases) - 1) if purchases else None
        row["replacement_spend_cents"] = (
            sum(cents(r, "amount") for r in purchases[1:]) if purchases else None
        )
        row["payments_received"] = len(receipts)
        amounts = [cents(r, "amount") for r in receipts]
        row["largest_received_payout_cents"] = max(amounts) if amounts else None
        row["median_received_payout_cents"] = median(amounts) if amounts else None
        account_receipts: dict[Any, int] = {}
        for receipt in receipts:
            account_receipts[receipt.get("account_number")] = account_receipts.get(
                receipt.get("account_number"), 0
            ) + cents(receipt, "amount")
        row["largest_account_receipt_share"] = (
            max(account_receipts.values()) / sum(amounts) if amounts else None
        )
        for metric in (
            "largest_received_payout_cents",
            "median_received_payout_cents",
            "largest_account_receipt_share",
        ):
            statuses[metric] = "reached" if receipts else "not_reached"
        counts = Counter(str(t.get("entry_trading_day") or t.get("trading_day")) for t in trades)
        if set(counts) - set(study.calendar):
            raise ValueError(f"funded entry outside evaluation calendar: {key}")
        n = len(study.calendar)
        row.update(
            funded_entries=len(trades),
            evaluated_dates=n,
            entry_dates=len(counts),
            average_entries_per_evaluated_date=len(trades) / n if n else None,
            entry_date_coverage=len(counts) / n if n else None,
            zero_entry_dates=sum(not counts[d] for d in study.calendar),
            one_entry_dates=sum(counts[d] == 1 for d in study.calendar),
            multiple_entry_dates=sum(counts[d] > 1 for d in study.calendar),
        )
        longest: list[str] = []
        current: list[str] = []
        for day in study.calendar:
            current = [*current, day] if not counts[day] else []
            if len(current) > len(longest):
                longest = current[:]
        row["longest_no_entry_evaluated_dates"] = len(longest)
        row["longest_no_entry_start"] = longest[0] if longest else None
        row["longest_no_entry_end"] = longest[-1] if longest else None
        row["processing_duration_seconds"] = (
            _duration(events, cutoff, "processing") if events else None
        )
        row["protection_duration_seconds"] = (
            _duration(events, cutoff, "secured") if events else None
        )
        milestones = [
            t
            for t in trades
            if not t.get("account_failed")
            and t.get("exit_utc")
            and cents(t, "balance_after") is not None
            and cents(t, "balance_after") >= 200_000
        ]
        milestones.sort(key=lambda t: (stamp(t["exit_utc"]).value, int(t.get("seq") or 0)))
        first = milestones[0] if milestones else None
        requests = sorted(
            [
                r
                for r in study.rows("payout_events", key, FIRM)
                if r.get("event") in {"requested", "secured"}
            ],
            key=_event_key,
        )
        row["first_request_utc"] = requests[0]["ts_utc"] if requests else None
        row["first_request_basis"] = "saved request event; eligibility not inferred"
        eligible = sorted(
            [
                r
                for r in study.rows("payout_events", key, FIRM)
                if r.get("event") == "eligibility_secured"
            ],
            key=_event_key,
        )
        row["first_eligibility_utc"] = eligible[0]["ts_utc"] if eligible else None
        row["first_eligibility_basis"] = "actual saved secured eligibility event"
        row["first_eligibility_event_seq"] = eligible[0].get("seq") if eligible else None
        row["first_eligibility_account"] = eligible[0].get("account_number") if eligible else None
        statuses["first_eligibility_utc"] = "reached" if eligible else "unavailable"
        for prefix, event, time_field in (
            ("first_flat_buffer", first, "exit_utc"),
            ("first_receipt", receipts[0] if receipts else None, "ts_utc"),
        ):
            status = "reached" if event and start else "not_reached" if start else "unavailable"
            row[prefix + "_status"] = status
            row[prefix + "_utc"] = event[time_field] if event else None
            for suffix in ("elapsed_seconds", "calendar_days", "evaluated_date_days"):
                row[prefix + "_" + suffix] = (
                    clock(start, event[time_field], study.calendar)[suffix]
                    if event and start
                    else None
                )
                statuses[prefix + "_" + suffix] = status
        row["first_flat_buffer_account"] = first.get("account_number") if first else None
        row["first_flat_buffer_trade_ref"] = first.get("trade_ref") if first else None
        row["accounts_failed_before_first_buffer"] = (
            sum(stamp(r["failed_utc"]) < stamp(first["exit_utc"]) for r in failures)
            if first
            else None
        )
        before = (
            [r for r in purchases if _event_key(r) < _event_key(receipts[0])] if receipts else []
        )
        row["purchases_before_first_receipt"] = len(before) if receipts else None
        row["spend_before_first_receipt_cents"] = (
            sum(cents(r, "amount") for r in before) if receipts else None
        )
        intervals = []
        if start:
            ends = [r["ts_utc"] for r in receipts]
            bounds = [start, *ends, cutoff]
            for i, (left, right) in enumerate(zip(bounds, bounds[1:], strict=False)):
                kind = (
                    "initial"
                    if i == 0
                    else "terminal"
                    if i == len(bounds) - 2
                    else "between_receipts"
                )
                intervals.append(
                    {
                        "start_utc": left,
                        "end_utc": right,
                        "kind": kind,
                        "right_censored": not receipts or kind == "terminal",
                        **clock(left, right, study.calendar),
                    }
                )
        row["no_receipt_intervals"] = intervals
        longest_interval = max(intervals, key=lambda r: r["elapsed_seconds"]) if intervals else {}
        row["longest_no_receipt_calendar_days"] = longest_interval.get("calendar_days")
        row["longest_no_receipt_kind"] = longest_interval.get("kind")
        row["longest_no_receipt_right_censored"] = longest_interval.get("right_censored")
        account_statuses = []
        conditional_account_days = []
        for journey in journeys:
            account = journey.get("account_number")
            reached = next((t for t in milestones if t.get("account_number") == account), None)
            paid = [r for r in receipts if r.get("account_number") == account]
            account_trades = [t for t in trades if t.get("account_number") == account]
            cushion_evidence = all(
                t.get("exit_utc") and cents(t, "balance_after") is not None for t in account_trades
            ) and (journey.get("trades") is None or len(account_trades) == journey["trades"])
            receipt_evidence = (
                journey.get("payouts_received") is None or len(paid) == journey["payouts_received"]
            )
            if reached and cushion_evidence:
                conditional_account_days.append(
                    clock(journey["created_utc"], reached["exit_utc"], study.calendar)[
                        "calendar_days"
                    ]
                )
            account_statuses.append(
                {
                    "account_number": account,
                    "cushion_status": "unavailable"
                    if not cushion_evidence
                    else "reached"
                    if reached
                    else "failed_before_reaching"
                    if journey.get("failed_utc")
                    else "right_censored",
                    "receipt_status": "unavailable"
                    if not receipt_evidence
                    else "reached"
                    if paid
                    else "failed_before_reaching"
                    if journey.get("failed_utc")
                    else "right_censored",
                    "cushion_utc": reached.get("exit_utc") if reached else None,
                    "receipt_utc": paid[0]["ts_utc"] if paid else None,
                }
            )
        row["account_milestone_statuses"] = account_statuses
        row["cushion_status_counts"] = dict(Counter(r["cushion_status"] for r in account_statuses))
        row["receipt_status_counts"] = dict(Counter(r["receipt_status"] for r in account_statuses))
        for prefix, counts in (
            ("account_buffer", row["cushion_status_counts"]),
            ("account_receipt", row["receipt_status_counts"]),
        ):
            for suffix, status in (
                ("reached", "reached"),
                ("failed", "failed_before_reaching"),
                ("right_censored", "right_censored"),
                ("unavailable", "unavailable"),
            ):
                row[prefix + "_" + suffix] = counts.get(status, 0)
        row["conditional_median_account_days_to_buffer"] = (
            median(conditional_account_days) if conditional_account_days else None
        )
        statuses["conditional_median_account_days_to_buffer"] = (
            "measured" if conditional_account_days else "not_reached"
        )
        for name, value in list(row.items()):
            if name not in statuses and isinstance(value, (int, float)):
                statuses[name] = "measured"
        output.append(row)
    for rank, row in enumerate(sort_rows(output, "total_cash"), 1):
        row["overall_cash_rank"] = rank if row.get("net_received_cash_cents") is not None else None
    return output


def measured(row: Mapping[str, Any], metric: str) -> bool:
    return row.get(metric) is not None and row.get("measurement_status", {}).get(metric) not in {
        "unavailable",
        "not_reached",
        "not_applicable",
        "right_censored",
    }


def sort_rows(rows: Sequence[dict], lens: str) -> list[dict]:
    if lens not in LENSES:
        raise ValueError("unknown comparison lens")

    def compare(left, right):
        for metric, direction in LENSES[lens][1]:
            lk, rk = measured(left, metric), measured(right, metric)
            if lk != rk:
                return -1 if lk else 1
            if lk and left[metric] != right[metric]:
                value = -1 if left[metric] < right[metric] else 1
                return value if direction == "asc" else -value
        return (left["configuration_id"] > right["configuration_id"]) - (
            left["configuration_id"] < right["configuration_id"]
        )

    return sorted(rows, key=cmp_to_key(compare))


def validate_targets(targets: Mapping[str, Any]) -> list[str]:
    errors = []
    for metric, target in targets.items():
        if metric not in TARGETS:
            errors.append(f"Unknown target: {metric}")
        elif TARGETS[metric][0] == "range" and target[0] > target[1]:
            errors.append("Minimum average entries must be at most the maximum.")
        elif any(
            not math.isfinite(float(v))
            for v in (target if isinstance(target, (list, tuple)) else [target])
        ):
            errors.append("Targets must be finite numbers.")
        elif metric == "entry_date_coverage" and not 0 <= target <= 1:
            errors.append("Entry-date coverage must be between 0% and 100%.")
        elif metric != "net_received_cash_cents" and any(
            v < 0 for v in (target if isinstance(target, (list, tuple)) else [target])
        ):
            errors.append("Counts, activity and waiting targets cannot be negative.")
    return errors


def target_violations(
    row: Mapping[str, Any], targets: Mapping[str, Any]
) -> tuple[list[dict], list[dict]]:
    failed, unknown = [], []
    for metric, target in targets.items():
        operator, label, unit = TARGETS[metric]
        if not measured(row, metric):
            unknown.append(
                {
                    "metric": metric,
                    "target": target,
                    "reason": "cannot verify this target",
                    "status": row.get("measurement_status", {}).get(metric, "unavailable"),
                }
            )
            continue
        actual = row[metric]
        boundary = (
            (target[0] if actual < target[0] else target[1]) if operator == "range" else target
        )
        passes = (
            target[0] <= actual <= target[1]
            if operator == "range"
            else actual >= target
            if operator == "minimum"
            else actual <= target
        )
        if not passes:
            failed.append(
                {
                    "metric": metric,
                    "requirement": label,
                    "actual": actual,
                    "target": target,
                    "miss": abs(actual - boundary),
                    "unit": unit,
                }
            )
    return failed, unknown


def filter_rows(
    rows: Sequence[dict],
    *,
    lens: str = "total_cash",
    targets: Mapping[str, Any] | None = None,
    search: str = "",
    categories: Mapping[str, Sequence[str]] | None = None,
    closest: bool = False,
) -> dict[str, Any]:
    targets, categories = targets or {}, categories or {}
    errors = validate_targets(targets)
    if errors:
        return {
            "rows": [],
            "matches": 0,
            "unknown": 0,
            "failed": 0,
            "errors": errors,
            "total": len(rows),
        }

    def searchable(row):
        descriptions = [policy_label(axis, str(row.get(axis, ""))) for axis in AXES]
        aliases = "partial half runner" if row.get("exit") in {"XP", "XG", "XE"} else "whole"
        return (
            " ".join(
                [
                    str(row.get("configuration_id", "")),
                    *descriptions,
                    aliases,
                    *[str(row.get(k, "")) for k in AXES],
                ]
            )
        ).casefold()

    candidates = [
        r
        for r in rows
        if (not search or all(word in searchable(r) for word in search.casefold().split()))
        and all(not values or r.get(axis) in values for axis, values in categories.items())
    ]
    ranked = sort_rows(candidates, lens)
    matches, failed_count, unknown_count = [], 0, 0
    annotated = []
    for row in ranked:
        violations, unknowns = target_violations(row, targets)
        projection = dict(
            row,
            target_violations=violations,
            unverifiable_targets=unknowns,
            failed_target_count=len(violations),
            unverifiable_target_count=len(unknowns),
        )
        annotated.append(projection)
        failed_count += bool(violations)
        unknown_count += bool(unknowns)
        if not violations and not unknowns:
            matches.append(projection)
    displayed = sorted(annotated, key=lambda r: r["failed_target_count"]) if closest else matches
    return {
        "rows": displayed,
        "matches": len(matches),
        "total": len(rows),
        "category_count": len(candidates),
        "unknown": unknown_count,
        "failed": failed_count,
        "errors": [],
        "closest": closest,
    }


def export_view(
    view: Mapping[str, Any],
    *,
    lens: str,
    targets: Mapping[str, Any],
    categories: Mapping[str, Any],
    search: str,
) -> str:
    return json.dumps(
        {
            "reporting_version": VERSION,
            "lens": lens,
            "targets": targets,
            "categories": categories,
            "search": search,
            "scope": "full_study",
            "population": "actual_funded",
            "ordering": "failed requirement count, selected lens; unverifiable targets separate"
            if view.get("closest")
            else "selected lens",
            "match_count": view["matches"],
            "complete_count": view["total"],
            "rows": view["rows"],
        },
        allow_nan=False,
        indent=2,
    )
