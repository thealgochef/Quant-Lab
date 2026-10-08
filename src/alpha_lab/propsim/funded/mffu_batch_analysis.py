"""Pure, saved-result analysis for the bounded 64-intent MFFU batch.

Cash comparisons use completed MyFundedFutures summaries in exact cents.
Cash-event timing comes from the saved ledger; entry-session, runner, and HTF
zone views describe funded trade P&L and never allocate received payouts to
trades. Missing historical HTF zone IDs stay in an explicit unknown bucket.
No replay, market read, approval mutation, or file write occurs here.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from alpha_lab.propsim.funded.reporting_accounts import account_token

SCHEMA = "ifsm_mffu_batch_analysis_v2"
FIRM_KEY = "myfundedfutures"
_CHICAGO = ZoneInfo("America/Chicago")
_CORE_SCHEDULES = ("S0", "S1")
_CORE_CAPS = ("U", "D1")
_CORE_CONTEXTS = ("F0", "FE", "FL", "FEL")
_CORE_EXITS = ("XP", "XF", "XG")
_FIELDS = (
    "variant_id", "family", "schedule", "daily_cap", "entry_context", "exit",
    "sizing", "overhead", "geometry",
)


def _timestamp(value: str | datetime, field: str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be an aware timestamp")
    return value.astimezone(UTC)


def _stamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _cents(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be exact cents")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.lstrip("-").isdecimal():
        return int(value)
    raise ValueError(f"{field} must be exact cents")


def _usd_cents(value: Any, field: str) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} must be exact USD")
    number = Decimal(str(value)) * 100
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{field} must be exact USD cents")
    return int(number)


def _row_cents(row: Mapping[str, Any], stem: str) -> int:
    if row.get(stem + "_cents") is not None:
        return _cents(row[stem + "_cents"], stem + "_cents")
    return _usd_cents(row.get(stem + "_usd"), stem + "_usd")


def _variant(item: Mapping[str, Any] | Any) -> dict[str, Any]:
    row = dict(item) if isinstance(item, Mapping) else item.model_dump()
    intent = json.loads(row["intent_json"]) if "intent_json" in row else row
    out = {field: intent.get(field, row.get(field)) for field in _FIELDS}
    if any(out[field] is None for field in _FIELDS):
        raise ValueError(f"variant is missing a declared axis: {out}")
    if row.get("variant_id") is not None and row["variant_id"] != out["variant_id"]:
        raise ValueError("plan variant identity differs from saved intent")
    return out


def _variants(items: Iterable[Mapping[str, Any] | Any]) -> dict[str, dict[str, Any]]:
    variants = [_variant(item) for item in items]
    ids = [row["variant_id"] for row in variants]
    if ids != [f"MCB{number:03d}" for number in range(1, 65)]:
        raise ValueError("analysis requires every ordered MCB001..MCB064 intent")
    core = [row for row in variants if row["family"] == "core_48"]
    actual = {
        (row["schedule"], row["daily_cap"], row["entry_context"], row["exit"])
        for row in core
    }
    expected = {
        (schedule, cap, context, exit_policy)
        for schedule in _CORE_SCHEDULES
        for cap in _CORE_CAPS
        for context in _CORE_CONTEXTS
        for exit_policy in _CORE_EXITS
    }
    if len(core) != 48 or actual != expected:
        raise ValueError("the 48 core interaction cells are not the full declared cross")
    if any(
        row["sizing"] != "Q10" or row["overhead"] != "O0" or row["geometry"] != "G0"
        for row in core
    ):
        raise ValueError("a core cell changed a noncore axis")
    return {row["variant_id"]: row for row in variants}


def _result_rows(result: Mapping[str, Any], variants: Mapping[str, dict]) -> dict[str, dict]:
    summaries = result.get("summaries_cents")
    if not isinstance(summaries, Mapping):
        raise ValueError("saved result lacks summaries_cents")
    if any(summary.get("firm_key") != FIRM_KEY for summary in summaries.values()):
        raise ValueError("the bounded result contains a different funded firm")
    if any(key not in {f"{variant_id}|{FIRM_KEY}" for variant_id in variants}
           for key in summaries):
        raise ValueError("the bounded result contains an undeclared configuration")
    rows: dict[str, dict] = {}
    for variant_id, axes in variants.items():
        summary = summaries.get(f"{variant_id}|{FIRM_KEY}")
        if not isinstance(summary, Mapping):
            rows[variant_id] = {
                **axes, "status": "unavailable", "reason": "missing_saved_summary",
                "net_cash_cents": None,
            }
            continue
        if summary.get("configuration") != variant_id or summary.get("firm_key") != FIRM_KEY:
            raise ValueError(f"saved summary identity mismatch for {variant_id}")
        if summary.get("status") != "Completed":
            rows[variant_id] = {
                **axes, "status": "unavailable",
                "reason": summary.get("reason") or summary.get("status") or "not_completed",
                "net_cash_cents": None,
            }
            continue
        received = _cents(summary.get("payouts_received_cents"), "payouts_received_cents")
        costs = _cents(summary.get("account_costs_cents"), "account_costs_cents")
        net = _cents(summary.get("net_cash_earned_cents"), "net_cash_earned_cents")
        if received - costs != net:
            raise ValueError(f"saved cash summary does not reconcile for {variant_id}")
        rows[variant_id] = {
            **axes, "status": "completed", "reason": None,
            "payouts_received_cents": received, "account_costs_cents": costs,
            "payouts_received_count": summary.get("payouts_received_count"),
            "trades_taken": summary.get("trades_taken"),
            "net_cash_cents": net,
        }
    return rows


def _delta(rows: Mapping[str, dict], terms: Mapping[str, int]) -> dict[str, Any]:
    missing = [variant_id for variant_id in terms if rows[variant_id]["status"] != "completed"]
    if missing:
        return {"status": "unavailable", "delta_cents": None, "unavailable_ids": missing}
    return {
        "status": "available",
        "delta_cents": sum(rows[variant_id]["net_cash_cents"] * sign
                           for variant_id, sign in terms.items()),
        "unavailable_ids": [],
    }


def _matched_pairs(
    rows: Mapping[str, dict], pairs: Iterable[Mapping[str, Any]], expected_count: int | None,
) -> list[dict]:
    pairs = list(pairs)
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError("comparison pair count differs from the declared handoff")
    seen: set[tuple[str, str]] = set()
    out = []
    for pair in pairs:
        base, challenger = pair["base_id"], pair["challenger_id"]
        if base not in rows or challenger not in rows or base == challenger:
            raise ValueError("comparison pair references an invalid variant")
        if (base, challenger) in seen:
            raise ValueError("duplicate directed comparison pair")
        seen.add((base, challenger))
        out.append({
            "base_id": base,
            "challenger_id": challenger,
            "changed_axis": pair.get("changed_axis"),
            "group": pair.get("group"),
            **_delta(rows, {challenger: 1, base: -1}),
        })
    return out


def _interaction_summary(cells: list[dict]) -> dict[str, Any]:
    deltas = [cell["delta_cents"] for cell in cells if cell["status"] == "available"]
    return {
        "available_cells": len(deltas),
        "unavailable_cells": len(cells) - len(deltas),
        "min_delta_cents": min(deltas) if deltas else None,
        "max_delta_cents": max(deltas) if deltas else None,
    }


def _interactions(rows: Mapping[str, dict]) -> dict[str, Any]:
    core = {
        (row["schedule"], row["daily_cap"], row["entry_context"], row["exit"]): key
        for key, row in rows.items() if row["family"] == "core_48"
    }
    fe_fl, cap_exit, schedule_cap = [], [], []
    for schedule in _CORE_SCHEDULES:
        for cap in _CORE_CAPS:
            for exit_policy in _CORE_EXITS:
                terms = {
                    core[schedule, cap, "FEL", exit_policy]: 1,
                    core[schedule, cap, "FE", exit_policy]: -1,
                    core[schedule, cap, "FL", exit_policy]: -1,
                    core[schedule, cap, "F0", exit_policy]: 1,
                }
                fe_fl.append({
                    "schedule": schedule, "daily_cap": cap, "exit": exit_policy,
                    **_delta(rows, terms),
                })
        for context in _CORE_CONTEXTS:
            for challenger_exit in ("XF", "XG"):
                terms = {
                    core[schedule, "D1", context, challenger_exit]: 1,
                    core[schedule, "U", context, challenger_exit]: -1,
                    core[schedule, "D1", context, "XP"]: -1,
                    core[schedule, "U", context, "XP"]: 1,
                }
                cap_exit.append({
                    "schedule": schedule, "entry_context": context,
                    "challenger_exit": challenger_exit, "base_exit": "XP",
                    **_delta(rows, terms),
                })
    for context in _CORE_CONTEXTS:
        for exit_policy in _CORE_EXITS:
            terms = {
                core["S1", "D1", context, exit_policy]: 1,
                core["S1", "U", context, exit_policy]: -1,
                core["S0", "D1", context, exit_policy]: -1,
                core["S0", "U", context, exit_policy]: 1,
            }
            schedule_cap.append({
                "entry_context": context, "exit": exit_policy,
                **_delta(rows, terms),
            })
    return {
        "fe_x_fl": {"cells": fe_fl, **_interaction_summary(fe_fl)},
        "cap_x_exit_vs_xp": {"cells": cap_exit, **_interaction_summary(cap_exit)},
        "schedule_x_cap": {"cells": schedule_cap, **_interaction_summary(schedule_cap)},
    }


def _interval(
    start: datetime, end: datetime, kind: str, evaluation_dates: Sequence[date],
    *, censored_start: bool = False, censored_end: bool = False,
) -> dict[str, Any]:
    start_day, end_day = start.astimezone(_CHICAGO).date(), end.astimezone(_CHICAGO).date()
    return {
        "kind": kind,
        "start_utc": _stamp(start), "end_utc": _stamp(end),
        "start_chicago_date": start_day.isoformat(),
        "end_chicago_date": end_day.isoformat(),
        "calendar_days": (end_day - start_day).days,
        "elapsed_seconds": int((end - start).total_seconds()),
        "evaluated_trading_days": sum(start_day < day <= end_day for day in evaluation_dates),
        "censored_start": censored_start, "censored_end": censored_end,
    }


def _waiting(
    rows: Mapping[str, dict], payout_events: Sequence[Mapping[str, Any]] | None,
    start: datetime, cutoff: datetime, evaluation_dates: Sequence[date],
) -> dict[str, dict]:
    if payout_events is None:
        return {key: {"status": "unavailable", "reason": "payout_events_absent"}
                for key in rows}
    receipts: dict[str, list[datetime]] = defaultdict(list)
    for event in payout_events:
        if event.get("firm_key") != FIRM_KEY or event.get("event") != "received":
            continue
        key = event.get("configuration")
        if key not in rows:
            continue
        stamp = _timestamp(event.get("received_utc") or event.get("ts_utc"), "receipt")
        if not start <= stamp <= cutoff:
            raise ValueError(f"received payout outside the authorized period: {key}")
        receipts[key].append(stamp)
    out = {}
    for key, row in rows.items():
        if row["status"] != "completed":
            out[key] = {"status": "unavailable", "reason": row["reason"]}
            continue
        times = sorted(receipts[key])
        expected_count = row.get("payouts_received_count")
        if expected_count is not None and len(times) != expected_count:
            out[key] = {"status": "unavailable", "reason": "receipt_event_count_mismatch"}
            continue
        if not times and row["payouts_received_cents"]:
            out[key] = {"status": "unavailable", "reason": "receipt_events_not_materialized"}
            continue
        initial = _interval(
            start, times[0] if times else cutoff, "initial", evaluation_dates,
            censored_start=True, censored_end=not times,
        )
        between = [
            _interval(left, right, "between_receipts", evaluation_dates)
            for left, right in zip(times, times[1:], strict=False)
        ]
        terminal = _interval(
            times[-1], cutoff, "terminal", evaluation_dates, censored_end=True
        ) if times else None
        all_intervals = [initial, *between, *([terminal] if terminal else [])]
        longest = max(all_intervals, key=lambda item: item["elapsed_seconds"])
        out[key] = {
            "status": "available",
            "received_count": len(times),
            "initial": initial,
            "max_between_receipts": (
                max(between, key=lambda item: item["elapsed_seconds"]) if between else None
            ),
            "terminal": terminal,
            "max_no_receipt_interval": longest,
            "between_status": "available" if between else "fewer_than_two_receipts",
            "terminal_status": "available" if terminal else "no_receipt",
        }
    return out


def _months(start: datetime, cutoff: datetime) -> list[str]:
    first = start.astimezone(_CHICAGO).date()
    last = cutoff.astimezone(_CHICAGO).date()
    year, month = first.year, first.month
    out = []
    while (year, month) <= (last.year, last.month):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def _cash_concentration(
    rows: Mapping[str, dict], cash_ledger: Sequence[Mapping[str, Any]] | None,
    start: datetime, cutoff: datetime,
) -> dict[str, dict]:
    if cash_ledger is None:
        return {key: {"status": "unavailable", "reason": "cash_ledger_absent"}
                for key in rows}
    by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in cash_ledger:
        if event.get("firm_key") == FIRM_KEY and event.get("configuration") in rows:
            by_key[event["configuration"]].append(event)
    months = _months(start, cutoff)
    out = {}
    for key, row in rows.items():
        if row["status"] != "completed":
            out[key] = {"status": "unavailable", "reason": row["reason"]}
            continue
        if not by_key[key] and (row["payouts_received_cents"] or row["account_costs_cents"]):
            out[key] = {"status": "unavailable", "reason": "cash_events_not_materialized"}
            continue
        buckets = {month: {"received_cents": 0, "account_costs_cents": 0}
                   for month in months}
        for event in by_key[key]:
            stamp = _timestamp(event["ts_utc"], "cash event")
            if not start <= stamp <= cutoff:
                raise ValueError(f"cash event outside the authorized period: {key}")
            month = stamp.astimezone(_CHICAGO).strftime("%Y-%m")
            amount = _row_cents(event, "amount")
            if amount < 0:
                raise ValueError("saved purchase and receipt amounts must be nonnegative")
            if event["kind"] == "payout_received":
                buckets[month]["received_cents"] += amount
            elif event["kind"] == "account_purchase":
                buckets[month]["account_costs_cents"] += amount
            else:
                raise ValueError(f"unsupported saved cash event kind: {event['kind']}")
        total_received = sum(value["received_cents"] for value in buckets.values())
        total_costs = sum(value["account_costs_cents"] for value in buckets.values())
        if (total_received, total_costs) != (
            row["payouts_received_cents"], row["account_costs_cents"]
        ):
            raise ValueError(f"saved cash ledger does not reconcile for {key}")
        monthly = []
        cumulative = 0
        for month, amounts in buckets.items():
            net = amounts["received_cents"] - amounts["account_costs_cents"]
            cumulative += net
            monthly.append({"month": month, **amounts, "net_cash_cents": net,
                            "cumulative_net_cash_cents": cumulative})
        years: dict[str, dict[str, int]] = {}
        for item in monthly:
            year = item["month"][:4]
            year_row = years.setdefault(year, {"received_cents": 0, "account_costs_cents": 0,
                                               "net_cash_cents": 0})
            for name in year_row:
                year_row[name] += item[name]
        out[key] = {
            "status": "available", "basis": "cash_event_chicago_date",
            "monthly": monthly, "yearly": years,
            "zero_receipt_months": [item["month"] for item in monthly
                                    if item["received_cents"] == 0],
        }
    return out


def _trade_concentration(
    rows: Mapping[str, dict], trades: Sequence[Mapping[str, Any]] | None,
    start: datetime, cutoff: datetime,
) -> dict[str, dict]:
    if trades is None:
        return {key: {"status": "unavailable", "reason": "funded_trades_absent"}
                for key in rows}
    by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in trades:
        if trade.get("firm_key") == FIRM_KEY and trade.get("configuration") in rows:
            by_key[trade["configuration"]].append(trade)
    out = {}
    for key, row in rows.items():
        if row["status"] != "completed":
            out[key] = {"status": "unavailable", "reason": row["reason"]}
            continue
        expected_trades = row.get("trades_taken")
        if expected_trades is not None and len(by_key[key]) != expected_trades:
            out[key] = {"status": "unavailable", "reason": "funded_trade_count_mismatch"}
            continue
        years: dict[str, dict[str, int]] = {}
        sessions: dict[str, dict[str, int]] = {}
        resolutions: dict[str, dict[str, int]] = {}
        accounts: dict[str, dict[str, int]] = {}
        unknown_account = {"trades": 0, "net_trade_pnl_cents": 0}
        htf_zones: dict[str, dict[str, int]] = {}
        unknown_htf_zone = {"trades": 0, "net_trade_pnl_cents": 0}
        scored = []
        total_trade_pnl_cents = 0
        for index, trade in enumerate(by_key[key]):
            stamp = _timestamp(trade["entry_utc"], "funded trade entry")
            if not start <= stamp <= cutoff:
                raise ValueError(f"funded trade entry outside the authorized period: {key}")
            money = _row_cents(trade, "net_pnl")
            total_trade_pnl_cents += money
            year = str(stamp.astimezone(_CHICAGO).year)
            session = str(trade.get("entry_session") or "unknown")
            account_id = account_token(trade)
            zone_id = trade.get("htf_zone_id")
            if zone_id is not None and (not isinstance(zone_id, str) or not zone_id.strip()):
                raise ValueError(f"funded trade has an invalid HTF zone ID: {key}")
            partial = _cents(trade.get("scale_out_quantity") or 0, "scale_out_quantity") > 0
            exit_kind = trade.get("exit_kind")
            resolution = (
                "partial_to_scheduled_close" if partial and exit_kind == "scheduled_close"
                else "partial_to_breakeven_stop" if partial and exit_kind == "breakeven_stop"
                else "partial_other" if partial else "no_partial"
            )
            for mapping, label in ((years, year), (sessions, session),
                                   (resolutions, resolution)):
                bucket = mapping.setdefault(label, {"trades": 0, "net_trade_pnl_cents": 0})
                bucket["trades"] += 1
                bucket["net_trade_pnl_cents"] += money
            account_bucket = (accounts.setdefault(
                account_id, {"trades": 0, "net_trade_pnl_cents": 0},
            ) if account_id is not None else unknown_account)
            account_bucket["trades"] += 1
            account_bucket["net_trade_pnl_cents"] += money
            zone_bucket = (htf_zones.setdefault(
                zone_id, {"trades": 0, "net_trade_pnl_cents": 0},
            ) if zone_id is not None else unknown_htf_zone)
            zone_bucket["trades"] += 1
            zone_bucket["net_trade_pnl_cents"] += money
            scored.append({
                "trade_id": trade.get("trade_ref") or trade.get("strategy_trade_id")
                            or f"{key}:{index}",
                "entry_utc": _stamp(stamp), "net_trade_pnl_cents": money,
                "resolution": resolution,
            })
        for grouped in (years, sessions, resolutions):
            if (sum(bucket["trades"] for bucket in grouped.values()) != len(by_key[key])
                    or sum(bucket["net_trade_pnl_cents"] for bucket in grouped.values())
                    != total_trade_pnl_cents):
                raise ValueError(f"funded trade concentration does not reconcile for {key}")
        if (sum(bucket["trades"] for bucket in htf_zones.values())
                + unknown_htf_zone["trades"] != len(by_key[key])
                or sum(bucket["net_trade_pnl_cents"] for bucket in htf_zones.values())
                + unknown_htf_zone["net_trade_pnl_cents"] != total_trade_pnl_cents):
            raise ValueError(f"funded HTF zone concentration does not reconcile for {key}")
        if (sum(bucket["trades"] for bucket in accounts.values())
                + unknown_account["trades"] != len(by_key[key])
                or sum(bucket["net_trade_pnl_cents"] for bucket in accounts.values())
                + unknown_account["net_trade_pnl_cents"] != total_trade_pnl_cents):
            raise ValueError(f"funded account concentration does not reconcile for {key}")
        unknown_count = unknown_htf_zone["trades"]
        zone_coverage = (
            "no_trades" if not by_key[key] else "unavailable" if unknown_count == len(by_key[key])
            else "partial_unknown" if unknown_count else "available"
        )
        out[key] = {
            "status": "available",
            "basis": "actual_funded_trade_net_pnl_not_received_cash",
            "yearly_by_entry": years, "by_entry_session": sessions,
            "by_resolution": resolutions,
            "by_account_id": dict(sorted(accounts.items())),
            "unknown_account": unknown_account,
            "account_coverage": {
                "status": (
                    "no_trades" if not by_key[key] else "unavailable"
                    if unknown_account["trades"] == len(by_key[key]) else "partial_unknown"
                    if unknown_account["trades"] else "available"
                ),
                "known_trades": len(by_key[key]) - unknown_account["trades"],
                "unknown_trades": unknown_account["trades"],
                "known_account_count": len(accounts),
                "reason": "account_identity_absent_or_null" if unknown_account["trades"] else None,
            },
            "by_htf_zone_id": dict(sorted(htf_zones.items())),
            "unknown_htf_zone": unknown_htf_zone,
            "htf_zone_coverage": {
                "status": zone_coverage,
                "known_trades": len(by_key[key]) - unknown_count,
                "unknown_trades": unknown_count,
                "known_zone_count": len(htf_zones),
                "reason": "htf_zone_id_absent_or_null" if unknown_count else None,
            },
            "largest_trades": sorted(
                scored, key=lambda item: (-item["net_trade_pnl_cents"], item["trade_id"])
            )[:5],
            "runner_net_note": "whole-trade P&L, not isolated residual-leg profit",
        }
    return out


_TRADE_FIELDS = (
    "stop_ticks", "target_ticks", "exit_ticks", "exit_kind", "scale_out_quantity",
)
_TRADE_AUDIT_FIELDS = (
    "account_id", "entry_trading_day", "quantity", "final_exit_quantity",
    "initial_risk_cents", "htf_zone_id",
)


def _funded_trade_index(
    rows: Mapping[str, dict], trades: Sequence[Mapping[str, Any]] | None,
    start: datetime, cutoff: datetime,
) -> dict[str, dict]:
    if trades is None:
        return {key: {"status": "unavailable", "reason": "funded_trades_absent"}
                for key in rows}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for trade in trades:
        key = trade.get("configuration")
        if trade.get("firm_key") == FIRM_KEY and key in rows:
            grouped[key].append(trade)
    out = {}
    for key, row in rows.items():
        if row["status"] != "completed":
            out[key] = {"status": "unavailable", "reason": row["reason"]}
            continue
        actual = grouped[key]
        expected = row.get("trades_taken")
        if expected is not None and len(actual) != expected:
            out[key] = {"status": "unavailable", "reason": "funded_trade_count_mismatch"}
            continue
        entries = {}
        reason = None
        for trade in actual:
            if any(trade.get(field) is None for field in
                   ("entry_utc", "entry_ticks", "direction")):
                reason = "funded_entry_identity_absent"
                break
            stamp = _timestamp(trade["entry_utc"], "funded trade entry")
            if not start <= stamp <= cutoff:
                raise ValueError(f"funded trade entry outside the authorized period: {key}")
            direction = str(trade["direction"]).lower().split(".")[-1]
            signature = (_stamp(stamp), direction, int(trade["entry_ticks"]))
            if signature in entries:
                reason = "duplicate_funded_entry_signature"
                break
            entries[signature] = trade
        out[key] = ({"status": "unavailable", "reason": reason} if reason else
                    {"status": "available", "entries": entries})
    return out


def _entry_id(signature: tuple[str, str, int]) -> str:
    return f"{signature[0]}|{signature[1]}|{signature[2]}"


def _trade_projection(trade: Mapping[str, Any]) -> dict:
    return {
        "trade_ref": trade.get("trade_ref") or trade.get("strategy_trade_id"),
        **{field: trade.get(field) for field in _TRADE_FIELDS},
        **{field: trade.get(field) for field in _TRADE_AUDIT_FIELDS},
        "account_id": account_token(trade),
        "reporting_account_key": trade.get("reporting_account_key"),
        "trading_costs_cents": (
            _row_cents(trade, "costs") if (trade.get("costs_cents") is not None
                                           or trade.get("costs_usd") is not None) else None
        ),
        "net_trade_pnl_cents": _row_cents(trade, "net_pnl"),
    }


def _matched_trade_diagnostics(
    rows: Mapping[str, dict], pairs: list[dict], index: Mapping[str, dict],
) -> None:
    """Attach actual funded trade changes; cash deltas remain account outcomes."""
    for pair in pairs:
        base_id, challenger_id = pair["base_id"], pair["challenger_id"]
        if pair["status"] != "available":
            pair["trade_diagnostics"] = {
                "status": "unavailable", "reason": "financial_row_unavailable",
            }
            continue
        base, challenger = index[base_id], index[challenger_id]
        if base["status"] != "available" or challenger["status"] != "available":
            pair["trade_diagnostics"] = {
                "status": "unavailable",
                "reason": (
                    f"{base_id}:{base.get('reason')};"
                    f"{challenger_id}:{challenger.get('reason')}"
                ),
            }
            continue
        base_entries, challenger_entries = base["entries"], challenger["entries"]
        common = sorted(base_entries.keys() & challenger_entries.keys())
        gained = sorted(challenger_entries.keys() - base_entries.keys())
        lost = sorted(base_entries.keys() - challenger_entries.keys())
        changed = []
        for signature in common:
            left = _trade_projection(base_entries[signature])
            right = _trade_projection(challenger_entries[signature])
            changed_fields = [field for field in (*_TRADE_FIELDS, "net_trade_pnl_cents")
                              if left[field] != right[field]]
            if changed_fields:
                changed.append({
                    "entry_key": _entry_id(signature), "changed_fields": changed_fields,
                    "base": left, "challenger": right,
                    "net_trade_pnl_delta_cents": (
                        right["net_trade_pnl_cents"] - left["net_trade_pnl_cents"]
                    ),
                })

        def additions(signatures, records):
            return [{"entry_key": _entry_id(signature), **_trade_projection(records[signature])}
                    for signature in signatures]

        pair["trade_diagnostics"] = {
            "status": "available",
            "basis": "actual_funded_trade_net_pnl_not_received_cash",
            "entry_match_basis": "UTC_entry_instant_direction_and_entry_ticks",
            "common_entries": len(common), "changed_common_entries": len(changed),
            "gained_entries": len(gained), "lost_entries": len(lost),
            "common_entry_pnl_delta_cents": sum(
                _row_cents(challenger_entries[key], "net_pnl")
                - _row_cents(base_entries[key], "net_pnl") for key in common
            ),
            "gained_entry_pnl_cents": sum(
                _row_cents(challenger_entries[key], "net_pnl") for key in gained
            ),
            "lost_entry_pnl_cents": sum(
                _row_cents(base_entries[key], "net_pnl") for key in lost
            ),
            "changed_common": changed,
            "gained": additions(gained, challenger_entries),
            "lost": additions(lost, base_entries),
        }


def _runner_resolution(trade: Mapping[str, Any]) -> str:
    partial = _cents(trade.get("scale_out_quantity") or 0, "scale_out_quantity") > 0
    if not partial:
        return "no_partial"
    if trade.get("exit_kind") == "scheduled_close":
        return "partial_to_daily_close"
    if trade.get("exit_kind") == "breakeven_stop":
        return "partial_to_entry_stop"
    return "partial_other"


def _entry_order_runner(index: Mapping[str, dict]) -> dict[str, dict]:
    out = {}
    for key, indexed in index.items():
        if indexed["status"] != "available":
            out[key] = {"status": "unavailable", "reason": indexed["reason"]}
            continue
        ordered = []
        for signature, trade in indexed["entries"].items():
            day = trade.get("entry_trading_day") or trade.get("trading_day")
            if not day:
                ordered = []
                break
            ordered.append((str(day), signature, trade))
        if indexed["entries"] and not ordered:
            out[key] = {"status": "unavailable", "reason": "entry_trading_day_absent"}
            continue
        ordered.sort(key=lambda item: (item[0], item[1]))
        by_order = {
            name: {"trades": 0, "net_trade_pnl_cents": 0, "by_resolution": {}}
            for name in ("first", "later")
        }
        previous_day = None
        for day, _signature, trade in ordered:
            label = "first" if day != previous_day else "later"
            previous_day = day
            money = _row_cents(trade, "net_pnl")
            bucket = by_order[label]
            bucket["trades"] += 1
            bucket["net_trade_pnl_cents"] += money
            resolution = _runner_resolution(trade)
            sub = bucket["by_resolution"].setdefault(
                resolution, {"trades": 0, "net_trade_pnl_cents": 0},
            )
            sub["trades"] += 1
            sub["net_trade_pnl_cents"] += money
        out[key] = {
            "status": "available", "basis": "actual_funded_trade_net_pnl_not_received_cash",
            "day_basis": "saved_entry_trading_day", "by_entry_order": by_order,
        }
    return out


def _count_bucket(counts: dict[str, dict[str, int]], name: str, value: Any) -> None:
    label = str(value if value is not None else "unknown")
    counts[name][label] = counts[name].get(label, 0) + 1


def _nominal_report_day(decision: datetime) -> date:
    chicago = decision.astimezone(_CHICAGO)
    return chicago.date() if chicago.hour >= 22 else chicago.date() - timedelta(days=1)


def _context_classifications(context: Mapping[str, Any]) -> dict[str, str]:
    """Classify saved decisions using only their recorded as-of context."""
    decision_raw = context.get("decision_ts_utc")
    decision = _timestamp(decision_raw, "context decision") if decision_raw else None
    classified = {}
    for prefix, status_key, eligible_key in (
        ("gamma", "gamma_status", "gamma_eligible_from_utc"),
        ("levels", "levels_status", "level_eligible_from_utc"),
    ):
        eligible_raw = context.get(eligible_key)
        if not eligible_raw:
            timing = "source_timestamp_unavailable"
        elif decision is None:
            timing = "decision_timestamp_unavailable"
        else:
            eligible = _timestamp(eligible_raw, eligible_key)
            if eligible > decision:
                raise ValueError(f"saved {prefix} context becomes eligible after its decision")
            timing = "eligible_by_nominal_schedule"
        classified[f"{prefix}_nominal_timing"] = timing
        status = context.get(status_key)
        classified[f"{prefix}_status"] = str(status if status is not None else "unknown")

    gamma_report = context.get("gamma_report_date")
    if gamma_report and decision:
        report_day = date.fromisoformat(str(gamma_report))
        nominal_day = _nominal_report_day(decision)
        if report_day > nominal_day:
            raise ValueError("saved gamma report date is after the nominal decision day")
        gamma_carry = ("current_nominal_day" if report_day == nominal_day
                       else "carried_prior_report")
    else:
        gamma_carry = "report_or_decision_date_unavailable"
    classified["gamma_report_carry"] = gamma_carry

    requested, level_report = (context.get("level_requested_date"),
                               context.get("level_report_date"))
    if requested and level_report:
        requested_day = date.fromisoformat(str(requested))
        report_day = date.fromisoformat(str(level_report))
        if report_day > requested_day:
            raise ValueError("saved level report date is after its requested date")
        level_carry = ("requested_report_day" if report_day == requested_day
                       else "carried_prior_report")
    else:
        level_carry = "request_or_report_date_unavailable"
    classified["level_report_carry"] = level_carry

    sign, age = context.get("gamma_sign"), context.get("positive_run_age")
    if sign != "positive":
        phase = "not_positive_or_unknown"
    elif age is None:
        phase = "positive_age_unknown"
    elif isinstance(age, int) and not isinstance(age, bool) and age >= 1:
        phase = "early_positive" if age <= 5 else "established_positive"
    else:
        raise ValueError("saved positive run age is invalid")
    classified["gamma_positive_phase"] = phase
    classified["positive_age_bound"] = (
        "lower_bound" if context.get("positive_run_age_is_lower_bound") else "exact_or_unknown"
    )
    return classified


def _decision_coverage(rows: Mapping[str, dict], batch: Mapping[str, Any]) -> dict[str, dict]:
    decisions = batch.get("decision_context")
    dispositions = {row["variant_id"]: row.get("status")
                    for row in batch.get("dispositions", [])}
    if not isinstance(decisions, list):
        return {key: {"status": "unavailable", "reason": "decision_context_absent"}
                for key in rows}
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for decision in decisions:
        key, stream = decision.get("configuration"), decision.get("stream")
        if key not in rows or stream not in {"strategy", "funded"}:
            raise ValueError("saved decision context has an undeclared configuration or stream")
        grouped[key][stream].append(decision)
    out = {}
    for key, row in rows.items():
        if row["status"] != "completed":
            out[key] = {"status": "unavailable", "reason": row["reason"]}
            continue
        if dispositions.get(key) in {"compatible_reused", "reused", "semantic_alias"}:
            out[key] = {"status": "unavailable",
                        "reason": "historical_policy_events_unavailable"}
            continue
        streams = {}
        for stream in ("strategy", "funded"):
            events = grouped[key][stream]
            counts: dict[str, dict[str, int]] = {
                name: {} for name in ("events", "actions", "rejection_reasons",
                                       "gamma_status", "levels_status", "gamma_sign",
                                       "geometry_fallback_reasons", "gamma_nominal_timing",
                                       "levels_nominal_timing", "gamma_report_carry",
                                       "level_report_carry", "gamma_positive_phase",
                                       "positive_age_bound")
            }

            quota_increments = 0
            actions_by_context_class: dict[str, dict[str, int]] = {}
            for event in events:
                _count_bucket(counts, "events", event.get("event"))
                _count_bucket(counts, "actions", event.get("action"))
                context = event.get("context") or {}
                for name in ("gamma_status", "levels_status", "gamma_sign"):
                    _count_bucket(counts, name, context.get(name))
                classified = _context_classifications(context)
                for name in ("gamma_nominal_timing", "levels_nominal_timing",
                             "gamma_report_carry", "level_report_carry",
                             "gamma_positive_phase", "positive_age_bound"):
                    _count_bucket(counts, name, classified[name])
                for name in ("gamma_status", "levels_status", "gamma_report_carry",
                             "level_report_carry"):
                    label = f"{name}:{classified[name]}"
                    action = str(event.get("action") or "unknown")
                    actions = actions_by_context_class.setdefault(label, {})
                    actions[action] = actions.get(action, 0) + 1
                for reason in event.get("reasons") or ():
                    _count_bucket(counts, "rejection_reasons", reason)
                if event.get("fallback_reason"):
                    _count_bucket(counts, "geometry_fallback_reasons",
                                  event["fallback_reason"])
                before, after = event.get("quota_before"), event.get("quota_after")
                if before is not None and after is not None and int(after) > int(before):
                    quota_increments += 1
            streams[stream] = {
                "decision_count": len(events), **counts,
                "actions_by_context_class": actions_by_context_class,
                "quota_increments": quota_increments,
                "timing_basis": "nominal_eligibility_not_observed_vendor_publication",
            }
        out[key] = {"status": "available", "streams": streams}
    return out


def analyze_mffu_batch(
    result: Mapping[str, Any],
    *,
    variants: Iterable[Mapping[str, Any] | Any],
    comparison_pairs: Iterable[Mapping[str, Any]],
    evaluation_dates: Sequence[str],
    expected_pair_count: int | None = 188,
    economic_result_id: str | None = None,
) -> dict[str, Any]:
    """Analyze the one saved MFFU result without rerunning or mutating it.

    ``variants`` are the 64 handoff matrix rows or frozen plan variant refs.
    ``comparison_pairs`` are parsed rows from its declared CSV. A failed or
    missing summary remains unavailable in every derived comparison.
    """
    axes = _variants(variants)
    dates = [date.fromisoformat(day) for day in evaluation_dates]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("evaluation_dates must be unique and ordered")
    period = result.get("period") or {}
    start = _timestamp(period.get("start_utc"), "period.start_utc")
    cutoff = _timestamp(period.get("cutoff_utc"), "period.cutoff_utc")
    if start >= cutoff:
        raise ValueError("analysis period has no positive duration")
    rows = _result_rows(result, axes)
    tables = dict(result.get("tables") or {})
    if economic_result_id is not None:
        from alpha_lab.propsim.funded.reporting_accounts import account_row

        tables["trades"] = [account_row(row, result_id=economic_result_id)
                            for row in tables.get("trades") or []]
    pairs = _matched_pairs(rows, comparison_pairs, expected_pair_count)
    trade_index = _funded_trade_index(rows, tables.get("trades"), start, cutoff)
    _matched_trade_diagnostics(rows, pairs, trade_index)
    completed = [row for row in rows.values() if row["status"] == "completed"]
    ranking = sorted(
        ({"variant_id": row["variant_id"], "net_cash_cents": row["net_cash_cents"]}
         for row in completed),
        key=lambda item: (-item["net_cash_cents"], item["variant_id"]),
    )
    return {
        "schema": SCHEMA, "firm_key": FIRM_KEY,
        "economic_result_id": economic_result_id,
        "reporting_account_version": "scoped_reporting_account_v1",
        "variant_count": len(rows), "completed_count": len(completed),
        "unavailable_count": len(rows) - len(completed),
        "cash_basis": "received_after_split_minus_all_account_purchases_exact_cents",
        "variants": list(rows.values()), "ranking": ranking,
        "matched_pairs": pairs, "interactions": _interactions(rows),
        "waiting_by_variant": _waiting(
            rows, tables.get("payout_events"), start, cutoff, dates
        ),
        "cash_concentration_by_variant": _cash_concentration(
            rows, tables.get("cash_ledger"), start, cutoff
        ),
        "trade_concentration_by_variant": _trade_concentration(
            rows, tables.get("trades"), start, cutoff
        ),
        "entry_order_runner_by_variant": _entry_order_runner(trade_index),
        "decision_context_coverage_by_variant": _decision_coverage(
            rows, result.get("mffu_batch") or {},
        ),
        "waiting_day_basis": (
            "Chicago calendar-date difference; evaluated dates strictly after interval start "
            "and through interval end; elapsed seconds use UTC timestamps"
        ),
    }
