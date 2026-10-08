"""Saved-evidence gamma and geometry, indexed once at reporting publication.

The overnight as-of adapter remains authoritative. Source enrichments are
explicit reporting annotations; trade cohorts never acquire payout economics.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_lenses import VERSION, cents, stamp
from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_provenance import (
    EXECUTED,
    checkpoint_origin,
    receipt_matches,
)
from alpha_lab.propsim.funded.reporting_legs import trade_legs

CATEGORIES = (
    "negative",
    "positive_early",
    "positive_established",
    "positive_age_unknown",
    "neutral_zero",
    "unknown",
)
CATEGORY_LABELS = {
    "negative": "Negative",
    "positive_early": "Positive · early (1–5 reports)",
    "positive_established": "Positive · established (6+ reports)",
    "positive_age_unknown": "Positive · age unknown",
    "neutral_zero": "Exactly zero",
    "unknown": "Unavailable",
}
CLOCKS = (
    "Evening / Asia",
    "London",
    "6:00–7:00 AM",
    "7:00–8:30 AM",
    "8:30–10:00 AM",
    "10:00 AM–noon",
    "Noon–1:00 PM",
    "1:00 PM–day end",
)
OUTCOMES = {
    "no_partial": "No actual first partial",
    "partial_entry_stop": "Partial · remaining stopped near entry",
    "partial_deadline": "Partial · remaining closed at deadline",
    "partial_other": "Partial · other final reason",
}


def category(gamma: Mapping[str, Any]) -> str:
    value = gamma.get("value", gamma.get("total_net_gex"))
    if (
        gamma.get("status") not in {"selected", "available"}
        or value is None
        or not math.isfinite(float(value))
    ):
        return "unknown"
    if value < 0:
        return "negative"
    if value == 0:
        return "neutral_zero"
    age = gamma.get("positive_run_age")
    bound = gamma.get("positive_run_lower_bound", 0) or 0
    if (age is not None and age >= 6) or bound >= 6:
        return "positive_established"
    if (
        age is not None
        and 1 <= age <= 5
        and not gamma.get("positive_run_age_is_lower_bound", False)
    ):
        return "positive_early"
    return "positive_age_unknown"


def clock_bin(value: Any) -> str:
    at = stamp(value).tz_convert("America/Chicago")
    minute = at.hour * 60 + at.minute
    if minute >= 17 * 60 or minute < 60:
        return CLOCKS[0]
    for end, label in (
        (360, CLOCKS[1]),
        (420, CLOCKS[2]),
        (510, CLOCKS[3]),
        (600, CLOCKS[4]),
        (720, CLOCKS[5]),
        (780, CLOCKS[6]),
        (1020, CLOCKS[7]),
    ):
        if minute < end:
            return label
    raise ValueError("clock display partition incomplete")


def trade_key(configuration: str, population: str, row: Mapping[str, Any]) -> str:
    ref = row.get("trade_ref") or row.get("strategy_trade_id") or row.get("trade_id")
    if not ref:
        raise ValueError("gamma view requires an exact saved trade identity")
    return f"{configuration}|{population}|{row.get('account_number', '')}|{ref}"


def first_checkpoint_time(row: Mapping[str, Any]) -> Any:
    """Actual partial fill or whole-at-target exit, including subsecond order."""
    if row.get("scale_out_quantity", 0):
        if row.get("scale_out_ns") is not None:
            return pd.Timestamp(int(row["scale_out_ns"]), tz="UTC").isoformat()
        return row.get("scale_out_utc") or row.get("scale_out_ts_utc")
    return row.get("exit_utc") if row.get("exit_kind") == "target" else None


def _decision_stamp(decision: Mapping[str, Any]) -> Any:
    return (decision.get("context") or {}).get("decision_ts_utc")


def _compact_receipt(decision: Mapping[str, Any] | None) -> dict | None:
    return (
        {
            k: decision.get(k)
            for k in (
                "event",
                "configuration",
                "stream",
                "firm_key",
                "account_number",
                "setup_id",
                "trade_id",
                "policy",
                "action",
                "reasons",
                "distance_ticks",
                "fallback_reason",
                "context",
            )
        }
        if decision
        else None
    )


def build_gamma(
    study: Any, index: Any, vendor_metrics: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """One bounded publication pass. Never called on a filter or detail click."""
    decisions = (study.result.get("mffu_batch") or {}).get("decision_context") or []
    saved_plan = (study.result.get("mffu_batch") or {}).get("plan") or {}
    sections = {
        v["name"]: json.loads(v["effective_section_json"]) for v in saved_plan.get("variants", [])
    }
    admissions: dict[tuple, list] = defaultdict(list)
    locks: dict[tuple, list] = defaultdict(list)
    targets: dict[tuple, list] = defaultdict(list)
    for decision in decisions:
        config, stream = decision.get("configuration"), decision.get("stream")
        at = _decision_stamp(decision)
        if decision.get("event") == "entry_admission" and decision.get("action") == "admit" and at:
            admissions[(config, stream, stamp(at).value)].append(decision)
        elif decision.get("event") == "parent_lock_distance" and at:
            locks[(config, stream, decision.get("setup_id"))].append(decision)
        elif decision.get("event") == "first_target":
            targets[(config, stream, decision.get("trade_id"))].append(decision)
    annotations = {
        (r.get("configuration"), r.get("stream"), r.get("historical_trade_id")): r
        for r in (study.result.get("mffu_batch") or {}).get("reuse_context_annotations") or []
    }
    maps: dict[str, dict] = {}
    timeline = []
    snapshots: dict[int, dict] = {}

    def snapshot(at):
        ns = stamp(at).value
        if ns not in snapshots:
            full = index.snapshot(stamp(at).to_pydatetime(warn=False))
            levels = full.get("levels") or {}
            set_id = levels.get("level_set_id")
            if set_id:
                maps[set_id] = levels
            snapshots[ns] = {
                "decision_time_utc": stamp(at).isoformat(),
                "policy_id": full["policy_id"],
                "gamma": full.get("gamma") or {},
                "level_set_id": set_id,
                "level_status": levels.get("status"),
                "historical_publication_verified": False,
            }
            metric = (vendor_metrics or {}).get((full.get("gamma") or {}).get("report_date"))
            if metric:
                selected = snapshots[ns]["gamma"]
                if metric.get("source_sha256") != selected.get("source_sha256") or metric.get(
                    "source_json_pointer"
                ) != selected.get("source_json_pointer"):
                    raise ValueError("vendor percentile does not bind the selected gamma source")
                selected["gex_percentile_1y"] = metric.get("gex_percentile_1y")
                selected["percentile_role"] = (
                    "provided vendor percentile; bound-source reporting annotation"
                )
        return snapshots[ns]

    # The immutable source index's independent level clock is reused verbatim.
    # This reporting read does not alter the adapter or any execution decision.
    for eligible in index._level_times:
        record = snapshot(eligible)
        if record.get("level_set_id") and (
            not timeline or timeline[-1]["level_set_id"] != record["level_set_id"]
        ):
            timeline.append(
                {
                    "eligible_from_utc": stamp(eligible).isoformat(),
                    "level_set_id": record["level_set_id"],
                }
            )
    rows = []
    geometries = []
    for population, table in (("funded", "trades"), ("strategy", "strategy_trades")):
        source_rows = (study.result.get("tables") or {}).get(table) or []
        membership = Counter(
            (
                t.get("configuration"),
                t.get("trade_ref") or t.get("strategy_trade_id") or t.get("trade_id"),
            )
            for t in source_rows
        )
        seen = set()
        for source_trade in source_rows:
            trade = dict(source_trade)
            if population == "strategy":
                trade["exit_kind"] = trade.get("resolution")
                if trade.get("risk_ticks") is not None:
                    trade["initial_risk_cents"] = (
                        int(trade["risk_ticks"])
                        * int(trade["quantity"])
                        * int(trade["tick_value_cents"])
                    )
            config = trade.get("configuration")
            day = str(trade.get("entry_trading_day") or trade.get("trading_day"))
            if (
                config not in study.configurations
                or day not in study.calendar
                or trade.get("is_warmup")
            ):
                continue
            unique = trade_key(config, population, trade)
            if unique in seen:
                raise ValueError("duplicate actual trade in gamma population")
            seen.add(unique)
            entry_time = trade["entry_utc"]
            ref = trade.get("trade_ref") or trade.get("strategy_trade_id") or trade.get("trade_id")
            candidates = admissions.get((config, population, stamp(entry_time).value), [])
            distinct = {d.get("setup_id"): d for d in candidates}
            admission = next(iter(distinct.values())) if len(distinct) == 1 else None
            setup_id = admission.get("setup_id") if admission else None
            lock_candidates = (
                [
                    d
                    for d in locks.get((config, population, setup_id), [])
                    if stamp(_decision_stamp(d)) <= stamp(entry_time)
                ]
                if setup_id
                else []
            )
            lock = (
                max(lock_candidates, key=lambda d: stamp(_decision_stamp(d)))
                if lock_candidates
                else None
            )
            target_candidates = targets.get((config, population, ref), [])
            target_receipt = target_candidates[0] if len(target_candidates) == 1 else None
            original = trade.get("entry_context") or {}
            annotation = annotations.get((config, population, ref))
            entry_snapshot = snapshot(entry_time)
            context_role = (
                "executed_saved_context"
                if original.get("policy_id") == entry_snapshot["policy_id"]
                else "posthoc_reporting_annotation"
            )
            if annotation:
                context_role = "posthoc_added_after_execution"
            # Verify executed values before enriching missing vendor values/maps.
            executed_gamma = original.get("gamma") or {}
            if executed_gamma and context_role == "executed_saved_context":
                for field in ("report_date", "sign", "positive_run_age", "value"):
                    if executed_gamma.get(field) != entry_snapshot["gamma"].get(field):
                        raise ValueError(
                            "executed gamma differs from bound as-of source: "
                            f"{config}/{ref}/{field}"
                        )
            net = cents(trade, "net_pnl")
            risk = cents(trade, "initial_risk")
            had_partial = bool(trade.get("scale_out_quantity"))
            outcome = (
                "partial_deadline"
                if had_partial and trade.get("exit_kind") == "scheduled_close"
                else "partial_entry_stop"
                if had_partial and trade.get("exit_kind") == "breakeven_stop"
                else "partial_other"
                if had_partial
                else "no_partial"
            )
            remaining = trade_legs(trade).remaining_cents if had_partial else None
            base = {
                "trade_key": unique,
                "trade_ref": ref,
                "configuration_id": config,
                "population": population,
                "firm": trade.get("firm_key"),
                "account_number": trade.get("account_number"),
                "seq": trade.get("seq"),
                "economic_result_id": study.result_id,
                "reporting_version": VERSION,
                "trading_date": day,
                "entry_utc": entry_time,
                "exit_utc": trade.get("exit_utc"),
                "entry_ticks": trade.get("entry_ticks"),
                "quantity": trade.get("quantity"),
                "net_pnl_cents": net,
                "net_initial_risk_units": net / risk
                if net is not None and risk and risk > 0
                else None,
                "unavailable_risk_reason": None
                if risk and risk > 0
                else "original positive initial risk unavailable",
                "outcome": outcome,
                "remaining_leg_net_cents": remaining,
                "had_partial": had_partial,
                "entry_policy_receipt": _compact_receipt(admission),
                "lock_policy_receipt": _compact_receipt(lock),
                "first_target_policy_receipt": _compact_receipt(target_receipt),
                "setup_id": setup_id,
                "context_role": context_role,
                "entry_snapshot": entry_snapshot,
                "lock_snapshot": snapshot(_decision_stamp(lock)) if lock else None,
                "vendor_percentile": entry_snapshot["gamma"].get("gex_percentile_1y"),
            }
            checkpoint_time = first_checkpoint_time(trade)
            base["first_checkpoint_utc"] = checkpoint_time
            base["first_checkpoint_status"] = (
                "observed_exact_time"
                if checkpoint_time
                else "reached_time_unavailable"
                if had_partial or trade.get("exit_kind") == "target"
                else "not_reached"
            )
            base["first_checkpoint_time_reason"] = (
                trade.get("scale_out_timestamp_status") or "exact first checkpoint time not saved"
                if had_partial and not checkpoint_time
                else None
            )
            base["first_checkpoint_branch"] = (
                "actual partial"
                if had_partial
                else "whole at first target"
                if checkpoint_time
                else "not reached"
            )
            origin = checkpoint_origin(
                trade=trade,
                population=population,
                candidates=target_candidates,
                checkpoint=checkpoint_time,
                section=sections.get(config, {}),
                index=index,
                unique_membership=membership[(config, ref)] == 1,
            )
            basis_time = (
                _decision_stamp(target_receipt)
                if population == "strategy" and origin["status"] == "verified"
                else checkpoint_time
            )
            base["first_checkpoint_snapshot"] = (
                snapshot(basis_time)
                if checkpoint_time and origin["status"] in {"verified", "annotation"}
                else None
            )
            base["first_checkpoint_origin"] = origin
            base["first_checkpoint_context_role"] = origin["role"]
            base["entry_context_role"] = context_role
            base["entry_context_usage"] = (
                "recorded_at_entry; use depends on the saved admission/quantity policy"
                if context_role in EXECUTED
                else "reporting_only"
            )
            base["lock_context_role"] = (
                "executed_saved_decision_receipt"
                if lock
                and receipt_matches(
                    lock.get("context") or {},
                    index.snapshot(stamp(_decision_stamp(lock)).to_pydatetime(warn=False)),
                )
                else "unresolved_checkpoint_evidence"
                if lock
                else "unavailable"
            )
            if origin["status"] != "verified":
                base["first_target_policy_receipt"] = None
            rows.append(base)
            if lock:
                lock_snapshot = base["lock_snapshot"]
                level_set = maps.get(lock_snapshot.get("level_set_id"), {})
                items = level_set.get("items") or {}
                upper, lower = (items.get(name, {}).get("price") for name in ("1D Max", "1D Min"))
                move = (
                    (upper - lower) / 2
                    if upper is not None and lower is not None and upper > lower
                    else None
                )
                policy = lock.get("policy")
                fraction = {
                    "implied_move_005_v1": 0.05,
                    "implied_move_0075_v1": 0.075,
                    "implied_move_010_v1": 0.10,
                }.get(policy)
                geometries.append(
                    {
                        "trade_key": unique,
                        "configuration_id": config,
                        "population": population,
                        "economic_result_id": study.result_id,
                        "reporting_version": VERSION,
                        "entry_utc": entry_time,
                        "lock_utc": _decision_stamp(lock),
                        "setup_id": setup_id,
                        "policy": policy,
                        "range_max_points": upper,
                        "range_min_points": lower,
                        "implied_half_range_points": move,
                        "configured_fraction": fraction,
                        "frozen_limit_ticks": lock.get("distance_ticks"),
                        "frozen_limit_points": lock["distance_ticks"] * 0.25
                        if lock.get("distance_ticks") is not None
                        else None,
                        "fallback_reason": lock.get("fallback_reason"),
                        "level_set_id": lock_snapshot.get("level_set_id"),
                        "qualifying_separation_points": None,
                        "separation_status": "exact opposing-pattern separation not saved",
                    }
                )
    return {
        "reporting_version": VERSION,
        "economic_result_id": study.result_id,
        "source_bundle_sha256": index.bundle_sha256,
        "source_table_sha256": dict(index.table_sha256),
        "trades": rows,
        "geometry": geometries,
        "level_maps": maps,
        "level_timeline": timeline,
        "intraday_note": "15,459 observations; level prices only; eight overlapping study "
        "dates. No intraday economic input. Qscore unavailable.",
        "units": {
            "price": "full index points",
            "gamma": "vendor supplied GEX units",
            "pnl": "cents",
            "risk": "original full-position structural risk",
        },
    }


def selected_rows(
    report: Mapping[str, Any],
    configuration: str,
    population: str = "funded",
    basis: str = "entry",
    *,
    categories: Sequence[str] = (),
    clocks: Sequence[str] = (),
    start: str | None = None,
    end: str | None = None,
    cursor: Any = None,
) -> list[dict]:
    rows = []
    for source in report.get("trades", []):
        if source["configuration_id"] != configuration or source["population"] != population:
            continue
        if start and source["trading_date"] < start or end and source["trading_date"] > end:
            continue
        if cursor is not None and stamp(source["entry_utc"]) > stamp(cursor):
            continue
        if basis not in {"entry", "first_1R_checkpoint", "parent_lock"}:
            raise ValueError("unsupported checkpoint basis")
        event = (
            source["entry_utc"]
            if basis == "entry"
            else (source.get("lock_policy_receipt") or {}).get("context", {}).get("decision_ts_utc")
            if basis == "parent_lock"
            else source.get("first_checkpoint_utc")
        )
        if not event or cursor is not None and stamp(event) > stamp(cursor):
            continue
        snapshot = (
            source["entry_snapshot"]
            if basis == "entry"
            else source.get("lock_snapshot")
            if basis == "parent_lock"
            else source.get("first_checkpoint_snapshot")
        )
        group, window = category((snapshot or {}).get("gamma", {})), clock_bin(event)
        if categories and group not in categories or clocks and window not in clocks:
            continue
        row = dict(
            source,
            basis=basis,
            checkpoint_utc=event,
            gamma_category=group,
            chicago_clock=window,
            selected_snapshot=snapshot,
            context_role=(
                source.get("entry_context_role", source.get("context_role"))
                if basis == "entry"
                else source.get("lock_context_role", "unavailable")
                if basis == "parent_lock"
                else source.get("first_checkpoint_context_role", "unavailable")
            ),
        )
        target_at = source.get("first_checkpoint_utc")
        if cursor is not None and (not target_at or stamp(target_at) > stamp(cursor)):
            for key in tuple(row):
                if key.startswith("first_checkpoint") or key == "first_target_policy_receipt":
                    row[key] = None
            row["had_partial"] = None
        if (
            cursor is not None
            and source.get("exit_utc")
            and stamp(source["exit_utc"]) > stamp(cursor)
        ):
            for key in (
                "net_pnl_cents",
                "net_initial_risk_units",
                "outcome",
                "remaining_leg_net_cents",
            ):
                row[key] = None
            row["exit_utc"] = None
        rows.append(row)
    return rows


def grouped(rows: Sequence[Mapping[str, Any]], *, by_clock: bool = False) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for row in rows:
        groups[(row["gamma_category"], row["chicago_clock"] if by_clock else "All windows")].append(
            row
        )
    result = []
    for (group, clock), trades in sorted(
        groups.items(),
        key=lambda item: (
            CATEGORIES.index(item[0][0]),
            CLOCKS.index(item[0][1]) if by_clock else 0,
        ),
    ):
        known = [t for t in trades if t.get("net_pnl_cents") is not None]
        rs = [
            t["net_initial_risk_units"]
            for t in known
            if t.get("net_initial_risk_units") is not None
        ]
        result.append(
            {
                "gamma_category": group,
                "chicago_clock": clock,
                "actual_trades": len(trades),
                "distinct_trading_dates": len({t["trading_date"] for t in trades}),
                "wins": sum(t["net_pnl_cents"] > 0 for t in known),
                "losses": sum(t["net_pnl_cents"] < 0 for t in known),
                "zero_outcomes": sum(t["net_pnl_cents"] == 0 for t in known),
                "outcome_not_yet_known": len(trades) - len(known),
                "net_pnl_cents": sum(t["net_pnl_cents"] for t in known),
                "mean_after_cost_initial_risk_units": sum(rs) / len(rs) if rs else None,
                "risk_count": len(rs),
                "unavailable_risk_count": len(known) - len(rs),
            }
        )
    return result


def outcome_partition(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    return [
        {
            "outcome": label,
            "trades": sum(r.get("outcome") == key for r in rows),
            "whole_trade_net_cents": sum(
                r["net_pnl_cents"]
                for r in rows
                if r.get("outcome") == key and r.get("net_pnl_cents") is not None
            ),
            "remaining_leg_net_cents": sum(
                r["remaining_leg_net_cents"]
                for r in rows
                if r.get("outcome") == key and r.get("remaining_leg_net_cents") is not None
            ),
            "remaining_leg_count": sum(
                r.get("outcome") == key and r.get("remaining_leg_net_cents") is not None
                for r in rows
            ),
        }
        for key, label in OUTCOMES.items()
    ]


def coverage(rows: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "actual_trades": len(rows),
        "executed": sum(r.get("context_role") in EXECUTED for r in rows),
        "posthoc": sum(
            "annotation" in str(r.get("context_role", ""))
            or r.get("context_role") == "posthoc_added_after_execution"
            for r in rows
        ),
        "unavailable": sum(
            r.get("context_role") not in EXECUTED
            and "annotation" not in str(r.get("context_role", ""))
            and r.get("context_role") != "posthoc_added_after_execution"
            for r in rows
        ),
        "origin_counts": dict(Counter(r.get("context_role", "unavailable") for r in rows)),
        "unknown_reasons": dict(
            Counter(
                (r.get("selected_snapshot") or {}).get("gamma", {}).get("status", "no snapshot")
                for r in rows
                if r.get("gamma_category") == "unknown"
            )
        ),
        "report_dates": sorted(
            {
                (r.get("selected_snapshot") or {}).get("gamma", {}).get("report_date")
                for r in rows
                if (r.get("selected_snapshot") or {}).get("gamma", {}).get("report_date")
            }
        ),
    }


def checkpoint_cards(trade: Mapping[str, Any], *, cursor: Any = None) -> list[dict]:
    cards = []
    for label, at, snapshot, receipt in (
        (
            "Supporting-pattern lock",
            (trade.get("lock_policy_receipt") or {}).get("context", {}).get("decision_ts_utc"),
            trade.get("lock_snapshot"),
            trade.get("lock_policy_receipt"),
        ),
        (
            "Entry",
            trade["entry_utc"],
            trade.get("entry_snapshot"),
            trade.get("entry_policy_receipt"),
        ),
        (
            "First 1R checkpoint",
            trade.get("first_checkpoint_utc"),
            trade.get("first_checkpoint_snapshot"),
            trade.get("first_target_policy_receipt"),
        ),
    ):
        if at and (cursor is None or stamp(at) <= stamp(cursor)):
            cards.append(
                {
                    "checkpoint": label,
                    "at_utc": at,
                    "snapshot": snapshot,
                    "recorded_policy": receipt,
                    "context_role": (
                        trade.get("first_checkpoint_context_role", "unavailable")
                        if label == "First 1R checkpoint"
                        else trade.get("lock_context_role", "unavailable")
                        if label == "Supporting-pattern lock"
                        else trade.get(
                            "entry_context_role", trade.get("context_role", "unavailable")
                        )
                    ),
                    "context_usage": (
                        (trade.get("first_checkpoint_origin") or {}).get("context_usage")
                        if label == "First 1R checkpoint"
                        else trade.get("entry_context_usage")
                        if label == "Entry"
                        else "recorded supporting-pattern distance decision"
                    ),
                    "branch": trade.get("first_checkpoint_branch")
                    if label == "First 1R checkpoint"
                    else None,
                    "quantity": trade.get("quantity") if label == "Entry" else None,
                }
            )
    return cards


def level_segments(
    report: Mapping[str, Any],
    *,
    start: Any,
    end: Any,
    cursor: Any = None,
    names: Sequence[str] = (),
) -> list[dict]:
    """Each historical map begins at eligibility; no line paints older candles."""
    stop = min(stamp(end), stamp(cursor)) if cursor is not None else stamp(end)
    begin = stamp(start)
    timeline = report.get("level_timeline") or []
    result = []
    for i, event in enumerate(timeline):
        eligible = stamp(event["eligible_from_utc"])
        next_time = stamp(timeline[i + 1]["eligible_from_utc"]) if i + 1 < len(timeline) else stop
        left, right = max(begin, eligible), min(stop, next_time)
        if left >= right:
            continue
        level_map = report["level_maps"][event["level_set_id"]]
        # Staleness is the saved adapter's calendar-age rule, not an inferred refresh.
        report_day = pd.Timestamp(level_map["report_date"])
        from alpha_lab.agents.data_infra.ifvg.menthorq_asof import MAX_AGE_CALENDAR_DAYS

        stale_at = (
            (report_day + pd.Timedelta(days=MAX_AGE_CALENDAR_DAYS + 1))
            .tz_localize("America/Chicago")
            .tz_convert("UTC")
        )
        right = min(right, stale_at)
        if left >= right:
            continue
        prices: dict[float, list] = defaultdict(list)
        for name, item in (level_map.get("items") or {}).items():
            if (not names or name in names) and item.get("price") is not None:
                prices[item["price"]].append({"name": name, "signed_vendor_gex": item.get("gex")})
        for price, aliases in prices.items():
            result.append(
                {
                    "start_utc": left.isoformat(),
                    "end_utc": right.isoformat(),
                    "eligible_from_utc": eligible.isoformat(),
                    "price_points": price,
                    "levels": aliases,
                    "level_set_id": event["level_set_id"],
                    "report_date": level_map.get("report_date"),
                }
            )
    return result


def export_rows(
    rows: Sequence[dict], *, report: Mapping[str, Any], basis: str, filters: Mapping[str, Any]
) -> str:
    return json.dumps(
        {
            "economic_result_id": report["economic_result_id"],
            "reporting_version": VERSION,
            "basis": basis,
            "filters": filters,
            "units": report["units"],
            "source_bundle_sha256": report["source_bundle_sha256"],
            "rows": rows,
            "groups": grouped(rows),
            "clock_groups": grouped(rows, by_clock=True),
            "outcomes": outcome_partition(rows),
        },
        allow_nan=False,
        indent=2,
    )
