"""The ONE immutable configuration-comparison result (screen and review folder read it).

``build_comparison_result`` turns the per-configuration worker outputs into
plain JSON tables and one summary per configuration-and-firm pair.
``validate_comparison`` re-derives every headline figure independently from the
pair ledgers and checks the single-account rules. Neither the screen nor the
export recomputes money.

Pairs are separate comparisons: nothing here adds configurations or firms
together into a portfolio total.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from alpha_lab.propsim.funded.campaign import TradingDay, display_time, machine_time, month_keys
from alpha_lab.propsim.funded.clock import chicago_date
from alpha_lab.propsim.funded.instance import STATUS_LABELS
from alpha_lab.propsim.funded.profiles import FundedFirmProfile

__all__ = [
    "COMPARISON_SCHEMA",
    "COMPARISON_MODE",
    "COMPARISON_TABLES",
    "DAILY_CLOSE_WORDING_CORRECTION_ID",
    "STOP_DIFFERENCE_CORRECTION_ID",
    "apply_reporting_corrections",
    "build_comparison_result",
    "stop_fill_difference_cents",
    "validate_comparison",
]

COMPARISON_SCHEMA = "funded_comparison_result_v1"
COMPARISON_MODE = "single_account_configuration_comparison"
COMPARISON_TABLES = (
    "pair_results", "configurations", "account_journeys", "cash_ledger", "payout_events",
    "monthly_results", "trades", "account_events", "rule_boundary_evidence",
    "cash_over_time", "execution_evidence", "strategy_metrics",
)
HOUR = 3_600_000_000_000
EXIT_RULE_TEXT = {
    "fixed_target_v1": "The whole position exits at the stop, the target or the daily close",
    "scale_out_half_breakeven_hold_to_close_v1": (
        "Half exits at the target (1R); the stop of the rest moves to the entry price and it "
        "is held to that stop or the daily close"),
}


def _usd(cents: int | None) -> float | None:
    return None if cents is None else round(cents / 100.0, 2)


def _hours(ns: int) -> float:
    return round(ns / HOUR, 2)


def _times(row: dict[str, Any], *names: str) -> None:
    for name in names:
        if name not in row:
            continue
        ns = row.pop(name)
        base = name[:-3]
        row[f"{base}_utc"] = machine_time(ns)
        row[f"{base}_chicago"] = display_time(ns)


def _money(row: dict[str, Any]) -> None:
    for key in [k for k in row if k.endswith("_cents")]:
        row[key[:-6] + "_usd"] = _usd(row.pop(key))


def _ready_split(ledger: dict, days: tuple[TradingDay, ...], cutoff_ns: int) -> tuple[int, int]:
    closed = sorted((d.deadline_ns, d.reopen_ns) for d in days)

    def closed_overlap(a: int, b: int) -> int:
        return sum(max(0, min(b, hi) - max(a, lo)) for lo, hi in closed if hi > a and lo < b)

    events = defaultdict(list)
    for row in ledger["account_events"]:
        if row.get("status_after") is not None:
            events[row["account_id"]].append((row["ts_ns"], row["status_after"]))
    shut = open_ = 0
    for rows in events.values():
        rows.sort()
        for (ts, status), nxt in zip(rows, [*rows[1:], (cutoff_ns, None)], strict=True):
            if status != "ready":
                continue
            end = min(nxt[0], cutoff_ns)
            if end <= ts:
                continue
            closed_part = closed_overlap(ts, end)
            shut += closed_part
            open_ += end - ts - closed_part
    return shut, open_


STOP_EXIT_KINDS = ("stop", "breakeven_stop")
#: reporting correction of results saved before the stop-difference fix
STOP_DIFFERENCE_CORRECTION_ID = "stop_difference_summary_final_quantity_v1"
DAILY_CLOSE_WORDING_CORRECTION_ID = "daily_close_wording_v1"


def stop_fill_difference_cents(trade: dict[str, Any], tick_value_cents: int) -> int:
    """Money a stop or break-even stop lost by filling worse than its stop price.

    ``(final stop - fill) x direction``, floored at zero, times the contracts that
    actually closed at that stop (after a half exit only the remaining half) and
    the tick value. It is a descriptive measure already inside the trade's profit;
    it is never subtracted again. Rows saved before the half exit existed carry
    no ``final_*`` fields and closed their whole quantity at their only stop.
    """

    if trade["exit_kind"] not in STOP_EXIT_KINDS:
        return 0
    stop = trade.get("final_stop_ticks")
    stop = trade["stop_ticks"] if stop is None else stop
    quantity = trade.get("final_exit_quantity")
    quantity = trade["quantity"] if quantity is None else quantity
    sign = 1 if trade["direction"] == "long" else -1
    return max(0, sign * (int(stop) - int(trade["exit_ticks"]))) * int(quantity) * int(
        tick_value_cents)


def _stop_difference_fields(trades: list[dict[str, Any]], tick_value_cents: int
                            ) -> dict[str, int]:
    differences = [stop_fill_difference_cents(t, tick_value_cents) for t in trades]
    return {"stop_exits_filled_worse_than_stop": sum(1 for d in differences if d),
            "stop_slippage_cents": sum(differences)}


def _pair_summary(pair: dict, firm: FundedFirmProfile, configuration: dict,
                  days: tuple[TradingDay, ...], cutoff_ns: int, tick_value_cents: int
                  ) -> dict[str, Any]:
    ledger = pair["ledger"]
    accounts = ledger["accounts"]
    requests = [r for r in ledger["payout_events"] if r["event"] == "requested"]
    receipts = [r for r in ledger["payout_events"] if r["event"] == "received"]
    received_ids = {r["request_id"] for r in receipts}
    pending = [r for r in requests if r["request_id"] not in received_ids]
    failed = [a for a in accounts if a["status"] == "failed"]
    durations: dict[str, int] = defaultdict(int)
    refused: dict[str, int] = defaultdict(int)
    for account in accounts:
        for status, ns in account["status_durations_ns"].items():
            durations[status] += ns
        for reason, count in account["blocked_entries"].items():
            refused[reason] += count
    live = accounts[-1]
    shut, open_ = _ready_split(ledger, days, cutoff_ns)
    trades = ledger["trades"]
    first_receipt = min((r["received_ns"] for r in receipts), default=None)
    return {
        "pair_id": pair["pair_id"],
        "configuration": configuration["configuration"],
        "configuration_label": configuration["display_name"],
        "firm_key": firm.firm_key,
        "firm": firm.firm_name,
        "status": "Completed",
        "net_cash_earned_cents": ledger["receipts"] - ledger["costs"],
        "payouts_received_cents": ledger["receipts"],
        "payouts_received_count": len(receipts),
        "largest_single_payout_cents": max((r["trader_cents"] for r in receipts), default=0),
        "account_costs_cents": ledger["costs"],
        "accounts_purchased": len(accounts),
        "accounts_lost_before_first_payout": sum(1 for a in failed
                                                 if a["payouts_received"] == 0),
        "accounts_lost_after_a_payout": sum(1 for a in failed if a["payouts_received"] > 0),
        "first_payout_ns": first_receipt,
        "max_unrecovered_spending_cents": ledger["max_shortfall"],
        "max_unrecovered_spending_ns": ledger["max_shortfall_ns"],
        "pending_gross_at_cutoff_cents": sum(r["gross_cents"] for r in pending),
        "pending_after_split_at_cutoff_cents": sum(r["trader_cents"] for r in pending),
        "payouts_processing_at_cutoff": len(pending),
        "secured_not_requested_at_cutoff": int(live["status"] == "secured"
                                               and (live["pending_payout"] or {}).get("state")
                                               != "release_next_day"),
        "profit_inside_live_account_cents": live["balance"] if live["status"] != "failed" else 0,
        "live_account_status_at_end": STATUS_LABELS[live["status"]],
        "trades_taken": len(trades),
        "entries_refused_payout_protection": refused.get("account_payout_protection", 0),
        "entries_refused_payout_processing": refused.get("account_payout_processing", 0),
        "account_liquidations_that_ended_a_strategy_trade": pair["forced_flat"],
        "trades_not_in_no_account_replay": pair["trades_not_in_reference"],
        "stop_exits": sum(1 for t in trades if t["exit_kind"] in STOP_EXIT_KINDS),
        **_stop_difference_fields(trades, tick_value_cents),
        "trades_with_approximated_minutes": sum(1 for t in trades
                                                if t["minutes_approximated"] > 0),
        "position_minutes_on_prints": sum(t["minutes_on_prints"] for t in trades),
        "position_minutes_approximated": sum(t["minutes_approximated"] for t in trades),
        "entry_candidates_also_blocked_by_strategy":
            pair.get("entry_candidates_also_blocked_by_strategy", 0),
        "hours_trading": _hours(durations.get("in_trade", 0)),
        "hours_payout_protection": _hours(durations.get("secured", 0)),
        "hours_payout_processing": _hours(durations.get("processing", 0)),
        "hours_ready_market_closed_or_locked": _hours(shut),
        "hours_ready_no_strategy_signal": _hours(open_),
    }


def build_comparison_result(*, context: dict[str, Any], outputs: list[dict[str, Any]],
                            failures: list[dict[str, Any]], profiles: tuple[FundedFirmProfile, ...],
                            trading_days: tuple[TradingDay, ...], start_ns: int, cutoff_ns: int,
                            settings: dict[str, Any]) -> dict[str, Any]:
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in COMPARISON_TABLES}
    summaries: dict[str, dict[str, Any]] = {}
    months = month_keys(start_ns, cutoff_ns)
    by_key = {p.firm_key: p for p in profiles}

    for output in sorted(outputs, key=lambda o: o["configuration"]):
        sizing = output.get("sizing") or {}
        exit_rule = EXIT_RULE_TEXT.get(output.get("exit_policy", "fixed_target_v1"), "")
        settings_rows = list(output["settings_plain"])
        if exit_rule and not any(r["setting"] == "Exit rule" for r in settings_rows):
            settings_rows.append({"setting": "Exit rule", "value": exit_rule})
        if sizing:
            settings_rows.append({"setting": "Position size", "value": (
                f"{sizing['quantity']} x {sizing['instrument_label']} per trade, "
                f"${sizing['cost_per_contract_mills'] / 1000:.3f} per contract per fill")})
        tables["configurations"].append({
            "configuration": output["configuration"],
            "configuration_label": output["display_name"],
            "settings": settings_rows,
            "axes": output["axes"],
        })
        if "strategy_trades_no_account" in output:
            tables["strategy_metrics"].append(strategy_metrics(output))
        tables["execution_evidence"].append({
            "configuration": output["configuration"],
            "no_account_replay_equals_saved_study": output["reference"]["equivalent"],
            "saved_study_trades": output["reference"]["saved_study_trades"],
            "no_account_replay_trades": output["reference"]["replayed_trades"],
            "resumed_run_identical": all(v["identical"] for v in output["resumed"].values()),
            "position_minutes_checked": output["prints"]["minutes_checked"],
            "position_minutes_rebuilt_exactly_from_prints":
                output["prints"]["minutes_rebuilt_exactly"],
            "missing_print_days": len(output["prints"]["missing_utc_days"]),
        })
        for firm_key, pair in output["pairs"].items():
            profile = by_key[firm_key]
            ledger = pair["ledger"]
            tick_value = output.get("sizing", {}).get("tick_value_cents",
                                                      settings.get("tick_value_cents"))
            summary = _pair_summary(pair, profile, output, trading_days, cutoff_ns,
                                    tick_value)
            summaries[pair["pair_id"]] = summary
            base = {"pair_id": pair["pair_id"], "configuration": output["configuration"],
                    "firm_key": firm_key, "firm": profile.firm_name}
            for account in ledger["accounts"]:
                journey = {
                    **base, "account_number": account["number"],
                    "created_ns": account["created_ns"], "failed_ns": account["failed_ns"],
                    "replaces_account_number": (account["number"] - 1
                                                if account["replaces"] else None),
                    "status_at_end": STATUS_LABELS[account["status"]],
                    "failure_reason": account["failure_reason"],
                    "lost_after_payout": account["status"] == "failed"
                    and account["payouts_received"] > 0,
                    "trades": account["trades"], "payouts_received": account["payouts_received"],
                    "received_cents": account["received_cents"],
                    "largest_payout_cents": account["largest_receipt_cents"],
                    "gross_requested_cents": account["gross_requested_cents"],
                    "final_balance_cents": account["balance"],
                    "final_floor_cents": account["floor"],
                    "entries_refused_payout_protection":
                        account["blocked_entries"].get("account_payout_protection", 0),
                    "entries_refused_payout_processing":
                        account["blocked_entries"].get("account_payout_processing", 0),
                    "hours_trading": _hours(account["status_durations_ns"].get("in_trade", 0)),
                    "hours_payout_protection":
                        _hours(account["status_durations_ns"].get("secured", 0)),
                    "hours_payout_processing":
                        _hours(account["status_durations_ns"].get("processing", 0)),
                }
                _times(journey, "created_ns", "failed_ns")
                _money(journey)
                tables["account_journeys"].append(journey)
            number = {a["account_id"]: a["number"] for a in ledger["accounts"]}
            for source_name, target, stamps in (
                ("cash_ledger", "cash_ledger", ("ts_ns",)),
                ("payout_events", "payout_events",
                 ("ts_ns", "secured_ns", "requested_ns", "due_ns", "received_ns")),
                ("account_events", "account_events", ("ts_ns", "due_ns")),
                ("trades", "trades", ("ts_ns", "entry_ns", "exit_ns", "min_equity_ns")),
                ("boundary_evidence", "rule_boundary_evidence", ("ts_ns",)),
            ):
                for raw in ledger[source_name]:
                    row = {**base, **{k: v for k, v in raw.items() if k != "pair_id"}}
                    if "account_id" in row:
                        row["account_number"] = number.get(row.pop("account_id"))
                    _times(row, *stamps)
                    _money(row)
                    tables[target].append(row)
            requested_ids = {r["request_id"] for r in ledger["payout_events"]
                             if r["event"] == "received"}
            for r in ledger["payout_events"]:
                if r["event"] == "requested" and r["request_id"] not in requested_ids:
                    tables["payout_events"].append({
                        **base, "account_number": number[r["account_id"]],
                        "event": "processing_not_received_at_cutoff",
                        "request_id": r["request_id"], "gross_usd": _usd(r["gross_cents"]),
                        "trader_usd": _usd(r["trader_cents"]),
                        "due_utc": machine_time(r["due_ns"]),
                        "due_chicago": display_time(r["due_ns"]),
                        "ts_utc": machine_time(cutoff_ns), "ts_chicago": display_time(cutoff_ns),
                    })
            if summary["secured_not_requested_at_cutoff"]:
                tables["payout_events"].append({
                    **base, "account_number": ledger["accounts"][-1]["number"],
                    "event": "secured_not_requested_at_cutoff",
                    "ts_utc": machine_time(cutoff_ns), "ts_chicago": display_time(cutoff_ns),
                    "realized_balance_usd": _usd(ledger["accounts"][-1]["balance"]),
                })
            buckets = {m: {"received": 0, "spent": 0, "payouts": 0, "bought": 0}
                       for m in months}
            for r in ledger["cash_ledger"]:
                day = chicago_date(r["ts_ns"])
                bucket = buckets[f"{day.year:04d}-{day.month:02d}"]
                if r["kind"] == "payout_received":
                    bucket["received"] += r["amount_cents"]
                    bucket["payouts"] += 1
                else:
                    bucket["spent"] += r["amount_cents"]
                    bucket["bought"] += 1
            cumulative = 0
            for month in months:
                bucket = buckets[month]
                cumulative += bucket["received"] - bucket["spent"]
                tables["monthly_results"].append({
                    **base, "month": month, "partial_month": month in (months[0], months[-1]),
                    "received_usd": _usd(bucket["received"]),
                    "account_costs_usd": _usd(bucket["spent"]),
                    "net_usd": _usd(bucket["received"] - bucket["spent"]),
                    "cumulative_net_usd": _usd(cumulative),
                    "payouts_received": bucket["payouts"], "accounts_bought": bucket["bought"],
                })
            for r in ledger["cash_ledger"]:
                tables["cash_over_time"].append({
                    **base, "ts_utc": machine_time(r["ts_ns"]),
                    "ts_chicago": display_time(r["ts_ns"]),
                    "received_usd": _usd(r["receipts_after_cents"]),
                    "account_costs_usd": _usd(r["costs_after_cents"]),
                    "net_cash_usd": _usd(r["net_cash_after_cents"]),
                })

    for failure in failures:
        for profile in profiles:
            pair_id = f"{failure['configuration']}|{profile.firm_key}"
            summaries[pair_id] = {
                "pair_id": pair_id, "configuration": failure["configuration"],
                "configuration_label": failure.get("display_name", failure["configuration"]),
                "firm_key": profile.firm_key, "firm": profile.firm_name,
                "status": "Not completed", "reason": failure["reason"],
            }

    for profile in profiles:
        completed = sorted(
            (s for s in summaries.values()
             if s["firm_key"] == profile.firm_key and s["status"] == "Completed"),
            key=lambda s: (-s["net_cash_earned_cents"], s["configuration"]))
        rank = 0
        previous = None
        for index, summary in enumerate(completed, start=1):
            if summary["net_cash_earned_cents"] != previous:
                rank = index
                previous = summary["net_cash_earned_cents"]
            summary["rank_within_firm"] = rank
    for summary in sorted(summaries.values(),
                          key=lambda s: (s["firm_key"], s.get("rank_within_firm", 10**6),
                                         s["configuration"])):
        row = dict(summary)
        _times(row, "first_payout_ns", "max_unrecovered_spending_ns")
        _money(row)
        tables["pair_results"].append(row)

    price_minutes = sum(o["prints"]["minutes_checked"] for o in outputs)
    price_exact = sum(o["prints"]["minutes_rebuilt_exactly"] for o in outputs)
    return {
        "schema_version": COMPARISON_SCHEMA,
        "mode": COMPARISON_MODE,
        **context,
        "period": {"start_utc": machine_time(start_ns), "start_chicago": display_time(start_ns),
                   "cutoff_utc": machine_time(cutoff_ns),
                   "cutoff_chicago": display_time(cutoff_ns), "months": months},
        "settings": settings,
        "price_evidence": {
            "position_minutes_checked": price_minutes,
            "position_minutes_rebuilt_exactly_from_prints": price_exact,
            "position_minutes_approximated": price_minutes - price_exact,
            "missing_print_days": sorted({d for o in outputs
                                          for d in o["prints"]["missing_utc_days"]}),
            "print_files": sorted({f["file"] for o in outputs for f in o["prints"]["files"]}),
        },
        "configurations_requested": len(outputs) + len(failures),
        "configurations_completed": len(outputs),
        "summaries": {k: _dollars(v) for k, v in summaries.items()},
        "summaries_cents": summaries,
        "tables": tables,
    }


def _pair_tick_values(result: dict[str, Any]) -> dict[str, int]:
    settings = result.get("settings") or {}
    sizing = settings.get("sizing_by_configuration") or {}
    default = int(settings.get("tick_value_cents") or 500)
    return {pair: int((sizing.get(s.get("configuration")) or {}).get("tick_value_cents")
                      or default)
            for pair, s in (result.get("summaries_cents") or {}).items()}


_OLD_DAILY_CLOSE = re.compile(
    r"^All positions closed by (?P<clock>.+?) Chicago \(earlier on shortened days\); "
    r"no overnight or weekend holding$")


def _stop_correction(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    summaries = result.get("summaries_cents") or {}
    trades: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in (result.get("tables") or {}).get("trades") or []:
        trades[t["pair_id"]].append(t)
    ticks = _pair_tick_values(result)
    changed: list[dict[str, Any]] = []
    fixed: dict[str, dict[str, int]] = {}
    for pair, s in summaries.items():
        if s.get("status") != "Completed" or "stop_slippage_cents" not in s:
            continue
        fields = _stop_difference_fields(trades.get(pair, []), ticks[pair])
        before = {k: s[k] for k in fields}
        if before != fields:
            fixed[pair] = fields
            changed.append({"pair_id": pair, "before": before, "after": fields})
    if not changed:
        return None
    out = dict(result)
    out["summaries_cents"] = {k: ({**v, **fixed[k]} if k in fixed else v)
                              for k, v in summaries.items()}
    dollars = dict(result.get("summaries") or {})
    for pair, fields in fixed.items():
        if pair in dollars:
            dollars[pair] = {**dollars[pair],
                             "stop_exits_filled_worse_than_stop":
                                 fields["stop_exits_filled_worse_than_stop"],
                             "stop_slippage_usd": _usd(fields["stop_slippage_cents"])}
    out["summaries"] = dollars
    tables = dict(result.get("tables") or {})
    tables["pair_results"] = [
        ({**row, "stop_exits_filled_worse_than_stop":
          fixed[row["pair_id"]]["stop_exits_filled_worse_than_stop"],
          "stop_slippage_usd": _usd(fixed[row["pair_id"]]["stop_slippage_cents"])}
         if row.get("pair_id") in fixed else row)
        for row in tables.get("pair_results") or []]
    out["tables"] = tables
    return out, {
        "correction_id": STOP_DIFFERENCE_CORRECTION_ID,
        "fields": ["stop_exits_filled_worse_than_stop", "stop_slippage_cents"],
        "description": (
            "Stop exits filled worse than the stop, and their dollar total, recomputed from "
            "the saved trade rows: the difference between the final stop and the fill, times "
            "the contracts actually closed at that stop, including break-even stops inside "
            "the half-exit minute. The saved summary had used the entry quantity and skipped "
            "those same-minute break-even stops. Summary-only: trade profits, balances, "
            "costs, payouts, net cash and ranks are unchanged."),
        "pairs_changed": len(changed),
        "changes": changed,
    }


def _daily_close_correction(result: dict[str, Any]
                            ) -> tuple[dict[str, Any], dict[str, Any]] | None:
    from alpha_lab.propsim.funded.comparison_describe import daily_close_text

    tables = result.get("tables") or {}
    rows, changed = [], 0
    for config in tables.get("configurations") or []:
        settings = []
        for item in config.get("settings") or []:
            match = _OLD_DAILY_CLOSE.match(str(item.get("value", "")))
            if item.get("setting") == "Daily close" and match:
                item = {**item, "value": daily_close_text(match.group("clock"))}
                changed += 1
            settings.append(item)
        rows.append({**config, "settings": settings})
    if not changed:
        return None
    return {**result, "tables": {**tables, "configurations": rows}}, {
        "correction_id": DAILY_CLOSE_WORDING_CORRECTION_ID,
        "fields": ["configurations.settings: Daily close"],
        "description": (
            "The daily-close setting said there was no overnight holding. The simulated rule "
            "closes every position by the scheduled daily close; a trading day starts at the "
            "5:00 PM Chicago reopen, so positions can be open past midnight inside one "
            "trading day. Wording only; nothing simulated changed."),
        "configurations_changed": changed,
    }


def apply_reporting_corrections(result: dict[str, Any]) -> dict[str, Any]:
    """Re-derive summary-only reporting fields of a SAVED result from its own rows.

    The saved result bytes and their hash never change. This returns a copy with
    (1) the stop-difference summary fields recomputed from the saved trade rows
    with :func:`stop_fill_difference_cents` and (2) the corrected daily-close
    wording, and records each change under ``reporting_corrections``. Money,
    balances, costs, payouts and ranks are untouched (the trade profits already
    used the correct quantities). A result that needs no correction is returned
    as is (the same object).
    """

    out = result
    records = list(result.get("reporting_corrections") or [])
    done = {r.get("correction_id") for r in records}
    for correct in (_stop_correction, _daily_close_correction):
        fixed = correct(out)
        if fixed is None:
            continue
        out, record = fixed
        if record["correction_id"] not in done:
            records.append(record)
    if out is result:
        return result
    return {**out, "reporting_corrections": records}


def _dollars(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary)
    _times(row, "first_payout_ns", "max_unrecovered_spending_ns")
    _money(row)
    return row


def validate_comparison(result: dict[str, Any], outputs: list[dict[str, Any]],
                        profiles: tuple[FundedFirmProfile, ...], cutoff_ns: int
                        ) -> dict[str, Any]:
    """Independent reconciliation and single-account rule checks (compact)."""

    by_key = {p.firm_key: p for p in profiles}
    checks: dict[str, Any] = {}
    ok = True
    seen_pairs: set[str] = set()
    for output in outputs:
        config_checks = {
            "no_account_replay_equals_saved_study": output["reference"]["equivalent"],
            "resumed_run_identical": bool(output["resumed"]) and all(
                v["identical"] for v in output["resumed"].values()),
        }
        for firm_key, pair in output["pairs"].items():
            profile = by_key[firm_key]
            ledger = pair["ledger"]
            summary = result["summaries_cents"][pair["pair_id"]]
            seen_pairs.add(pair["pair_id"])
            accounts = ledger["accounts"]
            cash = ledger["cash_ledger"]
            receipts = sum(r["amount_cents"] for r in cash if r["kind"] == "payout_received")
            purchases = [r for r in cash if r["kind"] == "account_purchase"]
            costs = sum(r["amount_cents"] for r in purchases)
            requests = [r for r in ledger["payout_events"] if r["event"] == "requested"]
            received = [r for r in ledger["payout_events"] if r["event"] == "received"]
            request_ids = {r["request_id"] for r in requests}
            # one live account at a time; each replacement bought at its predecessor's failure
            sequence_ok = all(a["status"] == "failed" for a in accounts[:-1]) and all(
                later["created_ns"] == earlier["failed_ns"]
                and later["replaces"] == earlier["account_id"]
                for earlier, later in zip(accounts, accounts[1:], strict=False))
            net_trades: dict[str, int] = defaultdict(int)
            for t in ledger["trades"]:
                net_trades[t["account_id"]] += t["net_pnl_cents"]
            gross: dict[str, int] = defaultdict(int)
            for r in requests:
                gross[r["account_id"]] += r["gross_cents"]
            locks: dict[str, list[tuple[int, int]]] = defaultdict(list)
            opened: dict[str, int] = {}
            for row in ledger["account_events"]:
                after, acct = row.get("status_after"), row["account_id"]
                if after in ("secured", "processing") and acct not in opened:
                    opened[acct] = row["ts_ns"]
                elif acct in opened and after not in ("secured", "processing"):
                    locks[acct].append((opened.pop(acct), row["ts_ns"]))
            for acct, start in opened.items():
                locks[acct].append((start, 2**62))
            created = {a["account_id"]: a["created_ns"] for a in accounts}
            pair_checks = {
                "net_cash_equals_receipts_minus_all_account_costs":
                    summary["net_cash_earned_cents"] == receipts - costs
                    == ledger["receipts"] - ledger["costs"],
                "every_account_purchase_charged_once_at_the_firm_price":
                    len(purchases) == len(accounts)
                    and all(r["amount_cents"] == profile.acquisition_cost_cents
                            for r in purchases)
                    and len({r["account_id"] for r in purchases}) == len(accounts),
                "at_most_one_live_account_at_a_time": sequence_ok,
                "no_purchase_at_or_after_the_cutoff":
                    all(a["created_ns"] < cutoff_ns for a in accounts),
                "each_receipt_matches_one_request_once":
                    len({r["request_id"] for r in received}) == len(received)
                    and all(r["request_id"] in request_ids for r in received),
                "no_receipt_before_its_due_time":
                    all(r["received_ns"] >= r["due_ns"] for r in received),
                "no_receipt_after_the_cutoff":
                    all(r["received_ns"] <= cutoff_ns for r in received),
                "gross_equals_trader_plus_firm_share":
                    all(r["gross_cents"] == r["trader_cents"] + r["firm_share_cents"]
                        for r in requests),
                "receipts_sum_to_trader_shares":
                    receipts == sum(r["trader_cents"] for r in received),
                "every_request_retains_the_cushion":
                    all(r["balance_after_cents"] == profile.retained_cushion_cents
                        for r in requests),
                "every_request_meets_the_gross_minimum":
                    all(r["gross_cents"] >= profile.minimum_gross_request_cents
                        for r in requests),
                "account_balances_reconcile_to_trades_and_withdrawals":
                    all(a["balance"] == net_trades[a["account_id"]] - gross[a["account_id"]]
                        for a in accounts),
                "no_entry_while_payout_protected_or_processing":
                    all(not (s < t["entry_ns"] < e) for t in ledger["trades"]
                        for s, e in locks.get(t["account_id"], [])),
                "replacement_never_trades_the_failure_event":
                    all(t["entry_ns"] > created[t["account_id"]] for t in ledger["trades"]
                        if created[t["account_id"]] > accounts[0]["created_ns"]),
                "rows_belong_to_this_pair_only":
                    all(row["pair_id"] == pair["pair_id"] for table in (
                        cash, ledger["payout_events"], ledger["account_events"],
                        ledger["trades"], ledger["boundary_evidence"]) for row in table),
                "no_position_open_at_the_cutoff": ledger["position"] is None,
            }
            config_checks[firm_key] = pair_checks
            ok = ok and all(pair_checks.values())
        checks[output["configuration"]] = config_checks
        ok = ok and config_checks["no_account_replay_equals_saved_study"] and config_checks[
            "resumed_run_identical"]
    expected = {f"{o['configuration']}|{p.firm_key}" for o in outputs for p in profiles}
    checks["one_result_per_configuration_and_firm"] = seen_pairs == expected
    ok = ok and seen_pairs == expected
    return {"passed": ok, "checks": checks}


def strategy_metrics(output: dict[str, Any]) -> dict[str, Any]:
    """Strategy measures of the configuration's no-account replay (evaluation days).

    Same definitions as the study metrics (``search/strategy_metrics.py``): per trade
    net R = (realized points - round-trip cost points) / risk points, with realized
    points = Strategy-Core's realized R x risk (the scale-out blends its two halves);
    a win is a positive result before costs; drawdown is the running peak (floored at
    zero) minus cumulative net R in exit order; time under water is the longest run of
    distinct trading days below that peak. The round-trip cost per contract is
    ``2 x cost per contract per fill / tick value`` ticks = 0.514 points for both the
    mini and the micro.
    """

    sizing = output.get("sizing") or {}
    mills = sizing.get("cost_per_contract_mills", 5140)
    tick_value = sizing.get("tick_value_cents", 500)
    cost_points = 2 * mills / 10 / tick_value * 0.25
    trades = sorted((x for x in output["strategy_trades_no_account"]
                     if not x["is_warmup"] and x.get("realized_r") is not None
                     and x.get("risk_ticks")),
                    key=lambda x: (x["resolution_ts_utc"] or "", x["trade_id"]))
    net, gross = [], []
    for x in trades:
        risk_points = x["risk_ticks"] * 0.25
        realized_points = x["realized_r"] * risk_points
        net.append((realized_points - cost_points) / risk_points)
        gross.append(x["realized_r"])
    equity = peak = drawdown = 0.0
    longest = 0
    current: set[str] = set()
    for x, r in zip(trades, net, strict=True):
        equity += r
        peak = max(peak, equity)
        below = max(0.0, peak) - equity
        drawdown = max(drawdown, below)
        if below > 1e-12:
            current.add(x.get("trading_day") or x["entry_ts_utc"][:10])
            longest = max(longest, len(current))
        else:
            current = set()
    wins = [r for r in net if r > 0]
    losses = [r for r in net if r <= 0]
    gross_win, gross_loss = sum(wins), -sum(losses)

    def side(x) -> str:
        return str(x["direction"]).lower().split(".")[-1]

    return {
        "configuration": output["configuration"],
        "configuration_label": output["display_name"],
        "trades": len(net),
        "long_trades": sum(1 for x in trades if side(x) in ("long", "bullish")),
        "short_trades": sum(1 for x in trades if side(x) in ("short", "bearish")),
        "win_rate_pct": round(100 * sum(1 for g in gross if g > 0) / len(gross), 1)
        if gross else None,
        "net_r_after_costs": round(sum(net), 2),
        "expectancy_r_per_trade": round(sum(net) / len(net), 3) if net else None,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else None,
        "max_drawdown_r": round(drawdown, 2),
        "longest_trading_days_under_water": longest,
        "round_trip_cost_points_per_contract": round(cost_points, 3),
    }
