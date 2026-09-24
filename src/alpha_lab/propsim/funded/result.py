"""The ONE immutable funded result that the screen and the review folder read.

``build_result`` turns finished firm instances into plain JSON-safe tables and
per-firm summaries, then ``validate_result`` re-derives every headline figure
independently from the ledgers (conservation, isolation, idempotency). The
presenter and the export both read the saved result; neither recomputes money.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

from alpha_lab.propsim.funded.campaign import (
    CampaignInputs,
    display_time,
    machine_time,
    month_keys,
)
from alpha_lab.propsim.funded.clock import chicago_date
from alpha_lab.propsim.funded.instance import STATUS_LABELS, FirmInstance
from alpha_lab.propsim.funded.paths import FIDELITY_LABELS

__all__ = [
    "RESULT_SCHEMA_VERSION",
    "TABLE_NAMES",
    "build_result",
    "validate_result",
    "result_sha256",
    "canonical_json",
]

RESULT_SCHEMA_VERSION = "funded_payout_result_v1"

TABLE_NAMES = (
    "instance_results",
    "account_journeys",
    "cash_ledger",
    "payout_events",
    "credit_events",
    "growth_events",
    "monthly_results",
    "trades",
    "account_events",
    "rule_boundary_evidence",
    "comparison",
    "cash_over_time",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def result_sha256(result: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()


def _dollars(cents: int | None) -> float | None:
    return None if cents is None else round(cents / 100.0, 2)


def _times(row: dict[str, Any], *names: str) -> None:
    for name in names:
        ns = row.pop(name, None)
        base = name[:-3] if name.endswith("_ns") else name
        row[f"{base}_utc"] = machine_time(ns)
        row[f"{base}_chicago"] = display_time(ns)


def _money(row: dict[str, Any]) -> None:
    for key in [k for k in row if k.endswith("_cents")]:
        row[key[: -len("_cents")] + "_usd"] = _dollars(row.pop(key))


def _hours(ns: int) -> float:
    return round(ns / 3_600_000_000_000, 2)


def _ready_split(firm: FirmInstance, inputs: CampaignInputs) -> tuple[int, int]:
    """Ready-without-position time split into market closed/locked vs open."""

    closed = sorted((d.deadline_ns, d.reopen_ns) for d in inputs.trading_days)

    def closed_overlap(a: int, b: int) -> int:
        total = 0
        for lo, hi in closed:
            if hi <= a or lo >= b:
                continue
            total += min(b, hi) - max(a, lo)
        return total

    events = defaultdict(list)
    for row in firm.account_events:
        if row.get("status_after") is not None:
            events[row["account_id"]].append((row["ts_ns"], row["status_after"]))
    market_closed = market_open = 0
    for rows in events.values():
        rows.sort()
        for (ts, status), nxt in zip(rows, [*rows[1:], (inputs.cutoff_ns, None)], strict=True):
            if status != "ready":
                continue
            end = min(nxt[0], inputs.cutoff_ns)
            if end <= ts:
                continue
            shut = closed_overlap(ts, end)
            market_closed += shut
            market_open += end - ts - shut
    return market_closed, market_open


def build_result(
    inputs: CampaignInputs,
    instances: dict[str, FirmInstance],
    *,
    run_identity: dict[str, Any],
    price_evidence: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``context`` carries the plan's question, source, owner decisions,
    material limitations, purpose and plan id into the one saved result."""

    firms = [instances[p.firm_key] for p in inputs.profiles]
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLE_NAMES}
    summaries: dict[str, dict[str, Any]] = {}
    months = month_keys(inputs.start_ns, inputs.cutoff_ns)
    first_month, last_month = months[0], months[-1]

    for firm in firms:
        profile = firm.profile
        name = profile.firm_name
        key = profile.firm_key

        def labeled(row: dict[str, Any], *, _key=key, _name=name) -> dict[str, Any]:
            out = {"firm_key": _key, "firm": _name}
            out.update({k: v for k, v in row.items() if k != "firm_key"})
            return out

        # ── payouts ──
        requests = [r for r in firm.payout_events if r["event"] == "requested"]
        receipts = [r for r in firm.payout_events if r["event"] == "received"]
        received_ids = {r["request_id"] for r in receipts}
        pending = [r for r in requests if r["request_id"] not in received_ids]
        secured_not_requested = [
            a for a in firm.accounts if a.status == "secured"
            and (a.pending_payout or {}).get("state") != "release_next_day"
        ]
        failed = [a for a in firm.accounts if a.status == "failed"]
        lost_before = [a for a in failed if a.payouts_received == 0]
        lost_after = [a for a in failed if a.payouts_received > 0]
        credit_costs = sum(r["amount_cents"] for r in firm.cash_ledger
                           if r["kind"] == "account_purchase" and r["funding"] == "monthly_credit")
        wallet_costs = sum(r["amount_cents"] for r in firm.cash_ledger
                           if r["kind"] == "account_purchase" and r["funding"] == "payout_wallet")
        blocked: dict[str, int] = defaultdict(int)
        durations: dict[str, int] = defaultdict(int)
        for account in firm.accounts:
            for reason, count in account.blocked_signals.items():
                blocked[reason] += count
            for status, ns in account.status_durations_ns.items():
                durations[status] += ns
        vacancy_wait_ns = sum(
            row.get("vacancy_wait_ns", 0) for row in firm.account_events
            if row["event"] == "vacancy_filled"
        ) + sum(inputs.cutoff_ns - v["opened_ns"] for v in firm.vacancies)
        alive = [a for a in firm.accounts if a.status != "failed"]
        ready_closed, ready_open = _ready_split(firm, inputs)
        largest = max((r["trader_cents"] for r in receipts), default=0)
        summary = {
            "firm_key": key,
            "firm": name,
            "net_cash_earned_cents": firm.receipts - firm.acquisition_costs,
            "payouts_received_cents": firm.receipts,
            "payouts_received_count": len(receipts),
            "largest_single_payout_cents": largest,
            "acquisition_costs_cents": firm.acquisition_costs,
            "acquisition_costs_from_credits_cents": credit_costs,
            "acquisition_costs_from_wallet_cents": wallet_costs,
            "accounts_bought": len(firm.accounts),
            "accounts_lost_before_first_payout": len(lost_before),
            "accounts_lost_after_a_payout": len(lost_after),
            "accounts_active_at_end": len(alive),
            "capacity_at_end": firm.capacity,
            "unused_credits_at_end": firm.credits,
            "payout_wallet_at_end_cents": firm.wallet,
            "gross_withdrawals_requested_cents": firm.gross_debited,
            "firm_share_cents": firm.firm_share_retained,
            "pending_gross_at_cutoff_cents": sum(r["gross_cents"] for r in pending),
            "pending_after_split_at_cutoff_cents": sum(r["trader_cents"] for r in pending),
            "payouts_processing_at_cutoff": len(pending),
            "secured_not_requested_at_cutoff": len(secured_not_requested),
            "trading_profit_inside_accounts_cents": sum(a.balance for a in alive),
            "growth_purchases": sum(1 for g in firm.growth_events if g["decision"] == "purchased"),
            "next_five_cost_cents": profile.cost_of_next_block_cents,
            "growth_threshold_cents": profile.growth_threshold_cents,
            "vacancies_waiting_at_end": len(firm.vacancies),
            "entries_blocked_payout_protection": blocked.get("payout_protection", 0),
            "entries_blocked_payout_processing": blocked.get("payout_processing", 0),
            "signals_missed_waiting_for_credit": firm.missed_for_credit,
            "hours_trading": _hours(durations.get("in_trade", 0)),
            "hours_payout_protection": _hours(durations.get("secured", 0)),
            "hours_payout_processing": _hours(durations.get("processing", 0)),
            "hours_ready_without_position": _hours(durations.get("ready", 0)),
            "hours_waiting_for_credit": _hours(vacancy_wait_ns),
            "hours_ready_market_closed_or_locked": _hours(ready_closed),
            "hours_ready_market_open_no_new_signal": _hours(ready_open),
            "trades_taken": len(firm.trades),
            "cost_per_account_cents": profile.acquisition_cost_cents,
            "trader_share_pct": profile.trader_share_pct,
            "max_minis": profile.max_mini_equivalent_tenths / 10,
        }
        spent = firm.acquisition_costs
        summary["net_cash_per_dollar_of_account_costs"] = (
            round((firm.receipts - spent) / spent, 4) if spent else None
        )
        summaries[key] = summary
        row = dict(summary)
        _money(row)
        tables["instance_results"].append(row)

        # ── account journeys ──
        for account in firm.accounts:
            journey = {
                "firm_key": key, "firm": name, "account_id": account.account_id,
                "account_number": account.number, "slot": account.slot + 1,
                "funding": account.funding, "replaces_account": account.replaces,
                "created_ns": account.created_ns, "failed_ns": account.failed_ns,
                "status_at_end": STATUS_LABELS[account.status],
                "failure_reason": account.failure_reason,
                "lost_after_payout": account.status == "failed" and account.payouts_received > 0,
                "trades": account.trades, "payouts_received": account.payouts_received,
                "received_cents": account.received_cents,
                "largest_payout_cents": account.largest_receipt_cents,
                "gross_requested_cents": account.gross_requested_cents,
                "final_balance_cents": account.balance,
                "final_floor_cents": account.floor,
                "entries_blocked_payout_protection": account.blocked_signals.get(
                    "payout_protection", 0),
                "entries_blocked_payout_processing": account.blocked_signals.get(
                    "payout_processing", 0),
                "hours_trading": _hours(account.status_durations_ns.get("in_trade", 0)),
                "hours_payout_protection": _hours(account.status_durations_ns.get("secured", 0)),
                "hours_payout_processing": _hours(
                    account.status_durations_ns.get("processing", 0)),
                "hours_ready_without_position": _hours(
                    account.status_durations_ns.get("ready", 0)),
            }
            _times(journey, "created_ns", "failed_ns")
            _money(journey)
            tables["account_journeys"].append(journey)

        for source, target, times in (
            (firm.cash_ledger, "cash_ledger", ("ts_ns",)),
            (firm.credit_events, "credit_events", ("ts_ns",)),
            (firm.growth_events, "growth_events", ("ts_ns",)),
            (firm.account_events, "account_events", ("ts_ns", "due_ns")),
            (firm.trades, "trades", ("ts_ns", "entry_ns", "exit_ns", "min_equity_ns")),
            (firm.boundary_evidence, "rule_boundary_evidence", ("ts_ns",)),
            (firm.payout_events, "payout_events",
             ("ts_ns", "secured_ns", "requested_ns", "due_ns", "received_ns")),
        ):
            for raw in source:
                row = labeled(dict(raw))
                if "vacancy_wait_ns" in row:
                    row["vacancy_wait_hours"] = _hours(row.pop("vacancy_wait_ns") or 0)
                _times(row, *[t for t in times if t in row or t == "ts_ns"])
                _money(row)
                tables[target].append(row)
        # cut-off states are explicit payout rows, never silent
        for r in pending:
            tables["payout_events"].append({
                "firm_key": key, "firm": name, "account_id": r["account_id"],
                "event": "processing_not_received_at_cutoff", "request_id": r["request_id"],
                "gross_usd": _dollars(r["gross_cents"]),
                "trader_usd": _dollars(r["trader_cents"]),
                "due_utc": machine_time(r["due_ns"]), "due_chicago": display_time(r["due_ns"]),
                "ts_utc": machine_time(inputs.cutoff_ns),
                "ts_chicago": display_time(inputs.cutoff_ns),
            })
        for account in secured_not_requested:
            tables["payout_events"].append({
                "firm_key": key, "firm": name, "account_id": account.account_id,
                "event": "secured_not_requested_at_cutoff",
                "ts_utc": machine_time(inputs.cutoff_ns),
                "ts_chicago": display_time(inputs.cutoff_ns),
                "realized_balance_usd": _dollars(account.balance),
            })

        # ── monthly results (every month, including zero and partial) ──
        by_month: dict[str, dict[str, int]] = {
            m: {"received": 0, "spent": 0, "payouts": 0, "purchases": 0} for m in months
        }
        for r in firm.cash_ledger:
            day = chicago_date(r["ts_ns"])
            month = f"{day.year:04d}-{day.month:02d}"
            bucket = by_month[month]
            if r["kind"] == "payout_received":
                bucket["received"] += r["amount_cents"]
                bucket["payouts"] += 1
            else:
                bucket["spent"] += r["amount_cents"]
                bucket["purchases"] += 1 if r["account_id"] else 5
        cumulative = 0
        for month in months:
            bucket = by_month[month]
            cumulative += bucket["received"] - bucket["spent"]
            tables["monthly_results"].append({
                "firm_key": key, "firm": name, "month": month,
                "partial_month": month in (first_month, last_month),
                "received_usd": _dollars(bucket["received"]),
                "account_costs_usd": _dollars(bucket["spent"]),
                "net_usd": _dollars(bucket["received"] - bucket["spent"]),
                "cumulative_net_usd": _dollars(cumulative),
                "payouts_received": bucket["payouts"],
                "accounts_bought": bucket["purchases"],
            })

        # ── cumulative cash series (one point per money event) ──
        for r in firm.cash_ledger:
            tables["cash_over_time"].append({
                "firm_key": key, "firm": name,
                "ts_utc": machine_time(r["ts_ns"]), "ts_chicago": display_time(r["ts_ns"]),
                "received_usd": _dollars(r["receipts_after_cents"]),
                "account_costs_usd": _dollars(r["costs_after_cents"]),
                "net_cash_usd": _dollars(r["net_cash_after_cents"]),
            })

    # ── comparison (separate operations, never summed) ──
    if len(firms) == 2:
        a, b = (summaries[p.firm_key] for p in inputs.profiles)
        for metric, label in (
            ("net_cash_earned_cents", "Net cash earned after all account costs"),
            ("payouts_received_cents", "Payouts received after the firm's share"),
            ("largest_single_payout_cents", "Largest single payout received"),
            ("acquisition_costs_cents", "Account costs"),
            ("accounts_lost_before_first_payout", "Accounts lost before any payout"),
            ("accounts_lost_after_a_payout", "Accounts lost after a payout"),
            ("net_cash_per_dollar_of_account_costs", "Net cash per dollar of account costs"),
        ):
            money = metric.endswith("_cents")
            av, bv = a[metric], b[metric]
            tables["comparison"].append({
                "measure": label,
                "unit": "USD" if money else "count" if "accounts" in metric else "ratio",
                a["firm"]: _dollars(av) if money else av,
                b["firm"]: _dollars(bv) if money else bv,
                "difference_first_minus_second": (
                    _dollars(av - bv) if money else
                    (None if av is None or bv is None else round(av - bv, 4))
                ),
            })

    ambiguous = [
        row for row in tables["rule_boundary_evidence"]
        if row.get("check") == "account_failure" and row.get("approximate")
    ]
    context = dict(context or {})
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_identity": run_identity,
        "funded_plan_id": context.get("funded_plan_id"),
        "purpose": context.get("purpose", "engineering_sample"),
        "question": context.get("question"),
        "source": context.get("source"),
        "owner_decisions": context.get("owner_decisions", []),
        "limitations": context.get("limitations", []),
        "period": {
            "start_utc": machine_time(inputs.start_ns),
            "start_chicago": display_time(inputs.start_ns),
            "cutoff_utc": machine_time(inputs.cutoff_ns),
            "cutoff_chicago": display_time(inputs.cutoff_ns),
            "months": months,
        },
        "settings": {
            "instrument": inputs.instrument,
            "quantity": inputs.quantity,
            "cost_per_side_usd": _dollars(inputs.cost_per_side_cents),
            "processing_clock": inputs.processing.model_dump(mode="json"),
            "firm_profiles": [p.model_dump(mode="json") for p in inputs.profiles],
            "strategy_executions": sum(1 for e in inputs.executions if not e.is_warmup),
        },
        "price_evidence": {
            **price_evidence,
            "labels": FIDELITY_LABELS,
            "approximate_failures": len(ambiguous),
        },
        "summaries": {k: _summary_dollars(v) for k, v in summaries.items()},
        "summaries_cents": summaries,
        "tables": tables,
    }
    return result


def _summary_dollars(summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary)
    _money(row)
    return row


def validate_result(result: dict[str, Any], instances: dict[str, FirmInstance],
                    resumed: dict[str, Any] | None) -> dict[str, Any]:
    """Independent reconciliation of the saved figures (compact outcomes)."""

    checks: dict[str, Any] = {}
    ok = True
    for key, firm in instances.items():
        s = result["summaries_cents"][key]
        receipts = sum(r["amount_cents"] for r in firm.cash_ledger
                       if r["kind"] == "payout_received")
        costs = sum(r["amount_cents"] for r in firm.cash_ledger
                    if r["kind"] == "account_purchase")
        wallet_spend = sum(r["amount_cents"] for r in firm.cash_ledger
                           if r["kind"] == "account_purchase" and r["funding"] == "payout_wallet")
        received_rows = [r for r in firm.payout_events if r["event"] == "received"]
        request_rows = [r for r in firm.payout_events if r["event"] == "requested"]
        credits_granted = sum(r["change"] for r in firm.credit_events
                              if r["kind"] == "monthly_grant")
        credits_used = -sum(r["change"] for r in firm.credit_events if r["kind"] == "credit_used")
        credit_purchases = sum(
            1 for r in firm.cash_ledger
            if r["kind"] == "account_purchase" and r["funding"] == "monthly_credit")
        trade_net = defaultdict(int)
        for t in firm.trades:
            trade_net[t["account_id"]] += t["net_pnl_cents"]
        gross_by_account = defaultdict(int)
        for r in request_rows:
            gross_by_account[r["account_id"]] += r["gross_cents"]
        balances_ok = all(
            a.balance == trade_net[a.account_id] - gross_by_account[a.account_id]
            - (a.open_trade["outcome"].entry_cost_cents if a.open_trade else 0)
            for a in firm.accounts
        )
        grant_ids = [r["grant_id"] for r in firm.credit_events if r["kind"] == "monthly_grant"]
        firm_checks = {
            "net_cash_equals_receipts_minus_all_account_costs":
                s["net_cash_earned_cents"] == receipts - costs,
            "wallet_equals_receipts_minus_wallet_purchases":
                firm.wallet == receipts - wallet_spend,
            "each_receipt_matches_one_request_once":
                len({r["request_id"] for r in received_rows}) == len(received_rows)
                and all(r["request_id"] in {q["request_id"] for q in request_rows}
                        for r in received_rows),
            "gross_equals_trader_plus_firm_share": all(
                r["gross_cents"] == r["trader_cents"] + r["firm_share_cents"]
                for r in request_rows),
            "receipts_sum_to_trader_shares": receipts == sum(
                r["trader_cents"] for r in received_rows),
            "no_receipt_before_its_due_time": all(
                r["received_ns"] >= r["due_ns"] for r in received_rows),
            "every_request_retains_the_cushion": all(
                r["balance_after_cents"] == firm.profile.retained_cushion_cents
                for r in request_rows),
            "every_request_meets_the_gross_minimum": all(
                r["gross_cents"] >= firm.profile.minimum_gross_request_cents
                for r in request_rows),
            "credits_conserved": credits_granted - credits_used == firm.credits
            and credits_used == credit_purchases,
            "monthly_grants_issued_once": len(grant_ids) == len(set(grant_ids)),
            "account_balances_reconcile_to_trades_and_withdrawals": balances_ok,
            "capacity_never_exceeds_20": firm.capacity <= firm.profile.max_capacity,
            "ledger_rows_belong_to_this_firm_only": all(
                row["firm_key"] == key for table in (
                    firm.cash_ledger, firm.credit_events, firm.growth_events,
                    firm.payout_events, firm.account_events, firm.trades)
                for row in table),
            "no_entries_while_secured_or_processing": _no_blocked_entries(firm),
        }
        checks[key] = firm_checks
        ok = ok and all(firm_checks.values())
    if resumed is not None:
        checks["resumed_state_equivalence"] = resumed
        ok = ok and all(item["identical"] for item in resumed.values())
    return {"passed": ok, "checks": checks}


def _no_blocked_entries(firm: FirmInstance) -> bool:
    """No trade entry falls inside an account's secured/processing interval."""

    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    open_at: dict[str, int] = {}
    for row in firm.account_events:
        after = row.get("status_after")
        account = row["account_id"]
        if after in ("secured", "processing") and account not in open_at:
            open_at[account] = row["ts_ns"]
        elif account in open_at and after not in ("secured", "processing"):
            intervals[account].append((open_at.pop(account), row["ts_ns"]))
    for account, start in open_at.items():
        intervals[account].append((start, 2**62))
    for trade in firm.trades:
        for start, end in intervals.get(trade["account_id"], []):
            if start < trade["entry_ns"] < end:
                return False
    return True
