"""SYNTHETIC configuration-comparison result for the screen and export tests.

Hand-made per-configuration worker outputs (shaped like the comparison worker's
``outputs``) are passed through the real ``build_comparison_result`` and
``validate_comparison``. Nothing here reads market data; the prices, dates and
payouts are invented. Passing tests built on this fixture proves only that the
screen and review folder handle every state, not that a historical study ran.

Covered states:

- Configuration "Alpha" (S0_D160): TakeProfitTrader receives a large payout,
  then loses that account and continues on a replacement (a stop filled worse
  than the stop, an entry refused during processing); MyFundedFutures loses its
  first account before any payout and never receives a payout (one trade uses
  the labeled one-minute approximation).
- Configuration "Beta" (S3_D80): TakeProfitTrader takes no trades and receives
  nothing; MyFundedFutures has a payout still processing at the cutoff.
- Configuration "Gamma" (S9_D40): did not complete (shown as "Not completed").
"""

from __future__ import annotations

import copy
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

from alpha_lab.propsim.funded.campaign import TradingDay
from alpha_lab.propsim.funded.clock import to_ns
from alpha_lab.propsim.funded.comparison_result import (
    build_comparison_result,
    validate_comparison,
)
from alpha_lab.propsim.funded.instance import split_gross
from alpha_lab.propsim.funded.profiles import (
    MYFUNDEDFUTURES_PROFILE,
    TAKEPROFITTRADER_PROFILE,
)

__all__ = [
    "START",
    "CUTOFF",
    "PROFILES",
    "FAILED_REASON",
    "comparison_fixture_result",
    "comparison_fixture_parts",
]

START = "2026-01-12T23:00:00Z"
CUTOFF = "2026-03-06T22:00:00Z"
PROFILES = (TAKEPROFITTRADER_PROFILE, MYFUNDEDFUTURES_PROFILE)
TICK_CENTS = 500
COST_PER_SIDE = 514
FAILED_REASON = "ValueError: recorded prices for 2026-02-03 are unavailable"
QUESTION = ("Which tested strategy configuration produces the most simulated cash received "
            "after the cost of every funded account used, over the same selected historical "
            "period?")


def _days() -> tuple[TradingDay, ...]:
    out = []
    day = date(2026, 1, 13)
    while day <= date(2026, 3, 6):
        if day.weekday() < 5:
            text = day.isoformat()
            out.append(TradingDay(trading_day=text, day_end_ns=to_ns(f"{text}T22:00:00Z"),
                                  reopen_ns=to_ns(f"{text}T23:00:00Z"),
                                  deadline_ns=to_ns(f"{text}T21:55:00Z")))
        day += timedelta(days=1)
    return tuple(out)


class _Pair:
    """Builds one configuration-and-firm ledger exactly as the engine shapes it."""

    def __init__(self, configuration: str, profile) -> None:
        self.pair_id = f"{configuration}|{profile.firm_key}"
        self.profile = profile
        self.accounts: list[dict[str, Any]] = []
        self.cash: list[dict[str, Any]] = []
        self.payouts: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.evidence: list[dict[str, Any]] = []
        self.receipts = 0
        self.costs = 0
        self.max_short = 0
        self.max_short_ns: int | None = None
        self.seq = 0
        self.forced_flat = 0

    def _next(self) -> int:
        self.seq += 1
        return self.seq

    def _cash(self, ts: int, kind: str, account_id: str, amount: int, detail: str) -> None:
        before_r, before_c = self.receipts, self.costs
        if kind == "payout_received":
            self.receipts += amount
        else:
            self.costs += amount
        net = self.receipts - self.costs
        if -net > self.max_short:
            self.max_short, self.max_short_ns = -net, ts
        self.cash.append({
            "pair_id": self.pair_id, "seq": self._next(), "kind": kind,
            "account_id": account_id, "detail": detail, "ts_ns": ts, "amount_cents": amount,
            "receipts_before_cents": before_r, "receipts_after_cents": self.receipts,
            "costs_before_cents": before_c, "costs_after_cents": self.costs,
            "net_cash_after_cents": net,
        })

    def _event(self, account: dict[str, Any], ts: int, event: str, before: str | None,
               after: str, reason: str, **extra: Any) -> None:
        account["status"] = after
        self.events.append({
            "pair_id": self.pair_id, "seq": self._next(), "account_id": account["account_id"],
            "event": event, "ts_ns": ts, "status_before": before, "status_after": after,
            "reason": reason, "balance_cents": account["balance"],
            "floor_cents": account["floor"], "replaces": account["replaces"], **extra,
        })

    def buy(self, when: str, detail: str) -> dict[str, Any]:
        ts = to_ns(when)
        previous = self.accounts[-1] if self.accounts else None
        number = len(self.accounts) + 1
        account = {
            "account_id": f"{self.pair_id}#{number}", "number": number, "created_ns": ts,
            "failed_ns": None, "replaces": previous["account_id"] if previous else None,
            "status": "ready", "failure_reason": None, "payouts_received": 0, "trades": 0,
            "received_cents": 0, "largest_receipt_cents": 0, "gross_requested_cents": 0,
            "balance": 0, "floor": -self.profile.loss_allowance_cents, "blocked_entries": {},
            "status_durations_ns": {}, "pending_payout": None,
        }
        self.accounts.append(account)
        self._cash(ts, "account_purchase", account["account_id"],
                   self.profile.acquisition_cost_cents, detail)
        self._event(account, ts, "created", None, "ready", detail)
        return account

    def trade(self, account: dict[str, Any], *, entry: str, exit: str, move_ticks: int,
              kind: str, ref: str, recorded_exit_offset: int = 0,
              approx_minutes: int = 0, fail: bool = False) -> None:
        entry_ns, exit_ns = to_ns(entry), to_ns(exit)
        entry_ticks = 100_000
        exit_ticks = entry_ticks + move_ticks
        stop_ticks = entry_ticks - 80
        gross = move_ticks * TICK_CENTS
        costs = 2 * COST_PER_SIDE
        net = gross - costs
        before = account["balance"]
        self._event(account, entry_ns, "status_change", "ready", "in_trade", "entered a trade",
                    trade_ref=ref)
        account["balance"] = before + net
        account["trades"] += 1
        minutes = max(1, (exit_ns - entry_ns) // 60_000_000_000)
        self.trades.append({
            "pair_id": self.pair_id, "seq": self._next(), "account_id": account["account_id"],
            "trade_ref": ref, "strategy_trade_id": ref, "trading_day": entry[:10],
            "direction": "long", "quantity": 1, "entry_chart": "one-minute entry chart",
            "entry_ticks": entry_ticks, "stop_ticks": stop_ticks,
            "target_ticks": entry_ticks + 160, "exit_ticks": exit_ticks, "exit_kind": kind,
            "exit_basis": "synthetic", "strategy_recorded_exit_kind": kind,
            "strategy_recorded_exit_ticks": exit_ticks + recorded_exit_offset,
            "entry_ns": entry_ns, "exit_ns": exit_ns, "ts_ns": exit_ns,
            "min_equity_ns": exit_ns, "min_equity_cents": before + min(0, net),
            "max_equity_cents": before + max(0, net),
            "gross_pnl_cents": gross, "costs_cents": costs, "net_pnl_cents": net,
            "initial_risk_cents": 80 * TICK_CENTS, "minutes_approximated": approx_minutes,
            "minutes_on_prints": minutes - approx_minutes, "observations_checked": 12,
            "approximate_exit": False, "account_failed": fail,
            "balance_before_cents": before, "balance_after_cents": account["balance"],
            "floor_before_cents": account["floor"], "floor_after_cents": account["floor"],
        })
        if fail:
            self.forced_flat += 1
            account["failed_ns"] = exit_ns
            account["failure_reason"] = "open-position equity reached the loss limit"
            self.evidence.append({
                "pair_id": self.pair_id, "seq": self._next(),
                "account_id": account["account_id"], "check": "account_failure",
                "outcome": "account lost", "detail": "equity at or below the loss limit",
                "ts_ns": exit_ns, "equity_cents": account["balance"],
                "floor_cents": account["floor"], "price_ticks": exit_ticks,
                "comparator": "at_or_below", "approximate": False, "trade_ref": ref,
            })
            self._event(account, exit_ns, "status_change", "in_trade", "failed",
                        "open-position equity reached the loss limit", trade_ref=ref)
        else:
            self._event(account, exit_ns, "status_change", "in_trade", "ready", "trade closed",
                        trade_ref=ref)

    def payout(self, account: dict[str, Any], *, secured: str, requested: str, due: str,
               received: bool, after_cutoff: bool = False) -> None:
        cushion = self.profile.retained_cushion_cents
        gross = account["balance"] - cushion
        trader, firm = split_gross(gross, self.profile.trader_share_pct)
        request_id = f"{account['account_id']}-payout-{gross}"
        s_ns, r_ns, d_ns = to_ns(secured), to_ns(requested), to_ns(due)
        common = {"pair_id": self.pair_id, "account_id": account["account_id"],
                  "request_id": request_id, "gross_cents": gross, "firm_share_cents": firm,
                  "trader_cents": trader, "secured_ns": s_ns, "requested_ns": r_ns,
                  "due_ns": d_ns, "clock_basis": "two_business_days"}
        self.payouts.append({
            "pair_id": self.pair_id, "seq": self._next(), "account_id": account["account_id"],
            "event": "eligibility_secured", "ts_ns": s_ns, "eligible_gross_cents": gross,
            "realized_balance_cents": account["balance"], "floor_cents": account["floor"],
            "trading_day": secured[:10],
        })
        self._event(account, s_ns, "status_change", "ready", "secured", "payout secured")
        before = account["balance"]
        account["balance"] = before - gross
        account["gross_requested_cents"] += gross
        self.payouts.append({**common, "seq": self._next(), "event": "requested", "ts_ns": r_ns,
                             "balance_before_cents": before,
                             "balance_after_cents": account["balance"],
                             "floor_cents": account["floor"], "after_cutoff": after_cutoff,
                             "trading_day": requested[:10]})
        self._event(account, r_ns, "status_change", "secured", "processing",
                    "full surplus requested", due_ns=d_ns, request_id=request_id)
        if not received:
            account["pending_payout"] = {"state": "processing", "request_id": request_id}
            return
        self.payouts.append({**common, "seq": self._next(), "event": "received", "ts_ns": d_ns,
                             "received_ns": d_ns})
        account["payouts_received"] += 1
        account["received_cents"] += trader
        account["largest_receipt_cents"] = max(account["largest_receipt_cents"], trader)
        self._cash(d_ns, "payout_received", account["account_id"], trader, "payout received")
        self._event(account, d_ns, "status_change", "processing", "ready", "payout received")

    def refused(self, account: dict[str, Any], when: str, reason: str) -> None:
        account["blocked_entries"][reason] = account["blocked_entries"].get(reason, 0) + 1
        self.evidence.append({
            "pair_id": self.pair_id, "seq": self._next(), "account_id": account["account_id"],
            "check": "entry_blocked_by_payout_policy", "outcome": "entry refused",
            "detail": reason.replace("_", " "), "ts_ns": to_ns(when),
        })

    def ledger(self, cutoff_ns: int) -> dict[str, Any]:
        for account in self.accounts:
            rows = sorted((e["ts_ns"], e["status_after"]) for e in self.events
                          if e["account_id"] == account["account_id"])
            durations: dict[str, int] = {}
            for (ts, status), nxt in zip(rows, [*rows[1:], (cutoff_ns, None)], strict=True):
                if status != "failed":
                    durations[status] = durations.get(status, 0) + min(nxt[0], cutoff_ns) - ts
            account["status_durations_ns"] = durations
        return {
            "receipts": self.receipts, "costs": self.costs, "max_shortfall": self.max_short,
            "max_shortfall_ns": self.max_short_ns, "accounts": self.accounts,
            "cash_ledger": self.cash, "payout_events": self.payouts,
            "account_events": self.events, "trades": self.trades,
            "boundary_evidence": self.evidence, "position": None,
        }


def _settings_plain(schedule: str, distance: int) -> list[dict[str, str]]:
    return [
        {"setting": "Entry hours", "value": schedule},
        {"setting": "Direction", "value": "Long only"},
        {"setting": "Higher-timeframe gap charts", "value": "one-hour and four-hour"},
        {"setting": "Supporting (parent) charts",
         "value": "five-minute and fifteen-minute"},
        {"setting": "Largest distance from the parent gap to the opposing gap",
         "value": f"{distance} ticks ({distance / 4:g} points)"},
        {"setting": "Profit target", "value": "1 times the initial risk"},
    ]


def _output(configuration: str, display: str, pairs: dict[str, _Pair], cutoff_ns: int,
            schedule: str, distance: int, trades: int) -> dict[str, Any]:
    return {
        "configuration": configuration, "display_name": display,
        "axes": {"schedule": configuration.split("_")[0], "distance_ticks": distance},
        "settings_plain": _settings_plain(schedule, distance),
        "reference": {"equivalent": True, "saved_study_trades": trades,
                      "replayed_trades": trades},
        "resumed": {key: {"identical": True, "events": 10, "checkpoint_after_events": 5}
                    for key in pairs},
        "prints": {"minutes_checked": 120, "minutes_rebuilt_exactly": 110,
                   "missing_utc_days": [], "files": [{"file": "2026-01-13/prints.parquet"}]},
        "pairs": {key: {"pair_id": pair.pair_id, "ledger": pair.ledger(cutoff_ns),
                        "forced_flat": pair.forced_flat, "trades_not_in_reference": 0}
                  for key, pair in pairs.items()},
    }


@lru_cache(maxsize=1)
def _build() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cutoff_ns = to_ns(CUTOFF)
    tpt, mff = TAKEPROFITTRADER_PROFILE, MYFUNDEDFUTURES_PROFILE

    # Alpha / TakeProfitTrader: large payout, later failure, replacement.
    a_tpt = _Pair("S0_D160", tpt)
    acct = a_tpt.buy(START, "first funded account")
    a_tpt.trade(acct, entry="2026-01-13T15:00:00Z", exit="2026-01-13T15:30:00Z",
                move_ticks=1_024, kind="target", ref="0f1e2d3c-aaaa-4bbb-8ccc-000000000001")
    a_tpt.payout(acct, secured="2026-01-13T15:30:00Z", requested="2026-01-13T22:00:00Z",
                 due="2026-01-15T22:00:00Z", received=True)
    a_tpt.refused(acct, "2026-01-14T15:00:00Z", "account_payout_processing")
    a_tpt.trade(acct, entry="2026-01-16T15:00:00Z", exit="2026-01-16T15:20:00Z",
                move_ticks=-420, kind="account_failure",
                ref="0f1e2d3c-aaaa-4bbb-8ccc-000000000002", fail=True)
    acct2 = a_tpt.buy("2026-01-16T15:20:00Z", "replacement for a lost account")
    a_tpt.trade(acct2, entry="2026-01-20T15:00:00Z", exit="2026-01-20T15:10:00Z",
                move_ticks=-84, kind="stop", recorded_exit_offset=4,
                ref="0f1e2d3c-aaaa-4bbb-8ccc-000000000003")

    # Alpha / MyFundedFutures: lost before any payout, zero payouts.
    a_mff = _Pair("S0_D160", mff)
    acct = a_mff.buy(START, "first funded account")
    a_mff.trade(acct, entry="2026-01-14T15:00:00Z", exit="2026-01-14T15:40:00Z",
                move_ticks=-400, kind="account_failure", approx_minutes=3,
                ref="0f1e2d3c-aaaa-4bbb-8ccc-000000000004", fail=True)
    acct2 = a_mff.buy("2026-01-14T15:40:00Z", "replacement for a lost account")
    a_mff.trade(acct2, entry="2026-01-21T21:00:00Z", exit="2026-01-21T21:55:00Z",
                move_ticks=12, kind="scheduled_close",
                ref="0f1e2d3c-aaaa-4bbb-8ccc-000000000005")

    # Beta / TakeProfitTrader: no trades, no payouts.
    b_tpt = _Pair("S3_D80", tpt)
    b_tpt.buy(START, "first funded account")

    # Beta / MyFundedFutures: a payout still processing at the cutoff.
    b_mff = _Pair("S3_D80", mff)
    acct = b_mff.buy(START, "first funded account")
    b_mff.trade(acct, entry="2026-03-05T15:00:00Z", exit="2026-03-05T15:45:00Z",
                move_ticks=600, kind="target", ref="0f1e2d3c-aaaa-4bbb-8ccc-000000000006")
    b_mff.payout(acct, secured="2026-03-05T15:45:00Z", requested="2026-03-05T22:00:00Z",
                 due="2026-03-09T21:00:00Z", received=False, after_cutoff=True)

    outputs = [
        _output("S0_D160", "Alpha | original three windows | distance 160 ticks (40 points)",
                {"takeprofittrader": a_tpt, "myfundedfutures": a_mff}, cutoff_ns,
                "Original three windows", 160, 5),
        _output("S3_D80", "Beta | New York morning only | distance 80 ticks (20 points)",
                {"takeprofittrader": b_tpt, "myfundedfutures": b_mff}, cutoff_ns,
                "New York morning only", 80, 1),
    ]
    failures = [{"configuration": "S9_D40",
                 "display_name": "Gamma | all sessions | distance 40 ticks (10 points)",
                 "reason": FAILED_REASON}]
    context = {
        "run_identity": {"funded_comparison_plan_id": "ab" * 32,
                         "engine_version": "funded_comparison_engine_v1",
                         "execution_model_id": "ordered_prints_stop_market_v2"},
        "funded_comparison_plan_id": "ab" * 32,
        "purpose": "engineering_sample",
        "question": QUESTION,
        "approval": None,
        "source": {"title": "Synthetic comparison fixture", "evaluation_first_day": "2026-01-13",
                   "evaluation_last_day": "2026-03-06", "evaluation_days": 39,
                   "warmup_days": 2},
        "owner_decisions": [
            {"decided_on": "2026-09-23", "subject": "Comparison mode",
             "decision": "One live funded account per configuration and firm; a lost account "
             "is replaced at once at the firm's price.", "status": "owner_confirmed"},
            {"decided_on": "2026-09-22", "subject": "MyFundedFutures equality before the lock",
             "decision": "Equity at or below the floor fails before it locks.",
             "status": "assumption"},
        ],
        "limitations": [
            "One historical path. The results compare configurations on that path; they are "
            "not probabilities of future payouts.",
            "A replacement account is assumed to be available at once when an account fails.",
            "Where recorded trades do not rebuild a minute exactly, that minute uses a "
            "labeled one-minute approximation (losing side first).",
        ],
    }
    settings = {
        "instrument": "mini", "instrument_label": "E-mini Nasdaq-100 (NQ)", "quantity": 1,
        "tick_value_cents": TICK_CENTS, "cost_per_side_usd": COST_PER_SIDE / 100,
        "processing_clock": {"basis": "two_business_days", "payment_time_chicago": "16:00",
                             "description": "Second business day after the request date; "
                             "paid at 4:00 PM Chicago."},
        "firm_profiles": [p.model_dump(mode="json") for p in PROFILES],
        "execution_model": {"id": "ordered_prints_stop_market_v2",
                            "description": "Synthetic execution description.",
                            "refused_entry_policy": "discard_refused_setup",
                            "replacement_policy": "immediate_fresh_account_same_configuration"},
    }
    result = build_comparison_result(
        context=context, outputs=outputs, failures=failures, profiles=PROFILES,
        trading_days=_days(), start_ns=to_ns(START), cutoff_ns=cutoff_ns, settings=settings)
    result["validation"] = validate_comparison(result, outputs, PROFILES, cutoff_ns)
    result["validation"]["checks"]["every_configuration_completed"] = not failures
    result["validation"]["checks"]["at_least_one_configuration_completed"] = bool(outputs)
    result["validation"]["configurations_not_completed"] = failures
    result["validation"]["passed"] = bool(result["validation"]["passed"] and outputs)
    return result, outputs


def comparison_fixture_result() -> dict[str, Any]:
    """A fresh deep copy of the synthetic comparison result (safe to mutate)."""

    return copy.deepcopy(_build()[0])


def comparison_fixture_parts() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result, outputs = _build()
    return copy.deepcopy(result), copy.deepcopy(outputs)
