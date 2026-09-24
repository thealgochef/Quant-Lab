"""One configuration x one firm: at most one live funded account at a time.

The single-account configuration-comparison mode (owner scope update,
September 23, 2026). A :class:`PairLedger` owns the chronological sequence of
accounts of ONE configuration at ONE firm:

1. The first fresh funded account is bought at the evaluation start and charged.
2. Only one account is ever alive. A payout-processing account is alive: it is
   never replaced or supplemented, and it cannot trade until processing ends.
3. On failure the account is finalized, and a fresh replacement is bought and
   charged at once, with no credit limit and no funding test. No purchase is
   made at or after the cutoff. The replacement trades only from a later
   strategy opportunity (the strategy driver clears the failed setup).
4. Nothing is ever reset: costs and receipts accumulate over the whole period.

There are no monthly credits, vacancies, wallets, growth blocks or copied
accounts here; those belong to the separate budgeted mode
(:mod:`alpha_lab.propsim.funded.instance`).

Same-timestamp precedence of timed events: payout-processing completion, then
the end-of-trading-day step (MyFundedFutures closing-balance floor update, then
the full-surplus request), then the next-day release of a failed request.
Position exits and entries are driven minute by minute by the pair engine.
"""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass, field
from typing import Any

from alpha_lab.propsim.funded.campaign import TradingDay
from alpha_lab.propsim.funded.clock import ProcessingClockPolicy, processing_due_ns
from alpha_lab.propsim.funded.instance import STATUS_LABELS, split_gross
from alpha_lab.propsim.funded.position_walk import (
    MinuteObservations,
    OpenPosition,
    PositionExit,
    open_position,
    walk_minute,
)
from alpha_lab.propsim.funded.profiles import FundedFirmProfile, breaches

__all__ = ["PairLedger", "PairAccount", "BLOCK_REASONS", "TIMED_PRIORITY"]

TIMED_PRIORITY = {"processing_complete": 2, "day_end": 5, "day_release": 6}

#: account-side reasons an entry is refused (the strategy driver passes these
#: to Strategy-Core's execution-time admission check)
BLOCK_REASONS = {
    "secured": "account_payout_protection",
    "processing": "account_payout_processing",
}


@dataclass
class PairAccount:
    account_id: str
    number: int
    created_ns: int
    replaces: str | None
    balance: int = 0
    floor: int = 0
    peak: int = 0
    status: str = "ready"
    status_since_ns: int = 0
    secured_ns: int | None = None
    secured_trading_day: str | None = None
    failed_ns: int | None = None
    failure_reason: str | None = None
    trades: int = 0
    payouts_received: int = 0
    received_cents: int = 0
    gross_requested_cents: int = 0
    largest_receipt_cents: int = 0
    pending_payout: dict[str, Any] | None = None
    blocked_entries: dict[str, int] = field(default_factory=dict)
    status_durations_ns: dict[str, int] = field(default_factory=dict)


class PairLedger:
    def __init__(self, *, pair_id: str, configuration: str, profile: FundedFirmProfile,
                 processing: ProcessingClockPolicy, quantity: int, tick_value_cents: int,
                 cost_per_side_cents: int, trading_days: tuple[TradingDay, ...],
                 start_ns: int, cutoff_ns: int, cost_per_contract_mills: int | None = None,
                 scale_out: bool = False) -> None:
        self.pair_id = pair_id
        self.configuration = configuration
        self.profile = profile
        self.processing = processing
        self.quantity = quantity
        self.tick_value_cents = tick_value_cents
        self.cost_per_side_cents = cost_per_side_cents
        #: exact cost per contract per fill in tenths of a cent (micros cost $0.514)
        self.cost_per_contract_mills = (cost_per_contract_mills
                                        if cost_per_contract_mills is not None
                                        else cost_per_side_cents * 10)
        self.scale_out = scale_out
        self.trading_days = trading_days
        self.start_ns = start_ns
        self.cutoff_ns = cutoff_ns
        self.accounts: list[PairAccount] = []
        self.position: OpenPosition | None = None
        self.receipts = 0
        self.costs = 0
        self.max_shortfall = 0
        self.max_shortfall_ns: int | None = None
        self.queue: list[list] = []
        self.qseq = 0
        self.seq = 0
        self.started = False
        self.finished = False
        self._open_strategy: dict[str, Any] | None = None
        # emitted tables (append-only)
        self.cash_ledger: list[dict] = []
        self.payout_events: list[dict] = []
        self.account_events: list[dict] = []
        self.trades: list[dict] = []
        self.boundary_evidence: list[dict] = []

    # ── helpers ───────────────────────────────────────────────────────────
    @property
    def current(self) -> PairAccount:
        return self.accounts[-1]

    @property
    def net_cash(self) -> int:
        return self.receipts - self.costs

    def _row(self, table: list[dict], ts_ns: int, **fields: Any) -> dict:
        self.seq += 1
        row = {"pair_id": self.pair_id, "seq": self.seq, "ts_ns": ts_ns, **fields}
        table.append(row)
        return row

    def _push(self, ts_ns: int, kind: str, *payload: Any) -> None:
        self.qseq += 1
        heapq.heappush(self.queue, [ts_ns, TIMED_PRIORITY[kind], self.qseq, kind, list(payload)])

    def _set_status(self, account: PairAccount, status: str, ts_ns: int, reason: str,
                    **detail: Any) -> None:
        elapsed = max(0, ts_ns - account.status_since_ns)
        account.status_durations_ns[account.status] = (
            account.status_durations_ns.get(account.status, 0) + elapsed)
        before = account.status
        account.status = status
        account.status_since_ns = ts_ns
        self._row(self.account_events, ts_ns, account_id=account.account_id,
                  event=f"{before}->{status}", status_before=before, status_after=status,
                  reason=reason, balance_cents=account.balance, floor_cents=account.floor,
                  **detail)

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self._buy(self.start_ns, replaces=None, reason="first funded account")
        for day in self.trading_days:
            if self.start_ns < day.day_end_ns <= self.cutoff_ns:
                self._push(day.day_end_ns, "day_end", day.trading_day)
            if self.start_ns < day.reopen_ns <= self.cutoff_ns:
                self._push(day.reopen_ns, "day_release")

    def _buy(self, ts_ns: int, *, replaces: str | None, reason: str) -> PairAccount:
        number = len(self.accounts) + 1
        account = PairAccount(
            account_id=f"{self.pair_id}#{number}", number=number, created_ns=ts_ns,
            replaces=replaces, balance=0, floor=-self.profile.loss_allowance_cents, peak=0,
            status="ready", status_since_ns=ts_ns,
        )
        self.accounts.append(account)
        before = (self.receipts, self.costs)
        self.costs += self.profile.acquisition_cost_cents
        self._track_shortfall(ts_ns)
        self._row(self.cash_ledger, ts_ns, kind="account_purchase",
                  amount_cents=self.profile.acquisition_cost_cents,
                  account_id=account.account_id, detail=reason,
                  receipts_before_cents=before[0], receipts_after_cents=self.receipts,
                  costs_before_cents=before[1], costs_after_cents=self.costs,
                  net_cash_after_cents=self.net_cash)
        self._row(self.account_events, ts_ns, account_id=account.account_id,
                  event="created", status_before=None, status_after="ready", reason=reason,
                  balance_cents=0, floor_cents=account.floor, replaces=replaces)
        return account

    def _track_shortfall(self, ts_ns: int) -> None:
        if -self.net_cash > self.max_shortfall:
            self.max_shortfall = -self.net_cash
            self.max_shortfall_ns = ts_ns

    # ── timed events ──────────────────────────────────────────────────────
    def run_until(self, ts_ns: int) -> None:
        """Process every timed event at or before ``ts_ns`` in fixed order."""

        while self.queue and self.queue[0][0] <= ts_ns:
            at, _prio, _seq, kind, payload = heapq.heappop(self.queue)
            if kind == "processing_complete":
                self._processing_complete(at, *payload)
            elif kind == "day_end":
                self._day_end(at, payload[0])
            elif kind == "day_release":
                self._day_release(at)
            else:  # pragma: no cover
                raise ValueError(kind)

    def _day_end(self, ts_ns: int, trading_day: str) -> None:
        account = self.current
        if account.status == "failed":  # pragma: no cover - replaced at once
            raise AssertionError("no live account at the end of the trading day")
        if self.position is not None:
            raise AssertionError("a position is open at the end of the trading day")
        if self.profile.threshold_update == "session_close_balance":
            prior = account.floor
            new = min(self.profile.floor_lock_cents,
                      max(prior, account.balance - self.profile.loss_allowance_cents))
            if new != prior:
                account.floor = new
                self._row(self.boundary_evidence, ts_ns, account_id=account.account_id,
                          check="floor_moved",
                          outcome="floor locked" if new >= self.profile.floor_lock_cents
                          else "floor raised", trade_ref=None, prior_floor_cents=prior,
                          new_floor_cents=new, peak_equity_cents=account.balance,
                          detail="scheduled session-close realized balance")
            if breaches(account.balance, account.floor,
                        self.profile.comparator_for(account.floor)):
                raise AssertionError("closed balance below its floor at day end")
        if account.status != "secured" or account.secured_trading_day != trading_day:
            return
        gross = max(0, account.balance - self.profile.retained_cushion_cents)
        if gross < self.profile.minimum_gross_request_cents:
            self._row(self.payout_events, ts_ns, account_id=account.account_id,
                      event="request_shortfall", trading_day=trading_day,
                      realized_balance_cents=account.balance, eligible_gross_cents=gross,
                      floor_cents=account.floor)
            account.pending_payout = {"state": "release_next_day"}
            return
        trader, firm = split_gross(gross, self.profile.trader_share_pct)
        due = processing_due_ns(ts_ns, self.processing)
        before = account.balance
        account.balance -= gross
        account.gross_requested_cents += gross
        request_id = f"{account.account_id}-payout-{account.gross_requested_cents}"
        account.pending_payout = {
            "state": "processing", "request_id": request_id, "gross": gross,
            "trader": trader, "firm": firm, "requested_ns": ts_ns, "due_ns": due,
            "secured_ns": account.secured_ns,
        }
        self._row(self.payout_events, ts_ns, account_id=account.account_id,
                  event="requested", trading_day=trading_day, request_id=request_id,
                  secured_ns=account.secured_ns, requested_ns=ts_ns, due_ns=due,
                  gross_cents=gross, firm_share_cents=firm, trader_cents=trader,
                  balance_before_cents=before, balance_after_cents=account.balance,
                  floor_cents=account.floor, clock_basis=self.processing.basis,
                  after_cutoff=due > self.cutoff_ns)
        self._set_status(account, "processing", ts_ns,
                         "full surplus requested at the end of the trading day",
                         request_id=request_id, due_ns=due)
        self._push(due, "processing_complete", account.account_id, request_id)

    def _day_release(self, ts_ns: int) -> None:
        account = self.current
        if account.status == "secured" and (account.pending_payout or {}).get(
                "state") == "release_next_day":
            account.pending_payout = None
            account.secured_ns = None
            account.secured_trading_day = None
            self._set_status(account, "ready", ts_ns,
                             "request not possible; released at the next trading day")

    def _processing_complete(self, ts_ns: int, account_id: str, request_id: str) -> None:
        account = self._account(account_id)
        pending = account.pending_payout
        if not pending or pending.get("request_id") != request_id:
            return  # idempotent
        if account.status != "processing":
            raise AssertionError("processing completion for an account not processing")
        trader = pending["trader"]
        before = (self.receipts, self.costs)
        self.receipts += trader
        self._row(self.cash_ledger, ts_ns, kind="payout_received", amount_cents=trader,
                  account_id=account.account_id,
                  detail=f"after-split payout for request {request_id}",
                  receipts_before_cents=before[0], receipts_after_cents=self.receipts,
                  costs_before_cents=before[1], costs_after_cents=self.costs,
                  net_cash_after_cents=self.net_cash)
        account.payouts_received += 1
        account.received_cents += trader
        account.largest_receipt_cents = max(account.largest_receipt_cents, trader)
        self._row(self.payout_events, ts_ns, account_id=account.account_id,
                  event="received", request_id=request_id,
                  requested_ns=pending["requested_ns"], due_ns=pending["due_ns"],
                  received_ns=ts_ns, gross_cents=pending["gross"],
                  firm_share_cents=pending["firm"], trader_cents=trader,
                  clock_basis=self.processing.basis, secured_ns=pending["secured_ns"])
        account.pending_payout = None
        account.secured_ns = None
        account.secured_trading_day = None
        self._set_status(account, "ready", ts_ns,
                         "processing complete; the payout lock is released",
                         request_id=request_id)

    # ── strategy interface ────────────────────────────────────────────────
    def gate(self) -> str | None:
        """Why the live account refuses a new entry now (None: it may enter)."""

        # while a position is open Strategy-Core itself refuses new entries
        return BLOCK_REASONS.get(self.current.status)

    def note_blocked(self, ts_ns: int, reason: str, detail: str) -> None:
        account = self.current
        account.blocked_entries[reason] = account.blocked_entries.get(reason, 0) + 1
        self._row(self.boundary_evidence, ts_ns, account_id=account.account_id,
                  check="entry_blocked_by_payout_policy", outcome="entry refused",
                  trade_ref=None, detail=f"{reason}: {detail}")

    def open(self, *, ts_ns: int, trade_ref: str, direction: str, entry_ticks: int,
             stop_ticks: int, target_ticks: int, trading_day: str,
             strategy: dict[str, Any]) -> None:
        account = self.current
        if self.position is not None or account.status != "ready":
            raise AssertionError(f"entry while the account is {account.status}")
        if ts_ns >= self.cutoff_ns:
            raise AssertionError("entry at or after the cutoff")
        position, failure = open_position(
            profile=self.profile, trade_ref=trade_ref, direction=direction, entry_ns=ts_ns,
            entry_ticks=entry_ticks, stop_ticks=stop_ticks, target_ticks=target_ticks,
            quantity=self.quantity, tick_value_cents=self.tick_value_cents,
            cost_per_side_cents=self.cost_per_side_cents, balance_cents=account.balance,
            floor_cents=account.floor, peak_cents=account.peak,
            cost_per_contract_mills=self.cost_per_contract_mills, scale_out=self.scale_out)
        account.trades += 1
        self.position = position
        self._open_strategy = dict(strategy, trading_day=trading_day)
        self._set_status(account, "in_trade", ts_ns, "entered a new strategy signal",
                         trade_ref=trade_ref)
        if failure is not None:
            self._close(failure, trading_day, fidelity="entry_fill")

    def on_minute(self, obs: MinuteObservations, *, deadline_minute: bool,
                  trading_day: str) -> PositionExit | None:
        if self.position is None:
            return None
        outcome = walk_minute(profile=self.profile, pos=self.position, obs=obs,
                              deadline_minute=deadline_minute)
        if outcome is not None:
            self._close(outcome, trading_day, fidelity=obs.fidelity)
        return outcome

    def _close(self, exit_: PositionExit, trading_day: str, *, fidelity: str) -> None:
        account = self.current
        pos = self.position
        strategy = self._open_strategy
        self.position = None
        account.balance = exit_.balance_after_cents
        account.floor = exit_.floor_cents
        account.peak = max(account.peak, exit_.peak_cents)
        net = exit_.balance_after_cents - pos.balance_before_cents
        self._row(
            self.trades, exit_.ts_ns, account_id=account.account_id,
            trade_ref=pos.trade_ref, trading_day=strategy["trading_day"],
            direction=pos.direction, quantity=pos.quantity, entry_ns=pos.entry_ns,
            entry_ticks=pos.entry_ticks,
            stop_ticks=pos.initial_stop_ticks if pos.scaled else pos.stop_ticks,
            final_stop_ticks=pos.stop_ticks,
            target_ticks=pos.target_ticks, exit_ns=exit_.ts_ns, exit_ticks=exit_.fill_ticks,
            exit_kind=exit_.kind, exit_basis=exit_.fill_basis,
            scale_out_ns=exit_.scale_out_ns, scale_out_ticks=exit_.scale_out_ticks,
            scale_out_quantity=exit_.scale_out_quantity,
            final_exit_quantity=pos.remaining_quantity,
            gross_pnl_cents=exit_.gross_pnl_cents,
            costs_cents=pos.entry_cost_cents + pos.partial_cost_cents + pos.exit_cost_cents,
            net_pnl_cents=net,
            initial_risk_cents=pos.initial_risk_cents,
            balance_before_cents=pos.balance_before_cents,
            balance_after_cents=exit_.balance_after_cents,
            floor_before_cents=pos.floor_before_cents, floor_after_cents=exit_.floor_cents,
            min_equity_cents=min(pos.min_equity_cents, exit_.balance_after_cents
                                 + pos.exit_cost_cents),
            min_equity_ns=pos.min_equity_ns, max_equity_cents=pos.max_equity_cents,
            observations_checked=pos.observations,
            minutes_on_prints=pos.minutes_on_prints,
            minutes_approximated=pos.minutes_approximated,
            approximate_exit=exit_.approximate, account_failed=exit_.account_failed,
            entry_chart=strategy.get("entry_chart"),
            strategy_trade_id=strategy.get("trade_id"),
        )
        for ts, prior, new, peak in pos.floor_transitions:
            self._row(self.boundary_evidence, ts, account_id=account.account_id,
                      check="floor_moved",
                      outcome="floor locked" if new >= self.profile.floor_lock_cents
                      else "floor raised", trade_ref=pos.trade_ref,
                      prior_floor_cents=prior, new_floor_cents=new, peak_equity_cents=peak,
                      detail="intraday peak of realized plus open equity")
        if exit_.account_failed:
            self._fail(account, exit_, pos)
            return
        self._set_status(account, "ready", exit_.ts_ns, f"strategy exit ({exit_.kind})",
                         trade_ref=pos.trade_ref)
        gross = max(0, account.balance - self.profile.retained_cushion_cents)
        if gross >= self.profile.minimum_gross_request_cents:
            account.secured_ns = exit_.ts_ns
            account.secured_trading_day = trading_day
            self._set_status(account, "secured", exit_.ts_ns,
                             "realized payout eligibility reached; no more entries today",
                             eligible_gross_cents=gross)
            self._row(self.payout_events, exit_.ts_ns, account_id=account.account_id,
                      event="eligibility_secured", trading_day=trading_day,
                      realized_balance_cents=account.balance, eligible_gross_cents=gross,
                      floor_cents=account.floor)

    def _fail(self, account: PairAccount, exit_: PositionExit, pos: OpenPosition) -> None:
        stage_text = {
            "at_entry_cost": "the entry cost reached the loss limit",
            "open_position": "open-position equity reached the loss limit",
            "after_exit_cost": "the exit cost took the balance to the loss limit",
        }[exit_.failure_stage]
        account.failed_ns = exit_.ts_ns
        account.failure_reason = stage_text
        for index, (t, px, eq, fl) in enumerate(exit_.evidence):
            self._row(self.boundary_evidence, t, account_id=account.account_id,
                      check="ordered_observation_before_failure",
                      outcome="failure observation" if index == len(exit_.evidence) - 1
                      else "observation", trade_ref=pos.trade_ref, price_ticks=px,
                      equity_cents=eq, floor_cents=fl)
        self._row(self.boundary_evidence, exit_.ts_ns, account_id=account.account_id,
                  check="account_failure", outcome="account lost", trade_ref=pos.trade_ref,
                  equity_cents=exit_.failure_equity_cents, floor_cents=exit_.floor_cents,
                  comparator=exit_.failure_comparator, price_ticks=exit_.fill_ticks,
                  detail=exit_.fill_basis, approximate=exit_.approximate)
        self._set_status(account, "failed", exit_.ts_ns, stage_text, trade_ref=pos.trade_ref,
                         paid_before_failure=account.payouts_received > 0)
        if exit_.ts_ns < self.cutoff_ns:
            self._buy(exit_.ts_ns, replaces=account.account_id,
                      reason=f"replacement for account {account.number}")

    def finish(self) -> None:
        """Run the calendar to the cutoff; pending money stays unreceived."""

        if self.finished:
            return
        if self.position is not None:
            raise AssertionError("a position is open at the cutoff")
        self.run_until(self.cutoff_ns)
        for account in self.accounts:
            elapsed = max(0, self.cutoff_ns - account.status_since_ns)
            account.status_durations_ns[account.status] = (
                account.status_durations_ns.get(account.status, 0) + elapsed)
            account.status_since_ns = self.cutoff_ns
        self.finished = True

    def _account(self, account_id: str) -> PairAccount:
        for account in self.accounts:
            if account.account_id == account_id:
                return account
        raise KeyError(account_id)

    # ── checkpoint ────────────────────────────────────────────────────────
    _FIXED = ("profile", "processing", "trading_days")

    def snapshot(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items()
                 if k not in (*self._FIXED, "accounts", "position")}
        state["accounts"] = [asdict(a) for a in self.accounts]
        state["position"] = None if self.position is None else self.position.to_json()
        return state

    def restore(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            if key in ("accounts", "position"):
                continue
            setattr(self, key, value)
        self.accounts = [PairAccount(**a) for a in state["accounts"]]
        self.position = (None if state["position"] is None
                         else OpenPosition.from_json(state["position"]))
        self.queue = [list(item) for item in state["queue"]]
        heapq.heapify(self.queue)

    def status_label(self) -> str:
        return STATUS_LABELS[self.current.status]
