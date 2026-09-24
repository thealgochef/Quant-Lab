"""One firm's funded operation: accounts, credits, payouts, wallet and growth.

An :class:`FirmInstance` is driven by :mod:`alpha_lab.propsim.funded.campaign`
through a single time-ordered event queue. It never reads or writes another
firm's state: every ledger row it emits carries its own ``firm_key`` and the
campaign checks that no money crosses instances.

Same-timestamp precedence (fixed, declared in the run rules):

1. position exits and failures (fills, costs and breach resolution first);
2. payout-processing completions (cash received, payout lock released);
3. growth review of those same-time receipts (one decision per firm per day);
4. monthly credit grants, then vacancy filling;
5. end-of-trading-day: MyFundedFutures floor update from the closing balance,
   THEN full-surplus payout requests, then day-release of failed requests;
6. new strategy entries (strictly after anything above at the same time).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from alpha_lab.propsim.funded.clock import ProcessingClockPolicy, processing_due_ns
from alpha_lab.propsim.funded.paths import ExecutionPath, StrategyExecution
from alpha_lab.propsim.funded.positions import (
    AccountMarks,
    PositionOutcome,
    simulate_position,
)
from alpha_lab.propsim.funded.profiles import FundedFirmProfile, breaches

__all__ = ["FundedAccount", "FirmInstance", "split_gross", "STATUS_LABELS"]

STATUS_LABELS = {
    "ready": "Ready to trade",
    "in_trade": "Trading",
    "secured": "Payout secured — finished for today",
    "processing": "Payout processing — trading paused",
    "failed": "Account lost",
}


def split_gross(gross_cents: int, trader_share_pct: int) -> tuple[int, int]:
    """(trader, firm) cents; trader share rounded half-up to the cent."""

    trader = int(
        (Decimal(gross_cents) * Decimal(trader_share_pct) / Decimal(100)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    )
    return trader, gross_cents - trader


@dataclass
class FundedAccount:
    account_id: str
    firm_key: str
    number: int
    slot: int
    created_ns: int
    funding: str  # initial_credit | replacement_credit | growth_wallet
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
    paid_before_failure: bool = False
    open_trade: dict[str, Any] | None = None
    pending_payout: dict[str, Any] | None = None
    trades: int = 0
    payouts_received: int = 0
    received_cents: int = 0
    gross_requested_cents: int = 0
    largest_receipt_cents: int = 0
    blocked_signals: dict[str, int] = field(default_factory=dict)
    status_durations_ns: dict[str, int] = field(default_factory=dict)


class FirmInstance:
    def __init__(
        self,
        *,
        profile: FundedFirmProfile,
        processing: ProcessingClockPolicy,
        quantity: int,
        tick_value_cents: int,
        cost_per_side_cents: int,
        start_ns: int,
    ) -> None:
        self.profile = profile
        self.processing = processing
        self.quantity = quantity
        self.tick_value_cents = tick_value_cents
        self.cost_per_side_cents = cost_per_side_cents
        self.start_ns = start_ns
        self.accounts: list[FundedAccount] = []
        self.capacity = 0
        self.credits = 0
        self.wallet = 0
        self.receipts = 0
        self.acquisition_costs = 0
        self.gross_debited = 0
        self.firm_share_retained = 0
        self.vacancies: list[dict[str, Any]] = []  # FIFO of unfilled slots
        self.granted_months: list[str] = []
        self.growth_days: list[str] = []
        self.ledger_seq = 0
        self.missed_for_credit = 0  # vacant slots x new strategy signals
        # emitted tables (append-only)
        self.cash_ledger: list[dict] = []
        self.credit_events: list[dict] = []
        self.growth_events: list[dict] = []
        self.payout_events: list[dict] = []
        self.account_events: list[dict] = []
        self.trades: list[dict] = []
        self.boundary_evidence: list[dict] = []

    # ── identity / helpers ────────────────────────────────────────────────
    @property
    def key(self) -> str:
        return self.profile.firm_key

    def _row(self, table: list[dict], ts_ns: int, **fields: Any) -> dict:
        self.ledger_seq += 1
        row = {"firm_key": self.key, "seq": self.ledger_seq, "ts_ns": ts_ns, **fields}
        table.append(row)
        return row

    def _set_status(self, account: FundedAccount, status: str, ts_ns: int, reason: str,
                    **detail: Any) -> None:
        elapsed = max(0, ts_ns - account.status_since_ns)
        account.status_durations_ns[account.status] = (
            account.status_durations_ns.get(account.status, 0) + elapsed
        )
        before = account.status
        account.status = status
        account.status_since_ns = ts_ns
        self._row(
            self.account_events, ts_ns, account_id=account.account_id,
            event=f"{before}->{status}", status_before=before, status_after=status,
            reason=reason, balance_cents=account.balance, floor_cents=account.floor,
            slot=account.slot, **detail,
        )

    def alive_accounts(self) -> list[FundedAccount]:
        return [a for a in self.accounts if a.status != "failed"]

    # ── money ─────────────────────────────────────────────────────────────
    def _cash(self, ts_ns: int, kind: str, *, amount: int, funding: str,
              account_id: str | None, detail: str) -> None:
        wallet_before = self.wallet
        receipts_before = self.receipts
        costs_before = self.acquisition_costs
        if kind == "payout_received":
            self.receipts += amount
            self.wallet += amount
        elif kind == "account_purchase":
            self.acquisition_costs += amount
            if funding == "payout_wallet":
                if self.wallet < amount:
                    raise AssertionError("wallet purchase exceeds the available wallet")
                self.wallet -= amount
        else:  # pragma: no cover - programming error
            raise ValueError(kind)
        self._row(
            self.cash_ledger, ts_ns, kind=kind, amount_cents=amount, funding=funding,
            account_id=account_id, detail=detail,
            wallet_before_cents=wallet_before, wallet_after_cents=self.wallet,
            receipts_before_cents=receipts_before, receipts_after_cents=self.receipts,
            costs_before_cents=costs_before, costs_after_cents=self.acquisition_costs,
            net_cash_after_cents=self.receipts - self.acquisition_costs,
        )

    @property
    def net_cash(self) -> int:
        return self.receipts - self.acquisition_costs

    # ── accounts ──────────────────────────────────────────────────────────
    def _new_account(self, ts_ns: int, *, slot: int, funding: str,
                     replaces: str | None) -> FundedAccount:
        number = len(self.accounts) + 1
        account = FundedAccount(
            account_id=f"{self.key}-{number:03d}", firm_key=self.key, number=number,
            slot=slot, created_ns=ts_ns, funding=funding, replaces=replaces,
            balance=0, floor=-self.profile.loss_allowance_cents, peak=0,
            status="ready", status_since_ns=ts_ns,
        )
        self.accounts.append(account)
        self._row(
            self.account_events, ts_ns, account_id=account.account_id,
            event="created", status_before=None, status_after="ready",
            reason=f"fresh funded account ({funding.replace('_', ' ')})",
            balance_cents=0, floor_cents=account.floor, slot=slot,
            replaces=replaces,
        )
        return account

    def _purchase_with_credit(self, ts_ns: int, *, slot: int, funding: str,
                              replaces: str | None, reason: str) -> FundedAccount:
        if self.credits <= 0:
            raise AssertionError("purchase without an available credit")
        self.credits -= 1
        account = self._new_account(ts_ns, slot=slot, funding=funding, replaces=replaces)
        self._row(
            self.credit_events, ts_ns, kind="credit_used", change=-1,
            credits_after=self.credits, grant_id=None, account_id=account.account_id,
            detail=reason,
        )
        self._cash(ts_ns, "account_purchase", amount=self.profile.acquisition_cost_cents,
                   funding="monthly_credit", account_id=account.account_id, detail=reason)
        return account

    def start(self) -> None:
        month = _month_id(self.start_ns)
        self._grant(self.start_ns, month, initial=True)
        for slot in range(self.profile.initial_accounts):
            self._purchase_with_credit(
                self.start_ns, slot=slot, funding="initial_credit", replaces=None,
                reason="initial funded account",
            )
        self.capacity = self.profile.initial_accounts

    def _grant(self, ts_ns: int, grant_id: str, *, initial: bool = False) -> None:
        if grant_id in self.granted_months:
            # idempotent: a resumed run can never issue a month twice
            return
        self.granted_months.append(grant_id)
        self.credits += self.profile.monthly_credits
        self._row(
            self.credit_events, ts_ns, kind="monthly_grant",
            change=self.profile.monthly_credits, credits_after=self.credits,
            grant_id=grant_id, account_id=None,
            detail=("initial month's five credits" if initial
                    else "five credits for the new Chicago calendar month"),
        )

    def monthly_grant(self, ts_ns: int, grant_id: str) -> None:
        self._grant(ts_ns, grant_id)
        self._fill_vacancies(ts_ns)

    def _fill_vacancies(self, ts_ns: int) -> None:
        while self.vacancies and self.credits > 0:
            vacancy = self.vacancies.pop(0)
            waited = ts_ns - vacancy["opened_ns"]
            account = self._purchase_with_credit(
                ts_ns, slot=vacancy["slot"], funding="replacement_credit",
                replaces=vacancy["failed_account_id"],
                reason=f"replacement for {vacancy['failed_account_id']}",
            )
            self._row(
                self.account_events, ts_ns, account_id=account.account_id,
                event="vacancy_filled", status_before=None, status_after="ready",
                reason="replacement bought with a monthly credit",
                balance_cents=0, floor_cents=account.floor, slot=vacancy["slot"],
                vacancy_wait_ns=waited, replaces=vacancy["failed_account_id"],
            )

    # ── strategy entries ──────────────────────────────────────────────────
    def on_signal(self, ts_ns: int, execution: StrategyExecution, path: ExecutionPath,
                  schedule_exit) -> None:
        self.missed_for_credit += len(self.vacancies)
        for account in self.accounts:
            if account.status == "failed":
                continue
            if account.created_ns >= ts_ns:
                continue
            if account.status != "ready":
                reason = {
                    "secured": "payout_protection",
                    "processing": "payout_processing",
                    "in_trade": "already_in_position",
                }[account.status]
                account.blocked_signals[reason] = account.blocked_signals.get(reason, 0) + 1
                if reason in ("payout_protection", "payout_processing"):
                    self._row(
                        self.boundary_evidence, ts_ns, account_id=account.account_id,
                        check="entry_blocked_by_payout_policy",
                        outcome="entry rejected", trade_id=execution.trade_id,
                        detail=f"account status: {account.status}",
                    )
                continue
            outcome = simulate_position(
                profile=self.profile,
                marks=AccountMarks(account.balance, account.floor, account.peak),
                execution=execution, path=path, quantity=self.quantity,
                tick_value_cents=self.tick_value_cents,
                cost_per_side_cents=self.cost_per_side_cents,
            )
            account.trades += 1
            account.balance -= outcome.entry_cost_cents
            account.open_trade = {"trade_id": execution.trade_id, "outcome": outcome}
            self._set_status(account, "in_trade", ts_ns, "entered the strategy's new signal",
                             trade_id=execution.trade_id)
            schedule_exit(outcome.exit_ns, self.key, account.account_id)

    # ── exits and failures ────────────────────────────────────────────────
    def on_position_closed(self, ts_ns: int, account_id: str, execution: StrategyExecution,
                           path: ExecutionPath, trading_day: str) -> None:
        account = self._account(account_id)
        outcome: PositionOutcome = account.open_trade["outcome"]
        account.open_trade = None
        balance_before_entry = outcome.balance_before_cents
        account.balance = outcome.balance_after_cents
        account.floor = outcome.floor_after_cents
        account.peak = outcome.peak_after_cents
        net = outcome.balance_after_cents - balance_before_entry
        self._row(
            self.trades, ts_ns, account_id=account.account_id, trade_id=execution.trade_id,
            trading_day=trading_day, direction=execution.direction,
            quantity=outcome.quantity, entry_ns=outcome.entry_ns,
            entry_ticks=outcome.entry_ticks, stop_ticks=execution.stop_ticks,
            strategy_exit_reason=execution.exit_reason,
            strategy_exit_ticks=execution.exit_ticks,
            exit_ns=outcome.exit_ns, exit_ticks=outcome.exit_ticks,
            exit_kind=outcome.exit_kind, gross_pnl_cents=outcome.gross_pnl_cents,
            costs_cents=outcome.entry_cost_cents + outcome.exit_cost_cents,
            net_pnl_cents=net, initial_risk_cents=outcome.initial_risk_cents,
            balance_before_cents=balance_before_entry,
            balance_after_cents=outcome.balance_after_cents,
            floor_before_cents=outcome.floor_before_cents,
            floor_after_cents=outcome.floor_after_cents,
            min_equity_cents=outcome.min_equity_cents,
            min_equity_ns=outcome.min_equity_ns,
            max_equity_cents=outcome.max_equity_cents,
            observations_checked=outcome.observations_checked,
            price_evidence=path.fidelity,
            approximate_price_path=path.is_approximate,
            entry_chart=execution.entry_chart,
            account_failed=outcome.failed,
        )
        for ts, prior, new, peak in outcome.floor_transitions:
            self._row(
                self.boundary_evidence, ts, account_id=account.account_id,
                check="floor_moved", outcome=(
                    "floor locked" if new >= self.profile.floor_lock_cents else "floor raised"
                ),
                trade_id=execution.trade_id, prior_floor_cents=prior,
                new_floor_cents=new, peak_equity_cents=peak,
                detail="intraday peak of realized plus open equity",
            )
        if outcome.failed:
            self._fail(account, ts_ns, outcome, execution, path)
            return
        self._set_status(account, "ready", ts_ns, f"strategy exit ({execution.exit_reason})",
                         trade_id=execution.trade_id)
        self._check_secure(account, ts_ns, trading_day)

    def _fail(self, account: FundedAccount, ts_ns: int, outcome: PositionOutcome,
              execution: StrategyExecution, path: ExecutionPath) -> None:
        stage_text = {
            "at_entry_cost": "the entry cost reached the loss limit",
            "open_position": "open-position equity reached the loss limit",
            "after_exit_cost": "the exit cost took the balance to the loss limit",
        }[outcome.failure_stage]
        account.failed_ns = ts_ns
        account.failure_reason = stage_text
        account.paid_before_failure = account.payouts_received > 0 or (
            account.pending_payout is not None
        )
        for index, (t, px, eq, fl) in enumerate(outcome.evidence):
            self._row(
                self.boundary_evidence, t, account_id=account.account_id,
                check="ordered_observation_before_failure", outcome=(
                    "failure observation" if index == len(outcome.evidence) - 1
                    else "observation"
                ),
                trade_id=execution.trade_id, price_ticks=px, equity_cents=eq,
                floor_cents=fl, detail=path.fidelity,
            )
        self._row(
            self.boundary_evidence, ts_ns, account_id=account.account_id,
            check="account_failure", outcome="account lost",
            trade_id=execution.trade_id, equity_cents=outcome.failure_equity_cents,
            floor_cents=outcome.failure_floor_cents,
            comparator=outcome.failure_comparator,
            price_ticks=outcome.exit_ticks, detail=outcome.failure_fill_basis,
            approximate=path.is_approximate,
        )
        self._set_status(account, "failed", ts_ns, stage_text, trade_id=execution.trade_id,
                         paid_before_failure=account.paid_before_failure)
        vacancy = {"slot": account.slot, "failed_account_id": account.account_id,
                   "opened_ns": ts_ns}
        self.vacancies.append(vacancy)
        self._row(
            self.account_events, ts_ns, account_id=account.account_id,
            event="vacancy_opened", status_before="failed", status_after="failed",
            reason="slot waits for a monthly credit" if self.credits == 0
            else "slot refilled with a monthly credit",
            balance_cents=account.balance, floor_cents=account.floor, slot=account.slot,
        )
        self._fill_vacancies(ts_ns)

    # ── payout lifecycle ──────────────────────────────────────────────────
    def _surplus(self, account: FundedAccount) -> int:
        return max(0, account.balance - self.profile.retained_cushion_cents)

    def _check_secure(self, account: FundedAccount, ts_ns: int, trading_day: str) -> None:
        gross = self._surplus(account)
        if gross >= self.profile.minimum_gross_request_cents:
            account.secured_ns = ts_ns
            account.secured_trading_day = trading_day
            self._set_status(account, "secured", ts_ns,
                             "realized payout eligibility reached; no more entries today",
                             eligible_gross_cents=gross)
            self._row(
                self.payout_events, ts_ns, account_id=account.account_id,
                event="eligibility_secured", trading_day=trading_day,
                realized_balance_cents=account.balance, eligible_gross_cents=gross,
                floor_cents=account.floor,
            )

    def end_of_trading_day(self, ts_ns: int, trading_day: str, cutoff_ns: int) -> None:
        # (a) MyFundedFutures: the closing-balance floor update FIRST
        if self.profile.threshold_update == "session_close_balance":
            for account in self.alive_accounts():
                if account.created_ns > ts_ns:
                    continue
                prior = account.floor
                new = min(self.profile.floor_lock_cents,
                          max(prior, account.balance - self.profile.loss_allowance_cents))
                if new != prior:
                    account.floor = new
                    self._row(
                        self.boundary_evidence, ts_ns, account_id=account.account_id,
                        check="floor_moved", outcome=(
                            "floor locked" if new >= self.profile.floor_lock_cents
                            else "floor raised"
                        ),
                        trade_id=None, prior_floor_cents=prior, new_floor_cents=new,
                        peak_equity_cents=account.balance,
                        detail="scheduled session-close realized balance",
                    )
                # the closing balance itself must respect the (new) floor
                if breaches(account.balance, account.floor,
                            self.profile.comparator_for(account.floor)):
                    raise AssertionError("closed balance below its floor at day end")
        # (b) one full-surplus request per secured account, while flat
        for account in self.alive_accounts():
            if account.status != "secured" or account.secured_trading_day != trading_day:
                continue
            gross = self._surplus(account)
            if gross < self.profile.minimum_gross_request_cents:
                self._row(
                    self.payout_events, ts_ns, account_id=account.account_id,
                    event="request_shortfall", trading_day=trading_day,
                    realized_balance_cents=account.balance, eligible_gross_cents=gross,
                    floor_cents=account.floor,
                )
                account.pending_payout = {"state": "release_next_day"}
                continue
            trader, firm = split_gross(gross, self.profile.trader_share_pct)
            due = processing_due_ns(ts_ns, self.processing)
            balance_before = account.balance
            account.balance -= gross
            account.gross_requested_cents += gross
            self.gross_debited += gross
            self.firm_share_retained += firm
            request_id = f"{account.account_id}-payout-{account.gross_requested_cents}"
            account.pending_payout = {
                "state": "processing", "request_id": request_id, "gross": gross,
                "trader": trader, "firm": firm, "requested_ns": ts_ns, "due_ns": due,
                "secured_ns": account.secured_ns,
            }
            self._row(
                self.payout_events, ts_ns, account_id=account.account_id,
                event="requested", trading_day=trading_day, request_id=request_id,
                secured_ns=account.secured_ns, requested_ns=ts_ns, due_ns=due,
                gross_cents=gross, firm_share_cents=firm, trader_cents=trader,
                balance_before_cents=balance_before, balance_after_cents=account.balance,
                floor_cents=account.floor, clock_basis=self.processing.basis,
                after_cutoff=due > cutoff_ns,
            )
            self._set_status(account, "processing", ts_ns,
                             "full surplus requested at the end of the trading day",
                             request_id=request_id, due_ns=due)

    def next_day_release(self, ts_ns: int) -> None:
        for account in self.alive_accounts():
            if account.status == "secured" and (account.pending_payout or {}).get(
                "state"
            ) == "release_next_day":
                account.pending_payout = None
                account.secured_ns = None
                account.secured_trading_day = None
                self._set_status(account, "ready", ts_ns,
                                 "request not possible; released at the next trading day")

    def processing_complete(self, ts_ns: int, account_id: str, request_id: str) -> None:
        account = self._account(account_id)
        pending = account.pending_payout
        if not pending or pending.get("request_id") != request_id:
            # idempotent: an already-posted receipt is never posted twice
            return
        if account.status != "processing":
            raise AssertionError("processing completion for an account not processing")
        trader = pending["trader"]
        self._cash(ts_ns, "payout_received", amount=trader, funding="payout",
                   account_id=account.account_id,
                   detail=f"after-split payout for request {request_id}")
        account.payouts_received += 1
        account.received_cents += trader
        account.largest_receipt_cents = max(account.largest_receipt_cents, trader)
        self._row(
            self.payout_events, ts_ns, account_id=account.account_id, event="received",
            request_id=request_id, requested_ns=pending["requested_ns"],
            due_ns=pending["due_ns"], received_ns=ts_ns, gross_cents=pending["gross"],
            firm_share_cents=pending["firm"], trader_cents=trader,
            clock_basis=self.processing.basis, secured_ns=pending["secured_ns"],
        )
        account.pending_payout = None
        account.secured_ns = None
        account.secured_trading_day = None
        self._set_status(account, "ready", ts_ns,
                         "processing complete; the payout lock is released",
                         request_id=request_id)

    # ── growth ────────────────────────────────────────────────────────────
    def review_growth(self, ts_ns: int, receipt_day: str) -> None:
        cost = self.profile.cost_of_next_block_cents
        threshold = self.profile.growth_threshold_cents
        wallet_before = self.wallet
        net_before = self.net_cash
        capacity_before = self.capacity
        if receipt_day in self.growth_days:
            decision, reason = "not_reviewed", "one five-account purchase already made today"
        elif self.capacity >= self.profile.max_capacity:
            decision, reason = "declined", "capacity is already at the 20-account maximum"
        elif cost * 10_000 > self.profile.growth_share_bps * self.wallet:
            decision, reason = "declined", "the wallet is below the 25% threshold"
        elif net_before <= 0:
            decision, reason = "declined", "cumulative net cash is not yet positive"
        else:
            decision, reason = "purchased", "the next five cost at most 25% of the wallet"
        if decision == "purchased":
            self.growth_days.append(receipt_day)
            self._cash(ts_ns, "account_purchase", amount=cost, funding="payout_wallet",
                       account_id=None, detail="five-account growth block")
            first_slot = self.capacity
            self.capacity += self.profile.capacity_step
            for slot in range(first_slot, self.capacity):
                self._new_account(ts_ns, slot=slot, funding="growth_wallet", replaces=None)
        self._row(
            self.growth_events, ts_ns, receipt_day=receipt_day, decision=decision,
            reason=reason, wallet_before_cents=wallet_before,
            wallet_after_cents=self.wallet, net_cash_before_cents=net_before,
            net_cash_after_cents=self.net_cash, block_cost_cents=cost,
            threshold_cents=threshold, capacity_before=capacity_before,
            capacity_after=self.capacity,
        )

    # ── bookkeeping ───────────────────────────────────────────────────────
    def _account(self, account_id: str) -> FundedAccount:
        for account in self.accounts:
            if account.account_id == account_id:
                return account
        raise KeyError(account_id)

    def close_durations(self, end_ns: int) -> None:
        for account in self.accounts:
            elapsed = max(0, end_ns - account.status_since_ns)
            account.status_durations_ns[account.status] = (
                account.status_durations_ns.get(account.status, 0) + elapsed
            )
            account.status_since_ns = end_ns

    def snapshot(self) -> dict[str, Any]:
        state = {k: v for k, v in self.__dict__.items()
                 if k not in ("profile", "processing", "accounts")}
        state["accounts"] = []
        for account in self.accounts:
            data = asdict(account)
            if account.open_trade is not None:
                data["open_trade"] = {
                    "trade_id": account.open_trade["trade_id"],
                    "outcome": asdict(account.open_trade["outcome"]),
                }
            state["accounts"].append(data)
        return state

    def restore(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            if key == "accounts":
                continue
            setattr(self, key, value)
        self.accounts = []
        for data in state["accounts"]:
            data = dict(data)
            open_trade = data.pop("open_trade")
            account = FundedAccount(**data)
            if open_trade is not None:
                outcome = dict(open_trade["outcome"])
                outcome["floor_transitions"] = tuple(
                    tuple(item) for item in outcome["floor_transitions"]
                )
                outcome["evidence"] = tuple(tuple(item) for item in outcome["evidence"])
                account.open_trade = {
                    "trade_id": open_trade["trade_id"],
                    "outcome": PositionOutcome(**outcome),
                }
            self.accounts.append(account)


def _month_id(ts_ns: int) -> str:
    from alpha_lab.propsim.funded.clock import chicago_date

    day = chicago_date(ts_ns)
    return f"{day.year:04d}-{day.month:02d}"
