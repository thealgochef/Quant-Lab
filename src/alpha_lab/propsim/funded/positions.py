"""One account's position on one ordered path — the open-equity loss check.

Two controls are kept separate (specification section 6):

* **Update** — WHEN the floor moves. TakeProfitTrader: at every observation,
  from the running peak of current equity (realized + open), capped at its lock.
  MyFundedFutures: never inside a position; only at the scheduled session close
  (handled by the instance, not here).
* **Enforce** — on EVERY ordered observation while the position exists, current
  equity (posted balance + open profit/loss) is compared with the floor that is
  active at that observation. Both firms can fail during an open trade.

Costs post at their fills: half the round trip at entry (before the first
observation) and half at the exit fill. Future costs are never pre-deducted.
Everything is exact integer cents and ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alpha_lab.propsim.funded.paths import (
    OBS_EXIT_TOUCH,
    ExecutionPath,
    StrategyExecution,
)
from alpha_lab.propsim.funded.profiles import FundedFirmProfile, breaches

__all__ = ["AccountMarks", "PositionOutcome", "simulate_position", "boundary_offset_ticks"]

#: how many ordered observations before a failure are kept as review evidence
BREACH_EVIDENCE_POINTS = 12


@dataclass(frozen=True)
class AccountMarks:
    """The account coordinates a position starts from (relative cents)."""

    balance_cents: int
    floor_cents: int
    peak_cents: int


@dataclass(frozen=True)
class PositionOutcome:
    trade_id: str
    quantity: int
    entry_ns: int
    entry_ticks: int
    entry_cost_cents: int
    exit_ns: int
    exit_ticks: int
    exit_cost_cents: int
    exit_kind: str  # stop | target | scheduled_close | account_failure
    gross_pnl_cents: int
    balance_before_cents: int
    balance_after_cents: int
    floor_before_cents: int
    floor_after_cents: int
    peak_after_cents: int
    failed: bool
    failure_stage: str | None  # at_entry_cost | open_position | after_exit_cost
    failure_ns: int | None
    failure_equity_cents: int | None
    failure_floor_cents: int | None
    failure_comparator: str | None
    failure_fill_basis: str | None
    min_equity_cents: int
    min_equity_ns: int
    max_equity_cents: int
    max_equity_ns: int
    observations_checked: int
    initial_risk_cents: int
    floor_transitions: tuple[tuple[int, int, int, int], ...]  # ts, prior, new, peak
    evidence: tuple[tuple[int, int, int, int], ...] = field(default=())  # ts, px, eq, floor
    strategy_exit_overridden: bool = False


def boundary_offset_ticks(room_cents: int, value_cents: int, comparator: str) -> int:
    """Largest favorable offset k (ticks) at which the account is in breach.

    With ``room = floor - balance`` and ``value`` = cents per tick for the
    whole position, equity at offset k is ``balance + k * value``; the account
    breaches at every offset ``<= k``.
    """

    if comparator == "at_or_below":
        return room_cents // value_cents
    return -((-room_cents) // value_cents) - 1


def _floor_after_peak(profile: FundedFirmProfile, floor: int, peak: int) -> int:
    return min(profile.floor_lock_cents, max(floor, peak - profile.loss_allowance_cents))


def simulate_position(
    *,
    profile: FundedFirmProfile,
    marks: AccountMarks,
    execution: StrategyExecution,
    path: ExecutionPath,
    quantity: int,
    tick_value_cents: int,
    cost_per_side_cents: int,
) -> PositionOutcome:
    sign = execution.sign
    value = tick_value_cents * quantity
    entry = execution.entry_ticks
    entry_ns = _entry_ns(execution)
    entry_cost = cost_per_side_cents * quantity
    exit_cost = cost_per_side_cents * quantity
    intraday = profile.threshold_update == "intraday_peak_equity"
    initial_risk = abs(entry - execution.stop_ticks) * value

    balance = marks.balance_cents - entry_cost
    floor = marks.floor_cents
    peak = marks.peak_cents
    transitions: list[tuple[int, int, int, int]] = []

    def fail_outcome(**kw) -> PositionOutcome:
        base = dict(
            trade_id=execution.trade_id, quantity=quantity, entry_ns=entry_ns,
            entry_ticks=entry, entry_cost_cents=entry_cost,
            exit_cost_cents=exit_cost, balance_before_cents=marks.balance_cents,
            floor_before_cents=marks.floor_cents, failed=True,
            initial_risk_cents=initial_risk, floor_transitions=tuple(transitions),
        )
        base.update(kw)
        return PositionOutcome(**base)

    # 1. entry cost posts at the entry fill; it can itself breach the floor
    comparator = profile.comparator_for(floor)
    if breaches(balance, floor, comparator):
        exit_ticks = entry
        after = balance - exit_cost
        return fail_outcome(
            exit_ns=entry_ns, exit_ticks=exit_ticks, exit_kind="account_failure",
            gross_pnl_cents=0, balance_after_cents=after, floor_after_cents=floor,
            peak_after_cents=peak, failure_stage="at_entry_cost", failure_ns=entry_ns,
            failure_equity_cents=balance, failure_floor_cents=floor,
            failure_comparator=comparator,
            failure_fill_basis="closed at the entry price when the entry cost reached the floor",
            min_equity_cents=balance, min_equity_ns=entry_ns,
            max_equity_cents=balance, max_equity_ns=entry_ns, observations_checked=0,
        )

    prices = path.price_ticks
    n = len(prices)
    if n and execution.exit_reason == "target" and int(path.kind[-1]) == OBS_EXIT_TOUCH:
        # a target is a resting limit: the account is marked at the target
        # price on the print that fills it, never at a better print price
        prices = prices.copy()
        prices[-1] = execution.exit_ticks
    offsets = (prices - entry) * sign
    equity = balance + offsets * value

    # 2. floor path while the position is open (update control)
    if intraday and n:
        running_peak = np.maximum.accumulate(np.maximum(equity, peak))
        floors = np.minimum(
            profile.floor_lock_cents,
            np.maximum(floor, running_peak - profile.loss_allowance_cents),
        )
    else:
        running_peak = np.full(n, peak, dtype=np.int64)
        floors = np.full(n, floor, dtype=np.int64)

    # 3. enforcement on every observation against the ACTIVE floor
    if n:
        locked = floors >= profile.floor_lock_cents
        inclusive_before = profile.comparator_before_lock == "at_or_below"
        inclusive_after = profile.comparator_after_lock == "at_or_below"
        inclusive = np.where(locked, inclusive_after, inclusive_before)
        breached = np.where(inclusive, equity <= floors, equity < floors)
        hit = int(np.argmax(breached)) if bool(breached.any()) else -1
    else:
        hit = -1

    last_checked = hit if hit >= 0 else n - 1
    if n:
        seg_eq = equity[: last_checked + 1]
        imin = int(np.argmin(seg_eq))
        imax = int(np.argmax(seg_eq))
        min_eq, min_ns = int(seg_eq[imin]), int(path.ts_ns[imin])
        max_eq, max_ns = int(seg_eq[imax]), int(path.ts_ns[imax])
    else:
        min_eq = max_eq = balance
        min_ns = max_ns = entry_ns

    if intraday and n:
        seg_floors = floors[: last_checked + 1]
        changes = np.flatnonzero(np.diff(np.concatenate(([floor], seg_floors))))
        prior = floor
        for index in changes:
            new = int(seg_floors[index])
            transitions.append((int(path.ts_ns[index]), prior, new, int(running_peak[index])))
            prior = new
        # keep the review evidence compact: first rise, lock event and last value
        if len(transitions) > 3:
            locks = [t for t in transitions if t[2] >= profile.floor_lock_cents][:1]
            keep = [transitions[0], *locks, transitions[-1]]
            dedup: list[tuple[int, int, int, int]] = []
            for item in keep:
                if item not in dedup:
                    dedup.append(item)
            transitions[:] = dedup

    if hit >= 0:
        floor_at = int(floors[hit])
        peak_at = int(running_peak[hit])
        comparator = profile.comparator_for(floor_at)
        fill_ticks = int(prices[hit])
        basis = "first recorded trade print at or through the floor (gap-through fill)"
        if bool(path.continuous[hit]) and hit > 0:
            # continuous leg: the floor-equivalent price is reached on the way
            k = boundary_offset_ticks(floor_at - balance, value, comparator)
            candidate = entry + sign * k
            lo, hi = sorted((int(prices[hit - 1]), int(prices[hit])))
            if lo <= candidate <= hi:
                fill_ticks = candidate
                basis = "assumed continuous candle leg reaches the floor price (approximation)"
        elif bool(path.continuous[hit]) and hit == 0:
            basis = "first assumed candle point at or through the floor (approximation)"
        fill_equity = balance + (fill_ticks - entry) * sign * value
        after = fill_equity - exit_cost
        start = max(0, hit - BREACH_EVIDENCE_POINTS + 1)
        evidence = tuple(
            (int(path.ts_ns[i]), int(prices[i]), int(equity[i]), int(floors[i]))
            for i in range(start, hit + 1)
        )
        return fail_outcome(
            exit_ns=int(path.ts_ns[hit]), exit_ticks=fill_ticks,
            exit_kind="account_failure",
            gross_pnl_cents=(fill_ticks - entry) * sign * value,
            balance_after_cents=after, floor_after_cents=floor_at,
            peak_after_cents=max(peak_at, peak), failure_stage="open_position",
            failure_ns=int(path.ts_ns[hit]), failure_equity_cents=int(equity[hit]),
            failure_floor_cents=floor_at, failure_comparator=comparator,
            failure_fill_basis=basis, min_equity_cents=min(min_eq, fill_equity),
            min_equity_ns=min_ns if min_eq <= fill_equity else int(path.ts_ns[hit]),
            max_equity_cents=max_eq, max_equity_ns=max_ns,
            observations_checked=hit + 1, evidence=evidence,
            strategy_exit_overridden=True,
        )

    # 4. the strategy's own exit fills; then its cost posts
    if n:
        floor = int(floors[-1])
        peak = int(running_peak[-1])
    exit_ns = path.strategy_exit_ns
    exit_ticks = execution.exit_ticks
    gross = (exit_ticks - entry) * sign * value
    fill_equity = balance + gross
    if intraday and fill_equity > peak:
        prior = floor
        peak = fill_equity
        floor = _floor_after_peak(profile, floor, peak)
        if floor != prior:
            transitions.append((exit_ns, prior, floor, peak))
    comparator = profile.comparator_for(floor)
    after = fill_equity - exit_cost
    common = dict(
        exit_ns=exit_ns, exit_ticks=exit_ticks, gross_pnl_cents=gross,
        balance_after_cents=after, floor_after_cents=floor, peak_after_cents=peak,
        min_equity_cents=min(min_eq, fill_equity), min_equity_ns=min_ns,
        max_equity_cents=max(max_eq, fill_equity), max_equity_ns=max_ns,
        observations_checked=n,
    )
    if breaches(fill_equity, floor, comparator) or breaches(after, floor, comparator):
        stage = "open_position" if breaches(fill_equity, floor, comparator) else "after_exit_cost"
        return fail_outcome(
            exit_kind=execution.exit_reason, failure_stage=stage, failure_ns=exit_ns,
            failure_equity_cents=after if stage == "after_exit_cost" else fill_equity,
            failure_floor_cents=floor, failure_comparator=comparator,
            failure_fill_basis=(
                "the exit cost took the closed balance to the floor"
                if stage == "after_exit_cost"
                else "the strategy's exit fill price is at or through the floor"
            ),
            **common,
        )
    return PositionOutcome(
        trade_id=execution.trade_id, quantity=quantity, entry_ns=entry_ns,
        entry_ticks=entry, entry_cost_cents=entry_cost, exit_cost_cents=exit_cost,
        exit_kind=execution.exit_reason, balance_before_cents=marks.balance_cents,
        floor_before_cents=marks.floor_cents, failed=False, failure_stage=None,
        failure_ns=None, failure_equity_cents=None, failure_floor_cents=None,
        failure_comparator=None, failure_fill_basis=None,
        initial_risk_cents=initial_risk, floor_transitions=tuple(transitions),
        **common,
    )


def _entry_ns(execution: StrategyExecution) -> int:
    from alpha_lab.propsim.funded.clock import to_ns

    return to_ns(execution.entry_ts_utc)


def exit_touch_index(path: ExecutionPath) -> int | None:
    kinds = np.flatnonzero(path.kind == OBS_EXIT_TOUCH)
    return int(kinds[0]) if len(kinds) else None
