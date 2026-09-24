"""One open position walked minute by minute on ordered observations.

Execution model ``ordered_prints_stop_market_v2`` (the configuration-comparison
mode; the September 22 pilot used ``recorded_strategy_prices_v1`` and is kept
as history):

* **Entry** at the strategy's confirming one-minute candle close, the price
  Strategy-Core records. Half the round-trip cost posts at the entry fill and
  can itself reach the loss floor.
* **Stop** is a stop-market order: it triggers on the first observation at or
  through the stop and fills AT THAT OBSERVATION'S PRICE (a print that gaps
  through the stop fills worse than the stop, never better).
* **Target** is a resting limit order: it fills at the target price on the
  first observation at or through it, never at a better price.
* **Account loss floor** is enforced on every observation against the floor
  active at that observation (TakeProfitTrader's floor rises with the running
  peak of open equity; MyFundedFutures' floor never moves inside a position).
  A breach liquidates at that observation's price. On one observation the
  account breach is resolved first, then the stop, then the target.
* **Scheduled daily close**: a position still open at the end of the deadline
  minute exits at that minute's closing trade.
* The exit cost posts after the exit fill; a balance at or through the floor
  after that cost also loses the account.

Observations are either ordered exchange trade prints (consecutive prints are
jumps: nothing is assumed between them) or, where a minute's prints do not
rebuild the strategy's candle exactly, a declared one-minute approximation
(open, losing extreme, winning extreme, close; legs inside the candle are
treated as continuous). Everything is exact integer cents and ticks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from alpha_lab.propsim.funded.positions import boundary_offset_ticks
from alpha_lab.propsim.funded.profiles import FundedFirmProfile, breaches

__all__ = [
    "EXECUTION_MODEL_ID",
    "EXECUTION_MODEL_TEXT",
    "MinuteObservations",
    "OpenPosition",
    "PositionExit",
    "open_position",
    "walk_minute",
    "minute_approximation",
]

EXECUTION_MODEL_ID = "ordered_prints_stop_market_v2"
EXECUTION_MODEL_TEXT = (
    "Entry at the confirming one-minute candle's close. Stops are stop-market orders "
    "filled at the first recorded trade at or through the stop; targets are limit "
    "orders filled at the target. The account loss limit is checked on every recorded "
    "trade; a breach closes the position at that trade's price. A position still open "
    "at the daily deadline closes at that minute's last trade. Costs post at each fill."
)

Fidelity = Literal["ordered_trade_prints", "minute_adverse_first"]
EVIDENCE_POINTS = 12


@dataclass(frozen=True)
class MinuteObservations:
    """The ordered observations of one one-minute candle."""

    open_ns: int
    close_ns: int
    close_ticks: int
    ts_ns: np.ndarray  # int64
    price_ticks: np.ndarray  # int64
    continuous: np.ndarray  # bool: leg FROM the previous point is continuous
    fidelity: Fidelity


def minute_approximation(open_ns: int, close_ns: int, o: int, h: int, low: int, c: int,
                         sign: int) -> MinuteObservations:
    """Declared approximation: open, losing extreme, winning extreme, close."""

    adverse, favorable = (low, h) if sign > 0 else (h, low)
    step = max(1, (close_ns - open_ns) // 4)
    return MinuteObservations(
        open_ns=open_ns, close_ns=close_ns, close_ticks=c,
        ts_ns=np.array([open_ns, open_ns + step, open_ns + 2 * step, close_ns - 1],
                       dtype=np.int64),
        price_ticks=np.array([o, adverse, favorable, c], dtype=np.int64),
        continuous=np.array([False, True, True, True]),
        fidelity="minute_adverse_first",
    )


@dataclass
class OpenPosition:
    trade_ref: str
    direction: str
    quantity: int
    value_cents: int  # cents per tick for the whole position
    entry_ns: int
    entry_ticks: int
    stop_ticks: int
    target_ticks: int
    entry_cost_cents: int
    exit_cost_cents: int
    balance_before_cents: int
    balance_cents: int  # after the entry cost
    floor_before_cents: int
    floor_cents: int
    peak_cents: int
    min_equity_cents: int
    min_equity_ns: int
    max_equity_cents: int
    max_equity_ns: int
    observations: int = 0
    minutes_on_prints: int = 0
    minutes_approximated: int = 0
    floor_transitions: list = field(default_factory=list)  # (ts, prior, new, peak)
    recent: deque = field(default_factory=lambda: deque(maxlen=EVIDENCE_POINTS))
    # exact per-contract cost in tenths of a cent (0 = legacy per-side cents only)
    cost_per_contract_mills: int = 0
    tick_value_cents: int = 0
    # scale-out exit: ``scale_quantity`` contracts exit at the target, the stop of
    # the rest moves to the entry price (break-even); the rest is held
    scale_out: bool = False
    scale_quantity: int = 0
    scaled: bool = False
    remaining_quantity: int = 0
    initial_stop_ticks: int | None = None
    scale_out_ns: int | None = None
    scale_out_ticks: int | None = None
    partial_gross_cents: int = 0
    partial_cost_cents: int = 0

    @property
    def sign(self) -> int:
        return 1 if self.direction == "long" else -1

    @property
    def initial_risk_cents(self) -> int:
        stop = self.initial_stop_ticks if self.scaled else self.stop_ticks
        per_tick = (self.tick_value_cents * self.quantity if self.tick_value_cents
                    else self.value_cents)
        return abs(self.entry_ticks - stop) * per_tick

    def fill_cost_cents(self, quantity: int) -> int:
        return fill_cost_cents(quantity, self.cost_per_contract_mills)

    def to_json(self) -> dict:
        data = dict(self.__dict__)
        data["floor_transitions"] = [list(t) for t in self.floor_transitions]
        data["recent"] = [list(t) for t in self.recent]
        return data

    @classmethod
    def from_json(cls, data: dict) -> OpenPosition:
        data = dict(data)
        transitions = [tuple(t) for t in data.pop("floor_transitions")]
        recent = deque((tuple(t) for t in data.pop("recent")), maxlen=EVIDENCE_POINTS)
        return cls(**data, floor_transitions=transitions, recent=recent)


def fill_cost_cents(quantity: int, mills: int) -> int:
    """Exact cost of one fill: quantity x cost per contract (tenths of a cent)."""

    total = quantity * mills
    if total % 10:
        raise ValueError(f"a fill of {quantity} contracts at {mills / 1000:.3f} dollars per "
                         "contract is not a whole number of cents")
    return total // 10


@dataclass(frozen=True)
class PositionExit:
    kind: str  # stop | target | breakeven_stop | scheduled_close | account_failure
    ts_ns: int
    fill_ticks: int
    gross_pnl_cents: int
    balance_after_cents: int  # after the exit cost
    floor_cents: int
    peak_cents: int
    fill_basis: str
    approximate: bool
    account_failed: bool
    failure_stage: str | None  # at_entry_cost | open_position | after_exit_cost
    failure_equity_cents: int | None
    failure_comparator: str | None
    evidence: tuple = ()  # (ts, price, equity, floor) before and at the exit
    # scale-out: the partial exit before the final one (gross includes both)
    scale_out_ns: int | None = None
    scale_out_ticks: int | None = None
    scale_out_quantity: int = 0
    scaled_in_exit_minute: bool = False


def open_position(*, profile: FundedFirmProfile, trade_ref: str, direction: str,
                  entry_ns: int, entry_ticks: int, stop_ticks: int, target_ticks: int,
                  quantity: int, tick_value_cents: int, cost_per_side_cents: int,
                  balance_cents: int, floor_cents: int, peak_cents: int,
                  cost_per_contract_mills: int | None = None, scale_out: bool = False
                  ) -> tuple[OpenPosition, PositionExit | None]:
    """Open at the entry fill; the entry cost can itself fail the account.

    ``cost_per_contract_mills`` (tenths of a cent per contract per fill) replaces
    ``cost_per_side_cents`` when given. ``scale_out`` exits half the contracts at
    the target and holds the rest at break-even (an even quantity is required).
    """

    mills = (cost_per_contract_mills if cost_per_contract_mills is not None
             else cost_per_side_cents * 10)
    if scale_out and (quantity < 2 or quantity % 2):
        raise ValueError("the scale-out exit needs an even number of contracts")
    cost = fill_cost_cents(quantity, mills)
    balance = balance_cents - cost
    position = OpenPosition(
        trade_ref=trade_ref, direction=direction, quantity=quantity,
        value_cents=tick_value_cents * quantity, entry_ns=entry_ns,
        entry_ticks=entry_ticks, stop_ticks=stop_ticks, target_ticks=target_ticks,
        entry_cost_cents=cost, exit_cost_cents=cost,
        balance_before_cents=balance_cents, balance_cents=balance,
        floor_before_cents=floor_cents, floor_cents=floor_cents, peak_cents=peak_cents,
        min_equity_cents=balance, min_equity_ns=entry_ns,
        max_equity_cents=balance, max_equity_ns=entry_ns,
        cost_per_contract_mills=mills, tick_value_cents=tick_value_cents,
        scale_out=scale_out, scale_quantity=quantity // 2 if scale_out else 0,
        remaining_quantity=quantity,
    )
    comparator = profile.comparator_for(floor_cents)
    if breaches(balance, floor_cents, comparator):
        after = balance - cost
        return position, PositionExit(
            kind="account_failure", ts_ns=entry_ns, fill_ticks=entry_ticks,
            gross_pnl_cents=0, balance_after_cents=after, floor_cents=floor_cents,
            peak_cents=peak_cents,
            fill_basis="closed at the entry price when the entry cost reached the loss limit",
            approximate=False, account_failed=True, failure_stage="at_entry_cost",
            failure_equity_cents=balance, failure_comparator=comparator,
            evidence=((entry_ns, entry_ticks, balance, floor_cents),),
        )
    return position, None


def _floor_path(profile: FundedFirmProfile, pos: OpenPosition, equity: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    if profile.threshold_update == "intraday_peak_equity":
        peaks = np.maximum.accumulate(np.maximum(equity, pos.peak_cents))
        floors = np.minimum(profile.floor_lock_cents,
                            np.maximum(pos.floor_cents, peaks - profile.loss_allowance_cents))
    else:
        peaks = np.full(len(equity), pos.peak_cents, dtype=np.int64)
        floors = np.full(len(equity), pos.floor_cents, dtype=np.int64)
    return peaks, floors


def _breach_mask(profile: FundedFirmProfile, equity: np.ndarray, floors: np.ndarray
                 ) -> np.ndarray:
    locked = floors >= profile.floor_lock_cents
    inclusive = np.where(locked, profile.comparator_after_lock == "at_or_below",
                         profile.comparator_before_lock == "at_or_below")
    return np.where(inclusive, equity <= floors, equity < floors)


def _record_floor_moves(pos: OpenPosition, ts: np.ndarray, peaks: np.ndarray,
                        floors: np.ndarray, upto: int, lock: int) -> None:
    prior = pos.floor_cents
    seg = floors[: upto + 1]
    for index in np.flatnonzero(np.diff(np.concatenate(([prior], seg)))):
        new = int(seg[index])
        # keep the review evidence compact: first rise, the lock, the last value
        if pos.floor_transitions and new < lock and len(pos.floor_transitions) >= 2:
            pos.floor_transitions[-1] = (int(ts[index]), pos.floor_transitions[-1][1], new,
                                         int(peaks[index]))
        else:
            pos.floor_transitions.append((int(ts[index]), prior, new, int(peaks[index])))
        prior = new


def walk_minute(*, profile: FundedFirmProfile, pos: OpenPosition, obs: MinuteObservations,
                deadline_minute: bool, _count_minute: bool = True) -> PositionExit | None:
    """Advance the open position through one minute; return its exit, if any."""

    sign = pos.sign
    px = obs.price_ticks
    n = len(px)
    if _count_minute and obs.fidelity == "ordered_trade_prints":
        pos.minutes_on_prints += 1
    elif _count_minute:
        pos.minutes_approximated += 1
    if n == 0:
        if deadline_minute:
            return _close_at(profile, pos, "scheduled_close", obs.close_ns, obs.close_ticks,
                             "the deadline minute's last recorded trade", obs)
        return None
    equity = pos.balance_cents + (px - pos.entry_ticks) * sign * pos.value_cents
    peaks, floors = _floor_path(profile, pos, equity)
    breach = _breach_mask(profile, equity, floors)
    stop_hit = px * sign <= pos.stop_ticks * sign
    # after the scale-out there is no target: the rest is held to break-even or the close
    target_hit = (px * sign >= pos.target_ticks * sign) if not pos.scaled else np.zeros(
        n, dtype=bool)
    any_hit = breach | stop_hit | target_hit
    if not bool(any_hit.any()):
        _absorb(pos, obs, equity, peaks, floors, n - 1, profile.floor_lock_cents)
        if deadline_minute:
            return _close_at(profile, pos, "scheduled_close", obs.close_ns, obs.close_ticks,
                             "the deadline minute's last recorded trade", obs)
        return None
    i = int(np.argmax(any_hit))
    kind, fill, basis = _resolve_hit(profile, pos, obs, equity, floors, breach, stop_hit,
                                     target_hit, i)
    if kind == "target" and pos.scale_out and not pos.scaled:
        return _scale_out_then_continue(profile, pos, obs, i, fill, deadline_minute)
    if kind == "stop" and pos.scaled:
        kind = "breakeven_stop"
        basis = basis.replace("the stop", "the break-even stop (the entry price)")
    # re-mark the triggering observation at the actual fill price
    eff = px[: i + 1].copy()
    eff[i] = fill
    equity = pos.balance_cents + (eff - pos.entry_ticks) * sign * pos.value_cents
    peaks, floors = _floor_path(profile, pos, equity)
    evidence = _evidence(pos, obs, eff, equity, floors, i)
    _absorb(pos, obs, equity, peaks, floors, i, profile.floor_lock_cents)
    fill_equity = int(equity[i])
    floor_at = int(floors[i])
    comparator = profile.comparator_for(floor_at)
    if kind == "account_failure":
        return PositionExit(
            kind=kind, ts_ns=int(obs.ts_ns[i]), fill_ticks=fill,
            gross_pnl_cents=(fill - pos.entry_ticks) * sign * pos.value_cents
            + pos.partial_gross_cents,
            balance_after_cents=fill_equity - pos.exit_cost_cents, floor_cents=floor_at,
            peak_cents=int(peaks[i]), fill_basis=basis,
            approximate=obs.fidelity != "ordered_trade_prints", account_failed=True,
            failure_stage="open_position", failure_equity_cents=fill_equity,
            failure_comparator=comparator, evidence=evidence, **_scale_fields(pos),
        )
    return _finish(profile, pos, kind, int(obs.ts_ns[i]), fill, fill_equity, floor_at,
                   int(peaks[i]), basis, obs.fidelity != "ordered_trade_prints", evidence)


def _resolve_hit(profile, pos, obs, equity, floors, breach, stop_hit, target_hit, i):
    sign = pos.sign
    px = obs.price_ticks
    continuous = bool(obs.continuous[i]) and i > 0
    if not continuous:
        # a jump: the observation's own price is the first executable price
        if breach[i]:
            return ("account_failure", int(px[i]),
                    "first recorded trade at or through the loss limit")
        if stop_hit[i]:
            return ("stop", int(px[i]),
                    "stop-market: first recorded trade at or through the stop"
                    if obs.fidelity == "ordered_trade_prints"
                    else "stop-market: the candle opened through the stop (approximation)")
        return ("target", pos.target_ticks, "limit order filled at the target price")
    # a continuous approximated leg: the first level met on the way
    prev = int(px[i - 1])
    if target_hit[i] and not stop_hit[i] and not breach[i]:
        return ("target", pos.target_ticks,
                "limit order filled at the target price (candle approximation)")
    k = boundary_offset_ticks(int(floors[i]) - pos.balance_cents, pos.value_cents,
                              profile.comparator_for(int(floors[i])))
    breach_px = pos.entry_ticks + sign * k
    lo, hi = sorted((prev, int(px[i])))
    breach_on_leg = bool(breach[i]) and lo <= breach_px <= hi
    stop_on_leg = bool(stop_hit[i])
    if breach_on_leg and (not stop_on_leg or breach_px * sign >= pos.stop_ticks * sign):
        return ("account_failure", breach_px,
                "assumed continuous candle leg reaches the loss limit (approximation)")
    if stop_on_leg:
        return ("stop", pos.stop_ticks,
                "assumed continuous candle leg reaches the stop (approximation)")
    return ("account_failure", int(px[i]),
            "candle point at or through the loss limit (approximation)")


def _evidence(pos, obs, eff, equity, floors, i):
    start = max(0, i - EVIDENCE_POINTS + 1)
    tail = [(int(obs.ts_ns[j]), int(eff[j]), int(equity[j]), int(floors[j]))
            for j in range(start, i + 1)]
    need = EVIDENCE_POINTS - len(tail)
    earlier = list(pos.recent)[-need:] if need > 0 else []
    return tuple([*earlier, *tail])


def _absorb(pos: OpenPosition, obs: MinuteObservations, equity, peaks, floors, upto: int,
            lock: int) -> None:
    seg = equity[: upto + 1]
    if len(seg):
        imin, imax = int(np.argmin(seg)), int(np.argmax(seg))
        if int(seg[imin]) < pos.min_equity_cents:
            pos.min_equity_cents, pos.min_equity_ns = int(seg[imin]), int(obs.ts_ns[imin])
        if int(seg[imax]) > pos.max_equity_cents:
            pos.max_equity_cents, pos.max_equity_ns = int(seg[imax]), int(obs.ts_ns[imax])
        _record_floor_moves(pos, obs.ts_ns, peaks, floors, upto, lock)
        pos.floor_cents = int(floors[upto])
        pos.peak_cents = int(peaks[upto])
        start = max(0, upto + 1 - EVIDENCE_POINTS)
        for j in range(start, upto + 1):
            pos.recent.append((int(obs.ts_ns[j]), int(obs.price_ticks[j]), int(equity[j]),
                               int(floors[j])))
    pos.observations += upto + 1


def _close_at(profile, pos, kind, ts_ns, fill, basis, obs) -> PositionExit:
    sign = pos.sign
    fill_equity = pos.balance_cents + (fill - pos.entry_ticks) * sign * pos.value_cents
    peak, floor = pos.peak_cents, pos.floor_cents
    if profile.threshold_update == "intraday_peak_equity" and fill_equity > peak:
        prior = floor
        peak = fill_equity
        floor = min(profile.floor_lock_cents, max(floor, peak - profile.loss_allowance_cents))
        if floor != prior:
            pos.floor_transitions.append((ts_ns, prior, floor, peak))
    return _finish(profile, pos, kind, ts_ns, fill, fill_equity, floor, peak, basis,
                   obs.fidelity != "ordered_trade_prints", tuple(pos.recent))


def _finish(profile, pos, kind, ts_ns, fill, fill_equity, floor, peak, basis, approximate,
            evidence) -> PositionExit:
    sign = pos.sign
    after = fill_equity - pos.exit_cost_cents
    comparator = profile.comparator_for(floor)
    at_fill = breaches(fill_equity, floor, comparator)
    after_cost = breaches(after, floor, comparator)
    failed = at_fill or after_cost
    return PositionExit(
        kind=kind, ts_ns=ts_ns, fill_ticks=fill,
        gross_pnl_cents=(fill - pos.entry_ticks) * sign * pos.value_cents
        + pos.partial_gross_cents,
        balance_after_cents=after, floor_cents=floor, peak_cents=peak, fill_basis=basis,
        approximate=approximate, account_failed=failed,
        failure_stage=None if not failed else (
            "open_position" if at_fill else "after_exit_cost"),
        failure_equity_cents=None if not failed else (fill_equity if at_fill else after),
        failure_comparator=comparator if failed else None, evidence=evidence,
        **_scale_fields(pos),
    )


def _scale_fields(pos: OpenPosition) -> dict:
    if not pos.scaled:
        return {}
    return {"scale_out_ns": pos.scale_out_ns, "scale_out_ticks": pos.scale_out_ticks,
            "scale_out_quantity": pos.scale_quantity}


def _scale_out_then_continue(profile: FundedFirmProfile, pos: OpenPosition,
                             obs: MinuteObservations, i: int, fill: int,
                             deadline_minute: bool) -> PositionExit | None:
    """Half exits at the target (limit price); the rest continues this minute."""

    sign = pos.sign
    eff = obs.price_ticks[: i + 1].copy()
    eff[i] = fill
    equity = pos.balance_cents + (eff - pos.entry_ticks) * sign * pos.value_cents
    peaks, floors = _floor_path(profile, pos, equity)
    _absorb(pos, obs, equity, peaks, floors, i, profile.floor_lock_cents)
    quantity = pos.scale_quantity
    gross = (fill - pos.entry_ticks) * sign * pos.tick_value_cents * quantity
    cost = pos.fill_cost_cents(quantity)
    pos.balance_cents += gross - cost
    pos.partial_gross_cents = gross
    pos.partial_cost_cents = cost
    pos.remaining_quantity -= quantity
    pos.value_cents = pos.tick_value_cents * pos.remaining_quantity
    pos.exit_cost_cents = pos.fill_cost_cents(pos.remaining_quantity)
    pos.scaled = True
    pos.initial_stop_ticks = pos.stop_ticks
    pos.stop_ticks = pos.entry_ticks
    pos.scale_out_ns = int(obs.ts_ns[i])
    pos.scale_out_ticks = fill
    # the rest of this minute starts at the fill point, where the loss limit is
    # checked again after the partial exit cost
    # the observation that crossed the target (a print beyond it, or the candle's
    # extreme) still counts for the remaining contracts' open equity and peak
    rest = MinuteObservations(
        open_ns=obs.open_ns, close_ns=obs.close_ns, close_ticks=obs.close_ticks,
        ts_ns=np.concatenate(([obs.ts_ns[i]], obs.ts_ns[i:])).astype(np.int64),
        price_ticks=np.concatenate(([fill], obs.price_ticks[i:])).astype(np.int64),
        continuous=np.concatenate(([False], obs.continuous[i:])).astype(bool),
        fidelity=obs.fidelity)
    result = walk_minute(profile=profile, pos=pos, obs=rest, deadline_minute=deadline_minute,
                         _count_minute=False)
    return None if result is None else replace(result, scaled_in_exit_minute=True)
