"""Conditional resampling of recorded trades with one firm's ledger rules.

Model ``conditional_firm_ledger_resampling_v1`` (:data:`MODEL_ID`; analytical
corrections A3 and A4, September 25, 2026; formerly "payout race — full
version"). The fixed closed-profit boundary diagnostic
(:func:`.resampling.payout_race`) counts which of two fixed cumulative
closed-profit levels a path crosses first. This model feeds the SAME resampled
draws of recorded trades through the existing one-account ledger of the funded
simulator (:class:`alpha_lab.propsim.funded.pair_ledger.PairLedger`), so the
selected firm's saved trailing or session-close floor, floor lock, payout
protection ("finished for today"), end-of-day request, processing clock and
replacement purchases apply. Only the ledger's public interface is called; no
firm rule is copied here and no simulator code is changed. Firm terms, the
processing clock and the sizing come from the saved result's frozen settings;
the trading calendar from the study's verified strategy package. The ledger's
greedy withdrawal policy (request the full surplus at the first trading-day end
after eligibility) is never changed by anything typed on a screen.

It is CONDITIONAL, not an exact fresh-account model at either firm
(:data:`LIMITATIONS`): the draws reuse this configuration's recorded funded
trades (already shaped by the historical accounts' entry selection and skipped
opportunities), in the study's fixed trade slots; trades cut short when an
account was lost keep their shortened results; and each trade is compressed to
its entry, lowest, highest and exit points, so a reversal between them is not
represented. The historical-order check (:func:`check_original`) confirms that
the adapter reproduces the saved order's five summary figures (net cash, payouts
received, payout count, accounts bought, account costs), its number of trades, and
each trade's net result and account-loss flag in order; it compares no fill times,
prices or quantities, account assignment, payout or failure times, or setup lineage,
and it does not show that other sampled orders, or a fresh account running the full
strategy, are modelled exactly.

How a saved trade becomes a synthetic trade (the adapter)
---------------------------------------------------------

The study's funded trades are the "slots": each keeps its own entry instant,
exit instant and trading day, in the original order. A resampled order puts
the recorded OUTCOME of one saved trade into each slot, as a short ordered price
path inside the trade's own time window:

* entry at the recorded entry price, exit at the recorded exit price (the path
  closes there at the slot's exit instant, the way the simulator closes a
  position at its daily deadline);
* the recorded lowest equity during the trade becomes a price point (the
  "worst point") and the recorded highest equity another (the "best point"),
  converted back to whole ticks with the trade's own quantity, tick value and
  costs;
* a half-exit trade touches its recorded 1R target, where the simulator's own
  scale-out exits half the contracts and moves the stop of the rest to the
  entry price;
* the order's stop sits one tick beyond the worst point and its target one tick
  beyond the best point, so neither triggers by itself: the recorded exit price
  is kept. The loss limit is still checked by the simulator on every point.

Which came first, the worst or the best point, is saved only partly (the time of
the lowest equity is saved, the time of the highest is not). Where the record
or the stored one-minute bars pin the order it is used; where they do not, the
best point is placed first. That placement is a fixed convention, not a proven
best or worst case: the intermediate observations between the stored points are
not saved, so a reversal between them is not represented. For example, with a
floor that trails intraday peak equity and locks at $0, a trade marked
$0 → −$1,000 → +$2,500 → −$100 → +$4,000 → +$3,000 fails on the reversal to
−$100, while one marked $0 → −$1,000 → +$2,500 → +$2,000 → +$4,000 → +$3,000
survives; both compress to the same four points (and the same order of the low
and the high), so no compressed placement can tell them apart. A firm whose
floor moves only at the session close still depends on the recorded-trade
selection, fixed slots and shortened results, so its figures are conditional
too. A trade that ended because an account was lost keeps its recorded
(truncated) result, like every other resampled figure.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
    FundedStudy,
    ordered_trades,
    pair_key,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.resampling import (
    resample_paths,
    share_below_ties_half,
)

__all__ = [
    "DEFAULT_FULL_PATHS",
    "FULL_PATH_CHOICES",
    "LIMITATIONS",
    "MODEL_ID",
    "FirmRules",
    "FullRace",
    "LedgerOutcome",
    "PathOutcome",
    "Slot",
    "TradeShape",
    "Validation",
    "build_inputs",
    "check_original",
    "extension_seed",
    "MinuteIndex",
    "open_source",
    "firm_rules",
    "firm_terms",
    "full_race",
    "horizon_days",
    "ledger_outcome",
    "path_outcome",
    "race_from_outcomes",
    "race_orders",
    "replay",
    "shared_trades",
    "slots_from_rows",
    "terms_digest",
    "trade_shape",
    "validate_original_order",
]

#: the model's identity: a cached or exported figure is only this model's when it
#: carries this id (correction A3)
MODEL_ID = "conditional_firm_ledger_resampling_v1"
#: what the model does not represent (correction A3); plain sentences for exports
LIMITATIONS: tuple[str, ...] = (
    "Source selection: every path redraws this configuration's recorded funded trades, "
    "which already reflect the historical accounts' entry selection and the opportunities "
    "they skipped.",
    "Fixed slots: each redrawn trade is placed in one of the study's fixed trade slots "
    "(the recorded entry time, exit time and trading day), not where the strategy would "
    "have traded on a fresh account.",
    "Shortened liquidation outcomes: a trade cut short when a historical account was lost "
    "keeps its shortened result.",
    "Compressed intratrade path: each trade is reduced to its entry, lowest, highest and "
    "exit points, so a reversal between them, a missing intermediate observation, is not "
    "represented.",
)

MINUTE_NS = 60_000_000_000
DAY_NS = 86_400_000_000_000
#: the ledger run is slower than the fixed-boundary diagnostic: fewer paths by default
DEFAULT_FULL_PATHS = 1_000
FULL_PATH_CHOICES = (500, 1_000, 2_000)
#: every resampled draw is this long, as in the fixed-boundary diagnostic, so the
#: ledger run uses exactly the same draws (their first ``min(slots, 200)``
#: trades; a longer study continues each path from a derived seed)
RACE_DRAW_LENGTH = 200


# ── the saved inputs ──────────────────────────────────────────────────────


def _cents(usd: Any) -> int:
    return int(round(float(usd) * 100))


def _ns(value: Any) -> int:
    stamp = utc_instant(value)
    if stamp is None:
        raise ValueError(f"missing time {value!r}")
    return int(stamp.value)


def firm_terms(study: FundedStudy, firm_key: str) -> dict[str, Any] | None:
    """The firm's frozen terms as saved in the result (``settings.firm_profiles``)."""

    for profile in (study.result.get("settings") or {}).get("firm_profiles") or []:
        if profile.get("firm_key") == firm_key:
            return dict(profile)
    return None


def terms_digest(terms: dict[str, Any] | None, processing_clock: dict[str, Any] | None) -> str:
    """SHA-256 of the saved firm terms and the saved processing clock (canonical JSON).

    Binds a cached conditional result to the exact account assumptions it used
    (correction A3): other terms or another clock give another digest.
    """

    payload = json.dumps({"firm_terms": terms, "processing_clock": processing_clock},
                         sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Slot:
    """One saved trade's place in time: entry, exit and trading day."""

    entry_ns: int
    exit_ns: int
    trading_day: str


@dataclass(frozen=True)
class TradeShape:
    """One saved trade's recorded outcome as a synthetic ordered price path."""

    index: int
    direction: str
    entry_ticks: int
    exit_ticks: int
    stop_ticks: int  # the order's stop: one tick beyond every point before the exit
    target_ticks: int  # the recorded target for a half exit; else beyond the best point
    path_ticks: tuple[int, ...]  # observations after the entry, ending at the exit
    net_cents: int  # recorded net result (for checks)
    scaled: bool
    order_pinned: bool  # the record fixes whether the worst or best point came first
    exact: bool  # worst and best points converted to whole ticks without rounding


@dataclass(frozen=True)
class FirmRules:
    """Everything the funded ledger needs for one configuration at one firm."""

    pair_id: str
    configuration: str
    firm_key: str
    firm_name: str
    profile: Any  # FundedFirmProfile
    processing: Any  # ProcessingClockPolicy
    quantity: int
    tick_value_cents: int
    cost_per_contract_mills: int
    scale_out: bool
    trading_days: tuple[Any, ...]  # TradingDay
    start_ns: int
    cutoff_ns: int

    @property
    def loss_allowance(self) -> float:
        return self.profile.loss_allowance_cents / 100

    @property
    def trigger(self) -> float:
        return (self.profile.retained_cushion_cents
                + self.profile.minimum_gross_request_cents) / 100


def _sizing(study: FundedStudy, configuration: str) -> tuple[int, int, int, bool]:
    settings = study.result.get("settings") or {}
    by_configuration = settings.get("sizing_by_configuration") or {}
    sizing = by_configuration.get(configuration)
    if sizing is not None:
        mills = int(round(float(sizing["cost_per_contract_per_fill_usd"]) * 1000))
        scale_out = str(sizing.get("exit_policy") or "fixed_target_v1") != "fixed_target_v1"
        return int(sizing["quantity"]), int(sizing["tick_value_cents"]), mills, scale_out
    if "quantity" in settings and "tick_value_cents" in settings:  # single-size plans
        return (int(settings["quantity"]), int(settings["tick_value_cents"]),
                int(round(float(settings["cost_per_side_usd"]) * 1000)), False)
    raise ValueError("the saved result does not record this configuration's size")


def firm_rules(study: FundedStudy, configuration: str, firm_key: str, *,
               source: Any) -> FirmRules:
    """Build the ledger inputs from the saved result and the verified package source.

    ``source`` is the study's :class:`ComparisonSource` (``open_comparison_source``
    on the plan's verified strategy package): its trading days, start and cutoff.
    """

    from alpha_lab.propsim.funded.clock import ProcessingClockPolicy
    from alpha_lab.propsim.funded.profiles import FundedFirmProfile

    terms = firm_terms(study, firm_key)
    if terms is None:
        raise ValueError(f"the saved result has no terms for {firm_key}")
    settings = study.result.get("settings") or {}
    clock = settings.get("processing_clock")
    if not clock:
        raise ValueError("the saved result has no payout-processing clock")
    quantity, tick_value, mills, scale_out = _sizing(study, configuration)
    profile = FundedFirmProfile.model_validate(terms)
    return FirmRules(
        pair_id=pair_key(configuration, firm_key), configuration=configuration,
        firm_key=firm_key, firm_name=profile.firm_name, profile=profile,
        processing=ProcessingClockPolicy.model_validate(clock), quantity=quantity,
        tick_value_cents=tick_value, cost_per_contract_mills=mills, scale_out=scale_out,
        trading_days=tuple(source.trading_days), start_ns=int(source.start_ns),
        cutoff_ns=int(source.cutoff_ns))


def slots_from_rows(rows: Sequence[dict[str, Any]]) -> tuple[Slot, ...]:
    return tuple(Slot(entry_ns=_ns(r["entry_utc"]), exit_ns=_ns(r["exit_utc"]),
                      trading_day=str(r["trading_day"])) for r in rows)


def _fill_cost(quantity: int, mills: int) -> int:
    from alpha_lab.propsim.funded.position_walk import fill_cost_cents

    return fill_cost_cents(quantity, mills) if quantity else 0


def _to_ticks(equity: int, base: int, per_tick: int, sign: int, *, worse: bool
              ) -> tuple[int, bool]:
    """Price (ticks from the entry) where ``base + move × per_tick`` equals ``equity``."""

    move, remainder = divmod(equity - base, per_tick)
    exact = remainder == 0
    if not exact and not worse:
        move += 1  # round a best point away from the entry only when inexact
    return sign * move, exact


@dataclass(frozen=True)
class MinuteIndex:
    """The study's stored one-minute E-mini bars, for ordering a trade's best and worst."""

    open_ns: np.ndarray
    high_ticks: np.ndarray
    low_ticks: np.ndarray

    @classmethod
    def from_frame(cls, frame: Any) -> MinuteIndex:
        stamps = frame["logical_open_ts_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        return cls(open_ns=np.asarray(stamps, dtype="datetime64[ns]").astype(np.int64),
                   high_ticks=np.asarray(frame["high_ticks"], dtype=np.int64),
                   low_ticks=np.asarray(frame["low_ticks"], dtype=np.int64))

    def first_reach(self, start_ns: int, end_ns: int, price: int, sign: int) -> int | None:
        """Open instant of the first minute after ``start_ns`` that reached ``price``."""

        lo = int(np.searchsorted(self.open_ns, start_ns, side="left"))
        hi = int(np.searchsorted(self.open_ns, end_ns, side="right"))
        if hi <= lo:
            return None
        window = self.high_ticks[lo:hi] >= price if sign > 0 else self.low_ticks[lo:hi] <= price
        if not bool(window.any()):
            return None
        return int(self.open_ns[lo + int(np.argmax(window))])


def _best_first(minutes: MinuteIndex | None, entry_ns: int, exit_ns: int, best: int,
                sign: int, low_ns: int | None) -> bool | None:
    """True/False when the stored minutes show whether the best point came first."""

    if minutes is None or low_ns is None:
        return None
    reached = minutes.first_reach(entry_ns, exit_ns, best, sign)
    if reached is None:
        return None
    low_minute = low_ns - low_ns % MINUTE_NS
    if reached < low_minute:
        return True
    if reached > low_minute:
        return False
    return None  # the same minute: the order inside it is not stored


def trade_shape(row: dict[str, Any], index: int, *, tick_value_cents: int, mills: int,
                minutes: MinuteIndex | None = None) -> TradeShape:
    """Turn one saved funded trade into a synthetic ordered path (see the module notes)."""

    sign = 1 if str(row["direction"]) == "long" else -1
    entry = int(row["entry_ticks"])
    exit_ = int(row["exit_ticks"])
    quantity = int(row["quantity"])
    half = int(row.get("scale_out_quantity") or 0)
    scaled = row.get("scale_out_ns") is not None and half > 0
    balance = _cents(row["balance_before_usd"])
    after_entry = balance - _fill_cost(quantity, mills)
    low, high = _cents(row["min_equity_usd"]), _cents(row["max_equity_usd"])
    entry_ns, exit_ns = _ns(row["entry_utc"]), _ns(row["exit_utc"])
    low_ns = _ns(row["min_equity_utc"]) if row.get("min_equity_utc") else None
    full = tick_value_cents * quantity
    exact = True

    if not scaled:
        worst_move, ok_w = _to_ticks(low, after_entry, full, 1, worse=True)
        best_move, ok_b = _to_ticks(high, after_entry, full, 1, worse=False)
        exact = ok_w and ok_b
        worst, best = entry + sign * worst_move, entry + sign * best_move
        low_at_exit = low_ns is not None and low_ns >= exit_ns
        target_exit = str(row.get("exit_kind")) == "target"
        hint = (None if low_at_exit or target_exit
                else _best_first(minutes, entry_ns, exit_ns, best, sign, low_ns))
        if low_at_exit:  # the lowest point was the exit itself: the best came first
            points, pinned = [best, exit_], True
        elif target_exit:  # the target fill is the highest point: the worst came first
            points, pinned = [worst, exit_], True
        elif hint is False:  # the stored minutes show the worst point came first
            points, pinned = [worst, best, exit_], True
        else:  # best first: shown by the minutes, or not pinned (a fixed convention,
            # not a proven best or worst case: reversals between points are not stored)
            points, pinned = [best, worst, exit_], hint is True
        before_exit = [entry, *points[:-1]]
        stop = min(p * sign for p in before_exit) * sign - sign
        target = max(p * sign for p in [*before_exit, exit_]) * sign + sign
        return TradeShape(index=index, direction=str(row["direction"]), entry_ticks=entry,
                          exit_ticks=exit_, stop_ticks=stop, target_ticks=target,
                          path_ticks=tuple(int(p) for p in points),
                          net_cents=_cents(row["net_pnl_usd"]), scaled=False,
                          order_pinned=pinned, exact=exact)

    target = int(row["target_ticks"])
    scale_ns = int(row["scale_out_ns"])
    rest = quantity - half
    after_half = (after_entry + (target - entry) * sign * tick_value_cents * half
                  - _fill_cost(half, mills))
    at_target_before_half = after_entry + (target - entry) * sign * full
    points: list[int] = []
    low_before_half = low_ns is not None and low_ns < scale_ns
    if low_before_half:
        worst_move, ok_w = _to_ticks(low, after_entry, full, 1, worse=True)
        exact = exact and ok_w
        points.append(entry + sign * worst_move)
    points.append(target)
    after: list[int] = []
    best = None
    if high > at_target_before_half:
        best_move, ok_b = _to_ticks(high, after_half, tick_value_cents * rest, 1, worse=False)
        exact = exact and ok_b
        best = entry + sign * best_move
        if best * sign > target * sign:
            after.append(best)
        else:
            best = None
    pinned = True
    if not low_before_half and low_ns is not None and low_ns < exit_ns:
        # the lowest point came after the half exit and before the final exit
        low_move, ok_l = _to_ticks(low, after_half, tick_value_cents * rest, 1, worse=True)
        exact = exact and ok_l
        low_price = entry + sign * low_move
        if low_price * sign > entry * sign:  # never below the break-even stop
            hint = (None if best is None
                    else _best_first(minutes, scale_ns, exit_ns, best, sign, low_ns))
            if hint is False:
                after.insert(0, low_price)
            else:
                after.append(low_price)
                pinned = best is None or hint is True
    points.extend(after)
    points.append(exit_)
    before_target = [entry, *points[:points.index(target)]]
    stop = min(p * sign for p in before_target) * sign - sign
    return TradeShape(index=index, direction=str(row["direction"]), entry_ticks=entry,
                      exit_ticks=exit_, stop_ticks=stop, target_ticks=target,
                      path_ticks=tuple(int(p) for p in points),
                      net_cents=_cents(row["net_pnl_usd"]), scaled=True,
                      order_pinned=pinned, exact=exact)


# ── one replay through the funded ledger ──────────────────────────────────


def _ledger(rules: FirmRules):
    from alpha_lab.propsim.funded.pair_ledger import PairLedger

    return PairLedger(
        pair_id=rules.pair_id, configuration=rules.configuration, profile=rules.profile,
        processing=rules.processing, quantity=rules.quantity,
        tick_value_cents=rules.tick_value_cents, cost_per_side_cents=0,
        trading_days=rules.trading_days, start_ns=rules.start_ns, cutoff_ns=rules.cutoff_ns,
        cost_per_contract_mills=rules.cost_per_contract_mills, scale_out=rules.scale_out)


def _observations(slot: Slot, shape: TradeShape):
    from alpha_lab.propsim.funded.position_walk import MinuteObservations

    # the entry price, then the recorded points: legs between them are treated as
    # continuous (a loss limit crossed on the way is met at its own level), and the
    # last step to the recorded exit is a jump so the exit fills at its recorded price
    points = (shape.entry_ticks, *shape.path_ticks)
    close_ns = max(slot.exit_ns, slot.entry_ns + len(points) + 1)
    step = max(1, (close_ns - slot.entry_ns) // (len(points) + 1))
    stamps = [slot.entry_ns + step * k for k in range(len(points))]
    continuous = np.ones(len(points), dtype=bool)
    continuous[0] = continuous[-1] = False
    return MinuteObservations(
        open_ns=slot.entry_ns, close_ns=close_ns, close_ticks=shape.exit_ticks,
        ts_ns=np.array(stamps, dtype=np.int64),
        price_ticks=np.array(points, dtype=np.int64), continuous=continuous,
        fidelity="minute_adverse_first")


def replay(rules: FirmRules, slots: Sequence[Slot], shapes: Sequence[TradeShape],
           order: Sequence[int]):
    """Run the funded ledger over the slots with ``shapes[order[i]]`` in slot ``i``.

    A slot whose account refuses entries (payout protection or processing) is
    counted as refused and its trade is skipped, as the simulator does with a
    refused strategy signal. Returns the finished ledger.
    """

    ledger = _ledger(rules)
    ledger.start()
    for number, (slot, pick) in enumerate(zip(slots, order, strict=False)):
        if slot.entry_ns >= rules.cutoff_ns:
            break
        ledger.run_until(slot.entry_ns - MINUTE_NS)
        reason = ledger.gate()
        if reason is not None:
            ledger.note_blocked(slot.entry_ns, reason,
                                "the resampled trade in this slot was refused")
            continue
        shape = shapes[int(pick)]
        ledger.open(ts_ns=slot.entry_ns, trade_ref=f"slot-{number + 1}",
                    direction=shape.direction, entry_ticks=shape.entry_ticks,
                    stop_ticks=shape.stop_ticks, target_ticks=shape.target_ticks,
                    trading_day=slot.trading_day,
                    strategy={"trade_id": f"resampled-{shape.index}"})
        if ledger.position is None:  # the entry cost alone reached the loss limit
            continue
        ledger.run_until(slot.entry_ns)
        ledger.on_minute(_observations(slot, shape), deadline_minute=True,
                         trading_day=slot.trading_day)
        if ledger.position is not None:  # pragma: no cover - the deadline always closes
            raise AssertionError("the synthetic trade did not close")
    ledger.finish()
    return ledger


@dataclass(frozen=True)
class LedgerOutcome:
    """What one replay produced (exact cents, one firm)."""

    net_cash_cents: int
    received_cents: int
    costs_cents: int
    payouts: int
    accounts: int
    #: per account bought: (payouts received, failed, trades taken)
    account_rows: tuple[tuple[int, bool, int], ...]
    #: the first account: "paid", "died" or "going", and trades taken until then
    first_fate: str
    first_trades: int | None
    trade_nets: tuple[int, ...] = ()
    trade_failed: tuple[bool, ...] = ()


def ledger_outcome(ledger) -> LedgerOutcome:
    first = ledger.accounts[0]
    receipts = [e for e in ledger.payout_events if e.get("event") == "received"]
    first_receipts = [e["ts_ns"] for e in receipts if e["account_id"] == first.account_id]
    first_trades = [t for t in ledger.trades if t["account_id"] == first.account_id]
    if first_receipts:
        paid_at = min(first_receipts)
        fate = "paid"
        trades_until = sum(1 for t in first_trades if t["exit_ns"] <= paid_at)
    elif first.failed_ns is not None:
        fate, trades_until = "died", len(first_trades)
    else:
        fate, trades_until = "going", None
    return LedgerOutcome(
        net_cash_cents=int(ledger.receipts - ledger.costs),
        received_cents=int(ledger.receipts), costs_cents=int(ledger.costs),
        payouts=len(receipts), accounts=len(ledger.accounts),
        account_rows=tuple((a.payouts_received, a.failed_ns is not None, a.trades)
                           for a in ledger.accounts),
        first_fate=fate, first_trades=trades_until,
        trade_nets=tuple(int(t["net_pnl_cents"]) for t in ledger.trades),
        trade_failed=tuple(bool(t["account_failed"]) for t in ledger.trades))


@dataclass(frozen=True)
class PathOutcome:
    """One resampled path through the ledger, to the cutoff (exact cents; correction A4).

    Received money is only what the ledger received; a payout requested but not
    received by the cutoff is counted separately (``requests_unresolved`` and its
    after-split ``unresolved_trader_cents``) and is never received cash. The
    ``first_*`` endpoints belong to the path's first account: eligibility = its
    first ``eligibility_secured`` event, request = its first ``requested`` event,
    receipt = its first ``received`` event, failure = its ``failed_ns``. Trades to
    an endpoint = that account's trades closed at or before the event; days =
    calendar days from the account's purchase. ``None`` = not reached by the cutoff.
    """

    path: int
    net_cash_cents: int
    received_cents: int
    costs_cents: int
    accounts_purchased: int
    accounts_failed: int
    accounts_open_at_cutoff: int
    payouts_received: int
    payouts_received_by_failed: int
    payouts_received_by_open: int
    requests_unresolved: int  # requested, no matching receipt by the cutoff
    unresolved_trader_cents: int  # their after-split amounts (NOT received)
    first_account_endpoint: str  # "received" | "failed" | "open"
    first_trades_to_eligibility: int | None
    first_trades_to_request: int | None
    first_trades_to_receipt: int | None
    first_trades_to_failure: int | None
    first_days_to_eligibility: float | None  # calendar days from the account's purchase
    first_days_to_request: float | None
    first_days_to_receipt: float | None
    first_days_to_failure: float | None


def path_outcome(ledger, path: int = 0) -> PathOutcome:
    """Read one finished ledger into a :class:`PathOutcome` (the ledger is not changed)."""

    first = ledger.accounts[0]
    mine = [e for e in ledger.payout_events if e.get("account_id") == first.account_id]

    def first_at(kind: str) -> int | None:
        stamps = [int(e["ts_ns"]) for e in mine if e.get("event") == kind]
        return min(stamps) if stamps else None

    eligible_at, requested_at = first_at("eligibility_secured"), first_at("requested")
    received_at = first_at("received")
    failed_at = None if first.failed_ns is None else int(first.failed_ns)
    first_exits = [int(t["exit_ns"]) for t in ledger.trades
                   if t["account_id"] == first.account_id]

    def trades_by(at: int | None) -> int | None:
        return None if at is None else sum(1 for exit_ns in first_exits if exit_ns <= at)

    def days_by(at: int | None) -> float | None:
        return None if at is None else (at - int(first.created_ns)) / DAY_NS

    if received_at is not None and (failed_at is None or received_at < failed_at):
        endpoint = "received"
    elif failed_at is not None:
        endpoint = "failed"
    else:
        endpoint = "open"
    received_ids = {e.get("request_id") for e in ledger.payout_events
                    if e.get("event") == "received"}
    unresolved = [e for e in ledger.payout_events
                  if e.get("event") == "requested" and e.get("request_id") not in received_ids]
    failed = [a for a in ledger.accounts if a.failed_ns is not None]
    still_open = [a for a in ledger.accounts if a.failed_ns is None]
    return PathOutcome(
        path=int(path), net_cash_cents=int(ledger.receipts - ledger.costs),
        received_cents=int(ledger.receipts), costs_cents=int(ledger.costs),
        accounts_purchased=len(ledger.accounts), accounts_failed=len(failed),
        accounts_open_at_cutoff=len(still_open),
        payouts_received=sum(1 for e in ledger.payout_events if e.get("event") == "received"),
        payouts_received_by_failed=sum(a.payouts_received for a in failed),
        payouts_received_by_open=sum(a.payouts_received for a in still_open),
        requests_unresolved=len(unresolved),
        unresolved_trader_cents=sum(int(e.get("trader_cents") or 0) for e in unresolved),
        first_account_endpoint=endpoint,
        first_trades_to_eligibility=trades_by(eligible_at),
        first_trades_to_request=trades_by(requested_at),
        first_trades_to_receipt=trades_by(received_at),
        first_trades_to_failure=trades_by(failed_at),
        first_days_to_eligibility=days_by(eligible_at),
        first_days_to_request=days_by(requested_at),
        first_days_to_receipt=days_by(received_at),
        first_days_to_failure=days_by(failed_at))


# ── validation: the original order must reproduce the saved result ────────


@dataclass(frozen=True)
class Validation:
    firm_key: str
    firm_name: str
    trades: int
    saved: dict[str, int]
    replayed: dict[str, int]
    trades_matching: int  # trades whose net result and account loss match the record
    refused_slots: int
    inexact_shapes: int
    unpinned_shapes: int
    replayed_trades: int | None = None  # how many trades the replay produced

    @property
    def exact(self) -> bool:
        return self.saved == self.replayed and self.trades_matching == self.trades

    @property
    def differences(self) -> dict[str, int]:
        return {k: self.replayed[k] - self.saved[k] for k in self.saved
                if self.replayed.get(k) != self.saved[k]}


_SAVED_KEYS = {"net_cash_cents": "net_cash_earned_cents",
               "received_cents": "payouts_received_cents",
               "payouts": "payouts_received_count", "accounts": "accounts_purchased",
               "costs_cents": "account_costs_cents"}


def build_inputs(study: FundedStudy, configuration: str, firm_key: str, *, source: Any,
                 minutes: MinuteIndex | None = None
                 ) -> tuple[FirmRules, tuple[Slot, ...], tuple[TradeShape, ...]]:
    rules = firm_rules(study, configuration, firm_key, source=source)
    rows = ordered_trades(study, configuration, firm_key)
    shapes = tuple(trade_shape(r, i, tick_value_cents=rules.tick_value_cents,
                               mills=rules.cost_per_contract_mills, minutes=minutes)
                   for i, r in enumerate(rows))
    return rules, slots_from_rows(rows), shapes


def validate_original_order(study: FundedStudy, configuration: str, firm_key: str, *,
                            source: Any, minutes: MinuteIndex | None = None) -> Validation:
    """Replay the saved order through the adapter and compare with the saved summary."""

    rules, slots, shapes = build_inputs(study, configuration, firm_key, source=source,
                                        minutes=minutes)
    return check_original(study, configuration, firm_key, rules, slots, shapes)


def check_original(study: FundedStudy, configuration: str, firm_key: str, rules: FirmRules,
                   slots: Sequence[Slot], shapes: Sequence[TradeShape]) -> Validation:
    """The saved order through the adapter, compared with the saved result.

    Compared: the five summary figures of ``_SAVED_KEYS`` (exact integers), the number
    of trades (a different count leaves ``trades_matching`` at 0), and, trade by trade
    in the saved order, each trade's net result (to the cent) and account-loss flag.
    Not compared: fill times, prices or quantities, which account took each trade,
    payout or failure times, and setup lineage.
    """

    ledger = replay(rules, slots, shapes, range(len(shapes)))
    outcome = ledger_outcome(ledger)
    summary = study.summary(configuration, firm_key) or {}
    saved = {mine: int(summary.get(theirs) or 0) for mine, theirs in _SAVED_KEYS.items()}
    replayed = {k: int(getattr(outcome, k)) for k in _SAVED_KEYS}
    rows = ordered_trades(study, configuration, firm_key)
    matching = sum(
        1 for row, net, failed in zip(rows, outcome.trade_nets, outcome.trade_failed,
                                      strict=False)
        if _cents(row["net_pnl_usd"]) == net and bool(row["account_failed"]) == failed)
    refused = sum(sum(a.blocked_entries.values()) for a in ledger.accounts)
    return Validation(
        firm_key=firm_key, firm_name=rules.firm_name, trades=len(shapes), saved=saved,
        replayed=replayed,
        trades_matching=matching if len(outcome.trade_nets) == len(rows) else 0,
        refused_slots=refused, inexact_shapes=sum(1 for s in shapes if not s.exact),
        unpinned_shapes=sum(1 for s in shapes if not s.order_pinned),
        replayed_trades=len(outcome.trade_nets))


# ── conditional resampling over the resampled draws ───────────────────────


def extension_seed(seed: int) -> int:
    """The seed that continues each path past ``RACE_DRAW_LENGTH`` trades (derived, fixed)."""

    return int(np.random.SeedSequence((int(seed), RACE_DRAW_LENGTH)).generate_state(1)[0])


def shared_trades(count: int) -> int:
    """How many trades of each full-version path are exactly the chart's draw."""

    return min(int(count), RACE_DRAW_LENGTH)


def race_orders(count: int, *, paths: int, seed: int, method: str = "blocks") -> np.ndarray:
    """Resampled draws as trade indices: the fixed-boundary diagnostic's own draws.

    The fixed-boundary diagnostic draws ``RACE_DRAW_LENGTH`` trades per path. The indices
    are drawn exactly the same way (same seed, method and length), so the first
    ``min(count, RACE_DRAW_LENGTH)`` trades of path *i* here are path *i* of the
    chart, for every path. A study with more trade slots than that continues
    each path with a second draw of the remaining slots from
    :func:`extension_seed` (those trades are not in the chart, which stops at
    ``RACE_DRAW_LENGTH``).
    """

    index = np.arange(count, dtype=float)
    draws = resample_paths(index, paths=paths, length=RACE_DRAW_LENGTH, method=method,
                           seed=seed)
    if count > RACE_DRAW_LENGTH:
        more = resample_paths(index, paths=paths, length=count - RACE_DRAW_LENGTH,
                              method=method, seed=extension_seed(seed))
        draws = np.concatenate([draws, more], axis=1)
    return draws[:, :count].astype(int)


@dataclass(frozen=True)
class FullRace:
    """Conditional resampling with one firm's ledger rules: every path to the cutoff.

    Shares (``paid``/``died``/``still_going``) are the first account's endpoint
    shares: received a first payout before failing / failed before any payout was
    received / neither by the cutoff. ``payouts_before_death`` =
    ``payouts_by_failed / failed_accounts`` — the average payouts among accounts
    that failed within the tested horizon (not a lifetime expectation); open
    accounts, their payouts and unresolved requests are reported separately.
    ``cash_per_account`` = (all paths' received after-split payouts − all account
    purchases) ÷ all accounts purchased, in dollars: a pooled ratio, not an
    average of each path's own ratio (correction A4).
    """

    firm_key: str
    firm_name: str
    paths: int
    seed: int
    method: str
    slots: int
    paid_share: float
    died_share: float
    still_going_share: float
    #: median trades of the first account among paths whose endpoint was the receipt
    typical_to_payout: float | None
    #: median trades of the first account among paths whose endpoint was the failure
    typical_to_limit: float | None
    #: payouts_by_failed ÷ failed_accounts (None when no account failed)
    payouts_before_death: float | None
    #: accounts that failed within the horizon (equal to ``failed_accounts``)
    lost_accounts: int
    #: pooled: (received − account costs) ÷ accounts purchased, dollars
    cash_per_account: float | None
    accounts_bought: int
    net_cash_bad: float
    net_cash_typical: float
    net_cash_good: float
    seconds: float = 0.0
    validation: Validation | None = field(default=None, compare=False)
    #: every run's net cash in dollars, in run order (for the actual result's standing)
    net_cash_values: tuple[float, ...] = field(default=(), compare=False, repr=False)
    model_id: str = MODEL_ID
    #: numerator and denominator of ``payouts_before_death``
    payouts_by_failed: int = 0
    failed_accounts: int = 0
    #: accounts still open at the cutoff and the payouts they had received
    open_accounts: int = 0
    payouts_by_open: int = 0
    #: requests with no receipt by the cutoff and their after-split amounts (not received)
    unresolved_requests: int = 0
    unresolved_trader_cents: int = 0
    received_cents_total: int = 0
    costs_cents_total: int = 0
    #: medians over the paths whose first account reached each endpoint (None: none did)
    median_trades_to_eligibility: float | None = None
    median_trades_to_request: float | None = None
    median_trades_to_receipt: float | None = None
    median_trades_to_failure: float | None = None
    median_days_to_eligibility: float | None = None
    median_days_to_request: float | None = None
    median_days_to_receipt: float | None = None
    median_days_to_failure: float | None = None
    #: the tested horizon: first and last trading day (``YYYY-MM-DD``) of the ledger run
    horizon_first_day: str | None = None
    horizon_last_day: str | None = None
    outcomes: tuple[PathOutcome, ...] = field(default=(), compare=False, repr=False)

    def share_below(self, actual: float) -> float | None:
        """Share of paths whose net cash was below ``actual``; ties count half."""

        return share_below_ties_half(self.net_cash_values, actual)


def horizon_days(rules: FirmRules, slots: Sequence[Slot] = ()) -> tuple[str | None, str | None]:
    """First and last trading day of the ledger run, from the race's own inputs.

    The rules' trading days whose day end falls after the start and at or before
    the cutoff (the days the ledger runs); the slots' days when there are none.
    """

    days = [d.trading_day for d in rules.trading_days
            if rules.start_ns < int(d.day_end_ns) <= rules.cutoff_ns]
    if not days:
        days = [d.trading_day for d in rules.trading_days] or [s.trading_day for s in slots]
    return (str(days[0]), str(days[-1])) if days else (None, None)


def _median(values: Sequence[float | int | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return float(np.median(present)) if present else None


def race_from_outcomes(rules: FirmRules, outcomes: Sequence[PathOutcome], *, paths: int,
                       seed: int, method: str, slots: int, seconds: float = 0.0,
                       validation: Validation | None = None,
                       horizon: tuple[str | None, str | None] = (None, None)) -> FullRace:
    """Pool the per-path outcomes into a :class:`FullRace` (every denominator exported)."""

    count = max(1, len(outcomes))
    paid = sum(1 for o in outcomes if o.first_account_endpoint == "received") / count
    died = sum(1 for o in outcomes if o.first_account_endpoint == "failed") / count
    received = sum(o.received_cents for o in outcomes)
    costs = sum(o.costs_cents for o in outcomes)
    bought = sum(o.accounts_purchased for o in outcomes)
    failed = sum(o.accounts_failed for o in outcomes)
    by_failed = sum(o.payouts_received_by_failed for o in outcomes)
    net_usd = np.asarray([o.net_cash_cents for o in outcomes], dtype=float) / 100
    if net_usd.size == 0:
        net_usd = np.zeros(1)
    return FullRace(
        firm_key=rules.firm_key, firm_name=rules.firm_name, paths=paths, seed=seed,
        method=method, slots=slots, paid_share=paid, died_share=died,
        # counted, not 1 − paid − died (which leaves a float residue such as 5.6e-17)
        still_going_share=sum(1 for o in outcomes
                              if o.first_account_endpoint == "open") / count,
        typical_to_payout=_median([o.first_trades_to_receipt for o in outcomes
                                   if o.first_account_endpoint == "received"]),
        typical_to_limit=_median([o.first_trades_to_failure for o in outcomes
                                  if o.first_account_endpoint == "failed"]),
        payouts_before_death=(by_failed / failed) if failed else None,
        lost_accounts=failed,
        cash_per_account=((received - costs) / bought / 100) if bought else None,
        accounts_bought=bought,
        net_cash_bad=float(np.percentile(net_usd, 5)),
        net_cash_typical=float(np.median(net_usd)),
        net_cash_good=float(np.percentile(net_usd, 95)),
        seconds=seconds, validation=validation,
        net_cash_values=tuple(float(o.net_cash_cents) / 100 for o in outcomes),
        model_id=MODEL_ID, payouts_by_failed=by_failed, failed_accounts=failed,
        open_accounts=sum(o.accounts_open_at_cutoff for o in outcomes),
        payouts_by_open=sum(o.payouts_received_by_open for o in outcomes),
        unresolved_requests=sum(o.requests_unresolved for o in outcomes),
        unresolved_trader_cents=sum(o.unresolved_trader_cents for o in outcomes),
        received_cents_total=received, costs_cents_total=costs,
        median_trades_to_eligibility=_median([o.first_trades_to_eligibility for o in outcomes]),
        median_trades_to_request=_median([o.first_trades_to_request for o in outcomes]),
        median_trades_to_receipt=_median([o.first_trades_to_receipt for o in outcomes]),
        # the failure clock covers the same population as the endpoint row: first
        # accounts that failed before any payout was received
        median_trades_to_failure=_median([o.first_trades_to_failure for o in outcomes
                                          if o.first_account_endpoint == "failed"]),
        median_days_to_eligibility=_median([o.first_days_to_eligibility for o in outcomes]),
        median_days_to_request=_median([o.first_days_to_request for o in outcomes]),
        median_days_to_receipt=_median([o.first_days_to_receipt for o in outcomes]),
        median_days_to_failure=_median([o.first_days_to_failure for o in outcomes
                                        if o.first_account_endpoint == "failed"]),
        horizon_first_day=horizon[0], horizon_last_day=horizon[1],
        outcomes=tuple(outcomes))


def full_race(rules: FirmRules, slots: Sequence[Slot], shapes: Sequence[TradeShape], *,
              paths: int, seed: int, method: str = "blocks",
              progress: Callable[[int, int], None] | None = None,
              validation: Validation | None = None) -> FullRace:
    """Every path: the study's calendar and slots with one resampled draw of the trades.

    The first account of each path gives the endpoint shares (received a first
    payout before failing; failed before any payout was received; neither by the
    cutoff). The whole path, with replacements, gives the payouts of accounts that
    failed within the horizon, the open accounts and unresolved requests at the
    cutoff, and the pooled net cash per purchased account. Nothing typed on a
    screen (such as the fixed diagnostic's boundaries) enters this run.
    """

    import time

    started = time.monotonic()
    orders = race_orders(len(shapes), paths=paths, seed=seed, method=method)
    outcomes: list[PathOutcome] = []
    for number, order in enumerate(orders):
        outcomes.append(path_outcome(replay(rules, slots, shapes, order), number))
        if progress is not None:
            progress(number + 1, paths)
    return race_from_outcomes(rules, outcomes, paths=paths, seed=seed, method=method,
                              slots=len(slots), seconds=round(time.monotonic() - started, 1),
                              validation=validation, horizon=horizon_days(rules, slots))


def open_source(plan: Any):
    """The study's verified strategy package as a comparison source (read only)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.market import study_package_root
    from alpha_lab.propsim.funded.comparison_source import open_comparison_source

    root = study_package_root(plan)
    if root is None:
        return None
    return open_comparison_source(Path(root))
