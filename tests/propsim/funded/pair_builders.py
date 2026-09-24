"""Deterministic synthetic days, prints and a scripted strategy for the pair engine.

The scripted strategy obeys the same one-slot rules as Strategy-Core's reducer:
one position at a time; entries at the confirming candle's close; its own exits
checked on later candles, stop first, then target, then the daily deadline; no
new entry on the candle a position resolves; an entry refused by the account is
discarded. Each synthetic minute is a list of trade prices; its candle is
rebuilt from them, so the prints always reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import numpy as np

from alpha_lab.propsim.funded.campaign import TradingDay
from alpha_lab.propsim.funded.clock import CHICAGO, NS, TWO_BUSINESS_DAYS_FED_1600, to_ns
from alpha_lab.propsim.funded.pair_engine import DayInput, PairRun, run_days
from alpha_lab.propsim.funded.pair_ledger import PairLedger
from alpha_lab.propsim.funded.position_walk import MinuteObservations
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES
from alpha_lab.propsim.funded.strategy_driver import EntrySignal, StepOutcome

BASE = 80_000  # ticks


def minute_bar(day: date, start: datetime, index: int, prices: list[int]):
    opened = start + timedelta(minutes=index)
    closed = opened + timedelta(minutes=1)
    return SimpleNamespace(
        trading_day=day, logical_open_ts_utc=opened, logical_close_ts_utc=closed,
        open_ts_utc=opened, close_ts_utc=closed, availability_ts_utc=closed,
        open_ticks=prices[0], high_ticks=max(prices), low_ticks=min(prices),
        close_ticks=prices[-1], trade_count=len(prices), prices=list(prices),
    )


@dataclass
class SyntheticDay:
    day: date
    minutes: list[list[int]]
    # minute index -> (direction, stop ticks, target ticks)
    signals: dict[int, tuple[str, int, int]] = field(default_factory=dict)
    is_evaluation: bool = True


def build(days: list[SyntheticDay], *, open_clock: time = time(9, 0)):
    """DayInputs, trading schedule, the evaluation start and cutoff."""

    inputs, schedule = [], []
    for spec in days:
        start = datetime.combine(spec.day, open_clock, tzinfo=CHICAGO)
        bars = [minute_bar(spec.day, start, i, prices) for i, prices in enumerate(spec.minutes)]
        for i, bar in enumerate(bars):
            bar.signal = spec.signals.get(i)
        deadline = to_ns(bars[-1].logical_close_ts_utc)
        trading_day = TradingDay(trading_day=spec.day.isoformat(),
                                 day_end_ns=deadline + 5 * 60 * NS,
                                 reopen_ns=deadline + 65 * 60 * NS, deadline_ns=deadline)
        schedule.append(trading_day)
        inputs.append(DayInput(trading_day=spec.day.isoformat(), bars_by_tf={60: bars},
                               levels_for=None, is_evaluation=spec.is_evaluation,
                               schedule=trading_day if spec.is_evaluation else None))
    evaluation = [d for d, s in zip(schedule, days, strict=True) if s.is_evaluation]
    first_eval = next(i for i, s in enumerate(days) if s.is_evaluation)
    start_ns = to_ns(inputs[first_eval].bars_by_tf[60][0].logical_open_ts_utc) - 60 * NS
    cutoff_ns = evaluation[-1].day_end_ns
    return inputs, tuple(evaluation), start_ns, cutoff_ns


class ScriptedPrints:
    def minute(self, bar, sign: int) -> MinuteObservations:
        open_ns = to_ns(bar.logical_open_ts_utc)
        close_ns = to_ns(bar.logical_close_ts_utc)
        n = len(bar.prices)
        step = (close_ns - open_ns) // (n + 1)
        return MinuteObservations(
            open_ns=open_ns, close_ns=close_ns, close_ticks=bar.close_ticks,
            ts_ns=np.array([open_ns + step * i for i in range(n)], dtype=np.int64),
            price_ticks=np.array(bar.prices, dtype=np.int64),
            continuous=np.zeros(n, dtype=bool), fidelity="ordered_trade_prints")


class ScriptedDriver:
    """Stand-in for Strategy-Core with the same one-slot semantics.

    With ``scale_out`` it follows Core's scale-out exit: the target candle scales
    out (no resolution), the stop moves to the entry price and is checked from the
    NEXT candle, and the rest is held to break-even or the deadline.
    """

    def __init__(self, scale_out: bool = False) -> None:
        self.scale_out = scale_out
        self.position = None
        self.forced_flat = 0
        self.discarded_setups = 0
        self.refusals: list[str] = []
        self.entries = 0
        self._bars: list = []
        self._deadline = None

    def begin_day(self, bars_by_tf, levels_for):
        self._bars = bars_by_tf[60]
        self._deadline = self._bars[-1].logical_close_ts_utc
        return list(self._bars)

    @property
    def in_position(self) -> bool:
        return self.position is not None

    def step(self, bar, gate):
        exits = []
        if self.position is not None and bar.logical_open_ts_utc >= self.position.opened:
            p = self.position
            long = p.direction == "long"
            scaled = getattr(p, "scaled", False)
            if (bar.low_ticks <= p.stop_ticks) if long else (bar.high_ticks >= p.stop_ticks):
                exits.append(self._resolve(bar, "breakeven_stop" if scaled else "stop",
                                           p.stop_ticks))
            elif not scaled and ((bar.high_ticks >= p.target_ticks) if long
                                 else (bar.low_ticks <= p.target_ticks)):
                if self.scale_out:
                    p.scaled = True
                    p.initial_stop = p.stop_ticks
                    p.stop_ticks = p.entry_ticks
                    if bar.logical_close_ts_utc == self._deadline:
                        exits.append(self._resolve(bar, "scheduled_close", bar.close_ticks))
                else:
                    exits.append(self._resolve(bar, "target", p.target_ticks))
            elif bar.logical_close_ts_utc == self._deadline:
                exits.append(self._resolve(bar, "scheduled_close", bar.close_ticks))
            if exits:
                return StepOutcome(entry=None, core_exits=tuple(exits), refused=())
        if self.position is not None or getattr(bar, "signal", None) is None:
            return StepOutcome(entry=None, core_exits=(), refused=())
        direction, stop, target = bar.signal
        if gate is not None:
            self.refusals.append(gate)
            return StepOutcome(entry=None, core_exits=(), refused=(gate,))
        self.entries += 1
        trade_id = f"T{self.entries}-{bar.logical_close_ts_utc.isoformat()}"
        self.position = SimpleNamespace(
            trade_id=trade_id, direction=direction, entry_ticks=bar.close_ticks,
            stop_ticks=stop, target_ticks=target, opened=bar.logical_close_ts_utc,
            entry_ts=bar.logical_close_ts_utc)
        return StepOutcome(
            entry=EntrySignal(trade_id=trade_id, direction=direction,
                              entry_ticks=bar.close_ticks, stop_ticks=stop,
                              target_ticks=target,
                              entry_chart="one-hour gap, five-minute parent chart, "
                                          "one-minute entry chart"),
            core_exits=(), refused=())

    def _resolve(self, bar, kind: str, price: int):
        p = self.position
        self.position = None
        return SimpleNamespace(
            trade_id=p.trade_id, direction=p.direction, entry_ts_utc=p.entry_ts,
            resolution_ts_utc=bar.logical_close_ts_utc, entry_ticks=p.entry_ticks,
            stop_ticks=getattr(p, "initial_stop", p.stop_ticks), target_ticks=p.target_ticks,
            resolution=kind, exit_ticks=price,
            risk_ticks=abs(p.entry_ticks - getattr(p, "initial_stop", p.stop_ticks)),
            realized_r=None, envelope=SimpleNamespace(trading_day=bar.trading_day),
            scale_out_ticks=p.target_ticks if getattr(p, "scaled", False) else None)

    def force_flat(self, *, counted: bool = True) -> None:
        assert self.position is not None
        self.position = None
        self.forced_flat += int(counted)

    def discard_refused_setup(self) -> None:
        self.discarded_setups += 1

    def end_day(self, trading_day, *, dataset_exhausted=False):
        assert self.position is None, "scripted strategy holds through the close"
        return []

    def checkpoint(self) -> dict:
        return {"forced_flat": self.forced_flat, "discarded_setups": self.discarded_setups,
                "entries": self.entries}

    def restore(self, state: dict) -> None:
        self.forced_flat = state["forced_flat"]
        self.discarded_setups = state["discarded_setups"]
        self.entries = state["entries"]

    def seed_hash(self) -> str:
        return f"{self.entries}:{self.forced_flat}:{self.discarded_setups}"


def ledger_for(firm_key: str, schedule, start_ns: int, cutoff_ns: int, *,
               pair_id: str | None = None, quantity: int = 1,
               cost_per_side_cents: int = 514, tick_value_cents: int = 500,
               cost_per_contract_mills: int | None = None,
               scale_out: bool = False) -> PairLedger:
    profile = FIRM_PROFILES[firm_key]
    return PairLedger(
        pair_id=pair_id or f"CFG|{firm_key}", configuration="CFG", profile=profile,
        processing=TWO_BUSINESS_DAYS_FED_1600, quantity=quantity,
        tick_value_cents=tick_value_cents, cost_per_side_cents=cost_per_side_cents,
        trading_days=schedule, start_ns=start_ns, cutoff_ns=cutoff_ns,
        cost_per_contract_mills=cost_per_contract_mills, scale_out=scale_out)


def run_pair(days: list[SyntheticDay], firm_key: str = "takeprofittrader", **kw):
    inputs, schedule, start_ns, cutoff_ns = build(days)
    ledger = ledger_for(firm_key, schedule, start_ns, cutoff_ns, **kw)
    run = PairRun(ledger.pair_id, ScriptedDriver(scale_out=kw.get("scale_out", False)),
                  ledger)
    prints = ScriptedPrints()
    run_days([run], inputs, lambda _day: (lambda: prints))
    ledger.finish()
    return run, ledger, (inputs, schedule, start_ns, cutoff_ns)


def weekdays(first: date, count: int) -> list[date]:
    out, cursor = [], first
    while len(out) < count:
        if cursor.weekday() < 5:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def flat(n: int, price: int = BASE) -> list[list[int]]:
    return [[price] for _ in range(n)]
