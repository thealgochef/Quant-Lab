"""Drive each configuration-and-firm pair through its own strategy replay.

For every trading day and every pair, candle by candle:

1. the pair's timed account events due by the candle's open are processed
   (payout completion, day-end request, next-day release);
2. an open position is walked through the candle's ordered observations
   (exit, stop, target, loss-limit breach, daily deadline);
3. the live account's admission reason is handed to the pair's OWN Strategy-Core
   reducer, which steps the candle;
4. a Core entry opens the account's position at the recorded fill; a refused
   entry is counted and its setup discarded; a position the account already
   closed (loss limit) is cleared from the reducer so later setups can form.

Warmup days run the strategy with no account (its trades are not booked),
exactly like the reference study. A separate REFERENCE run of the same
configuration, with no account at all, reproduces the saved study's trades and
proves the driver is equivalent to the normal replay on this exact path.

Pairs share only immutable inputs (the candles, levels and prints of the day).
Each pair has its own reducer, account ledger and money.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from alpha_lab.propsim.funded.campaign import TradingDay
from alpha_lab.propsim.funded.pair_ledger import BLOCK_REASONS, PairLedger
from alpha_lab.propsim.funded.print_minutes import bar_window_ns

__all__ = ["DayInput", "PairRun", "EngineConsistencyError", "run_days", "run_one_day"]


class EngineConsistencyError(AssertionError):
    """The strategy and the account disagree about an open position."""


@dataclass(frozen=True)
class DayInput:
    trading_day: str
    bars_by_tf: dict[int, list]
    levels_for: Any
    is_evaluation: bool
    schedule: TradingDay | None
    exhausted: bool = False


@dataclass
class PairRun:
    pair_id: str
    driver: Any
    ledger: PairLedger | None  # None: the no-account reference run
    strategy_trades: list[dict] = field(default_factory=list)  # Core's own resolutions
    warmup_trades: int = 0
    also_blocked: int = 0
    open_rows: dict[str, dict] = field(default_factory=dict)

    def to_state(self) -> dict[str, Any]:
        return {"pair_id": self.pair_id, "driver": self.driver.checkpoint(),
                "ledger": None if self.ledger is None else self.ledger.snapshot(),
                "strategy_trades": list(self.strategy_trades),
                "warmup_trades": self.warmup_trades, "also_blocked": self.also_blocked}


def _core_trade(record, *, warmup: bool) -> dict[str, Any]:
    return {
        "trade_id": str(record.trade_id), "direction": str(record.direction),
        "entry_ts_utc": record.entry_ts_utc.isoformat(),
        "resolution_ts_utc": (record.resolution_ts_utc.isoformat()
                              if record.resolution_ts_utc else None),
        "entry_ticks": int(record.entry_ticks), "stop_ticks": int(record.stop_ticks),
        "target_ticks": int(record.target_ticks), "resolution": str(record.resolution),
        "exit_ticks": None if record.exit_ticks is None else int(record.exit_ticks),
        "is_warmup": warmup,
        "risk_ticks": int(record.risk_ticks),
        "realized_r": None if record.realized_r is None else float(record.realized_r),
        "trading_day": record.envelope.trading_day.isoformat(),
        "scale_out_ticks": getattr(record, "scale_out_ticks", None),
    }


def run_one_day(run: PairRun, day: DayInput, prints) -> None:
    driver = run.driver
    bars = driver.begin_day(day.bars_by_tf, day.levels_for)
    ledger = run.ledger if (run.ledger is not None and day.is_evaluation) else None
    if ledger is not None and not ledger.started:
        ledger.start()
    deadline = day.schedule.deadline_ns if day.schedule is not None else None
    for bar in bars:
        open_ns, close_ns = bar_window_ns(bar)
        exited = None
        gate = None
        if ledger is not None:
            if bar.trading_day.isoformat() != day.trading_day:
                raise EngineConsistencyError("candle trading day differs from the schedule")
            ledger.run_until(open_ns)
            if ledger.position is not None:
                exited = ledger.on_minute(prints().minute(bar, ledger.position.sign),
                                          deadline_minute=close_ns == deadline,
                                          trading_day=day.trading_day)
                if exited is not None:
                    row = ledger.trades[-1]
                    run.open_rows[row["strategy_trade_id"]] = row
            gate = ledger.gate()
        out = driver.step(bar, gate)
        if ledger is None:
            for record in out.core_exits:
                run.strategy_trades.append(_core_trade(record, warmup=not day.is_evaluation))
                if not day.is_evaluation:
                    run.warmup_trades += 1
            continue
        for record in out.core_exits:
            if ledger.position is not None:
                raise EngineConsistencyError(
                    f"the strategy resolved {record.trade_id} while the account position "
                    "is still open")
            run.strategy_trades.append(_core_trade(record, warmup=False))
            row = run.open_rows.pop(str(record.trade_id), None)
            if row is not None:
                row["strategy_recorded_exit_kind"] = str(record.resolution)
                row["strategy_recorded_exit_ticks"] = (
                    None if record.exit_ticks is None else int(record.exit_ticks))
        # the prints can show the break-even stop inside the very minute the half exited;
        # the strategy checks its break-even stop only from the next candle
        same_minute_breakeven = (exited is not None and exited.kind == "breakeven_stop"
                                 and exited.scaled_in_exit_minute)
        if exited is not None and driver.in_position:
            if not same_minute_breakeven and (
                    not exited.account_failed or exited.kind != "account_failure"):
                setup = driver.reducer._setup
                raise EngineConsistencyError(
                    "the account closed a position the strategy still holds without a "
                    f"loss-limit liquidation ({run.pair_id}, {exited.kind} at "
                    f"{exited.fill_ticks} on the candle opening {bar.open_ts_utc}; strategy "
                    f"stop {setup.stop_ticks}, target {setup.tp_ticks}; candle "
                    f"{bar.open_ticks}/{bar.high_ticks}/{bar.low_ticks}/{bar.close_ticks})")
            driver.force_flat(counted=not same_minute_breakeven)
            row = run.open_rows.pop(ledger.trades[-1]["strategy_trade_id"], None)
            if row is not None:
                row["strategy_recorded_exit_kind"] = (
                    "breakeven_stop_within_the_scale_out_minute" if same_minute_breakeven
                    else "ended_by_account_liquidation")
                row["strategy_recorded_exit_ticks"] = None
        run.also_blocked += out.also_blocked_by_strategy
        if out.refused:
            reasons = [r for r in out.refused if r in BLOCK_REASONS.values()]
            if reasons:
                ledger.note_blocked(close_ns, reasons[0],
                                    "the strategy's entry signal was refused and discarded")
                driver.discard_refused_setup()
        if out.entry is not None:
            if gate is not None:
                raise EngineConsistencyError("Strategy-Core entered despite an account refusal")
            signal = out.entry
            ledger.open(ts_ns=close_ns, trade_ref=signal.trade_id, direction=signal.direction,
                        entry_ticks=signal.entry_ticks, stop_ticks=signal.stop_ticks,
                        target_ticks=signal.target_ticks, trading_day=day.trading_day,
                        strategy={"trade_id": signal.trade_id,
                                  "entry_chart": signal.entry_chart})
            if ledger.position is None:
                # the entry cost itself lost the account: the trade ends here
                ledger.trades[-1]["strategy_recorded_exit_kind"] = "ended_by_account_liquidation"
                ledger.trades[-1]["strategy_recorded_exit_ticks"] = None
                driver.force_flat()
        if exited is not None and run.open_rows:
            raise EngineConsistencyError("an account exit was not matched by the strategy")
    tail = driver.end_day(date.fromisoformat(day.trading_day), dataset_exhausted=day.exhausted)
    if tail and ledger is not None:
        raise EngineConsistencyError("a strategy position remained open after the daily close")
    for record in tail:
        run.strategy_trades.append(_core_trade(record, warmup=not day.is_evaluation))


def run_days(runs: list[PairRun], days: Iterable[DayInput],
             prints_factory: Callable[[DayInput], Callable[[], Any]], *,
             on_day_start: Callable[[int, DayInput], None] | None = None) -> None:
    """Run all pairs over the days; ``prints_factory(day)`` returns a lazy loader."""

    for index, day in enumerate(days):
        if on_day_start is not None:
            on_day_start(index, day)
        prints = prints_factory(day)
        for run in runs:
            run_one_day(run, day, prints)
        release = getattr(prints, "release", None)
        if release is not None:
            release()  # count the day's print evidence, then free its arrays
