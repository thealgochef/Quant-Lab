"""Account-driven Strategy-Core replay for ONE configuration and ONE account sequence.

Strategy-Core is used unmodified. Its IFVG reducer holds one setup slot and
steps one completed one-minute candle at a time, so an external account can
steer it through two narrow seams, both exercised by tests against the pinned
Core commit:

* **Admission.** Before a candle is stepped, the account's current refusal
  reason (payout protection or payout processing) is appended to the reducer's
  own execution-time admission check (``execution_block_reasons``), the check
  Core already runs at the entry fill for session windows, market hours and the
  mandatory daily-close lock. A refused entry is recorded by Core as a blocked
  candidate; the driver then DISCARDS that setup so it can never be filled
  later from a stale signal (specification: disarm stale entry signals).
* **Account liquidation.** When the account layer closes a position earlier
  than the strategy would have (a loss-limit breach), the driver clears the
  reducer's position slot after that candle. Core therefore never continues
  the dead trade to its old stop or target, and it may form a NEW setup from
  the next candle onward (never on the liquidation candle itself, matching
  Core's own "no new setup on a resolution bar" rule).

Everything else — detectors, registries, swings, gap validity, day seeds — is
Core's own state, carried from day to day through Core's ``end_seed``. Each
driver owns its state; drivers never share a reducer. With no account
interference the driver reproduces Core's ``run_day`` chain exactly (tested
against the archived daily-close study).
"""

from __future__ import annotations

import base64
import pickle
from dataclasses import dataclass
from datetime import date
from typing import Any

__all__ = ["CoreStrategyDriver", "StepOutcome", "EntrySignal", "CORE_SEAMS"]

CORE_SEAMS = (
    "IfvgReducer.execution_block_reasons (per-instance admission extension)",
    "IfvgReducer._setup (cleared to None for account liquidation / refused setups)",
)

_SECONDS_LABEL = {60: "one-minute", 180: "three-minute", 300: "five-minute",
                  600: "ten-minute", 900: "fifteen-minute", 1800: "thirty-minute",
                  3600: "one-hour", 14400: "four-hour"}


def chart_name(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    return _SECONDS_LABEL.get(int(seconds), f"{int(seconds)}-second")


@dataclass(frozen=True)
class EntrySignal:
    trade_id: str
    direction: str
    entry_ticks: int
    stop_ticks: int
    target_ticks: int
    entry_chart: str


@dataclass(frozen=True)
class StepOutcome:
    entry: EntrySignal | None
    core_exits: tuple[Any, ...]  # Core ExecutedTradeRecord(s) resolved on this candle
    refused: tuple[str, ...]  # entries Core would have filled but the account refused
    #: candidates Core blocked for its own reasons as well (never counted as refusals)
    also_blocked_by_strategy: int = 0


class CoreStrategyDriver:
    def __init__(self, section: Any, *, tick_size: float) -> None:
        self.section = section
        self.tick_size = tick_size
        self.seed = None
        self._orch = None
        self._gate: str | None = None
        self.forced_flat = 0
        self.discarded_setups = 0

    # ── day lifecycle ─────────────────────────────────────────────────────
    def begin_day(self, bars_by_tf: dict[int, list], levels_for) -> list:
        from strategy_core.strategies.ifvg_smc.replay import DayOrchestrator

        orch = DayOrchestrator(section=self.section, seed=self.seed,
                               tick_size=self.tick_size, levels_for=levels_for)
        orch.reset_funnel()
        for tf, bars in bars_by_tf.items():
            if tf == 60:
                continue
            for bar in bars:
                orch.on_higher_tf_bar(bar)
        reducer = orch._reducer
        original = type(reducer).execution_block_reasons

        def admission(execution_ts, *, session_doc, _r=reducer, _orig=original):
            blocks = _orig(_r, execution_ts, session_doc=session_doc)
            return blocks + ((self._gate,) if self._gate else ())

        reducer.execution_block_reasons = admission
        self._orch = orch
        return list(bars_by_tf.get(60, ()))

    def end_day(self, trading_day: date, *, dataset_exhausted: bool = False) -> list:
        orch = self._orch
        tail = list(orch.finalize_day(trading_day))
        if dataset_exhausted:
            tail.extend(orch.finalize_dataset(trading_day))
        self.seed = orch.end_seed(trading_day)
        self._orch = None
        return [e.record for e in tail if e.kind == "executed_trade"]

    # ── per candle ────────────────────────────────────────────────────────
    @property
    def reducer(self):
        return self._orch._reducer

    @property
    def in_position(self) -> bool:
        return bool(self._orch._reducer.active_trade_count)

    def step(self, bar, gate: str | None) -> StepOutcome:
        self._gate = gate
        try:
            emitted = self._orch.on_decision_bar(bar)
        finally:
            self._gate = None
        exits = tuple(e.record for e in emitted if e.kind == "executed_trade")
        refused: list[str] = []
        also_blocked = 0
        entry = None
        for emission in emitted:
            if emission.kind != "entry_candidate" or gate is None:
                continue
            blocks = tuple(emission.record.block_reasons or ())
            # only a candidate whose SOLE block is the account's is an entry the strategy
            # would have made; Core's own blocks (unratified retest, outside the entry
            # hours, daily cap...) leave its setup exactly as Core would
            if blocks == (gate,):
                refused.append(gate)
            elif gate in blocks:
                also_blocked += 1
        setup = self._orch._reducer._setup
        if (setup is not None and setup.phase == "S5"
                and setup.entry_ts_utc == bar.availability_ts_utc
                and any(e.kind == "eligible_decision" for e in emitted)):
            geometry = setup.geometry
            entry_tf = None
            if geometry is not None:
                entry_tf = (geometry.entry_fvg.timeframe_seconds
                            if geometry.entry_fvg is not None else 60)
            direction = "long" if int(setup.stop_ticks) < int(setup.entry_ticks) else "short"
            label = str(getattr(setup.direction, "value", setup.direction)).lower()
            if label not in (direction, "bullish" if direction == "long" else "bearish",
                             "bull" if direction == "long" else "bear"):
                raise AssertionError(f"direction {label!r} contradicts the stop placement")
            entry = EntrySignal(
                trade_id=str(setup.trade_id), direction=direction,
                entry_ticks=int(setup.entry_ticks), stop_ticks=int(setup.stop_ticks),
                target_ticks=int(setup.tp_ticks),
                entry_chart=_entry_chart(geometry, entry_tf),
            )
        return StepOutcome(entry=entry, core_exits=exits, refused=tuple(refused),
                           also_blocked_by_strategy=also_blocked)

    def force_flat(self, *, counted: bool = True) -> None:
        """The account closed the position: Core must not continue that trade.

        ``counted`` is False for the scale-out remainder that the prints close at
        break-even inside the scale-out minute (not an account liquidation).
        """

        reducer = self._orch._reducer
        if reducer._setup is None or reducer._setup.phase != "S5":
            raise AssertionError("force_flat without an open strategy position")
        reducer._setup = None
        if counted:
            self.forced_flat += 1
        else:
            self.breakeven_cleared = getattr(self, "breakeven_cleared", 0) + 1

    def discard_refused_setup(self) -> None:
        """A refused entry disarms its setup (never filled later from a stale signal)."""

        reducer = self._orch._reducer
        setup = reducer._setup
        if setup is not None and setup.phase == "S4":
            reducer._setup = None
            self.discarded_setups += 1

    # ── checkpoint ────────────────────────────────────────────────────────
    def checkpoint(self) -> dict[str, Any]:
        if self._orch is not None:
            raise AssertionError("checkpoints are taken between trading days")
        return {"seed": base64.b64encode(pickle.dumps(self.seed)).decode("ascii"),
                "forced_flat": self.forced_flat, "discarded_setups": self.discarded_setups}

    def restore(self, state: dict[str, Any]) -> None:
        self.seed = pickle.loads(base64.b64decode(state["seed"]))  # noqa: S301 - own checkpoint
        self.forced_flat = state["forced_flat"]
        self.discarded_setups = state["discarded_setups"]

    def seed_hash(self) -> str | None:
        from strategy_core.strategies.ifvg_smc.state import seed_hash

        return None if self.seed is None else seed_hash(self.seed)


def _entry_chart(geometry, entry_tf) -> str:
    if geometry is None:
        return "unknown"
    return (f"{chart_name(geometry.htf.timeframe_seconds)} gap, "
            f"{chart_name(geometry.parent.timeframe_seconds)} parent chart, "
            f"{chart_name(entry_tf)} entry chart")
