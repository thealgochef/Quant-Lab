"""Strategy executions and the ordered price paths accounts are checked on.

An :class:`ExecutionPath` holds every price observation made while the
strategy's position EXISTS: strictly after the entry fill, up to and including
the observation at which the strategy's own exit happens. Two evidence
classes exist and are never conflated:

* ``ordered_trade_prints`` — actual exchange trade prints from local MBP-1
  files, in source order. Consecutive prints are JUMPS: nothing is assumed
  between them, and a liquidation fills at the first print at/through the
  floor (gap-through), never at a better threshold price.
* ``minute_bars_adverse_first`` / ``minute_bars_favorable_first`` — a declared
  APPROXIMATION from one-minute candles. Within a candle, legs between the
  open, the two extremes and the close are treated as continuous, in the
  declared order; candle-to-candle moves are jumps. Results built on these
  paths are labeled approximate wherever the order can change survival.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from alpha_lab.agents.data_infra.ifvg.search.identities import FrozenContract

__all__ = [
    "PathFidelity",
    "StrategyExecution",
    "ExecutionPath",
    "OBS_PRINT",
    "OBS_BAR_POINT",
    "OBS_EXIT_TOUCH",
    "minute_scenario_path",
    "FIDELITY_LABELS",
]

PathFidelity = Literal[
    "ordered_trade_prints",
    "minute_bars_adverse_first",
    "minute_bars_favorable_first",
]

FIDELITY_LABELS: dict[str, str] = {
    "ordered_trade_prints": "Ordered exchange trade prints (exact relative to recorded prints)",
    "minute_bars_adverse_first": "One-minute candles, losing side assumed first (approximation)",
    "minute_bars_favorable_first": "One-minute candles, winning side assumed first (approximation)",
}

OBS_PRINT = 0
OBS_BAR_POINT = 1
OBS_EXIT_TOUCH = 2


class StrategyExecution(FrozenContract):
    """One actual strategy execution from a completed, verified study."""

    trade_id: str
    trading_day: str
    direction: Literal["long", "short"]
    entry_ts_utc: str
    entry_ticks: int
    stop_ticks: int
    target_ticks: int | None
    exit_ts_utc: str
    exit_ticks: int
    exit_reason: Literal["stop", "target", "scheduled_close"]
    is_warmup: bool
    entry_chart: str
    tick_size_points: str = "0.25"

    @property
    def sign(self) -> int:
        return 1 if self.direction == "long" else -1


@dataclass(frozen=True)
class ExecutionPath:
    """Observations while the position exists (numpy arrays, same length)."""

    trade_id: str
    fidelity: PathFidelity
    ts_ns: np.ndarray  # int64
    price_ticks: np.ndarray  # int64
    continuous: np.ndarray  # bool: the leg FROM the previous point is continuous
    kind: np.ndarray  # int8: OBS_*
    strategy_exit_ns: int
    source_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        n = len(self.ts_ns)
        if not (len(self.price_ticks) == len(self.continuous) == len(self.kind) == n):
            raise ValueError("path arrays differ in length")
        if n and np.any(np.diff(self.ts_ns) < 0):
            raise ValueError(f"path {self.trade_id} is not time ordered")

    @property
    def is_approximate(self) -> bool:
        return self.fidelity != "ordered_trade_prints"


def minute_scenario_path(
    execution: StrategyExecution,
    bars: list[tuple[int, int, int, int, int, int]],
    *,
    order: Literal["adverse_first", "favorable_first"],
    source_ids: tuple[str, ...] = (),
) -> ExecutionPath:
    """Build the declared one-minute approximation.

    ``bars`` are complete one-minute candles AFTER the entry fill, in time
    order, as ``(open_ts_ns, close_ts_ns, open, high, low, close)`` ticks, up
    to and including the strategy's resolution candle. The resolution candle
    is truncated at the strategy's recorded exit level (stop/target), reached
    along the declared order; scheduled exits keep the full deadline candle.
    """

    sign = execution.sign
    ts: list[int] = []
    px: list[int] = []
    cont: list[bool] = []
    kinds: list[int] = []

    def emit(t: int, p: int, continuous: bool, kind: int = OBS_BAR_POINT) -> None:
        ts.append(t)
        px.append(p)
        cont.append(continuous)
        kinds.append(kind)

    exit_level = execution.exit_ticks
    last_index = len(bars) - 1
    for index, (open_ns, close_ns, o, h, low, c) in enumerate(bars):
        adverse = low if sign > 0 else h
        favorable = h if sign > 0 else low
        legs = (adverse, favorable) if order == "adverse_first" else (favorable, adverse)
        # points inside one candle share its window; spread them for ordering
        step = max(1, (close_ns - open_ns) // 4)
        emit(open_ns, o, False)
        resolving = index == last_index and execution.exit_reason != "scheduled_close"
        if resolving:
            # the exit level is reached on the leg that moves toward it
            toward_stop = execution.exit_reason == "stop"
            first, second = legs
            first_is_adverse = first == adverse
            if toward_stop == first_is_adverse:
                emit(open_ns + step, exit_level, True, OBS_EXIT_TOUCH)
            else:
                emit(open_ns + step, first, True)
                emit(open_ns + 2 * step, exit_level, True, OBS_EXIT_TOUCH)
            break
        emit(open_ns + step, legs[0], True)
        emit(open_ns + 2 * step, legs[1], True)
        emit(close_ns, c, True)
    strategy_exit_ns = ts[-1] if ts else 0
    if execution.exit_reason == "scheduled_close" and bars:
        strategy_exit_ns = bars[-1][1]
    return ExecutionPath(
        trade_id=execution.trade_id,
        fidelity=(
            "minute_bars_adverse_first"
            if order == "adverse_first"
            else "minute_bars_favorable_first"
        ),
        ts_ns=np.asarray(ts, dtype=np.int64),
        price_ticks=np.asarray(px, dtype=np.int64),
        continuous=np.asarray(cont, dtype=bool),
        kind=np.asarray(kinds, dtype=np.int8),
        strategy_exit_ns=strategy_exit_ns,
        source_ids=source_ids,
    )
