"""The evaluation walk: barrier-options semantics over per-trade equity paths.

Walks days in order, trades in intra-day order, at 1 contract. Balance updates
on realized points (fill column selectable). The trailing floor ratchets on EOD
balance highs (capped at the starting balance when the ruleset locks); breach
is checked in real time at every equity observation the breach mode provides:

- ``realized_only`` — equity observed at trade closes and EOD.
- ``unrealized_adverse_first`` — each trade's path first visits
  entry − MAE, then entry + MFE, then settles at its realized points. On the
  adverse leg the floor and the daily-loss level are BARRIERS on one
  monotonically falling equity path: whichever level is higher is touched
  first. A soft-DLL touch force-closes the trade AT the DLL level (day halted,
  not a bust); a floor touch is a bust. Trades without excursions (``mae_pts``
  is None/NaN) contribute realized-only observations.

PASS is evaluated at EOD: total profit ≥ target, min-days satisfied, and (when
a consistency rule exists) best single day ≤ pct × total — otherwise the walk
keeps going and later days can dilute the ratio.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import date

from alpha_lab.propsim.models import Ruleset, TradePath, WalkResult

BREACH_MODES = ("realized_only", "unrealized_adverse_first")
FILL_COLUMNS = ("optimistic", "conservative")

_SUPPORTED_TRAIL_STYLES = ("eod_floor_realtime_breach",)


def _excursion(value: float | None) -> float | None:
    """A usable non-negative excursion magnitude, or None (NaN-safe)."""
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return abs(value)


class EvaluationWalk:
    """Stateful walk over trading days; feed days until a verdict lands."""

    def __init__(self, ruleset: Ruleset, *, column: str, breach_mode: str) -> None:
        if ruleset.trail_style not in _SUPPORTED_TRAIL_STYLES:
            msg = (
                f"Unsupported trail_style {ruleset.trail_style!r}. "
                f"Supported: {', '.join(_SUPPORTED_TRAIL_STYLES)}"
            )
            raise ValueError(msg)
        if column not in FILL_COLUMNS:
            msg = f"Unknown fill column {column!r}. Known: {', '.join(FILL_COLUMNS)}"
            raise ValueError(msg)
        if breach_mode not in BREACH_MODES:
            msg = f"Unknown breach mode {breach_mode!r}. Known: {', '.join(BREACH_MODES)}"
            raise ValueError(msg)
        self._rs = ruleset
        self._column = column
        self._mode = breach_mode
        self.balance = ruleset.starting_balance
        # The starting balance counts as the day-0 EOD: day 1's floor is
        # start − trail.
        self._eod_peak = ruleset.starting_balance
        self.floor = self._compute_floor()
        self._day_pnls: list[float] = []
        self._days = 0
        self._halted_days = 0
        self._skipped_trades = 0
        self._verdict: str | None = None
        self._bust_reason: str | None = None
        self._days_to_outcome: int | None = None
        self._min_floor_distance = ruleset.starting_balance - self.floor

    def _compute_floor(self) -> float:
        floor = self._eod_peak - self._rs.trail_amount
        if self._rs.trail_locks_at_start:
            floor = min(floor, self._rs.starting_balance)
        return floor

    def _points(self, trade: TradePath) -> float:
        if self._column == "conservative":
            return trade.points_conservative
        return trade.points_optimistic

    def _observe(self, equity: float) -> None:
        self._min_floor_distance = min(self._min_floor_distance, equity - self.floor)

    def _bust(self, reason: str, day_start_balance: float) -> str:
        self._verdict = "bust"
        self._bust_reason = reason
        self._days_to_outcome = self._days + 1
        # Bank the partial day so best-day bookkeeping stays coherent.
        self._day_pnls.append(self.balance - day_start_balance)
        self._days += 1
        return "bust"

    def play_day(self, trades: Sequence[TradePath]) -> str | None:
        """Play one trading day; returns the verdict if it lands, else None."""
        if self._verdict is not None:
            msg = f"walk already terminated with verdict {self._verdict!r}"
            raise RuntimeError(msg)
        rs = self._rs
        day_start = self.balance
        halted = False
        for trade in trades:
            if halted:
                self._skipped_trades += 1
                continue
            pnl = self._points(trade) * rs.point_value
            mae = _excursion(trade.mae_pts) if self._mode == "unrealized_adverse_first" else None
            if mae is not None:
                adverse_equity = self.balance - mae * rs.point_value
                dll_level = (
                    day_start - rs.dll_amount if rs.dll_amount is not None else None
                )
                floor_crossed = adverse_equity <= self.floor
                dll_crossed = dll_level is not None and adverse_equity <= dll_level
                if floor_crossed and (not dll_crossed or self.floor >= dll_level):
                    # The floor is the first (highest) barrier on the falling leg.
                    self._observe(adverse_equity)
                    return self._bust("trailing_floor", day_start)
                if dll_crossed:
                    # The DLL is touched first: force-close AT the DLL level —
                    # the deeper excursion never happens because we are flat.
                    self._observe(dll_level)
                    self.balance = dll_level
                    if not rs.dll_soft:
                        return self._bust("daily_loss_limit", day_start)
                    halted = True
                    continue
                self._observe(adverse_equity)
            # Settle at the realized points.
            self.balance += pnl
            self._observe(self.balance)
            if self.balance <= self.floor:
                return self._bust("trailing_floor", day_start)
            if rs.dll_amount is not None and (self.balance - day_start) <= -rs.dll_amount:
                if not rs.dll_soft:
                    return self._bust("daily_loss_limit", day_start)
                halted = True
        # End of day.
        self._days += 1
        if halted:
            self._halted_days += 1
        self._day_pnls.append(self.balance - day_start)
        self._observe(self.balance)
        if self.balance <= self.floor:
            self._verdict = "bust"
            self._bust_reason = "trailing_floor"
            self._days_to_outcome = self._days
            return "bust"
        total = self.balance - rs.starting_balance
        if total >= rs.profit_target and (rs.min_days is None or self._days >= rs.min_days):
            best_day = max(self._day_pnls)
            if rs.consistency_pct is None or best_day <= (rs.consistency_pct / 100.0) * total:
                self._verdict = "pass"
                self._days_to_outcome = self._days
                return "pass"
        # Ratchet the floor on the EOD balance (never moves down).
        self._eod_peak = max(self._eod_peak, self.balance)
        self.floor = max(self.floor, self._compute_floor())
        self._observe(self.balance)
        return None

    @property
    def verdict(self) -> str | None:
        return self._verdict

    def result(self) -> WalkResult:
        total = self.balance - self._rs.starting_balance
        best_day_ratio: float | None = None
        if self._day_pnls and total > 0:
            best_day_ratio = max(self._day_pnls) / total
        verdict = self._verdict or ("incomplete" if self._days else "no_trades")
        return WalkResult(
            verdict=verdict,
            days_to_outcome=self._days_to_outcome,
            bust_reason=self._bust_reason,
            best_day_ratio=best_day_ratio,
            min_floor_distance=self._min_floor_distance if self._days else None,
            final_balance=self.balance,
            days_walked=self._days,
            halted_days=self._halted_days,
            skipped_trades=self._skipped_trades,
        )


def group_by_day(trades: Iterable[TradePath]) -> list[tuple[date, list[TradePath]]]:
    """Chronological (day, trades-in-entry-order) blocks for walking/resampling."""
    by_day: dict[date, list[TradePath]] = {}
    for trade in trades:
        by_day.setdefault(trade.day, []).append(trade)
    return [
        (day, sorted(day_trades, key=lambda t: t.entry_ts))
        for day, day_trades in sorted(by_day.items())
    ]


def walk_days(
    day_blocks: Sequence[tuple[date, Sequence[TradePath]]],
    ruleset: Ruleset,
    *,
    column: str,
    breach_mode: str,
) -> WalkResult:
    """The as-sequenced historical walk over the given day blocks."""
    walk = EvaluationWalk(ruleset, column=column, breach_mode=breach_mode)
    for _, day_trades in day_blocks:
        if walk.play_day(day_trades) is not None:
            break
    return walk.result()
