"""Typed containers for the prop-firm evaluation walker."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class TradePath:
    """One completed trade, normalized from any source.

    Points are per-contract instrument points (signed: positive = profit on the
    trade). ``mfe_pts``/``mae_pts`` are the trade's maximum favorable/adverse
    excursions as non-negative point magnitudes relative to entry; ``None`` when
    the source cannot supply them — the unrealized breach mode then treats that
    trade realized-only. ``day`` uses the 18:00 ET trading-day roll (a close
    after the roll still belongs to its OPEN's trading day: trades are dated by
    entry).
    """

    day: date
    entry_ts: datetime
    points_optimistic: float
    points_conservative: float
    mfe_pts: float | None
    mae_pts: float | None
    resolution: str | None


@dataclass(frozen=True)
class Ruleset:
    """A prop-firm evaluation ruleset. Presets are data (see ``presets``).

    ``trail_style`` selects the drawdown-floor mechanic (breach is always
    checked in real time at every equity observation the breach mode provides):

    - ``"eod_floor_realtime_breach"`` — the floor ratchets on END-OF-DAY
      balance highs (floor = max of all prior EOD balances − ``trail_amount``).
    - ``"intraday_peak_trail"`` — the floor trails PEAK equity including
      unrealized: in the unrealized breach modes each trade's favorable leg
      (entry + MFE observed) raises the peak and floor = max(floor,
      peak − ``trail_amount``); in ``realized_only`` the peak updates from
      realized equity at trade closes + EOD.
    - ``"static_floor"`` — the floor is fixed at start − ``trail_amount``
      forever (never ratchets).

    ``trail_locks_at_start`` caps the floor at the starting balance (it never
    trails above it). ``dll_amount`` is the daily loss limit; ``dll_hard`` True
    means a DLL touch is a BUST — False (soft) halts the day (remaining trades
    skipped) without busting. ``consistency_pct`` blocks a pass while the best
    single day exceeds ``pct`` × total profit. ``max_eval_days`` is the day
    budget: a walk still unresolved after that many days lands ``"expired"``
    on day max+1 (None = no expiry). ``point_value`` is dollars per instrument
    point per contract.
    """

    starting_balance: float
    profit_target: float
    trail_amount: float
    trail_style: str
    trail_locks_at_start: bool
    dll_amount: float | None
    dll_hard: bool
    consistency_pct: float | None
    min_days: int | None
    max_eval_days: int | None
    point_value: float


@dataclass(frozen=True)
class WalkResult:
    """Outcome of one evaluation walk over a day sequence.

    ``verdict``: ``"pass"`` | ``"bust"`` | ``"expired"`` (the ruleset's
    ``max_eval_days`` budget ran out) | ``"incomplete"`` (data exhausted
    before any verdict) | ``"no_trades"`` (empty input). ``days_to_outcome``
    counts the day the verdict landed on (1-based; ``"expired"`` lands on day
    max+1, the first day the eval may no longer trade); ``None`` while
    incomplete.
    ``best_day_ratio`` = best single-day P&L / total profit (``None`` when the
    total is not positive). ``min_floor_distance`` is the closest equity ever
    came to the active floor, in dollars (``None`` for an empty walk).
    """

    verdict: str
    days_to_outcome: int | None
    bust_reason: str | None
    best_day_ratio: float | None
    min_floor_distance: float | None
    final_balance: float
    days_walked: int
    halted_days: int
    skipped_trades: int
