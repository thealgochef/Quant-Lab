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

    ``trail_style`` currently supports only ``"eod_floor_realtime_breach"``:
    the drawdown floor ratchets on END-OF-DAY balance highs (floor = max of all
    prior EOD balances − ``trail_amount``) but a BREACH of the floor is checked
    in real time. ``trail_locks_at_start`` caps the floor at the starting
    balance (it never trails above it). ``dll_amount`` is the daily loss limit;
    ``dll_soft`` means a DLL hit halts the day (remaining trades skipped) and is
    NOT a bust. ``consistency_pct`` blocks a pass while the best single day
    exceeds ``pct`` × total profit. ``point_value`` is dollars per instrument
    point per contract.
    """

    starting_balance: float
    profit_target: float
    trail_amount: float
    trail_style: str
    trail_locks_at_start: bool
    dll_amount: float | None
    dll_soft: bool
    consistency_pct: float | None
    min_days: int | None
    point_value: float


@dataclass(frozen=True)
class WalkResult:
    """Outcome of one evaluation walk over a day sequence.

    ``verdict``: ``"pass"`` | ``"bust"`` | ``"incomplete"`` (data exhausted
    before either) | ``"no_trades"`` (empty input). ``days_to_outcome`` counts
    the day the verdict landed on (1-based); ``None`` while incomplete.
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
