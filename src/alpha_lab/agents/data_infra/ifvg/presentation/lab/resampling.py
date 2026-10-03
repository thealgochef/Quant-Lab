"""Risk and simulation: resampled paths of ONE configuration's own recorded funded trades.

``CALCULATIONS.md`` "Risk and simulation" (analytical corrections A2 and A5,
September 25, 2026). Every path DRAWS the saved net results of one
configuration's funded trades at one firm WITH REPLACEMENT — it is a bootstrap
path, not a reordering: a path can repeat some trades and leave out others, so
its total differs from the recorded total (a pure reordering of the same trades
would always end at the recorded total). These paths describe the recorded
trades under that sampling model; they are not a forecast. Two methods:

- ``blocks`` ("Keep streaks together"): blocks of 10 consecutive recorded trades
  drawn with replacement at random start positions and joined, then cut to the
  path length;
- ``shuffle`` ("Draw single trades"): single recorded trades drawn with
  replacement.

Every function takes an explicit ``seed`` and path count; the same inputs give
identical results (rule 9). :func:`payout_race` is a fixed closed-profit
boundary diagnostic (which of two fixed cumulative closed-profit levels a path
crosses first); it is not a payout or account-failure model. The conditional
resampling through each firm's ledger rules lives in :mod:`.firm_race`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "BLOCK_SIZE",
    "METHODS",
    "TIE_TOLERANCE",
    "DrawdownGrowth",
    "EquityFan",
    "PayoutRace",
    "StreakDistribution",
    "drawdown_growth",
    "equity_fan",
    "longest_losing_runs",
    "payout_race",
    "resample_paths",
    "running_fall",
    "sampling_words",
    "share_below_ties_half",
    "streak_distribution",
]

BLOCK_SIZE = 10
# correction A5: these are draws with replacement, never "shuffles" of the same trades
METHODS = {"blocks": "Keep streaks together", "shuffle": "Draw single trades"}
PERCENTILES = (5, 25, 50, 75, 95)
#: two values within half a cent of each other are a tie (money is exact cents)
TIE_TOLERANCE = 0.005


def share_below_ties_half(values: Sequence[float], actual: float, *,
                          tolerance: float = TIE_TOLERANCE) -> float | None:
    """Share of ``values`` below ``actual``, ties (within ``tolerance``) counting half.

    The one percentile rule for every "the recorded result is above X% of the
    resampled paths" sentence (correction A5). ``None`` when there are no values.
    """

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None
    below = float((data < actual - tolerance).sum())
    ties = float((np.abs(data - actual) <= tolerance).sum())
    return (below + 0.5 * ties) / data.size


def sampling_words(method: str, trades: int) -> str:
    """How a path draws the recorded trades: ``blocks of 10`` or ``one at a time``.

    With no more trades than one block, :func:`resample_paths` draws single
    trades for both methods, and the words say so.
    """

    if method == "blocks" and trades > BLOCK_SIZE:
        return f"blocks of {BLOCK_SIZE}"
    return "one at a time"


def resample_paths(values: Sequence[float], *, paths: int, length: int, method: str,
                   seed: int, block: int = BLOCK_SIZE) -> np.ndarray:
    """A (paths × length) array of trade results drawn WITH replacement.

    Not a permutation: a path can repeat some recorded trades and omit others.
    """

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("no trades to resample")
    rng = np.random.default_rng(seed)
    if method == "shuffle" or data.size <= block:
        return data[rng.integers(0, data.size, (paths, length))]
    if method != "blocks":
        raise ValueError(f"unknown resampling method {method!r}")
    blocks_needed = -(-length // block)
    starts = rng.integers(0, data.size - block + 1, (paths, blocks_needed))
    index = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(paths, -1)
    return data[index[:, :length]]


@dataclass(frozen=True)
class PayoutRace:
    """Fixed closed-profit boundary diagnostic (correction A2; name kept for callers).

    ``loss_limit`` is the LOWER boundary and ``trigger`` the UPPER boundary of
    cumulative closed profit; ``paid_*`` = the upper boundary crossed first,
    ``died_*`` = the lower boundary crossed first. Closed trade results only: no
    trailing or locking floor, no losses inside open trades, no payout requests,
    processing or receipts. It is not a payout or account-failure model.
    """

    paths: int
    seed: int
    method: str
    loss_limit: float
    trigger: float
    max_trades: int
    paid_share: float
    died_share: float
    still_going_share: float
    typical_to_payout: float | None
    typical_to_limit: float | None
    #: cumulative share paid / died at each trade 0..chart_trades
    paid_by_trade: tuple[float, ...] = ()
    died_by_trade: tuple[float, ...] = ()


def payout_race(values: Sequence[float], *, loss_limit: float = -2000.0,
                trigger: float = 2600.0, paths: int = 20_000, seed: int, method: str = "blocks",
                max_trades: int = 200, chart_trades: int = 40) -> PayoutRace:
    """Each path starts at $0 and runs until its cumulative closed profit crosses a boundary."""

    draws = resample_paths(values, paths=paths, length=max_trades, method=method, seed=seed)
    cumulative = np.cumsum(draws, axis=1)
    hit_paid = cumulative >= trigger
    hit_died = cumulative <= loss_limit
    never = max_trades + 1
    first_paid = np.where(hit_paid.any(axis=1), hit_paid.argmax(axis=1) + 1, never)
    first_died = np.where(hit_died.any(axis=1), hit_died.argmax(axis=1) + 1, never)
    paid = first_paid < first_died
    died = first_died < first_paid
    trades = np.arange(chart_trades + 1)
    paid_by = [(paid & (first_paid <= t)).mean() for t in trades]
    died_by = [(died & (first_died <= t)).mean() for t in trades]
    return PayoutRace(
        paths=paths, seed=seed, method=method, loss_limit=loss_limit, trigger=trigger,
        max_trades=max_trades,
        paid_share=float(paid.mean()), died_share=float(died.mean()),
        still_going_share=float(1.0 - paid.mean() - died.mean()),
        typical_to_payout=float(np.median(first_paid[paid])) if paid.any() else None,
        typical_to_limit=float(np.median(first_died[died])) if died.any() else None,
        paid_by_trade=tuple(float(v) for v in paid_by),
        died_by_trade=tuple(float(v) for v in died_by),
    )


def longest_losing_runs(draws: np.ndarray) -> np.ndarray:
    """Longest run of losing or break-even trades (≤ $0) in each path."""

    losing = draws <= 0
    best = np.zeros(draws.shape[0], dtype=int)
    current = np.zeros(draws.shape[0], dtype=int)
    for column in range(draws.shape[1]):
        current = np.where(losing[:, column], current + 1, 0)
        best = np.maximum(best, current)
    return best


def _max_drawdown_rows(cumulative: np.ndarray) -> np.ndarray:
    with_start = np.concatenate([np.zeros((cumulative.shape[0], 1)), cumulative], axis=1)
    return (np.maximum.accumulate(with_start, axis=1) - with_start).max(axis=1)


@dataclass(frozen=True)
class EquityFan:
    method: str
    paths: int
    seed: int
    trades: int
    #: percentile → values at trade 0..trades (trade 0 is $0)
    bands: dict[int, tuple[float, ...]] = field(default_factory=dict)
    samples: tuple[tuple[float, ...], ...] = ()
    end_values: tuple[float, ...] = ()
    bad_end: float = 0.0
    typical_end: float = 0.0
    good_end: float = 0.0
    bad_case_drawdown: float = 0.0
    below_zero_share: float = 0.0
    streak_typical: float = 0.0
    streak_bad: float = 0.0
    streaks: tuple[int, ...] = ()


def equity_fan(values: Sequence[float], *, method: str, paths: int = 20_000, seed: int,
               sample_paths: int = 10) -> EquityFan:
    """Percentile bands of the trade path, sample paths, end values and streaks."""

    data = np.asarray(values, dtype=float)
    draws = resample_paths(data, paths=paths, length=data.size, method=method, seed=seed)
    cumulative = np.cumsum(draws, axis=1)
    with_start = np.concatenate([np.zeros((paths, 1)), cumulative], axis=1)
    bands = {p: tuple(float(v) for v in np.percentile(with_start, p, axis=0))
             for p in PERCENTILES}
    ends = cumulative[:, -1]
    drawdowns = _max_drawdown_rows(cumulative)
    streaks = longest_losing_runs(draws)
    pick = np.linspace(0, paths - 1, sample_paths).astype(int)
    return EquityFan(
        method=method, paths=paths, seed=seed, trades=int(data.size), bands=bands,
        samples=tuple(tuple(float(v) for v in with_start[i]) for i in pick),
        end_values=tuple(float(v) for v in ends),
        bad_end=float(np.percentile(ends, 5)), typical_end=float(np.median(ends)),
        good_end=float(np.percentile(ends, 95)),
        bad_case_drawdown=float(np.percentile(drawdowns, 95)),
        below_zero_share=float((ends < 0).mean()),
        streak_typical=float(np.median(streaks)), streak_bad=float(np.percentile(streaks, 95)),
        streaks=tuple(int(v) for v in streaks),
    )


@dataclass(frozen=True)
class StreakDistribution:
    #: run length → share of paths
    shares: dict[int, float]
    typical: float
    bad: float


def streak_distribution(streaks: Sequence[int]) -> StreakDistribution:
    values = np.asarray(streaks, dtype=int)
    lengths, counts = np.unique(values, return_counts=True)
    return StreakDistribution(
        shares={int(k): float(c / values.size) for k, c in zip(lengths, counts, strict=True)},
        typical=float(np.median(values)), bad=float(np.percentile(values, 95)))


def running_fall(cumulative: np.ndarray) -> np.ndarray:
    """Largest fall so far from a previous high of each cumulative closed-profit path.

    ``cumulative`` is (paths × points) or one path, starting at $0. Running high
    minus value, then its running maximum. This is a trading-profit measure only:
    it knows nothing of an account floor, its lock, withdrawals or losses inside
    an open trade (correction A2).
    """

    values = np.asarray(cumulative, dtype=float)
    axis = values.ndim - 1
    return np.maximum.accumulate(np.maximum.accumulate(values, axis=axis) - values, axis=axis)


@dataclass(frozen=True)
class DrawdownGrowth:
    """Sampled closed-profit drawdown from a previous high (correction A2).

    Resampled cumulative CLOSED profit with no withdrawals and no account floor.
    ``touched_share`` (name kept) is the share of paths whose closed profit has
    fallen at least ``limit`` below a previous high by each trade. It is NOT the
    share of funded accounts that fail: a floor that has locked (at $0 or +$100)
    need not end an account after such a fall, and a loss inside an open trade can
    end an account whose closed result recovers.
    """

    method: str
    paths: int
    seed: int
    #: the fall size counted (dollars); defaults to the firm's saved loss allowance
    limit: float
    #: per trade 0..horizon: median, 75th and 95th percentile of the largest fall so far
    typical: tuple[float, ...]
    worse_1_in_4: tuple[float, ...]
    worse_1_in_20: tuple[float, ...]
    #: per trade 0..horizon: share of paths whose largest fall so far is at least ``limit``
    touched_share: tuple[float, ...]
    #: per checkpoint trade count: (5th, median, 95th) of cumulative closed profit
    profit_at: dict[int, tuple[float, float, float]] = field(default_factory=dict)


def drawdown_growth(values: Sequence[float], *, paths: int = 20_000, seed: int,
                    method: str = "blocks", horizon: int = 100, limit: float = 2000.0,
                    checkpoints: Sequence[int] = (10, 20, 40, 100)) -> DrawdownGrowth:
    """Largest closed-profit fall from a previous high, trades 0..horizon, no withdrawals."""

    draws = resample_paths(values, paths=paths, length=horizon, method=method, seed=seed)
    cumulative = np.concatenate([np.zeros((paths, 1)), np.cumsum(draws, axis=1)], axis=1)
    worst = running_fall(cumulative)
    profit_at = {int(c): (float(np.percentile(cumulative[:, c], 5)),
                          float(np.median(cumulative[:, c])),
                          float(np.percentile(cumulative[:, c], 95)))
                 for c in checkpoints if c <= horizon}
    return DrawdownGrowth(
        method=method, paths=paths, seed=seed, limit=limit,
        typical=tuple(float(v) for v in np.median(worst, axis=0)),
        worse_1_in_4=tuple(float(v) for v in np.percentile(worst, 75, axis=0)),
        worse_1_in_20=tuple(float(v) for v in np.percentile(worst, 95, axis=0)),
        touched_share=tuple(float(v) for v in (worst >= limit).mean(axis=0)),
        profit_at=profit_at,
    )
