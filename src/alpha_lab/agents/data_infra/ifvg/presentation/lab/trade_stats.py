"""Trades tab measures (``CALCULATIONS.md`` "Trades tab") over one configuration at one firm.

All inputs are the saved funded trade rows in execution order (every account of
the configuration at that firm). A winner has a net result above $0; a loser
is at or below $0. Nothing is recomputed from prices.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "RESULT_BINS",
    "DrawdownEpisodes",
    "Excursion",
    "PerformanceColumn",
    "close_reason",
    "drawdown_episodes",
    "excursions",
    "excursion_counts",
    "fills_text",
    "longest_streak",
    "performance_column",
    "result_distribution",
]

#: (label, low inclusive, high exclusive) — under −$750, then $250 steps to
#: $1,000, then $1k–2k, $2k–4k and over $4k
RESULT_BINS: tuple[tuple[str, float, float], ...] = (
    ("under −$750", -np.inf, -750.0),
    ("−$750", -750.0, -500.0),
    ("−$500", -500.0, -250.0),
    ("−$250", -250.0, 0.0),
    ("$0", 0.0, 250.0),
    ("$250", 250.0, 500.0),
    ("$500", 500.0, 750.0),
    ("$750", 750.0, 1000.0),
    ("$1k–2k", 1000.0, 2000.0),
    ("$2k–4k", 2000.0, 4000.0),
    ("over $4k", 4000.0, np.inf),
)


def _results(trades: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(t.get("net_pnl_usd") or 0.0) for t in trades], dtype=float)


def result_distribution(trades: Sequence[dict[str, Any]]) -> list[tuple[str, int, float, float]]:
    """(label, count, low, high) per bin; a bin holds results in [low, high)."""

    values = _results(trades)
    return [(label, int(((values >= lo) & (values < hi)).sum()), lo, hi)
            for label, lo, hi in RESULT_BINS]


def longest_streak(mask: Sequence[bool]) -> int:
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


@dataclass(frozen=True)
class DrawdownEpisodes:
    count: int
    #: trades spent below the previous high, averaged over episodes
    average_length: float | None
    average_depth: float | None
    worst_depth: float | None
    longest_length: int | None


def drawdown_episodes(results: Sequence[float]) -> DrawdownEpisodes:
    """Episodes where the trade path sits below its running high.

    An episode starts at a high and ends when the path gets back to that high;
    its length is the number of trades spent below it (an unrecovered episode
    runs to the last trade), its depth the largest fall from that high.
    """

    path = np.concatenate([[0.0], np.cumsum(np.asarray(results, dtype=float))])
    lengths: list[int] = []
    depths: list[float] = []
    index, n = 1, len(path)
    while index < n:
        peak_at = index - 1
        peak = float(np.max(path[: index]))
        if path[index] < peak:
            start = peak_at
            end = index
            deepest = 0.0
            while end < n and path[end] < peak:
                deepest = max(deepest, peak - float(path[end]))
                end += 1
            lengths.append(end - start - 1)
            depths.append(deepest)
            index = end
        else:
            index += 1
    if not lengths:
        return DrawdownEpisodes(0, None, None, None, None)
    return DrawdownEpisodes(count=len(lengths), average_length=float(np.mean(lengths)),
                            average_depth=float(np.mean(depths)), worst_depth=max(depths),
                            longest_length=max(lengths))


@dataclass(frozen=True)
class PerformanceColumn:
    """One column (All, Long or Short) of the performance summary; None when empty."""

    trades: int
    win_rate: float | None
    gross_profit: float | None
    gross_loss: float | None
    average_win: float | None
    average_loss: float | None
    largest_win: float | None
    largest_loss: float | None
    longest_winning: int | None
    longest_losing: int | None
    drawdowns: DrawdownEpisodes | None


def performance_column(trades: Sequence[dict[str, Any]], direction: str | None = None
                       ) -> PerformanceColumn:
    rows = [t for t in trades if direction is None or t.get("direction") == direction]
    values = _results(rows)
    if values.size == 0:
        return PerformanceColumn(0, None, None, None, None, None, None, None, None, None, None)
    wins, losses = values[values > 0], values[values <= 0]
    return PerformanceColumn(
        trades=int(values.size),
        win_rate=float(wins.size / values.size),
        gross_profit=float(wins.sum()),
        gross_loss=float(losses.sum()),
        average_win=float(wins.mean()) if wins.size else None,
        average_loss=float(losses.mean()) if losses.size else None,
        largest_win=float(values.max()),
        largest_loss=float(values.min()),
        longest_winning=longest_streak(values > 0),
        longest_losing=longest_streak(values <= 0),
        drawdowns=drawdown_episodes(values),
    )


@dataclass(frozen=True)
class Excursion:
    worst: float  # balance before − lowest equity during the trade
    best: float  # highest equity during the trade − balance before
    result: float
    winner: bool


def excursions(trades: Sequence[dict[str, Any]]) -> list[Excursion]:
    out = []
    for t in trades:
        before = float(t.get("balance_before_usd") or 0.0)
        low = t.get("min_equity_usd")
        high = t.get("max_equity_usd")
        if low is None or high is None:
            continue
        result = float(t.get("net_pnl_usd") or 0.0)
        out.append(Excursion(worst=before - float(low), best=float(high) - before,
                             result=result, winner=result > 0))
    return out


def excursion_counts(items: Sequence[Excursion], threshold: float = 250.0
                     ) -> tuple[int, int, int, int]:
    """Counts: (losers never ``threshold`` in profit, losers,
    winners never ``threshold`` against, winners)."""

    losers = [e for e in items if not e.winner]
    winners = [e for e in items if e.winner]
    return (sum(e.best < threshold for e in losers), len(losers),
            sum(e.worst < threshold for e in winners), len(winners))


_REASONS = {
    "stop": "Stop",
    "breakeven_stop": "Break-even stop",
    "target": "Target",
    "scheduled_close": "3:55 PM close",
    "account_failure": "Account loss limit",
}


def close_reason(trade: dict[str, Any]) -> str:
    kind = str(trade.get("exit_kind") or "")
    return _REASONS.get(kind, kind.replace("_", " ").capitalize() or "Not recorded")


def fills_text(trade: dict[str, Any]) -> str:
    """``10 at 25,874.75`` or, for a half exit, ``5 at 25,743.50 · 5 at 25,752.75``."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.format import points

    exit_ticks = trade.get("exit_ticks")
    if trade.get("scale_out_ticks") is not None:
        first = f"{int(trade.get('scale_out_quantity') or 0)} at " \
                f"{points(trade.get('scale_out_ticks'), from_ticks=True)}"
        rest = f"{int(trade.get('final_exit_quantity') or 0)} at " \
               f"{points(exit_ticks, from_ticks=True)}"
        return f"{first} · {rest}"
    return f"{int(trade.get('quantity') or 0)} at {points(exit_ticks, from_ticks=True)}"
