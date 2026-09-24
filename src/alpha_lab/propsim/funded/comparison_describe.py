"""Plain-English strategy settings for one resolved configuration.

Shown in the comparison screen and the review folder: only the settings needed
to understand a result, with full chart-timeframe names and Chicago times in
the 12-hour clock. No hashes or internal field names.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

__all__ = ["daily_close_text", "describe_section", "short_label"]

_TF = {"1m": "one-minute", "3m": "three-minute", "5m": "five-minute", "10m": "ten-minute",
       "15m": "fifteen-minute", "30m": "thirty-minute", "1H": "one-hour", "4H": "four-hour"}
_SESSION = {"asia": "Asia", "london": "London", "ny": "New York"}


def _clock(text: str, *, minus_hours: int = 0) -> str:
    moment = datetime.strptime(text, "%H:%M") - timedelta(hours=minus_hours)
    return moment.strftime("%I:%M %p").lstrip("0")


def _charts(values) -> str:
    names = [_TF.get(v, v) for v in values]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def entry_hours(section: Any) -> str:
    policy = section.entry_schedule_policy
    if policy == "all_open_market_v1":
        return "All open-market hours"
    if policy == "explicit_windows_v1":
        return "; ".join(f"{_clock(a)} to {_clock(b)} Chicago"
                         for a, b in section.entry_schedule_windows)
    # the original windows are defined in New York time (one hour ahead of Chicago)
    parts = []
    for key in section.enabled_entry_sessions:
        start, end = section.doc_sessions[key]
        parts.append(f"{_SESSION.get(key, key)} {_clock(start, minus_hours=1)} to "
                     f"{_clock(end, minus_hours=1)}")
    return "Original three windows: " + ", ".join(parts) + " Chicago"


def daily_close_text(clock: str) -> str:
    """The scheduled daily close rule in plain words (it is not a no-overnight rule)."""

    return (f"All positions closed by {clock} Chicago on every trading day (earlier on "
            "shortened days). A trading day runs from the 5:00 PM Chicago reopen, so a "
            "position may stay open past midnight inside one trading day; none is held "
            "through the daily close, a closed market or a weekend")


def describe_section(section: Any) -> list[dict[str, str]]:
    ticks = section.opposing_parent_distance_ticks_max
    minimum = section.opposing_min_gap_ticks
    direction = ("Long and short" if section.enable_longs and section.enable_shorts
                 else "Long only" if section.enable_longs else "Short only")
    close = (daily_close_text(_clock(section.daily_close_time))
             if section.holding_policy == "scheduled_daily_close_v1"
             else "Positions may be held across sessions")
    cap = section.max_executed_trades_per_day
    return [
        {"setting": "Entry hours", "value": entry_hours(section)},
        {"setting": "Direction", "value": direction},
        {"setting": "Higher-timeframe gap charts", "value": _charts(section.htf_timeframes)},
        {"setting": "Supporting (parent) charts", "value": _charts(section.parent_timeframes)},
        {"setting": "Largest distance from the parent gap to the opposing gap",
         "value": f"{ticks} ticks ({ticks / 4:g} points)"},
        {"setting": "Smallest opposing gap",
         "value": "no minimum" if minimum is None else
         f"{minimum} tick{'s' if minimum != 1 else ''} ({minimum / 4:g} points)"},
        {"setting": "Profit target",
         "value": "equal to the initial risk (1 to 1)" if section.tp_r_multiple == 1
         else f"{section.tp_r_multiple:g} times the initial risk"},
        {"setting": "Stop", "value": f"Beyond the setup's swing extreme plus "
                                     f"{section.sl_buffer_ticks} tick"},
        {"setting": "Trades per day",
         "value": "one position at a time, no daily limit" if cap is None
         else f"one position at a time, at most {cap} per day"},
        {"setting": "Daily close", "value": close},
    ]


def short_label(display_name: str) -> str:
    """A compact but readable configuration label from the study's own name."""

    return display_name.replace(" | ", "; ")
