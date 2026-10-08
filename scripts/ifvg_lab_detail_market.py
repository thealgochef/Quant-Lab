"""Configuration detail · Market conditions (mock 07).

Each study trading day is labeled from the stored one-minute E-mini bars
(``lab/market.py``): rising or falling, volatile or quiet. Two separately
versioned label sets (analytical correction A1, September 25, 2026), chosen by
the "Labels" switch in the header:

- Retrospective (default, ``retrospective_daily_close_v1``): the day's own
  close against the close 10 trading days earlier, volatility against the
  whole study's median. A description with hindsight, not known at entry.
- Known at entry (``entry_known_prior_closes_v1``): only closes completed
  before the day's 5:00 PM Chicago open and thresholds from earlier measures.
  Trades are then labeled by their entry's trading day only.

The tab shows, for ONE configuration at ONE firm:

- the definition sentence and the four condition cards (plus the
  not-enough-history line), labeling trades by entry day or by exit day
  (the switch in the header, remembered in the shared per-result context;
  retrospective labels only);
- the day-by-day trading profit with the background shaded by each day's label,
  and a caption naming the longest falling stretch and its result;
- how long each condition lasts (transition counts and row percentages);
- one entry-condition measure against the trade result in R, with its switch
  and a link-strength badge.

Everything is read only: saved funded trades and stored bars, nothing after
the study's cutoff. The pure helpers below carry no Streamlit state so they
can be tested on their own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab import market
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import (
    COLORS,
    FONT_SANS,
    css_var,
    palette,
    style_chart,
)

__all__ = [
    "BY_LABELS",
    "CONDITION_COLORS",
    "ENTRY_DAY_NOTE",
    "LABEL_MODES",
    "LINK_CLEAR",
    "LINK_NONE_BELOW",
    "LINK_SIGNIFICANCE",
    "MEASURE_WORDS",
    "NAMED_STRETCH_DAYS",
    "MarketView",
    "TransitionCell",
    "build_view",
    "condition_colors",
    "daily_by",
    "day_span",
    "definition_sentence",
    "entry_caption",
    "entry_figure",
    "header_right",
    "history_line",
    "link_strength",
    "moved_trades",
    "profit_figure",
    "render",
    "same_day_sentence",
    "stretch_caption",
    "transition_cells",
]

#: the header switch: label trades by the entry's or the exit's trading day
BY_LABELS = {"entry": "By entry day", "exit": "By exit day"}
_BY_KEY = "market_by"
_MEASURE_KEY = "market_measure"

#: the second header switch (correction A1): which label set the whole tab follows
LABEL_MODES = {market.RETROSPECTIVE: "Retrospective", market.ENTRY_KNOWN: "Known at entry"}
_MODE_KEY = "market_labels"
#: shown in place of the "Label trades by" switch when labels are known at entry:
#: labeling by the exit's day would use information from after the entry
ENTRY_DAY_NOTE = "Known at entry labels each trade by its entry's trading day."
_MODE_HELP = (
    "Retrospective labels describe each trading day with hindsight, from its own "
    "close and the whole study's typical volatility; they were not known when a "
    "trade was entered. Known at entry labels use only closes completed before the "
    "trading day's 5:00 PM Chicago open."
)

#: the definition line of each label set (correction A1), naming its version
_RETROSPECTIVE_DEFINITION = (
    f"Retrospective labels ({market.RETROSPECTIVE_VERSION}): each trading day is labeled "
    "from its own last close (4:00 PM, earlier on a shortened day) against the close "
    f"{market.LOOKBACK} trading days earlier, and its volatility against the median of the "
    "whole study. They describe the day with hindsight; they were not known when a trade "
    "was entered."
)
_ENTRY_KNOWN_DEFINITION = (
    f"Labels known at entry ({market.ENTRY_KNOWN_VERSION}): each trading day is labeled "
    "only from closes completed before its 5:00 PM open — the last completed close against "
    f"the one {market.LOOKBACK} trading days earlier, and the last {market.LOOKBACK} daily "
    "changes against the median of that day's and earlier days' measures. An evening entry "
    f"belongs to the next day's trading day. Days with fewer than {market.LOOKBACK + 1} "
    f"earlier closes or {market.MIN_HISTORY} volatility measures up to that day show "
    f'"{market.NOT_ENOUGH}", which is not a zero.'
)

#: each condition's palette color NAME (blue-light shades rising, orange-light falling);
#: the light values are the mock source's, the dark theme gives the same names dark values
_CONDITION_KEYS = {
    "Rising · quiet": "blue_band",
    "Rising · volatile": "blue_shade",
    "Falling · quiet": "orange_shade",
    "Falling · volatile": "orange_mid",
    market.NOT_ENOUGH: "header_row",  # the mock's grey start band
}

#: condition colors from the mock source: the LIGHT palette's values (the design reference);
#: screen code reads :func:`condition_colors` (charts) or :func:`_condition_var` (HTML)
CONDITION_COLORS = {label: COLORS[key] for label, key in _CONDITION_KEYS.items()}


def condition_colors(theme: str | None = None) -> dict[str, str]:
    """Condition → color value of the active theme (or of ``theme``), for chart shapes.

    Called when a figure is built, never at import time, so a chart drawn on the
    dark theme takes the dark values. On the light theme it equals
    ``CONDITION_COLORS``.
    """

    c = palette(theme)
    return {label: c[key] for label, key in _CONDITION_KEYS.items()}


def _condition_var(label: str) -> str:
    """``var(--lab-…)`` of a condition's color, for inline HTML (cards, legend, table)."""

    return css_var(_CONDITION_KEYS.get(label, "header_row"))


#: link-strength badge thresholds (decision log, rule 8): "No clear link" when the
#: correlation's size is under 0.1 or it is not significant at the 5% level;
#: otherwise "Weak" under 0.3 and "Clear" from 0.3, with its direction.
LINK_NONE_BELOW = 0.1
LINK_CLEAR = 0.3
LINK_SIGNIFICANCE = 0.05

#: a falling stretch this long or longer is named in the profit chart's caption
NAMED_STRETCH_DAYS = 5

#: words for each entry measure: (left end, right end, subject of the caption, chart title);
#: the time measure has clock ticks, so its ends are not named in words
MEASURE_WORDS = {
    "volatility": ("Calm", "Busy", "Busier markets at entry", "Volatility at entry vs result"),
    "trend": (
        "Falling",
        "Rising",
        "A stronger rise in the hour before entry",
        "Trend at entry vs result",
    ),
    "volume": ("Light", "Heavy", "Heavier trading before entry", "Volume at entry vs result"),
    "time": ("", "", "Later entries in the trading day", "Time of day at entry vs result"),
}
#: x-axis title when the measure's own axis words don't fit the ticks shown
_AXIS_TITLES = {"time": "Entry time, Chicago (the trading day opens at 5:00 PM)"}

_NOT_AVAILABLE = (
    "The stored one-minute E-mini bars for this study aren't available, so its "
    "days can't be labeled by market condition."
)
_NO_CALENDAR = (
    "This study's trading calendar isn't available, so its days can't be labeled "
    "by market condition."
)


# ── pure helpers ──────────────────────────────────────────────────────────


def _day(value: str) -> date:
    return date.fromisoformat(str(value)[:10])


def day_span(first: str, last: str, *, joiner: str = "–") -> str:
    """``January 13–15``, ``January 30 – February 2`` or ``January 13`` (no year).

    ``joiner="to"`` gives the caption form ``March 18 to April 7``.
    """

    start, end = _day(first), _day(last)
    if start == end:
        return f"{start:%B} {start.day}"
    if joiner == "to":
        return f"{start:%B} {start.day} to {end:%B} {end.day}"
    if (start.year, start.month) == (end.year, end.month):
        return f"{start:%B} {start.day}–{end.day}"
    return f"{start:%B} {start.day} – {end:%B} {end.day}"


def definition_sentence(labels: market.ConditionLabels) -> str:
    """The sentence under the tabs for the labels' mode, with the first labeled day.

    Correction A1: the retrospective labels say they were not known at entry; the
    entry-known labels say what they are built from. Each names its version.
    """

    entry_known = labels.mode == market.ENTRY_KNOWN
    head = _ENTRY_KNOWN_DEFINITION if entry_known else _RETROSPECTIVE_DEFINITION
    if labels.first_labeled is None:
        if entry_known:
            return (f"{head} No day in this study has enough closes completed before its "
                    "open in the stored bars, so none can be labeled.")
        return (f"{head} No day in this study has {market.LOOKBACK} earlier trading days in "
                "the stored bars, so none can be labeled.")
    first = fmt.date_long(labels.first_labeled)
    if next(iter(labels.labels.values()), None) == market.NOT_ENOUGH:
        return f"{head} Labels start {first}; the first days lack enough history."
    return f"{head} Labels start {first}."


def daily_by(trades: Sequence[Mapping[str, Any]], calendar: Sequence[str], *,
             by: str = "entry") -> tuple[list[tuple[str, float]], int]:
    """Net result per study trading day with each trade on its entry or exit trading day.

    Returns the daily series (0 on days without a trade) and how many trades
    fell outside the calendar (never silently added to another day).
    """

    totals = dict.fromkeys(calendar, 0.0)
    outside = 0
    for trade in trades:
        day = market.trade_day(dict(trade), by)
        if day in totals:
            totals[day] += float(trade.get("net_pnl_usd") or 0.0)
        else:
            outside += 1
    return [(day, round(value, 2)) for day, value in totals.items()], outside


def moved_trades(trades: Sequence[Mapping[str, Any]]) -> int:
    """How many trades closed on a later trading day than the one they opened in."""

    return sum(1 for t in trades
               if market.trade_day(dict(t), "exit") != market.trade_day(dict(t), "entry"))


def same_day_sentence(moved: int, total: int) -> str:
    """One sentence on what the entry/exit switch changes for these trades (rule 4)."""

    if total == 0:
        return "This configuration has no funded trades at this firm."
    if moved == 0:
        return ("Every trade here closed inside the trading day it opened, so labeling by "
                "entry day or by exit day gives the same counts.")
    return (f"{fmt.count(moved, 'trade')} of {total} closed on a later trading day than "
            "they opened; the exit-day view counts them under their exit day's label.")


def history_line(cards: Sequence[market.ConditionCard], labels: market.ConditionLabels,
                 total_trades: int) -> str:
    """``Not enough history (January 13–15): 5 trades, −$664. Shares are of 114 …``."""

    shares = f"Shares are of {fmt.count(total_trades, 'funded trade')}."
    days = [d for d, lab in labels.labels.items() if lab == market.NOT_ENOUGH]
    card = next((c for c in cards if c.label == market.NOT_ENOUGH), None)
    if not days or card is None:
        return shares
    span = day_span(days[0], days[-1])
    if card.trades == 0:
        return f"Not enough history ({span}): no trades. {shares}"
    return (f"Not enough history ({span}): {fmt.count(card.trades, 'trade')}, "
            f"{fmt.money_whole(card.net, signed=True)}. {shares}")


def _stretch_result(net: float) -> str:
    if round(net) < 0:
        return f"lost {fmt.money_whole(abs(net))}"
    return f"made {fmt.money_whole(net, signed=True)}"


def stretch_caption(items: Sequence[market.Stretch]) -> str:
    """Caption under the shaded profit chart (decision log, rule 1 and rule 3).

    Names the longest falling stretch (runs of one label; ties go to the later
    run, as ``market.longest_stretch`` does) with its result, then every other
    falling stretch of five days or more, then the longest stretch of any label.
    """

    labeled = [s for s in items if s.label != market.NOT_ENOUGH]
    if not labeled:
        return "No day in this period could be labeled, so there are no stretches to compare."
    falling = [s for s in labeled if s.label.startswith("Falling")]
    parts: list[str] = []
    longest_falling = max(falling, key=lambda s: (s.days, s.first_day)) if falling else None
    if longest_falling is None:
        parts.append("No day in this period was labeled falling.")
    else:
        s = longest_falling
        parts.append(f"The longest falling stretch — "
                     f"{day_span(s.first_day, s.last_day, joiner='to')}, "
                     f"{fmt.count(s.days, 'trading day')}, all {_volatility(s.label)} — "
                     f"{_stretch_result(s.net)}.")
        others = sorted((o for o in falling if o is not s and o.days >= NAMED_STRETCH_DAYS),
                        key=lambda o: o.first_day)
        if others:
            named = [f"{day_span(o.first_day, o.last_day)} ({_volatility(o.label)}) "
                     f"{_stretch_result(o.net)}" for o in others]
            listed = named[0] if len(named) == 1 else ", ".join(named[:-1]) + " and " + named[-1]
            parts.append(f"The other falling {'stretch' if len(named) == 1 else 'stretches'} "
                         f"of {_WORDS.get(NAMED_STRETCH_DAYS, NAMED_STRETCH_DAYS)} days or "
                         f"more: {listed}.")
        else:
            parts.append(f"No other falling stretch lasted "
                         f"{_WORDS.get(NAMED_STRETCH_DAYS, NAMED_STRETCH_DAYS)} days or more.")
    overall = market.longest_stretch(labeled)
    if overall is not None and overall is not longest_falling:
        parts.append(f"The longest stretch of any kind — "
                     f"{day_span(overall.first_day, overall.last_day, joiner='to')}, "
                     f"{fmt.count(overall.days, 'trading day')}, "
                     f"{overall.label.lower().replace(' · ', ' and ')} — "
                     f"{_stretch_result(overall.net)}.")
    return " ".join(parts)


_WORDS = {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 10: "ten"}


def _volatility(label: str) -> str:
    """``Falling · volatile`` → ``volatile``."""

    return label.split(" · ", 1)[-1]


@dataclass(frozen=True)
class TransitionCell:
    count: int
    share: float | None  # of the row's total; None when the row has no days
    diagonal: bool


def transition_cells(table: Mapping[str, Mapping[str, int]]) -> list[list[TransitionCell]]:
    """Rows in ``CONDITIONS`` order: count and row percentage of each next-day label."""

    rows = []
    for i, today in enumerate(market.CONDITIONS):
        counts = [int((table.get(today) or {}).get(tomorrow, 0))
                  for tomorrow in market.CONDITIONS]
        total = sum(counts)
        rows.append([TransitionCell(n, (n / total) if total else None, i == j)
                     for j, n in enumerate(counts)])
    return rows


def link_strength(correlation: float | None, p_value: float | None) -> tuple[str, str]:
    """Badge text and tone for the entry-condition link (thresholds in the decision log)."""

    if correlation is None:
        return "Not enough trades", "neutral"
    size = abs(correlation)
    if size < LINK_NONE_BELOW or p_value is None or p_value >= LINK_SIGNIFICANCE:
        return "No clear link", "neutral"
    direction = "positive" if correlation > 0 else "negative"
    strength = "Clear" if size >= LINK_CLEAR else "Weak"
    return f"{strength} {direction} link", "blue"


def entry_caption(measure: market.EntryMeasure, total_trades: int) -> str:
    """``Correlation 0.04 across 114 trades. Busier markets at entry didn't …``."""

    subject = MEASURE_WORDS.get(measure.measure, MEASURE_WORDS["volatility"])[2]
    measured = len(measure.x)
    if measure.correlation is None:
        return (f"Only {fmt.count(measured, 'trade')} could be measured, too few for a "
                "correlation.")
    across = (f"across {fmt.count(measured, 'trade')}" if measured == total_trades
              else f"across {measured} of {fmt.count(total_trades, 'trade')} (the others had "
                   "too few earlier bars)")
    head = f"Correlation {fmt.number(measure.correlation)} {across}"
    text, _ = link_strength(measure.correlation, measure.p_value)
    if text == "No clear link":
        if abs(measure.correlation) >= LINK_NONE_BELOW:
            head += f", not significant at the {LINK_SIGNIFICANCE:.0%} level"
        bigger = "bigger" if measure.correlation >= 0 else "smaller"
        return f"{head}. {subject} didn't produce {bigger} results here."
    bigger = "bigger" if measure.correlation > 0 else "smaller"
    return f"{head}. {subject} went with {bigger} results here ({text.lower()})."


@dataclass(frozen=True)
class MarketView:
    """Everything the tab shows for one configuration at one firm, labeled one way."""

    by: str
    labels: market.ConditionLabels
    cards: tuple[market.ConditionCard, ...]
    daily: tuple[tuple[str, float], ...]
    stretches: tuple[market.Stretch, ...]
    transitions: dict[str, dict[str, int]]
    total_trades: int
    moved: int
    outside: int = 0
    notes: tuple[str, ...] = field(default=())


def build_view(labels: market.ConditionLabels, trades: Sequence[Mapping[str, Any]],
               calendar: Sequence[str], *, by: str = "entry") -> MarketView:
    """The cards, daily series, stretches and transitions for one labeling choice.

    Entry-known labels always label a trade by its entry's trading day (correction A1):
    the exit's day would use information from after the entry.
    """

    by = by if by in BY_LABELS else "entry"
    if getattr(labels, "mode", market.RETROSPECTIVE) == market.ENTRY_KNOWN:
        by = "entry"
    rows = [dict(t) for t in trades]
    daily, outside = daily_by(rows, calendar, by=by)
    return MarketView(
        by=by, labels=labels,
        cards=tuple(market.condition_cards(labels, rows, by=by)),
        daily=tuple(daily), stretches=tuple(market.stretches(labels, daily)),
        transitions=market.transition_table(labels), total_trades=len(rows),
        moved=moved_trades(rows), outside=outside)


def _month_ticks(days: Sequence[str]) -> tuple[list[int], list[str]]:
    values, text, seen = [], [], set()
    for index, day in enumerate(days):
        month = str(day)[:7]
        if month not in seen:
            seen.add(month)
            values.append(index)
            text.append(f"{_day(day):%B}")
    return values, text


def profit_figure(daily: Sequence[tuple[str, float]], labels: market.ConditionLabels):
    """Running total of trading profit by trading day, background shaded by label."""

    import plotly.graph_objects as go

    days = [d for d, _ in daily]
    running, total = [], 0.0
    for _, value in daily:
        total += value
        running.append(round(total, 2))
    c = palette()  # the active theme's values, read when the figure is built
    shades = condition_colors()
    fig = go.Figure()
    start = 0
    for index in range(1, len(days) + 1):
        if index == len(days) or labels.labels.get(days[index]) != labels.labels.get(days[start]):
            label = labels.labels.get(days[start], market.NOT_ENOUGH)
            fig.add_shape(type="rect", xref="x", yref="paper", x0=start - 0.5, x1=index - 0.5,
                          y0=0, y1=1, fillcolor=shades.get(label, c["header_row"]),
                          line={"width": 0}, layer="below")
            start = index
    custom = [[fmt.date_long(d), labels.labels.get(d, market.NOT_ENOUGH), fmt.money(v, signed=True),
               fmt.money(r, signed=True)] for (d, v), r in zip(daily, running, strict=True)]
    fig.add_trace(go.Scatter(
        x=list(range(len(days))), y=running, mode="lines", customdata=custom,
        line={"color": c["ink"], "width": 2.5}, name="Trading profit",
        hovertemplate=("%{customdata[0]} · %{customdata[1]}<br>Day %{customdata[2]} · "
                       "running total %{customdata[3]}<extra></extra>")))
    fig.add_hline(y=0, line={"color": c["zero_line"], "dash": "dot", "width": 1})
    style_chart(fig, height=330, x_title="Trading day (Chicago dates)",
                y_title="Trading profit, running total")
    ticks, text = _month_ticks(days)
    fig.update_xaxes(range=[-0.5, len(days) - 0.5], tickvals=ticks, ticktext=text,
                     showgrid=False,
                     tickfont={"family": FONT_SANS, "size": 12, "color": c["muted"]})
    return fig


def _clock_from_open(hours: float) -> str:
    minutes = int(round(hours * 60)) + 17 * 60
    hour, minute = (minutes // 60) % 24, minutes % 60
    return f"{hour % 12 or 12}:{minute:02d} {'AM' if hour < 12 else 'PM'}"


def _x_text(measure: str, value: float) -> str:
    if measure == "volatility":
        return f"{value * 100:.3f}% of price per minute"
    if measure == "trend":
        return f"{value * 100:+.2f}% over the hour"
    if measure == "volume":
        return f"{value:,.0f} contracts"
    return f"entered {_clock_from_open(value)}"


def entry_figure(measure: market.EntryMeasure):
    """Scatter of one entry measure against the trade's result in R (winners blue)."""

    import plotly.graph_objects as go

    c = palette()  # the active theme's values, read when the figure is built
    fig = go.Figure()
    groups = ((True, c["blue"], "Winning trade"),
              (False, c["orange"], "Losing or break-even trade"))
    for winner, color, name in groups:
        xs = [x for x, w in zip(measure.x, measure.winner, strict=True) if w is winner]
        rs = [r for r, w in zip(measure.r_multiple, measure.winner, strict=True) if w is winner]
        custom = [[_x_text(measure.measure, x), f"{r:+.2f}R"]
                  for x, r in zip(xs, rs, strict=True)]
        fig.add_trace(go.Scatter(
            x=xs, y=rs, mode="markers", name=name, customdata=custom,
            marker={"color": color, "size": 8, "opacity": 0.78, "line": {"width": 0}},
            hovertemplate="%{customdata[0]}<br>Result %{customdata[1]}<extra></extra>"))
    fig.add_hline(y=0, line={"color": c["rule"], "width": 1})
    clock = measure.measure == "time"
    style_chart(fig, height=320 if clock else 300, money_axis=None, y_title="Result in R")
    fig.update_yaxes(ticksuffix="R", tickformat="~g")
    if clock:  # a time axis: Chicago 12-hour clock ticks
        ticks = list(range(0, 24, 4))
        fig.update_xaxes(tickvals=ticks, ticktext=[_clock_from_open(t) for t in ticks],
                         range=[-0.5, 23.5])
    else:  # the mock labels the ends in words instead of numbers
        fig.update_xaxes(showticklabels=False)
    if measure.measure == "trend" and measure.x and min(measure.x) < 0 < max(measure.x):
        # where the hour before entry turns from falling to rising
        fig.add_vline(x=0, line={"color": c["zero_line"], "dash": "dot", "width": 1})
    # the x axis is titled in words, with its two ends named (mock: Calm … Busy)
    left, right, _, _ = MEASURE_WORDS.get(measure.measure, MEASURE_WORDS["volatility"])
    title = _AXIS_TITLES.get(measure.measure) or market.ENTRY_MEASURES.get(
        measure.measure, market.ENTRY_MEASURES["volatility"])[1]
    shift = -40 if clock else -16
    for x, anchor, text in ((0, "left", left), (0.5, "center", title), (1, "right", right)):
        if not text:
            continue
        fig.add_annotation(x=x, xref="paper", y=0, yref="paper", yshift=shift, xanchor=anchor,
                           yanchor="top", showarrow=False, text=text,
                           font={"size": 12, "color": c["muted"]})
    fig.update_layout(margin={"l": 8, "r": 16, "t": 12, "b": 60 if clock else 36})
    return fig


# ── HTML pieces ───────────────────────────────────────────────────────────


def _condition_card(card: market.ConditionCard) -> h.Markup:
    color = _condition_var(card.label)
    if card.trades:
        net_html = h.esc(fmt.money_whole(card.net, signed=True))
        trades = f"{card.trades} · {fmt.percent(card.share_of_trades)}"
        win = fmt.percent(card.win_rate)
        average = fmt.money_whole(card.average)
    else:  # no trade on these days: say so, never a $0 result (rule 6)
        net_html = h.placeholder("No trades")
        trades, win, average = "0", fmt.MISSING, fmt.MISSING
    row = ('<span>{}</span><span style="font-family:var(--lab-mono);text-align:right">{}</span>')
    stats = "".join(row.format(h.esc(k), h.esc(v)) for k, v in
                    (("Trades", trades), ("Win rate", win), ("Average trade", average)))
    return h.Markup(
        f'<div class="lab" style="background:{css_var("panel")};'
        f'border:1px solid {css_var("rule")};'
        f"border-top:6px solid {color};border-radius:12px;padding:16px 18px;display:flex;"
        'flex-direction:column;gap:6px;min-width:0">'
        '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px">'
        f'<div style="font-size:15px;font-weight:600">{h.esc(card.label)}</div>'
        f'<div style="font-size:13px;color:{css_var("muted")};white-space:nowrap">'
        f'{h.esc(fmt.count(card.days, "day"))}</div></div>'
        '<div style="font-family:var(--lab-mono);font-size:24px;font-weight:500">'
        f"{net_html}</div>"
        '<div style="display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:13px;'
        f'color:{css_var("body")}">{stats}</div></div>')


def _cards(view: MarketView) -> h.Markup:
    cards = [c for c in view.cards if c.label in market.CONDITIONS]
    return h.grid([_condition_card(c) for c in cards], 4)


def _transition_card(view: MarketView) -> h.Markup:
    rows = transition_cells(view.transitions)
    head = [f'<div style="color:{css_var("muted")};padding:6px 4px">From / to</div>']
    for label in market.CONDITIONS:
        head.append(f'<div style="padding:6px;text-align:center;font-weight:600;'
                    f'background:{_condition_var(label)};border-radius:4px">'
                    f"{h.esc(label.replace(' · ', ' '))}</div>")
    body = []
    for label, cells in zip(market.CONDITIONS, rows, strict=True):
        body.append('<div style="padding:10px 4px;font-weight:600">'
                    f"{h.esc(label.replace(' · ', ' '))}</div>")
        for cell in cells:
            share = fmt.percent(cell.share) if cell.share is not None else fmt.MISSING
            if cell.diagonal:
                body.append(
                    '<div style="padding:10px 6px;text-align:center;'
                    f'background:{css_var("ink")};color:{css_var("on_ink")};border-radius:4px">'
                    '<div style="font-family:var(--lab-mono);'
                    f'font-size:15px">{h.esc(share)}</div><div style="font-size:11px">'
                    f"{h.esc(fmt.count(cell.count, 'day'))}</div></div>")
            else:
                body.append(
                    '<div style="padding:10px 6px;text-align:center;'
                    f'background:{css_var("soft_panel")};border-radius:4px">'
                    '<div style="font-family:var(--lab-mono);font-size:15px">'
                    f'{h.esc(share)}</div><div style="font-size:11px;color:{css_var("muted")}">'
                    f"{cell.count}</div></div>")
    grid = ('<div style="display:grid;grid-template-columns:120px repeat(4,minmax(0,1fr));'
            f'gap:4px;font-size:13px">{"".join(head)}{"".join(body)}</div>')
    totals = [sum(c.count for c in cells) for cells in rows]
    few = any(total < 30 for total in totals)
    parts = [f'<div style="font-size:14px;color:{css_var("body_2")}">Chance tomorrow\'s label '
             'follows today\'s. The diagonal is "stays the same."</div>', grid]
    if few:
        parts.append(f'<div style="font-size:13px;line-height:1.5;color:{css_var("muted")}">'
                     "Few days per row — treat the off-diagonal numbers as rough. The "
                     "multi-year run fills this in properly.</div>")
    return h.card(h.Markup("".join(parts)), title="How long each condition lasts")


def _legend(view: MarketView) -> h.Markup:
    items = [(_condition_var(label), label) for label in market.CONDITIONS]
    if any(lab == market.NOT_ENOUGH for lab in view.labels.labels.values()):
        items.append((_condition_var(market.NOT_ENOUGH), market.NOT_ENOUGH))
    return h.legend(items)


_CARD_CSS = """
<style>
/* the entry-measure switch as separate pills (mock 07), not joined buttons */
.st-key-ifvg_lab_card_market_entry [role="radiogroup"] { flex-wrap: wrap; gap: 8px; }
.st-key-ifvg_lab_card_market_entry [role="radiogroup"] > label,
.st-key-ifvg_lab_card_market_entry [role="radiogroup"] > label:first-child,
.st-key-ifvg_lab_card_market_entry [role="radiogroup"] > label:last-child {
  border-radius: 999px; margin: 0; min-height: 36px; padding: 6px 12px;
}
.st-key-ifvg_lab_card_market_entry [role="radiogroup"] > label p { font-size: 13px; }
</style>
"""


# ── cached data (read only) ───────────────────────────────────────────────


class _BarsUnavailableError(Exception):
    """The verified package's one-minute bars can't be read now.

    Raised inside the cached functions and caught by their callers: Streamlit never
    caches an exception, so a package restored later is found on the next run.
    """


def _minutes(store_root: str, result_id: str):
    from ifvg_lab_cache import index_minutes

    minutes = index_minutes(store_root, result_id)
    if minutes is None:
        raise _BarsUnavailableError
    return minutes


def _source_signature(store_root: str, result_id: str) -> tuple:
    from pathlib import Path

    from ifvg_lab_ui import funded_study

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.external_catalog import (
        resolve_registered_result,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.registered_market import (
        cache_signature,
    )

    # Historical package views retain their established process-cache behavior.
    if resolve_registered_result(result_id, external_store_root=Path(store_root)) is None:
        return ()
    plan = getattr(funded_study(store_root, result_id), "plan", None)
    if getattr(getattr(plan, "source", None), "kind", None) == "verified_task_b_registered_inputs":
        return cache_signature(plan)
    return ()


@st.cache_data(show_spinner="Labeling each trading day by market condition…", max_entries=8)
def _cached_labels(
    store_root: str, result_id: str, mode: str = market.RETROSPECTIVE,
    source_signature: tuple = (),
) -> market.ConditionLabels | None:
    """One label set per mode (retrospective or known at entry), cached separately."""

    from ifvg_lab_ui import funded_study

    minutes = _minutes(store_root, result_id)
    study = funded_study(store_root, result_id)
    if not study.calendar:  # a fact of the saved result: kept
        return None
    if mode == market.ENTRY_KNOWN:
        return market.entry_known_labels(
            market.daily_closes(minutes), market.daily_close_times(minutes), study.calendar
        )
    return market.condition_labels(market.daily_closes(minutes), study.calendar)


def _labels(
    store_root: str, result_id: str, mode: str = market.RETROSPECTIVE
) -> market.ConditionLabels | None:
    try:
        return _cached_labels(store_root, result_id, mode, _source_signature(store_root, result_id))
    except _BarsUnavailableError:
        return None


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_view(
    store_root: str,
    result_id: str,
    configuration: str,
    firm_key: str,
    by: str,
    mode: str = market.RETROSPECTIVE,
    source_signature: tuple = (),
) -> MarketView | None:
    from ifvg_lab_ui import funded_study

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    labels = _cached_labels(store_root, result_id, mode, source_signature)
    if labels is None:
        return None
    study = funded_study(store_root, result_id)
    return build_view(labels, ordered_trades(study, configuration, firm_key), study.calendar, by=by)


def _view(
    store_root: str,
    result_id: str,
    configuration: str,
    firm_key: str,
    by: str,
    mode: str = market.RETROSPECTIVE,
) -> MarketView | None:
    try:
        return _cached_view(store_root, result_id, configuration, firm_key, by, mode,
                            _source_signature(store_root, result_id))
    except _BarsUnavailableError:
        return None


@st.cache_data(show_spinner="Measuring the market at each entry…", max_entries=64)
def _cached_entry(
    store_root: str, result_id: str, configuration: str, firm_key: str, measure: str,
    source_signature: tuple = (),
) -> market.EntryMeasure:
    from ifvg_lab_ui import funded_study

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    minutes = _minutes(store_root, result_id)
    study = funded_study(store_root, result_id)
    return market.entry_measure(
        minutes, ordered_trades(study, configuration, firm_key), measure=measure
    )


def _entry(
    store_root: str, result_id: str, configuration: str, firm_key: str, measure: str
) -> market.EntryMeasure | None:
    try:
        return _cached_entry(store_root, result_id, configuration, firm_key, measure,
                             _source_signature(store_root, result_id))
    except _BarsUnavailableError:
        return None


# ── Streamlit ─────────────────────────────────────────────────────────────


def _mode(ctx) -> str:
    """The chosen label set (retrospective unless "Known at entry" was chosen)."""

    mode = ctx.context.get(_MODE_KEY, market.RETROSPECTIVE)
    return mode if mode in LABEL_MODES else market.RETROSPECTIVE


def header_right(st_module, ctx) -> None:
    """The "Labels" switch and the "By entry day / By exit day" switch (mock 07).

    Correction A1: with labels known at entry a trade is labeled by its entry's trading
    day only, so the second switch gives way to a one-line note; its remembered choice
    is kept for when the retrospective labels are shown again.
    """

    from ifvg_lab_ui import show, switch

    mode = switch("Labels", list(LABEL_MODES), key=_MODE_KEY, value=_mode(ctx),
                  format_func=LABEL_MODES.get, help=_MODE_HELP, st_module=st_module)
    ctx.context[_MODE_KEY] = mode
    if mode == market.ENTRY_KNOWN:
        show(h.Markup(f'<div class="lab" style="font-size:13px;line-height:1.4;'
                      f'color:{css_var("muted")}">{h.esc(ENTRY_DAY_NOTE)}</div>'), st_module)
        return
    by = switch("Label trades by", list(BY_LABELS), key=_BY_KEY,
                value=ctx.context.get(_BY_KEY, "entry"), format_func=BY_LABELS.get,
                help="Label each trade by the market condition of the trading day it was "
                     "entered, or of the trading day it closed. A trading day runs from the "
                     "5:00 PM open to the 4:00 PM close.", st_module=st_module)
    ctx.context[_BY_KEY] = by


def _entry_panel(st_module, ctx, total_trades: int) -> None:
    from ifvg_lab_ui import plot, show, switch

    options = list(market.ENTRY_MEASURES)
    current = ctx.context.get(_MEASURE_KEY, "volatility")
    with st_module.container(key="ifvg_lab_card_market_entry"):
        show(h.Markup(_CARD_CSS), st_module)
        head = st_module.empty()
        measure_key = switch(
            "Entry measure", options, key=_MEASURE_KEY, value=current,
            format_func=lambda k: market.ENTRY_MEASURES[k][0],
            help="Which market measure at the moment of entry to plot against the trade's "
                 "result: volatility, trend, volume or time of day.", st_module=st_module)
        ctx.context[_MEASURE_KEY] = measure_key
        measure = _entry(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key,
                         measure_key)
        title = MEASURE_WORDS[measure_key][3]
        definition = market.ENTRY_MEASURES[measure_key][2]
        if measure is None:
            with head:
                show(h.section_title(title, size="h3"), st_module)
            show(h.placeholder(_NOT_AVAILABLE), st_module)
            return
        badge_text, tone = link_strength(measure.correlation, measure.p_value)
        with head:
            show(h.Markup('<div class="lab" style="display:flex;justify-content:space-between;'
                          f'align-items:baseline;gap:12px"><div class="lab-h3">{h.esc(title)}'
                          f"</div>{h.badge(badge_text, tone)}</div>"), st_module)
        if not measure.x:
            show(h.placeholder("No trade had enough earlier bars to measure."), st_module)
            return
        plot(entry_figure(measure), key=f"market_entry_{measure_key}", st_module=st_module)
        key = h.legend([(css_var("blue"), "Winning trade"),
                        (css_var("orange"), "Losing or break-even trade")])
        show(h.Markup(
            '<div class="lab" style="display:flex;flex-direction:column;gap:8px">'
            f'<div style="font-size:14px;line-height:1.5;color:{css_var("body")}">'
            f"{h.esc(entry_caption(measure, total_trades))}</div>{key}"
            f'<div class="lab-line" style="font-size:13px">How this is measured: '
            f"{h.esc(definition)} Result = the trade's net result ÷ its initial risk. "
            f"\"No clear link\" means a correlation under {LINK_NONE_BELOW} in size or not "
            f"significant at the {LINK_SIGNIFICANCE:.0%} level; under {LINK_CLEAR} is weak, "
            f"{LINK_CLEAR} and above is clear.</div></div>"), st_module)


def _render_price(st_module, ctx) -> None:
    from ifvg_lab_ui import plot, show

    mode = _mode(ctx)
    # labels known at entry: always the entry's trading day (correction A1)
    by = ctx.context.get(_BY_KEY, "entry") if mode == market.RETROSPECTIVE else "entry"
    if not ctx.study.calendar:
        show(h.placeholder(_NO_CALENDAR), st_module)
        return
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.registered_market import (
        MarketSourceError,
    )

    try:
        view = _view(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key, by, mode=mode)
    except MarketSourceError as error:
        show(h.alert("This study's market inputs could not be verified.", str(error)), st_module)
        return
    if view is None:
        show(h.placeholder(_NOT_AVAILABLE), st_module)
        return
    show(
        h.Markup(
            '<div class="lab" style="font-size:14px;line-height:1.5;'
            f'color:{css_var("body_2")};margin-top:-8px">'
            f"{h.esc(definition_sentence(view.labels))}</div>"
        ),
        st_module,
    )
    show(_cards(view), st_module)
    lines = [history_line(view.cards, view.labels, view.total_trades)]
    if mode == market.RETROSPECTIVE:  # the entry/exit switch is shown only here
        lines.append(same_day_sentence(view.moved, view.total_trades))
    if view.outside:
        lines.append(
            f"{fmt.count(view.outside, 'trade')} fell outside the study's trading "
            "days on this labeling and "
            f"{'is' if view.outside == 1 else 'are'} left out of the daily profit "
            "below."
        )
    show(
        h.Markup(
            f'<div class="lab" style="font-size:13px;color:{css_var("muted")};'
            'margin-top:-10px;line-height:1.5">' + "<br>".join(h.esc(x) for x in lines) + "</div>"
        ),
        st_module,
    )
    with st_module.container(key="ifvg_lab_card_market_profit"):
        show(
            h.Markup(
                '<div class="lab" style="display:flex;justify-content:space-between;'
                'align-items:baseline;gap:16px;flex-wrap:wrap"><div class="lab-h3">'
                "Trading profit, shaded by market condition</div>"
                f"{_legend(view)}</div>"
            ),
            st_module,
        )
        plot(
            profit_figure(view.daily, view.labels),
            key=f"market_profit_{mode}_{view.by}",
            st_module=st_module,
        )
        show(
            h.Markup(
                '<div class="lab" style="font-size:14px;line-height:1.5;'
                f'color:{css_var("body")}">'
                f"{h.esc(stretch_caption(view.stretches))}</div>"
            ),
            st_module,
        )
    left, right = st_module.columns(2, gap="medium")
    with left:
        show(_transition_card(view), st_module)
    with right:
        _entry_panel(st_module, ctx, view.total_trades)


def render(st_module, ctx) -> None:
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import is_mffu_plan

    if is_mffu_plan(getattr(ctx.study, "plan", None)):
        selected = ctx.context.get("market_surface", "Price behavior")
        options = ["Price behavior", "Gamma & expected move"]
        selected = st_module.radio(
            "Market conditions view",
            options,
            index=options.index(selected) if selected in options else 0,
            horizontal=True,
            key=f"market_surface_{ctx.result_id[:16]}",
        )
        ctx.context["market_surface"] = selected
        if selected == "Gamma & expected move":
            from ifvg_lab_mffu_views import render_gamma

            render_gamma(st_module, ctx)
            return
    _render_price(st_module, ctx)
