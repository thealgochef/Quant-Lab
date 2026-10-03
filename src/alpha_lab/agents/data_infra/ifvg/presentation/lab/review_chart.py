"""Trade review charts (mocks 09, 09b): candles, zones and fills as pure Plotly figures.

Candles come from the study's stored one-minute E-mini bars, aggregated to the
chosen size by their Chicago clock (a 10-minute candle holds the minutes that
opened 7:00–7:09 PM and is named by its opening minute). Zones and setup times
come from the linked saved setup record; fills, stops and the result from the
saved funded trade row. Nothing is re-computed from the strategy.

Point in time: with a ``moment``, only one-minute bars that had CLOSED by then
are used (an aggregated candle can therefore be partly formed, as it looked at
that moment); markings, lines and the axis range use only what was known then,
and the rest of the trading day is covered and labeled "Hidden: everything
after …". The x range is the trading day's SCHEDULED 5:00 PM open → 4:00 PM
close (:func:`chart_window`), never the first and last stored bar, so a day
whose later bars are missing, changed or extended draws the same hidden chart
(correction A7). Full history keeps the bar-derived window.

A related setup record (another configuration's, identity not established;
correction A6) draws its zones labelled as related context.
"""

from __future__ import annotations

from datetime import time, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO, chicago_wall
from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab.review_panels import (
    TradeView,
    clock,
    day_close,
    exit_words,
    hidden_sentence,
    setup_events,
    short,
    when,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.setup_records import SetupRecord
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import (
    FONT_MONO,
    FONT_SANS,
    palette,
    rgba,
    style_chart,
)

__all__ = [
    "CANDLE_SIZES",
    "aggregate",
    "chart_window",
    "day_minutes",
    "scheduled_window",
    "setup_figure",
    "setup_window",
    "trading_window",
    "visible_minutes",
    "whole_trade_figure",
    "whole_trade_caption",
]

CANDLE_SIZES = (1, 5, 10, 15, 30)
# Fills are read from the active palette when a figure is built (``theme.palette`` and
# ``theme.rgba``): the hidden-time cover is header_row, the in-trade wash blue at 0.06, the
# gap zones blue_band at 0.75 (whole trade) and 0.85 (parent gap), midnight zero_line.


# ── bars ──────────────────────────────────────────────────────────────────


def day_minutes(minutes: pd.DataFrame, trading_day: str) -> pd.DataFrame:
    """The stored one-minute bars of one trading day (5:00 PM open → 4:00 PM close)."""

    if minutes is None or minutes.empty:
        return pd.DataFrame(columns=["logical_open_ts_utc", "logical_close_ts_utc", "open",
                                     "high", "low", "close"])
    day = minutes[minutes["trading_day"].astype(str) == str(trading_day)]
    return day.sort_values("logical_open_ts_utc").reset_index(drop=True)


def trading_window(day: pd.DataFrame, trading_day: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(open, close) instants of the trading day: from its bars, else 5:00 PM → 4:00 PM."""

    if day is not None and not day.empty:
        return (pd.Timestamp(day["logical_open_ts_utc"].min()),
                pd.Timestamp(day["logical_close_ts_utc"].max()))
    closing = pd.Timestamp(str(trading_day)[:10]).date()
    opening = closing - timedelta(days=1)
    start = pd.Timestamp.combine(opening, time(17, 0)).tz_localize(CHICAGO)
    end = pd.Timestamp.combine(closing, time(16, 0)).tz_localize(CHICAGO)
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def scheduled_window(view: TradeView) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The trade's trading day as scheduled: 5:00 PM open → 4:00 PM close (never the bars).

    The day is the saved trading day, else the one the entry belongs to
    (:func:`review_panels.day_close`).
    """

    closing = day_close(view).tz_convert(CHICAGO).date()
    return trading_window(None, closing.isoformat())


def chart_window(day: pd.DataFrame | None, view: TradeView, *,
                 moment: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The whole-trade chart's time range and header words.

    Point in time (a ``moment``) always uses the scheduled day, so the range can't
    depend on bars after the moment (correction A7); Full history uses the stored
    bars' first open and last close, else the scheduled day.
    """

    if moment is not None or day is None or day.empty:
        return scheduled_window(view)
    return trading_window(day, view.trading_day)


def visible_minutes(frame: pd.DataFrame, moment: pd.Timestamp | None) -> pd.DataFrame:
    """Bars that had closed by ``moment`` (all bars without a moment)."""

    if moment is None or frame.empty:
        return frame
    return frame[frame["logical_close_ts_utc"] <= moment]


def aggregate(frame: pd.DataFrame, size_minutes: int) -> pd.DataFrame:
    """Candles of ``size_minutes`` from one-minute bars, named by their opening minute."""

    columns = ["open_utc", "close_utc", "open", "high", "low", "close", "minutes"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    data = frame.sort_values("logical_open_ts_utc").reset_index(drop=True)
    if size_minutes <= 1:
        out = data[["logical_open_ts_utc", "logical_close_ts_utc", "open", "high", "low",
                    "close"]].rename(columns={"logical_open_ts_utc": "open_utc",
                                              "logical_close_ts_utc": "close_utc"})
        return out.assign(minutes=1)[columns]
    # whole-hour zone offsets: flooring the UTC instant equals flooring the Chicago clock
    # for every size that divides an hour (1, 5, 10, 15, 30 minutes)
    bucket = data["logical_open_ts_utc"].dt.floor(f"{int(size_minutes)}min")
    grouped = data.assign(_bucket=bucket).groupby("_bucket", sort=True)
    out = pd.DataFrame({
        "close_utc": grouped["logical_close_ts_utc"].max(),
        "open": grouped["open"].first(), "high": grouped["high"].max(),
        "low": grouped["low"].min(), "close": grouped["close"].last(),
        "minutes": grouped["open"].size(),
    })
    out.index.name = "open_utc"
    return out.reset_index()[columns]


# ── shared drawing helpers ────────────────────────────────────────────────


def _wall(at: Any) -> pd.Timestamp | None:
    return chicago_wall(at) if at is not None else None


def _pts(ticks: Any) -> float | None:
    return None if ticks is None else int(ticks) * 0.25


def _known(at: Any, moment: pd.Timestamp | None) -> bool:
    return at is not None and (moment is None or at <= moment)


def _candles(fig: go.Figure, candles: pd.DataFrame, size_minutes: int) -> None:
    if candles.empty:
        return
    x = [_wall(t) + timedelta(minutes=size_minutes / 2) for t in candles["open_utc"]]
    text = [f"{clock(t)} candle · open {o:,.2f} · high {hi:,.2f} · low {lo:,.2f} · "
            f"close {c:,.2f}" for t, o, hi, lo, c in zip(candles["open_utc"], candles["open"],
                                                         candles["high"], candles["low"],
                                                         candles["close"], strict=True)]
    c = palette()
    fig.add_trace(go.Candlestick(
        x=x, open=candles["open"], high=candles["high"], low=candles["low"],
        close=candles["close"], name="Candles", text=text, hoverinfo="text",
        increasing={"line": {"color": c["blue"], "width": 1}, "fillcolor": c["blue"]},
        decreasing={"line": {"color": c["orange"], "width": 1}, "fillcolor": c["orange"]},
        whiskerwidth=0, showlegend=False))


def _y_range(values: list[float], *, minimum_span: float = 8.0) -> tuple[float, float]:
    values = [v for v in values if v is not None]
    if not values:
        return 0.0, 1.0
    low, high = min(values), max(values)
    span = max(high - low, minimum_span)
    pad = span * 0.08
    middle = (low + high) / 2
    return middle - span / 2 - pad, middle + span / 2 + pad


def _hline(fig, x0, x1, y, *, color, dash="solid", width=1.5, name=None) -> None:
    if x0 is None or x1 is None or y is None or x1 <= x0:
        return
    fig.add_trace(go.Scatter(x=[x0, x1], y=[y, y], mode="lines", hoverinfo="text" if name else
                             "skip", text=[name, name] if name else None,
                             line={"color": color, "dash": dash, "width": width},
                             showlegend=False))


def _label(fig, x, y, text, *, color=None, bold=False, xanchor="left", yanchor="bottom",
           size=12, yshift=0, xshift=0, bg=None) -> None:
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>" if bold else text, showarrow=False,
                       xanchor=xanchor, yanchor=yanchor, yshift=yshift, xshift=xshift,
                       font={"family": FONT_SANS, "size": size,
                             "color": color or palette()["body"]},
                       bgcolor=bg, borderpad=3 if bg else 0)


def _entry_marker(fig, view: TradeView, *, size: int = 13) -> None:
    c = palette()
    fig.add_trace(go.Scatter(
        x=[_wall(view.entry_utc)], y=[_pts(view.entry_ticks)], mode="markers",
        marker={"symbol": "triangle-up" if view.long else "triangle-down", "size": size,
                "color": c["ink"], "line": {"color": c["panel"], "width": 1}},
        hoverinfo="text", text=[f"Entry {_price(view.entry_ticks)} · "
                                f"{short(view.entry_utc)}"], showlegend=False))


def _exit_dots(fig, view: TradeView, moment: pd.Timestamp | None, *, size: int = 11) -> None:
    xs, ys, text = [], [], []
    if view.half_utc is not None and _known(view.half_utc, moment):
        xs.append(_wall(view.half_utc))
        ys.append(_pts(view.half_ticks))
        text.append(f"Half out: {view.half_quantity} at {_price(view.half_ticks)} · "
                    f"{short(view.half_utc)}")
    if _known(view.exit_utc, moment) and view.exit_ticks is not None:
        xs.append(_wall(view.exit_utc))
        ys.append(_pts(view.exit_ticks))
        text.append(f"Out: {view.exit_quantity} at {_price(view.exit_ticks)} · "
                    f"{short(view.exit_utc)}")
    if xs:
        c = palette()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", hoverinfo="text", text=text,
                                 marker={"size": size, "color": c["blue"],
                                         "line": {"color": c["panel"], "width": 1.5}},
                                 showlegend=False))


def _price(ticks: Any) -> str:
    return fmt.points(ticks, from_ticks=True)


def _cover(fig, start_wall, end_wall, text: str | None) -> None:
    if start_wall is None or end_wall is None or end_wall <= start_wall:
        return
    # nothing after the moment is drawn at all; the cover only marks the hidden time, so it
    # sits under the markers that end exactly at the moment
    c = palette()
    fig.add_shape(type="rect", xref="x", yref="paper", x0=start_wall, x1=end_wall, y0=0, y1=1,
                  fillcolor=c["header_row"], line={"width": 0}, layer="between")
    if text:
        middle = start_wall + (end_wall - start_wall) / 2
        fig.add_annotation(x=middle, y=0.5, xref="x", yref="paper", text=text, showarrow=False,
                           font={"family": FONT_SANS, "size": 14, "color": c["body_2"]},
                           bgcolor=c["panel"], bordercolor=c["rule"], borderwidth=1,
                           borderpad=8)


def _axes(fig, *, x_range, y_range, height: int, dtick_hours: float | None = None) -> None:
    c = palette()
    style_chart(fig, height=height, x_title="Chicago time", y_title="Price (index points)",
                money_axis=None)
    fig.update_layout(margin={"l": 8, "r": 12, "t": 8, "b": 8}, hovermode="closest",
                      xaxis_rangeslider_visible=False, dragmode=False)
    xaxis: dict[str, Any] = {"range": list(x_range), "tickformat": "%-I:%M %p",
                             "hoverformat": "%b %-d, %-I:%M %p", "showgrid": False,
                             "tickfont": {"family": FONT_SANS, "size": 12,
                                          "color": c["muted"]}}
    if dtick_hours:
        xaxis["dtick"] = int(dtick_hours * 3600 * 1000)
    fig.update_xaxes(**xaxis)
    fig.update_yaxes(range=list(y_range), tickformat=",.2~f", side="left",
                     tickfont={"family": FONT_MONO, "size": 12, "color": c["muted"]})


def _daily_close(view: TradeView, window_end: pd.Timestamp,
                 close_clock: time | None) -> pd.Timestamp | None:
    if close_clock is None:
        return None
    end_local = window_end.tz_convert(CHICAGO)
    at = pd.Timestamp.combine(end_local.date(), close_clock).tz_localize(CHICAGO)
    return at.tz_convert("UTC")


def _midnight(window: tuple[pd.Timestamp, pd.Timestamp]) -> pd.Timestamp | None:
    start_local = window[0].tz_convert(CHICAGO)
    midnight = pd.Timestamp.combine(start_local.date() + timedelta(days=1), time(0, 0))
    at = midnight.tz_localize(CHICAGO).tz_convert("UTC")
    return at if window[0] < at < window[1] else None


# ── whole trade ───────────────────────────────────────────────────────────


def _result_box_text(view: TradeView) -> str:
    kind = view.exit_kind
    part = "last " if view.half_utc is not None else ""
    head = {"scheduled_close": f"{clock(view.exit_utc)} close",
            "target": f"Target at {clock(view.exit_utc)}",
            "breakeven_stop": f"Stop at entry, {clock(view.exit_utc)}",
            "stop": f"Stop at {clock(view.exit_utc)}",
            "account_failure": f"Account loss limit, {clock(view.exit_utc)}",
            }.get(kind, f"Out at {clock(view.exit_utc)}")
    first = f"{head} · {part}{view.exit_quantity} at {_price(view.exit_ticks)}"
    result = fmt.money(view.net, signed=True, missing="not in this study's export")
    return f"{first}<br><b>Trade result {result}</b>"


def _zone_label(gap, y_range: tuple[float, float], *, related: bool = False) -> str:
    low, high = _pts(gap.low_ticks), _pts(gap.high_ticks)
    name = "Four-hour gap" if gap.timeframe_seconds == 14400 else (
        "One-hour gap" if gap.timeframe_seconds == 3600 else "Higher-timeframe gap")
    if related:  # another configuration's record: context, not this setup (correction A6)
        name = f"Related {name[0].lower()}{name[1:]} (context)"
    if low < y_range[0] and high > y_range[1]:
        where = f"covers everything shown ({low:,.2f}–{high:,.2f})"
    elif low < y_range[0]:
        where = f"below {high:,.2f}"
    elif high > y_range[1]:
        where = f"above {low:,.2f}"
    else:
        where = f"{low:,.2f}–{high:,.2f}"
    return f"{name} {where} · confirmed {short(gap.confirmed_utc)}"


def whole_trade_figure(day: pd.DataFrame, view: TradeView, record: SetupRecord | None, *,
                       size_minutes: int = 10, moment: pd.Timestamp | None = None,
                       zones: bool = True, stops: bool = True, times: bool = True,
                       close_clock: time | None = time(15, 55), height: int = 430
                       ) -> go.Figure:
    """The trade's whole trading day: candles, the gap zone, fills, stops and the result."""

    window = chart_window(day, view, moment=moment)
    shown = visible_minutes(day, moment)
    candles = aggregate(shown, size_minutes)
    fig = go.Figure()
    c = palette()
    values = list(candles["low"]) + list(candles["high"])
    entry_known = _known(view.entry_utc, moment)
    if entry_known:
        values += [_pts(view.entry_ticks)]
        if stops:
            values += [_pts(view.stop_ticks)]
            if view.target_ticks is not None:
                values.append(_pts(view.target_ticks))
    for at, ticks in ((view.half_utc, view.half_ticks), (view.exit_utc, view.exit_ticks)):
        if _known(at, moment) and ticks is not None:
            values.append(_pts(ticks))
    y_range = _y_range(values)
    x_range = (_wall(window[0]), _wall(window[1]))
    last = min(window[1], moment) if moment is not None else window[1]
    # in-trade shading (up to the exit, or to the moment)
    if entry_known:
        end = view.exit_utc if moment is None else min(view.exit_utc, moment)
        fig.add_shape(type="rect", xref="x", yref="paper", x0=_wall(view.entry_utc),
                      x1=_wall(end), y0=0, y1=1, fillcolor=rgba("blue", 0.06),
                      line={"width": 0},
                      layer="below")
    if zones and record is not None and record.htf is not None:
        gap = record.htf
        start = max(window[0], gap.confirmed_utc) if gap.confirmed_utc is not None else window[0]
        if start < last:
            fig.add_shape(type="rect", xref="x", yref="y", x0=_wall(start), x1=_wall(last),
                          y0=_pts(gap.low_ticks), y1=_pts(gap.high_ticks),
                          fillcolor=rgba("blue_band", 0.75), line={"width": 0}, layer="below")
            share = 0.0  # point in time: never placed by where the exit came
            if moment is None:
                early_exit = view.exit_utc < window[0] + (window[1] - window[0]) / 2
                share = 0.62 if early_exit else 0.5
            _label(fig, _wall(start) + (_wall(last) - _wall(start)) * share,
                   max(y_range[0], _pts(gap.low_ticks)),
                   _zone_label(gap, y_range, related=not record.identity_established),
                   color=c["blue_dark"], bold=True, yshift=4, xanchor="left")
    if times:
        midnight = _midnight(window)
        if midnight is not None and _known(midnight, moment):
            fig.add_vline(x=_wall(midnight), line={"color": c["zero_line"], "dash": "dot",
                                                    "width": 1})
            _label(fig, _wall(midnight), y_range[0], "Midnight · same trading day",
                   color=c["muted"], xshift=4, yshift=4)
        close_at = _daily_close(view, window[1], close_clock)
        if close_at is not None and _known(close_at, moment):
            fig.add_vline(x=_wall(close_at), line={"color": c["orange"], "dash": "dot",
                                                    "width": 1})
    if stops and entry_known:
        first_end = view.half_utc or view.exit_utc
        first_end = first_end if moment is None else min(first_end, moment)
        _hline(fig, _wall(view.entry_utc), _wall(first_end), _pts(view.stop_ticks),
               color=c["orange"], width=2, name=f"Initial stop {_price(view.stop_ticks)}")
        if view.target_ticks is not None:
            target_end = view.half_utc or view.exit_utc
            target_end = target_end if moment is None else min(target_end, moment)
            _hline(fig, _wall(view.entry_utc), _wall(target_end), _pts(view.target_ticks),
                   color=c["blue_line"], dash="dot", width=1.2,
                   name=f"Target {_price(view.target_ticks)}")
        if view.half_utc is not None and _known(view.half_utc, moment) \
                and view.final_stop_ticks is not None:
            end = view.exit_utc if moment is None else min(view.exit_utc, moment)
            _hline(fig, _wall(view.half_utc), _wall(end), _pts(view.final_stop_ticks),
                   color=c["blue"], dash="dash", width=1.6,
                   name=f"Moved stop {_price(view.final_stop_ticks)}")
            moved = ("moved to entry" if view.final_stop_ticks == view.entry_ticks
                     else "moved")
            if end > view.half_utc + timedelta(minutes=60):
                _label(fig, _wall(end), _pts(view.final_stop_ticks),
                       f"Stop on the last {view.exit_quantity} {moved}, "
                       f"{_price(view.final_stop_ticks)}", color=c["blue_dark"],
                       xanchor="right", xshift=-6, yshift=2)
    _candles(fig, candles, size_minutes)
    if entry_known:
        _entry_marker(fig, view)
        _label(fig, _wall(view.entry_utc), _pts(view.entry_ticks),
               f"Entry {_price(view.entry_ticks)} · {clock(view.entry_utc)}",
               color=c["ink"], bold=True, yanchor="top", yshift=-12, xshift=8)
    _exit_dots(fig, view, moment)
    if _known(view.exit_utc, moment) and view.exit_ticks is not None:
        exit_y = _pts(view.exit_ticks)
        upper = exit_y > (y_range[0] + y_range[1]) / 2
        early = view.exit_utc < window[0] + (window[1] - window[0]) / 2
        fig.add_annotation(x=_wall(view.exit_utc), y=exit_y, text=_result_box_text(view),
                           showarrow=False, xanchor="left" if early else "right",
                           yanchor="top" if upper else "bottom",
                           xshift=12 if early else -12, yshift=-12 if upper else 12, align="left",
                           font={"family": FONT_SANS, "size": 12, "color": c["on_ink"]},
                           bgcolor=c["ink"], borderpad=8)
    if moment is not None and moment < window[1]:
        _cover(fig, _wall(moment), _wall(window[1]), hidden_sentence(moment, view.entry_utc))
    _axes(fig, x_range=x_range, y_range=y_range, height=height, dtick_hours=3)
    return fig


def whole_trade_caption(view: TradeView, *, moment: pd.Timestamp | None = None) -> str:
    """One sentence about the trade, using its recorded numbers (point in time aware)."""

    side = "Bought" if view.long else "Sold short"
    text = (f"{side} {view.quantity} at {_price(view.entry_ticks)} at "
            f"{clock(view.entry_utc)} with the stop at {_price(view.stop_ticks)}")
    if moment is not None and moment < view.exit_utc:
        if not _known(view.entry_utc, moment):
            return (f"Showing the trading day up to {when(moment, view.entry_utc)}, before the "
                    "entry; the entry, the exit and the result are hidden.")
        if view.half_utc is not None and _known(view.half_utc, moment):
            text += (f"; half came out at {_price(view.half_ticks)} at "
                     f"{clock(view.half_utc)}")
        return (text + f". Everything after {when(moment, view.entry_utc)}, including the exit "
                "and the result, is hidden.")
    if view.half_utc is not None:
        text += (f"; half came out at {_price(view.half_ticks)} at "
                 f"{clock(view.half_utc)} and the rest at "
                 f"{_price(view.exit_ticks)} ({exit_words(view)}, "
                 f"{when(view.exit_utc, view.entry_utc)})")
    else:
        text += (f"; it closed at {_price(view.exit_ticks)} ({exit_words(view)}, "
                 f"{when(view.exit_utc, view.entry_utc)})")
    result = fmt.money(view.net, signed=True, missing="a result not in this study's export")
    return text + f", for {result} after costs."


# ── the setup (one-minute candles around the entry) ───────────────────────


def setup_window(view: TradeView, record: SetupRecord | None, *, before: int = 25,
                 after: int = 18, longest: int = 60, moment: pd.Timestamp | None = None
                 ) -> tuple[pd.Timestamp, pd.Timestamp]:
    """About 25 minutes before the entry (back to the parent gap's first candle, at most an
    hour) to 18 minutes after, on whole five-minute marks; widened to keep a first exit
    that came within 45 minutes in view — only when that exit is already known (point in
    time never lets the window's width reveal when the trade ended). In point in time the
    parent gap widens the window only once it has formed (correction A7: no bound comes
    from anything after the moment)."""

    start = view.entry_utc - timedelta(minutes=before)
    parent = record.parent if record is not None else None
    if parent is not None and parent.first_open_utc and (
            moment is None or _known(parent.confirmed_utc, moment)):
        start = min(start, parent.first_open_utc - timedelta(minutes=5))
    start = max(start, view.entry_utc - timedelta(minutes=longest))
    start = start.floor("5min")
    end = view.entry_utc + timedelta(minutes=after)
    closing = view.half_utc or view.exit_utc  # keep the first exit in view when it is close
    if (closing is not None and end <= closing <= view.entry_utc + timedelta(minutes=45)
            and (moment is None or closing <= moment)):
        end = closing + timedelta(minutes=3)
    end = end.ceil("5min") if end != end.floor("5min") else end
    return start, end


def _number(fig, x, y, number: int) -> None:
    c = palette()
    fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[str(number)],
                             textposition="middle center", hoverinfo="skip",
                             textfont={"family": FONT_SANS, "size": 11, "color": c["on_ink"]},
                             marker={"size": 19, "color": c["ink"]}, showlegend=False))


def setup_figure(day: pd.DataFrame, view: TradeView, record: SetupRecord | None, *,
                 moment: pd.Timestamp | None = None, key: list | None = None,
                 height: int = 340) -> tuple[go.Figure, tuple[pd.Timestamp, pd.Timestamp]]:
    """One-minute candles around the entry with numbered markers matching ``key``."""

    start, end = setup_window(view, record, moment=moment)
    frame = day[(day["logical_open_ts_utc"] >= start) & (day["logical_open_ts_utc"] < end)] \
        if not day.empty else day
    candles = aggregate(visible_minutes(frame, moment), 1)
    last = end if moment is None else min(end, max(moment, start))
    fig = go.Figure()
    c = palette()
    values = list(candles["low"]) + list(candles["high"])
    if _known(view.entry_utc, moment):
        values += [_pts(view.entry_ticks), _pts(view.stop_ticks)]
        if view.target_ticks is not None:
            values.append(_pts(view.target_ticks))
    if _known(view.exit_utc, moment) and view.exit_utc < end and view.exit_ticks is not None:
        values.append(_pts(view.exit_ticks))
    if record is not None:
        for gap in (record.parent, record.opposing):
            if gap is not None and _known(gap.confirmed_utc, moment):
                values += [_pts(gap.low_ticks), _pts(gap.high_ticks)]
    if view.half_utc is not None and _known(view.half_utc, moment) and view.half_utc < end:
        values.append(_pts(view.half_ticks))
    y_range = _y_range(values, minimum_span=6.0)
    span = y_range[1] - y_range[0]
    y_range = (y_range[0] - span * 0.04, y_range[1] + span * 0.12)  # room for the markers
    roles = [role for role, _, _ in (key or [])]
    numbers = {role: i for i, role in enumerate(roles, start=1)}
    minute = timedelta(minutes=1)
    if record is not None:
        parent = record.parent
        if parent is not None and _known(parent.confirmed_utc, moment):
            x0 = max(parent.first_open_utc or start, start)
            fig.add_shape(type="rect", xref="x", yref="y", x0=_wall(x0), x1=_wall(last),
                          y0=_pts(parent.low_ticks), y1=_pts(parent.high_ticks),
                          fillcolor=rgba("blue_band", 0.85), line={"width": 0}, layer="below")
            if "parent" in numbers:
                _number(fig, _wall(x0 + minute * 1.5),
                        (_pts(parent.low_ticks) + _pts(parent.high_ticks)) / 2,
                        numbers["parent"])
        # the close-through as the key has it (review_panels.SetupEvent): known at its candle's
        # close; the candle's opening minute only places it
        inversion = next((e for e in setup_events(record) if e.role == "inversion"), None)
        inversion_open = inversion.candle_open if inversion is not None else None
        inversion_close = inversion.known_at if inversion is not None else None
        opposing = record.opposing
        if opposing is not None and _known(opposing.confirmed_utc, moment):
            x0 = max(opposing.first_open_utc or start, start)
            x1 = inversion_close or opposing.confirmed_utc
            x1 = min(x1, last) if x1 is not None else last
            low, high = _pts(opposing.low_ticks), _pts(opposing.high_ticks)
            if high - low < span * 0.02:
                hover = (f"Opposing gap {low:,.2f}–{high:,.2f}" if record.identity_established
                         else f"Related opposing gap {low:,.2f}–{high:,.2f} (context)")
                _hline(fig, _wall(x0), _wall(x1), (low + high) / 2, color=c["orange"],
                       width=4, name=hover)
            else:
                fig.add_shape(type="rect", xref="x", yref="y", x0=_wall(x0), x1=_wall(x1),
                              y0=low, y1=high, fillcolor=rgba("orange", 0.25),
                              line={"color": c["orange"], "width": 1}, layer="below")
            if "opposing" in numbers:
                _number(fig, _wall(x0 + (x1 - x0) / 2) if x1 > x0 else _wall(x0),
                        (low + high) / 2 - span * 0.07, numbers["opposing"])
        # shown exactly when the key lists it: in full history whenever it was recorded, at a
        # point in time only once its candle's recorded close is reached
        if inversion is not None and (moment is None or _known(inversion_close, moment)):
            if inversion_open is not None and inversion_close is not None:
                fig.add_shape(type="rect", xref="x", yref="paper", x0=_wall(inversion_open),
                              x1=_wall(inversion_close), y0=0, y1=1,
                              fillcolor=rgba("ink", 0.07), line={"width": 0}, layer="below")
            anchor = (inversion_open + minute / 2 if inversion_open is not None
                      else inversion_close)
            if "inversion" in numbers:
                _number(fig, _wall(anchor), y_range[1] - span * 0.05, numbers["inversion"])
    if _known(view.entry_utc, moment):
        first_end = view.half_utc or view.exit_utc
        first_end = min(first_end, last)
        _hline(fig, _wall(view.entry_utc), _wall(first_end), _pts(view.stop_ticks),
               color=c["orange"], dash="dash", width=2,
               name=f"Initial stop {_price(view.stop_ticks)}")
        near_end = first_end >= end - (end - start) * 0.3
        _label(fig, _wall(first_end), _pts(view.stop_ticks),
               f"Initial stop {_price(view.stop_ticks)}", color=c["orange_dark"], size=11,
               xshift=-4 if near_end else 6, xanchor="right" if near_end else "left",
               yanchor="bottom" if near_end else "middle", yshift=2 if near_end else 0)
        if view.target_ticks is not None:
            target_end = min(view.half_utc or view.exit_utc, last)
            _hline(fig, _wall(view.entry_utc), _wall(target_end), _pts(view.target_ticks),
                   color=c["blue"], dash="dash", width=2,
                   name=f"Target {_price(view.target_ticks)}")
        _entry_marker(fig, view, size=12)
        if "entry" in numbers:
            _number(fig, _wall(view.entry_utc + minute * 1.2),
                    _pts(view.entry_ticks) - span * 0.07, numbers["entry"])
    if view.half_utc is not None and _known(view.half_utc, moment) and view.half_utc < end:
        if view.final_stop_ticks is not None:
            stop_end = min(view.exit_utc, last)
            _hline(fig, _wall(view.half_utc), _wall(stop_end), _pts(view.final_stop_ticks),
                   color=c["blue"], dash="dash", width=1.6,
                   name=f"Moved stop {_price(view.final_stop_ticks)}")
            words = ("Stop moved to entry" if view.final_stop_ticks == view.entry_ticks
                     else f"Stop moved to {_price(view.final_stop_ticks)}")
            if stop_end - view.half_utc >= timedelta(minutes=4):
                _label(fig, _wall(view.half_utc + (stop_end - view.half_utc) / 2),
                       _pts(view.final_stop_ticks), words, color=c["blue_dark"], size=11,
                       xanchor="center", yshift=3)
        if "half" in numbers:
            _number(fig, _wall(view.half_utc), _pts(view.half_ticks) + span * 0.08,
                    numbers["half"])
    elif "exit" in numbers and _known(view.exit_utc, moment) and view.exit_utc < end:
        _number(fig, _wall(view.exit_utc), _pts(view.exit_ticks) + span * 0.08,
                numbers["exit"])
    _candles(fig, candles, 1)
    _exit_dots(fig, _within(view, end), moment, size=10)
    if moment is not None and moment < end:
        _cover(fig, _wall(max(moment, start)), _wall(end), None)
    _axes(fig, x_range=(_wall(start), _wall(end)), y_range=y_range, height=height)
    fig.update_xaxes(dtick=10 * 60 * 1000, tickformat="%-I:%M")
    return fig, (start, end)


def _within(view: TradeView, end: pd.Timestamp) -> TradeView:
    """The view with fills after the setup window dropped (so no dot is drawn off-chart)."""

    from dataclasses import replace

    changes: dict[str, Any] = {}
    if view.half_utc is not None and view.half_utc >= end:
        changes.update(half_utc=None)
    if view.exit_utc >= end:
        changes.update(exit_ticks=None)
    return replace(view, **changes) if changes else view
