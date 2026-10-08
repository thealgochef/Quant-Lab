"""Configuration detail · Trades (mock 06).

For ONE configuration at ONE firm of one saved funded result:

- "Trade results": the funded trades' net results in the eleven result bins;
- "Performance summary": All, Long and Short columns;
- "Account equity range during each trade": lowest and highest account
  equity relative to its starting balance against the final result;
- "Trades": the trade list with both fills of a half exit; the row where an
  account's loss limit ended a winning trade is tinted, and every row opens that
  exact trade in Trade review;
- "More": the earlier trade table with every column it had (rule 12).

Every figure comes from the saved funded trade rows through ``lab.trade_stats``
(CALCULATIONS.md "Trades tab"); nothing is recomputed from prices, nothing is
saved, and firms are never added together.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab import trade_stats as ts
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var, palette, style_chart

__all__ = [
    "CALM",
    "FIRST_ROWS",
    "SHOW_ALL",
    "TAIL",
    "distribution_caption",
    "distribution_card",
    "earlier_rows",
    "excursion_figure",
    "excursion_pairs",
    "excursion_sentence",
    "handle_action",
    "money_ticks",
    "performance_card",
    "render",
    "risk_row",
    "trade_action",
    "trades_card",
]

#: rows shown before "Show all"
FIRST_ROWS = 8
#: the threshold of the counts sentence under the excursion charts
CALM = 250.0
#: results above this make the long right tail named in the distribution caption
TAIL = 2000.0
#: the shared per-result context key remembering "Show all"
SHOW_ALL = "trades_show_all"

_WORDS = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten")
#: a CSS variable name only (no palette value is read at import time)
_MUTED_DASH = h.Markup(f'<span style="color:{css_var("muted")};font-family:var(--lab-sans)">'
                       f"{fmt.MISSING}</span>")


def _at(value: Any):
    """The stored instant, truncated to the whole second for display (never the next minute).

    Exit instants are stored with nanoseconds; the shared formatter would warn
    about discarding them, so they are dropped here first. Stored values are
    never changed.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant

    instant = utc_instant(value)
    return None if instant is None else instant.floor("s")


def _net(trade: dict[str, Any]) -> float:
    return float(trade.get("net_pnl_usd") or 0.0)


def _count_words(n: int, singular: str, plural: str | None = None) -> str:
    """``Ten trades`` at the start of a sentence (digits above ten)."""

    word = _WORDS[n] if 0 <= n < len(_WORDS) else f"{n:,}"
    return f"{word} {singular if n == 1 else (plural or singular + 's')}"


# ── trade results (distribution) ──────────────────────────────────────────


def _bin_title(count: int, low: float, high: float) -> str:
    trades = fmt.count(count, "trade")
    if math.isinf(low):
        return f"{trades} lost more than {fmt.money_whole(-high)}"
    if math.isinf(high):
        return f"{trades} made {fmt.money_whole(low)} or more"
    return f"{trades} from {fmt.money_whole(low)} up to {fmt.money_whole(high)}"


def distribution_caption(trades: Sequence[dict[str, Any]]) -> str:
    """One or two sentences naming what the distribution shows, with the actual numbers.

    "Small" = the worst loss is at most half the best win; "bunched" = every
    loss fits inside the chart's $250 steps (under $1,000). Both are stated with
    the numbers they rest on.
    """

    values = [_net(t) for t in trades]
    if not values:
        return "No funded trades at this firm."
    losses = [v for v in values if v <= 0]
    wins = [v for v in values if v > 0]
    parts: list[str] = []
    if not losses:
        parts.append("No trade lost money.")
    else:
        worst = min(losses)
        best = max(wins) if wins else None
        losers = fmt.count(len(losses), "loser")
        small = best is not None and abs(worst) <= best / 2
        bunched = abs(worst) < 1000
        if small and bunched:
            parts.append(f"Losses stay small and bunched: the worst of {losers} is "
                         f"{fmt.money_whole(worst)}.")
        elif small:
            parts.append(f"Losses stay small next to the wins: the worst of {losers} is "
                         f"{fmt.money_whole(worst)}.")
        else:
            against = f", against a best trade of {fmt.money_whole(best)}" if best else ""
            parts.append(f"Losses run large: the worst of {losers} is "
                         f"{fmt.money_whole(worst)}{against}.")
    tail = sum(1 for v in values if v > TAIL)
    if tail:
        verb = "makes" if tail == 1 else "make"
        parts.append(f"{_count_words(tail, 'trade')} above {fmt.money_whole(TAIL)} {verb} the "
                     "long right tail.")
    elif wins:
        parts.append(f"No trade made more than {fmt.money_whole(TAIL)}; the best made "
                     f"{fmt.money_whole(max(wins))}.")
    return " ".join(parts)


def distribution_card(trades: Sequence[dict[str, Any]]) -> h.Markup:
    """Mock 06 "Trade results": eleven bars, losers orange, winners blue, counts on top."""

    bins = ts.result_distribution(trades)
    tallest = max((count for _, count, _, _ in bins), default=0) or 1
    columns, labels, spoken = [], [], []
    for label, count, low, high in bins:
        color = css_var("orange") if high <= 0 else css_var("blue")
        height = round(160 * count / tallest)
        title = _bin_title(count, low, high)
        spoken.append(title)
        columns.append(
            f'<div title="{h.esc(title)}" style="flex:1;display:flex;flex-direction:column;'
            'align-items:center;gap:4px;justify-content:flex-end;min-width:0">'
            f'<span class="lab-mono" style="font-size:11px">{count}</span>'
            f'<div style="width:100%;height:{height}px;background:{color}"></div></div>')
        labels.append(f'<span style="flex:1;min-width:0">{h.esc(label)}</span>')
    chart = (
        f'<div role="img" aria-label="{h.esc("; ".join(spoken))}" '
        'style="display:flex;flex-direction:column;gap:6px">'
        '<div style="display:flex;align-items:flex-end;gap:6px;height:180px;'
        f'border-bottom:1px solid {css_var("rule")}">{"".join(columns)}</div>'
        f'<div style="display:flex;gap:6px;font-size:11px;color:{css_var("muted")};'
        f'text-align:center;line-height:1.3">{"".join(labels)}</div>'
        f'<div style="font-size:12px;color:{css_var("muted")}">Result per trade after costs; '
        "each bar runs from its label up to the next.</div></div>")
    caption = (f'<div style="font-size:14px;line-height:1.5;color:{css_var("body")}">'
               f"{h.esc(distribution_caption(trades))}</div>")
    return h.card(h.Markup(chart + caption),
                  title=f"Trade results, {fmt.count(len(trades), 'funded trade')}")


# ── performance summary ───────────────────────────────────────────────────

_MEASURES = ("Trades · win rate", "Gross profit", "Gross loss", "Average win · average loss",
             "Largest win · largest loss", "Longest winning · losing streak",
             "Drawdowns · average length", "Average drawdown · worst", "Longest drawdown")


def _column_values(col: ts.PerformanceColumn) -> list[Any]:
    if col.trades == 0:
        return [_MUTED_DASH] * len(_MEASURES)
    wins = col.average_win is not None
    losses = col.average_loss is not None
    dd = col.drawdowns
    has_dd = dd is not None and dd.count > 0
    return [
        f"{col.trades:,} · {fmt.percent(col.win_rate, decimals=1)}",
        fmt.money(col.gross_profit),
        fmt.money(col.gross_loss),
        f"{fmt.money_whole(col.average_win)} · {fmt.money_whole(col.average_loss)}",
        f"{fmt.money_whole(col.largest_win) if wins else fmt.MISSING} · "
        f"{fmt.money_whole(col.largest_loss) if losses else fmt.MISSING}",
        f"{col.longest_winning} · {col.longest_losing}",
        (f"{dd.count} · {fmt.number(dd.average_length, decimals=1)} trades" if has_dd
         else "None"),
        (f"{fmt.money_whole(dd.average_depth)} · {fmt.money_whole(dd.worst_depth)}" if has_dd
         else fmt.MISSING),
        fmt.count(int(dd.longest_length or 0), "trade") if has_dd else fmt.MISSING,
    ]


_PERF_CSS = (".lab-trades-perf .lab-table td{padding:7px 8px;font-size:14px}"
             ".lab-trades-perf .lab-table td:first-child{min-width:120px}"
             ".lab-trades-perf .lab-table th{padding:8px 8px}"
             ".lab-trades-perf .lab-table td:first-child,"
             ".lab-trades-perf .lab-table th:first-child{padding-left:0}"
             ".lab-trades-perf .lab-table td:last-child,"
             ".lab-trades-perf .lab-table th:last-child{padding-right:0}")


def _nowrap(value: Any) -> h.Markup:
    """Keep each half of ``$7,350 · −$950`` on one line; wrap only at the separator."""

    if isinstance(value, h.Markup) or " · " not in str(value):
        return h.Markup(f'<span style="white-space:nowrap">{h.esc(value)}</span>')
    first, _, second = str(value).partition(" · ")
    return h.Markup(f'<span style="white-space:nowrap">{h.esc(first)} ·</span> '
                    f'<span style="white-space:nowrap">{h.esc(second)}</span>')


def _direction(settings: Sequence[tuple[str, str]]) -> str:
    """``long`` / ``short`` / ``both`` from the configuration's readable Direction setting."""

    value = dict(settings).get("Direction", "").strip().lower()
    if value.startswith("long only"):
        return "long"
    if value.startswith("short only"):
        return "short"
    return "both"


def performance_card(trades: Sequence[dict[str, Any]], settings: Sequence[tuple[str, str]],
                     firm: str) -> h.Markup:
    """Mock 06 "Performance summary": All, Long and Short columns of the funded trades."""

    direction = _direction(settings)
    columns_data = {
        "all": ts.performance_column(trades),
        "long": ts.performance_column(trades, "long"),
        "short": ts.performance_column(trades, "short"),
    }
    values = {key: [_nowrap(v) for v in _column_values(col)]
              for key, col in columns_data.items()}
    rows = [h.Row({"measure": measure, **{key: values[key][i] for key in values}})
            for i, measure in enumerate(_MEASURES)]
    # an empty column stays narrow (mock: 80 px); filled columns share the rest evenly
    filled = [key for key, col in columns_data.items() if col.trades]
    empty_share = 10
    label_share = 30 if len(filled) < 3 else 28
    each = (100 - label_share - empty_share * (3 - len(filled))) // max(len(filled), 1)
    widths = {key: f"{each if key in filled else empty_share}%" for key in columns_data}
    table = h.table([h.Column("measure", "", width=f"{label_share}%"),
                     h.Column("all", "All", "right", width=widths["all"]),
                     h.Column("long", "Long", "right", width=widths["long"]),
                     h.Column("short", "Short", "right", width=widths["short"])],
                    rows, plain=True, wrap=False)
    # the mock's compact rows (7 px) instead of the ranking table's 12 px
    table = h.Markup(f"<style>{_PERF_CSS}</style>"
                     f'<div class="lab-trades-perf">{table}</div>')
    notes = []
    if direction == "long":
        notes.append("Short column stays empty on long-only configurations.")
    elif direction == "short":
        notes.append("Long column stays empty on short-only configurations.")
    else:
        for side in ("long", "short"):
            if columns_data[side].trades == 0:
                notes.append(f"No {side} trades were taken at {firm}.")
    notes.append("A drawdown runs from a high of the running result until that high is "
                  "regained; its length counts the trades spent below the high.")
    body = "".join(f'<div class="lab-line" style="font-size:13px">{h.esc(n)}</div>'
                   for n in notes)
    return h.card(h.Markup(f"{table}{body}"), title="Performance summary")


# ── excursions ────────────────────────────────────────────────────────────


def excursion_pairs(trades: Sequence[dict[str, Any]]
                    ) -> tuple[list[tuple[ts.Excursion, dict[str, Any]]], int]:
    """(excursion, trade) for every trade with a recorded equity range; and the count left out."""

    pairs = []
    for trade in trades:
        found = ts.excursions([trade])
        if found:
            pairs.append((found[0], trade))
    return pairs, len(trades) - len(pairs)


def excursion_sentence(pairs: Sequence[tuple[ts.Excursion, dict[str, Any]]]) -> str:
    """``36 of 47 losers were never more than $250 in profit before …``."""

    items = [e for e, _ in pairs]
    calm_losers, losers, calm_winners, winners = ts.excursion_counts(items, CALM)
    threshold = fmt.money_whole(CALM)
    parts = []
    if losers:
        at_stop = all(str(t.get("exit_kind")) in ("stop", "breakeven_stop")
                      for e, t in pairs if not e.winner)
        verb = "were" if losers != 1 else "was"
        ending = "before the stop" if at_stop else "before they closed"
        parts.append(f"{calm_losers} of {fmt.count(losers, 'loser')} {verb} never more than "
                     f"{threshold} in profit {ending}.")
    if winners:
        parts.append(f"{calm_winners} of {fmt.count(winners, 'winner')} went less than "
                     f"{threshold} against you first.")
    return " ".join(parts) or "No trades with a recorded range during the trade."


def money_ticks(low: float, high: float, *, most: int = 5) -> tuple[list[float], list[str]]:
    """Round money ticks covering [low, high] labelled like ``−$1k``, ``$0``, ``$2k``."""

    span = max(float(high) - float(low), 1.0)
    steps = (50, 100, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000, 100000)
    step = next((s for s in steps if span / s <= most), steps[-1])
    first = math.floor(float(low) / step) * step
    last = math.ceil(float(high) / step) * step
    values = [first + i * step for i in range(int(round((last - first) / step)) + 1)]
    return values, [fmt.money_short(v) for v in values]


def excursion_figure(pairs: Sequence[tuple[ts.Excursion, dict[str, Any]]], which: str):
    """Worst (``which="worst"``) or best point during each trade against its final result."""

    import plotly.graph_objects as go

    worst = which == "worst"
    c = palette()  # the active theme's values, read when the figure is built
    fig = go.Figure()
    for winner, name, color in ((True, "Winner", c["blue"]), (False, "Loser", c["orange"])):
        chosen = [(e, t) for e, t in pairs if e.winner is winner]
        if not chosen:
            continue
        fig.add_trace(go.Scatter(
            x=[e.worst if worst else e.best for e, _ in chosen],
            y=[e.result for e, _ in chosen],
            mode="markers", name=name,
            marker={"size": 8, "color": color, "opacity": 0.78, "line": {"width": 0}},
            customdata=[[str(t.get("account_number")), fmt.chicago_short(_at(t.get("entry_utc"))),
                         fmt.money(e.worst if worst else e.best),
                         fmt.money(e.result, signed=True)] for e, t in chosen],
            hovertemplate=("Account %{customdata[0]}, entered %{customdata[1]}<br>"
                           + ("Account equity below start: " if worst else
                              "Account equity above start: ")
                           + "%{customdata[2]}<br>Final result: %{customdata[3]}"
                           f"<extra>{name}</extra>")))
    style_chart(fig, height=300, money_axis=None,
                x_title="Account equity below starting balance" if worst
                else "Account equity above starting balance",
                y_title="Final result after costs")
    xs = [e.worst if worst else e.best for e, _ in pairs] or [0.0]
    ys = [e.result for e, _ in pairs] or [0.0]
    x_ticks, x_text = money_ticks(min(0.0, min(xs)), max(xs), most=4)
    y_ticks, y_text = money_ticks(min(0.0, min(ys)), max(ys))
    fig.update_xaxes(tickvals=x_ticks, ticktext=x_text)
    fig.update_yaxes(tickvals=y_ticks, ticktext=y_text)
    fig.add_hline(y=0, line_color=c["rule"], line_width=1)
    return fig


def _excursion_header() -> h.Markup:
    dot = ('<span style="display:inline-flex;align-items:center;gap:6px">'
           '<span style="width:10px;height:10px;border-radius:5px;background:{c}"></span>'
           "{t}</span>")
    legend = (f'<div style="display:flex;gap:16px;font-size:13px;color:{css_var("body_2")}">'
              + dot.format(c=css_var("blue"), t="Winner")
              + dot.format(c=css_var("orange"), t="Loser") + "</div>")
    return h.Markup(
        '<div class="lab" style="display:flex;justify-content:space-between;'
        'align-items:baseline;gap:16px;flex-wrap:wrap">'
        '<div class="lab-card-title">Account equity range during each trade</div>'
        f"{legend}</div>")


def _subtitle(text: str) -> h.Markup:
    return h.Markup(f'<div class="lab" style="font-size:14px;font-weight:600">{h.esc(text)}'
                    "</div>")


# ── trade list ────────────────────────────────────────────────────────────


def risk_row(trade: dict[str, Any]) -> bool:
    """An account's loss limit ended a WINNING trade (tinted in the list)."""

    return str(trade.get("exit_kind")) == "account_failure" and _net(trade) > 0


def trade_action(trade: dict[str, Any]) -> str:
    """``trade:<funded trade seq>|<account number>`` — sent back when the row is clicked."""

    return f"trade:{int(trade['seq'])}|{int(trade.get('account_number') or 0)}"


def _why(trade: dict[str, Any]) -> Any:
    reason = ts.close_reason(trade)
    if str(trade.get("exit_kind")) == "account_failure":
        return h.Markup(f'<span style="color:{css_var("orange_dark")};font-weight:600">'
                        f"{h.esc(reason)}</span>")
    return reason


def trades_card(trades: Sequence[dict[str, Any]], *, show_all: bool) -> h.Markup:
    """Mock 06 "Trades": the first eight trades (or all), each row opening Trade review."""

    columns = [
        h.Column("account", "Account", width="70px"),
        h.Column("entry", "Entry"),
        h.Column("exit", "Exit"),
        h.Column("price", "Entry price", "right"),
        h.Column("stop", "Stop", "right"),
        h.Column("filled", "Filled", mono=True),
        h.Column("why", "Why it closed"),
        h.Column("result", "Result", "right"),
    ]
    shown = list(trades) if show_all else list(trades)[:FIRST_ROWS]
    rows = []
    for trade in shown:
        entry = fmt.chicago_short(_at(trade.get("entry_utc")))
        rows.append(h.Row(
            {"account": str(trade.get("account_number") or fmt.MISSING),
             "entry": entry,
             "exit": fmt.chicago_short(_at(trade.get("exit_utc"))),
             "price": fmt.points(trade.get("entry_ticks"), from_ticks=True),
             "stop": fmt.points(trade.get("stop_ticks"), from_ticks=True),
             "filled": ts.fills_text(trade),
             "why": _why(trade),
             "result": fmt.money(trade.get("net_pnl_usd"), signed=True)},
            tint="risk" if risk_row(trade) else None,
            action=trade_action(trade),
            label=f"Open the Account {trade.get('account_number')} trade entered {entry} "
                  "in Trade review"))
    table = h.table(columns, rows, plain=True, wrap=False)
    total = len(trades)
    showing = (f"Showing all {total:,}" if len(shown) == total
               else f"Showing {len(shown):,} of {total:,}")
    toggle = ""
    if total > FIRST_ROWS:
        toggle = str(h.link(f"Show the first {FIRST_ROWS}" if show_all else "Show all",
                            "showall"))
    foot = (f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding-top:8px;font-size:14px;color:{css_var("body_2")}">'
            f"<span>{h.esc(showing)} · click any trade to open it in Trade review</span>"
            f'<span style="font-weight:500">{toggle}</span></div>')
    return h.card(h.Markup(f"{table}{foot}"), title="Trades",
                  right="Prices in index points · results after costs")


def handle_action(action: str | None, ctx, trades: Sequence[dict[str, Any]],
                  st_module) -> None:
    """A click in the trade list: open that exact trade in Trade review, or show all rows."""

    if not action:
        return
    kind, _, value = str(action).partition(":")
    if kind == "showall":
        ctx.context[SHOW_ALL] = not ctx.context.get(SHOW_ALL, False)
        st_module.rerun()
        return
    if kind != "trade":
        return
    seq_text, _, account_text = value.partition("|")
    try:
        seq, account = int(seq_text), int(account_text)
    except ValueError:
        return
    if not any(t.get("seq") is not None and int(t["seq"]) == seq for t in trades):
        return  # never open a trade this configuration and firm did not take
    from ifvg_lab_nav import open_trade_review

    open_trade_review(ctx.target, configuration=ctx.configuration, firm_key=ctx.firm_key,
                      account_number=account, trade_seq=seq, back_tab="Trades",
                      st_module=st_module)


# ── "More": the earlier trade table (rule 12) ─────────────────────────────


def earlier_rows(study, configuration: str, firm_key: str) -> list[dict[str, str]]:
    """Every column of the earlier funded trade table (repair R2/R3 presenter), plus the day.

    Times use the one Chicago format of the redesign (rule 11); every other
    value is the earlier presenter's own text.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        present_pair_detail,
    )

    detail = present_pair_detail(study.result, configuration, firm_key)
    trades = ordered_trades(study, configuration, firm_key)
    aligned = len(detail.trades) == len(trades) and all(
        row.account == f"Account {t.get('account_number')}"
        for row, t in zip(detail.trades, trades, strict=False))
    rows = []
    for index, row in enumerate(detail.trades):
        shown = row.display()
        out: dict[str, str] = {"Trade": shown.pop("Trade")}
        if aligned:
            trade = trades[index]
            out["Trading day"] = fmt.trading_day_label(trade.get("trading_day"))
            shown["Entry (Chicago)"] = fmt.chicago_long(_at(trade.get("entry_utc")))
            shown["Exit (Chicago)"] = fmt.chicago_long(_at(trade.get("exit_utc")))
        out.update(shown)
        rows.append(out)
    return rows


def _more(st_module, ctx) -> None:
    import pandas as pd
    from ifvg_lab_ui import show

    with st_module.expander("More: every column of the earlier trade table", expanded=False):
        show(h.Markup('<div class="lab-line">Direction, size, exit price, points gained or '
                      "lost, initial risk, where each price came from and whether a stop "
                      "filled worse than the stop, for every trade. Prices are index points; "
                      "results are after costs.</div>"), st_module)
        rows = earlier_rows(ctx.study, ctx.configuration, ctx.firm_key)
        if not rows:
            show(h.placeholder("No trades were taken in this period."), st_module)
            return
        st_module.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                            key=f"ifvg_lab_v1_trades_more_{ctx.configuration}_{ctx.firm_key}")


# ── page ──────────────────────────────────────────────────────────────────


def _grid_start(items: Sequence[h.Markup]) -> h.Markup:
    """Two cards side by side, each as tall as its own content (mock: top-aligned)."""

    return h.Markup('<div class="lab lab-grid" style="grid-template-columns:repeat(2, '
                    'minmax(0, 1fr));gap:16px;align-items:start">'
                    + "".join(str(i) for i in items) + "</div>")


def render(st_module, ctx) -> None:
    from ifvg_lab_ui import clickable, plot, show

    trades = ordered_trades(ctx.study, ctx.configuration, ctx.firm_key)
    if not trades:
        show(h.card(h.placeholder(f"No funded trades were taken by this configuration at "
                                  f"{ctx.firm}."), title="Trades"), st_module)
        return
    settings = ctx.study.settings(ctx.configuration)
    show(_grid_start([distribution_card(trades),
                      performance_card(trades, settings, ctx.firm)]), st_module)

    pairs, left_out = excursion_pairs(trades)
    with st_module.container(key="ifvg_lab_card_trades_excursions"):
        show(_excursion_header(), st_module)
        if pairs:
            left, right = st_module.columns(2, gap="large")
            with left:
                show(_subtitle("Lowest account equity vs final result"), st_module)
                plot(excursion_figure(pairs, "worst"), key="trades_worst", st_module=st_module)
            with right:
                show(_subtitle("Highest account equity vs final result"), st_module)
                plot(excursion_figure(pairs, "best"), key="trades_best", st_module=st_module)
        sentence = excursion_sentence(pairs)
        how = ("Measured on the account's recorded lowest and highest equity during each "
               "trade, after costs.")
        if left_out:
            how += (f" {fmt.count(left_out, 'trade')} without a recorded equity range "
                    f"{'is' if left_out == 1 else 'are'} left out.")
        show(h.Markup(f'<div class="lab" style="font-size:14px;line-height:1.5;'
                      f'color:{css_var("body")}">{h.esc(sentence)}</div>'
                      f'<div class="lab-line" style="font-size:13px">{h.esc(how)}</div>'),
             st_module)

    show_all = bool(ctx.context.get(SHOW_ALL, False))
    action = clickable(trades_card(trades, show_all=show_all),
                       key=f"trades_{ctx.configuration}_{ctx.firm_key}", st_module=st_module)
    handle_action(action, ctx, trades, st_module)
    _more(st_module, ctx)
