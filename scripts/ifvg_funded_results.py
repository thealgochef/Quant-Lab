"""Completed-results screen for a funded-account payout simulation.

``render_funded_results(st, result)`` shows the ONE saved funded result through
the pure presenter :func:`present_funded_result`. It never recomputes money.
Firms stay separate everywhere (side-by-side cards, one line per firm on
charts, one tab per firm for detail); there is no grand total across firms.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from alpha_lab.agents.data_infra.ifvg.presentation.funded_results import (
    AccountRow,
    FirmView,
    FundedResultView,
    chicago_datetime,
    present_funded_result,
)

__all__ = [
    "render_funded_results",
    "cash_over_time_figure",
    "monthly_figure",
    "journeys_figure",
    "account_path_figure",
]

KEY_PREFIX = "funded_results_v1_"

#: One consistent style per firm: color plus a non-color cue (dash + marker).
FIRM_STYLES = (
    {"color": "#1f5fa8", "dash": "solid", "symbol": "circle"},
    {"color": "#c0621b", "dash": "dash", "symbol": "square"},
)
_RECEIVED_COLOR = "#2e7d32"
_COST_COLOR = "#8d6e63"
_FLOOR_COLOR = "#b71c1c"


def _style(index: int) -> dict[str, str]:
    return FIRM_STYLES[index % len(FIRM_STYLES)]


def _plotly(st, fig, key: str) -> None:
    st.plotly_chart(fig, key=f"{KEY_PREFIX}{key}", config={"displaylogo": False})


# ── charts ────────────────────────────────────────────────────────────────


def cash_over_time_figure(view: FundedResultView) -> go.Figure:
    """Cumulative received / account costs / net cash, shared Chicago date axis."""

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=(
            "Payouts received after the firm's share (cumulative, US dollars)",
            "Account costs (cumulative, US dollars)",
            "Net cash earned after all account costs (cumulative, US dollars)",
        ),
    )
    for index, firm in enumerate(view.firms):
        style = _style(index)
        xs = [chicago_datetime(p.ts_utc) for p in firm.cash]
        # Extend each step line to the end of the period so a flat stretch
        # (no payouts) is visible instead of the line stopping early.
        end = chicago_datetime(view.period_end_utc)
        series = (
            [p.received_usd for p in firm.cash],
            [p.account_costs_usd for p in firm.cash],
            [p.net_cash_usd for p in firm.cash],
        )
        for row, values in enumerate(series, start=1):
            x = list(xs)
            y = list(values)
            if end is not None and y:
                x.append(end)
                y.append(y[-1])
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="lines+markers", name=firm.firm, legendgroup=firm.firm,
                    showlegend=row == 1, line={"color": style["color"], "dash": style["dash"],
                                               "shape": "hv"},
                    marker={"symbol": style["symbol"], "size": 5},
                    hovertemplate=(f"{firm.firm}<br>%{{x|%B %-d, %Y %-I:%M %p}} Chicago"
                                   "<br>$%{y:,.2f}<extra></extra>"),
                ),
                row=row, col=1,
            )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_xaxes(title_text="Date (Chicago time)", row=3, col=1)
    fig.update_layout(height=640, margin={"t": 60, "b": 40, "l": 60, "r": 20},
                      legend={"orientation": "h", "y": 1.08, "x": 0},
                      hovermode="closest")
    return fig


def monthly_figure(view: FundedResultView) -> go.Figure:
    """Monthly received cash and account costs, one panel per firm (never summed)."""

    firms = view.firms or ()
    fig = make_subplots(
        rows=1, cols=max(1, len(firms)), shared_yaxes=True,
        subplot_titles=[f.firm for f in firms] or ["No firms"],
    )
    for col, firm in enumerate(firms, start=1):
        labels = [m.label for m in firm.months]
        for name, values, color, pattern in (
            ("Payouts received", [m.received_usd for m in firm.months], _RECEIVED_COLOR, ""),
            ("Account costs", [m.account_costs_usd for m in firm.months], _COST_COLOR, "/"),
        ):
            fig.add_trace(
                go.Bar(
                    x=labels, y=values, name=name, legendgroup=name, showlegend=col == 1,
                    marker={"color": color, "pattern": {"shape": pattern}},
                    text=[f"${v:,.0f}" for v in values], textposition="outside",
                    hovertemplate=f"{firm.firm} · {name}<br>%{{x}}<br>$%{{y:,.2f}}<extra></extra>",
                ),
                row=1, col=col,
            )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f", title_text="US dollars", col=1)
    fig.update_layout(barmode="group", height=420,
                      margin={"t": 60, "b": 40, "l": 60, "r": 20},
                      legend={"orientation": "h", "y": 1.15, "x": 0})
    return fig


def journeys_figure(view: FundedResultView, firm: FirmView) -> go.Figure:
    """One row per account: start, processing pauses, payouts, loss, replacement."""

    fig = go.Figure()
    end = chicago_datetime(view.period_end_utc)
    labels = [a.label for a in firm.accounts]
    by_id = {a.account_id: a for a in firm.accounts}
    shown: set[str] = set()

    def legend(name: str) -> bool:
        if name in shown:
            return False
        shown.add(name)
        return True

    for account in firm.accounts:
        start = chicago_datetime(account.created_utc)
        stop = chicago_datetime(account.failed_utc) if account.lost else end
        name = "Account alive"
        fig.add_trace(go.Scatter(
            x=[start, stop], y=[account.label, account.label], mode="lines",
            line={"color": "#607d8b", "width": 3}, name=name, legendgroup=name,
            showlegend=legend(name),
            hovertemplate=f"{account.label}<br>{account.status}<extra></extra>",
        ))
        for begin, finish in account.processing_intervals_utc:
            name = "Payout processing (trading paused)"
            fig.add_trace(go.Scatter(
                x=[chicago_datetime(begin), chicago_datetime(finish) if finish else end],
                y=[account.label, account.label], mode="lines",
                line={"color": "#f9a825", "width": 9}, name=name, legendgroup=name,
                showlegend=legend(name),
                hovertemplate=f"{account.label}<br>Payout processing<extra></extra>",
            ))
        name = "Started (fresh funded account)"
        fig.add_trace(go.Scatter(
            x=[start], y=[account.label], mode="markers", name=name, legendgroup=name,
            showlegend=legend(name),
            marker={"symbol": "triangle-right", "size": 10, "color": "#37474f"},
            hovertemplate=f"{account.label} started<br>{account.created}<br>"
            f"{account.funding}<extra></extra>",
        ))
        if account.receipts_utc:
            name = "Payout received ($ after split)"
            fig.add_trace(go.Scatter(
                x=[chicago_datetime(ts) for ts, _ in account.receipts_utc],
                y=[account.label] * len(account.receipts_utc), mode="markers+text",
                text=[f"${cents / 100:,.0f}" for _, cents in account.receipts_utc],
                textposition="top center", textfont={"size": 9},
                name=name, legendgroup=name, showlegend=legend(name),
                marker={"symbol": "diamond", "size": 11, "color": _RECEIVED_COLOR},
                hovertemplate=f"{account.label}<br>Payout received %{{text}}"
                "<extra></extra>",
            ))
        if account.lost:
            name = "Account lost"
            fig.add_trace(go.Scatter(
                x=[chicago_datetime(account.failed_utc)], y=[account.label],
                mode="markers+text", text=["lost"], textposition="middle right",
                name=name, legendgroup=name, showlegend=legend(name),
                marker={"symbol": "x", "size": 12, "color": _FLOOR_COLOR},
                hovertemplate=f"{account.label} lost<br>{account.failed}<br>"
                f"{account.failure_reason}<extra></extra>",
            ))
        if account.replaces_id and account.replaces_id in by_id:
            old = by_id[account.replaces_id]
            name = "Replacement link"
            fig.add_trace(go.Scatter(
                x=[chicago_datetime(old.failed_utc), start], y=[old.label, account.label],
                mode="lines", line={"color": "#9e9e9e", "dash": "dot", "width": 1},
                name=name, legendgroup=name, showlegend=legend(name),
                hovertemplate=f"{account.label} replaces {old.label}<extra></extra>",
            ))
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(labels)),
                     title_text="Account")
    fig.update_xaxes(title_text="Date (Chicago time)", range=[
        chicago_datetime(view.period_start_utc), end])
    fig.update_layout(height=max(320, 26 * len(labels) + 140),
                      margin={"t": 40, "b": 40, "l": 90, "r": 20},
                      legend={"orientation": "h", "y": 1.06, "x": 0})
    return fig


def account_path_figure(account: AccountRow) -> go.Figure:
    """Realized balance and the lowest open equity versus the loss limit."""

    fig = go.Figure()
    steps = [p for p in account.path if p.floor_usd is not None]
    lows = [p for p in account.path if p.floor_usd is None]
    fig.add_trace(go.Scatter(
        x=[chicago_datetime(p.ts_utc) for p in steps], y=[p.balance_usd for p in steps],
        mode="lines+markers", name="Account balance (profit since start)",
        line={"color": "#1f5fa8", "shape": "hv"}, marker={"symbol": "circle", "size": 6},
        text=[p.what for p in steps],
        hovertemplate="%{text}<br>%{x|%B %-d, %Y %-I:%M %p} Chicago"
        "<br>Balance $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[chicago_datetime(p.ts_utc) for p in steps], y=[p.floor_usd for p in steps],
        mode="lines", name="Loss limit (account is lost at this level)",
        line={"color": _FLOOR_COLOR, "dash": "dash", "shape": "hv"},
        hovertemplate="Loss limit $%{y:,.2f}<extra></extra>",
    ))
    if lows:
        fig.add_trace(go.Scatter(
            x=[chicago_datetime(p.ts_utc) for p in lows], y=[p.balance_usd for p in lows],
            mode="markers", name="Lowest open-position equity in a trade",
            marker={"symbol": "triangle-down", "size": 9, "color": "#6a1b9a"},
            hovertemplate="Lowest open equity $%{y:,.2f}<br>"
            "%{x|%B %-d, %Y %-I:%M %p} Chicago<extra></extra>",
        ))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f",
                     title_text="US dollars relative to the $50,000 start")
    fig.update_xaxes(title_text="Date (Chicago time)")
    fig.update_layout(height=360, margin={"t": 30, "b": 40, "l": 70, "r": 20},
                      legend={"orientation": "h", "y": 1.15, "x": 0})
    return fig


# ── screen ────────────────────────────────────────────────────────────────


def _md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown never renders money as math."""

    return str(text).replace("$", "\\$")


def _status(st, line) -> None:
    {"info": st.info, "warning": st.warning, "error": st.error,
     "success": st.success}.get(line.level, st.info)(_md(line.text))


def _clear_stale(st, key: str, options: list[Any]) -> None:
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def render_funded_results(st, result: dict[str, Any]) -> None:
    """Render the completed funded-payout result (read-only; launches nothing)."""

    view = present_funded_result(result)
    st.subheader(view.headline)
    if view.question:
        st.markdown(_md(f"**Question.** {view.question}"))
    st.markdown(f"**Market period.** {view.period_text}")
    st.markdown(_md(f"**Strategy.** {view.strategy_text}"))
    if view.settings_text:
        st.caption(_md(f"Settings: {view.settings_text}."))
    for line in view.status:
        _status(st, line)

    if not view.firms:
        st.warning("This result contains no firm simulations to show.")
        _limitations(st, view)
        return

    columns = st.columns(len(view.firms))
    for column, firm in zip(columns, view.firms, strict=True):
        column.markdown(f"### {firm.firm}")
        for card in firm.cards:
            column.metric(card.label, card.value, help=_md(card.note))
        for notice in firm.notices:
            column.warning(_md(notice))

    _limitations(st, view)

    st.markdown("#### Comparison of the two separate operations")
    names = [f.firm for f in view.firms]
    first, second = (names + ["", ""])[:2]
    st.dataframe(
        [
            {"Measure": row.measure,
             **{name: value for name, value in row.values},
             f"Difference ({first} minus {second})": row.difference}
            for row in view.comparison
        ],
        hide_index=True, width="stretch",
    )
    st.caption(_md(view.comparison_note))

    st.markdown("#### Cash over time")
    st.caption("Each firm is drawn separately; nothing is added across firms. Pending "
               "payouts appear only when they are actually received.")
    _plotly(st, cash_over_time_figure(view), "cash_over_time")

    st.markdown("#### Month by month")
    st.caption("Every month of the period is shown, including months with no payouts and "
               "the partial first and last months.")
    _plotly(st, monthly_figure(view), "monthly")

    tabs = st.tabs([f"{firm.firm} — details" for firm in view.firms])
    for tab, firm in zip(tabs, view.firms, strict=True):
        with tab:
            _firm_details(st, view, firm)

    if view.decisions:
        st.markdown("#### Owner decisions used by this run")
        st.dataframe(
            [{"Decided": d.decided_on, "Subject": d.subject, "Decision": d.decision,
              "Status": d.status} for d in view.decisions],
            hide_index=True, width="stretch",
        )


def _limitations(st, view: FundedResultView) -> None:
    st.markdown("#### What this result cannot tell you")
    st.markdown(_md("\n".join(f"- {item}" for item in view.limitations)))


def _firm_details(st, view: FundedResultView, firm: FirmView) -> None:
    key = firm.firm_key
    st.markdown(f"##### {firm.firm}: money, accounts and capacity")
    st.dataframe(
        [{"Item": f.label, "Value": f.value, "What it means": f.note} for f in firm.facts],
        hide_index=True, width="stretch",
    )
    st.markdown(f"##### {firm.firm}: time without trading, by reason")
    st.dataframe(
        [{"Reason": w.reason, "Time (summed over accounts)": w.hours_text,
          "Signals affected": w.count_text} for w in firm.waiting],
        hide_index=True, width="stretch",
    )
    st.caption("A payout pause is a deliberate policy, not a strategy drought.")

    st.markdown(f"##### {firm.firm}: account journeys")
    if not firm.accounts:
        st.info("No accounts were bought by this firm in this period.")
        return
    _plotly(st, journeys_figure(view, firm), f"journeys_{key}")
    st.dataframe(
        [{"Account": a.label, "Status at the end": a.status, "Started": a.created,
          "How it was paid for": a.funding, "Replaces": a.replaces,
          "Replaced by": a.replaced_by, "Trades": a.trades,
          "Payouts received": a.payouts_received, "Received after split": a.received,
          "Largest payout": a.largest_payout, "Lost": a.failed,
          "Why it was lost": a.failure_reason}
         for a in firm.accounts],
        hide_index=True, width="stretch",
    )

    account_key = f"{KEY_PREFIX}account_{key}"
    ids = [a.account_id for a in firm.accounts]
    labels = {a.account_id: f"{a.label} — {a.status}" for a in firm.accounts}
    _clear_stale(st, account_key, ids)
    selected_id = st.selectbox(f"Choose a {firm.firm} account", ids,
                               format_func=labels.get, key=account_key)
    account = next(a for a in firm.accounts if a.account_id == selected_id)
    _account_detail(st, firm, account)


def _account_detail(st, firm: FirmView, account: AccountRow) -> None:
    key = firm.firm_key
    st.markdown(_md(f"**{account.label}** · {account.status} · started {account.created} · "
                    f"{account.funding}"))
    if account.lost:
        paid = (" It had already paid out " + account.received + " after the firm's share, "
                "which it keeps.") if account.payouts_received else (
            " It was lost before any payout.")
        st.error(_md(f"Lost {account.failed}. Reason: {account.failure_reason}.{paid}"))
    st.caption(_md(f"Final balance {account.final_balance} relative to the $50,000 start; "
                   f"final loss limit {account.final_floor}. Gross requested "
                   f"{account.gross_requested}."))
    if account.path:
        _plotly(st, account_path_figure(account), f"path_{key}")
    if account.payouts:
        st.dataframe(
            [{"Payout state": p.state, "Secured": p.secured, "Requested": p.requested,
              "Paid or due": p.paid_or_due, "Gross withdrawal": p.gross,
              "Firm's share": p.firm_share, "After the firm's share": p.after_split}
             for p in account.payouts],
            hide_index=True, width="stretch",
        )
    else:
        st.info("This account received no payouts.")

    trades = [t for t in firm.trades if t.account_id == account.account_id]
    if not trades:
        st.info("This account took no trades.")
        return
    st.dataframe(
        [{"Trade": t.number, "Entry (Chicago)": t.entry_time, "Exit (Chicago)": t.exit_time,
          "Direction": t.direction, "Size": t.quantity, "Entry price": t.entry_price,
          "Exit price": t.exit_price, "Result after costs": t.net_result,
          "Close reason": t.close_reason}
         for t in trades],
        hide_index=True, width="stretch",
    )
    trade_key = f"{KEY_PREFIX}trade_{key}"
    options = list(range(len(trades)))
    stale = st.session_state.get(f"{trade_key}_account")
    if stale != account.account_id:
        st.session_state.pop(trade_key, None)
        st.session_state[f"{trade_key}_account"] = account.account_id
    _clear_stale(st, trade_key, options)
    index = st.selectbox("Choose a trade", options,
                         format_func=lambda i: trades[i].label, key=trade_key)
    trade = trades[index]
    st.markdown(f"**Trade {trade.number}** on {trade.trading_day}")
    st.dataframe(
        [
            {"Item": "Direction and size", "Value": f"{trade.direction}, {trade.quantity}"},
            {"Item": "Entry (Chicago)", "Value": trade.entry_time},
            {"Item": "Exit (Chicago)", "Value": trade.exit_time},
            {"Item": "Entry price (points)", "Value": trade.entry_price},
            {"Item": "Protective stop (points)", "Value": trade.stop_price},
            {"Item": "Exit price (points)", "Value": trade.exit_price},
            {"Item": "Points moved in the trade's favor", "Value": trade.points_moved},
            {"Item": "Result before costs", "Value": trade.gross_result},
            {"Item": "Trading costs", "Value": trade.costs},
            {"Item": "Result after costs", "Value": trade.net_result},
            {"Item": "Initial risk (entry to stop)", "Value": trade.initial_risk},
            {"Item": "Lowest open-position equity", "Value": trade.lowest_open_equity},
            {"Item": "Close reason", "Value": trade.close_reason},
            {"Item": "Price evidence", "Value": trade.price_evidence},
        ],
        hide_index=True, width="stretch",
    )
    if trade.approximate:
        st.warning("This trade was checked against one-minute candles. The order of prices "
                   "inside each minute is assumed, so its loss-limit outcome is an "
                   "approximation.")
