"""Completed-results screen for a funded configuration comparison.

``render_funded_comparison_results(st, result)`` shows the ONE saved comparison
result through the pure presenter
:mod:`alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison`. It
never recomputes money. Every configuration-and-firm pair is its own
comparison: firms are separate tabs and nothing is summed across configurations
or firms. Only standard Streamlit widgets and charts are used (keyboard
usable); matplotlib is used only by the review-folder export.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
    STRATEGY_METRICS_NOTE,
    ComparisonView,
    PairDetail,
    cash_chart_points,
    completed_configurations,
    present_comparison,
    present_pair_detail,
    present_strategy_metrics,
)
from alpha_lab.agents.data_infra.ifvg.presentation.funded_results import chicago_datetime

__all__ = ["render_funded_comparison_results"]

KEY_PREFIX = "funded_comparison_v1_"
_LEVELS = {"info", "warning", "error", "success"}


def _md(text: str) -> str:
    """Escape dollar signs so Streamlit markdown never reads them as math."""

    return str(text).replace("$", r"\$")


def _table(st, rows: list[dict[str, str]], key: str, empty: str) -> None:
    if not rows:
        st.caption(empty)
        return
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", key=KEY_PREFIX + key)


def _static_table(st, rows: list[dict[str, str]], empty: str) -> None:
    """Short key-value tables: every row visible and long text wrapped."""

    if not rows:
        if empty:
            st.caption(empty)
        return
    frame = pd.DataFrame(rows)
    st.table(frame.set_index(frame.columns[0]))


def _header(st, view: ComparisonView) -> None:
    st.subheader(view.title)
    st.markdown(_md(f"**Question:** {view.question}"))
    st.markdown(
        f"**Period:** {view.period_text}  \n"
        f"**Firms:** {view.firms_text}  \n"
        f"**Position size:** {view.size_text}  \n"
        f"**Configurations:** {view.configurations_text}")
    for line in view.status:
        getattr(st, line.level if line.level in _LEVELS else "info")(_md(line.text))
    st.caption(view.future_note)


def _comparison(st, view: ComparisonView) -> None:
    st.markdown("#### All configurations, ranked by net cash earned")
    st.caption("Net cash earned = payouts received after the firm's share minus the cost of "
               "every funded account purchased. Each firm is a separate comparison; "
               "configurations and firms are never added together.")
    if not view.firm_tables:
        st.info("This result contains no firm comparisons.")
        return
    tabs = st.tabs([table.firm for table in view.firm_tables])
    for tab, table in zip(tabs, view.firm_tables, strict=True):
        with tab:
            _table(st, [row.display() for row in table.rows], f"table_{table.firm_key}",
                   "No configurations were tested with this firm.")
            st.caption(_md(table.note))
            for row in table.rows:
                if not row.completed:
                    st.warning(_md(f"{row.label}: {row.reason}"))


def _cash_chart(st, detail: PairDetail) -> None:
    points = cash_chart_points(detail)
    if not points:
        st.caption("No cash movements were recorded.")
        return
    frame = pd.DataFrame(
        [{"Chicago time": when, "Payouts received": received, "Account costs": costs,
          "Net cash": net} for when, received, costs, net in points]).set_index("Chicago time")
    st.line_chart(frame, y_label="US dollars (cumulative)", x_label="Date (Chicago time)")


def _monthly_chart(st, detail: PairDetail) -> None:
    if not detail.months:
        return
    # categorical month names (a date axis would draw weekly ticks and sliver bars);
    # the "01 " prefix keeps calendar order when the chart sorts the labels
    frame = pd.DataFrame(
        [{"Month": f"{i:02d} {pd.Timestamp(f'{m.month}-01'):%B %Y}",
          "Payouts received": m.received_usd, "Account costs": m.account_costs_usd}
         for i, m in enumerate(detail.months, start=1)]).set_index("Month")
    st.bar_chart(frame, stack=False, y_label="US dollars", x_label="Month")


def _accounts(st, detail: PairDetail, key: str) -> None:
    st.markdown("#### Account replacement history")
    _table(st, [a.display() for a in detail.accounts], f"accounts_{key}",
           "No accounts were purchased.")
    if not detail.accounts:
        return
    labels = [a.label for a in detail.accounts]
    chosen = st.selectbox("Show one account", labels, key=f"{KEY_PREFIX}account_{key}")
    account = detail.accounts[labels.index(chosen)]
    st.markdown(_md(f"**{account.label}:** {account.payout_outcome.lower()}; final balance "
                    f"{account.final_balance} against a loss limit of "
                    f"{account.final_loss_limit}."))
    path = [p for p in account.balance_path
            if p.balance_usd is not None and p.loss_limit_usd is not None]
    if path:
        frame = pd.DataFrame(
            [{"Chicago time": chicago_datetime(p.ts_utc), "Account balance": p.balance_usd,
              "Loss limit": p.loss_limit_usd} for p in path]).set_index("Chicago time")
        st.line_chart(frame, y_label="US dollars", x_label="Date (Chicago time)")
    _table(st, [p.display() for p in account.payouts], f"payouts_{key}_{account.number}",
           "This account requested no payouts.")


def _detail(st, result: dict[str, Any], view: ComparisonView) -> None:
    st.markdown("#### Detail for one configuration and firm")
    configurations = completed_configurations(result)
    if not configurations:
        st.warning("No configuration completed, so there is no detail to show.")
        return
    keys = [key for key, _ in configurations]
    labels = dict(configurations)
    # open on the first firm's top-ranked configuration
    leader = next((r.configuration for t in view.firm_tables for r in t.rows
                   if r.completed and r.configuration in labels), keys[0])
    configuration = st.selectbox("Configuration", keys, format_func=lambda k: labels[k],
                                 index=keys.index(leader), key=f"{KEY_PREFIX}configuration")
    firms = {t.firm_key: t.firm for t in view.firm_tables}
    firm_key = st.radio("Firm", list(firms), format_func=lambda k: firms[k], horizontal=True,
                        key=f"{KEY_PREFIX}firm")
    detail = present_pair_detail(result, configuration, firm_key)
    st.markdown(_md(f"**{detail.label}** with **{detail.firm}**"
                    + (f" — rank {detail.rank_text} for this firm" if detail.completed
                       else "")))
    if not detail.completed:
        for notice in detail.notices:
            st.warning(_md(notice))
        return
    columns = st.columns(len(detail.headline))
    for column, fact in zip(columns, detail.headline, strict=True):
        column.metric(fact.label, fact.value, help=_md(fact.note) if fact.note else None)
    for notice in detail.notices:
        st.info(_md(notice))

    st.markdown("#### Configuration settings")
    _static_table(st, [{"Setting": name, "Value": value}
                       for name, value in detail.settings],
                  "No settings were saved for this configuration.")

    st.markdown("#### Payout timing and spending")
    _static_table(st, [{"Measure": f.label, "Value": f.value, "Note": f.note}
                       for f in detail.facts], "")

    st.markdown("#### Payouts by month")
    _monthly_chart(st, detail)
    _table(st, [m.display() for m in detail.months], "months", "No months in the period.")

    st.markdown("#### Cumulative payouts, account costs and net cash")
    _cash_chart(st, detail)

    _accounts(st, detail, f"{configuration}_{firm_key}")

    st.markdown("#### Where the time went")
    st.caption("Account time summed over this configuration's accounts. Payout protection and "
               "processing are deliberate pauses, not a lack of strategy signals.")
    _table(st, [{"Reason": t.reason, "Time": t.hours_text, "Note": t.note}
                for t in detail.time_split], "time", "")
    _table(st, [{"Refused entries": f.label, "Count": f.value} for f in detail.refused],
           "refused", "")

    st.markdown("#### Trades")
    st.caption("Prices are index points. Results are after trading costs.")
    _table(st, [t.display() for t in detail.trades], "trades",
           "No trades were taken in this period.")


def render_funded_comparison_results(st, result: dict[str, Any]) -> None:
    """Render the completed configuration comparison (read-only)."""

    view = present_comparison(result)
    _header(st, view)
    _comparison(st, view)
    metrics = present_strategy_metrics(result)
    if metrics:
        st.markdown("#### Strategy measures without accounts")
        st.caption(STRATEGY_METRICS_NOTE)
        st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True,
                     key="funded_comparison_v1_strategy_metrics")
    _detail(st, result, view)
    st.markdown("#### Material limitations")
    for item in view.limitations:
        st.markdown(_md(f"- {item}"))
    st.caption(view.future_note)
    if view.decisions:
        with st.expander("Owner decisions and assumptions used"):
            _table(st, [{"Decided": d.decided_on, "Subject": d.subject,
                         "Decision": d.decision, "Status": d.status}
                        for d in view.decisions], "decisions", "")
