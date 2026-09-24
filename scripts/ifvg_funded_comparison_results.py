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


CONTEXT_KEY = KEY_PREFIX + "selected_context"


def selected_context(st, result_id: str) -> dict[str, Any]:
    """The one selected firm / configuration / account for this saved result.

    Kept per result identity in the browser session (not in widget state, which
    Streamlit discards when the page is left), so Back and navigation to Trade
    review and back restore it, and another result never inherits it.
    """

    contexts = st.session_state.setdefault(CONTEXT_KEY, {})
    return contexts.setdefault(result_id, {})


def _comparison(st, view: ComparisonView, context: dict[str, Any], scope: str) -> None:
    st.markdown("#### All configurations, ranked by net cash earned")
    st.caption("Net cash earned = payouts received after the firm's share minus the cost of "
               "every funded account purchased. Each firm is a separate comparison; "
               "configurations and firms are never added together.")
    if not view.firm_tables:
        st.info("This result contains no firm comparisons.")
        return
    firms = {table.firm_key: table for table in view.firm_tables}
    keys = list(firms)
    saved = context.get("firm_key")
    firm_key = st.radio(
        "Firm (the ranking and the detail below both show this firm)", keys,
        index=keys.index(saved) if saved in keys else 0, horizontal=True,
        format_func=lambda k: firms[k].firm, key=f"{KEY_PREFIX}{scope}_firm")
    context["firm_key"] = firm_key
    table = firms[firm_key]
    _table(st, [row.display() for row in table.rows], f"{scope}_table_{table.firm_key}",
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


def _accounts(st, detail: PairDetail, key: str, context: dict[str, Any]) -> int | None:
    """Render the pair's accounts; the account shown is returned and remembered."""

    st.markdown("#### Account replacement history")
    _table(st, [a.display() for a in detail.accounts], f"accounts_{key}",
           "No accounts were purchased.")
    if not detail.accounts:
        return None
    labels = [a.label for a in detail.accounts]
    numbers = [a.number for a in detail.accounts]
    pair = f"{detail.configuration}|{detail.firm_key}"
    remembered = context.get("account") or {}
    # an account is restored only for its own configuration and firm, never carried over
    index = (numbers.index(remembered["number"])
             if remembered.get("pair") == pair and remembered.get("number") in numbers else 0)
    # keyed by result, configuration and firm: an account of another pair never carries over
    chosen = st.selectbox(f"Show one account ({detail.firm})", labels, index=index,
                          key=f"{KEY_PREFIX}account_{key}")
    account = detail.accounts[labels.index(chosen)]
    context["account"] = {"pair": pair, "number": account.number}
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
    return account.number


def _open_trade_review(target: dict[str, Any]) -> None:
    """Open Trade review on this exact result, configuration, firm and account."""

    import streamlit

    streamlit.session_state["ifvg_funded_review_pending"] = dict(target)
    streamlit.session_state["ifvg_study_v1_open_review"] = True


def _detail(st, result: dict[str, Any], view: ComparisonView, context: dict[str, Any],
            scope: str, result_id: str) -> None:
    st.markdown("#### Detail for one configuration and firm")
    configurations = completed_configurations(result)
    if not configurations or not view.firm_tables:
        st.warning("No configuration completed, so there is no detail to show.")
        return
    keys = [key for key, _ in configurations]
    labels = dict(configurations)
    firms = {t.firm_key: t for t in view.firm_tables}
    firm_key = context.get("firm_key") if context.get("firm_key") in firms else next(iter(firms))
    firm = firms[firm_key]
    saved = context.get("configuration")
    if saved in labels:
        index = keys.index(saved)
    else:
        # nothing chosen yet for this result: open on this firm's top-ranked configuration
        leader = next((r.configuration for r in firm.rows
                       if r.completed and r.configuration in labels), keys[0])
        index = keys.index(leader)
        st.caption(_md(f"Opened on {firm.firm}'s top-ranked configuration. Choosing the other "
                       "firm above keeps the configuration chosen here."))
    configuration = st.selectbox("Configuration", keys, format_func=lambda k: labels[k],
                                 index=index, key=f"{KEY_PREFIX}{scope}_configuration")
    context["configuration"] = configuration
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
    _table(st, [m.display() for m in detail.months], f"{scope}_months",
           "No months in the period.")

    st.markdown("#### Cumulative payouts, account costs and net cash")
    _cash_chart(st, detail)

    account_number = _accounts(st, detail, f"{scope}_{configuration}_{firm_key}", context)

    st.markdown("#### Where the time went")
    st.caption("Account time summed over this configuration's accounts. Payout protection and "
               "processing are deliberate pauses, not a lack of strategy signals.")
    _table(st, [{"Reason": t.reason, "Time": t.hours_text, "Note": t.note}
                for t in detail.time_split], f"{scope}_time", "")
    _table(st, [{"Refused entries": f.label, "Count": f.value} for f in detail.refused],
           f"{scope}_refused", "")

    st.markdown("#### Trades")
    st.caption("Prices are index points. Results are after trading costs.")
    _table(st, [t.display() for t in detail.trades], f"{scope}_trades",
           "No trades were taken in this period.")
    plan_id = result.get("funded_comparison_plan_id")
    if detail.trades and plan_id:
        st.button(f"Review these trades in Trade review ({detail.firm})",
                  key=f"{KEY_PREFIX}{scope}_open_review", on_click=_open_trade_review,
                  args=({"plan_id": plan_id, "result_id": result_id,
                         "configuration": configuration, "firm_key": firm_key,
                         "account_number": account_number},),
                  help="Opens the recorded trades of this configuration and firm, on the "
                       "account shown above, on the strategy's original bars, each with its "
                       "own review.")


def render_funded_comparison_results(st, result: dict[str, Any], *,
                                     result_id: str | None = None) -> None:
    """Render the completed configuration comparison (read-only).

    ``result_id`` is the saved result's identity; widget keys and the selected
    firm / configuration context are scoped to it.
    """

    result_id = result_id or str(result.get("funded_comparison_plan_id") or "result")
    scope = result_id[:16]
    context = selected_context(st, result_id)
    view = present_comparison(result)
    _header(st, view)
    _comparison(st, view, context, scope)
    metrics = present_strategy_metrics(result)
    if metrics:
        st.markdown("#### Strategy measures without accounts")
        st.caption(STRATEGY_METRICS_NOTE)
        st.dataframe(pd.DataFrame(metrics), hide_index=True, use_container_width=True,
                     key=f"funded_comparison_v1_{scope}_strategy_metrics")
    _detail(st, result, view, context, scope, result_id)
    st.markdown("#### Material limitations")
    for item in view.limitations:
        st.markdown(_md(f"- {item}"))
    st.caption(view.future_note)
    if view.decisions:
        with st.expander("Owner decisions and assumptions used"):
            _table(st, [{"Decided": d.decided_on, "Subject": d.subject,
                         "Decision": d.decision, "Status": d.status}
                        for d in view.decisions], f"{scope}_decisions", "")
