"""Funded comparison trades inside the existing Trade review (repair R3).

Reached from Trade review → Study executions (a completed funded comparison in
the Study selector) or from a funded result's "Review these trades" action.
It shows one recorded funded trade exactly as the saved result recorded it —
run, configuration, firm, account and trade identity — with the strategy's
original bars and, where the strategy saved them, the gap zones. The chart and
review form are the existing ones; judgments go to the existing ledger under
keys that belong to this result, pair, account and trade only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ifvg_search_review import (
    _bars,
    _evidence,
    add_gap_zones,
    candle_figure,
    finish_figure,
    mark_time,
)

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import (
    chicago_label,
    chicago_wall,
    utc_instant,
)
from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
    completed_configurations,
)
from alpha_lab.agents.data_infra.ifvg.presentation.funded_trade_review import (
    funded_trade_facts,
    pair_trades,
    plan_strategy_package,
    review_keys,
    strategy_trade_links,
    trade_option_label,
)
from alpha_lab.agents.data_infra.ifvg.visual_review_store import append_review, list_reviews

__all__ = ["render_funded_trade_review"]

_KEY = "ifvg_funded_review_"
_TIMEFRAME_NAMES = {60: "1-minute bars", 3600: "1-hour bars", 14400: "4-hour bars"}


@st.cache_resource(show_spinner="Verifying the saved funded result…")
def _result(store_root: str, result_id: str):
    from alpha_lab.propsim.funded.comparison_runner import load_comparison_result

    return load_comparison_result(Path(store_root), result_id)


@st.cache_resource(show_spinner="Verifying the strategy study this comparison used…")
def _strategy_source(store_root: str, plan_id: str):
    """The plan's bound strategy package (run id and manifest hash must match)."""

    from alpha_lab.propsim.funded.comparison_runner import load_plan
    from alpha_lab.propsim.funded.comparison_source import open_comparison_source
    from alpha_lab.propsim.funded.sources import ARCHIVE_ROOT

    plan = load_plan(Path(store_root), plan_id)
    package = plan_strategy_package(plan, ARCHIVE_ROOT)
    return None if package is None else open_comparison_source(package)


def _firms(result) -> dict[str, str]:
    order: dict[str, str] = {}
    for profile in (result.get("settings") or {}).get("firm_profiles") or []:
        order.setdefault(profile["firm_key"], profile.get("firm_name") or profile["firm_key"])
    for summary in (result.get("summaries_cents") or {}).values():
        order.setdefault(summary["firm_key"], summary.get("firm") or summary["firm_key"])
    return order


def _back_to_result(plan_id: str, result_id: str, configuration: str, firm_key: str,
                    account_key: str) -> None:
    """Return to this result with the same configuration, firm and account selected."""

    state = st.session_state
    state["ifvg_workspace_destination"] = "My studies"
    state["ifvg_workspace_screen"] = "detail"
    state["ifvg_workspace_selected_study"] = plan_id
    contexts = state.setdefault("funded_comparison_v1_selected_context", {})
    context = contexts.setdefault(result_id, {})
    context.update(configuration=configuration, firm_key=firm_key)
    account = state.get(account_key)
    if isinstance(account, int):
        context["account"] = {"pair": f"{configuration}|{firm_key}", "number": account}


def render_funded_trade_review(st_module, roots, study) -> None:
    from alpha_lab.propsim.funded.comparison_runner import is_v2, load_plan

    store = Path(roots["store_root"])
    plan_id, result_id = study.key, study.state["result_id"]
    try:
        result = _result(str(store), result_id)
        plan = load_plan(store, plan_id)
    except Exception:
        st_module.error("This funded comparison's saved result failed verification and is "
                        "not shown.")
        return
    if result.get("funded_comparison_plan_id") != plan_id:
        st_module.error("The saved result does not belong to this comparison plan; it is not "
                        "shown.")
        return
    scope = result_id[:16]
    status = getattr(study, "status", "Completed")
    if status == "Failed":
        st_module.warning("This saved result failed its internal money checks; its trades are "
                          "shown for review only.")
    elif status == "Incomplete":
        st_module.info("Some configurations of this comparison did not complete; only the "
                       "completed ones have trades here.")
    configurations = completed_configurations(result)
    if not configurations:
        st_module.info("No configuration completed in this funded comparison, so it has no "
                       "trades to review.")
        return
    keys, labels = [k for k, _ in configurations], dict(configurations)
    firms = _firms(result)
    config_key, firm_key_key = f"{_KEY}{scope}_configuration", f"{_KEY}{scope}_firm"
    target = st_module.session_state.pop(_KEY + "target", None) or {}
    if target.get("result_id") == result_id:
        if target.get("configuration") in labels:
            st_module.session_state[config_key] = target["configuration"]
        if target.get("firm_key") in firms:
            st_module.session_state[firm_key_key] = target["firm_key"]
    if st_module.session_state.get(config_key) not in labels:
        st_module.session_state.pop(config_key, None)
    configuration = st_module.selectbox("Configuration", keys, format_func=labels.get,
                                        key=config_key)
    firm_key = st_module.radio("Firm", list(firms), format_func=firms.get, horizontal=True,
                               key=firm_key_key)
    pair = f"{configuration}|{firm_key}"
    account_key = f"{_KEY}{scope}_{pair}_account"
    st_module.button("← Back to this funded result", key=f"{_KEY}{scope}_back",
                     on_click=_back_to_result,
                     args=(plan_id, result_id, configuration, firm_key, account_key))
    summary = next((s for s in (result.get("summaries_cents") or {}).values()
                    if s.get("configuration") == configuration
                    and s.get("firm_key") == firm_key), None)
    if summary is None or summary.get("status") != "Completed":
        st_module.warning("This configuration did not complete with this firm, so it has no "
                          "recorded trades. It is not a zero result.")
        return
    trades = pair_trades(result, configuration, firm_key)
    st_module.caption(f"{len(trades):,} recorded funded trades · {firms[firm_key]} · one live "
                      "account at a time · prices and times as the saved result recorded them")
    if not trades:
        st_module.info("No trades were taken by this configuration with this firm.")
        return
    accounts = sorted({int(t["account_number"]) for t in trades})
    if (target.get("result_id") == result_id and target.get("configuration") == configuration
            and target.get("firm_key") == firm_key
            and target.get("account_number") in accounts):
        # the account shown on the results page (never one of another firm/configuration)
        st_module.session_state[account_key] = target["account_number"]
    account = st_module.selectbox(
        "Account", ["all", *accounts], key=account_key,
        format_func=lambda a: "All accounts" if a == "all" else f"Account {a}")
    shown = [(i, t) for i, t in enumerate(trades, start=1)
             if account == "all" or int(t["account_number"]) == account]
    options = {int(t["seq"]): trade_option_label(i, t) for i, t in shown}
    trade_key = f"{_KEY}{scope}_{pair}_trade"
    if st_module.session_state.get(trade_key) not in options:
        st_module.session_state.pop(trade_key, None)
    seq = st_module.selectbox("Trade", list(options), format_func=options.get, key=trade_key)
    trade = next(t for t in trades if int(t["seq"]) == seq)

    variant = next((v for v in getattr(plan, "variants", ()) or () if v.name == configuration),
                   None)
    instrument = variant.instrument if variant is not None else getattr(plan, "instrument", None)
    facts = funded_trade_facts(result, trade, instrument=instrument)
    cols = st_module.columns(4)
    cols[0].metric("Entry", f"{int(trade['entry_ticks']) * 0.25:,.2f}")
    cols[1].metric("Initial stop", f"{int(trade['stop_ticks']) * 0.25:,.2f}")
    cols[2].metric("Target", f"{int(trade['target_ticks']) * 0.25:,.2f}"
                   if trade.get("target_ticks") is not None else "None")
    net = float(trade.get("net_pnl_usd") or 0)
    cols[3].metric("Result after costs", f"{'-' if net < 0 else '+'}${abs(net):,.2f}")
    # table cells are rendered as Markdown: escape "$" so amounts are not read as math
    st_module.table(pd.DataFrame([(name, value.replace("$", r"\$")) for name, value in
                                  facts.rows], columns=["Recorded fact", "Value"])
                    .set_index("Recorded fact"))
    for notice in facts.notices:
        st_module.caption(notice)
    if is_v2(plan) and variant is not None and variant.exit_policy != "fixed_target_v1":
        st_module.caption("The accounts follow the recorded exchange trades in order, so a half "
                          "exit and a break-even stop can both happen inside one minute.")
    _chart(st_module, roots, store, plan_id, plan, configuration, trade, facts)
    _review(st_module, roots, store, plan_id, result_id, trade)


def _chart(st_module, roots, store, plan_id, plan, configuration, trade, facts) -> None:
    source = None
    try:
        source = _strategy_source(str(store), plan_id)
    except Exception:
        source = None
    if source is None:
        st_module.warning("The strategy study this comparison used is not available or failed "
                          "verification, so the chart is unavailable. The recorded trade facts "
                          "above come from the saved result.")
        return
    geometry_core, bars_core = strategy_trade_links(plan, source, configuration)
    evidence_root = source.package.root.parent / "store"
    day = str(trade["trading_day"])
    try:
        bars_evidence = _evidence(str(evidence_root), bars_core)
        bars = _bars(str(roots["repo_root"]), bars_core, day, bars_evidence)
    except Exception:
        st_module.warning("The original bars for this trading day could not be verified, so the "
                          "chart is unavailable. The recorded trade facts above are shown.")
        return
    geometry = None
    if geometry_core is None:
        st_module.caption("Gap zones are not shown: this configuration was not part of the "
                          "verified strategy study, so no strategy record of its gaps exists.")
    else:
        try:
            executed = _evidence(str(evidence_root), geometry_core).dataset.tables[
                RecordTable.EXECUTED_TRADE]
        except Exception:
            st_module.caption("Gap zones are not shown: the strategy record for this trade "
                              "could not be verified.")
        else:
            match = executed.loc[executed.trade_id == str(trade["strategy_trade_id"])]
            geometry = None if match.empty else match.iloc[0]
            if geometry is None:
                st_module.caption("Gap zones are not shown: the verified strategy study has no "
                                  "record of this trade's identity. A funded trade can differ "
                                  "from the strategy study when an account rule changed the "
                                  "strategy's path.")
    available = set(int(v) for v in bars.timeframe_ticks.unique())
    timeframes = [60]
    if geometry is not None:
        timeframes += [int(geometry.geometry_parent_timeframe_seconds),
                       int(geometry.geometry_htf_timeframe_seconds)]
    else:
        timeframes += [tf for tf in (3600, 14400) if tf in available]
    timeframes = sorted(set(tf for tf in timeframes if tf in available))
    if st_module.session_state.get(f"{_KEY}timeframe") not in timeframes:
        st_module.session_state.pop(f"{_KEY}timeframe", None)
    timeframe = st_module.selectbox(
        "Chart timeframe", timeframes, key=f"{_KEY}timeframe",
        format_func=lambda v: _TIMEFRAME_NAMES.get(v, f"{v // 60}-minute bars"))
    entry, exit_ = utc_instant(trade["entry_utc"]), utc_instant(trade["exit_utc"])
    frame = bars.loc[bars.timeframe_ticks == timeframe].copy()
    if timeframe == 60:
        frame = frame.loc[frame.logical_close_ts_utc.between(
            entry - pd.Timedelta(minutes=60), exit_ + pd.Timedelta(minutes=20))]
    figure, x = candle_figure(frame)
    start, end = chicago_wall(entry), chicago_wall(exit_)
    if facts.entry_ticks is not None:
        _segment(figure, "Entry price", start, end, facts.entry_ticks, "#2563eb", at_start=True)
    if facts.target_ticks is not None:
        _segment(figure, "Target", start, end, facts.target_ticks, "#15803d")
    for stop in facts.stops:
        _segment(figure, stop.label, chicago_wall(stop.start), chicago_wall(stop.end),
                 stop.price_ticks, "#dc2626")
    heights = iter((1.10, 1.05, 1.0, 0.95))
    points = []
    for marker in facts.markers:
        at = mark_time(figure, marker.label, marker.ts_utc, next(heights, 0.9))
        if marker.price_ticks is not None:
            points.append((at, int(marker.price_ticks) * 0.25, marker.label,
                           chicago_label(marker.ts_utc, seconds=True)))
    if points:
        figure.add_trace(go.Scatter(
            x=[p[0] for p in points], y=[p[1] for p in points], mode="markers",
            marker=dict(size=10, color="#0f172a", symbol="diamond"),
            text=[f"{p[2]}: {p[1]:,.2f} at {p[3]}" for p in points], hoverinfo="text",
            name="Recorded fills"))
    if geometry is not None and st_module.checkbox("Show gap zones", value=True,
                                                   key=f"{_KEY}gaps"):
        add_gap_zones(figure, geometry, timeframe, x.max())
    finish_figure(figure, "Recorded funded trade and the strategy's original bars")
    st_module.plotly_chart(figure, width="stretch", key="ifvg_funded_trade_chart", theme=None)
    st_module.caption("Markers sit at the recorded fill instants. Candles are the strategy's "
                      "original bars; movement inside a minute is not drawn, and prices after "
                      "the exit are context only, not profit captured.")


def _segment(figure, label, x0, x1, ticks, color, *, at_start=False) -> None:
    figure.add_shape(type="line", x0=x0, x1=x1, y0=ticks * 0.25, y1=ticks * 0.25,
                     line=dict(color=color, dash="dot"))
    # the entry label sits at the start, the others at the end, so a stop moved to the
    # entry price never prints over the entry label
    figure.add_annotation(x=x0 if at_start else x1, y=ticks * 0.25, text=label,
                          showarrow=False, xanchor="right" if at_start else "left",
                          font=dict(size=11, color=color))


def _review(st_module, roots, store, plan_id, result_id, trade) -> None:
    from ifvg_verifier_tab import _CANDIDATE_DETAIL_VERDICTS, _review_form

    chart_key, case_key, pair_ref = review_keys(result_id, plan_id, trade)

    def save(**values):
        from alpha_lab.propsim.funded.comparison_runner import load_comparison_result

        verified = load_comparison_result(store, result_id)  # hash re-checked
        recorded = [t for t in (verified.get("tables") or {}).get("trades") or []
                    if t.get("pair_id") == trade["pair_id"] and t.get("seq") == trade["seq"]]
        if len(recorded) != 1 or recorded[0].get("strategy_trade_id") != trade.get(
                "strategy_trade_id"):
            raise ValueError("the saved funded trade could not be verified again")
        return append_review(
            repo_root=Path(roots["repo_root"]), replay_chart_artifact_id=chart_key,
            pair_ref=pair_ref, candidate_id=case_key, decision_id=None,
            trade_id=str(trade.get("strategy_trade_id")), **values)

    with st_module.expander("Review", expanded=True):
        st_module.caption("Funded-trade reviews are kept for this result, firm, account and "
                          "trade only; they never change the strategy study's reviews.")
        _review_form(
            st_module, prefix="funded_review", case_key=case_key,
            existing=list_reviews(repo_root=Path(roots["repo_root"]), candidate_id=case_key,
                                  replay_chart_artifact_id=chart_key),
            detail_fields=_CANDIDATE_DETAIL_VERDICTS, on_save=save)
