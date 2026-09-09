"""Trade review for exact saved search children, using their original bars."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import configuration_name, load_studies
from alpha_lab.agents.data_infra.ifvg.search.review_evidence import (
    load_search_day_bars,
    load_search_review_evidence,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import list_search_runs, load_search_state
from alpha_lab.agents.data_infra.ifvg.visual_review_store import append_review, list_reviews


@st.cache_resource(show_spinner="Verifying the saved execution evidence…")
def _evidence(root: str, core_id: str):
    return load_search_review_evidence(Path(root), core_id)


@st.cache_data(show_spinner=False)
def _bars(repo: str, core_id: str, day: str, _evidence):
    return load_search_day_bars(Path(repo), _evidence, day)


def render_trade_review(st_module, roots):
    """Keep the existing context reviewer, and offer exact search executions."""
    pending = st_module.session_state.pop("ifvg_search_review_pending", None)
    if pending:
        st_module.session_state["ifvg_review_source"] = "Study executions"
        st_module.session_state["ifvg_search_review_search"] = pending[0]
        st_module.session_state["ifvg_search_review_core"] = pending[1]
    source = st_module.radio(
        "Review source",
        ("Verified context", "Study executions"),
        horizontal=True,
        key="ifvg_review_source",
    )
    if source == "Verified context":
        from ifvg_lab_tab import render_ifvg_replay_tab

        render_ifvg_replay_tab(st_module)
        return
    st_module.header("Study trade review")
    runs = list_search_runs(Path(roots["state_root"]), Path(roots["store_root"]))
    runs = [run for run in runs if not run.archived]
    if not runs:
        st_module.info("No saved searches are available for review.")
        return
    studies, _issues = load_studies(roots)
    saved_names = {study.key: study.name for study in studies}
    names = {
        run.search_id: saved_names.get(run.search_id, f"Saved study {index + 1}")
        for index, run in enumerate(runs)
    }
    search = st_module.selectbox(
        "Study", list(names), format_func=names.get, key="ifvg_search_review_search"
    )
    state = load_search_state(Path(roots["state_root"]), search)
    children = [
        c
        for c in (state or {}).get("children", ())
        if c.get("core_replay_id") and c["state"] in ("completed", "reused")
    ]
    labels = {c["core_replay_id"]: configuration_name(c["axis_value_ids"]) for c in children}
    if not labels:
        st_module.info("This study has no completed execution evidence.")
        return
    core_key = "ifvg_search_review_core"
    if st_module.session_state.get(core_key) not in labels:
        st_module.session_state[core_key] = next(iter(labels))
    core_id = st_module.selectbox(
        "Configuration", list(labels), format_func=labels.get, key=core_key
    )
    try:
        evidence = _evidence(str(roots["store_root"]), core_id)
    except Exception:
        st_module.error("This configuration's saved evidence failed verification.")
        return
    trades = evidence.dataset.tables[RecordTable.EXECUTED_TRADE].sort_values("entry_ts_utc")
    show_warmup = st_module.checkbox("Include warmup trades", key="ifvg_search_review_warmup")
    if not show_warmup:
        trades = trades.loc[~trades.is_warmup]
    if trades.empty:
        st_module.info("No executions in this selection.")
        return
    st_module.caption(
        f"{len(trades)} executed trades · Original saved prices and bars · Research review"
    )
    trade_names = {
        row.trade_id: (
            f"{i + 1}. {row.entry_ts_utc.tz_convert('America/New_York'):%Y-%m-%d %H:%M} ET · "
            f"{row.resolution} · {row.entry_ticks * 0.25:.2f}"
        )
        for i, row in enumerate(trades.itertuples())
    }
    key = "ifvg_search_review_trade_" + core_id
    if st_module.session_state.get(key) not in trade_names:
        st_module.session_state[key] = next(iter(trade_names))
    selected = st_module.selectbox("Trade", list(trade_names), format_func=trade_names.get, key=key)
    row = trades.loc[trades.trade_id == selected].iloc[0]
    day = str(row.trading_day)[:10]
    try:
        bars = _bars(str(roots["repo_root"]), core_id, day, evidence)
    except Exception:
        st_module.error(
            "The original input bars could not be verified; chart and review are unavailable."
        )
        return
    if row.is_warmup:
        st_module.warning("Warmup execution: excluded from the corrected research metrics.")
    cols = st_module.columns(4)
    for col, label, value in zip(
        cols[:3],
        ("Entry", "Stop", "Target"),
        (row.entry_ticks, row.stop_ticks, row.target_ticks),
        strict=True,
    ):
        col.metric(label, f"{value * 0.25:.2f}")
    cols[3].metric("Gross outcome (R)", f"{row.realized_r:+.2f}")
    st_module.caption(
        "Outcome uses the saved next-bar, stop-first resolver. Transaction costs are applied "
        "in study results."
    )
    timeframes = sorted(
        {60, int(row.geometry_parent_timeframe_seconds), int(row.geometry_htf_timeframe_seconds)}
    )
    timeframe = st_module.selectbox(
        "Chart timeframe", timeframes, format_func=lambda value: f"{value // 60} minute bars"
    )
    frame = bars.loc[bars.timeframe_ticks == timeframe].copy()
    if timeframe == 60:
        frame = frame.loc[
            frame.logical_close_ts_utc.between(
                row.entry_ts_utc - pd.Timedelta(minutes=60),
                row.resolution_ts_utc + pd.Timedelta(minutes=20),
            )
        ]
    x = frame.logical_close_ts_utc.dt.tz_convert("America/New_York")
    figure = go.Figure(
        go.Candlestick(
            x=x,
            open=frame.open_ticks * 0.25,
            high=frame.high_ticks * 0.25,
            low=frame.low_ticks * 0.25,
            close=frame.close_ticks * 0.25,
            name="Original bars",
        )
    )
    for label, price, color in (
        ("Entry", row.entry_ticks, "#2563eb"),
        ("Stop", row.stop_ticks, "#dc2626"),
        ("Target", row.target_ticks, "#15803d"),
    ):
        figure.add_hline(y=price * 0.25, line_color=color, line_dash="dot", annotation_text=label)
    for label, ts in (
        ("Inversion", row.geometry_inversion_bar_logical_close_ts_utc),
        ("Entry", row.entry_ts_utc),
        ("Exit", row.resolution_ts_utc),
    ):
        figure.add_shape(
            type="line",
            x0=ts.tz_convert("America/New_York"),
            x1=ts.tz_convert("America/New_York"),
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#64748b", dash="dot"),
        )
        figure.add_annotation(
            x=ts.tz_convert("America/New_York"),
            y={"Inversion": 1.10, "Entry": 1.05, "Exit": 1.0}[label],
            yref="paper",
            text=label,
            showarrow=False,
        )
    if st_module.checkbox("Show gap zones", value=True):
        roles = (
            ("opposing", "entry_fvg")
            if timeframe == 60
            else ("parent",)
            if timeframe == int(row.geometry_parent_timeframe_seconds)
            else ("htf",)
        )
        for role in roles:
            prefix = "geometry_" + role
            low, high = row.get(prefix + "_gap_low_ticks"), row.get(prefix + "_gap_high_ticks")
            confirmed = row.get(prefix + "_confirmed_ts_utc")
            if pd.notna(low) and pd.notna(high) and pd.notna(confirmed):
                figure.add_shape(
                    type="rect",
                    x0=confirmed.tz_convert("America/New_York"),
                    x1=x.max(),
                    y0=low * 0.25,
                    y1=high * 0.25,
                    fillcolor="#f59e0b",
                    opacity=0.15,
                    line_width=0,
                )
    figure.update_layout(
        title="Saved execution and original bars",
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#17212B"),
        height=530,
        xaxis_rangeslider_visible=False,
        xaxis_title="New York time",
        yaxis_title="NQ price",
        margin=dict(l=35, r=20, t=60, b=40),
    )
    st_module.plotly_chart(figure, width="stretch", key="ifvg_search_trade_chart", theme=None)
    with st_module.expander("Execution and lifecycle details"):
        events = {
            "Tap": row.geometry_tap_bar_logical_close_ts_utc,
            "Parent confirmed": row.geometry_parent_confirmed_ts_utc,
            "Parent retest": row.geometry_lock_bar_logical_close_ts_utc,
            "Opposing gap": row.geometry_opposing_confirmed_ts_utc,
            "Inversion": row.geometry_inversion_bar_logical_close_ts_utc,
            "Entry": row.entry_ts_utc,
            "Exit": row.resolution_ts_utc,
        }
        st_module.dataframe(
            [
                {"Event": name, "Time (ET)": str(value.tz_convert("America/New_York"))}
                for name, value in events.items()
            ],
            hide_index=True,
            width="stretch",
        )
        st_module.caption(
            f"Risk: {row.risk_ticks * 0.25:g} points · "
            f"MFE: {row.mfe_ticks / row.risk_ticks:.3f} R · "
            f"MAE: {row.mae_ticks / row.risk_ticks:.3f} R. "
            "Excursions include the resolution bar and can exceed the realized 1R outcome."
        )
    with st_module.expander("Review", expanded=True):
        from ifvg_verifier_tab import _CANDIDATE_DETAIL_VERDICTS, _review_form

        def save(**values):
            verified = load_search_review_evidence(Path(roots["store_root"]), core_id)
            load_search_day_bars(Path(roots["repo_root"]), verified, day)
            return append_review(
                repo_root=Path(roots["repo_root"]),
                replay_chart_artifact_id="",
                pair_ref=verified.reference,
                candidate_id=str(row.candidate_id),
                decision_id=str(row.decision_id),
                trade_id=str(row.trade_id),
                **values,
            )

        _review_form(
            st_module,
            prefix="search_review",
            case_key=str(row.candidate_id),
            existing=list_reviews(
                repo_root=Path(roots["repo_root"]), candidate_id=str(row.candidate_id)
            ),
            detail_fields=_CANDIDATE_DETAIL_VERDICTS,
            on_save=save,
        )
