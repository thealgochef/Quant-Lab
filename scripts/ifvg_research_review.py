"""Human review fields and chart/table alternatives; exact joins stay in providers."""

from __future__ import annotations

import re

import pandas as pd

_IDENTIFIER = re.compile(r"\b[0-9a-f]{12,}\b(?:…)?", re.I)
_PATH = re.compile(r"[A-Za-z]:[\\/][^\s<]+|(?:scripts|src|data)/[^\s<]+")
_FVG_ID = re.compile(r"\b\d+s:\d{4}-\d{2}-\d{2}:\d+\b")
_PHASE = re.compile(r"(?:\(phase\s+)?\bS\d+[a-z]?\b\)?", re.I)


def setup_phase_name(value):
    return {
        "S0": "No active setup",
        "S1": "Waiting for parent retest",
        "S2": "Parent locked",
        "S3": "Opposing gap armed",
        "S4": "Inversion confirmed",
        "S5": "Trade entered",
    }.get(str(value), "Legacy stage " + str(value).removeprefix("S"))


def words(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    text = str(value)
    text = _IDENTIFIER.sub("", text)
    text = _PATH.sub("", text)
    text = _FVG_ID.sub("", text)
    text = _PHASE.sub(
        lambda match: setup_phase_name(re.search(r"S\d+[a-z]?", match[0], re.I)[0]), text
    )
    return text.replace("_", " ").strip(" ·")


def research_figure(figure, *, title):
    """Remove technical hover metadata from the presentation copy only."""
    import plotly.graph_objects as go

    result = go.Figure(figure)
    result.update_layout(
        title=title,
        meta=None,
        height=540,
        margin=dict(l=30, r=20, t=50, b=45),
        template="plotly_white",
        font=dict(color="#17212B"),
    )
    for trace in result.data:
        if trace.name:
            trace.name = words(trace.name)
        for field in ("text", "hovertext"):
            value = getattr(trace, field, None)
            if value is None:
                continue

            def clean(text):
                lines = str(text).split("<br>")
                return "<br>".join(
                    words(line)
                    for line in lines
                    if not any(
                        token in line.lower()
                        for token in (
                            "parent clocks",
                            "remaining window bars",
                            "open windows",
                            "source=",
                            "source:",
                            "{",
                            '["',
                        )
                    )
                )

            setattr(
                trace,
                field,
                clean(value) if isinstance(value, str) else [clean(item) for item in value],
            )
        trace.meta = None
        if hasattr(trace, "customdata"):
            trace.customdata = None
    for annotation in result.layout.annotations or ():
        annotation.text = words(annotation.text)
    return result


def bar_table(st, bars_by_pane, *, key):
    with st.expander("Chart data table"):
        choices = [value for value, frame in bars_by_pane.items() if not frame.empty]
        if not choices:
            st.info("No chart bars are available for the selected range.")
            return
        timeframe = st.selectbox(
            "Table timeframe",
            choices,
            format_func=lambda value: f"{int(value) // 60} minute bars",
            key=key,
        )
        frame = bars_by_pane[timeframe]
        allowed = [
            column
            for column in (
                "close_ts_utc",
                "ts_utc",
                "ts_event",
                "timestamp",
                "logical_close_ts",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_ticks",
                "high_ticks",
                "low_ticks",
                "close_ticks",
            )
            if column in frame
        ]
        table = frame[allowed].copy()
        if not any(
            column in allowed for column in ("ts_utc", "ts_event", "timestamp", "logical_close_ts")
        ) and isinstance(frame.index, pd.DatetimeIndex):
            table.insert(0, "Time", frame.index)
        table.columns = [words(column).capitalize() for column in table.columns]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download chart data",
            table.to_csv(index=False),
            file_name="trade-chart-data.csv",
            mime="text/csv",
            key=key + "_download",
        )


def candidate_panel(st, ctx, evidence, row):
    from ifvg_verifier_tab import _render_review_section

    st.subheader("Actual execution")
    execution = evidence.execution
    if execution is None:
        if row.get("blocked"):
            st.warning("No trade was executed. Blocking reason: " + words(row.get("block_reasons")))
        else:
            st.info("No actual execution is visible at the selected point in time.")
    else:
        cols = st.columns(3)
        cols[0].metric("Entry", f"{execution['entry_ticks'] * 0.25:.2f}")
        cols[1].metric("Stop", f"{execution['stop_ticks'] * 0.25:.2f}")
        if "realized_r" in execution:
            # Provider realized_r is the saved execution outcome; no invented
            # cost adjustment or hypothetical label is substituted here.
            cols[2].metric("Realized outcome (R)", f"{execution['realized_r']:+.2f}")
        else:
            cols[2].caption("Outcome withheld at this point in time")
    if st.checkbox("Show evidence details", key="ifvg_review_evidence_details"):
        with st.expander("Lifecycle"):
            rows = []
            order = ("tap", "parent", "lock", "opposing", "inversion", "entry", "resolution")
            for stage, gate in evidence.stage_gates.items():
                visible = evidence.mode != "point_in_time" or (
                    stage in order
                    and evidence.stage in order
                    and order.index(stage) <= order.index(evidence.stage)
                )
                rows.append(
                    {
                        "Event": words(stage),
                        "Time (UTC)": str(gate.ts_utc) if visible else "Withheld",
                    }
                )
            st.dataframe(rows, hide_index=True, width="stretch")
        if evidence.counterfactual_labels:
            with st.expander("Hypothetical outcome labels"):
                st.caption("These hypothetical labels are separate from actual execution.")
                frame = pd.DataFrame(list(evidence.counterfactual_labels))
                columns = [
                    key
                    for key in ("label_family", "label", "censored", "mfe_r", "mae_r")
                    if key in frame
                ]
                frame = frame[columns].rename(
                    columns={key: words(key).capitalize() for key in columns}
                )
                st.dataframe(frame, hide_index=True, width="stretch")
        if evidence.model:
            with st.expander("Model probabilities"):
                st.caption(
                    "Offline out-of-sample predictions for hypothetical outcomes. "
                    "These are separate from trade results and reviewer judgment."
                )
                rows = []
                for tier, payload in evidence.model.items():
                    model_row = {"Feature tier": words(tier)}
                    for name, value in payload.items():
                        if name in ("probability", "p_positive", "prediction", "status", "reason"):
                            model_row[words(name).capitalize()] = (
                                words(value) if isinstance(value, str) else value
                            )
                    rows.append(model_row)
                st.dataframe(rows, hide_index=True, width="stretch")
    _render_review_section(st, ctx, evidence, row)


def setup_panel(st, ctx, bundle, evidence, row):
    from ifvg_verifier_tab import _render_setup_review_section, _setup_candidate_ids

    st.subheader("Setup outcome")
    if evidence.stage != "terminal":
        st.caption("Later outcomes are withheld at the selected point in time.")
    else:
        st.write(words(row.get("terminal_reason")))
        if row.get("candidate_less"):
            st.info("This setup produced no entry opportunity and no trade.")
        else:
            st.caption(
                f"{int(row.get('candidate_count', 0))} entry opportunities. "
                "Open an opportunity to inspect its execution or blocking reason."
            )
    candidates = _setup_candidate_ids(row)
    if candidates and evidence.stage == "terminal":
        selected = st.selectbox(
            "Linked entry opportunity",
            candidates,
            format_func=lambda value: f"Entry opportunity {candidates.index(value) + 1}",
            key="ifvg_setup_linked_candidate",
        )
        if st.button("Review this opportunity"):
            st.session_state["ifvg_context_v1_pending_jump"] = ("candidate_id", selected)
            st.rerun()
    with st.expander("Lifecycle events"):
        allowed = [
            key
            for key in (
                "event_kind",
                "stage",
                "ts_utc",
                "selected",
                "drop_reason",
                "fill_kind",
                "fill_depth_ticks",
            )
            if key in evidence.events
        ]
        table = evidence.events[allowed].copy()
        for column in table.select_dtypes(include=["object", "string"]).columns:
            table[column] = table[column].map(words)
        table.columns = [words(column).capitalize() for column in table.columns]
        st.dataframe(table, hide_index=True, width="stretch")
    _render_setup_review_section(st, ctx, bundle, evidence)
