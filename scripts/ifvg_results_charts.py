"""Pure Plotly builders for the study results surfaces (R4; FUX §§19–28).

Every builder is Streamlit-free, deterministic, and bounded: figures honor
explicit layer budgets and return an ``OmissionReport`` (reused from
``ifvg_verifier_charts``) describing anything truncated — honest truncation,
never silent (FUX §33). Timestamps are converted for DISPLAY through the
repository display-timezone helper; underlying data stays UTC.

Accessibility: no meaning is carried by color alone — glyph classes, dash
patterns, marker shapes, and text labels always accompany it (FUX §32.1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go
from ifvg_verifier_charts import (
    DISPLAY_TIMEZONE,
    OmissionReport,
    to_display_timezone,
)

from alpha_lab.agents.data_infra.ifvg.search.charter import OBJECTIVE_DIRECTIONS
from alpha_lab.agents.data_infra.ifvg.study_status import (
    HEATMAP_GLYPHS,
    TIMELINE_MARKERS,
)

__all__ = [
    "STUDY_LAYER_BUDGETS",
    "direction_colorscale",
    "build_funnel_figure",
    "build_funnel_delta_figure",
    "build_frontier_figure",
    "build_sensitivity_heatmap",
    "build_firm_matrix_figure",
    "build_survival_figure",
    "build_payout_distribution_figure",
    "build_account_timeline_figure",
    "DISPLAY_TIMEZONE",
    "OmissionReport",
]

#: Per-surface trace budgets (points/lines/markers). Exceeding rows are
#: dropped deterministically from the END of the sorted input and reported.
STUDY_LAYER_BUDGETS: Mapping[str, int] = {
    "frontier_points": 256,  # the max_child_count ceiling
    "heatmap_cells": 400,
    "firm_matrix_cells": 256,
    "survival_lines": 12,
    "payout_samples": 20_000,
    "timeline_events": 400,
}

_DASH_CYCLE = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")


def direction_colorscale(metric_key: str | None) -> tuple[str, str]:
    """UI-1 (plan F-05): the colorscale follows the metric's REGISTERED
    optimization direction — ``RdYlGn_r`` for a ``minimize`` metric (worst
    drawdown / highest breach probability never render green) — with the
    direction spelled out for the colorbar title. An unregistered key is
    treated as descriptive: the neutral maximize scale with an explicit
    "direction unregistered" note rather than a silent assumption."""

    direction = OBJECTIVE_DIRECTIONS.get(metric_key or "")
    if direction == "minimize":
        return "RdYlGn_r", "lower is better"
    if direction == "maximize":
        return "RdYlGn", "higher is better"
    return "RdYlGn", "direction unregistered"

#: Plotly marker symbols matching the FUX §28 glyphs (shape ≠ color-only).
_TIMELINE_SYMBOLS = {
    "payout": "triangle-down-open",
    "fee": "diamond-open",
    "breach": "x",
    "replacement": "square-open",
}


def _budget(
    rows: Sequence[Any], budget_key: str, omissions: OmissionReport
) -> Sequence[Any]:
    budget = STUDY_LAYER_BUDGETS[budget_key]
    if len(rows) <= budget:
        return rows
    omissions.add(budget_key, len(rows) - budget, f"budget {budget}")
    return rows[:budget]


def build_funnel_figure(
    stage_labels: Sequence[str], counts: Sequence[int | None]
) -> go.Figure:
    """The Plotly funnel ACCOMPANYING the keyboard buttons (FUX §16.3)."""

    values = [count if count is not None else 0 for count in counts]
    text = [
        str(count) if count is not None else "pending"
        for count in counts
    ]
    figure = go.Figure(
        go.Funnel(
            y=list(stage_labels),
            x=values,
            text=text,
            textinfo="text",
        )
    )
    figure.update_layout(
        margin=dict(l=8, r=8, t=24, b=8), height=280, showlegend=False
    )
    return figure


def build_funnel_delta_figure(
    counter_deltas: Sequence[Mapping[str, Any]],
) -> go.Figure:
    """Per-counter baseline→challenger deltas (FUX §25 funnel panel)."""

    counters = [str(row["counter"]) for row in counter_deltas]
    deltas = [float(row.get("delta") or 0) for row in counter_deltas]
    text = [
        f"{row.get('baseline', '—')} → {row.get('challenger', '—')}"
        for row in counter_deltas
    ]
    figure = go.Figure(
        go.Bar(x=deltas, y=counters, orientation="h", text=text)
    )
    figure.update_layout(
        margin=dict(l=8, r=8, t=24, b=8),
        height=max(240, 28 * len(counters) + 80),
        xaxis_title="challenger − baseline",
        showlegend=False,
    )
    return figure


def build_frontier_figure(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_id: str | None = None,
) -> tuple[go.Figure, OmissionReport]:
    """The payout-reliability frontier (FUX §19).

    Row fields: ``config_id``, ``name``, ``expected_net_payout_90d`` (x),
    ``payout_probability_per_rolling_30d`` (y), ``breach_probability_90d``
    (color), optional ``median_account_lifetime_days`` (size — a uniform
    size plus an omission entry when unavailable), ``feasible`` (infeasible/
    blocked rows are NEVER silently plotted as feasible), optional
    ``firm_context``/``evidence_scope`` hover text.
    """

    omissions = OmissionReport()
    ordered = sorted(rows, key=lambda row: str(row.get("config_id")))
    ordered = list(_budget(ordered, "frontier_points", omissions))
    feasible = [row for row in ordered if row.get("feasible", True)]
    excluded = [row for row in ordered if not row.get("feasible", True)]
    if excluded:
        omissions.add(
            "frontier_points",
            len(excluded),
            "infeasible/blocked configurations are listed, not plotted",
        )
    sizes: list[float] = []
    missing_lifetime = 0
    for row in feasible:
        lifetime = row.get("median_account_lifetime_days")
        if lifetime is None:
            missing_lifetime += 1
            sizes.append(14.0)
        else:
            sizes.append(10.0 + min(float(lifetime), 180.0) / 6.0)
    if missing_lifetime:
        omissions.add(
            "frontier_points",
            missing_lifetime,
            "size encoding degraded: median_account_lifetime_days missing",
        )
    figure = go.Figure(
        go.Scatter(
            x=[row["expected_net_payout_90d"] for row in feasible],
            y=[row["payout_probability_per_rolling_30d"] for row in feasible],
            mode="markers+text",
            text=[
                "★" if row.get("config_id") == selected_id else ""
                for row in feasible
            ],
            textposition="top center",
            customdata=[
                [
                    row.get("config_id"),
                    row.get("name"),
                    row.get("firm_context", "—"),
                    row.get("evidence_scope", "—"),
                ]
                for row in feasible
            ],
            hovertemplate=(
                "%{customdata[1]}<br>expected 90-day net payout: %{x:,.0f}"
                "<br>P(payout / rolling 30d): %{y:.2f}"
                "<br>breach P(90d): %{marker.color:.2f}"
                "<br>firm/policy: %{customdata[2]}"
                "<br>evidence: %{customdata[3]}<extra></extra>"
            ),
            marker=dict(
                size=sizes,
                color=[
                    row.get("breach_probability_90d") for row in feasible
                ],
                colorscale="RdYlGn_r",
                colorbar=dict(title="breach P(90d)"),
                line=dict(width=1),
            ),
        )
    )
    figure.update_layout(
        xaxis_title="expected 90-day net payout ($)",
        yaxis_title="P(at least one payout per 30 days)",
        margin=dict(l=8, r=8, t=24, b=8),
        height=420,
    )
    return figure, omissions


def build_sensitivity_heatmap(
    cells: Sequence[Mapping[str, Any]],
    *,
    row_axis: str,
    col_axis: str,
    metric_label: str,
    metric_key: str | None = None,
) -> tuple[go.Figure, OmissionReport]:
    """The parameter-sensitivity heatmap with glyph classes (FUX §20).

    Cell fields: ``row_value``, ``col_value``, ``value`` (float | None),
    ``cell_class`` (a ``HEATMAP_GLYPHS`` key), ``sample_count``. The glyph
    is drawn as a text annotation on every cell so color never carries the
    class alone; the table twin renders the same records. ``metric_key``
    selects the direction-aware colorscale (plan F-05).
    """

    omissions = OmissionReport()
    bounded = list(_budget(list(cells), "heatmap_cells", omissions))
    row_values = sorted({str(cell["row_value"]) for cell in bounded})
    col_values = sorted({str(cell["col_value"]) for cell in bounded})
    grid: dict[tuple[str, str], Mapping[str, Any]] = {}
    for cell in bounded:
        coordinate = (str(cell["row_value"]), str(cell["col_value"]))
        if coordinate in grid:
            # callers aggregate before building; a duplicate coordinate is
            # reported, never silently collapsed (FUX §33)
            omissions.add(
                "heatmap_cells",
                1,
                f"duplicate cell at {coordinate} — last value kept",
            )
        grid[coordinate] = cell
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for row_value in row_values:
        z_row: list[float | None] = []
        text_row: list[str] = []
        for col_value in col_values:
            cell = grid.get((row_value, col_value))
            if cell is None:
                z_row.append(None)
                text_row.append(HEATMAP_GLYPHS["insufficient_data"])
            else:
                z_row.append(
                    None if cell.get("value") is None else float(cell["value"])
                )
                text_row.append(
                    HEATMAP_GLYPHS.get(str(cell.get("cell_class")), "·")
                )
        z.append(z_row)
        text.append(text_row)
    colorscale, direction_note = direction_colorscale(metric_key)
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=col_values,
            y=row_values,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            colorbar=dict(title=f"{metric_label} ({direction_note})"),
            hoverongaps=False,
        )
    )
    figure.update_layout(
        xaxis_title=col_axis,
        yaxis_title=row_axis,
        margin=dict(l=8, r=8, t=24, b=8),
        height=max(300, 40 * len(row_values) + 120),
    )
    return figure, omissions


def build_firm_matrix_figure(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric_label: str,
    metric_key: str | None = None,
) -> tuple[go.Figure, OmissionReport]:
    """Strategy-configuration × firm matrix (FUX §21).

    Cell fields: ``config_name``, ``firm_label``, ``value`` (float | None),
    ``status_text`` (shown in the cell — missing/blocked/unsupported
    REASONS render as text, never as a fake zero), ``universal`` (bool).
    """

    omissions = OmissionReport()
    bounded = list(_budget(list(cells), "firm_matrix_cells", omissions))
    rows = sorted({str(cell["config_name"]) for cell in bounded})
    cols = sorted({str(cell["firm_label"]) for cell in bounded})
    grid = {
        (str(cell["config_name"]), str(cell["firm_label"])): cell
        for cell in bounded
    }
    z: list[list[float | None]] = []
    text: list[list[str]] = []
    for row in rows:
        z_row: list[float | None] = []
        text_row: list[str] = []
        for col in cols:
            cell = grid.get((row, col))
            if cell is None or cell.get("value") is None:
                z_row.append(None)
                text_row.append(
                    str((cell or {}).get("status_text") or "no data")
                )
            else:
                z_row.append(float(cell["value"]))
                marker = "◆ " if cell.get("universal") else ""
                text_row.append(f"{marker}{float(cell['value']):.2f}")
        z.append(z_row)
        text.append(text_row)
    colorscale, direction_note = direction_colorscale(metric_key)
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=cols,
            y=rows,
            text=text,
            texttemplate="%{text}",
            colorscale=colorscale,
            colorbar=dict(title=f"{metric_label} ({direction_note})"),
            hoverongaps=False,
        )
    )
    figure.update_layout(
        xaxis_title="firm / account contract",
        yaxis_title="strategy configuration (◆ = universal)",
        margin=dict(l=8, r=8, t=24, b=8),
        height=max(300, 40 * len(rows) + 120),
    )
    return figure, omissions


def build_survival_figure(
    curves: Sequence[Mapping[str, Any]],
) -> tuple[go.Figure, OmissionReport]:
    """Account-survival step curves, one line per firm/policy (FUX §22).

    Curve fields: ``label``, ``days`` (x), ``survival`` (y in [0,1]),
    optional ``mode_caption``. Distinct dash patterns ride WITH color.
    """

    omissions = OmissionReport()
    bounded = list(_budget(list(curves), "survival_lines", omissions))
    figure = go.Figure()
    for index, curve in enumerate(bounded):
        figure.add_trace(
            go.Scatter(
                x=list(curve["days"]),
                y=list(curve["survival"]),
                mode="lines",
                name=str(curve["label"]),
                line=dict(
                    shape="hv",
                    dash=_DASH_CYCLE[index % len(_DASH_CYCLE)],
                    width=2,
                ),
            )
        )
    figure.update_layout(
        xaxis_title="trading days",
        yaxis_title="P(account alive)",
        yaxis=dict(range=[0, 1.02]),
        margin=dict(l=8, r=8, t=24, b=8),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return figure, omissions


def build_payout_distribution_figure(
    samples: Sequence[float],
    *,
    horizon_label: str,
    quantiles: Mapping[str, float],
) -> tuple[go.Figure, OmissionReport]:
    """Payout distribution for one horizon (FUX §23). P10 drawn prominent."""

    omissions = OmissionReport()
    bounded = list(_budget(list(samples), "payout_samples", omissions))
    figure = go.Figure(
        go.Histogram(x=bounded, nbinsx=60, name=horizon_label)
    )
    for name, dash, width in (
        ("p10", "solid", 3),  # the lower tail leads (FUX §23)
        ("median", "dash", 2),
        ("mean", "dot", 2),
        ("p90", "dash", 1),
    ):
        value = quantiles.get(name)
        if value is None:
            continue
        figure.add_vline(
            x=float(value),
            line_dash=dash,
            line_width=width,
            annotation_text=f"{name.upper()} {float(value):,.0f}",
            annotation_position="top",
        )
    figure.update_layout(
        xaxis_title=f"net payout — {horizon_label}",
        yaxis_title="paths",
        margin=dict(l=8, r=8, t=48, b=8),
        height=360,
        showlegend=False,
    )
    return figure, omissions


def build_account_timeline_figure(
    events: Sequence[Mapping[str, Any]],
    *,
    eligibility_windows: Sequence[tuple[str, str]] = (),
) -> tuple[go.Figure, OmissionReport]:
    """The account timeline (FUX §28) from ordered event envelopes.

    ``events`` are ``PropAccountEventEnvelope`` dicts already sorted by
    ``event_ordinal`` (the provider guarantees the total order — chart
    timestamps are display only). Balance/threshold lines derive from
    ``equity_update`` / ``threshold_ratchet`` / ``daily_halt`` payloads;
    payout/fee/breach/replacement markers use the exact FUX shapes with
    text labels; payout-eligibility windows are shaded.
    """

    omissions = OmissionReport()
    bounded = list(_budget(list(events), "timeline_events", omissions))
    figure = go.Figure()
    balance_x: list[str] = []
    balance_y: list[float] = []
    floor_x: list[str] = []
    floor_y: list[float] = []
    for event in bounded:
        ts = str(event.get("event_ts_utc"))
        payload = event.get("payload") or {}
        if event.get("event_type") == "equity_update":
            balance_x.append(ts)
            balance_y.append(float(payload.get("new_equity", 0.0)))
        elif event.get("event_type") == "threshold_ratchet":
            floor_x.append(ts)
            floor_y.append(float(payload.get("new_floor", 0.0)))
    if balance_x:
        figure.add_trace(
            go.Scatter(
                x=balance_x,
                y=balance_y,
                mode="lines",
                name="balance",
                line=dict(width=2),
            )
        )
    if floor_x:
        figure.add_trace(
            go.Scatter(
                x=floor_x,
                y=floor_y,
                mode="lines",
                name="trailing threshold",
                line=dict(shape="hv", dash="dash", width=2),
            )
        )
    daily_halts = [
        event for event in bounded if event.get("event_type") == "daily_halt"
    ]
    if daily_halts:
        # FUX §28: the daily-loss threshold is a SEPARATE LINE (step) —
        # markers alone only when a single observation exists
        figure.add_trace(
            go.Scatter(
                x=[str(event.get("event_ts_utc")) for event in daily_halts],
                y=[
                    float((event.get("payload") or {}).get("threshold_value", 0.0))
                    for event in daily_halts
                ],
                mode="lines+markers" if len(daily_halts) > 1 else "markers",
                name="daily-loss threshold",
                line=dict(shape="hv", dash="dot", width=2),
                marker=dict(symbol="line-ew-open", size=12),
            )
        )
    phase_transitions = [
        event
        for event in bounded
        if event.get("event_type") == "phase_transition"
    ]
    if phase_transitions:
        figure.add_trace(
            go.Scatter(
                x=[
                    str(event.get("event_ts_utc"))
                    for event in phase_transitions
                ],
                y=[_marker_y(event) for event in phase_transitions],
                mode="markers+text",
                name="phase transition",
                text=[
                    f"{(event.get('payload') or {}).get('from_phase', '?')}"
                    f"→{(event.get('payload') or {}).get('to_phase', '?')}"
                    for event in phase_transitions
                ],
                textposition="bottom center",
                marker=dict(symbol="arrow-up-open", size=12),
            )
        )
    for start, end in eligibility_windows:
        figure.add_vrect(
            x0=start,
            x1=end,
            fillcolor="LightGreen",
            opacity=0.15,
            line_width=0,
            annotation_text="payout eligibility",
            annotation_position="top left",
        )
    for kind, glyph in TIMELINE_MARKERS.items():
        marked = [
            event for event in bounded if event.get("event_type") == kind
        ]
        if not marked:
            continue
        figure.add_trace(
            go.Scatter(
                x=[str(event.get("event_ts_utc")) for event in marked],
                y=[_marker_y(event) for event in marked],
                mode="markers+text",
                name=f"{kind} {glyph}",
                text=[glyph] * len(marked),
                textposition="top center",
                customdata=[
                    [
                        event.get("event_id"),
                        event.get("event_ordinal"),
                        event.get("source_trade_id") or "—",
                        event.get("source_setup_id") or "—",
                    ]
                    for event in marked
                ],
                hovertemplate=(
                    f"{kind} {glyph}<br>ordinal %{{customdata[1]}}"
                    "<br>trade %{customdata[2]}"
                    "<br>setup %{customdata[3]}<extra></extra>"
                ),
                marker=dict(symbol=_TIMELINE_SYMBOLS[kind], size=12),
            )
        )
    figure.update_layout(
        xaxis_title=f"time ({DISPLAY_TIMEZONE} display)",
        yaxis_title="account value ($)",
        margin=dict(l=8, r=8, t=24, b=8),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    to_display_timezone(figure)
    return figure, omissions


def _marker_y(event: Mapping[str, Any]) -> float:
    payload = event.get("payload") or {}
    for key in (
        "observed_equity",
        "new_equity",
        "trader_amount",
        "amount",
        "reset_fee",
    ):
        value = payload.get(key)
        if value is not None:
            return float(value)
    return 0.0
