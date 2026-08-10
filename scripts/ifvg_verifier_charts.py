"""Pure Plotly builders for the IFVG visual trade verifier (context_v1 lane).

No Streamlit imports — every function is testable headlessly.  The builder
consumes the provider's :class:`CandidateEvidence` plus pre-sliced bar frames
and returns one three-pane figure (1m execution / parent TF / HTF) together
with an :class:`OmissionReport`.

Rendering rules enforced here:

* zone rectangles begin at ``confirmed_ts`` — never drawn backward;
* zones with no persisted fill/invalidation end carry "lifecycle end unknown";
* actual execution = solid entry/stop/target + filled marker; counterfactual
  label path = dashed + hollow marker; blocked candidates get a reason chip
  and never look executable;
* MFE/MAE render as dashed LEVELS labeled "magnitude only — timing not
  persisted";
* the ``experimental_q40_open`` watermark appears iff 240m evidence is shown;
* per-layer shape budgets degrade gracefully into the omission report — the
  chart never fails on budget excess and never omits silently.

The numeric coordinates of everything drawn are registered in
``fig.layout.meta`` and read back via :func:`figure_geometry_index` so golden
tests can check them against the source artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ifvg_lab_charts import TICK_SIZE, ZONE_COLORS, rgba
from plotly.subplots import make_subplots

__all__ = [
    "LAYER_BUDGETS",
    "VerifierLayers",
    "OmissionReport",
    "build_setup_figure",
    "build_verifier_figure",
    "figure_geometry_index",
    "session_band_intervals",
]

LAYER_BUDGETS = {
    "zones": 16,
    "sessions": 30,
    "structure": 40,
    "displacement": 12,
    "pools": 12,
    "markers": 60,
    # setup-mode layers (additive; candidate-mode budgets unchanged)
    "setup_taps": 60,
    "setup_fills": 80,
    "setup_deaths": 30,
    "setup_intervals": 30,
}

_STAGE_MARKER_SYMBOLS = {
    "tap": "circle",
    "parent": "diamond",
    "lock": "square",
    "opposing": "triangle-right",
    "inversion": "x",
    "entry": "star",
    "resolution": "circle-open",
}

_SESSION_COLORS = {
    "asia": "rgba(76,120,168,0.06)",
    "london": "rgba(230,159,0,0.06)",
    "ny": "rgba(89,161,79,0.06)",
}

_Q40_TEXT = "experimental_q40_open — not canonical context"


@dataclass(frozen=True)
class VerifierLayers:
    structure: bool = False
    displacement: bool = False
    pools: bool = False
    sessions: bool = True
    zone_projection: bool = True


@dataclass
class OmissionReport:
    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, layer: str, omitted: int, detail: str) -> None:
        if omitted > 0:
            self.entries.append({"layer": layer, "omitted": int(omitted), "detail": detail})

    def summary_lines(self) -> list[str]:
        return [
            f"{entry['omitted']} {entry['detail']} omitted from the {entry['layer']} layer"
            for entry in self.entries
        ]


def _budget(
    items: list[Any], layer: str, report: OmissionReport, *, detail: str
) -> list[Any]:
    limit = LAYER_BUDGETS[layer]
    if len(items) <= limit:
        return items
    report.add(layer, len(items) - limit, detail)
    return items[:limit]


def _pane_for_timeframe(timeframe: int, parent_tf: int, htf_tf: int) -> int:
    if timeframe == 60:
        return 1
    if timeframe == parent_tf:
        return 2
    if timeframe in (3600, 14400) or timeframe == htf_tf:
        return 3
    return 2


def _bar_frame_xohlc(bars: pd.DataFrame, tick_size: float) -> dict[str, Any]:
    if "logical_open_ts_utc" in bars.columns:
        x = pd.to_datetime(bars["logical_open_ts_utc"], utc=True)
    else:
        x = pd.to_datetime(bars["close_ts_utc"], utc=True) - pd.Timedelta(seconds=60)
    return {
        "x": x,
        "open": bars["open_ticks"] * tick_size,
        "high": bars["high_ticks"] * tick_size,
        "low": bars["low_ticks"] * tick_size,
        "close": bars["close_ticks"] * tick_size,
    }


def _pane_y_range(
    bars: pd.DataFrame | None,
    tick_size: float,
    *,
    extra_prices: list[float | None] | tuple = (),
) -> list[float] | None:
    """Bar-driven y-range with a small pad; overlays never dictate the scale."""
    if bars is None or not len(bars):
        return None
    low = float(bars["low_ticks"].min()) * tick_size
    high = float(bars["high_ticks"].max()) * tick_size
    prices = [price for price in extra_prices if price is not None]
    if prices:
        low = min(low, min(prices))
        high = max(high, max(prices))
    span = max(high - low, tick_size * 8)
    pad = span * 0.06
    return [low - pad, high + pad]


def _bar_high_price(bars: pd.DataFrame, ts: pd.Timestamp, tick_size: float) -> float | None:
    if bars is None or not len(bars):
        return None
    column = "close_ts_utc" if "close_ts_utc" in bars.columns else "logical_close_ts_utc"
    closes = pd.to_datetime(bars[column], utc=True).astype("int64").to_numpy()
    index = int(np.searchsorted(closes, pd.Timestamp(ts).value, side="right")) - 1
    index = min(max(index, 0), len(bars) - 1)
    return float(bars["high_ticks"].iloc[index]) * tick_size


def session_band_intervals(
    schemes: dict[str, Any],
    scheme_name: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    """UTC session intervals over [start, end] from the exact scheme geometry."""
    scheme = schemes.get(scheme_name)
    if not scheme:
        return []
    zone = ZoneInfo(scheme["timezone"])
    intervals: list[dict[str, Any]] = []
    day = (start_ts.tz_convert(zone).date() - timedelta(days=1))
    last_day = end_ts.tz_convert(zone).date() + timedelta(days=1)
    while day <= last_day:
        for name, window in scheme["sessions"].items():
            window_start = time.fromisoformat(window["start"])
            window_end = time.fromisoformat(window["end"])
            begin = datetime.combine(day, window_start, tzinfo=zone)
            end_day = day + timedelta(days=1) if window["crosses_midnight"] else day
            finish = datetime.combine(end_day, window_end, tzinfo=zone)
            begin_utc = pd.Timestamp(begin).tz_convert("UTC")
            finish_utc = pd.Timestamp(finish).tz_convert("UTC")
            if finish_utc < start_ts or begin_utc > end_ts:
                continue
            intervals.append(
                {
                    "session": name,
                    "start": max(begin_utc, start_ts),
                    "end": min(finish_utc, end_ts),
                }
            )
        day = day + timedelta(days=1)
    return intervals


def _iso(ts: Any) -> str | None:
    if ts is None:
        return None
    stamp = pd.Timestamp(ts)
    if pd.isna(stamp):
        return None
    return stamp.isoformat()


def build_verifier_figure(
    *,
    evidence: Any,
    bars_by_pane: dict[int, pd.DataFrame],
    parent_tf: int,
    htf_tf: int,
    range_bounds: tuple[pd.Timestamp, pd.Timestamp],
    layers: VerifierLayers | None = None,
    blocked_reasons: str | None = None,
    tick_size: float = TICK_SIZE,
) -> tuple[go.Figure, OmissionReport]:
    """One synchronized three-pane figure plus the omission report."""
    layers = layers if layers is not None else VerifierLayers()
    report = OmissionReport()
    start_ts, end_ts = (pd.Timestamp(range_bounds[0]), pd.Timestamp(range_bounds[1]))
    meta: dict[str, Any] = {
        "candidate_id": evidence.candidate_id,
        "mode": evidence.mode,
        "stage": evidence.stage,
        "range": {"start": _iso(start_ts), "end": _iso(end_ts)},
        "zones": {},
        "risk": {},
        "execution": {},
        "mfe_mae": {},
        "stage_markers": {},
        "counterfactual": bool(evidence.execution is None),
        "blocked_reasons": blocked_reasons,
    }

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=(
            "1m execution",
            f"parent {parent_tf // 60}m",
            f"HTF {htf_tf // 60}m" if htf_tf < 3600 else f"HTF {htf_tf // 3600}H",
        ),
    )
    bars_1m = bars_by_pane.get(60, pd.DataFrame())
    for row, timeframe in ((1, 60), (2, parent_tf), (3, htf_tf)):
        bars = bars_by_pane.get(timeframe, pd.DataFrame())
        if not len(bars):
            continue
        fig.add_trace(
            go.Candlestick(
                **_bar_frame_xohlc(bars, tick_size),
                name=f"{timeframe}s",
                showlegend=False,
                increasing_line_color="#59A14F",
                decreasing_line_color="#E45756",
                increasing_line_width=1,
                decreasing_line_width=1,
            ),
            row=row,
            col=1,
        )

    # ── sessions ─────────────────────────────────────────────────────────────
    if layers.sessions:
        intervals = session_band_intervals(
            evidence.sessions.get("schemes", {}), "doc", start_ts, end_ts
        )
        intervals = _budget(intervals, "sessions", report, detail="session bands")
        for interval in intervals:
            fig.add_vrect(
                x0=interval["start"],
                x1=interval["end"],
                fillcolor=_SESSION_COLORS.get(interval["session"], "rgba(120,120,120,0.05)"),
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )

    # ── FVG zones ────────────────────────────────────────────────────────────
    zone_hover_x: list[Any] = []
    zone_hover_y: list[float] = []
    zone_hover_text: list[str] = []
    inversion_gate = evidence.stage_gates.get("inversion")
    inversion_ts = inversion_gate.ts_utc if inversion_gate is not None else None
    shown_240m = htf_tf == 14400
    zones = _budget(list(evidence.zones), "zones", report, detail="FVG zones")
    for zone in zones:
        x0 = max(pd.Timestamp(zone.confirmed_ts_utc), start_ts)
        if pd.Timestamp(zone.confirmed_ts_utc) > end_ts:
            continue
        if zone.timeframe_seconds == 14400:
            shown_240m = True
        y0 = zone.gap_low_ticks * tick_size
        y1 = zone.gap_high_ticks * tick_size
        color = ZONE_COLORS.get(zone.role, ZONE_COLORS["recomputed"])
        native_row = _pane_for_timeframe(zone.timeframe_seconds, parent_tf, htf_tf)
        segments: list[tuple[pd.Timestamp, pd.Timestamp, str]]
        if (
            zone.role == "opposing"
            and inversion_ts is not None
            and start_ts <= inversion_ts <= end_ts
            and pd.Timestamp(zone.confirmed_ts_utc) < inversion_ts
        ):
            segments = [(x0, inversion_ts, "pre"), (inversion_ts, end_ts, "post")]
        else:
            segments = [(x0, end_ts, "solid")]
        target_rows = {native_row}
        if layers.zone_projection and native_row != 1:
            target_rows.add(1)
        for target_row in sorted(target_rows):
            projected = target_row != native_row
            for seg_start, seg_end, phase in segments:
                if phase == "pre":
                    style = {
                        "fillcolor": rgba(color, 0.03 if projected else 0.06),
                        "line": {"color": color, "width": 1, "dash": "dash"},
                    }
                elif phase == "post":
                    inverted = ZONE_COLORS["entry_fvg"]
                    style = {
                        "fillcolor": rgba(inverted, 0.06 if projected else 0.14),
                        "line": {"color": inverted, "width": 1.5, "dash": "solid"},
                    }
                else:
                    style = {
                        "fillcolor": rgba(color, 0.08 if projected else 0.18),
                        "line": {"color": color, "width": 1 if projected else 1.5},
                    }
                fig.add_shape(
                    type="rect",
                    x0=seg_start,
                    x1=seg_end,
                    y0=y0,
                    y1=y1,
                    row=target_row,
                    col=1,
                    **style,
                )
        zone_hover_x.append(x0)
        zone_hover_y.append((y0 + y1) / 2)
        zone_hover_text.append(
            f"{zone.role} FVG {zone.fvg_id}<br>"
            f"tf {zone.timeframe_seconds}s · {zone.direction}<br>"
            f"[{y0:.2f}, {y1:.2f}]<br>"
            f"confirmed {_iso(zone.confirmed_ts_utc)}<br>"
            "lifecycle end unknown (fill/invalidation not persisted)"
        )
        meta["zones"][zone.role] = {
            "fvg_id": zone.fvg_id,
            "timeframe_seconds": zone.timeframe_seconds,
            "x0": _iso(x0),
            "confirmed_ts": _iso(zone.confirmed_ts_utc),
            "y0": y0,
            "y1": y1,
        }
    if zone_hover_x:
        fig.add_trace(
            go.Scatter(
                x=zone_hover_x,
                y=zone_hover_y,
                mode="markers",
                marker={"size": 8, "opacity": 0.01},
                hovertext=zone_hover_text,
                hoverinfo="text",
                showlegend=False,
                name="zones",
            ),
            row=1,
            col=1,
        )

    # ── risk / execution overlay ─────────────────────────────────────────────
    risk = evidence.risk
    executed = evidence.execution is not None
    entry_anchor = pd.Timestamp(evidence.range_row["entry_anchor_ts"])
    display_end = min(pd.Timestamp(evidence.range_row["display_end_ts"]), end_ts)
    if "entry_ticks" in risk:
        entry_price = risk["entry_ticks"] * tick_size
        stop_price = risk["stop_ticks"] * tick_size
        target_price = risk["target_ticks"] * tick_size
        dash = "solid" if executed else "dash"
        line_start = max(entry_anchor, start_ts)
        for price, color, label in (
            (entry_price, "#1F77B4", "entry"),
            (stop_price, "#D62728", "stop"),
            (target_price, "#2CA02C", "target (1R)"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[line_start, display_end],
                    y=[price, price],
                    mode="lines",
                    line={"color": color, "width": 1.6, "dash": dash},
                    name=label if executed else f"{label} (counterfactual)",
                    legendgroup="risk",
                    showlegend=label == "entry",
                    hovertext=(
                        f"{label}: {price:.2f}"
                        + ("" if executed else " — counterfactual label path")
                    ),
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
        # R/R box: entry→stop red, entry→target green.
        for bound, color in ((stop_price, "#D62728"), (target_price, "#2CA02C")):
            fig.add_shape(
                type="rect",
                x0=line_start,
                x1=display_end,
                y0=min(entry_price, bound),
                y1=max(entry_price, bound),
                fillcolor=rgba(color, 0.05),
                line_width=0,
                layer="below",
                row=1,
                col=1,
            )
        marker_symbol = (
            "triangle-up" if risk.get("direction") == "LONG" else "triangle-down"
        )
        fig.add_trace(
            go.Scatter(
                x=[max(entry_anchor, start_ts)],
                y=[entry_price],
                mode="markers",
                marker={
                    "symbol": marker_symbol,
                    "size": 13,
                    "color": "#1F77B4" if executed else "rgba(0,0,0,0)",
                    "line": {"color": "#1F77B4", "width": 2},
                },
                name="actual entry" if executed else "counterfactual entry",
                showlegend=False,
                hovertext=(
                    f"{'actual' if executed else 'counterfactual'} entry @ {entry_price:.2f}"
                ),
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        if risk.get("manipulation_swing_ticks") is not None:
            swing_price = risk["manipulation_swing_ticks"] * tick_size
            fig.add_trace(
                go.Scatter(
                    x=[line_start, display_end],
                    y=[swing_price, swing_price],
                    mode="lines",
                    line={"color": "#8C564B", "width": 1, "dash": "dashdot"},
                    name="structural swing (stop basis)",
                    legendgroup="risk",
                    showlegend=False,
                    hovertext=(
                        f"structural swing {swing_price:.2f} "
                        f"(+{risk.get('sl_buffer_ticks')} tick buffer)"
                    ),
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
        meta["risk"] = {
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "entry_ts": _iso(entry_anchor),
            "end_ts": _iso(display_end),
            "actual": executed,
        }

    if executed and "resolution" in evidence.execution:
        resolution_ts = pd.Timestamp(evidence.execution["resolution_ts_utc"])
        entry_price = evidence.execution["entry_ticks"] * tick_size
        direction = 1 if risk.get("direction") == "LONG" else -1
        mfe_price = entry_price + direction * evidence.execution["mfe_ticks"] * tick_size
        mae_price = entry_price - direction * evidence.execution["mae_ticks"] * tick_size
        for price, label in (
            (mfe_price, "MFE magnitude only — timing not persisted"),
            (mae_price, "MAE magnitude only — timing not persisted"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[max(entry_anchor, start_ts), min(resolution_ts, end_ts)],
                    y=[price, price],
                    mode="lines",
                    line={"color": "#7F7F7F", "width": 1, "dash": "dot"},
                    name=label,
                    legendgroup="excursion",
                    showlegend=False,
                    hovertext=f"{label}: {price:.2f}",
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
        if start_ts <= resolution_ts <= end_ts:
            fig.add_annotation(
                x=resolution_ts,
                y=entry_price,
                text=(
                    f"{evidence.execution['resolution']} · "
                    f"{evidence.execution['bars_after_entry_to_resolution']} bars · "
                    f"{evidence.execution['realized_r']:+.2f}R"
                ),
                showarrow=True,
                arrowhead=2,
                font={"size": 10},
                row=1,
                col=1,
            )
        meta["execution"] = {
            "resolution": evidence.execution["resolution"],
            "resolution_ts": _iso(resolution_ts),
            "realized_r": evidence.execution["realized_r"],
        }
        meta["mfe_mae"] = {"mfe_price": mfe_price, "mae_price": mae_price}

    if blocked_reasons:
        fig.add_trace(
            go.Scatter(
                x=[max(entry_anchor, start_ts)],
                y=[_bar_high_price(bars_1m, entry_anchor, tick_size) or 0],
                mode="markers+text",
                marker={"symbol": "x-thin", "size": 12, "color": "#8C8C8C",
                        "line": {"color": "#8C8C8C", "width": 2}},
                text=["blocked"],
                textposition="top center",
                textfont={"size": 9, "color": "#8C8C8C"},
                name="blocked candidate",
                showlegend=False,
                hovertext=f"blocked: {blocked_reasons}",
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # ── stage markers ────────────────────────────────────────────────────────
    marker_x, marker_y, marker_symbols, marker_text = [], [], [], []
    for stage_name, gate in evidence.stage_gates.items():
        if not (start_ts <= gate.ts_utc <= end_ts):
            continue
        anchor = _bar_high_price(bars_1m, gate.ts_utc, tick_size)
        if anchor is None:
            continue
        marker_x.append(gate.ts_utc)
        marker_y.append(anchor * 1.0006)
        marker_symbols.append(_STAGE_MARKER_SYMBOLS.get(stage_name, "circle"))
        marker_text.append(f"{stage_name} · {_iso(gate.ts_utc)}<br>{gate.source_kind}")
        meta["stage_markers"][stage_name] = _iso(gate.ts_utc)
    if marker_x:
        fig.add_trace(
            go.Scatter(
                x=marker_x,
                y=marker_y,
                mode="markers",
                marker={"symbol": marker_symbols, "size": 9, "color": "#4C4C4C"},
                hovertext=marker_text,
                hoverinfo="text",
                name="FSM stages",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ── structure layer ──────────────────────────────────────────────────────
    if layers.structure and len(evidence.structure):
        rows = [
            row
            for row in evidence.structure.to_dict("records")
            if row.get("break_ts") is not None and not pd.isna(row.get("break_ts"))
            and start_ts <= pd.Timestamp(row["break_ts"]) <= end_ts
        ]
        rows = _budget(rows, "structure", report, detail="structure break markers")
        for row in rows:
            timeframe = int(row["source_timeframe_seconds"])
            if timeframe == 14400:
                shown_240m = True
            pane = _pane_for_timeframe(timeframe, parent_tf, htf_tf)
            pane_bars = bars_by_pane.get(60 if pane == 1 else parent_tf if pane == 2 else htf_tf)
            anchor = _bar_high_price(
                pane_bars if pane_bars is not None else bars_1m,
                pd.Timestamp(row["break_ts"]),
                tick_size,
            )
            if anchor is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[pd.Timestamp(row["break_ts"])],
                    y=[anchor * 1.0008],
                    mode="markers+text",
                    marker={"symbol": "diamond-tall", "size": 8, "color": "#B279A2"},
                    text=[str(row.get("last_break_type") or "").upper()],
                    textposition="top center",
                    textfont={"size": 8},
                    hovertext=(
                        f"{row.get('last_break_type')} {row.get('last_break_direction')} "
                        f"@ {row['source_timeframe']}<br>"
                        f"{row.get('high_relationship')}/{row.get('low_relationship')} · "
                        f"direction {row.get('structure_direction')}<br>"
                        f"stage {row['stage']} (provider evidence — not recomputed)"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                    name="structure",
                ),
                row=pane,
                col=1,
            )

    # ── displacement layer ───────────────────────────────────────────────────
    if layers.displacement and len(evidence.displacement):
        rows = [
            row
            for row in evidence.displacement.to_dict("records")
            if row.get("start_ts") is not None and not pd.isna(row.get("start_ts"))
        ]
        rows = _budget(rows, "displacement", report, detail="displacement windows")
        for row in rows:
            window_start = max(pd.Timestamp(row["start_ts"]), start_ts)
            window_end = min(
                pd.Timestamp(row["end_ts"]) if not pd.isna(row.get("end_ts")) else end_ts,
                end_ts,
            )
            if window_end < start_ts or window_start > end_ts:
                continue
            valid = bool(row.get("valid"))
            fig.add_vrect(
                x0=window_start,
                x1=window_end,
                fillcolor="rgba(178,121,162,0.10)" if valid else "rgba(140,140,140,0.08)",
                line={"color": "#B279A2" if valid else "#8C8C8C", "width": 1, "dash": "dot"},
                layer="below",
                row=1,
                col=1,
            )
            mid = window_start + (window_end - window_start) / 2
            anchor = _bar_high_price(bars_1m, mid, tick_size)
            metrics = (
                f"path efficiency {row.get('metrics_path_efficiency_abs')}<br>"
                f"body fraction {row.get('metrics_body_fraction_mean')}<br>"
                f"opposing wick {row.get('metrics_opposing_wick_fraction_mean')}<br>"
                f"overlap {row.get('metrics_overlap_fraction_mean')}<br>"
                f"max directional run {row.get('metrics_max_consecutive_directional_bars')}<br>"
                f"normalized move {row.get('metrics_setup_net_move_normalized')}"
                if valid
                else f"INVALID: {row.get('missing_reason')}"
            )
            fig.add_trace(
                go.Scatter(
                    x=[mid],
                    y=[anchor or 0],
                    mode="markers",
                    marker={"size": 8, "opacity": 0.01},
                    hovertext=(
                        f"displacement {row['window_kind']} ({row['stage']})<br>"
                        f"{_iso(window_start)} → {_iso(window_end)} (exact bar interval)<br>"
                        + metrics
                    ),
                    hoverinfo="text",
                    showlegend=False,
                    name="displacement",
                ),
                row=1,
                col=1,
            )

    # ── EQH/EQL pools layer ──────────────────────────────────────────────────
    if layers.pools and len(evidence.pools):
        pool_rows = evidence.pools.to_dict("records")
        selected_pools = {
            str(row["pool_id"])
            for row in evidence.sweep_links.to_dict("records")
            if row.get("selected")
        }
        pool_rows.sort(key=lambda row: (str(row["pool_pool_id"]) not in selected_pools,))
        pool_rows = _budget(pool_rows, "pools", report, detail="non-linked pools")
        for row in pool_rows:
            confirmation = row.get("pool_confirmation_ts")
            if confirmation is None or pd.isna(confirmation):
                continue
            band_start = max(pd.Timestamp(confirmation), start_ts)
            swept_ts = row.get("pool_sweep_ts")
            band_end = (
                min(pd.Timestamp(swept_ts), end_ts)
                if row.get("pool_swept") and not pd.isna(swept_ts)
                else end_ts
            )
            if band_end < start_ts or band_start > end_ts:
                continue
            y0 = row["pool_lower_bound_ticks"] * tick_size
            y1 = row["pool_upper_bound_ticks"] * tick_size
            pool_type = str(row.get("pool_pool_type", "pool"))
            color = "#E45756" if pool_type == "eqh" else "#4C78A8"
            fig.add_shape(
                type="rect",
                x0=band_start,
                x1=band_end,
                y0=y0,
                y1=y1,
                fillcolor=rgba(color, 0.10),
                line={"color": color, "width": 1, "dash": "dot"},
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[band_start],
                    y=[(y0 + y1) / 2],
                    mode="markers",
                    marker={"size": 8, "opacity": 0.01},
                    hovertext=(
                        f"{pool_type.upper()} pool {row['pool_pool_id']}<br>"
                        f"[{y0:.2f}, {y1:.2f}] · confirmed {_iso(confirmation)}<br>"
                        f"active={row.get('pool_active')} swept={row.get('pool_swept')} "
                        f"reclaimed={row.get('pool_reclaimed')}"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                    name="pools",
                ),
                row=1,
                col=1,
            )
        members = evidence.pool_members.to_dict("records")
        member_x = [
            pd.Timestamp(row["swing_pivot_ts"])
            for row in members
            if not pd.isna(row.get("swing_pivot_ts"))
            and start_ts <= pd.Timestamp(row["swing_pivot_ts"]) <= end_ts
        ]
        member_y = [
            row["swing_price_ticks"] * tick_size
            for row in members
            if not pd.isna(row.get("swing_pivot_ts"))
            and start_ts <= pd.Timestamp(row["swing_pivot_ts"]) <= end_ts
        ]
        if member_x:
            fig.add_trace(
                go.Scatter(
                    x=member_x,
                    y=member_y,
                    mode="markers",
                    marker={"symbol": "diamond", "size": 7, "color": "#F1CE63"},
                    name="pool member swings",
                    showlegend=False,
                    hovertext="pool member swing (pivot; confirmed later)",
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
        for row in evidence.sweep_links.to_dict("records"):
            sweep_ts = row.get("sweep_ts")
            if sweep_ts is None or pd.isna(sweep_ts):
                continue
            sweep_stamp = pd.Timestamp(sweep_ts)
            if not (start_ts <= sweep_stamp <= end_ts):
                continue
            anchor = _bar_high_price(bars_1m, sweep_stamp, tick_size)
            fig.add_trace(
                go.Scatter(
                    x=[sweep_stamp],
                    y=[anchor or 0],
                    mode="markers",
                    marker={
                        "symbol": "x",
                        "size": 10,
                        "color": "#E45756" if row.get("selected") else "#8C8C8C",
                    },
                    hovertext=(
                        f"sweep {row['sweep_link_id']}<br>"
                        f"depth {row.get('sweep_depth_ticks')} ticks · "
                        f"qualifies_opposing_leg={row.get('qualifies_opposing_leg')} · "
                        f"selected={bool(row.get('selected'))}"
                    ),
                    hoverinfo="text",
                    showlegend=False,
                    name="sweeps",
                ),
                row=1,
                col=1,
            )
            reclaim_ts = row.get("reclaim_ts")
            if reclaim_ts is not None and not pd.isna(reclaim_ts):
                reclaim_stamp = pd.Timestamp(reclaim_ts)
                if start_ts <= reclaim_stamp <= end_ts:
                    fig.add_trace(
                        go.Scatter(
                            x=[reclaim_stamp],
                            y=[_bar_high_price(bars_1m, reclaim_stamp, tick_size) or 0],
                            mode="markers",
                            marker={"symbol": "circle-open", "size": 10, "color": "#59A14F"},
                            hovertext="reclaim candle",
                            hoverinfo="text",
                            showlegend=False,
                            name="reclaims",
                        ),
                        row=1,
                        col=1,
                    )

    # ── Q-40 watermark: only when 240m evidence is displayed ────────────────
    if shown_240m:
        fig.add_annotation(
            xref="x3 domain",
            yref="y3 domain",
            x=0.98,
            y=0.92,
            text=_Q40_TEXT,
            showarrow=False,
            font={"size": 10, "color": "#B25D00"},
            opacity=0.85,
        )
    meta["watermark_240m"] = shown_240m

    point_in_time = evidence.mode == "point_in_time"
    title = (
        f"[POINT-IN-TIME · {evidence.stage}] candidate {evidence.candidate_id[:12]}…"
        if point_in_time
        else f"[FULL AUDIT] candidate {evidence.candidate_id[:12]}…"
    )
    fig.update_layout(
        title=title,
        height=720,
        hovermode="x unified",
        hoversubplots="axis",
        paper_bgcolor="#FFF8EC" if point_in_time else "white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": -0.06},
        margin={"l": 40, "r": 20, "t": 60, "b": 30},
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(
        range=[start_ts, end_ts],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )
    for axis in ("xaxis2", "xaxis3"):
        fig.update_layout({f"{axis}": {"rangeslider": {"visible": False}}})
    # Candlestick traces implicitly attach an x-rangeslider, which locks the
    # linked y-axes (fixedrange) even when the slider is hidden — clear it so
    # dragging on a y-axis pans/scales it manually.
    fig.update_yaxes(tickformat=".2f", fixedrange=False)

    # Price-to-scale: each pane's y-range follows its OWN bars (plus the
    # nearby risk/excursion levels on the 1m pane) — a distant HTF zone or
    # pool band clips at the pane edge instead of squashing the candles.
    # The autoscale modebar button still restores the include-everything view.
    execution_prices = [
        meta["risk"].get(name)
        for name in ("entry_price", "stop_price", "target_price")
    ] + [meta["mfe_mae"].get(name) for name in ("mfe_price", "mae_price")]
    meta["y_ranges"] = {}
    for row, timeframe in ((1, 60), (2, parent_tf), (3, htf_tf)):
        y_range = _pane_y_range(
            bars_by_pane.get(timeframe),
            tick_size,
            extra_prices=execution_prices if row == 1 else (),
        )
        if y_range is not None:
            fig.update_yaxes(range=y_range, autorange=False, row=row, col=1)
            meta["y_ranges"][str(row)] = y_range

    meta["shape_count"] = len(fig.layout.shapes)
    meta["omissions"] = list(report.entries)
    fig.update_layout(meta=meta)
    return fig, report


# ── setup-mode figure (additive; the candidate/trade builder above is
# untouched — its figures stay byte-identical) ───────────────────────────────

#: Death/expiry grammar for setup mode: grey/red + dashed. A candidate-less
#: setup must NEVER look executable, so this builder draws no entry/stop/
#: target lines and never uses the entry/trade marker symbols
#: (star / triangle-up / triangle-down) for terminal events.
_SETUP_DEATH_COLOR = "#B0413E"
_SETUP_NEUTRAL_COLOR = "#8C8C8C"


def _in_window(ts: Any, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    if ts is None or pd.isna(ts):
        return False
    return start_ts <= pd.Timestamp(ts) <= end_ts


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or not len(frame):
        return []
    return frame.to_dict("records")


def build_setup_figure(
    *,
    evidence: Any,
    bars_by_pane: dict[int, pd.DataFrame],
    parent_tf: int | None,
    htf_tf: int | None,
    range_bounds: tuple[pd.Timestamp, pd.Timestamp],
    tick_size: float = TICK_SIZE,
) -> tuple[go.Figure, OmissionReport]:
    """Setup-mode three-pane figure (activation→display-end range).

    Consumes the provider's ``SetupEvidence`` (duck-typed) plus pre-sliced bar
    frames. Overlays: fill/death bars (hover carries fill depth + prior/new
    reached extreme), tap-candidate markers with drop reasons, parentless
    interval vrects, and a terminal marker with the death/expiry reason. The
    Q-40 watermark appears iff a 240m pane is shown.
    """
    report = OmissionReport()
    start_ts, end_ts = (pd.Timestamp(range_bounds[0]), pd.Timestamp(range_bounds[1]))
    range_row = evidence.range_row
    candidate_less = bool(range_row.get("candidate_less"))
    gated = evidence.stage is not None and evidence.stage != "terminal"
    meta: dict[str, Any] = {
        "setup_id": evidence.setup_id,
        "mode": "setup",
        "stage": evidence.stage,
        "candidate_less": candidate_less,
        "range": {"start": _iso(start_ts), "end": _iso(end_ts)},
        "taps": [],
        "fills": [],
        "deaths": [],
        "parentless_intervals": [],
        "terminal": None,
    }

    parent_title = (
        f"parent {parent_tf // 60}m" if parent_tf else "parent — none selected"
    )
    if htf_tf is None:
        htf_title = "HTF — unknown"
    elif htf_tf < 3600:
        htf_title = f"HTF {htf_tf // 60}m"
    else:
        htf_title = f"HTF {htf_tf // 3600}H"
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=("1m", parent_title, htf_title),
    )
    pane_rows: tuple[tuple[int, int | None], ...] = ((1, 60), (2, parent_tf), (3, htf_tf))
    for row, timeframe in pane_rows:
        if timeframe is None:
            continue
        bars = bars_by_pane.get(timeframe, pd.DataFrame())
        if not len(bars):
            continue
        fig.add_trace(
            go.Candlestick(
                **_bar_frame_xohlc(bars, tick_size),
                name=f"{timeframe}s",
                showlegend=False,
                increasing_line_color="#59A14F",
                decreasing_line_color="#E45756",
                increasing_line_width=1,
                decreasing_line_width=1,
            ),
            row=row,
            col=1,
        )
    bars_1m = bars_by_pane.get(60, pd.DataFrame())

    # ── tap-candidate markers (drop reasons in hover) ────────────────────────
    taps = [
        row
        for row in _records(getattr(evidence, "tap_candidates", None))
        if _in_window(row.get("envelope_ts_utc"), start_ts, end_ts)
    ]
    taps = _budget(taps, "setup_taps", report, detail="HTF tap candidates")
    tap_x, tap_y, tap_symbols, tap_colors, tap_text = [], [], [], [], []
    for row in taps:
        ts = pd.Timestamp(row["envelope_ts_utc"])
        anchor = _bar_high_price(bars_1m, ts, tick_size)
        if anchor is None:
            continue
        selected = bool(row.get("selected"))
        drop_reason = row.get("drop_reason")
        has_drop = drop_reason is not None and not pd.isna(drop_reason)
        tap_x.append(ts)
        tap_y.append(anchor * 1.0008)
        tap_symbols.append("circle" if selected else "circle-open")
        tap_colors.append("#4C78A8" if selected else _SETUP_NEUTRAL_COLOR)
        tap_text.append(
            f"HTF tap {row.get('fvg_fvg_id')}<br>"
            f"selected={selected}"
            + (f"<br>drop: {drop_reason}" if not selected and has_drop else "")
        )
        meta["taps"].append(
            {
                "ts": _iso(ts),
                "fvg_id": row.get("fvg_fvg_id"),
                "selected": selected,
                "drop_reason": drop_reason if has_drop else None,
            }
        )
    if tap_x:
        fig.add_trace(
            go.Scatter(
                x=tap_x,
                y=tap_y,
                mode="markers",
                marker={"symbol": tap_symbols, "size": 9, "color": tap_colors},
                hovertext=tap_text,
                hoverinfo="text",
                name="HTF tap candidates",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ── fill-event bars (depth + prior/new reached in hover) ─────────────────
    fills = [
        row
        for row in _records(getattr(evidence, "fill_events", None))
        if _in_window(row.get("envelope_ts_utc"), start_ts, end_ts)
    ]
    fills = _budget(fills, "setup_fills", report, detail="fill-event bars")
    fill_x, fill_y, fill_text = [], [], []
    for row in fills:
        ts = pd.Timestamp(row["envelope_ts_utc"])
        kind = str(row.get("event_kind"))
        killing = kind == "filled"
        color = _SETUP_DEATH_COLOR if killing else _SETUP_NEUTRAL_COLOR
        fig.add_vrect(
            x0=max(ts - pd.Timedelta(seconds=60), start_ts),
            x1=ts,
            fillcolor=rgba(color, 0.14 if killing else 0.08),
            line={"color": color, "width": 1, "dash": "dash"},
            layer="below",
            row=1,
            col=1,
        )
        anchor = _bar_high_price(bars_1m, ts, tick_size)
        fill_x.append(ts)
        fill_y.append((anchor or 0) * 1.0004)
        fill_text.append(
            f"fill {kind} · {row.get('fvg_fvg_id')} ({row.get('fvg_role')})<br>"
            f"depth {row.get('fill_depth_ticks')} ticks<br>"
            f"reached {row.get('prior_reached_ticks')} → "
            f"{row.get('new_reached_ticks')} (prior → new extreme)"
        )
        meta["fills"].append(
            {
                "ts": _iso(ts),
                "fvg_id": row.get("fvg_fvg_id"),
                "fill_kind": kind,
                "fill_depth_ticks": row.get("fill_depth_ticks"),
                "prior_reached_ticks": row.get("prior_reached_ticks"),
                "new_reached_ticks": row.get("new_reached_ticks"),
            }
        )
    if fill_x:
        fig.add_trace(
            go.Scatter(
                x=fill_x,
                y=fill_y,
                mode="markers",
                marker={"symbol": "line-ns-open", "size": 10,
                        "color": _SETUP_NEUTRAL_COLOR},
                hovertext=fill_text,
                hoverinfo="text",
                name="fill events",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ── slot-death bars (depth + prior/new + window clocks in hover) ─────────
    deaths = [
        row
        for row in _records(getattr(evidence, "slot_deaths", None))
        if _in_window(row.get("envelope_ts_utc"), start_ts, end_ts)
    ]
    deaths = _budget(deaths, "setup_deaths", report, detail="slot-death events")
    death_x, death_y, death_text = [], [], []
    for row in deaths:
        ts = pd.Timestamp(row["envelope_ts_utc"])
        terminated = bool(row.get("setup_terminated"))
        color = _SETUP_DEATH_COLOR if terminated else _SETUP_NEUTRAL_COLOR
        fig.add_vline(
            x=ts,
            line={"color": color, "width": 1.5, "dash": "dash"},
            row=1,
            col=1,
        )
        anchor = _bar_high_price(bars_1m, ts, tick_size)
        death_x.append(ts)
        death_y.append((anchor or 0) * 1.0012)
        death_text.append(
            f"slot death · {row.get('death_reason')} (phase {row.get('phase')})<br>"
            f"depth {row.get('fill_depth_ticks')} ticks · reached "
            f"{row.get('prior_reached_ticks')} → {row.get('new_reached_ticks')}<br>"
            f"parent clocks {row.get('parent_clocks')}<br>"
            f"remaining window bars {row.get('remaining_window_bars_by_tf')}<br>"
            f"open windows {row.get('open_window_timeframes')}"
        )
        meta["deaths"].append(
            {
                "ts": _iso(ts),
                "death_reason": row.get("death_reason"),
                "phase": row.get("phase"),
                "terminated": terminated,
                "fill_depth_ticks": row.get("fill_depth_ticks"),
                "prior_reached_ticks": row.get("prior_reached_ticks"),
                "new_reached_ticks": row.get("new_reached_ticks"),
            }
        )
    if death_x:
        fig.add_trace(
            go.Scatter(
                x=death_x,
                y=death_y,
                mode="markers",
                marker={"symbol": "x-thin", "size": 11, "color": _SETUP_DEATH_COLOR,
                        "line": {"color": _SETUP_DEATH_COLOR, "width": 2}},
                hovertext=death_text,
                hoverinfo="text",
                name="slot deaths (death/expiry)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # ── parentless-interval vrects ───────────────────────────────────────────
    intervals = [
        row
        for row in _records(getattr(evidence, "parentless_intervals", None))
        if not pd.isna(row.get("start_ts_utc"))
        and not pd.isna(row.get("end_ts_utc"))
        and pd.Timestamp(row["end_ts_utc"]) >= start_ts
        and pd.Timestamp(row["start_ts_utc"]) <= end_ts
    ]
    intervals = _budget(intervals, "setup_intervals", report, detail="parentless intervals")
    for row in intervals:
        interval_start = max(pd.Timestamp(row["start_ts_utc"]), start_ts)
        interval_end = min(pd.Timestamp(row["end_ts_utc"]), end_ts)
        fig.add_vrect(
            x0=interval_start,
            x1=interval_end,
            fillcolor="rgba(140,140,140,0.10)",
            line={"color": _SETUP_NEUTRAL_COLOR, "width": 1, "dash": "dot"},
            layer="below",
            row=1,
            col=1,
        )
        mid = interval_start + (interval_end - interval_start) / 2
        anchor = _bar_high_price(bars_1m, mid, tick_size)
        fig.add_trace(
            go.Scatter(
                x=[mid],
                y=[anchor or 0],
                mode="markers",
                marker={"size": 8, "opacity": 0.01},
                hovertext=(
                    f"parentless interval {row.get('interval_id')}<br>"
                    f"{row.get('bars_count')} counted bars · "
                    f"end: {row.get('end_reason')}"
                ),
                hoverinfo="text",
                showlegend=False,
                name="parentless intervals",
            ),
            row=1,
            col=1,
        )
        meta["parentless_intervals"].append(
            {
                "interval_id": row.get("interval_id"),
                "start": _iso(interval_start),
                "end": _iso(interval_end),
                "bars_count": row.get("bars_count"),
            }
        )

    # ── terminal marker (death/expiry grammar — never an entry/trade look) ───
    terminal_ts = range_row.get("terminal_ts_utc")
    terminal_reason = range_row.get("terminal_reason")
    phase_at_death = range_row.get("phase_at_death")
    if _in_window(terminal_ts, start_ts, end_ts):
        stamp = pd.Timestamp(terminal_ts)
        fig.add_vline(
            x=stamp,
            line={"color": _SETUP_DEATH_COLOR, "width": 2, "dash": "dash"},
            row=1,
            col=1,
        )
        anchor = _bar_high_price(bars_1m, stamp, tick_size)
        fig.add_trace(
            go.Scatter(
                x=[stamp],
                y=[(anchor or 0) * 1.0016],
                mode="markers",
                marker={"symbol": "x-thin", "size": 13, "color": _SETUP_DEATH_COLOR,
                        "line": {"color": _SETUP_DEATH_COLOR, "width": 2}},
                hovertext=(
                    f"setup terminal · {terminal_reason} (phase {phase_at_death})"
                ),
                hoverinfo="text",
                name="setup terminal (death/expiry)",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=stamp,
            y=(anchor or 0) * 1.0024,
            text=f"{terminal_reason} · {phase_at_death}",
            showarrow=True,
            arrowhead=2,
            font={"size": 10, "color": _SETUP_DEATH_COLOR},
            row=1,
            col=1,
        )
    meta["terminal"] = {
        "ts": _iso(terminal_ts) if terminal_ts is not None else None,
        "reason": terminal_reason,
        "phase": phase_at_death,
    }

    # ── Q-40 watermark: only when a 240m pane is shown ───────────────────────
    shown_240m = 14400 in {parent_tf, htf_tf}
    if shown_240m:
        fig.add_annotation(
            xref="x3 domain",
            yref="y3 domain",
            x=0.98,
            y=0.92,
            text=_Q40_TEXT,
            showarrow=False,
            font={"size": 10, "color": "#B25D00"},
            opacity=0.85,
        )
    meta["watermark_240m"] = shown_240m

    flags = " · candidate-less" if candidate_less else ""
    prefix = (
        f"[SETUP{flags} · PIT {evidence.stage}]" if gated else f"[SETUP{flags}]"
    )
    fig.update_layout(
        title=f"{prefix} setup {evidence.setup_id[:12]}…",
        height=720,
        hovermode="x unified",
        hoversubplots="axis",
        paper_bgcolor="#FFF8EC" if gated else "white",
        plot_bgcolor="white",
        legend={"orientation": "h", "y": -0.06},
        margin={"l": 40, "r": 20, "t": 60, "b": 30},
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(
        range=[start_ts, end_ts],
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )
    for axis in ("xaxis2", "xaxis3"):
        fig.update_layout({f"{axis}": {"rangeslider": {"visible": False}}})
    # Candlestick traces implicitly attach an x-rangeslider that locks the
    # linked y-axes even when hidden — clear it (same fix as candidate mode).
    fig.update_yaxes(tickformat=".2f", fixedrange=False)
    meta["y_ranges"] = {}
    for row, timeframe in pane_rows:
        if timeframe is None:
            continue
        y_range = _pane_y_range(bars_by_pane.get(timeframe), tick_size)
        if y_range is not None:
            fig.update_yaxes(range=y_range, autorange=False, row=row, col=1)
            meta["y_ranges"][str(row)] = y_range

    meta["shape_count"] = len(fig.layout.shapes)
    meta["omissions"] = list(report.entries)
    fig.update_layout(meta=meta)
    return fig, report


def figure_geometry_index(fig: go.Figure) -> dict[str, Any]:
    """The numeric index of everything the builder drew (for golden tests)."""
    meta = fig.layout.meta
    if meta is None:
        return {}
    return dict(meta)


def collapse_to_execution_pane(fig):
    """Display-only collapse of a built three-pane figure to the 1m execution
    pane (row 1): parent/HTF-row traces, shapes, and annotations are hidden
    and the execution pane takes the full height. A Q-40 watermark carried by
    the full figure is RE-ANCHORED to the visible pane rather than dropped —
    the experimental-anchor warning is never hidden while any 240m-derived
    content (projected zones) can still be on screen. Geometry meta and the
    omission report are untouched."""
    hidden_axes = {"x2", "x3", "y2", "y3"}

    def _axis_token(ref) -> str:
        return str(ref).split(" ")[0] if ref is not None else ""

    for trace in fig.data:
        if getattr(trace, "xaxis", "x") in ("x2", "x3"):
            trace.visible = False
    kept_shapes = tuple(
        shape
        for shape in (fig.layout.shapes or ())
        if _axis_token(shape.xref) not in hidden_axes
        and _axis_token(shape.yref) not in hidden_axes
    )
    # row-1's vertical span BEFORE re-domaining — paper-referenced
    # annotations (make_subplots places the subplot titles this way) below it
    # belong to the hidden rows.
    row1_domain = fig.layout.yaxis.domain or (0.0, 1.0)
    row1_bottom = float(row1_domain[0])
    kept_annotations = []
    for annotation in fig.layout.annotations or ():
        tokens = {_axis_token(annotation.xref), _axis_token(annotation.yref)}
        is_q40 = "q40" in str(annotation.text or "").lower()
        if not tokens.isdisjoint(hidden_axes):
            if is_q40:
                annotation.update(xref="x domain", yref="y domain", x=0.99, y=0.03)
                kept_annotations.append(annotation)
            continue
        if (
            _axis_token(annotation.yref) == "paper"
            and annotation.y is not None
            and float(annotation.y) < row1_bottom - 1e-9
        ):
            if is_q40:
                annotation.update(xref="x domain", yref="y domain", x=0.99, y=0.03)
                kept_annotations.append(annotation)
            continue
        kept_annotations.append(annotation)
    # direct assignment: update_layout would MERGE the arrays element-wise
    # instead of replacing them, resurrecting the hidden rows' shapes.
    fig.layout.shapes = kept_shapes
    fig.layout.annotations = tuple(kept_annotations)
    fig.update_layout(
        yaxis={"domain": [0.02, 1.0]},
        yaxis2={"visible": False, "domain": [0.0, 0.005]},
        yaxis3={"visible": False, "domain": [0.008, 0.013]},
        xaxis={"showticklabels": True},
        xaxis2={"visible": False, "showticklabels": False},
        xaxis3={"visible": False, "showticklabels": False},
    )
    return fig
