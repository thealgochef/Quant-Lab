"""Tests for the pure setup-mode chart builder (synthetic evidence only).

The candidate/trade builder is exercised by ``test_ifvg_verifier_charts.py``
and stays untouched; these tests cover ``build_setup_figure`` — the
death/expiry styling grammar (a candidate-less setup must never look
executable), the fill/death/tap/parentless overlays, the Q-40 watermark rule,
and the budget → omission-report degradation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_verifier_charts as charts  # noqa: E402


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


_START = _ts("2026-01-07T10:00:00Z")
_END = _ts("2026-01-07T11:40:00Z")

#: Marker symbols reserved for executable candidate/trade evidence — the
#: setup builder must never emit them (styling-grammar contract).
_EXECUTABLE_SYMBOLS = {"triangle-up", "triangle-down", "star"}
_EXECUTABLE_NAMES = {
    "entry",
    "stop",
    "target (1R)",
    "actual entry",
    "counterfactual entry",
}


def _bars_1m(minutes: int = 100) -> pd.DataFrame:
    rows = []
    for index in range(minutes):
        close = _START + pd.Timedelta(minutes=index + 1)
        rows.append(
            {
                "bar_id": f"60s:2026-01-07:{index}",
                "close_ts_utc": close,
                "open_ticks": 103000 + index,
                "high_ticks": 103050 + index,
                "low_ticks": 102950 + index,
                "close_ticks": 103020 + index,
                "volume": 10,
                "trade_count": 4,
            }
        )
    return pd.DataFrame(rows)


def _bars_tf(timeframe: int) -> pd.DataFrame:
    rows = []
    count = max(1, (100 * 60) // timeframe)
    for index in range(count):
        open_ts = _START + pd.Timedelta(seconds=timeframe * index)
        rows.append(
            {
                "bar_id": f"{timeframe}s:2026-01-07:{index}",
                "logical_open_ts_utc": open_ts,
                "logical_close_ts_utc": open_ts + pd.Timedelta(seconds=timeframe),
                "open_ticks": 103000,
                "high_ticks": 103100,
                "low_ticks": 102900,
                "close_ticks": 103040,
                "volume": 100,
                "trade_count": 40,
            }
        )
    return pd.DataFrame(rows)


class _SetupEvidence:
    """Minimal duck-typed SetupEvidence for the pure builder."""

    def __init__(
        self,
        *,
        candidate_less: bool = True,
        stage: str | None = None,
        htf_tf: int = 14400,
        parent_tf: int | None = 300,
    ) -> None:
        self.setup_id = "setup-1234567890abcdef"
        self.stage = stage
        self.range_row = {
            "setup_id": self.setup_id,
            "activation_ts_utc": _START,
            "terminal_ts_utc": _ts("2026-01-07T11:30:00Z"),
            "display_end_ts_utc": _END,
            "terminal_reason": "invalidated_htf_filled",
            "phase_at_death": "S1",
            "candidate_less": candidate_less,
            "candidate_count": 0 if candidate_less else 1,
            "parentless_interval_count": 1,
            "htf_tf_seconds": htf_tf,
            "parent_tf_seconds": parent_tf,
        }
        self.tap_candidates = pd.DataFrame(
            [
                {
                    "envelope_ts_utc": _ts("2026-01-07T10:05:00Z"),
                    "fvg_fvg_id": f"{htf_tf}s:2026-01-06:1",
                    "selected": True,
                    "drop_reason": None,
                },
                {
                    "envelope_ts_utc": _ts("2026-01-07T10:06:00Z"),
                    "fvg_fvg_id": f"{htf_tf}s:2026-01-06:2",
                    "selected": False,
                    "drop_reason": "rank_below_cap",
                },
            ]
        )
        self.fill_events = pd.DataFrame(
            [
                {
                    "envelope_ts_utc": _ts("2026-01-07T10:45:00Z"),
                    "fvg_fvg_id": f"{htf_tf}s:2026-01-06:1",
                    "fvg_role": "htf",
                    "event_kind": "touched",
                    "fill_depth_ticks": 12,
                    "prior_reached_ticks": 103000,
                    "new_reached_ticks": 103012,
                },
                {
                    "envelope_ts_utc": _ts("2026-01-07T11:30:00Z"),
                    "fvg_fvg_id": f"{htf_tf}s:2026-01-06:1",
                    "fvg_role": "htf",
                    "event_kind": "filled",
                    "fill_depth_ticks": 30,
                    "prior_reached_ticks": 103012,
                    "new_reached_ticks": 103042,
                },
            ]
        )
        self.slot_deaths = pd.DataFrame(
            [
                {
                    "envelope_ts_utc": _ts("2026-01-07T11:30:00Z"),
                    "death_ts_utc": _ts("2026-01-07T11:30:00Z"),
                    "death_reason": "invalidated_htf_filled",
                    "phase": "S1",
                    "setup_terminated": True,
                    "fill_depth_ticks": 30,
                    "prior_reached_ticks": 103012,
                    "new_reached_ticks": 103042,
                    "parent_clocks": "{}",
                    "remaining_window_bars_by_tf": "{}",
                    "open_window_timeframes": "[]",
                }
            ]
        )
        self.parentless_intervals = pd.DataFrame(
            [
                {
                    "interval_id": "interval-1",
                    "start_ts_utc": _ts("2026-01-07T10:30:00Z"),
                    "end_ts_utc": _ts("2026-01-07T10:50:00Z"),
                    "bars_count": 20,
                    "end_reason": "successor_selected",
                }
            ]
        )
        self.terminal = self.slot_deaths.iloc[0].to_dict()
        self.gating_report = {"stage": stage, "hidden_events": 0, "total_events": 5}


def _build(evidence: _SetupEvidence, htf_tf: int = 14400, parent_tf: int | None = 300):
    bars = {60: _bars_1m()}
    if parent_tf:
        bars[parent_tf] = _bars_tf(parent_tf)
    bars[htf_tf] = _bars_tf(htf_tf)
    return charts.build_setup_figure(
        evidence=evidence,
        bars_by_pane=bars,
        parent_tf=parent_tf,
        htf_tf=htf_tf,
        range_bounds=(_START, _END),
    )


class TestGeometryMeta:
    def test_meta_matches_evidence(self) -> None:
        evidence = _SetupEvidence()
        fig, _report = _build(evidence)
        index = charts.figure_geometry_index(fig)
        assert index["setup_id"] == evidence.setup_id
        assert index["mode"] == "setup"
        assert index["candidate_less"] is True
        assert index["terminal"]["reason"] == "invalidated_htf_filled"
        assert index["terminal"]["phase"] == "S1"
        assert index["terminal"]["ts"] == "2026-01-07T11:30:00+00:00"
        fills = {entry["fill_kind"]: entry for entry in index["fills"]}
        assert fills["filled"]["fill_depth_ticks"] == 30
        assert fills["filled"]["prior_reached_ticks"] == 103012
        assert fills["filled"]["new_reached_ticks"] == 103042
        assert index["deaths"][0]["death_reason"] == "invalidated_htf_filled"
        assert index["parentless_intervals"][0]["interval_id"] == "interval-1"
        assert index["parentless_intervals"][0]["bars_count"] == 20

    def test_stage_gate_metadata_and_pit_tint(self) -> None:
        evidence = _SetupEvidence(stage="activation")
        fig, _report = _build(evidence)
        assert charts.figure_geometry_index(fig)["stage"] == "activation"
        assert "PIT activation" in fig.layout.title.text
        assert fig.layout.paper_bgcolor == "#FFF8EC"
        full = _SetupEvidence(stage="terminal")
        fig2, _ = _build(full)
        assert "PIT" not in fig2.layout.title.text
        assert fig2.layout.paper_bgcolor == "white"


class TestStylingGrammar:
    def test_candidate_less_setup_never_looks_executable(self) -> None:
        evidence = _SetupEvidence(candidate_less=True)
        fig, _report = _build(evidence)
        assert "candidate-less" in fig.layout.title.text
        names = {trace.name for trace in fig.data if trace.name}
        assert not names & _EXECUTABLE_NAMES
        for trace in fig.data:
            marker = getattr(trace, "marker", None)
            symbol = getattr(marker, "symbol", None)
            symbols = (
                set(symbol)
                if isinstance(symbol, (tuple, list))
                else {symbol}
                if symbol
                else set()
            )
            assert not symbols & _EXECUTABLE_SYMBOLS, trace.name

    def test_terminal_marker_uses_death_expiry_styling(self) -> None:
        evidence = _SetupEvidence()
        fig, _report = _build(evidence)
        terminal = [t for t in fig.data if t.name == "setup terminal (death/expiry)"]
        assert terminal, "terminal marker must be drawn"
        assert terminal[0].marker.symbol == "x-thin"
        assert terminal[0].marker.color == charts._SETUP_DEATH_COLOR
        dashed_lines = [
            shape
            for shape in fig.layout.shapes
            if shape.type == "line" and shape.line.dash == "dash"
        ]
        assert dashed_lines, "terminal/death vlines must be dashed"
        texts = [a.text for a in fig.layout.annotations if a.text]
        assert any("invalidated_htf_filled" in text for text in texts)

    def test_candidate_full_setup_title_has_no_candidate_less_flag(self) -> None:
        evidence = _SetupEvidence(candidate_less=False)
        fig, _report = _build(evidence)
        assert "candidate-less" not in fig.layout.title.text
        assert charts.figure_geometry_index(fig)["candidate_less"] is False


class TestOverlays:
    def test_tap_markers_carry_drop_reasons(self) -> None:
        evidence = _SetupEvidence()
        fig, _report = _build(evidence)
        taps = [t for t in fig.data if t.name == "HTF tap candidates"]
        assert taps
        hover = "".join(taps[0].hovertext)
        assert "rank_below_cap" in hover
        index = charts.figure_geometry_index(fig)
        assert index["taps"][1]["drop_reason"] == "rank_below_cap"
        assert index["taps"][0]["selected"] is True

    def test_fill_hover_shows_depth_and_prior_new_extreme(self) -> None:
        evidence = _SetupEvidence()
        fig, _report = _build(evidence)
        fills = [t for t in fig.data if t.name == "fill events"]
        assert fills
        hover = "".join(fills[0].hovertext)
        assert "depth 30 ticks" in hover
        assert "103012 → 103042" in hover

    def test_parentless_interval_vrect_present(self) -> None:
        evidence = _SetupEvidence()
        fig, _report = _build(evidence)
        hover_traces = [t for t in fig.data if t.name == "parentless intervals"]
        assert len(hover_traces) == 1
        assert "interval-1" in "".join(hover_traces[0].hovertext)
        dotted_rects = [
            shape
            for shape in fig.layout.shapes
            if shape.type == "rect" and shape.line.dash == "dot"
        ]
        assert dotted_rects, "parentless interval must render as a vrect"


class TestWatermark:
    def test_watermark_present_iff_240m_pane(self) -> None:
        with_240 = _SetupEvidence(htf_tf=14400)
        fig, _ = _build(with_240, htf_tf=14400)
        texts = [a.text for a in fig.layout.annotations]
        assert any("experimental_q40_open" in (t or "") for t in texts)
        assert charts.figure_geometry_index(fig)["watermark_240m"] is True

        without_240 = _SetupEvidence(htf_tf=3600)
        fig2, _ = _build(without_240, htf_tf=3600)
        texts2 = [a.text for a in fig2.layout.annotations]
        assert not any("experimental_q40_open" in (t or "") for t in texts2)
        assert charts.figure_geometry_index(fig2)["watermark_240m"] is False


class TestBudgetsAndDegradation:
    def test_fill_budget_excess_reports_omission_not_exception(self) -> None:
        evidence = _SetupEvidence()
        evidence.fill_events = pd.DataFrame(
            [
                {
                    "envelope_ts_utc": _START + pd.Timedelta(minutes=index),
                    "fvg_fvg_id": f"14400s:2026-01-06:{index}",
                    "fvg_role": "htf",
                    "event_kind": "touched",
                    "fill_depth_ticks": index,
                    "prior_reached_ticks": 103000,
                    "new_reached_ticks": 103000 + index,
                }
                for index in range(100)
            ]
        )
        fig, report = _build(evidence)
        assert fig is not None
        entries = [e for e in report.entries if e["layer"] == "setup_fills"]
        assert entries and entries[0]["omitted"] == 100 - charts.LAYER_BUDGETS[
            "setup_fills"
        ]

    def test_missing_parent_pane_degrades_gracefully(self) -> None:
        evidence = _SetupEvidence(parent_tf=None)
        fig, _report = _build(evidence, parent_tf=None)
        assert fig is not None
        titles = [a.text for a in fig.layout.annotations if a.text]
        assert any("none selected" in (t or "") for t in titles)


def test_collapse_to_execution_pane_hides_tf_panes_and_keeps_q40() -> None:
    """Display-only collapse: parent/HTF-row traces, shapes, and subplot
    titles are hidden, the 1m pane takes the full height, and a Q-40
    watermark is re-anchored to the visible pane instead of dropped."""
    import plotly.graph_objects as go
    from ifvg_verifier_charts import collapse_to_execution_pane
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.24, 0.24],
        subplot_titles=("1m execution", "parent 10m", "HTF 4H"),
    )
    for row in (1, 2, 3):
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], name=f"row{row}"), row=row, col=1)
    fig.add_shape(
        type="rect", x0=1, x1=2, y0=1, y1=2, xref="x2", yref="y2"
    )
    fig.add_shape(
        type="rect", x0=1, x1=2, y0=1, y1=2, xref="x", yref="y"
    )
    fig.add_annotation(
        text="experimental_q40_open — not canonical context",
        xref="x3 domain",
        yref="y3 domain",
        x=0.99,
        y=0.98,
        showarrow=False,
    )
    collapsed = collapse_to_execution_pane(fig)

    visibility = {
        trace.name: trace.visible for trace in collapsed.data
    }
    assert visibility["row1"] is None or visibility["row1"] is True
    assert visibility["row2"] is False and visibility["row3"] is False
    # only the row-1 shape survives.
    assert len(collapsed.layout.shapes) == 1
    assert str(collapsed.layout.shapes[0].xref).startswith("x")
    assert "2" not in str(collapsed.layout.shapes[0].xref)
    # subplot titles for rows 2/3 are gone; the q40 watermark survives,
    # re-anchored onto the visible pane.
    texts = [str(a.text) for a in collapsed.layout.annotations]
    assert "parent 10m" not in texts and "HTF 4H" not in texts
    assert any("q40" in text.lower() for text in texts)
    q40 = next(a for a in collapsed.layout.annotations if "q40" in str(a.text).lower())
    assert q40.xref == "x domain" and q40.yref == "y domain"
    # the execution pane takes the full height; tf-pane axes are hidden.
    assert collapsed.layout.yaxis.domain[1] == 1.0
    assert collapsed.layout.yaxis.domain[0] <= 0.05
    assert collapsed.layout.yaxis2.visible is False
    assert collapsed.layout.yaxis3.visible is False
