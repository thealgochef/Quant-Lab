"""R4 display-site regressions: the IFVG verifier and research-review screens
show Chicago time (12-hour clock, CST/CDT named) while every stored instant,
machine field, export column and row order stays exactly as recorded.

Synthetic evidence only; no store, ledger or study is read or written.
"""

from __future__ import annotations

import io
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_research_review as research  # noqa: E402
import ifvg_verifier_charts as charts  # noqa: E402
import ifvg_verifier_tab as tab  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import (  # noqa: E402
    FsmStage,
    StageGate,
    ZoneEvidence,
)

_CHICAGO_AXIS_TITLE = "Chicago time (CST/CDT)"


def _utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_convert("UTC")


def _wall(text: str) -> pd.Timestamp:
    """A naive Chicago wall-clock expectation, written out by hand."""
    return pd.Timestamp(text)


# ── synthetic candidate evidence ─────────────────────────────────────────────


def _bars_1m(start: pd.Timestamp, minutes: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "bar_id": f"60s:{start.date()}:{index}",
                "close_ts_utc": start + pd.Timedelta(minutes=index + 1),
                "open_ticks": 103000 + index,
                "high_ticks": 103050 + index,
                "low_ticks": 102950 + index,
                "close_ticks": 103020 + index,
                "volume": 10,
                "trade_count": 4,
            }
            for index in range(minutes)
        ]
    )


def _bars_tf(start: pd.Timestamp, timeframe: int) -> pd.DataFrame:
    rows = []
    for index in range(max(1, (120 * 60) // timeframe)):
        open_ts = start + pd.Timedelta(seconds=timeframe * index)
        rows.append(
            {
                "bar_id": f"{timeframe}s:{start.date()}:{index}",
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


def _gate(stage: FsmStage, ts: pd.Timestamp, ordinal: int) -> StageGate:
    return StageGate(
        stage=stage,
        ts_utc=ts,
        event_cursor=f"{ts.isoformat()}|60|x",
        trace_ordinal=ordinal,
        source_event_id=f"{stage.value}-event",
        source_kind=f"lifecycle:{stage.value}",
    )


def _evidence(start: pd.Timestamp, *, mode: str = "full_audit", stage=None):
    def at(minutes: int) -> pd.Timestamp:
        return start + pd.Timedelta(minutes=minutes)

    zone = ZoneEvidence(
        role="parent",
        fvg_id="parent-zone",
        timeframe_seconds=300,
        direction="bullish",
        gap_low_ticks=102980,
        gap_high_ticks=103010,
        size_ticks=30,
        a_bar_id="300s:x:5",
        c_bar_id="300s:x:7",
        a_open_ts_utc=at(-600),
        confirmed_ts_utc=at(20),
        trading_day=str(start.date()),
    )
    return SimpleNamespace(
        candidate_id="cand-1234567890ab",
        mode=mode,
        stage=stage,
        zones=(zone,),
        stage_gates={
            "tap": _gate(FsmStage.TAP, at(5), 1),
            "entry": _gate(FsmStage.ENTRY, at(60), 5),
            "resolution": _gate(FsmStage.RESOLUTION, at(90), 6),
        },
        risk={
            "direction": "LONG",
            "entry_ticks": 103020,
            "stop_ticks": 102984,
            "target_ticks": 103056,
        },
        execution=None,
        counterfactual_labels=(),
        structure=pd.DataFrame(),
        displacement=pd.DataFrame(),
        pools=pd.DataFrame(),
        pool_members=pd.DataFrame(),
        sweep_links=pd.DataFrame(),
        sessions={"entry_session": "ny", "schemes": {}},
        model={},
        range_row={
            "entry_anchor_ts": at(60),
            "display_end_ts": at(90),
            "parent_timeframe_seconds": 300,
            "htf_timeframe_seconds": 3600,
        },
    )


def _candidate_figure(start: pd.Timestamp):
    evidence = _evidence(start)
    bars = {60: _bars_1m(start), 300: _bars_tf(start, 300), 3600: _bars_tf(start, 3600)}
    fig, _report = charts.build_verifier_figure(
        evidence=evidence,
        bars_by_pane=bars,
        parent_tf=300,
        htf_tf=3600,
        range_bounds=(start, start + pd.Timedelta(minutes=120)),
    )
    return evidence, bars, charts.to_display_timezone(fig)


# ── verifier chart: instants converted, anchoring kept ───────────────────────


@pytest.mark.parametrize(
    ("start_text", "entry_wall", "range_start_wall", "entry_label"),
    [
        # winter: CST = UTC-6
        ("2026-01-07T10:00:00Z", "2026-01-07T05:00:00", "2026-01-07T04:00:00",
         "Jan 7, 2026 5:00 AM CST"),
        # summer: CDT = UTC-5
        ("2026-06-05T13:00:00Z", "2026-06-05T09:00:00", "2026-06-05T08:00:00",
         "Jun 5, 2026 9:00 AM CDT"),
    ],
)
def test_verifier_chart_shows_chicago_wall_time_and_marker_stays_on_its_candle(
    start_text: str, entry_wall: str, range_start_wall: str, entry_label: str
) -> None:
    start = _utc(start_text)
    evidence, bars, fig = _candidate_figure(start)
    entry_ts = evidence.stage_gates["entry"].ts_utc

    candle = next(t for t in fig.data if t.type == "candlestick" and t.name == "60s")
    candle_x = [pd.Timestamp(value) for value in candle.x]
    # the bar whose close is the entry gate (the builder's anchor bar)
    closes = list(bars[60]["close_ts_utc"])
    anchor_index = closes.index(entry_ts)
    assert candle_x[anchor_index] == _wall(entry_wall) - pd.Timedelta(seconds=60)
    assert candle_x[anchor_index].tzinfo is None

    markers = next(t for t in fig.data if t.name == "FSM stages")
    stage_index = list(evidence.stage_gates).index("entry")
    marker_x = pd.Timestamp(markers.x[stage_index])
    assert marker_x == _wall(entry_wall)
    # same conversion for candle and marker: still one bar apart, on the bar high
    assert marker_x - candle_x[anchor_index] == pd.Timedelta(seconds=60)
    high = float(bars[60]["high_ticks"].iloc[anchor_index]) * charts.TICK_SIZE
    assert markers.y[stage_index] == pytest.approx(high * 1.0006)

    # hover text names the Chicago zone instead of a raw UTC ISO instant
    hover = markers.hovertext[stage_index]
    assert entry_label in hover
    assert "+00:00" not in hover and entry_ts.isoformat() not in hover
    zone_hover = next(t for t in fig.data if t.name == "zones").hovertext[0]
    assert "confirmed " in zone_hover and ("CST" in zone_hover or "CDT" in zone_hover)
    assert "+00:00" not in zone_hover

    # axis range, 12-hour ticks/hover and the zone title
    assert pd.Timestamp(fig.layout.xaxis.range[0]) == _wall(range_start_wall)
    for axis_name in ("xaxis", "xaxis2", "xaxis3"):
        axis = getattr(fig.layout, axis_name)
        assert "%p" in axis.tickformat and "%-I" in axis.tickformat
        assert "%p" in axis.hoverformat
    assert fig.layout.xaxis3.title.text == _CHICAGO_AXIS_TITLE

    # machine geometry (ISO UTC) is unchanged by the display conversion
    meta = charts.figure_geometry_index(fig)
    assert meta["stage_markers"]["entry"] == entry_ts.isoformat()
    assert meta["risk"]["entry_ts"] == entry_ts.isoformat()


def test_collapsed_verifier_chart_titles_the_visible_axis() -> None:
    start = _utc("2026-01-07T10:00:00Z")
    evidence = _evidence(start)
    fig, _report = charts.build_verifier_figure(
        evidence=evidence,
        bars_by_pane={60: _bars_1m(start), 300: _bars_tf(start, 300),
                      3600: _bars_tf(start, 3600)},
        parent_tf=300,
        htf_tf=3600,
        range_bounds=(start, start + pd.Timedelta(minutes=120)),
    )
    out = charts.to_display_timezone(charts.collapse_to_execution_pane(fig))
    assert out.layout.xaxis.title.text == _CHICAGO_AXIS_TITLE


def test_display_time_names_the_zone_and_keeps_precision() -> None:
    # fall-back day: the repeated 1:30 AM hour is told apart by CDT/CST
    first = charts.display_time(_utc("2026-11-01T06:30:00Z"))
    second = charts.display_time(_utc("2026-11-01T07:30:00Z"))
    assert first == "Nov 1, 2026 1:30 AM CDT"
    assert second == "Nov 1, 2026 1:30 AM CST"
    # spring-forward boundary
    assert charts.display_time(_utc("2026-03-08T07:59:00Z")) == "Mar 8, 2026 1:59 AM CST"
    assert charts.display_time(_utc("2026-03-08T08:00:00Z")) == "Mar 8, 2026 3:00 AM CDT"
    # midnight and noon on a 12-hour clock
    assert charts.display_time(_utc("2026-01-07T06:00:00Z")) == "Jan 7, 2026 12:00 AM CST"
    assert charts.display_time(_utc("2026-01-07T18:00:00Z")) == "Jan 7, 2026 12:00 PM CST"
    # seconds and fractions are shown only when the instant has them
    assert (
        charts.display_time(pd.Timestamp("2026-01-07T10:00:05.123456789Z"))
        == "Jan 7, 2026 4:00:05.123456789 AM CST"
    )
    # a zone-less value is flagged, unless the caller names the UTC convention
    assert "time zone not recorded" in charts.display_time("2026-01-07 10:00:00")
    assert (
        charts.display_time("2026-01-07 10:00:00", naive="utc") == "Jan 7, 2026 4:00 AM CST"
    )
    assert charts.display_time(None) == "—"


def test_account_timeline_axis_is_chicago_wall_time() -> None:
    from ifvg_results_charts import build_account_timeline_figure

    figure, _omissions = build_account_timeline_figure(
        [
            {
                "event_type": "daily_halt",
                "event_ts_utc": "2026-01-07T14:00:00+00:00",
                "event_ordinal": 1,
                "event_id": "eh1",
                "payload": {"threshold_value": 49_100.0},
            }
        ]
    )
    halt = next(t for t in figure.data if t.name == "daily-loss threshold")
    assert pd.Timestamp(halt.x[0]) == _wall("2026-01-07T08:00:00")  # 8:00 AM CST
    assert figure.layout.xaxis.title.text == _CHICAGO_AXIS_TITLE
    assert "%p" in figure.layout.xaxis.hoverformat


# ── verifier tab: selector label, captions, saved-review time ───────────────


def test_setup_selector_label_is_chicago_time(monkeypatch) -> None:
    monkeypatch.setattr(tab, "technical_details_enabled", lambda: False)
    aware = pd.Series({"activation_ts_utc": _utc("2026-01-13T23:02:00Z")})
    naive = pd.Series({"activation_ts_utc": pd.Timestamp("2026-06-05T15:30:00")})
    assert tab._setup_label(aware) == "Setup activated Jan 13, 2026 5:02 PM CST"
    # the column is UTC by definition, so a zone-less value follows that convention
    assert tab._setup_label(naive) == "Setup activated Jun 5, 2026 10:30 AM CDT"


def test_chart_captions_no_longer_claim_eastern_or_utc_hover() -> None:
    source = (_SCRIPTS / "ifvg_verifier_tab.py").read_text(encoding="utf-8")
    assert "Eastern (" not in source
    assert "Hover ISO timestamps remain UTC" not in source
    assert "Chicago time (CST/CDT)" in tab._CHART_TIME_CAPTION
    assert source.count("st_module.caption(_CHART_TIME_CAPTION)") == 2


_REVIEWED_AT = "2026-09-08T13:54:03.359391+00:00"


def _review_app() -> None:
    from types import SimpleNamespace

    import ifvg_verifier_tab
    import pandas as pd
    import streamlit as st

    ctx = SimpleNamespace(
        replay=SimpleNamespace(artifact_id="c" * 64),
        pair_ref=SimpleNamespace(as_dict=lambda: {"profile_name": "profile"}),
    )
    evidence = SimpleNamespace(candidate_id="cand-000000000001")
    row = pd.Series({"decision_id": "dec-1", "trade_id": None})
    ifvg_verifier_tab._render_review_section(st, ctx, evidence, row)


def test_saved_review_time_is_shown_in_chicago_and_the_ledger_is_unchanged(
    monkeypatch,
) -> None:
    ledger = pd.DataFrame(
        [
            {
                "reviewed_at": _REVIEWED_AT,
                "reviewer": "owner",
                "overall_verdict": "correct",
                "tags": ["late_entry"],
                "notes": "checked",
            },
            {
                "reviewed_at": "2026-01-13T23:02:00+00:00",
                "reviewer": "owner",
                "overall_verdict": "questionable",
                "tags": [],
                "notes": "winter",
            },
        ]
    )
    before = ledger.copy(deep=True)
    monkeypatch.setattr(tab, "list_reviews", lambda **kwargs: ledger)
    monkeypatch.setattr(tab, "append_review", lambda **kwargs: kwargs)
    monkeypatch.setattr(tab, "export_csv", lambda **kwargs: "")
    at = apptest.AppTest.from_function(_review_app, default_timeout=60)
    at.run()
    assert not at.exception
    shown = at.dataframe[0].value
    # the stored order is kept; only the screen text changes
    assert list(shown["Reviewed at"]) == [
        "Sep 8, 2026 8:54 AM CDT",
        "Jan 13, 2026 5:02 PM CST",
    ]
    assert _REVIEWED_AT not in shown.to_string()
    pd.testing.assert_frame_equal(ledger, before)


# ── research review: lifecycle tables and chart data table ───────────────────


class _Screen:
    def __init__(self) -> None:
        self.frames: list = []
        self.downloads: list = []

    def subheader(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def caption(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def checkbox(self, *args, **kwargs):
        return True

    def expander(self, *args, **kwargs):
        return nullcontext()

    def selectbox(self, label, options, **kwargs):
        return list(options)[0]

    def dataframe(self, data, *args, **kwargs):
        self.frames.append(pd.DataFrame(data))

    def download_button(self, label, data, **kwargs):
        self.downloads.append(data)


def _stub_review(monkeypatch) -> None:
    monkeypatch.setattr(tab, "_render_review_section", lambda *args, **kwargs: None)
    monkeypatch.setattr(tab, "_render_setup_review_section", lambda *args, **kwargs: None)


def test_candidate_lifecycle_table_shows_chicago_and_keeps_withholding(monkeypatch) -> None:
    _stub_review(monkeypatch)
    start = _utc("2026-01-07T10:00:00Z")
    full = _evidence(start)
    screen = _Screen()
    research.candidate_panel(screen, None, full, pd.Series({"blocked": False}))
    table = screen.frames[0]
    assert list(table.columns) == ["Event", "Time (Chicago)"]
    assert list(table["Time (Chicago)"]) == [
        "Jan 7, 2026 4:05 AM CST",
        "Jan 7, 2026 5:00 AM CST",
        "Jan 7, 2026 5:30 AM CST",
    ]
    # point-in-time: later stages stay withheld exactly as before
    pit = _evidence(start, mode="point_in_time", stage="tap")
    screen = _Screen()
    research.candidate_panel(screen, None, pit, pd.Series({"blocked": False}))
    assert list(screen.frames[0]["Time (Chicago)"]) == [
        "Jan 7, 2026 4:05 AM CST",
        "Withheld",
        "Withheld",
    ]


def test_setup_lifecycle_events_show_chicago_in_stored_order(monkeypatch) -> None:
    _stub_review(monkeypatch)
    # stored (instant) order across the fall-back hour: text order would differ
    events = pd.DataFrame(
        {
            "event_kind": ["tap_candidate", "fill"],
            "stage": ["S1", "S2"],
            "ts_utc": [_utc("2026-11-01T06:30:00Z"), _utc("2026-11-01T07:10:00Z")],
        }
    )
    evidence = SimpleNamespace(stage="terminal", events=events)
    row = pd.Series(
        {
            "terminal_reason": "expired",
            "candidate_less": True,
            "candidate_count": 0,
            "candidate_ids": "[]",
        }
    )
    screen = _Screen()
    research.setup_panel(screen, None, None, evidence, row)
    table = screen.frames[0]
    assert "Time (Chicago)" in table.columns and "Ts utc" not in table.columns
    assert list(table["Time (Chicago)"]) == [
        "Nov 1, 2026 1:30 AM CDT",
        "Nov 1, 2026 1:10 AM CST",
    ]
    assert events["ts_utc"].iloc[0] == _utc("2026-11-01T06:30:00Z")  # input untouched


def test_bar_table_shows_chicago_and_the_csv_keeps_utc() -> None:
    start = _utc("2026-01-07T10:00:00Z")
    bars = _bars_1m(start, minutes=3)
    screen = _Screen()
    research.bar_table(screen, {60: bars}, key="bars")
    table = screen.frames[0]
    assert table.columns[0] == "Bar close (Chicago)"
    assert list(table["Bar close (Chicago)"]) == [
        "Jan 7, 2026 4:01 AM CST",
        "Jan 7, 2026 4:02 AM CST",
        "Jan 7, 2026 4:03 AM CST",
    ]
    assert "Close ts utc" not in table.columns
    exported = pd.read_csv(io.StringIO(screen.downloads[0]))
    # the machine column keeps its UTC instants and position; the label is appended
    assert list(exported.columns)[0] == "Close ts utc"
    assert list(exported.columns)[-1] == "Bar close (Chicago)"
    assert [pd.Timestamp(value) for value in exported["Close ts utc"]] == list(
        bars["close_ts_utc"]
    )
