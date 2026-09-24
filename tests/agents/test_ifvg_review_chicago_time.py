"""Trade review and research candidate charts show Chicago time (repair R4)."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


def _review_screen(monkeypatch, tmp_path):
    import ifvg_search_review as ui
    import ifvg_verifier_tab as verifier

    ts = pd.Timestamp("2026-01-13T04:31Z")  # January 12, 2026 10:31 PM CST
    row = dict(trade_id="t", candidate_id="c", decision_id="d", is_warmup=False,
               trading_day="2026-01-13", entry_ts_utc=ts,
               resolution_ts_utc=ts + pd.Timedelta(minutes=10), resolution="stop",
               entry_ticks=103535, stop_ticks=103499, target_ticks=103571, realized_r=-1,
               risk_ticks=36, mfe_ticks=10, mae_ticks=36,
               geometry_parent_timeframe_seconds=300, geometry_htf_timeframe_seconds=3600)
    for key in ("tap_bar_logical_close", "parent_confirmed", "lock_bar_logical_close",
                "opposing_confirmed", "inversion_bar_logical_close"):
        row[f"geometry_{key}_ts_utc"] = ts - pd.Timedelta(minutes=1)
    evidence = SimpleNamespace(reference={}, dataset=SimpleNamespace(
        tables={RecordTable.EXECUTED_TRADE: pd.DataFrame([row])}))
    bars = pd.DataFrame(dict(timeframe_ticks=[60, 60], logical_close_ts_utc=[
        ts, ts + pd.Timedelta(minutes=1)], open_ticks=[1, 1], high_ticks=[2, 2],
        low_ticks=[0, 0], close_ticks=[1, 1]))
    seen: dict = {"labels": [], "frames": [], "figures": []}

    class Screen:
        session_state = {"ifvg_search_review_pending": ("study", "core")}

        def radio(self, _label, _options, **kw):
            return self.session_state[kw["key"]]

        def selectbox(self, label, options, **kw):
            if label == "Trade":
                seen["labels"] = [kw["format_func"](o) for o in options]
            return options[0]

        def checkbox(self, label, **kw):
            return False

        def columns(self, count):
            return [self] * count

        def expander(self, *args, **kw):
            return nullcontext()

        def dataframe(self, data, **kw):
            seen["frames"].append(data)

        def plotly_chart(self, figure, **kw):
            seen["figures"].append(figure)

        def __getattr__(self, name):
            return lambda *args, **kw: None

    monkeypatch.setattr(ui, "list_search_runs",
                        lambda *a: [SimpleNamespace(search_id="study", archived=False)])
    monkeypatch.setattr(ui, "load_studies",
                        lambda *a: ([SimpleNamespace(key="study", name="Named")], []))
    monkeypatch.setattr(ui, "load_search_state", lambda *a: {"children": [
        dict(core_replay_id="core", state="completed", axis_value_ids={})]})
    monkeypatch.setattr(ui, "configuration_name", lambda *a: "Baseline")
    monkeypatch.setattr(ui, "_evidence", lambda *a: evidence)
    monkeypatch.setattr(ui, "_bars", lambda *a: bars)
    monkeypatch.setattr(ui, "list_reviews", lambda **kw: pd.DataFrame())
    monkeypatch.setattr(verifier, "_review_form", lambda *a, **kw: None)
    ui.render_trade_review(Screen(), {k: tmp_path for k in ("repo_root", "store_root",
                                                             "state_root")})
    return seen, ts


def test_trade_selector_lifecycle_and_axis_use_chicago_time(monkeypatch, tmp_path):
    seen, ts = _review_screen(monkeypatch, tmp_path)
    assert seen["labels"] == ["1. Jan 12, 2026 10:31 PM CST · stop · 25,883.75"]
    lifecycle = seen["frames"][0]
    assert lifecycle[-2] == {"Event": "Entry",
                             "Time (Chicago)": "January 12, 2026 10:31:00 PM CST"}
    figure = seen["figures"][0]
    assert figure.layout.xaxis.title.text == "Chicago time (CST/CDT)"
    candle_x = list(figure.data[0].x)
    assert pd.Timestamp(candle_x[0]) == pd.Timestamp("2026-01-12 22:31:00")
    entry_marker = next(s for s in figure.layout.shapes if s.type == "line"
                        and pd.Timestamp(s.x0) == pd.Timestamp("2026-01-12 22:31:00"))
    assert entry_marker is not None  # the marker sits on its candle
    assert ts == pd.Timestamp("2026-01-13T04:31Z")  # stored instant unchanged


def test_research_candidate_chart_is_on_the_chicago_clock():
    import ifvg_research_pipeline as pipeline

    close = pd.to_datetime(["2026-06-05T15:30:00Z", "2026-06-05T15:31:00Z"], utc=True)
    bars = pd.DataFrame({"close_ts_utc": close, "open_ticks": [1, 1], "high_ticks": [2, 2],
                         "low_ticks": [0, 0], "close_ticks": [1, 1]})
    x = pipeline._chicago_walls(bars["close_ts_utc"], naive="utc")
    assert str(x.iloc[0]) == "2026-06-05 10:30:00"  # summer: CDT, not a fixed offset
    assert pipeline._chicago("2026-01-13T23:02:00Z") == "January 13, 2026 5:02 PM CST"
