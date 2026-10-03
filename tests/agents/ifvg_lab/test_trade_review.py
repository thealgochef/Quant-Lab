"""Trade review (mocks 09, 09b): funded trades on stored bars, point in time, review form.

Pure tests use a synthetic trade and synthetic one-minute bars. The reference
tests read the saved funded variation study, its verified strategy package and
stored bars on this computer (read only; skipped where absent). Every review
saved here goes to pytest's ``tmp_path`` — never the real review ledger.
"""

from __future__ import annotations

import contextlib
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_chart as rc
from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_panels as rp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
TPT = "takeprofittrader"
HAVE_STUDY = (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file()

ROW = {
    "seq": 380, "account_number": 6, "direction": "long", "quantity": 10,
    "entry_utc": "2026-04-13T00:07:00Z", "entry_ticks": 99884, "stop_ticks": 99799,
    "target_ticks": 99969, "scale_out_ns": 1776039027251840803, "scale_out_ticks": 99969,
    "scale_out_quantity": 5, "final_exit_quantity": 5, "final_stop_ticks": 99884,
    "exit_utc": "2026-04-13T20:55:00Z", "exit_ticks": 102305, "exit_kind": "scheduled_close",
    "net_pnl_usd": 6254.72, "costs_usd": 10.28, "balance_before_usd": 1821.10,
    "balance_after_usd": 8075.82, "initial_risk_usd": 425.0, "trading_day": "2026-04-13",
    "minutes_on_prints": 1247, "minutes_approximated": 1, "approximate_exit": False,
    "account_failed": False,
}


def _minutes(start="2026-04-12T22:00:00Z", count=1380, base=24950.0) -> pd.DataFrame:
    opens = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    price = [base + (i % 37) * 0.25 + i * 0.5 for i in range(count)]
    return pd.DataFrame({
        "trading_day": "2026-04-13", "logical_open_ts_utc": opens,
        "logical_close_ts_utc": opens + pd.Timedelta(minutes=1), "open": price,
        "high": [p + 2 for p in price], "low": [p - 2 for p in price],
        "close": [p + 1 for p in price]})


# ── vocabulary ────────────────────────────────────────────────────────────


def test_overall_choices_are_the_ledger_verdicts_in_order():
    from alpha_lab.agents.data_infra.ifvg.presentation.review_vocabulary import VERDICT_LABELS

    assert [k for k, _ in rp.OVERALL_CHOICES] == list(VERDICT_LABELS)
    assert dict(rp.OVERALL_CHOICES)["correct"] == "Correct — matches how I'd trade it"
    assert dict(rp.OVERALL_CHOICES)["questionable"] == VERDICT_LABELS["questionable"]


def test_step_by_step_maps_onto_existing_verdict_fields_and_not_reviewed_saves_nothing():
    verdicts = rp.review_verdicts("correct", {
        "htf": "Agree", "Parent gap and retest are right": "Not reviewed",
        "Entry and stop are right": "Disagree", "Exit handling is right": "Unclear"})
    assert verdicts == {"overall_verdict": "correct", "htf_verdict": "correct",
                        "entry_verdict": "incorrect", "stop_verdict": "incorrect",
                        "outcome_verdict": "insufficient_evidence"}
    assert "parent_verdict" not in verdicts


def test_every_tag_is_a_ledger_tag_and_the_ledger_accepts_the_new_ones(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.visual_review_store import REVIEW_TAGS, append_review

    assert {key for _, key in rp.TAGS} <= set(REVIEW_TAGS)
    assert REVIEW_TAGS[:15][0] == "wrong_htf_fvg" and "fill_depth_disputed" in REVIEW_TAGS
    record = append_review(repo_root=tmp_path, replay_chart_artifact_id="c", pair_ref={},
                           candidate_id="funded:x", decision_id=None, trade_id="t",
                           reviewer="Test", verdicts={"overall_verdict": "correct"},
                           tags=[key for _, key in rp.TAGS], notes="")
    assert record["tags"] == sorted(key for _, key in rp.TAGS)


# ── the recorded trade ────────────────────────────────────────────────────


def _text(lines) -> str:
    import re

    return "\n".join(f"{line.label} | {re.sub('<[^>]+>', '', str(line.text))}"
                     for line in lines)


def test_recorded_lines_match_the_mock_wording():
    view = rp.TradeView.from_row(ROW)
    text = _text(rp.recorded_lines(view, instrument="micro"))
    assert text.splitlines() == [
        "7:07 PM | Bought 10 micros at 24,971.00",
        "Initial stop | 24,949.75 · risk $425.00",
        "7:10 PM | Sold 5 at 24,992.25 · target reached, stop on the rest moved to entry",
        "Apr 13, 3:55 PM | Sold 5 at 25,576.25 · scheduled daily close",
        "Result | +$6,254.72 after $10.28 costs",
        "Account 6 balance | $1,821.10 → $8,075.82",
    ]


def test_point_in_time_hides_the_exit_result_and_balance():
    view = rp.TradeView.from_row(ROW)
    moment = rp.default_moment(view)
    # the first five-minute clock mark after the 7:07 PM entry — not the half exit
    assert moment == pd.Timestamp("2026-04-13T00:10:00Z") and rp.clock(moment) == "7:10 PM"
    text = _text(rp.recorded_lines(view, instrument="micro", moment=moment))
    assert "3:55" not in text and "6,254.72" not in text and "8,075.82" not in text
    assert "Sold 5" not in text  # the half came out at 7:10:27 PM, after 7:10:00
    later = _text(rp.recorded_lines(view, instrument="micro", moment=view.half_utc))
    assert "Sold 5 at 24,992.25" in later and "6,254.72" not in later
    early = _text(rp.recorded_lines(view, moment=view.entry_utc - pd.Timedelta(minutes=1)))
    assert early == ""
    assert rp.trade_label(view) == "Apr 12, 7:07 PM · +$6,254.72"
    assert rp.trade_label(view, hide_result=True) == "Apr 12, 7:07 PM · result hidden"


def _later_story(row: dict) -> dict:
    """The same trade up to its entry, with everything after it different."""

    return {**row, "account_number": 2, "account_failed": True, "scale_out_ns": None,
            "scale_out_ticks": None, "scale_out_quantity": None, "final_exit_quantity": 10,
            "final_stop_ticks": row["stop_ticks"], "exit_kind": "account_failure",
            "exit_utc": "2026-04-13T00:12:00Z", "exit_ticks": row["stop_ticks"] - 40,
            "net_pnl_usd": -2210.0, "balance_before_usd": 900.0, "balance_after_usd": -1310.0}


def test_point_in_time_moments_never_come_from_the_exit_or_later_accounts():
    view = rp.TradeView.from_row(ROW)
    other = rp.TradeView.from_row(_later_story(ROW))
    shown = [(m.at, m.label) for m in rp.moments(view, None)]
    # a trade lost within minutes offers exactly the same clock as one held to the close
    assert shown == [(m.at, m.label) for m in rp.moments(other, None)]
    assert rp.default_moment(view) == rp.default_moment(other)
    labels = [label for _, label in shown]
    assert labels[:14] == ["7:07 PM · entry", "7:10 PM", "7:15 PM", "7:20 PM", "7:25 PM",
                           "7:30 PM", "7:35 PM", "7:40 PM", "7:45 PM", "7:50 PM", "7:55 PM",
                           "8:00 PM", "8:05 PM", "9:00 PM"]
    assert labels[-3:] == ["Apr 13, 2:00 PM", "Apr 13, 3:00 PM", "Apr 13, 4:00 PM"]
    assert shown[-1][0] == rp.day_close(view) == rc.trading_window(None, "2026-04-13")[1]
    # named by clock time only: nothing about the half exit, the exit or an account
    for label in labels:
        for word in ("half", "out", "exit", "target", "stop", "close", "Account", "result"):
            assert word not in label, label
    assert view.half_utc not in {at for at, _ in shown}
    assert view.exit_utc not in {at for at, _ in shown}
    # the trade picker leaves the account out in point in time (a later trade's new
    # account would show this account was lost)
    assert rp.trade_label(other, hide_result=True, account=True) == (
        "Apr 12, 7:07 PM · result hidden")
    assert rp.trade_label(view, account=True) == "Apr 12, 7:07 PM · Account 6 · +$6,254.72"


def test_point_in_time_clock_marks_end_at_the_close_for_a_late_entry():
    late = rp.TradeView.from_row({**ROW, "entry_utc": "2026-04-13T20:52:00Z"})  # 3:52 PM
    assert [m.label for m in rp.moments(late, None)] == [
        "3:52 PM · entry", "3:55 PM", "4:00 PM"]
    at_close = rp.TradeView.from_row({**ROW, "entry_utc": "2026-04-13T21:00:00Z"})
    assert rp.default_moment(at_close) == at_close.entry_utc  # no mark after the close
    undated = rp.TradeView.from_row({**ROW, "trading_day": ""})
    assert rp.day_close(undated) == rp.day_close(rp.TradeView.from_row(ROW))


def test_whole_position_trades_get_their_own_key_item():
    row = {**ROW, "scale_out_ns": None, "scale_out_ticks": None, "scale_out_quantity": None,
           "final_exit_quantity": 10, "final_stop_ticks": 99799, "exit_kind": "stop",
           "exit_ticks": 99799,
           "exit_utc": "2026-04-13T00:20:00Z"}
    view = rp.TradeView.from_row(row)
    items = rp.setup_key(None, view)
    assert [role for role, _, _ in items] == ["entry", "exit"]
    assert items[1][1] == "Stopped out: out at 24,949.75, 7:20 PM"
    # the default moment is a clock mark, the same whatever the exit (no half exit here)
    assert rp.default_moment(view) == pd.Timestamp("2026-04-13T00:10:00Z")


# ── candles and point in time ─────────────────────────────────────────────


def test_candles_aggregate_by_their_opening_minute():
    minutes = _minutes(count=25)
    candles = rc.aggregate(minutes, 10)
    assert len(candles) == 3 and list(candles["minutes"]) == [10, 10, 5]
    first = minutes.iloc[:10]
    assert candles.iloc[0]["open"] == first["open"].iloc[0]
    assert candles.iloc[0]["close"] == first["close"].iloc[-1]
    assert candles.iloc[0]["high"] == first["high"].max()
    assert candles.iloc[0]["low"] == first["low"].min()
    assert candles.iloc[0]["open_utc"] == pd.Timestamp("2026-04-12T22:00:00Z")


def test_point_in_time_uses_only_bars_closed_by_the_moment():
    minutes = _minutes(count=30)
    moment = pd.Timestamp("2026-04-12T22:12:30Z")
    shown = rc.visible_minutes(minutes, moment)
    assert shown["logical_close_ts_utc"].max() <= moment and len(shown) == 12
    candles = rc.aggregate(shown, 10)
    assert list(candles["minutes"]) == [10, 2]  # the forming candle, as it looked then


def test_trading_window_runs_from_the_5_pm_open_to_the_4_pm_close():
    start, end = rc.trading_window(pd.DataFrame(), "2026-04-13")
    assert (rp.short(start), rp.short(end)) == ("Apr 12, 5:00 PM", "Apr 13, 4:00 PM")


def _trace_points(fig):
    xs, ys = [], []
    for trace in fig.data:
        if trace.type == "candlestick":
            xs += list(trace.x)
            ys += list(trace.high) + list(trace.low)
        else:
            xs += [x for x in (trace.x or ()) if x is not None]
            ys += [y for y in (trace.y or ()) if y is not None]
    return xs, ys


def test_the_point_in_time_chart_draws_nothing_after_the_moment():
    view = rp.TradeView.from_row(ROW)
    minutes = _minutes()
    moment = view.half_utc
    fig = rc.whole_trade_figure(minutes, view, None, size_minutes=10, moment=moment)
    xs, ys = _trace_points(fig)
    wall = rc._wall(moment)
    # candles are centered in their bucket: the forming 7:10 candle holds no later minute
    assert max(xs) <= wall + pd.Timedelta(minutes=5)
    later = minutes[minutes["logical_close_ts_utc"] > moment]
    assert fig.layout.yaxis.range[1] < later["high"].max()  # the axis doesn't leak later highs
    assert all("Trade result" not in (a.text or "") for a in fig.layout.annotations)
    assert any(a.text == "Hidden: everything after 7:10 PM" for a in fig.layout.annotations)
    full = rc.whole_trade_figure(minutes, view, None, size_minutes=10)
    assert any("Trade result +$6,254.72" in (a.text or "") for a in full.layout.annotations)


def test_setup_chart_window_and_markers():
    view = rp.TradeView.from_row(ROW)
    start, end = rc.setup_window(view, None)
    assert (rp.clock(start), rp.clock(end)) == ("6:40 PM", "7:25 PM")
    fig, _ = rc.setup_figure(_minutes(), view, None, key=rp.setup_key(None, view))
    numbers = [t.text[0] for t in fig.data if getattr(t, "mode", None) == "markers+text"]
    assert numbers == ["1", "2"]  # entry and half exit when no setup record exists


# ── theme: no hard-coded color; figures follow the active palette ─────────


#: a hex color or an rgb()/rgba() with numeric channels (``theme.rgba("blue", …)`` is a call)
_COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{6}\b|rgba?\(\s*\d")


class _FakeSt:
    """Enough of ``st`` for the cards: containers are plain context managers."""

    def container(self, **_kwargs):
        return contextlib.nullcontext()


def test_trade_review_css_and_inline_markup_use_palette_variables_only(monkeypatch):
    import ifvg_lab_trade_review as review
    import ifvg_lab_ui

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.setup_records import NO_RECORD

    assert not _COLOR_LITERAL.search(review._CSS)
    assert "var(--lab-panel)" in review._CSS and "var(--lab-blue)" in review._CSS
    for module in (review, rc):  # nothing in either module names a color value
        assert not _COLOR_LITERAL.search(Path(module.__file__).read_text(encoding="utf-8"))
    shown: list[str] = []
    monkeypatch.setattr(ifvg_lab_ui, "show",
                        lambda markup, st_module=None: shown.append(str(markup)))
    view = rp.TradeView.from_row(ROW)
    study = types.SimpleNamespace(plan=object())
    fake = _FakeSt()
    review._whole_trade_card(fake, None, view, None, NO_RECORD, moment=None, size=10,
                             zones=True, stops=True, times=True, close_clock=None,
                             instrument=None, key="t")
    review._recorded_card(fake, study, {}, "cfg", TPT, ROW, view, [ROW], instrument="micro",
                          moment=view.entry_utc - pd.Timedelta(minutes=1))
    review._formed_card(fake, study, "cfg", view, None, moment=None)
    markup = "\n".join(shown)
    assert "Up candle" in markup and "Nothing was recorded yet" in markup
    assert "How the setup formed" in markup
    assert not _COLOR_LITERAL.search(markup), markup
    for name in ("blue", "orange", "blue-band", "blue-light", "body-2", "muted", "blue-dark",
                 "ink", "light-rule"):
        assert f"var(--lab-{name})" in markup, name


def test_review_figures_take_the_dark_palette_when_the_theme_is_dark():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    view = rp.TradeView.from_row(ROW)
    minutes = _minutes()
    light = rc.whole_trade_figure(minutes, view, None, size_minutes=10)
    candles = next(t for t in light.data if t.type == "candlestick")
    assert candles.increasing.line.color == theme.COLORS["blue"]  # headless: light, as before
    theme.set_theme_resolver(lambda: "dark")
    try:
        fig = rc.whole_trade_figure(minutes, view, None, size_minutes=10)
        hidden = rc.whole_trade_figure(minutes, view, None, size_minutes=10, moment=view.half_utc)
        setup, _ = rc.setup_figure(minutes, view, None, key=rp.setup_key(None, view))
    finally:
        theme.set_theme_resolver(None)
    dark = theme.DARK_COLORS
    candles = next(t for t in fig.data if t.type == "candlestick")
    assert candles.increasing.line.color == dark["blue"]
    assert candles.decreasing.fillcolor == dark["orange"]
    entry = next(t for t in fig.data
                 if getattr(getattr(t, "marker", None), "symbol", None) == "triangle-up")
    assert entry.marker.color == dark["ink"] and entry.marker.line.color == dark["panel"]
    result = next(a for a in fig.layout.annotations if "Trade result" in (a.text or ""))
    assert result.font.color == dark["on_ink"] and result.bgcolor == dark["ink"]
    assert theme.rgba("blue", 0.06, theme="dark") in {s.fillcolor for s in fig.layout.shapes}
    assert fig.layout.xaxis.tickfont.color == dark["muted"]
    assert fig.layout.yaxis.tickfont.color == dark["muted"]
    cover = next(a for a in hidden.layout.annotations if (a.text or "").startswith("Hidden"))
    assert cover.bgcolor == dark["panel"] and cover.font.color == dark["body_2"]
    assert dark["header_row"] in {s.fillcolor for s in hidden.layout.shapes}
    number = next(t for t in setup.data if getattr(t, "mode", None) == "markers+text")
    assert number.textfont.color == dark["on_ink"] and number.marker.color == dark["ink"]
    # the resolver is gone again: the next figure is light
    again = rc.whole_trade_figure(minutes, view, None, size_minutes=10)
    assert next(t for t in again.data if t.type == "candlestick").increasing.line.color == (
        theme.COLORS["blue"])


# ── the reference trade, end to end ───────────────────────────────────────


@pytest.fixture(scope="module")
def reference():
    if not HAVE_STUDY:
        pytest.skip("the saved funded variation study is not on this computer")
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        open_funded_study,
        ordered_trades,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.setup_records import (
        find_setup_record,
        load_setup_record_source,
    )

    study = open_funded_study(STORE, RESULT_ID)
    root = market.study_package_root(study.plan)
    if root is None:
        pytest.skip("the verified strategy package is not on this computer")
    variant = next(v for v in study.plan.variants if v.name == LEADER)
    row = next(t for t in ordered_trades(study, LEADER, TPT) if t["seq"] == 380)
    record = find_setup_record(row, configuration=LEADER, source=load_setup_record_source(root),
                               axis_value_ids=dict(variant.axis_value_ids),
                               in_verified_study=variant.in_verified_study,
                               base_configuration=study.plan.base_configuration)
    minutes = market.load_index_minutes(root, cutoff_utc=study.result["period"]["cutoff_utc"])
    return study, row, record, minutes


def test_reference_trade_review_values(reference):
    study, row, record, minutes = reference
    view = rp.TradeView.from_row(row)
    # correction A6: the leader is not a verified member; its record is related context
    assert not record.identity_established
    steps = [(s.number, s.text, s.time_text) for s in
             rp.setup_steps(record, view, distance_cap=80)]
    # closeout review follow-up: each time is when the step became known. The tap and the
    # close-through are known when their candle closes (the saved record's bar_close: 5:19 PM
    # and 7:06 PM); the candle is named by its opening minute. The mock's 5:18/7:05 PM were
    # the candles' opening minutes, not when either event was known.
    assert record.bar_close("tap_bar") == pd.Timestamp("2026-04-12T22:19:00Z")
    assert record.bar_close("inversion_bar") == pd.Timestamp("2026-04-13T00:06:00Z")
    assert steps == [
        (1, "Four-hour gap, 24,451.50–25,086.75", "Apr 8, 1:00 AM"),
        (2, "The 5:18 PM candle taps into it", "Apr 12, 5:19 PM"),
        (3, "Five-minute parent gap forms", "7:00 PM"),
        # correction A6: this configuration's 80-tick limit is not claimed for a related record
        (4, "Opposing 1-minute gap, 3 ticks", "7:03 PM"),
        (5, "The 7:05 PM candle closes through it", "7:06 PM"),
        (6, "Entry", "7:07 PM"),
    ]
    assert [text for _, text, _ in rp.setup_key(record, view)] == [
        "Five-minute parent gap, 24,937.25–24,966.50, confirmed 7:00 PM",
        "Opposing one-minute gap, 24,961.75–24,962.50, confirmed 7:03 PM",
        "The 7:05 PM candle closes through it",
        "Entry 24,971.00 at 7:07 PM · initial stop 24,949.75",
        "Half out at 24,992.25 (1R), 7:10 PM · stop on the rest moves to entry",
    ]
    text = _text(rp.recorded_lines(view, instrument="micro"))
    assert "Sold 5 at 25,576.25 · scheduled daily close" in text
    assert "+$6,254.72 after $10.28 costs" in text
    day = rc.day_minutes(minutes, view.trading_day)
    window = rc.trading_window(day, view.trading_day)
    assert (rp.short(window[0]), rp.short(window[1])) == ("Apr 12, 5:00 PM", "Apr 13, 4:00 PM")
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    assert view.seq in rp.five_largest(ordered_trades(study, LEADER, TPT))
    moment = rp.default_moment(view)
    assert rp.clock(moment) == "7:10 PM"
    offered = rp.moments(view, record)
    # correction A6: a related record adds no setup moments — only the entry and clock marks
    assert [m.label for m in offered][:3] == ["7:07 PM · entry", "7:10 PM", "7:15 PM"]
    assert offered == rp.moments(view, None)
    other = rp.TradeView.from_row(_later_story(row))
    assert offered == rp.moments(other, record)  # nothing after the entry shapes the list
    fig = rc.whole_trade_figure(day, view, record, moment=moment)
    xs, _ys = _trace_points(fig)
    assert max(xs) <= rc._wall(moment) + pd.Timedelta(minutes=5)
    assert fig.layout.yaxis.range[1] < 25_100  # the later rally to 25,576 stays hidden


# ── the page in Streamlit (AppTest) ───────────────────────────────────────


def _app():
    import ifvg_lab_trade_review as page
    import streamlit as st

    page.render_trade_review_page(st, page._TEST_ROOTS)


@pytest.fixture
def page(monkeypatch, tmp_path):
    if not HAVE_STUDY:
        pytest.skip("the saved funded variation study is not on this computer")
    import ifvg_lab_trade_review as page
    import ifvg_lab_ui

    clicks: dict[str, str] = {}

    def fake_clickable(markup, *, key, st_module=None):  # components v2 needs a browser
        import streamlit as st

        (st_module or st).html(str(markup))
        return clicks.pop(key, None)

    monkeypatch.setattr(ifvg_lab_ui, "clickable", fake_clickable)
    target = {"result_id": RESULT_ID, "store_root": str(STORE), "app": "ifsm",
              "study_key": "78406b15cdd9c1363b7cf626f7e45d48e58f6aa4e634a25dc00a3e57bdb28336",
              "plan_id": "78406b15cdd9c1363b7cf626f7e45d48e58f6aa4e634a25dc00a3e57bdb28336",
              "name": "Funded variation study — 64 configurations", "status": "Completed"}
    roots = {"store_root": STORE, "repo_root": tmp_path, "state_root": tmp_path / "jobs",
             "draft_root": tmp_path / "drafts"}
    monkeypatch.setattr(page, "_TEST_ROOTS", roots, raising=False)
    monkeypatch.setattr(page, "_funded_targets", lambda st_module, roots: [dict(target)])
    return {"target": target, "roots": roots, "ledger": tmp_path, "clicks": clicks}


def _run(state: dict):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_app, default_timeout=90)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _review_target(point: bool = False, back: str | None = "Trades") -> dict:
    return {"ifvg_lab_v1_review_target": {
        "source": "Funded trades", "result_id": RESULT_ID, "configuration": LEADER,
        "firm_key": TPT, "account_number": 6, "trade_seq": 380, "point_in_time": point,
        "back_tab": back}}


def _page_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "warning", "info", "error", "html"):
        try:
            parts += [str(getattr(e, "body", None) or getattr(e, "value", ""))
                      for e in at.get(kind)]
        except Exception:
            continue
    return "\n".join(parts)


def test_the_link_from_the_trades_tab_opens_that_exact_trade(page):
    at = _run(_review_target())
    assert not at.exception, at.exception
    scope = RESULT_ID[:16]
    assert at.selectbox(key=f"ifvg_lab_v1_review_config_{scope}").value == LEADER
    assert at.selectbox(key=f"ifvg_lab_v1_review_firm_{scope}").value == TPT
    assert at.selectbox(key=f"ifvg_lab_v1_review_account_{scope}").value == 6
    assert at.selectbox(key=f"ifvg_lab_v1_review_trade_{scope}").value == 380
    text = _page_text(at)
    assert "Bought 10 micros at" in text and "+$6,254.72" in text
    # correction A6: another configuration's record is related context, not this setup
    import html

    assert ("Related context from configuration S1_D80_W1_P1's saved setup record"
            in html.unescape(text))
    context = at.session_state["funded_comparison_v1_selected_context"][RESULT_ID]
    assert context["firm_key"] == TPT and context["configuration"] == LEADER


def test_point_in_time_hides_the_result_on_the_page(page):
    at = _run(_review_target(point=True))
    assert not at.exception, at.exception
    text = _page_text(at)
    assert "Exit and result hidden, so you judge the setup as it looked at 7:10 PM." in text
    assert "6,254.72" not in text and "25,576.25" not in text and "8,075.82" not in text
    scope = RESULT_ID[:16]
    trade = at.selectbox(key=f"ifvg_lab_v1_review_trade_{scope}")
    assert "result hidden" in trade.format_func(trade.value)


def test_point_in_time_pickers_name_nothing_from_after_the_moment(page):
    at = _run(_review_target(point=True))
    assert not at.exception, at.exception
    scope = RESULT_ID[:16]
    moment = at.selectbox(key=f"ifvg_lab_v1_review_moment_{scope}_{LEADER}_{TPT}_380")
    assert moment.format_func(moment.value) == "7:10 PM"
    options = list(moment.options)
    assert "7:07 PM · entry" in options and options[-1] == "Apr 13, 4:00 PM"
    assert not any("half" in o or "exit" in o or "Account" in o for o in options)
    text = _page_text(at)
    assert "Sold 5 at" not in text  # the half exit (7:10:27 PM) is after 7:10:00 PM
    at.selectbox(key=f"ifvg_lab_v1_review_account_{scope}").set_value("all").run()
    assert not at.exception, at.exception
    trade = at.selectbox(key=f"ifvg_lab_v1_review_trade_{scope}")
    assert trade.value == 380
    names = list(trade.options)  # the labels as drawn: only trades entered by 7:10 PM
    assert len(names) == 77  # fix F5: the 37 later trades are not listed
    assert all("Account" not in n and n.endswith("· result hidden") for n in names)
    accounts = list(at.selectbox(key=f"ifvg_lab_v1_review_account_{scope}").options)
    assert accounts == ["All accounts"] + [f"Account {n}" for n in range(1, 7)]
    text = _page_text(at)
    assert "trade 77 at this firm" in text and "of 114" not in text
    # a moment after the exit shows the trade as it ended, and says so
    close = pd.Timestamp("2026-04-13T21:00:00Z")  # Apr 13, 4:00 PM Chicago
    moment = at.selectbox(key=f"ifvg_lab_v1_review_moment_{scope}_{LEADER}_{TPT}_380")
    assert moment.format_func(close.isoformat()) == options[-1]
    moment.set_value(close.isoformat()).run()
    assert not at.exception, at.exception
    text = _page_text(at)
    assert "Later candles are hidden; the trade had ended by then." in text
    assert "+$6,254.72" in text
    # full history restores every trade, every account and the total
    at.radio(key="ifvg_lab_v1_review_mode").set_value("Full history").run()
    assert not at.exception, at.exception
    assert len(at.selectbox(key=f"ifvg_lab_v1_review_trade_{scope}").options) == 114
    assert len(at.selectbox(key=f"ifvg_lab_v1_review_account_{scope}").options) == 7
    assert "trade 77 of 114 at this firm" in _page_text(at)


def test_back_link_shows_only_while_its_result_is_open(page, monkeypatch):
    import ifvg_lab_trade_review as review

    other = dict(page["target"], result_id="0" * 64, name="Another funded study — 2 configs")
    monkeypatch.setattr(review, "_funded_targets",
                        lambda st_module, roots: [dict(page["target"]), dict(other)])
    at = _run(_review_target())
    assert not at.exception, at.exception
    assert "Back to this configuration" in _page_text(at)
    at.selectbox(key="ifvg_lab_v1_review_study").set_value("0" * 64).run()
    assert not at.exception, at.exception
    assert "Back to this configuration" not in _page_text(at)  # it would do nothing here
    at.selectbox(key="ifvg_lab_v1_review_study").set_value(RESULT_ID).run()
    assert "Back to this configuration" in _page_text(at)


def _app_drilldown():
    import ifvg_lab_trade_review as page
    import ifvg_workspace
    import streamlit as st
    from ifvg_ui_common import queue_replay_drilldown

    if not st.session_state.get("test_drilldown_queued"):
        # what 'Inspect supporting trade' / 'Review selected opportunity' do on click
        st.session_state["test_drilldown_queued"] = True
        kind, value = page._TEST_JUMP
        queue_replay_drilldown(st, kind, value, pair_label="pair", toast=lambda _m: None)
    ifvg_workspace.render_workspace(st, roots=page._TEST_ROOTS)


@pytest.mark.parametrize(("kind", "source", "mode"), [
    ("candidate_id", "Verified context", "candidate"),
    ("trade_id", "Verified context", "candidate"),
    ("setup_id", "Setups not taken", "setup"),
])
def test_a_verifier_drilldown_opens_its_case_in_trade_review(page, monkeypatch, kind, source,
                                                             mode):
    import ifvg_lab_tab
    import ifvg_lab_trade_review as review
    from streamlit.testing.v1 import AppTest

    calls = []

    def verifier(st_module):
        ss = st_module.session_state
        calls.append(("context", ss.get("ifvg_context_v1_selection_mode"),
                      ss.get("ifvg_context_v1_pending_jump"),
                      ss.get("ifvg_context_v1_replay_pair")))

    monkeypatch.setattr(ifvg_lab_tab, "render_ifvg_replay_tab", verifier)
    monkeypatch.setattr(review, "_render_funded",
                        lambda st_module, roots, targets: calls.append(("funded",)))
    monkeypatch.setattr(review, "_TEST_JUMP", (kind, "abc"), raising=False)
    at = AppTest.from_function(_app_drilldown, default_timeout=90).run()
    assert not at.exception, at.exception
    assert at.session_state["ifvg_workspace_destination"] == "Trade review"
    assert calls == [("context", mode, (kind, "abc"), "pair")]
    assert at.radio(key="ifvg_lab_v1_review_source").value == source
    # the jump routes once: while the reviewer still holds it, another source can be chosen
    at.radio(key="ifvg_lab_v1_review_source").set_value("Funded trades").run()
    assert not at.exception, at.exception
    assert calls[-1] == ("funded",)
    at.run()
    assert calls[-1] == ("funded",)


def test_a_funded_link_wins_over_a_queued_verifier_jump(page, monkeypatch):
    import ifvg_lab_tab
    import ifvg_lab_trade_review as review

    calls = []
    monkeypatch.setattr(ifvg_lab_tab, "render_ifvg_replay_tab",
                        lambda st_module: calls.append("context"))
    monkeypatch.setattr(review, "_render_funded",
                        lambda st_module, roots, targets: calls.append("funded"))
    at = _run({**_review_target(), "ifvg_context_v1_pending_jump": ("candidate_id", "abc")})
    assert not at.exception, at.exception
    assert calls == ["funded"]


def test_every_configuration_picker_label_is_unique_on_both_saved_results():
    if not HAVE_STUDY:
        pytest.skip("the saved funded studies are not on this computer")
    import ifvg_lab_trade_review as review

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study

    other = "92f5a08d3b81ae83eddde0525a04220ea9e1e9eb849d1118303d5de7efd743f4"
    for result_id in (RESULT_ID, other):
        if not (STORE / "funded_comparison_results" / result_id / "result.json").is_file():
            continue
        study = open_funded_study(STORE, result_id)
        configurations = list(study.configurations)
        names, labels = review.configuration_labels(study, configurations)
        assert len(set(labels.values())) == len(configurations), result_id
        assert len({n.full for n in names.values()}) == len(configurations), result_id


def test_picker_labels_stay_unique_when_short_names_and_second_lines_collide():
    import ifvg_lab_trade_review as review

    labels = review._picker_labels({"A": "All hours · Long", "B": "All hours · Long",
                                    "C": "All hours · Long"},
                                   {"A": "1R · x", "B": "1R · y", "C": "1R · y"})
    assert len(set(labels.values())) == 3
    assert labels["A"] == "All hours · Long · x"


def test_a_missing_strategy_package_is_not_remembered(monkeypatch, tmp_path):
    import ifvg_lab_trade_review as review
    import ifvg_lab_ui

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market

    found: list = [None]
    monkeypatch.setattr(ifvg_lab_ui, "funded_study",
                        lambda store_root, result_id: type("S", (), {"plan": object()})())
    monkeypatch.setattr(market, "study_package_root", lambda plan: found[0])
    review._located_package_root.clear()
    assert review._package_root(str(tmp_path), "r" * 64) is None
    found[0] = tmp_path  # the archive is restored
    assert review._package_root(str(tmp_path), "r" * 64) == str(tmp_path)
    found[0] = None  # a found package stays cached for the process
    assert review._package_root(str(tmp_path), "r" * 64) == str(tmp_path)
    review._located_package_root.clear()


def test_a_review_saves_under_this_firm_account_and_trade_only(page):
    from alpha_lab.agents.data_infra.ifvg.visual_review_store import list_reviews

    at = _run(_review_target())
    assert not at.exception, at.exception
    overall = next(r for r in at.radio if r.label == "Overall")
    overall.set_value("correct").run()
    at.text_input(key="ifvg_lab_v1_review_reviewer").input("Test reviewer").run()
    entry = next(s for s in at.selectbox if s.label == "Entry and stop are right")
    entry.set_value("Agree").run()
    next(b for b in at.button if b.label == "Save review").click().run()
    assert not at.exception, at.exception
    saved = list_reviews(repo_root=page["ledger"])
    assert len(saved) == 1
    row = saved.iloc[0]
    assert row["replay_chart_artifact_id"] == f"funded_comparison_result:{RESULT_ID}"
    assert row["candidate_id"] == (f"funded:{RESULT_ID}:{LEADER}|{TPT}#6:"
                                   "e3dcb3e9-a0d2-526c-892c-99a3ab649b57")
    assert row["pair_ref"]["firm_key"] == TPT and row["pair_ref"]["account_number"] == "6"
    assert row["pair_ref"]["funded_trade_seq"] == "380"
    assert row["overall_verdict"] == "correct"
    assert row["entry_verdict"] == row["stop_verdict"] == "correct"
    assert pd.isna(row["htf_verdict"]) or row["htf_verdict"] is None
    assert "Earlier reviews of this trade · 1" in _page_text(at)


def test_back_returns_to_this_configurations_trades_tab(page):
    at = _run(_review_target())
    assert not at.exception, at.exception
    page["clicks"]["review_header"] = "back"
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["ifvg_workspace_screen"] == "funded_detail"
    context = at.session_state["funded_comparison_v1_selected_context"][RESULT_ID]
    assert context["tab"] == "Trades" and context["configuration"] == LEADER
    assert context["firm_key"] == TPT


def test_previous_and_next_follow_the_pairs_execution_order(page):
    at = _run(_review_target())
    scope = RESULT_ID[:16]
    at.button(key="ifvg_lab_v1_review_next").click().run()
    assert not at.exception, at.exception
    following = at.selectbox(key=f"ifvg_lab_v1_review_trade_{scope}").value
    assert following != 380
    at.button(key="ifvg_lab_v1_review_previous").click().run()
    assert at.selectbox(key=f"ifvg_lab_v1_review_trade_{scope}").value == 380


def test_other_sources_reuse_the_existing_reviewers(page, monkeypatch):
    import ifvg_lab_tab
    import ifvg_search_review

    calls = []
    monkeypatch.setattr(ifvg_search_review, "render_trade_review",
                        lambda st_module, roots, **kw: calls.append(("strategy", kw)))
    monkeypatch.setattr(ifvg_lab_tab, "render_ifvg_replay_tab",
                        lambda st_module: calls.append(
                            ("context", st_module.session_state.get(
                                "ifvg_context_v1_selection_mode"))))
    for source in ("Strategy trades", "Setups not taken", "Verified context"):
        at = _run({"ifvg_lab_v1_review_source_value": source})
        assert not at.exception, at.exception
    assert calls == [("strategy", {"source": "Study executions", "include_funded": False}),
                     ("context", "setup"), ("context", "candidate")]


def test_a_loss_limit_exit_shows_the_recorded_check_and_the_replacement():
    from tests.propsim.funded.comparison_fixture import comparison_fixture_result

    result = comparison_fixture_result()
    row = next(t for t in result["tables"]["trades"]
               if t["pair_id"] == "S0_D160|takeprofittrader" and t["account_failed"])
    lines = dict(rp.failure_lines(result["tables"], row))
    assert "the loss limit" in lines["Loss limit"]
    assert lines["Replacement"].startswith("Account 2 started ")
    view = rp.TradeView.from_row(row)
    assert rp.exit_words(view) == "account loss limit reached"
    hidden = rp.recorded_lines(view, moment=view.entry_utc, extra=list(lines.items()))
    assert all(line.label not in ("Loss limit", "Replacement") for line in hidden)


def test_save_and_next_trade_saves_then_opens_the_next_trade(page):
    from alpha_lab.agents.data_infra.ifvg.visual_review_store import list_reviews

    at = _run(_review_target())
    scope = RESULT_ID[:16]
    next(r for r in at.radio if r.label == "Overall").set_value("insufficient_evidence").run()
    at.text_input(key="ifvg_lab_v1_review_reviewer").input("Test reviewer").run()
    next(b for b in at.button if b.label == "Save and next trade").click().run()
    assert not at.exception, at.exception
    saved = list_reviews(repo_root=page["ledger"])
    assert len(saved) == 1 and saved.iloc[0]["pair_ref"]["funded_trade_seq"] == "380"
    assert at.selectbox(key=f"ifvg_lab_v1_review_trade_{scope}").value != 380
    # the new trade's form starts empty: a verdict never carries across trades
    assert next(r for r in at.radio if r.label == "Overall").value is None


def test_the_firm_picker_is_the_shared_firm_and_the_candle_size_changes_the_chart(page):
    at = _run(_review_target())
    scope = RESULT_ID[:16]
    at.selectbox(key=f"ifvg_lab_v1_review_firm_{scope}").set_value("myfundedfutures").run()
    assert not at.exception, at.exception
    context = at.session_state["funded_comparison_v1_selected_context"][RESULT_ID]
    assert context["firm_key"] == "myfundedfutures"
    assert "MyFundedFutures" in _page_text(at)
    at.selectbox(key="ifvg_lab_v1_review_candles").set_value(5).run()
    assert "5-minute candles" in _page_text(at)
    at.checkbox(key="ifvg_lab_v1_review_zones").uncheck().run()
    assert not at.exception, at.exception
