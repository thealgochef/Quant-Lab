"""Configuration detail · Market conditions tab (mock 07).

Pure helpers are tested on small hand-made inputs. The reference section reads
the saved funded variation study and its verified strategy package on this
computer (read only) and checks the CALCULATIONS.md market-condition values as
the tab shows them; it is skipped where those local records are absent.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import ifvg_lab_detail_market as dm  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.lab import market  # noqa: E402

STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
TPT = "takeprofittrader"
MINUS = "−"

RQ, RV, FQ, FV = market.CONDITIONS
NE = market.NOT_ENOUGH


def _labels(mapping: dict[str, str]) -> market.ConditionLabels:
    first = next((d for d, lab in mapping.items() if lab != NE), None)
    return market.ConditionLabels(labels=dict(mapping), first_labeled=first,
                                  volatility_median=0.01)


def _trade(day: str, net: float, *, entry_utc: str | None = None,
           exit_utc: str | None = None) -> dict:
    return {"trading_day": day, "net_pnl_usd": net, "entry_utc": entry_utc,
            "exit_utc": exit_utc or f"{day}T15:00:00Z", "initial_risk_usd": 100.0}


# ── pure helpers ──────────────────────────────────────────────────────────


def test_day_span_forms():
    assert dm.day_span("2026-01-13", "2026-01-15") == "January 13–15"
    assert dm.day_span("2026-01-30", "2026-02-02") == "January 30 – February 2"
    assert dm.day_span("2026-03-18", "2026-04-07", joiner="to") == "March 18 to April 7"
    assert dm.day_span("2026-01-13", "2026-01-13") == "January 13"


def test_definition_sentence_uses_the_first_labeled_day():
    labels = _labels({"2026-01-13": NE, "2026-01-14": RQ})
    text = dm.definition_sentence(labels)
    # correction A1: the old "labeled from the previous 10 trading days" head was wrong (the
    # day's own close is used); these labels are named retrospective, with their version
    assert text.startswith("Retrospective labels (retrospective_daily_close_v1): each trading "
                           "day is labeled from its own last close (4:00 PM, earlier on a "
                           "shortened day) against the close 10 trading days earlier, and its "
                           "volatility against the median of the whole study. They describe the "
                           "day with hindsight; they were not known when a trade was entered.")
    assert text.endswith("Labels start January 14, 2026; the first days lack enough history.")
    none = dm.definition_sentence(_labels({"2026-01-13": NE}))
    assert "none can be labeled" in none


def test_daily_by_moves_a_trade_that_closes_after_the_5_pm_open():
    calendar = ["2026-01-13", "2026-01-14"]
    # entered on trading day January 13, closed 5:30 PM Chicago = trading day January 14
    crossing = _trade("2026-01-13", 300.0, exit_utc="2026-01-13T23:30:00Z")
    same = _trade("2026-01-14", -100.0, exit_utc="2026-01-14T16:00:00Z")
    by_entry, outside = dm.daily_by([crossing, same], calendar, by="entry")
    assert by_entry == [("2026-01-13", 300.0), ("2026-01-14", -100.0)] and outside == 0
    by_exit, _ = dm.daily_by([crossing, same], calendar, by="exit")
    assert by_exit == [("2026-01-13", 0.0), ("2026-01-14", 200.0)]
    assert dm.moved_trades([crossing, same]) == 1
    late = _trade("2026-01-14", 50.0, exit_utc="2026-01-14T23:10:00Z")  # exit day Jan 15
    assert dm.daily_by([late], calendar, by="exit")[1] == 1  # counted, never re-homed


def test_same_day_sentence():
    assert dm.same_day_sentence(0, 114) == (
        "Every trade here closed inside the trading day it opened, so labeling by entry day "
        "or by exit day gives the same counts.")
    assert dm.same_day_sentence(2, 10).startswith("2 trades of 10 closed on a later trading day")
    assert "no funded trades" in dm.same_day_sentence(0, 0)


def test_history_line():
    labels = _labels({"2026-01-13": NE, "2026-01-14": NE, "2026-01-15": NE, "2026-01-16": RQ})
    trades = [_trade("2026-01-13", -500.0), _trade("2026-01-15", -164.0),
              _trade("2026-01-16", 100.0)]
    cards = market.condition_cards(labels, trades)
    assert dm.history_line(cards, labels, 3) == (
        f"Not enough history (January 13–15): 2 trades, {MINUS}$664. "
        "Shares are of 3 funded trades.")
    quiet = market.condition_cards(labels, trades[2:])
    assert dm.history_line(quiet, labels, 1) == (
        "Not enough history (January 13–15): no trades. Shares are of 1 funded trade.")


def test_stretch_caption_names_the_longest_falling_stretch_and_the_others():
    items = [
        market.Stretch(FQ, "2026-01-19", "2026-01-26", 6, -1701.12),
        market.Stretch(RQ, "2026-01-27", "2026-01-29", 3, 361.10),
        market.Stretch(FV, "2026-02-05", "2026-02-18", 10, -1668.10),
        market.Stretch(FV, "2026-03-18", "2026-04-07", 15, 10536.36),
        market.Stretch(FQ, "2026-04-08", "2026-04-09", 2, 50.0),
        market.Stretch(RQ, "2026-04-14", "2026-05-07", 18, 8598.02),
    ]
    assert dm.stretch_caption(items) == (
        "The longest falling stretch — March 18 to April 7, 15 trading days, all volatile — "
        "made +$10,536. The other falling stretches of five days or more: January 19–26 "
        "(quiet) lost $1,701 and February 5–18 (volatile) lost $1,668. The longest stretch "
        "of any kind — April 14 to May 7, 18 trading days, rising and quiet — made +$8,598.")
    rising_only = [market.Stretch(RQ, "2026-01-16", "2026-01-20", 3, 10.0)]
    assert dm.stretch_caption(rising_only).startswith("No day in this period was labeled falling.")
    alone = [market.Stretch(FV, "2026-03-18", "2026-04-07", 15, -20.0)]
    assert dm.stretch_caption(alone) == (
        "The longest falling stretch — March 18 to April 7, 15 trading days, all volatile — "
        "lost $20. No other falling stretch lasted five days or more.")


def test_transition_cells_row_percentages_and_diagonal():
    table = {a: dict.fromkeys(market.CONDITIONS, 0) for a in market.CONDITIONS}
    table[RQ].update({RQ: 8, RV: 2})
    cells = dm.transition_cells(table)
    assert [(c.count, c.share, c.diagonal) for c in cells[0]] == [
        (8, 0.8, True), (2, 0.2, False), (0, 0.0, False), (0, 0.0, False)]
    assert all(c.share is None for c in cells[1])  # a row with no days: no percentages
    assert [c.diagonal for c in cells[3]] == [False, False, False, True]


def test_link_strength_thresholds():
    assert dm.link_strength(None, None) == ("Not enough trades", "neutral")
    assert dm.link_strength(0.04, 0.64) == ("No clear link", "neutral")
    assert dm.link_strength(0.09, 0.001) == ("No clear link", "neutral")  # too small
    assert dm.link_strength(-0.15, 0.106) == ("No clear link", "neutral")  # not significant
    assert dm.link_strength(0.2, 0.01) == ("Weak positive link", "blue")
    assert dm.link_strength(-0.35, 0.001) == ("Clear negative link", "blue")


def _measure(kind, xs, rs, correlation, p_value):
    return market.EntryMeasure(measure=kind, x=tuple(xs), r_multiple=tuple(rs),
                               winner=tuple(r > 0 for r in rs), correlation=correlation,
                               p_value=p_value, clear_link=False)


def test_entry_caption():
    m = _measure("volatility", [1, 2, 3], [0.5, -1, 2], 0.04, 0.6)
    assert dm.entry_caption(m, 3) == (
        "Correlation 0.04 across 3 trades. Busier markets at entry didn't produce bigger "
        "results here.")
    m = _measure("time", [1, 2, 3], [0.5, -1, 2], -0.15, 0.1)
    assert dm.entry_caption(m, 5) == (
        "Correlation −0.15 across 3 of 5 trades (the others had too few earlier bars), not "
        "significant at the 5% level. Later entries in the trading day didn't produce "
        "smaller results here.")
    m = _measure("volume", [1, 2, 3], [0.5, -1, 2], 0.4, 0.001)
    assert dm.entry_caption(m, 3).endswith(
        "Heavier trading before entry went with bigger results here (clear positive link).")
    m = _measure("trend", [1], [0.5], None, None)
    assert "too few for a correlation" in dm.entry_caption(m, 1)


def test_build_view_and_profit_figure():
    labels = _labels({"2026-01-13": NE, "2026-01-14": RQ, "2026-01-15": RQ,
                      "2026-02-02": FV})
    calendar = list(labels.labels)
    trades = [_trade("2026-01-13", -100.0), _trade("2026-01-14", 250.0),
              _trade("2026-02-02", 40.0)]
    view = dm.build_view(labels, trades, calendar, by="exit")
    assert view.by == "exit" and view.total_trades == 3 and view.moved == 0
    assert [(c.label, c.trades, c.net) for c in view.cards if c.trades] == [
        (RQ, 1, 250.0), (FV, 1, 40.0), (NE, 1, -100.0)]
    assert dm.build_view(labels, trades, calendar, by="nonsense").by == "entry"
    fig = dm.profit_figure(view.daily, labels)
    assert list(fig.data[0].y) == [-100.0, 150.0, 150.0, 190.0]
    shading = [s for s in fig.layout.shapes if s.type == "rect"]
    assert [(s.x0, s.x1, s.fillcolor) for s in shading] == [
        (-0.5, 0.5, dm.CONDITION_COLORS[NE]), (0.5, 2.5, dm.CONDITION_COLORS[RQ]),
        (2.5, 3.5, dm.CONDITION_COLORS[FV])]
    assert list(fig.layout.xaxis.ticktext) == ["January", "February"]
    assert list(fig.layout.xaxis.tickvals) == [0, 3]


def test_entry_figure_splits_winners_and_uses_clock_ticks_for_time():
    fig = dm.entry_figure(_measure("time", [0.5, 14.5, 16.0], [2.0, -1.0, 0.0], 0.1, 0.5))
    assert [len(t.x) for t in fig.data] == [1, 2]  # winners, then losers and break-even
    assert list(fig.layout.xaxis.ticktext)[:3] == ["5:00 PM", "9:00 PM", "1:00 AM"]
    assert [a.text for a in fig.layout.annotations] == [
        "Entry time, Chicago (the trading day opens at 5:00 PM)"]
    fig = dm.entry_figure(_measure("volatility", [0.0002, 0.0003], [1.0, -1.0], 0.1, 0.5))
    assert fig.layout.xaxis.showticklabels is False
    assert fig.layout.yaxis.ticksuffix == "R"
    assert [a.text for a in fig.layout.annotations] == [
        "Calm", "Average one-minute range over the last 20 minutes", "Busy"]


def test_condition_colors_follow_the_mock():
    assert dm.CONDITION_COLORS == {RQ: "#DCE6F2", RV: "#B7CBE3", FQ: "#F6E4D6",
                                   FV: "#EBC5A8", NE: "#EFECE4"}


_COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{6}\b|rgba?\(")


def test_condition_colors_follow_the_theme_and_html_has_no_color_literal():
    """Theme contract: the constant stays light; render-time code follows the palette."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    assert dm.condition_colors() == dm.CONDITION_COLORS  # headless: the light palette
    dark = dm.condition_colors("dark")
    assert list(dark) == list(dm.CONDITION_COLORS)
    assert dark == {RQ: theme.DARK_COLORS["blue_band"], RV: theme.DARK_COLORS["blue_shade"],
                    FQ: theme.DARK_COLORS["orange_shade"], FV: theme.DARK_COLORS["orange_mid"],
                    NE: theme.DARK_COLORS["header_row"]}
    labels = _labels({"2026-01-13": NE, "2026-01-14": RQ, "2026-02-02": FV})
    view = dm.build_view(labels, [_trade("2026-01-14", 250.0)], list(labels.labels))
    cards, transitions, legend = dm._cards(view), dm._transition_card(view), dm._legend(view)
    for markup in (cards, transitions, legend):
        assert not _COLOR_LITERAL.search(str(markup)), str(markup)[:300]
    assert "background:var(--lab-panel)" in str(cards)
    assert "border-top:6px solid var(--lab-blue-band)" in str(cards)
    assert "background:var(--lab-ink);color:var(--lab-on-ink)" in str(transitions)
    assert 'style="background:var(--lab-header-row)"' in str(legend)
    theme.set_theme_resolver(lambda: "dark")
    try:
        profit = dm.profit_figure(view.daily, labels)
        entry = dm.entry_figure(_measure("trend", [-0.01, 0.02], [1.0, -1.0], 0.1, 0.5))
    finally:
        theme.set_theme_resolver(None)
    shading = [s.fillcolor for s in profit.layout.shapes if s.type == "rect"]
    assert shading == [dark[NE], dark[RQ], dark[FV]]
    assert profit.data[0].line.color == theme.DARK_COLORS["ink"]
    zero = [s for s in profit.layout.shapes if s.type == "line"]
    assert zero and zero[0].line.color == theme.DARK_COLORS["zero_line"]
    assert entry.data[0].marker.color == theme.DARK_COLORS["blue"]
    assert entry.data[1].marker.color == theme.DARK_COLORS["orange"]
    assert {s.line.color for s in entry.layout.shapes if s.type == "line"} == {
        theme.DARK_COLORS["rule"], theme.DARK_COLORS["zero_line"]}
    # with the resolver removed the figures are light again (nothing was kept at import time)
    assert dm.profit_figure(view.daily, labels).data[0].line.color == theme.COLORS["ink"]


# ── reference values (local saved study, read only) ───────────────────────

_saved = pytest.mark.skipif(
    not (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file(),
    reason="the saved funded variation study is not on this computer")


@pytest.fixture(scope="module")
def reference():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        open_funded_study,
        ordered_trades,
    )

    study = open_funded_study(STORE, RESULT_ID)
    root = market.study_package_root(study.plan)
    if root is None:
        pytest.skip("the verified strategy package is not on this computer")
    minutes = market.load_index_minutes(root, cutoff_utc=study.result["period"]["cutoff_utc"])
    labels = market.condition_labels(market.daily_closes(minutes), study.calendar)
    trades = ordered_trades(study, LEADER, TPT)
    return study, minutes, labels, trades


@_saved
def test_reference_cards_lines_and_caption(reference):
    study, _, labels, trades = reference
    view = dm.build_view(labels, trades, study.calendar, by="entry")
    cards = {c.label: (c.days, c.trades, dm.fmt.money_whole(c.net, signed=True),
                       dm.fmt.percent(c.share_of_trades), dm.fmt.percent(c.win_rate),
                       dm.fmt.money_whole(c.average)) for c in view.cards}
    assert cards[RQ] == (39, 41, "+$13,546", "36%", "63%", "$330")
    assert cards[RV] == (18, 14, "+$15,844", "12%", "57%", "$1,132")
    assert cards[FQ] == (13, 15, "+$56", "13%", "53%", "$4")
    assert cards[FV] == (34, 39, "+$8,949", "34%", "62%", "$229")
    assert dm.definition_sentence(labels).endswith(
        "Labels start January 16, 2026; the first days lack enough history.")
    assert dm.history_line(view.cards, labels, view.total_trades) == (
        f"Not enough history (January 13–15): 5 trades, {MINUS}$664. "
        "Shares are of 114 funded trades.")
    assert view.moved == 0 and view.outside == 0
    exit_view = dm.build_view(labels, trades, study.calendar, by="exit")
    assert [(c.label, c.trades, c.net) for c in exit_view.cards] == [
        (c.label, c.trades, c.net) for c in view.cards]
    caption = dm.stretch_caption(view.stretches)
    assert caption.startswith("The longest falling stretch — March 18 to April 7, 15 trading "
                              "days, all volatile — made +$10,536.")
    assert ("January 19–26 (quiet) lost $1,701 and February 5–18 (volatile) lost $1,668"
            in caption)
    rows = dm.transition_cells(view.transitions)
    assert [(c.count, dm.fmt.percent(c.share)) for c in rows[0]] == [
        (32, "82%"), (3, "8%"), (2, "5%"), (2, "5%")]
    assert [(c.count, dm.fmt.percent(c.share)) for c in rows[3]] == [
        (0, "0%"), (5, "15%"), (1, "3%"), (27, "82%")]
    fig = dm.profit_figure(view.daily, labels)
    assert len(fig.data[0].y) == 107
    assert round(fig.data[0].y[-1], 2) == round(sum(float(t["net_pnl_usd"]) for t in trades), 2)


def test_unavailable_bars_are_not_cached(monkeypatch):
    """Review fix: bars missing on first open, then restored, show without a restart."""

    import uuid
    from types import SimpleNamespace

    import ifvg_lab_cache
    import ifvg_lab_ui

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_data

    bars: dict[str, object] = {"frame": None}  # the package lookup: first missing
    monkeypatch.setattr(ifvg_lab_cache, "index_minutes", lambda *_args: bars["frame"])
    monkeypatch.setattr(ifvg_lab_ui, "funded_study",
                        lambda *_args: SimpleNamespace(calendar=["2026-01-13"]))
    monkeypatch.setattr(funded_data, "ordered_trades", lambda *_args: [])
    monkeypatch.setattr(market, "daily_closes", lambda minutes: f"closes of {minutes}")
    monkeypatch.setattr(market, "condition_labels", lambda closes, _calendar: f"labels {closes}")
    monkeypatch.setattr(market, "entry_measure",
                        lambda minutes, _trades, measure: f"{measure} of {minutes}")
    monkeypatch.setattr(dm, "build_view", lambda labels, *_args, by: f"view by {by}: {labels}")
    key = (f"cache-test-{uuid.uuid4().hex}", "result")
    assert dm._labels(*key) is None
    assert dm._view(*key, "cfg", "takeprofittrader", "entry") is None
    assert dm._entry(*key, "cfg", "takeprofittrader", "volatility") is None
    bars["frame"] = "bars"  # the package is back
    assert dm._labels(*key) == "labels closes of bars"
    assert dm._view(*key, "cfg", "takeprofittrader", "entry") == (
        "view by entry: labels closes of bars")
    assert dm._entry(*key, "cfg", "takeprofittrader", "volatility") == "volatility of bars"
    bars["frame"] = None  # once read, the labels stay cached for the process
    assert dm._labels(*key) == "labels closes of bars"


def _tab_page(store_root, result_id, options, fake_calendar=None):
    """AppTest page: the header switch and the tab for one configuration at one firm.

    Runs in this test process, where ``scripts/`` is already importable.
    """

    from types import SimpleNamespace

    import ifvg_lab_cache
    import ifvg_lab_detail_market as dm
    import streamlit as st

    context = st.session_state.setdefault("ctx_dict", dict(options))
    original = ifvg_lab_cache.index_minutes
    try:
        if fake_calendar is not None:  # stored bars missing: a stand-in study, no records read
            ifvg_lab_cache.index_minutes = lambda *args: None
            study = SimpleNamespace(calendar=list(fake_calendar))
        else:
            from ifvg_lab_ui import funded_study

            study = funded_study(store_root, result_id)
        ctx = SimpleNamespace(study=study, store_root=store_root, result_id=result_id,
                              configuration="S1-T1-H14-P1-L-SO", firm_key="takeprofittrader",
                              context=context)
        dm.header_right(st, ctx)
        dm.render(st, ctx)
    finally:
        ifvg_lab_cache.index_minutes = original


def _html(at) -> str:
    import html

    return html.unescape(" ".join(node.proto.body for node in at.get("html")))


def test_tab_says_what_is_missing_without_stored_bars():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_tab_page, args=(str(REPO / "no-such-store"),
                                                "missing-bars-test", {}, ["2026-01-13"]),
                               default_timeout=60)
    at.run()
    assert not at.exception
    assert ("The stored one-minute E-mini bars for this study aren't available, so its days "
            "can't be labeled by market condition.") in _html(at)
    assert not at.get("plotly_chart")


@_saved
def test_tab_renders_exit_day_and_time_of_day_in_streamlit(reference):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(
        _tab_page, args=(str(STORE), RESULT_ID, {"market_by": "exit", "market_measure": "time"}),
        default_timeout=120)
    at.run()
    assert not at.exception
    body = _html(at)
    for text in ("Rising · quiet", "Not enough history (January 13–15): 5 trades",
                 "labeling by entry day or by exit day gives the same counts",
                 "Trading profit, shaded by market condition", "How long each condition lasts",
                 "Time of day at entry vs result", "No clear link"):
        assert text in body, text
    assert len(at.get("plotly_chart")) == 2
    assert at.session_state["ctx_dict"]["market_by"] == "exit"


@_saved
def test_reference_entry_measure_caption(reference):
    _, minutes, _, trades = reference
    measure = market.entry_measure(minutes, trades, measure="volatility")
    assert dm.link_strength(measure.correlation, measure.p_value) == ("No clear link",
                                                                     "neutral")
    assert dm.entry_caption(measure, len(trades)) == (
        "Correlation 0.04 across 114 trades. Busier markets at entry didn't produce bigger "
        "results here.")
