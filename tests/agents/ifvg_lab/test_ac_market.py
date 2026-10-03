"""Analytical correction A1: market-condition labels known at entry versus retrospective.

The retrospective labels (``retrospective_daily_close_v1``) describe trading day D from
its own close and a whole-study threshold, so they were not known at entry. The
entry-known labels (``entry_known_prior_closes_v1``) use only closes completed before
D's 5:00 PM Chicago open. These tests use a small synthetic series (module constants
below, for the lead's evidence file) and, where it is on this computer, the saved
reference study through its verified package reader (read only).
"""

from __future__ import annotations

import html
import sys
import uuid
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import ifvg_lab_detail_market as dm  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.lab import market  # noqa: E402

STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
TPT = "takeprofittrader"
CHICAGO = "America/Chicago"
NE = market.NOT_ENOUGH
MINUS = "−"

RETROSPECTIVE_TEXT = (
    "Retrospective labels (retrospective_daily_close_v1): each trading day is labeled from "
    "its own last close (4:00 PM, earlier on a shortened day) against the close 10 trading "
    "days earlier, and its volatility against the median of the whole study. They describe "
    "the day with hindsight; they were not known when a trade was entered.")
ENTRY_KNOWN_TEXT = (
    "Labels known at entry (entry_known_prior_closes_v1): each trading day is labeled only "
    "from closes completed before its 5:00 PM open — the last completed close against the "
    "one 10 trading days earlier, and the last 10 daily changes against the median of "
    "that day's and earlier days' measures. An evening entry belongs to the next day's "
    "trading day. Days with fewer than 11 earlier closes or 10 volatility measures up to "
    "that day show \"Not enough history\", which is not a zero.")

# ── synthetic cases (module constants for the lead's evidence file) ────────

#: 40 stored trading days, Monday February 2 to Friday March 27, 2026 (weekends fall
#: between them; daylight saving starts Sunday March 8), each closing at 4:00 PM Chicago.
#: Calm days, then a volatile fall (February 23 – March 6), then a calm rise.
A1_BASE = {
    "days": [d.date().isoformat() for d in pd.bdate_range("2026-02-02", periods=40)],
    "closes": [
        20000.0, 20040.0, 20020.0, 20060.0, 20040.0, 20080.0, 20060.0, 20100.0, 20080.0,
        20120.25, 20100.25, 20140.5, 20120.25, 20160.5, 20140.25, 19737.5, 19974.25,
        19614.75, 19811.0, 19513.75, 19904.0, 19665.25, 19822.5, 19624.25, 19918.5,
        19998.25, 20018.25, 19978.25, 20098.0, 20128.25, 20108.0, 20208.5, 20249.0,
        20188.25, 20269.0, 20289.25, 20350.0, 20309.25, 20349.75, 20370.0],
    "close_clock": "4:00 PM America/Chicago on each trading day's date",
}
#: day D: Tuesday March 17, 2026 (31 earlier stored days), opening Monday 5:00 PM Chicago
A1_DAY = "2026-03-17"

A1_CASES = [
    {"id": "A1-append-future-days",
     "inputs": {"day": A1_DAY, "change": {"kind": "append ten later trading days", "closes": {
         "2026-03-30": 21000.0, "2026-03-31": 19000.0, "2026-04-01": 21500.0,
         "2026-04-02": 18500.0, "2026-04-03": 22000.0, "2026-04-06": 18000.0,
         "2026-04-07": 22500.0, "2026-04-08": 17500.0, "2026-04-09": 23000.0,
         "2026-04-10": 17000.0}}},
     "expected": {"label": "Rising · quiet", "labels_through_day_unchanged": True}},
    {"id": "A1-change-own-close",
     "inputs": {"day": A1_DAY, "change": {"kind": "change D's own close (later same-day "
                                                  "prices)", "closes": {A1_DAY: 19000.0}}},
     "expected": {"label": "Rising · quiet", "labels_through_day_unchanged": True,
                  "retrospective_label_before": "Rising · quiet",
                  "retrospective_label_after": "Falling · volatile"}},
    {"id": "A1-change-later-volatility",
     "inputs": {"day": A1_DAY, "change": {"kind": "alternate ±5% moves on every day after D",
                                          "closes": {
         "2026-03-18": 21219.0, "2026-03-19": 20158.0, "2026-03-20": 21166.0,
         "2026-03-23": 20107.75, "2026-03-24": 21113.25, "2026-03-25": 20057.5,
         "2026-03-26": 21060.5, "2026-03-27": 20007.5}}},
     "expected": {"label": "Rising · quiet", "labels_through_day_unchanged": True,
                  "later_entry_known_labels_changed": True,
                  "retrospective_threshold_changed": True}},
    {"id": "A1-daytime-and-evening-entry",
     "inputs": {"day": A1_DAY, "trades": [
         {"trading_day": A1_DAY, "entry_utc": "2026-03-17T10:00:00-05:00",
          "exit_utc": "2026-03-17T11:00:00-05:00", "net_pnl_usd": 120.0,
          "initial_risk_usd": 100.0},
         {"trading_day": A1_DAY, "entry_utc": "2026-03-16T19:07:00-05:00",
          "exit_utc": "2026-03-16T21:00:00-05:00", "net_pnl_usd": -80.0,
          "initial_risk_usd": 100.0}]},
     "expected": {"entry_trading_days": [A1_DAY, A1_DAY], "label": "Rising · quiet",
                  "label_known_by_each_entry": True}},
    {"id": "A1-insufficient-history",
     "inputs": {"trades": [
         {"trading_day": "2026-02-05", "net_pnl_usd": -200.0,
          "exit_utc": "2026-02-05T15:00:00-06:00"},
         {"trading_day": "2026-02-23", "net_pnl_usd": 50.0,
          "exit_utc": "2026-02-23T15:00:00-06:00"},
         {"trading_day": A1_DAY, "net_pnl_usd": 300.0,
          "exit_utc": "2026-03-17T15:00:00-05:00"}]},
     "expected": {"not_enough_days": 20, "first_labeled": "2026-03-02",
                  "not_enough_card": {"trades": 2, "net": -150.0},
                  "condition_card_trades": 1,
                  "history_line": (f"Not enough history (February 2–27): 2 trades, "
                                   f"{MINUS}$150. Shares are of 3 funded trades.")}},
    {"id": "A1-instant-guard",
     "inputs": {"day": A1_DAY, "previous_day": "2026-03-16",
                "recorded_close_chicago": ["2026-03-16T17:00:00", "2026-03-16T17:01:00",
                                           "2026-03-16T16:59:59.999999999"]},
     "expected": {"excluded": [True, True, False],
                  "last_input_day": ["2026-03-13", "2026-03-13", "2026-03-16"]}},
    {"id": "A1-page-entry-day-only",
     "inputs": {"context": {"market_by": "exit"}, "switch_to": "entry_known", "trades": [
         {"trading_day": "2026-03-09", "entry_utc": "2026-03-09T14:00:00-05:00",
          "exit_utc": "2026-03-09T17:30:00-05:00", "net_pnl_usd": 400.0,
          "initial_risk_usd": 100.0}]},
     "expected": {"entry_day_label": "Falling · volatile", "exit_day_label": "Rising · volatile",
                  "card_with_the_trade": "Falling · volatile",
                  "remembered_market_by": "exit"}},
]

INVARIANCE_CASES = ("A1-append-future-days", "A1-change-own-close",
                    "A1-change-later-volatility")


def _case(case_id: str) -> dict:
    return next(c for c in A1_CASES if c["id"] == case_id)


def _close_instant(day: str) -> pd.Timestamp:
    return pd.Timestamp(f"{day} 16:00", tz=CHICAGO).tz_convert("UTC")


def case_series(case: dict | None = None) -> tuple[pd.Series, pd.Series, list[str]]:
    """Daily closes, close instants and calendar of ``A1_BASE`` with a case's change applied."""

    closes = dict(zip(A1_BASE["days"], A1_BASE["closes"], strict=True))
    change = ((case or {}).get("inputs") or {}).get("change") or {}
    closes.update(change.get("closes") or {})
    days = sorted(closes)
    return (pd.Series([closes[d] for d in days], index=days, dtype=float),
            pd.Series([_close_instant(d) for d in days], index=days), days)


def _html(at) -> str:
    return html.unescape(" ".join(node.proto.body for node in at.get("html")))


# ── the entry-known calculation ───────────────────────────────────────────


def test_a1_entry_known_label_matches_a_hand_computation():
    """A1: D's entry-known label, recomputed by hand from the closes completed before its open."""

    closes, times, days = case_series()
    i = days.index(A1_DAY)
    # every earlier stored day closed at 4:00 PM, before D's 5:00 PM open the evening before
    trend = "Rising" if closes.iloc[i - 1] > closes.iloc[i - 11] else "Falling"

    def measure(k: int) -> float:  # the 10 daily changes among the 11 closes before day k
        return float(closes.iloc[k - 11:k].pct_change().dropna().std(ddof=1))

    measures = [measure(k) for k in range(11, i + 1)]  # every day up to D with 11 closes
    threshold = float(pd.Series(measures).median())
    expected = f"{trend} · {'volatile' if measures[-1] > threshold else 'quiet'}"
    item = market.entry_known_days(closes, times, days)[A1_DAY]
    assert (item.inputs, item.last_input_day) == (i, days[i - 1])
    assert item.open_utc == pd.Timestamp("2026-03-16 17:00", tz=CHICAGO)
    assert item.trend == trend and item.measure == pytest.approx(measures[-1], rel=1e-12)
    assert item.measures == len(measures) and item.threshold == pytest.approx(threshold,
                                                                               rel=1e-12)
    assert item.label == expected == "Rising · quiet"


@pytest.mark.parametrize("case_id", INVARIANCE_CASES)
def test_a1_future_and_same_day_changes_leave_earlier_labels_unchanged(case_id):
    """A1: appending days, changing D's own close or later volatility changes no label up to D."""

    case = _case(case_id)
    before = market.entry_known_days(*case_series())
    closes, times, days = case_series(case)
    after = market.entry_known_days(closes, times, days)
    through = [d for d in A1_BASE["days"] if d <= A1_DAY]
    # every field (inputs, measure, threshold, label) of D and of every earlier day
    assert [after[d] for d in through] == [before[d] for d in through]
    labels = market.entry_known_labels(closes, times, days)
    assert [labels.labels[d] for d in through] == [before[d].label for d in through]
    assert labels.labels[A1_DAY] == case["expected"]["label"]
    if case["expected"].get("later_entry_known_labels_changed"):  # the change is material
        assert any(after[d].label != before[d].label
                   for d in A1_BASE["days"] if d > A1_DAY)


def test_a1_daytime_and_evening_entries_take_their_trading_days_label():
    """A1: a 10:00 AM entry on D and a 7:07 PM entry the evening before both get D's label."""

    case = _case("A1-daytime-and-evening-entry")
    trades = case["inputs"]["trades"]
    expected = case["expected"]["label"]
    for trade in trades:  # the saved trading day agrees with the entry instant
        assert market.trading_day_of(trade["entry_utc"]) == trade["trading_day"] == A1_DAY
        assert market.trade_day(trade, "entry") == A1_DAY
    for change in (None, *(_case(c) for c in INVARIANCE_CASES)):
        closes, times, days = case_series(change)
        labels = market.entry_known_labels(closes, times, days)
        item = market.entry_known_days(closes, times, days)[A1_DAY]
        assert [labels.labels[market.trade_day(t, "entry")] for t in trades] == [expected] * 2
        cards = {c.label: c.trades for c in market.condition_cards(labels, trades)}
        assert cards[expected] == 2 and cards[NE] == 0
        for trade in trades:  # the label was known by each entry: its inputs closed before
            entry = pd.Timestamp(trade["entry_utc"]).tz_convert("UTC")
            assert item.last_input_close_utc < item.open_utc <= entry


def test_a1_retrospective_labels_use_later_prices_negative_control():
    """A1 negative control: D's own close and later closes change the retrospective view."""

    base, _, days = case_series()
    before = market.condition_labels(base, days)
    own = _case("A1-change-own-close")
    closes, _, days_own = case_series(own)
    after = market.condition_labels(closes, days_own)
    assert before.labels[A1_DAY] == own["expected"]["retrospective_label_before"]
    assert after.labels[A1_DAY] == own["expected"]["retrospective_label_after"]
    later = _case("A1-change-later-volatility")
    closes, _, days_later = case_series(later)
    # the whole-study threshold moves when only days after D change
    assert market.condition_labels(closes, days_later).volatility_median != pytest.approx(
        before.volatility_median)


def test_a1_insufficient_history_is_its_own_state_not_a_zero():
    """A1: early days show "Not enough history"; their trades sit there, never in a condition."""

    case = _case("A1-insufficient-history")
    trades = case["inputs"]["trades"]
    expected = case["expected"]
    closes, times, days = case_series()
    detail = market.entry_known_days(closes, times, days)
    labels = market.entry_known_labels(closes, times, days)
    early = days[:expected["not_enough_days"]]
    assert all(labels.labels[d] == NE for d in early)
    assert labels.first_labeled == expected["first_labeled"] == days[20]
    # no measure (not a zero) before 11 completed closes; no threshold before 10 measures
    assert all(detail[d].trend is None and detail[d].measure is None for d in days[:11])
    assert all(detail[d].measure is not None and detail[d].threshold is None
               for d in days[11:20])
    assert (detail[days[19]].measures, detail[days[20]].measures) == (9, 10)
    cards = {c.label: c for c in market.condition_cards(labels, trades)}
    assert (cards[NE].trades, cards[NE].net) == (expected["not_enough_card"]["trades"],
                                                 expected["not_enough_card"]["net"])
    assert sum(cards[c].trades for c in market.CONDITIONS) == expected["condition_card_trades"]
    view = dm.build_view(labels, trades, days)
    markup = html.unescape(str(dm._cards(view)))
    assert markup.count("No trades") == 3 and "$0" not in markup  # never shown as $0
    assert dm.history_line(view.cards, labels, view.total_trades) == expected["history_line"]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_a1_a_previous_close_at_or_after_the_open_is_excluded(index):
    """A1: an earlier day whose recorded close is at or after D's 5:00 PM open isn't used."""

    case = _case("A1-instant-guard")
    recorded = case["inputs"]["recorded_close_chicago"][index]
    closes, times, days = case_series()
    times = times.copy()
    times[case["inputs"]["previous_day"]] = pd.Timestamp(recorded, tz=CHICAGO).tz_convert("UTC")
    detail = market.entry_known_days(closes, times, days)
    item = detail[A1_DAY]
    assert item.last_input_day == case["expected"]["last_input_day"][index]
    assert item.last_input_close_utc < item.open_utc
    if case["expected"]["excluded"][index]:
        assert item.inputs == 30
        # the same trend and measure as Monday's own, whose inputs end on Friday
        monday = detail[case["inputs"]["previous_day"]]
        assert (item.trend, item.measure) == (monday.trend, monday.measure)
    else:
        assert item.inputs == 31


def test_a1_trading_day_opens_and_evening_entries_on_the_chicago_clock():
    """A1: a Monday opens Sunday 5:00 PM, across daylight saving; evenings map to the next day."""

    assert market.trading_day_open_utc("2026-03-02") == pd.Timestamp("2026-03-01 23:00",
                                                                     tz="UTC")  # CST
    assert market.trading_day_open_utc("2026-03-09") == pd.Timestamp("2026-03-08 22:00",
                                                                     tz="UTC")  # CDT
    monday = market.entry_known_days(*case_series())["2026-03-16"]
    assert monday.last_input_day == "2026-03-13"  # Friday's close; the weekend adds nothing
    assert market.trading_day_of("2026-03-16T19:07:00-05:00") == "2026-03-17"
    assert market.trading_day_of("2026-03-16T16:59:00-05:00") == "2026-03-16"
    assert market.trading_day_of("2026-03-15T17:00:00-05:00") == "2026-03-16"  # Sunday
    assert market.trading_day_of(None) is None


def test_a1_label_sets_carry_their_versions():
    """A1: retrospective and entry-known labels are versioned separately."""

    closes, times, days = case_series()
    retro = market.condition_labels(closes, days)
    known = market.entry_known_labels(closes, times, days)
    assert (retro.mode, retro.version) == ("retrospective", "retrospective_daily_close_v1")
    assert (known.mode, known.version) == ("entry_known", "entry_known_prior_closes_v1")
    assert market.RETROSPECTIVE_VERSION == "retrospective_daily_close_v1"
    assert market.ENTRY_KNOWN_VERSION == "entry_known_prior_closes_v1"
    # labels built without naming a mode (earlier callers) are the retrospective kind
    plain = market.ConditionLabels(labels={}, first_labeled=None, volatility_median=None)
    assert (plain.mode, plain.version) == ("retrospective", "retrospective_daily_close_v1")
    assert known.volatility_median == market.entry_known_days(closes, times, days)[
        days[-1]].threshold


def test_a1_definition_line_and_entry_day_labelling_follow_the_label_set():
    """A1: each label set has its own definition line; known-at-entry labels by entry day only."""

    closes, times, days = case_series()
    retro = market.condition_labels(closes, days)
    known = market.entry_known_labels(closes, times, days)
    assert dm.definition_sentence(retro) == (
        f"{RETROSPECTIVE_TEXT} Labels start February 16, 2026; the first days lack enough "
        "history.")
    assert dm.definition_sentence(known) == (
        f"{ENTRY_KNOWN_TEXT} Labels start March 2, 2026; the first days lack enough history.")
    none = replace(known, labels=dict.fromkeys(days, NE), first_labeled=None)
    assert dm.definition_sentence(none) == (
        f"{ENTRY_KNOWN_TEXT} No day in this study has enough closes completed before its open "
        "in the stored bars, so none can be labeled.")
    trades = _case("A1-page-entry-day-only")["inputs"]["trades"]
    assert dm.build_view(known, trades, days, by="exit").by == "entry"
    assert dm.build_view(retro, trades, days, by="exit").by == "exit"
    assert dm.LABEL_MODES == {"retrospective": "Retrospective",
                              "entry_known": "Known at entry"}
    assert dm.ENTRY_DAY_NOTE == "Known at entry labels each trade by its entry's trading day."


# ── the tab (AppTest) ─────────────────────────────────────────────────────


def _synthetic_page(result_id, options, days, closes, trades):
    """AppTest page on synthetic stored bars: the header switches and the tab.

    One one-minute bar per trading day, closing at 4:00 PM Chicago. Nothing saved is
    read: the bar reader, the study and its trades are stand-ins for this run.
    """

    from types import SimpleNamespace

    import ifvg_lab_cache
    import ifvg_lab_detail_market as dm
    import ifvg_lab_ui
    import pandas as pd
    import streamlit as st

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_data

    stamps = [pd.Timestamp(f"{d} 16:00", tz="America/Chicago").tz_convert("UTC") for d in days]
    minutes = pd.DataFrame({
        "trading_day": list(days), "logical_close_ts_utc": stamps,
        "logical_open_ts_utc": [s - pd.Timedelta(minutes=1) for s in stamps],
        "open": closes, "high": closes, "low": closes, "close": closes,
        "volume": [100] * len(days)})
    study = SimpleNamespace(calendar=list(days))
    context = st.session_state.setdefault("ctx_dict", dict(options))
    saved = (ifvg_lab_cache.index_minutes, ifvg_lab_ui.funded_study, funded_data.ordered_trades)
    try:
        ifvg_lab_cache.index_minutes = lambda *args: minutes
        ifvg_lab_ui.funded_study = lambda *args: study
        funded_data.ordered_trades = lambda *args: [dict(t) for t in trades]
        ctx = SimpleNamespace(study=study, store_root="synthetic-a1-store", result_id=result_id,
                              configuration="synthetic", firm_key="takeprofittrader",
                              context=context)
        dm.header_right(st, ctx)
        dm.render(st, ctx)
    finally:
        ifvg_lab_cache.index_minutes, ifvg_lab_ui.funded_study, funded_data.ordered_trades = saved


def test_a1_known_at_entry_switch_changes_the_definition_and_forces_entry_day_labels():
    """A1: the "Known at entry" switch changes the definition line and labels by entry day."""

    from streamlit.testing.v1 import AppTest

    case = _case("A1-page-entry-day-only")
    trades = case["inputs"]["trades"]
    closes, times, days = case_series()
    known = market.entry_known_labels(closes, times, days)
    assert known.labels["2026-03-09"] == case["expected"]["entry_day_label"]
    assert known.labels["2026-03-10"] == case["expected"]["exit_day_label"]
    assert market.trade_day(trades[0], "exit") == "2026-03-10"
    at = AppTest.from_function(
        _synthetic_page, args=(f"a1-page-{uuid.uuid4().hex}", case["inputs"]["context"], days,
                               list(A1_BASE["closes"]), trades), default_timeout=60)
    at.run()
    assert not at.exception
    body = _html(at)
    assert RETROSPECTIVE_TEXT in body and "Labels known at entry" not in body
    assert {"ifvg_lab_v1_market_labels", "ifvg_lab_v1_market_by"} <= {r.key for r in at.radio}
    assert dm.ENTRY_DAY_NOTE not in body

    at.radio(key="ifvg_lab_v1_market_labels").set_value(case["inputs"]["switch_to"]).run()
    assert not at.exception
    body = _html(at)
    assert ENTRY_KNOWN_TEXT in body and "Retrospective labels (" not in body
    assert dm.ENTRY_DAY_NOTE in body
    assert "ifvg_lab_v1_market_by" not in {r.key for r in at.radio}  # no exit-day choice
    assert at.session_state["ctx_dict"]["market_labels"] == "entry_known"
    assert at.session_state["ctx_dict"]["market_by"] == case["expected"]["remembered_market_by"]
    by_entry = dm.build_view(known, trades, days, by="entry")
    by_exit = replace(by_entry, cards=tuple(market.condition_cards(known, trades, by="exit")))
    assert html.unescape(str(dm._cards(by_entry))) in body
    assert html.unescape(str(dm._cards(by_exit))) not in body
    card = next(c for c in by_entry.cards if c.trades)
    assert card.label == case["expected"]["card_with_the_trade"]
    assert "labeling by entry day or by exit day" not in body  # the switch's sentence is gone

    at.radio(key="ifvg_lab_v1_market_labels").set_value("retrospective").run()
    assert not at.exception
    assert at.radio(key="ifvg_lab_v1_market_by").value == "exit"  # the remembered choice
    assert RETROSPECTIVE_TEXT in _html(at)


# ── reference study (local saved records, read only) ──────────────────────

_saved = pytest.mark.skipif(
    not (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file(),
    reason="the saved funded variation study is not on this computer")

#: filled by the reference check for the lead's evidence (printed; run pytest with -s)
REAL_DATA_RECORD: dict = {}


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
    return study, minutes, ordered_trades(study, LEADER, TPT)


@_saved
def test_a1_reference_study_entry_known_labels_use_only_earlier_closes(reference):
    """A1: on the saved study, entry-known labels use only inputs before each day's open."""

    study, minutes, trades = reference
    calendar = list(study.calendar)
    closes, times = market.daily_closes(minutes), market.daily_close_times(minutes)
    retro = market.condition_labels(closes, calendar)  # unchanged retrospective reference
    assert retro.version == "retrospective_daily_close_v1" and retro.first_labeled == "2026-01-16"
    assert Counter(retro.labels.values()) == {
        "Rising · quiet": 39, "Falling · volatile": 34, "Rising · volatile": 18,
        "Falling · quiet": 13, NE: 3}
    known = market.entry_known_labels(closes, times, calendar)
    detail = market.entry_known_days(closes, times, calendar)
    assert known.version == "entry_known_prior_closes_v1" and list(known.labels) == calendar
    assert set(known.labels.values()) <= {*market.CONDITIONS, NE}
    labeled = [d for d in calendar if known.labels[d] != NE]
    assert labeled and known.first_labeled == labeled[0]
    start = calendar.index(labeled[0])
    assert labeled == calendar[start:]  # too little history only at the start
    for day in labeled:
        item = detail[day]
        assert item.last_input_day < day and item.last_input_close_utc < item.open_utc, day
        assert item.inputs >= 11 and item.measures >= 10, day
    for day in calendar:  # the same label from the stored days before D alone
        earlier = [d for d in closes.index if d < day]
        alone = market.entry_known_labels(closes[earlier], times[earlier], [day])
        assert alone.labels[day] == known.labels[day], day
    cards = market.condition_cards(known, trades)
    REAL_DATA_RECORD.update({
        "version": known.version, "first_labeled": known.first_labeled,
        "last_threshold": known.volatility_median,
        "day_counts": dict(Counter(known.labels.values())),
        "same_label_as_retrospective_days": sum(
            known.labels[d] == retro.labels[d] for d in calendar),
        "leader_takeprofittrader_cards": {c.label: {"days": c.days, "trades": c.trades,
                                                     "net": round(c.net, 2)} for c in cards}})
    print(f"\nA1 entry-known reference record: {REAL_DATA_RECORD}")
    assert sum(c.trades for c in cards) == len(trades)


def _saved_page(store_root, result_id, options):
    """AppTest page: the header switches and the tab for the reference configuration."""

    from types import SimpleNamespace

    import ifvg_lab_detail_market as dm
    import streamlit as st
    from ifvg_lab_ui import funded_study

    context = st.session_state.setdefault("ctx_dict", dict(options))
    ctx = SimpleNamespace(study=funded_study(store_root, result_id), store_root=store_root,
                          result_id=result_id, configuration="S1-T1-H14-P1-L-SO",
                          firm_key="takeprofittrader", context=context)
    dm.header_right(st, ctx)
    dm.render(st, ctx)


@_saved
def test_a1_reference_tab_renders_known_at_entry(reference):
    """A1: the saved study's tab shows the entry-known definition and the entry-day note."""

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(
        _saved_page, args=(str(STORE), RESULT_ID, {"market_labels": "entry_known",
                                                   "market_by": "exit"}),
        default_timeout=120)
    at.run()
    assert not at.exception
    body = _html(at)
    assert ENTRY_KNOWN_TEXT in body and dm.ENTRY_DAY_NOTE in body
    assert "ifvg_lab_v1_market_by" not in {r.key for r in at.radio}
    assert len(at.get("plotly_chart")) == 2
