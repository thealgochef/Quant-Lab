"""Setup events are known when they complete (closeout review follow-up, item 1).

The strategy engine records a tap (a candle's high/low reaching the gap) and a close
through the opposing gap (a candle's body close) at that candle's close; no earlier
observation is saved. A gap is known at its confirmation (its third candle's close). So
at every point in time, Trade review's "How the setup formed" text, the setup chart's
key and markers, and the moment picker must use one instant per event: its completion.
A candle's opening time only names the candle ("the 7:05 PM candle").

Synthetic records only (the ``_record`` fixture and ``ROW`` of test_ac_trade_review);
no saved study or market data is read. Every expectation pins the absolute instant
(``record.bar_close``), so moving the text, key and moments together to the candle's
opening time would still fail. These tests use only functions that exist before and
after the correction, so the uncorrected source fails them as ordinary assertions.

Cases (``SETUP_TIMING_CASES``): the owner's boundary (an own record whose opposing gap is
confirmed at 7:05 PM Chicago, the minute its close-through candle opens; that candle
closes at 7:06 PM), one nanosecond before each completion, at each completion, a related
record at 7:05 PM, one nanosecond before 7:06 PM, 7:06 PM and the entry, and records whose
completion time was not recorded. Further tests: every offered moment and each completion's
edges for text, key and chart together; full-history times; missing completion evidence in
full history; a tap with no recorded candle; a record without its higher-timeframe gap; a
close-through recorded without its opening minute; the page's "How the setup formed" card;
and the April anchors.
"""

from __future__ import annotations

import dataclasses
import types

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_chart as rc
from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_panels as rp
from tests.agents.ifvg_lab.test_ac_trade_review import (
    ROW,
    _bars,
    _FakeSt,
    _markup_text,
    _record,
    _utc,
)

#: the owner's reproduction: opposing gap confirmed 7:05 PM (00:05Z); the close-through
#: candle opens 7:05 PM and closes 7:06 PM (00:06Z); entry 7:07 PM
BOUNDARY = {"opposing_confirmed": "2026-04-13T00:05:00Z", "inversion": "2026-04-13T00:05:00Z"}
RELATED = {"link": "same_execution", "configuration": "S1_D80_W1_P1"}
ONE_NS = pd.Timedelta(1, "ns")
DAY_CLOSE = "2026-04-13T21:00:00Z"  # 4:00 PM Chicago, the last moment offered
NOT_RECORDED = "(its close was not recorded)"
TIMES_NOTE = "Each time is when that step became known"


def _make(inputs: dict):
    record = _record(inputs.get("link", "exact"), inputs.get("configuration", "CFG"),
                     **inputs.get("record", {}))
    for bar in inputs.get("missing_close", ()):  # completion evidence not recorded
        record = dataclasses.replace(record, bars={**record.bars,
                                                   bar: (record.bar_open(bar), None)})
    for bar in inputs.get("missing_open", ()):
        record = dataclasses.replace(record, bars={**record.bars,
                                                   bar: (None, record.bar_close(bar))})
    for bar in inputs.get("no_times", ()):
        record = dataclasses.replace(record, bars={k: v for k, v in record.bars.items()
                                                   if k != bar})
    if inputs.get("without_htf"):
        record = dataclasses.replace(record, gaps={k: v for k, v in record.gaps.items()
                                                   if k != "htf"})
    return record


def _is_tap(step) -> bool:
    return "taps into" in step.text


def _is_close_through(step) -> bool:
    return "closes through it" in step.text


def _chart(record, view, moment):
    """The setup chart at ``moment`` with the key it is drawn for: (marker numbers, band)."""

    key = rp.setup_key(record, view, moment=moment)
    bars = _bars("2026-04-12T23:40:00Z", 60, base=24_960.0, trading_day="2026-04-13")
    figure, _window = rc.setup_figure(bars, view, record, moment=moment, key=key)
    numbers = [t.text[0] for t in figure.data if getattr(t, "mode", None) == "markers+text"]
    opened, closed = record.bar_open("inversion_bar"), record.bar_close("inversion_bar")
    band = (opened is not None and closed is not None
            and any(s.yref == "paper" and s.x0 == rc._wall(opened) and s.x1 == rc._wall(closed)
                    for s in figure.layout.shapes))
    return key, numbers, band


SETUP_TIMING_CASES = [
    {"id": "own_7_05_pm_opposing_moment_comes_before_the_close_through",
     "inputs": {"record": BOUNDARY, "cursor": "2026-04-13T00:05:00Z"},
     "expected": {"selectable": True, "moment_label": "7:05 PM · the opposing gap has formed",
                  "step_numbers": [1, 2, 3, 4], "close_through_step": False,
                  "key_roles": ["parent", "opposing"], "chart_numbers": ["1", "2"],
                  "close_through_band": False}},
    {"id": "own_one_nanosecond_before_the_close_through_completes",
     "inputs": {"record": BOUNDARY, "cursor": "2026-04-13T00:05:59.999999999Z"},
     "expected": {"selectable": False, "step_numbers": [1, 2, 3, 4],
                  "close_through_step": False, "key_roles": ["parent", "opposing"],
                  "chart_numbers": ["1", "2"], "close_through_band": False}},
    {"id": "own_at_the_close_through_7_06_pm",
     "inputs": {"record": BOUNDARY, "cursor": "2026-04-13T00:06:00Z"},
     "expected": {"selectable": True,
                  "moment_label": "7:06 PM · the 7:05 PM candle has closed through the "
                                  "opposing gap",
                  "step_numbers": [1, 2, 3, 4, 5], "close_through_step": True,
                  "close_through_at": "2026-04-13 00:06:00+00:00",
                  "close_through_time": "7:06 PM",
                  "key_roles": ["parent", "opposing", "inversion"],
                  "chart_numbers": ["1", "2", "3"], "close_through_band": True}},
    {"id": "own_tap_one_nanosecond_before_its_candle_closes",
     "inputs": {"record": {}, "cursor": "2026-04-12T22:18:59.999999999Z"},
     "expected": {"selectable": False, "step_numbers": [1], "tap_step": False,
                  "key_roles": []}},
    {"id": "own_tap_at_its_candle_close_5_19_pm",
     "inputs": {"record": {}, "cursor": "2026-04-12T22:19:00Z"},
     "expected": {"selectable": True,
                  "moment_label": "5:19 PM · the 5:18 PM candle has tapped the four-hour gap",
                  "step_numbers": [1, 2], "tap_step": True,
                  "tap_at": "2026-04-12 22:19:00+00:00", "tap_time": "Apr 12, 5:19 PM",
                  "key_roles": []}},
    {"id": "related_record_at_7_05_pm_hides_the_close_through",
     "inputs": {**RELATED, "record": BOUNDARY, "cursor": "2026-04-13T00:05:00Z"},
     "expected": {"selectable": False, "step_numbers": [1, 2, 3, 4],
                  "close_through_step": False, "key_roles": ["parent", "opposing"],
                  "chart_numbers": ["1", "2"], "close_through_band": False}},
    {"id": "related_record_one_nanosecond_before_the_close_through_completes",
     "inputs": {**RELATED, "record": BOUNDARY, "cursor": "2026-04-13T00:05:59.999999999Z"},
     "expected": {"selectable": False, "step_numbers": [1, 2, 3, 4],
                  "close_through_step": False, "key_roles": ["parent", "opposing"],
                  "chart_numbers": ["1", "2"], "close_through_band": False}},
    {"id": "related_record_at_the_close_through_7_06_pm",
     "inputs": {**RELATED, "record": BOUNDARY, "cursor": "2026-04-13T00:06:00Z"},
     "expected": {"selectable": False, "step_numbers": [1, 2, 3, 4, 5],
                  "close_through_step": True,
                  "close_through_at": "2026-04-13 00:06:00+00:00",
                  "close_through_time": "7:06 PM",
                  "key_roles": ["parent", "opposing", "inversion"],
                  "chart_numbers": ["1", "2", "3"], "close_through_band": True}},
    {"id": "related_record_at_the_entry_7_07_pm",
     "inputs": {**RELATED, "record": BOUNDARY, "cursor": "2026-04-13T00:07:00Z"},
     "expected": {"selectable": True, "moment_label": "7:07 PM · entry",
                  "step_numbers": [1, 2, 3, 4, 5, 6], "close_through_step": True,
                  "close_through_at": "2026-04-13 00:06:00+00:00",
                  "close_through_time": "7:06 PM",
                  "key_roles": ["parent", "opposing", "inversion", "entry"]}},
    {"id": "own_close_through_without_a_recorded_close_is_never_known",
     "inputs": {"record": BOUNDARY, "missing_close": ["inversion_bar"], "cursor": DAY_CLOSE},
     "expected": {"selectable": True, "close_through_step": False,
                  "closed_through_moment_offered": False,
                  "key_roles": ["parent", "opposing", "entry", "half"]}},
    {"id": "own_tap_without_a_recorded_close_is_never_known",
     "inputs": {"record": {}, "missing_close": ["tap_bar"], "cursor": DAY_CLOSE},
     "expected": {"selectable": True, "tap_step": False, "tapped_moment_offered": False}},
]


def setup_timing_actual(case: dict) -> dict:
    """What text, key, chart and moments show at the case's cursor (see the module docs)."""

    inputs = case["inputs"]
    record, view = _make(inputs), rp.TradeView.from_row(ROW)
    cursor = _utc(inputs["cursor"])
    steps = rp.setup_steps(record, view, distance_cap=80, moment=cursor)
    key = rp.setup_key(record, view, moment=cursor)
    offered = {m.at: m.label for m in rp.moments(view, record)}
    tap = [s for s in steps if _is_tap(s)]
    through = [s for s in steps if _is_close_through(s)]
    actual = {
        "selectable": cursor in offered,
        "moment_label": offered.get(cursor),
        "step_numbers": [s.number for s in steps],
        "tap_step": bool(tap), "close_through_step": bool(through),
        "tap_at": str(tap[0].at) if tap else None,
        "tap_time": tap[0].time_text if tap else None,
        "close_through_at": str(through[0].at) if through else None,
        "close_through_time": through[0].time_text if through else None,
        "key_roles": [role for role, _text, _at in key],
        "closed_through_moment_offered": any("closed through" in label
                                             for label in offered.values()),
        "tapped_moment_offered": any("tapped" in label for label in offered.values()),
    }
    if "chart_numbers" in case["expected"]:
        _key, actual["chart_numbers"], actual["close_through_band"] = _chart(record, view, cursor)
    return actual


@pytest.mark.parametrize("case", SETUP_TIMING_CASES, ids=[c["id"] for c in SETUP_TIMING_CASES])
def test_setup_timing_cases(case):
    actual = setup_timing_actual(case)
    assert {k: actual[k] for k in case["expected"]} == case["expected"]


def _instants(record, view) -> list[pd.Timestamp]:
    """Every offered moment, plus each candle event's opening, completion − 1 ns and completion."""

    found = {m.at for m in rp.moments(view, record)}
    for bar in ("tap_bar", "inversion_bar"):
        opened, closed = record.bar_open(bar), record.bar_close(bar)
        found |= {opened, closed - ONE_NS, closed}
    return sorted(found)


@pytest.mark.parametrize("inputs", [
    {"record": BOUNDARY},
    {"record": {}},
    {**RELATED, "record": BOUNDARY},
], ids=["own_boundary_record", "own_default_record", "related_boundary_record"])
def test_text_key_chart_and_moments_know_each_candle_event_at_its_close(inputs):
    """At every offered moment and at each completion's edges: a candle event is shown exactly
    when the cursor has reached that candle's recorded close, in the text, the key and the
    chart (its number and its shaded candle) alike, and no shown step is dated after the
    cursor."""

    record, view = _make(inputs), rp.TradeView.from_row(ROW)
    tap_close, through_close = record.bar_close("tap_bar"), record.bar_close("inversion_bar")
    for at in _instants(record, view):
        steps = rp.setup_steps(record, view, distance_cap=80, moment=at)
        key, numbers, band = _chart(record, view, at)
        roles = [role for role, _text, _at in key]
        in_text = any(_is_close_through(s) for s in steps)
        in_key = "inversion" in roles
        in_chart = in_key and str(roles.index("inversion") + 1) in numbers
        assert in_text == in_key == in_chart == band == (at >= through_close), at
        assert any(_is_tap(s) for s in steps) == (at >= tap_close), at
        assert all(s.at is not None and s.at <= at for s in steps), at
        assert all(item_at is not None and item_at <= at for _role, _text, item_at in key), at


def test_full_history_dates_each_candle_event_at_its_close_and_names_the_candle_by_its_open():
    """Full history: the time next to a tap or a close-through is its candle's close (when it
    became known); the candle is named by its opening minute, in the steps and in the key. The
    same instant is the key's and the moment picker's."""

    record, view = _make({"record": BOUNDARY}), rp.TradeView.from_row(ROW)
    steps = rp.setup_steps(record, view, distance_cap=80)
    tap = next(s for s in steps if _is_tap(s))
    through = next(s for s in steps if _is_close_through(s))
    assert tap.at == record.bar_close("tap_bar") == _utc("2026-04-12T22:19:00Z")
    assert tap.time_text == "Apr 12, 5:19 PM" and tap.text == "The 5:18 PM candle taps into it"
    assert through.at == record.bar_close("inversion_bar") == _utc("2026-04-13T00:06:00Z")
    assert through.time_text == "7:06 PM"
    assert through.text == "The 7:05 PM candle closes through it"
    key = {role: (text, at) for role, text, at in rp.setup_key(record, view)}
    assert key["inversion"] == ("The 7:05 PM candle closes through it", through.at)
    labels = {m.label: m.at for m in rp.moments(view, record)}
    assert labels["7:06 PM · the 7:05 PM candle has closed through the opposing gap"] == through.at
    assert labels["5:19 PM · the 5:18 PM candle has tapped the four-hour gap"] == tap.at
    # every other step keeps its availability instant: gaps at confirmation, entry at entry
    by_number = {s.number: s for s in steps}
    assert by_number[1].at == record.htf.confirmed_utc
    assert by_number[3].at == record.parent.confirmed_utc
    assert by_number[4].at == record.opposing.confirmed_utc
    assert by_number[6].at == view.entry_utc and by_number[6].time_text == "7:07 PM"


@pytest.mark.parametrize("bar", ["tap_bar", "inversion_bar"])
def test_missing_completion_evidence_is_never_dated_by_the_candle_open(bar):
    """Without a recorded close, the event is never known at any moment and offers no moment.
    Full history still lists it, named by its candle, marked "(its close was not recorded)",
    with no time; the numbering is unchanged."""

    record = _make({"record": BOUNDARY, "missing_close": [bar]})
    view = rp.TradeView.from_row(ROW)
    is_event = _is_tap if bar == "tap_bar" else _is_close_through
    for m in rp.moments(view, record):
        assert not any(is_event(s) for s in rp.setup_steps(record, view, moment=m.at)), m
    words = "tapped" if bar == "tap_bar" else "closed through"
    assert not any(words in m.label for m in rp.moments(view, record))
    steps = rp.setup_steps(record, view)  # full history
    events = [s for s in steps if is_event(s)]
    assert [s.number for s in steps] == [1, 2, 3, 4, 5, 6]
    assert len(events) == 1
    event = events[0]
    candle = "5:18 PM" if bar == "tap_bar" else "7:05 PM"
    assert event.at is None and event.time_text == fmt.MISSING
    assert event.text.startswith(f"The {candle} candle") and event.text.endswith(NOT_RECORDED)


def test_full_history_key_and_chart_agree_when_a_close_through_close_is_missing():
    """Full history lists a close-through without a recorded close in the key; the chart marks
    it (on its candle, without a shaded completion) so every key number has its marker."""

    record = _make({"record": BOUNDARY, "missing_close": ["inversion_bar"]})
    view = rp.TradeView.from_row(ROW)
    key, numbers, band = _chart(record, view, None)
    roles = [role for role, _text, _at in key]
    assert "inversion" in roles and not band
    assert {r: text for r, text, _at in key}["inversion"].endswith(NOT_RECORDED)
    assert str(roles.index("inversion") + 1) in numbers


def test_a_tap_with_no_recorded_candle_is_listed_but_never_known():
    """A record with its higher-timeframe gap but no tap-candle times still lists the tap in
    full history (not recorded, no time) and never shows it at a moment."""

    record = _make({"record": BOUNDARY, "no_times": ["tap_bar"]})
    view = rp.TradeView.from_row(ROW)
    steps = rp.setup_steps(record, view)
    taps = [s for s in steps if _is_tap(s)]
    assert [s.number for s in steps] == [1, 2, 3, 4, 5, 6]
    assert len(taps) == 1 and taps[0].at is None and taps[0].time_text == fmt.MISSING
    assert taps[0].text.endswith(NOT_RECORDED)
    for m in rp.moments(view, record):
        assert not any(_is_tap(s) for s in rp.setup_steps(record, view, moment=m.at)), m


def test_a_record_without_its_higher_timeframe_gap_still_shows_its_tap_at_the_close():
    """The moment picker offers a tap even without the higher-timeframe gap; the steps show
    the same tap at the same instant."""

    record = _make({"record": BOUNDARY, "without_htf": True})
    view = rp.TradeView.from_row(ROW)
    close = _utc("2026-04-12T22:19:00Z")
    labels = {m.at: m.label for m in rp.moments(view, record)}
    assert labels[close] == "5:19 PM · the 5:18 PM candle has tapped the higher-timeframe gap"
    before = rp.setup_steps(record, view, moment=close - ONE_NS)
    at = rp.setup_steps(record, view, moment=close)
    assert not any(_is_tap(s) for s in before)
    taps = [s for s in at if _is_tap(s)]
    assert len(taps) == 1
    tap = taps[0]
    assert tap.text == "The 5:18 PM candle taps into the higher-timeframe gap"
    assert tap.at == close and tap.time_text == "Apr 12, 5:19 PM"


def test_a_close_through_recorded_without_its_opening_minute_agrees_in_text_key_and_chart():
    """With only the close recorded, the close-through is known at that close; text, key and
    chart all show it from then on (the chart's number sits on the close)."""

    record = _make({"record": BOUNDARY, "missing_open": ["inversion_bar"]})
    view = rp.TradeView.from_row(ROW)
    close = _utc("2026-04-13T00:06:00Z")
    for at, shown in ((close - ONE_NS, False), (close, True), (None, True)):
        steps = rp.setup_steps(record, view, moment=at)
        key, numbers, _band = _chart(record, view, at)
        roles = [role for role, _text, _at in key]
        assert any(_is_close_through(s) for s in steps) == ("inversion" in roles) == shown, at
        if shown:
            assert str(roles.index("inversion") + 1) in numbers, at
            assert next(s for s in steps if _is_close_through(s)).at == close
    assert "the — candle" not in " ".join(m.label for m in rp.moments(view, record))


def test_the_formed_card_hides_the_close_through_until_its_candle_closes_and_explains_times(
        monkeypatch):
    """The page's "How the setup formed" card follows the same instant as the key: absent at the
    selectable 7:05 PM moment, present from 7:06 PM (own record); it says what the times mean."""

    import ifvg_lab_trade_review as review
    import ifvg_lab_ui

    shown: list[str] = []
    monkeypatch.setattr(ifvg_lab_ui, "show",
                        lambda markup, st_module=None: shown.append(str(markup)))
    study = types.SimpleNamespace(plan=object())
    seen = {}
    for inputs in ({"record": BOUNDARY}, {**RELATED, "record": BOUNDARY}):
        record, view = _make(inputs), rp.TradeView.from_row(ROW)
        for label, at in (("7:05 PM", "2026-04-13T00:05:00Z"),
                          ("7:06 PM", "2026-04-13T00:06:00Z")):
            shown.clear()
            review._formed_card(_FakeSt(), study, "CFG", view, record, moment=_utc(at))
            text = _markup_text("\n".join(shown))
            assert rp.formed_card_title(record) in text and TIMES_NOTE in text
            seen[(record.link, label)] = "closes through it" in text
    assert seen == {("exact", "7:05 PM"): False, ("exact", "7:06 PM"): True,
                    ("same_execution", "7:05 PM"): False, ("same_execution", "7:06 PM"): True}


def test_the_april_anchors_are_kept():
    """The actual April trade's entry (7:07 PM), default cursor (7:10 PM) and the hidden half
    exit at 7:10:27.251840803 PM stay exactly as recorded."""

    view = rp.TradeView.from_row(ROW)
    cursor = rp.default_moment(view)
    assert view.entry_utc == _utc("2026-04-13T00:07:00Z") and rp.clock(view.entry_utc) == "7:07 PM"
    assert cursor == _utc("2026-04-13T00:10:00Z") and rp.clock(cursor) == "7:10 PM"
    assert view.half_utc.value == 1776039027251840803 and view.half_utc > cursor
    record = _make({"record": BOUNDARY})
    key = rp.setup_key(record, view, moment=cursor)
    assert [role for role, _text, _at in key] == ["parent", "opposing", "inversion", "entry"]
    steps = rp.setup_steps(record, view, distance_cap=80, moment=cursor)
    assert [s.number for s in steps] == [1, 2, 3, 4, 5, 6]
