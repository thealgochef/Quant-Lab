"""Registered entry windows through the installed Core orchestrator and reducer."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
from strategy_core.candles._ids import make_bar_id
from strategy_core.strategies.ifvg_smc.records import IfvgEmission
from strategy_core.strategies.ifvg_smc.reducer import IfvgStepInput
from strategy_core.strategies.ifvg_smc.replay import DayOrchestrator
from strategy_core.strategies.ifvg_smc.section import (
    default_ifvg_smc_section,
    ifvg_profile_hash,
)
from strategy_core.structures.fvg import Fvg, FvgState, GapDirection
from strategy_core.types import Bar, BarKind, CloseReason, Direction

from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import resolve_axis_overrides

_ET = ZoneInfo("America/New_York")
_WINDOW_ID = "ny_0700_1030"
_VALUE_ID = f"enabled_entry_sessions.{_WINDOW_ID}"


def _bar(index: int, close: datetime, ohlc: tuple[int, int, int, int]) -> Bar:
    trading_day = close.astimezone(_ET).date()
    return Bar(
        timeframe_ticks=60,
        trading_day=trading_day,
        bar_index=index,
        bar_id=make_bar_id(60, trading_day, index, BarKind.TIME),
        open_ts_utc=close - timedelta(seconds=58),
        close_ts_utc=close - timedelta(seconds=1),
        open_ticks=ohlc[0],
        high_ticks=ohlc[1],
        low_ticks=ohlc[2],
        close_ticks=ohlc[3],
        volume=1,
        trade_count=2,
        is_complete=True,
        is_partial=False,
        close_reason=CloseReason.COMPLETE,
        kind=BarKind.TIME,
        logical_open_ts_utc=close - timedelta(minutes=1),
        logical_close_ts_utc=close,
    )


def _drive_candidate(
    monkeypatch: pytest.MonkeyPatch,
    entry_ts: datetime,
    direction: Direction,
) -> tuple[DayOrchestrator, Bar, tuple[IfvgEmission, ...]]:
    """Adapt Core's synthetic tap/parent/opposing/inversion/fresh-entry golden.

    Inject only discovered structure evidence; actual orchestrator session stamps,
    reducer guards, entry fills, and trade resolution remain under test.
    """
    selected = {"enabled_entry_sessions": _VALUE_ID}
    if direction is Direction.SHORT:
        selected["enable_shorts"] = "enable_shorts.true"
    section = resolve_profile_config(
        {"section_overrides": resolve_axis_overrides(selected)}
    ).section
    orchestrator = DayOrchestrator(section=section, seed=None)
    prices = (
        (10030, 10032, 10015, 10028),
        (10028, 10031, 10022, 10029),
        (10024, 10026, 10016, 10022),
        (10020, 10021, 9992, 9995),
        (10000, 10016, 9998, 10014),
        (10015, 10018, 10010, 10016),
    )
    if direction is Direction.SHORT:
        prices = tuple((20000 - o, 20000 - low, 20000 - h, 20000 - c) for o, h, low, c in prices)
    bars = tuple(
        _bar(index, entry_ts - timedelta(minutes=5 - index), ohlc)
        for index, ohlc in enumerate(prices)
    )

    def gap(
        name: str, tf: int, low: int, high: int, confirmed: datetime, *, opposing: bool = False
    ) -> Fvg:
        bullish = direction is Direction.LONG
        if direction is Direction.SHORT:
            low, high = 20000 - high, 20000 - low
        if opposing:
            bullish = not bullish
        return Fvg(
            fvg_id=f"session-fixture:{name}",
            timeframe_seconds=tf,
            direction=GapDirection.BULLISH if bullish else GapDirection.BEARISH,
            gap_low_ticks=low,
            gap_high_ticks=high,
            size_ticks=high - low,
            a_bar_id=f"{name}:a",
            c_bar_id=f"{name}:c",
            a_open_ts_utc=confirmed - timedelta(seconds=tf * 3),
            confirmed_ts_utc=confirmed,
            trading_day=bars[0].trading_day,
        )

    htf = gap("htf", 3600, 10000, 10020, entry_ts - timedelta(hours=2))
    discoveries = {
        1: {300: (gap("parent", 300, 10010, 10018, bars[1].availability_ts_utc),)},
        3: {60: (gap("opposing", 60, 10008, 10012, bars[3].availability_ts_utc, opposing=True),)},
        5: {60: (gap("entry", 60, 10014, 10015, bars[5].availability_ts_utc),)},
    }
    reduce_step = orchestrator._reducer.step

    def scripted_structures(inp: IfvgStepInput) -> tuple[IfvgEmission, ...]:
        index = inp.bar_1m.bar_index
        return reduce_step(
            replace(
                inp,
                new_fvgs=discoveries.get(index, {}),
                htf_live=(FvgState(fvg=htf),) if index == 0 else (),
                fill_events=(),
                recent_swing_highs=(),
                recent_swing_lows=(),
            )
        )

    monkeypatch.setattr(orchestrator._reducer, "step", scripted_structures)
    emissions: tuple[IfvgEmission, ...] = ()
    for bar in bars:
        emissions = orchestrator.on_decision_bar(bar)
    return orchestrator, bars[-1], emissions


def test_registered_window_changes_entry_config_without_changing_market_sessions() -> None:
    baseline = resolve_profile_config()
    baseline_payload = baseline.section.model_dump(mode="json")
    overrides = resolve_axis_overrides({"enabled_entry_sessions": _VALUE_ID})
    custom = resolve_profile_config({"section_overrides": overrides})

    assert custom.section.doc_sessions == {_WINDOW_ID: ("07:00", "10:30")}
    assert custom.section.enabled_entry_sessions == (_WINDOW_ID,)
    assert custom.section.session_scheme == baseline.section.session_scheme
    assert custom.section.outside_session_policy == baseline.section.outside_session_policy
    assert custom.section_config_hash != baseline.section_config_hash
    assert resolve_profile_config().section.model_dump(mode="json") == baseline_payload
    assert baseline.section_config_hash == ifvg_profile_hash(default_ifvg_smc_section())


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
@pytest.mark.parametrize("day", [date(2026, 1, 14), date(2026, 7, 15)])
@pytest.mark.parametrize(
    ("wall", "allowed"),
    [
        (time(6, 59), False),
        (time(7, 0), True),
        (time(7, 30), True),  # Between the standard London and NY doc sessions.
        (time(8, 30), True),
        (time(10, 29), True),
        (time(10, 30), False),
        (time(11, 0), False),
    ],
)
def test_registered_window_gates_actual_entries_at_confirmation_close(
    monkeypatch: pytest.MonkeyPatch,
    direction: Direction,
    day: date,
    wall: time,
    allowed: bool,
) -> None:
    entry_ts = datetime.combine(day, wall, tzinfo=_ET).astimezone(UTC)
    orchestrator, entry_bar, emissions = _drive_candidate(monkeypatch, entry_ts, direction)
    candidates = [
        item.record
        for item in emissions
        if item.kind == "entry_candidate" and item.record.entry_family == "fresh_fvg_continuation"
    ]
    decisions = [item.record for item in emissions if item.kind == "eligible_decision"]

    assert len(candidates) == 1
    assert len(decisions) == int(allowed)
    assert orchestrator._reducer.active_trade_count == int(allowed)
    assert ("out_of_session" in candidates[0].block_reasons) is (not allowed)
    if allowed:
        assert decisions[0].envelope.entry_session == _WINDOW_ID
        assert candidates[0].in_doc_session == _WINDOW_ID
        assert decisions[0].envelope.profile_hash == orchestrator._profile_hash
    if wall == time(7, 30):
        assert candidates[0].in_engine_session == "none"
    if wall == time(10, 30):
        # Last print was inside the window; confirmation availability is outside.
        assert entry_bar.close_ts_utc.astimezone(_ET).time() == time(10, 29, 59)
        assert entry_bar.availability_ts_utc.astimezone(_ET).time() == time(10, 30)


@pytest.mark.parametrize("direction", [Direction.LONG, Direction.SHORT])
@pytest.mark.parametrize("day", [date(2026, 1, 14), date(2026, 7, 15)])
def test_open_trade_can_resolve_after_the_entry_window(
    monkeypatch: pytest.MonkeyPatch, direction: Direction, day: date
) -> None:
    entry_ts = datetime.combine(day, time(10, 29), tzinfo=_ET).astimezone(UTC)
    orchestrator, _, emissions = _drive_candidate(monkeypatch, entry_ts, direction)
    decision = next(item.record for item in emissions if item.kind == "eligible_decision")
    if direction is Direction.LONG:
        prices = (
            decision.entry_ticks,
            decision.target_ticks + 1,
            decision.entry_ticks - 1,
            decision.target_ticks,
        )
    else:
        prices = (
            decision.entry_ticks,
            decision.entry_ticks + 1,
            decision.target_ticks - 1,
            decision.target_ticks,
        )
    exit_bar = _bar(7, entry_ts + timedelta(minutes=2), prices)
    assert orchestrator._doc_session(exit_bar.availability_ts_utc) == "none"
    trades = [
        item.record
        for item in orchestrator.on_decision_bar(exit_bar)
        if item.kind == "executed_trade"
    ]
    assert len(trades) == 1
    assert trades[0].resolution == "target"
    assert trades[0].envelope.entry_session == _WINDOW_ID
    assert trades[0].envelope.profile_hash == orchestrator._profile_hash
    assert orchestrator._reducer.active_trade_count == 0
