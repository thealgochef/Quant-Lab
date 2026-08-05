"""Small deterministic Strategy-Core context fixture for Quant-Lab v3 tests."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
from strategy_core.candles._ids import make_bar_id
from strategy_core.strategies.ifvg_smc.context_config import ContextFeatureConfig
from strategy_core.strategies.ifvg_smc.context_features import IfvgContextObserver
from strategy_core.strategies.ifvg_smc.reducer import (
    IfvgReducer,
    IfvgReducerConfig,
    IfvgStepInput,
)
from strategy_core.strategies.ifvg_smc.replay import (
    ContextPerformanceTrace,
    _runtime_scheme,
)
from strategy_core.strategies.ifvg_smc.section import (
    IFVG_STRATEGY_VERSION,
    default_ifvg_smc_section,
)
from strategy_core.structures.fvg import Fvg, FvgState, GapDirection
from strategy_core.types import Bar, BarKind, CloseReason

from alpha_lab.agents.data_infra.ifvg.capture_driver import (
    ContextCaptureDayResult,
    flatten_emissions,
    normalize_context_days,
)
from alpha_lab.agents.data_infra.ifvg.context_contracts import ContextRecordTable
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable

DAY = date(2026, 1, 13)
T0 = datetime(2026, 1, 12, 15, tzinfo=UTC)


def _bar(index: int, o: int, h: int, low: int, c: int) -> Bar:
    logical_open = T0 + timedelta(minutes=index)
    logical_close = logical_open + timedelta(minutes=1)
    return Bar(
        timeframe_ticks=60,
        trading_day=DAY,
        bar_index=index,
        bar_id=make_bar_id(60, DAY, index, BarKind.TIME),
        open_ts_utc=logical_open + timedelta(seconds=1),
        close_ts_utc=logical_close - timedelta(seconds=1),
        open_ticks=o,
        high_ticks=h,
        low_ticks=low,
        close_ticks=c,
        volume=1,
        trade_count=1,
        is_complete=True,
        is_partial=False,
        close_reason=CloseReason.COMPLETE,
        kind=BarKind.TIME,
        logical_open_ts_utc=logical_open,
        logical_close_ts_utc=logical_close,
    )


def _fvg(
    seconds: int,
    direction: GapDirection,
    low: int,
    high: int,
    ident: str,
    confirmed: datetime,
) -> Fvg:
    return Fvg(
        fvg_id=f"{seconds}s:{direction.value}:{ident}",
        timeframe_seconds=seconds,
        direction=direction,
        gap_low_ticks=low,
        gap_high_ticks=high,
        size_ticks=high - low,
        a_bar_id=f"{ident}:a",
        c_bar_id=f"{ident}:c",
        a_open_ts_utc=confirmed - timedelta(seconds=seconds * 3),
        confirmed_ts_utc=confirmed,
        trading_day=DAY,
    )


def _step(
    bar: Bar,
    *,
    new_fvgs: dict[int, tuple[Fvg, ...]] | None = None,
    htf_live: tuple[FvgState, ...] = (),
) -> IfvgStepInput:
    return IfvgStepInput(
        bar_1m=bar,
        tf_bars_closed={},
        new_fvgs=new_fvgs or {},
        fill_events=(),
        htf_live=htf_live,
        levels=(),
        recent_swing_highs=(),
        recent_swing_lows=(),
        session_engine="ny",
        session_doc="ny",
    )


def context_fixture():
    section = default_ifvg_smc_section()
    reducer = IfvgReducer(
        IfvgReducerConfig.from_section(
            section,
            tick_size=0.25,
            strategy_id="ifvg_smc",
            strategy_version=IFVG_STRATEGY_VERSION,
        )
    )
    observer = IfvgContextObserver(
        config=ContextFeatureConfig(),
        scheme=_runtime_scheme(section.session_scheme),
        symbol="NQ",
        strategy_core_commit="1" * 40,
        strategy_core_source_tree_hash="2" * 64,
    )
    htf = _fvg(
        3600,
        GapDirection.BULLISH,
        10000,
        10020,
        "htf",
        T0 - timedelta(hours=2),
    )
    parent_bar = _bar(1, 10028, 10031, 10022, 10029)
    opposing_bar = _bar(3, 10020, 10021, 9992, 9995)
    entry_bar = _bar(5, 10015, 10018, 10010, 10016)
    parent = _fvg(
        300,
        GapDirection.BULLISH,
        10010,
        10018,
        "parent",
        parent_bar.availability_ts_utc,
    )
    opposing = _fvg(
        60,
        GapDirection.BEARISH,
        10008,
        10012,
        "opposing",
        opposing_bar.availability_ts_utc,
    )
    entry = _fvg(
        60,
        GapDirection.BULLISH,
        10014,
        10015,
        "entry",
        entry_bar.availability_ts_utc,
    )
    steps = (
        _step(
            _bar(0, 10030, 10032, 10015, 10028),
            htf_live=(FvgState(fvg=htf),),
        ),
        _step(parent_bar, new_fvgs={300: (parent,)}),
        _step(_bar(2, 10024, 10026, 10016, 10022)),
        _step(opposing_bar, new_fvgs={60: (opposing,)}),
        _step(_bar(4, 10000, 10016, 9998, 10014)),
        _step(entry_bar, new_fvgs={60: (entry,)}),
    )
    emissions = []
    events = []
    for step in steps:
        observer.advance_step((), step.bar_1m, step.new_fvgs)
        emitted = reducer.step(step)
        emissions.extend(emitted)
        events.extend(observer.capture(emitted))

    core_rows = flatten_emissions(emissions, entering_seed_hash=None)
    day = ContextCaptureDayResult(
        date_str=DAY.isoformat(),
        core_rows=core_rows,
        funnel={},
        entering_seed_hash=None,
        entering_context_seed_hash=None,
        end_seed=None,
        end_context_seed=observer.snapshot(),
        context_events=tuple(events),
        confirmed_swings=observer.drain_confirmed_swings(),
        pool_lifecycle_events=observer.drain_lifecycle_records(),
        sweep_link_events=observer.drain_sweep_link_records(),
        performance_trace=ContextPerformanceTrace((), (), ()),
    )
    tables = normalize_context_days((day,), warmup_days=0)
    core = {table: pd.DataFrame() for table in RecordTable}
    core[RecordTable.ENTRY_CANDIDATE] = core_rows.loc[
        core_rows["kind"] == RecordTable.ENTRY_CANDIDATE.value
    ].drop(columns="kind").reset_index(drop=True)
    core[RecordTable.ELIGIBLE_DECISION] = core_rows.loc[
        core_rows["kind"] == RecordTable.ELIGIBLE_DECISION.value
    ].drop(columns="kind").reset_index(drop=True)
    core[RecordTable.EXECUTED_TRADE] = pd.DataFrame(
        {"trade_id": tables[ContextRecordTable.TRADE_CONTEXT_LINK]["trade_id"]}
    )
    return day, tables, core, tuple(emissions)
