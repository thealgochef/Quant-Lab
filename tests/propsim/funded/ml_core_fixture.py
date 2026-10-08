"""Small synthetic Core fixture derived from its v2 characterization contract."""
from datetime import UTC, date, datetime, timedelta

from strategy_core.candles._ids import make_bar_id
from strategy_core.strategies.ifvg_smc.reducer import IfvgReducer, IfvgStepInput
from strategy_core.structures.fvg import Fvg, FvgState, GapDirection
from strategy_core.types import Bar, BarKind, CloseReason

_DAY = date(2026, 1, 13)
_T0 = datetime(2026, 1, 13, 16, tzinfo=UTC)

def _bar(index: int, o: int, h: int, low: int, c: int) -> Bar:
    logical_open = _T0 + timedelta(minutes=index)
    logical_close = logical_open + timedelta(minutes=1)
    return Bar(
        timeframe_ticks=60,
        trading_day=_DAY,
        bar_index=index,
        bar_id=make_bar_id(60, _DAY, index, BarKind.TIME),
        open_ts_utc=logical_open + timedelta(seconds=3),
        close_ts_utc=logical_close - timedelta(seconds=2),
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
    tf: int,
    direction: GapDirection,
    lo: int,
    hi: int,
    *,
    confirmed: datetime,
    ident: str,
) -> Fvg:
    return Fvg(
        fvg_id=f"{tf}s:{direction}:{ident}",
        timeframe_seconds=tf,
        direction=direction,
        gap_low_ticks=lo,
        gap_high_ticks=hi,
        size_ticks=hi - lo,
        a_bar_id=f"{ident}:a",
        c_bar_id=f"{ident}:c",
        a_open_ts_utc=confirmed - timedelta(seconds=tf * 3),
        confirmed_ts_utc=confirmed,
        trading_day=_DAY,
    )

def _step(bar: Bar, **kwargs) -> IfvgStepInput:
    return IfvgStepInput(
        bar_1m=bar,
        tf_bars_closed=kwargs.get("tf_bars_closed", {}),
        new_fvgs=kwargs.get("new_fvgs", {}),
        fill_events=kwargs.get("fill_events", ()),
        htf_live=kwargs.get("htf_live", ()),
        levels=(),
        recent_swing_highs=(),
        recent_swing_lows=(),
        session_engine=kwargs.get("session_engine", "ny"),
        session_doc=kwargs.get("session_doc", "ny"),
    )

def _drive_to_inversion(reducer: IfvgReducer) -> tuple[Fvg, Fvg, Fvg]:
    htf = _fvg(
        3600,
        GapDirection.BULLISH,
        10000,
        10020,
        confirmed=_T0 - timedelta(hours=2),
        ident="htf",
    )
    reducer.step(
        _step(_bar(0, 10030, 10032, 10015, 10028), htf_live=(FvgState(fvg=htf),))
    )
    parent_bar = _bar(1, 10028, 10031, 10022, 10029)
    parent = _fvg(
        300,
        GapDirection.BULLISH,
        10010,
        10018,
        confirmed=parent_bar.logical_close_ts_utc,
        ident="parent",
    )
    reducer.step(
        _step(
            parent_bar,
            new_fvgs={300: (parent,)},
            tf_bars_closed={300: parent_bar},
        )
    )
    reducer.step(_step(_bar(2, 10024, 10026, 10016, 10022)))
    opposing_bar = _bar(3, 10020, 10021, 9992, 9995)
    opposing = _fvg(
        60,
        GapDirection.BEARISH,
        10008,
        10012,
        confirmed=opposing_bar.logical_close_ts_utc,
        ident="opposing",
    )
    reducer.step(_step(opposing_bar, new_fvgs={60: (opposing,)}))
    reducer.step(_step(_bar(4, 10000, 10016, 9998, 10014)))
    assert reducer.phase == "S4"
    return htf, parent, opposing
