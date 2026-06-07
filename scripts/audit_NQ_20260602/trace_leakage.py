# ruff: noqa: E501
"""C-section: empirical leakage timestamp traces on real touches (READ-ONLY).

For a few real RTH touches, print the ACTUAL min/max ts_event of the data each feature
window queries, plus decision_ts / entry-print ts / first-forward-bar close, to prove:
 C1 approach trades strictly < touch ; C2 interaction ticks in [touch, decision] (<=decision)
 C3 first forward bar close strictly > decision ; C4 entry print ts <= decision.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from strategy_core import build_zones, detect_touches
from strategy_core.constants import DECISION_OFFSET_MINUTES, RTH_END

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _build_bars_for_date,
    _compute_levels_for_date,
    _ensure_et_index,
    _get_session_hl_for_date,
)
from alpha_lab.agents.data_infra.ml.engine_decision import (
    TRADE_TICK,
    _query_quotes,
    _query_trades,
    _trade_price_at,
    bars_et_to_engine,
    levels_to_engine,
)
from alpha_lab.agents.data_infra.tick_store import TickStore

DATA_DIR = Path("C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento")
SYM = "NQ"
ET = ZoneInfo("US/Eastern")
util = DashboardUtilityConfig(
    tp_points=15.0,
    sl_points=15.0,
    trap_mfe_min=5.0,
    interaction_window_minutes=5,
    level_proximity_pts=0.5,
    bar_type="147t",
    include_approach_features=True,
    approach_window_minutes=15,
)


def trace(date_str, max_touches=3):
    # warm prev day for levels
    prev = (date.fromisoformat(date_str) - timedelta(days=1)).isoformat()
    st = (None, None, None)
    st = _get_session_hl_for_date(DATA_DIR, SYM, prev, util, *st)
    bars = _ensure_et_index(_build_bars_for_date(DATA_DIR, SYM, date_str, util))
    levels = _compute_levels_for_date(bars, date_str, *st)
    td = date.fromisoformat(date_str)
    eng_bars = bars_et_to_engine(bars, td, TRADE_TICK)
    touches = detect_touches(
        eng_bars,
        _zones := build_zones(levels_to_engine(levels)),
        tick_size=TRADE_TICK,
        trading_day=td,
    )
    store = TickStore(DATA_DIR)
    store.register_symbol_date(SYM, date_str)
    store.register_symbol_date(SYM, prev)
    shown = 0
    for tch in touches:
        tt = pd.Timestamp(tch.bar_ts_utc).tz_convert(ET)
        if not (tt.time() >= RTH_END.replace(hour=9, minute=30) and tt.time() < RTH_END):
            continue
        dts = tch.bar_ts_utc + timedelta(minutes=DECISION_OFFSET_MINUTES)
        dts_et = pd.Timestamp(dts).tz_convert(ET)
        appr_start = tch.bar_ts_utc - timedelta(minutes=util.approach_window_minutes)
        atr = _query_trades(store, SYM, appr_start, tch.bar_ts_utc)  # [t-15m, t)
        _query_quotes(store, SYM, appr_start, tch.bar_ts_utc)
        itk = store.query_tick_feature_rows(
            SYM, tch.bar_ts_utc, tch.bar_ts_utc + timedelta(minutes=5), price_source="trade"
        )
        entry = _trade_price_at(store, SYM, dts)
        # entry print ts: most-recent <=dts
        fwd = [
            b
            for b in eng_bars
            if b.close_ts_utc > dts and b.close_ts_utc < pd.Timestamp(f"{date_str} 16:15", tz=ET)
        ]
        print(f"\n=== touch {tch.level_type} {tch.direction.value} @ {tt} (decision {dts_et}) ===")
        print(
            f"  C1 approach trades: n={len(atr)} ts[min..max]={pd.to_datetime(atr.ts_event.min())}..{pd.to_datetime(atr.ts_event.max())}  (must be < touch {pd.Timestamp(tch.bar_ts_utc)})  OK={pd.to_datetime(atr.ts_event.max()) < pd.Timestamp(tch.bar_ts_utc).tz_localize('UTC') if atr.ts_event.max().tzinfo is None else pd.to_datetime(atr.ts_event.max()) < pd.Timestamp(tch.bar_ts_utc)}"
        )
        print(
            f"  C2 interaction ticks: n={len(itk)} ts[min..max]={pd.to_datetime(itk.ts_event.min())}..{pd.to_datetime(itk.ts_event.max())}  (must be <= decision {pd.Timestamp(dts)})"
        )
        print(
            f"  C4 entry print: price={entry} (most-recent trade <= decision {pd.Timestamp(dts)})"
        )
        print(
            f"  C3 first forward bar close: {pd.Timestamp(fwd[0].close_ts_utc) if fwd else None}  (must be > decision {pd.Timestamp(dts)})"
        )
        shown += 1
        if shown >= max_touches:
            break
    store.close()


if __name__ == "__main__":
    import sys

    for d in sys.argv[1:] or ["2025-06-05", "2025-06-13"]:
        print(f"\n################ {d} ################")
        trace(d)
