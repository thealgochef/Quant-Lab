# ruff: noqa: N806,SIM115
"""DIAGNOSTIC (read-only): enrich the HONEST v3 NY touch population (no model, no gate).

Reproduces the v3 dataset's touch generation (availability-ENFORCED, full-prior-day
PDH/PDL, re-clocked ny 09:00-17:00) and, for every NY touch, records the honest
decision-time outcome — REUSING the audit harness machinery verbatim:
  * realistic entry = front-month TRADE price at touch + DECISION_OFFSET (5m)
    (engine_decision._trade_price_at), 1-tick adverse slippage per side,
  * tp=15 / sl=15 MAE-first resolve (strategy_core.resolve_honest_outcome for the
    label; _honest_scan — copied verbatim from audit_NQ_20260602/enrich_dates.py — for
    the realistic PnL with exit slippage and a cutoff mark-to-market),
  * v3 flatten 16:40 ET / forward cutoff 17:00 ET (the engine constants).
The ONLY change vs the gated audit is downstream (the report): no model, no 0.70 gate.

Writes one parquet per date to ny_enriched/ (resumable). NO features (not needed for
the baseline) -> fast. NO model, NO retrain, NO writes outside scripts/v3_verify.
PYTHONPATH=src.  Usage: python ny_baseline_enrich.py --start-idx I --end-idx J
(processes sorted dates_used[I:J]; warms prev_full_hl from dates_used[I-1]).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from strategy_core import (
    Direction,
    HonestEntryDrop,
    build_zones,
    classify_session,
    detect_touches,
    resolve_honest_outcome,
)
from strategy_core.constants import DECISION_OFFSET_MINUTES, NO_RESOLUTION, RTH_END

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _build_bars_for_date,
    _compute_levels_for_date,
    _ensure_et_index,
)
from alpha_lab.agents.data_infra.ml.engine_decision import (
    TRADE_TICK,
    _trade_price_at,
    bars_et_to_engine,
    levels_to_engine,
)
from alpha_lab.agents.data_infra.tick_store import TickStore

ET = "US/Eastern"
DATA_DIR = Path("C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento")
SYMBOL = "NQ"
OUTDIR = Path(__file__).parent / "ny_enriched"
SLIP_PTS = 0.25  # 1 tick adverse per side (verbatim from enrich_dates.py)
# Model's exact label policy (tp15/sl15/int5/147t) — UNCHANGED, no config edit.
UTIL = DashboardUtilityConfig(
    tp_points=15.0,
    sl_points=15.0,
    trap_mfe_min=5.0,
    interaction_window_minutes=5,
    level_proximity_pts=0.5,
    bar_type="147t",
    include_approach_features=False,
    approach_window_minutes=15,
)


def _honest_scan(direction, entry_price, forward_bars_et, tp, sl, tick):
    """VERBATIM from audit_NQ_20260602/enrich_dates.py: MAE-first TP/SL from REAL entry
    on engine forward bars; mark at last close if neither. gross includes exit slippage;
    entry slippage already baked into entry_price by caller. Returns (exit_reason, gross_pts)."""
    is_long = direction == Direction.LONG
    tp_lvl = entry_price + tp if is_long else entry_price - tp
    sl_lvl = entry_price - sl if is_long else entry_price + sl
    for bar in forward_bars_et:
        hi = bar.high_ticks * tick
        lo = bar.low_ticks * tick
        if is_long:
            if lo <= sl_lvl:
                ex = sl_lvl - SLIP_PTS
                return "sl", ex - entry_price
            if hi >= tp_lvl:
                ex = tp_lvl - SLIP_PTS
                return "tp", ex - entry_price
        else:
            if hi >= sl_lvl:
                ex = sl_lvl + SLIP_PTS
                return "sl", entry_price - ex
            if lo <= tp_lvl:
                ex = tp_lvl + SLIP_PTS
                return "tp", entry_price - ex
    last_close = forward_bars_et[-1].close_ticks * tick
    if is_long:
        return "cutoff", (last_close - SLIP_PTS) - entry_price
    return "cutoff", entry_price - (last_close + SLIP_PTS)


def enrich_one(date_str: str, prev_full_hl):
    """Return (DataFrame of NY-touch records, new_full_hl)."""
    bars = _build_bars_for_date(DATA_DIR, SYMBOL, date_str, UTIL)
    if bars.empty:
        return pd.DataFrame(), prev_full_hl
    be = _ensure_et_index(bars)
    new_full = (float(be["high"].max()), float(be["low"].min()))
    levels = _compute_levels_for_date(be, date_str, prev_full_hl, None, None)
    if not levels:
        return pd.DataFrame(), new_full

    td = date.fromisoformat(date_str)
    tick = TRADE_TICK
    eng_bars = bars_et_to_engine(be, td, tick)
    # GATED v3 zones (availability ENFORCED).
    zones_av = build_zones(levels_to_engine(levels, with_availability=True))
    avail_by_rep = {round(z.representative_price, 6): z.available_from for z in zones_av}
    zones = build_zones(levels_to_engine(levels, with_availability=True))
    touches = detect_touches(eng_bars, zones, tick_size=tick, trading_day=td)
    if not touches:
        return pd.DataFrame(), new_full

    store = TickStore(DATA_DIR)
    store.register_symbol_date(SYMBOL, date_str)

    def _tp(ts_utc):
        return _trade_price_at(store, SYMBOL, ts_utc)

    rth_cutoff = datetime.combine(td, RTH_END, tzinfo=ZoneInfo(ET))
    rows = []
    try:
        for t in touches:
            if classify_session(t.bar_ts_utc).session != "ny":
                continue  # NY-only universe
            bar_ts_et = pd.Timestamp(t.bar_ts_utc).tz_convert(ET)
            decision_ts_utc = t.bar_ts_utc + timedelta(minutes=DECISION_OFFSET_MINUTES)
            decision_ts_et = pd.Timestamp(decision_ts_utc).tz_convert(ET)
            A = avail_by_rep.get(round(t.representative_price, 6))
            rec = {
                "date": date_str,
                "event_ts": bar_ts_et,
                "decision_ts": decision_ts_et,
                "direction": t.direction.value,
                "level_type": t.level_type,
                "representative_price": float(t.representative_price),
                # look-ahead self-check within NY (must be False — gate guarantees it).
                "lookahead": bool(A is not None and t.bar_ts_utc < A),
                "eligible": None,
                "drop_reason": None,
                "label": None,
                "label_encoded": np.nan,
                "max_mfe": np.nan,
                "max_mae": np.nan,
                "entry_price": np.nan,
                "honest_exit_reason": None,
                "honest_gross_pts": np.nan,
            }

            result = resolve_honest_outcome(
                t,
                eng_bars,
                _tp,
                tick_size=tick,
                tp_points=UTIL.tp_points,
                sl_points=UTIL.sl_points,
                trap_mfe_min=UTIL.trap_mfe_min,
                decision_offset_minutes=UTIL.interaction_window_minutes,
            )

            if isinstance(result, HonestEntryDrop):
                rec["drop_reason"] = result.reason  # flatten / cutoff / no_fill / no_forward
                # eligible = decision before flatten AND before cutoff (the n ceiling).
                rec["eligible"] = result.reason not in ("flatten", "cutoff")
                rows.append(rec)
                continue

            # decision was in-window -> eligible.
            rec["eligible"] = True
            rec["label"] = result.label
            rec["label_encoded"] = (
                result.label_encoded if result.label_encoded is not None else np.nan
            )
            rec["max_mfe"] = result.max_mfe
            rec["max_mae"] = result.max_mae
            rec["drop_reason"] = "kept" if result.label != NO_RESOLUTION else "no_resolution"

            entry_price = _tp(decision_ts_utc)
            rec["entry_price"] = float(entry_price) if entry_price is not None else np.nan
            forward = [
                b
                for b in eng_bars
                if b.close_ts_utc > decision_ts_utc and b.close_ts_utc < rth_cutoff
            ]
            if entry_price is not None and forward:
                is_long = t.direction == Direction.LONG
                slip_entry = entry_price + SLIP_PTS if is_long else entry_price - SLIP_PTS
                ex_reason, gross = _honest_scan(
                    t.direction, slip_entry, forward, UTIL.tp_points, UTIL.sl_points, tick
                )
                rec["honest_exit_reason"] = ex_reason
                rec["honest_gross_pts"] = gross
            rows.append(rec)
    finally:
        store.close()
    return pd.DataFrame(rows), new_full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-idx", type=int, required=True)
    ap.add_argument("--end-idx", type=int, required=True)  # exclusive
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    ev = json.load(open("models/NQ_20260602_232808/evaluation.json"))
    dates = sorted(ev["dates_used"])
    lo, hi = args.start_idx, min(args.end_idx, len(dates))
    shard = dates[lo:hi]

    # Seed prev_full_hl from the IMMEDIATELY-prior dates_used date (PDH/PDL is a
    # 1-step carry: PDH/PDL[i] == full_hl[i-1], computed from date i-1's bars alone),
    # so this exactly reproduces the rebuild's per-date PDH/PDL. idx 0 -> None.
    prev_full_hl = None
    if lo > 0:
        warm = dates[lo - 1]
        wb = _build_bars_for_date(DATA_DIR, SYMBOL, warm, UTIL)
        if not wb.empty:
            wbe = _ensure_et_index(wb)
            prev_full_hl = (float(wbe["high"].max()), float(wbe["low"].min()))

    for ds in shard:
        out_path = OUTDIR / f"{ds}.parquet"
        if out_path.exists():
            # already done; still advance carry from this date's bars
            b = _build_bars_for_date(DATA_DIR, SYMBOL, ds, UTIL)
            if not b.empty:
                be = _ensure_et_index(b)
                prev_full_hl = (float(be["high"].max()), float(be["low"].min()))
            continue
        try:
            df, prev_full_hl = enrich_one(ds, prev_full_hl)
        except Exception as e:
            sys.stderr.write(f"FAIL {ds}: {e}\n")
            continue
        df.to_parquet(out_path, index=False)  # may be empty (0 NY touches that day)
        print(f"{ds}: {len(df)} NY touches", flush=True)
    print(f"SHARD DONE [{lo}:{hi}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
