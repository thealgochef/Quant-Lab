"""AUDIT enrichment (READ-ONLY): per-touch full population for model NQ_20260602_232808.

Reuses PRODUCTION functions verbatim (engine_decision / dashboard_utility_builder /
strategy_core) but does NOT drop touches: every detected touch is recorded with its
drop_reason, the 6 features (where computable), and an honest decision-time TP15/SL15
trade outcome (entered touches only; no_resolution marked-to-market at the 16:15 cutoff
close). This is the shared artifact for B1 (drop accounting) and B2/F (honest edge).

Mirrors strategy_core.resolve_honest_outcome's drop ORDER exactly so the kept set
reconciles to the production 384.  Writes one parquet per date (resumable).

CQL requires PYTHONPATH=src.  Run in date-sharded chunks for parallelism; each shard
warms 3 prior trading days so prev-day NY H/L (PDH/PDL) is correct.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import strategy_core as sc
from strategy_core import Direction, HonestEntryDrop, build_zones, detect_touches, resolve_honest_outcome
from strategy_core.constants import DECISION_OFFSET_MINUTES, FLATTEN_TIME, RTH_END, NO_RESOLUTION

from alpha_lab.agents.data_infra.ml.config import (
    DashboardUtilityConfig, MLPipelineConfig, ModelConfig, WalkForwardConfig,
)
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _build_bars_for_date, _compute_levels_for_date, _ensure_et_index, _get_session_hl_for_date,
)
from alpha_lab.agents.data_infra.ml.engine_decision import (
    TRADE_TICK, bars_et_to_engine, levels_to_engine,
    compute_interaction_features_engine, compute_approach_features_engine, _trade_price_at,
)
from alpha_lab.agents.data_infra.tick_store import TickStore

ET = "US/Eastern"
DATA_DIR = Path("C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento")
SYMBOL = "NQ"
OUTDIR = Path(__file__).parent / "enriched"
SLIP_PTS = 0.25  # 1 tick adverse per side

# THIS model's config (hash 3d2f8466, verified).
THIS_CONFIG = MLPipelineConfig(
    training_mode="dashboard_utility",
    walk_forward=WalkForwardConfig(train_days=30, test_days=7, gap_days=1, expanding=False),
    model=ModelConfig(iterations=500, depth=4, learning_rate=0.03, loss_function="MultiClass",
                      auto_class_weights="Balanced", rfecv_enabled=False, rfecv_min_features=5),
    dashboard_utility=DashboardUtilityConfig(
        tp_points=15.0, sl_points=15.0, trap_mfe_min=5.0, interaction_window_minutes=5,
        level_proximity_pts=0.5, bar_type="147t", include_approach_features=True,
        approach_window_minutes=15),
    tick_size=0.25, instrument="NQ")


def _in_ny_rth(ts_et: pd.Timestamp) -> bool:
    t = ts_et.time()
    return RTH_END.replace(hour=9, minute=30) <= t < RTH_END  # 09:30 <= t < 16:15


def _honest_scan(direction, entry_price, forward_bars_et, tp, sl, tick):
    """MAE-first TP/SL from REAL entry on engine forward bars; mark at last close if neither.
    forward_bars_et: list of engine Bar (already (decision, cutoff)). Returns (exit_reason, gross_pts)
    gross includes exit slippage; entry slippage already baked into entry_price by caller."""
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


def enrich_one_date(date_str, config, prev_state):
    util = config.dashboard_utility
    bars = _build_bars_for_date(DATA_DIR, SYMBOL, date_str, util)
    if bars.empty:
        return pd.DataFrame(), _get_session_hl_for_date(DATA_DIR, SYMBOL, date_str, util, *prev_state)
    bars_et = _ensure_et_index(bars)
    levels = _compute_levels_for_date(bars_et, date_str, *prev_state)
    new_state = _get_session_hl_for_date(DATA_DIR, SYMBOL, date_str, util, *prev_state)
    if not levels:
        return pd.DataFrame(), new_state

    td = date.fromisoformat(date_str)
    tick = TRADE_TICK
    eng_bars = bars_et_to_engine(bars_et, td, tick)
    zones = build_zones(levels_to_engine(levels))
    touches = detect_touches(eng_bars, zones, tick_size=tick, trading_day=td)
    if not touches:
        return pd.DataFrame(), new_state

    store = TickStore(DATA_DIR)
    store.register_symbol_date(SYMBOL, date_str)

    def _tp_for(ts_utc):
        return _trade_price_at(store, SYMBOL, ts_utc)

    rth_cutoff = datetime.combine(td, RTH_END, tzinfo=ZoneInfo(ET))
    rows = []
    try:
        for touch in touches:
            bar_ts_et = pd.Timestamp(touch.bar_ts_utc).tz_convert(ET)
            decision_ts_utc = touch.bar_ts_utc + timedelta(minutes=DECISION_OFFSET_MINUTES)
            decision_ts_et = pd.Timestamp(decision_ts_utc).tz_convert(ET)

            rec = {
                "date": date_str,
                "event_ts": bar_ts_et,
                "decision_ts": decision_ts_et,
                "direction": touch.direction.value,
                "level_type": touch.level_type,
                "representative_price": float(touch.representative_price),
                "session_rth": bool(_in_ny_rth(bar_ts_et)),
                "decision_rth": bool(_in_ny_rth(decision_ts_et)),
                "drop_reason": None, "label": None, "label_encoded": np.nan,
                "max_mfe": np.nan, "max_mae": np.nan,
                "entry_price": np.nan, "honest_exit_reason": None, "honest_gross_pts": np.nan,
                "int_time_beyond_level": np.nan, "int_time_within_2pts": np.nan,
                "int_absorption_ratio": np.nan, "app_avg_trade_size": np.nan,
                "app_large_trade_vol_pct": np.nan, "app_max_spread": np.nan,
            }

            result = resolve_honest_outcome(
                touch, eng_bars, _tp_for, tick_size=tick,
                tp_points=util.tp_points, sl_points=util.sl_points,
                trap_mfe_min=util.trap_mfe_min, decision_offset_minutes=util.interaction_window_minutes,
            )

            # Features (computed regardless, needed to score no_resolution touches too).
            feats = compute_interaction_features_engine(
                touch, DATA_DIR, SYMBOL, util, price_source="trade", tick_size=tick,
                window_minutes=util.interaction_window_minutes)
            appr = compute_approach_features_engine(touch, DATA_DIR, SYMBOL, util)
            if feats:
                rec.update(feats)
            if appr:
                rec.update(appr)

            if isinstance(result, HonestEntryDrop):
                rec["drop_reason"] = result.reason  # flatten / cutoff / no_fill / no_forward
                rows.append(rec)
                continue

            # OutcomeResult (entered touch).
            rec["label"] = result.label
            rec["label_encoded"] = result.label_encoded if result.label_encoded is not None else np.nan
            rec["max_mfe"] = result.max_mfe
            rec["max_mae"] = result.max_mae
            entry_price = _tp_for(decision_ts_utc)
            rec["entry_price"] = float(entry_price) if entry_price is not None else np.nan
            # Honest trade scan from slipped entry on (decision, cutoff) bars.
            forward = [b for b in eng_bars if b.close_ts_utc > decision_ts_utc and b.close_ts_utc < rth_cutoff]
            if entry_price is not None and forward:
                is_long = touch.direction == Direction.LONG
                slip_entry = entry_price + SLIP_PTS if is_long else entry_price - SLIP_PTS
                ex_reason, gross = _honest_scan(touch.direction, slip_entry, forward,
                                                util.tp_points, util.sl_points, tick)
                rec["honest_exit_reason"] = ex_reason
                rec["honest_gross_pts"] = gross

            # Drop classification (mirror production order: no_resolution BEFORE feature drop).
            if result.label == NO_RESOLUTION:
                rec["drop_reason"] = "no_resolution"
            elif feats is None:
                rec["drop_reason"] = "feature_drop"
            else:
                rec["drop_reason"] = "kept"
            rows.append(rec)
    finally:
        store.close()

    return pd.DataFrame(rows), new_state


def available_dates(start, end):
    out = []
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    d = d0
    while d <= d1:
        ds = d.isoformat()
        if (DATA_DIR / SYMBOL / ds).is_dir():
            out.append(ds)
        d += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--warm", type=int, default=4, help="prior trading days to warm level state")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Warm-up: walk back ~warm trading days before start to seed prev_ny_hl (PDH/PDL).
    warm_start = (date.fromisoformat(args.start) - timedelta(days=args.warm + 4)).isoformat()
    all_dates = available_dates(warm_start, args.end)
    target = set(available_dates(args.start, args.end))

    prev_state = (None, None, None)
    util = THIS_CONFIG.dashboard_utility
    for ds in all_dates:
        out_path = OUTDIR / f"{ds}.parquet"
        if ds in target and out_path.exists():
            # already done; still advance level state
            prev_state = _get_session_hl_for_date(DATA_DIR, SYMBOL, ds, util, *prev_state)
            continue
        try:
            df, prev_state = enrich_one_date(ds, THIS_CONFIG, prev_state)
        except Exception as e:
            sys.stderr.write(f"FAIL {ds}: {e}\n")
            try:
                prev_state = _get_session_hl_for_date(DATA_DIR, SYMBOL, ds, util, *prev_state)
            except Exception:
                pass
            continue
        if ds in target:
            df.to_parquet(out_path, index=False)
            n_keep = int((df["drop_reason"] == "kept").sum()) if not df.empty else 0
            print(f"{ds}: {len(df)} touches, {n_keep} kept", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
