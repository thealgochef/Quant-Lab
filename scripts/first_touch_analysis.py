"""
First-touch-only rule diagnostic.

Analyses how London/Asia/pre-market touches "spend" level zones before
NY RTH opens, reducing the number of tradeable signals during RTH.

Runs the 30-day backtest with instrumented touch tracking to produce:
1. Per-day table: levels available vs touched per session
2. Counterfactual: per-session first-touch (each session resets zone state)
3. Alternative: only RTH touches count as "spent"
4. CSV output: data/experiment/first_touch_analysis.csv

Usage:
    python scripts/first_touch_analysis.py
    python scripts/first_touch_analysis.py --start 2025-06-02 --end 2025-07-08
"""

from __future__ import annotations

import argparse
import copy
import sys
import time as time_mod
import uuid
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from alpha_lab.dashboard.engine.level_engine import LevelEngine
from alpha_lab.dashboard.engine.models import (
    LevelSide,
    LevelZone,
    ObservationStatus,
    TouchEvent,
    TradeDirection,
)
from alpha_lab.dashboard.engine.observation_manager import ObservationManager
from alpha_lab.dashboard.engine.feature_computer import FeatureComputer
from alpha_lab.dashboard.model.model_manager import ModelManager
from alpha_lab.dashboard.model.prediction_engine import PredictionEngine
from alpha_lab.dashboard.pipeline.price_buffer import PriceBuffer
from alpha_lab.dashboard.pipeline.rithmic_client import BBOUpdate, TradeUpdate

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")

DATA_DIR = Path("data/databento/NQ")
MODEL_PATH = Path("data/models/dashboard_3feature_v1.cbm")
OUTPUT_CSV = Path("data/experiment/first_touch_analysis.csv")


# ---- Session classifier (same as touch_detector.py) ----

def classify_session(ts_utc: datetime) -> str:
    ts_et = ts_utc.astimezone(ET)
    t = ts_et.time()
    if t >= time(18, 0) or t < time(1, 0):
        return "asia"
    if time(1, 0) <= t < time(8, 0):
        return "london"
    if time(8, 0) <= t < time(9, 30):
        return "pre_market"
    if time(9, 30) <= t < time(16, 15):
        return "ny_rth"
    return "post_market"


# ---- Data loading (from run_backtest.py) ----

def get_available_dates() -> list[str]:
    dates = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and (d / "mbp10.parquet").exists():
            dates.append(d.name)
    return dates


def detect_front_month(conn: duckdb.DuckDBPyConnection, mbp_path: str) -> str:
    rows = conn.execute(f"""
        SELECT symbol, count(*) AS n
        FROM read_parquet('{mbp_path}')
        WHERE action = 'T' AND symbol NOT LIKE '%-%'
        GROUP BY symbol ORDER BY n DESC LIMIT 1
    """).fetchall()
    if not rows:
        raise ValueError(f"No trades in {mbp_path}")
    return rows[0][0]


def load_trades_for_date(
    conn: duckdb.DuckDBPyConnection,
    date_str: str,
    front_month: str,
) -> pd.DataFrame:
    mbp_path = str(DATA_DIR / date_str / "mbp10.parquet")
    df = conn.execute(f"""
        SELECT
            ts_event,
            CAST(price AS DOUBLE) AS price,
            CAST(size AS INTEGER) AS size,
            side,
            CAST(bid_px_00 AS DOUBLE) AS bid_price,
            CAST(ask_px_00 AS DOUBLE) AS ask_price,
            CAST(bid_sz_00 AS INTEGER) AS bid_size,
            CAST(ask_sz_00 AS INTEGER) AS ask_size
        FROM read_parquet('{mbp_path}')
        WHERE action = 'T'
          AND symbol = '{front_month}'
          AND bid_px_00 > 0 AND ask_px_00 > 0
        ORDER BY ts_event
    """).fetchdf()
    return df


# ---- Touch tracking state ----

class TouchTracker:
    """Tracks all touch events per zone per session for diagnostic analysis."""

    def __init__(self) -> None:
        self.touches: list[dict] = []  # All touch events with metadata

    def record_touch(
        self,
        zone_id: str,
        zone_price: Decimal,
        zone_side: str,
        session: str,
        timestamp: datetime,
        trade_price: Decimal,
        direction: str,
    ) -> None:
        self.touches.append({
            "zone_id": zone_id,
            "zone_price": float(zone_price),
            "zone_side": zone_side,
            "session": session,
            "timestamp": timestamp,
            "trade_price": float(trade_price),
            "direction": direction,
        })


# ---- Counterfactual touch detector (no spending) ----

def run_counterfactual_touches(
    zones_snapshot: list[dict],
    trades_df: pd.DataFrame,
    front_month: str,
    mode: str,
) -> list[dict]:
    """Replay ticks against zone snapshots with alternative first-touch rules.

    modes:
      "per_session" - reset touch state at each session boundary
      "rth_only"    - only RTH touches count as "spent"
    """
    # Build working zone list: {zone_id, price, side, is_touched, touched_session}
    zones = []
    for z in zones_snapshot:
        zones.append({
            "zone_id": z["zone_id"],
            "price": z["price"],
            "side": z["side"],
            "is_touched": False,
            "touched_session": None,
            "touch_events": [],
        })

    current_session = None

    for _, row in trades_df.iterrows():
        ts = pd.Timestamp(row["ts_event"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        ts_utc = ts.tz_convert(UTC).to_pydatetime()
        trade_price = Decimal(str(round(float(row["price"]), 2)))

        session = classify_session(ts_utc)

        # Per-session mode: reset touches at session boundary
        if mode == "per_session" and session != current_session:
            for z in zones:
                z["is_touched"] = False
                z["touched_session"] = None
            current_session = session

        # RTH-only mode: pre-RTH touches don't count
        if mode == "rth_only" and session != current_session:
            current_session = session

        # Time cutoff: 3:49 PM CT
        ts_ct = ts_utc.astimezone(CT)
        if ts_ct.hour > 15 or (ts_ct.hour == 15 and ts_ct.minute > 49):
            continue

        for z in zones:
            if z["is_touched"]:
                continue

            touched = False
            if z["side"] == "high" and trade_price >= z["price"]:
                touched = True
                direction = "short"
            elif z["side"] == "low" and trade_price <= z["price"]:
                touched = True
                direction = "long"

            if touched:
                if mode == "rth_only" and session != "ny_rth":
                    # Record the touch but don't mark as spent
                    z["touch_events"].append({
                        "session": session,
                        "timestamp": ts_utc,
                        "price": float(trade_price),
                        "direction": direction,
                        "spent": False,
                    })
                    continue

                z["is_touched"] = True
                z["touched_session"] = session
                z["touch_events"].append({
                    "session": session,
                    "timestamp": ts_utc,
                    "price": float(trade_price),
                    "direction": direction,
                    "spent": True,
                })

    return zones


# ---- Main analysis ----

def analyze_day(
    price_buffer: PriceBuffer,
    date_str: str,
    conn: duckdb.DuckDBPyConnection,
    feature_computer: FeatureComputer,
    model_manager: ModelManager,
) -> dict:
    """Run first-touch analysis for one trading day."""
    trading_date = date.fromisoformat(date_str)

    # Compute levels at RTH open (same as run_backtest.py)
    level_engine = LevelEngine(price_buffer)
    rth_open = datetime.combine(
        trading_date, time(9, 30), tzinfo=ET,
    ).astimezone(UTC)
    levels = level_engine.compute_levels(trading_date, current_time=rth_open)
    zones = level_engine.get_active_zones()

    total_levels = len(levels)
    total_zones = len(zones)

    # Snapshot zones for counterfactual analysis
    zones_snapshot = []
    for z in zones:
        zones_snapshot.append({
            "zone_id": z.zone_id,
            "price": z.representative_price,
            "side": z.side.value,
        })

    # Load ticks for the day
    mbp_path = str(DATA_DIR / date_str / "mbp10.parquet")
    front_month = detect_front_month(conn, mbp_path)
    trades_df = load_trades_for_date(conn, date_str, front_month)

    if len(trades_df) == 0:
        return _empty_result(date_str)

    # ---- ACTUAL behavior: replay with per-day first-touch ----
    # Set up fresh level engine + touch detector for this day
    level_engine_actual = LevelEngine(price_buffer)
    level_engine_actual.compute_levels(trading_date, current_time=rth_open)

    touch_tracker = TouchTracker()
    obs_mgr = ObservationManager(feature_computer)
    pred_engine = PredictionEngine(model_manager)

    # Track touches per session
    touches_by_session: dict[str, int] = {
        "asia": 0, "london": 0, "pre_market": 0, "ny_rth": 0, "post_market": 0,
    }
    rth_signals = 0
    rth_executable = 0
    all_signals = 0
    predictions_by_session: dict[str, list] = {
        "asia": [], "london": [], "pre_market": [], "ny_rth": [], "post_market": [],
    }

    # Wire observation -> prediction
    def _on_obs_complete(window) -> None:
        nonlocal all_signals
        if window.status != ObservationStatus.COMPLETED:
            return
        pred_engine.predict(window)

    obs_mgr.on_observation_complete(_on_obs_complete)

    def _on_prediction(prediction) -> None:
        nonlocal rth_signals, rth_executable, all_signals
        session = prediction.observation.event.session
        all_signals += 1
        predictions_by_session[session].append({
            "predicted_class": prediction.predicted_class,
            "is_executable": prediction.is_executable,
            "direction": prediction.trade_direction.value,
        })
        if session == "ny_rth":
            rth_signals += 1
            if prediction.is_executable:
                rth_executable += 1

    pred_engine.on_prediction(_on_prediction)

    latest_price = None

    for _, row in trades_df.iterrows():
        ts = pd.Timestamp(row["ts_event"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(UTC)
        ts_utc = ts.tz_convert(UTC).to_pydatetime()
        trade_price = Decimal(str(round(float(row["price"]), 2)))
        bid_price = Decimal(str(round(float(row["bid_price"]), 2)))
        ask_price = Decimal(str(round(float(row["ask_price"]), 2)))
        latest_price = trade_price

        # Add to price buffer for future days
        trade_update = TradeUpdate(
            timestamp=ts_utc,
            price=trade_price,
            size=int(row["size"]),
            aggressor_side="BUY" if row["side"] == "A" else "SELL",
            symbol=front_month,
        )
        price_buffer.add_trade(trade_update)

        bbo = BBOUpdate(
            timestamp=ts_utc,
            bid_price=bid_price,
            bid_size=int(row["bid_size"]),
            ask_price=ask_price,
            ask_size=int(row["ask_size"]),
            symbol=front_month,
        )
        obs_mgr.on_bbo(bbo)

        # Time cutoff: 3:49 PM CT
        ts_ct = ts_utc.astimezone(CT)
        if ts_ct.hour > 15 or (ts_ct.hour == 15 and ts_ct.minute > 49):
            obs_mgr.on_trade(trade_update)
            continue

        # Check for touches
        session = classify_session(ts_utc)
        for zone in level_engine_actual.get_active_zones():
            touched = False
            if zone.side == LevelSide.HIGH and trade_price >= zone.representative_price:
                touched = True
                direction = TradeDirection.SHORT
            elif zone.side == LevelSide.LOW and trade_price <= zone.representative_price:
                touched = True
                direction = TradeDirection.LONG
            else:
                continue

            if touched:
                level_engine_actual.mark_zone_touched(zone.zone_id, ts_utc)
                touches_by_session[session] = touches_by_session.get(session, 0) + 1
                touch_tracker.record_touch(
                    zone.zone_id, zone.representative_price, zone.side.value,
                    session, ts_utc, trade_price, direction.value,
                )

                # Fire observation window
                event = TouchEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=ts_utc,
                    level_zone=zone,
                    trade_direction=direction,
                    price_at_touch=trade_price,
                    session=session,
                )
                obs_mgr.start_observation(event)
                break  # Only one touch per tick

        obs_mgr.on_trade(trade_update)

    # Compute touched-before-RTH
    pre_rth_touches = (
        touches_by_session.get("asia", 0)
        + touches_by_session.get("london", 0)
        + touches_by_session.get("pre_market", 0)
    )

    available_at_rth = total_zones - pre_rth_touches

    # ---- Counterfactual: per-session first-touch ----
    cf_per_session = run_counterfactual_touches(
        zones_snapshot, trades_df, front_month, "per_session",
    )
    cf_rth_touches_per_session = sum(
        1 for z in cf_per_session
        for e in z["touch_events"]
        if e["session"] == "ny_rth" and e["spent"]
    )
    cf_total_touches_per_session = sum(
        len(z["touch_events"]) for z in cf_per_session
    )

    # ---- Counterfactual: RTH-only spending ----
    cf_rth_only = run_counterfactual_touches(
        zones_snapshot, trades_df, front_month, "rth_only",
    )
    cf_rth_touches_rth_only = sum(
        1 for z in cf_rth_only
        for e in z["touch_events"]
        if e["session"] == "ny_rth" and e["spent"]
    )
    cf_non_rth_touches_rth_only = sum(
        1 for z in cf_rth_only
        for e in z["touch_events"]
        if e["session"] != "ny_rth" and not e["spent"]
    )

    return {
        "date": date_str,
        "front_month": front_month,
        "total_levels": total_levels,
        "total_zones": total_zones,
        "touched_asia": touches_by_session.get("asia", 0),
        "touched_london": touches_by_session.get("london", 0),
        "touched_pre_market": touches_by_session.get("pre_market", 0),
        "touched_before_rth": pre_rth_touches,
        "available_at_rth": available_at_rth,
        "touched_rth": touches_by_session.get("ny_rth", 0),
        "rth_signals": rth_signals,
        "rth_executable": rth_executable,
        "all_signals": all_signals,
        # Counterfactual: per-session
        "cf_per_session_rth_touches": cf_rth_touches_per_session,
        "cf_per_session_total_touches": cf_total_touches_per_session,
        # Counterfactual: RTH-only spending
        "cf_rth_only_rth_touches": cf_rth_touches_rth_only,
        "cf_rth_only_non_rth_touches_ignored": cf_non_rth_touches_rth_only,
        # Session prediction breakdown
        "sigs_asia": len(predictions_by_session.get("asia", [])),
        "sigs_london": len(predictions_by_session.get("london", [])),
        "sigs_pre_market": len(predictions_by_session.get("pre_market", [])),
        "sigs_ny_rth": len(predictions_by_session.get("ny_rth", [])),
        # Touch details for debugging
        "_touch_details": touch_tracker.touches,
        "_predictions_by_session": predictions_by_session,
    }


def _empty_result(date_str: str) -> dict:
    return {
        "date": date_str, "front_month": "", "total_levels": 0,
        "total_zones": 0, "touched_asia": 0, "touched_london": 0,
        "touched_pre_market": 0, "touched_before_rth": 0,
        "available_at_rth": 0, "touched_rth": 0, "rth_signals": 0,
        "rth_executable": 0, "all_signals": 0,
        "cf_per_session_rth_touches": 0, "cf_per_session_total_touches": 0,
        "cf_rth_only_rth_touches": 0, "cf_rth_only_non_rth_touches_ignored": 0,
        "sigs_asia": 0, "sigs_london": 0, "sigs_pre_market": 0,
        "sigs_ny_rth": 0,
        "_touch_details": [], "_predictions_by_session": {},
    }


# ---- Printing ----

def print_results(results: list[dict]) -> None:
    print(f"\n{'=' * 100}")
    print("  FIRST-TOUCH DIAGNOSTIC: Level Consumption Before RTH")
    print(f"{'=' * 100}")

    # ---- Per-day table ----
    print(f"\n  --- Current Behavior: Per-Day First-Touch (levels spent across all sessions) ---\n")
    header = (
        f"  {'Date':10s}  {'Zones':>5s}  "
        f"{'Asia':>5s}  {'Lon':>5s}  {'Pre':>5s}  "
        f"{'PreRTH':>6s}  {'@RTH':>5s}  "
        f"{'RTH Tch':>7s}  {'RTH Sig':>7s}  {'RTH Exec':>8s}  "
        f"{'All Sig':>7s}"
    )
    print(header)
    print(f"  {'-' * 90}")

    totals = {
        "zones": 0, "asia": 0, "london": 0, "pre": 0,
        "pre_rth": 0, "at_rth": 0, "rth_tch": 0,
        "rth_sig": 0, "rth_exec": 0, "all_sig": 0,
    }

    for r in results:
        print(
            f"  {r['date']:10s}  {r['total_zones']:>5d}  "
            f"{r['touched_asia']:>5d}  {r['touched_london']:>5d}  "
            f"{r['touched_pre_market']:>5d}  "
            f"{r['touched_before_rth']:>6d}  {r['available_at_rth']:>5d}  "
            f"{r['touched_rth']:>7d}  {r['rth_signals']:>7d}  "
            f"{r['rth_executable']:>8d}  "
            f"{r['all_signals']:>7d}"
        )
        totals["zones"] += r["total_zones"]
        totals["asia"] += r["touched_asia"]
        totals["london"] += r["touched_london"]
        totals["pre"] += r["touched_pre_market"]
        totals["pre_rth"] += r["touched_before_rth"]
        totals["at_rth"] += r["available_at_rth"]
        totals["rth_tch"] += r["touched_rth"]
        totals["rth_sig"] += r["rth_signals"]
        totals["rth_exec"] += r["rth_executable"]
        totals["all_sig"] += r["all_signals"]

    n = len(results)
    print(f"  {'-' * 90}")
    print(
        f"  {'TOTAL':10s}  {totals['zones']:>5d}  "
        f"{totals['asia']:>5d}  {totals['london']:>5d}  "
        f"{totals['pre']:>5d}  "
        f"{totals['pre_rth']:>6d}  {totals['at_rth']:>5d}  "
        f"{totals['rth_tch']:>7d}  {totals['rth_sig']:>7d}  "
        f"{totals['rth_exec']:>8d}  "
        f"{totals['all_sig']:>7d}"
    )
    if n > 0:
        print(
            f"  {'AVG/day':10s}  {totals['zones']/n:>5.1f}  "
            f"{totals['asia']/n:>5.1f}  {totals['london']/n:>5.1f}  "
            f"{totals['pre']/n:>5.1f}  "
            f"{totals['pre_rth']/n:>6.1f}  {totals['at_rth']/n:>5.1f}  "
            f"{totals['rth_tch']/n:>7.1f}  {totals['rth_sig']/n:>7.1f}  "
            f"{totals['rth_exec']/n:>8.1f}  "
            f"{totals['all_sig']/n:>7.1f}"
        )

    # ---- Key insight: what % of zones are consumed before RTH? ----
    if totals["zones"] > 0:
        pct_consumed = totals["pre_rth"] / totals["zones"] * 100
        print(f"\n  >> {totals['pre_rth']}/{totals['zones']} zones "
              f"({pct_consumed:.1f}%) consumed before RTH opens")
        pct_rth_touch = totals["rth_tch"] / totals["zones"] * 100
        print(f"  >> {totals['rth_tch']}/{totals['zones']} zones "
              f"({pct_rth_touch:.1f}%) touched during RTH")
        pct_untouched = (totals["zones"] - totals["pre_rth"] - totals["rth_tch"]) / totals["zones"] * 100
        untouched = totals["zones"] - totals["pre_rth"] - totals["rth_tch"]
        print(f"  >> {untouched}/{totals['zones']} zones "
              f"({pct_untouched:.1f}%) never touched all day")

    # ---- Counterfactual: per-session ----
    print(f"\n\n  --- Counterfactual A: Per-SESSION First-Touch (reset touch state each session) ---\n")
    print(f"  In this mode, a London touch does NOT spend the zone for RTH.")
    print(f"  Each session gets its own independent first-touch.\n")

    header_cf = (
        f"  {'Date':10s}  {'Zones':>5s}  "
        f"{'Actual RTH':>10s}  {'CF RTH Tch':>10s}  "
        f"{'CF Total':>8s}  {'Delta':>6s}"
    )
    print(header_cf)
    print(f"  {'-' * 60}")

    cf_total_actual_rth = 0
    cf_total_cf_rth = 0

    for r in results:
        actual_rth = r["touched_rth"]
        cf_rth = r["cf_per_session_rth_touches"]
        delta = cf_rth - actual_rth
        cf_total_actual_rth += actual_rth
        cf_total_cf_rth += cf_rth
        print(
            f"  {r['date']:10s}  {r['total_zones']:>5d}  "
            f"{actual_rth:>10d}  {cf_rth:>10d}  "
            f"{r['cf_per_session_total_touches']:>8d}  "
            f"{delta:>+6d}"
        )

    print(f"  {'-' * 60}")
    delta_total = cf_total_cf_rth - cf_total_actual_rth
    print(
        f"  {'TOTAL':10s}  {totals['zones']:>5d}  "
        f"{cf_total_actual_rth:>10d}  {cf_total_cf_rth:>10d}  "
        f"{'':>8s}  {delta_total:>+6d}"
    )
    if cf_total_actual_rth > 0:
        pct_increase = (cf_total_cf_rth - cf_total_actual_rth) / cf_total_actual_rth * 100
        print(f"\n  >> Per-session first-touch would yield {cf_total_cf_rth} RTH touches "
              f"vs {cf_total_actual_rth} actual ({pct_increase:+.0f}%)")

    # ---- Counterfactual: RTH-only spending ----
    print(f"\n\n  --- Counterfactual B: Only RTH Touches Count as 'Spent' ---\n")
    print(f"  In this mode, London/Asia touches are ignored for spending.")
    print(f"  All {totals['zones']} zones remain available at RTH open every day.\n")

    header_cf2 = (
        f"  {'Date':10s}  {'Zones':>5s}  "
        f"{'Actual RTH':>10s}  {'CF RTH Tch':>10s}  "
        f"{'Non-RTH Ign':>11s}  {'Delta':>6s}"
    )
    print(header_cf2)
    print(f"  {'-' * 65}")

    cf2_total_actual_rth = 0
    cf2_total_cf_rth = 0

    for r in results:
        actual_rth = r["touched_rth"]
        cf_rth = r["cf_rth_only_rth_touches"]
        non_rth_ignored = r["cf_rth_only_non_rth_touches_ignored"]
        delta = cf_rth - actual_rth
        cf2_total_actual_rth += actual_rth
        cf2_total_cf_rth += cf_rth
        print(
            f"  {r['date']:10s}  {r['total_zones']:>5d}  "
            f"{actual_rth:>10d}  {cf_rth:>10d}  "
            f"{non_rth_ignored:>11d}  "
            f"{delta:>+6d}"
        )

    print(f"  {'-' * 65}")
    delta2_total = cf2_total_cf_rth - cf2_total_actual_rth
    print(
        f"  {'TOTAL':10s}  {totals['zones']:>5d}  "
        f"{cf2_total_actual_rth:>10d}  {cf2_total_cf_rth:>10d}  "
        f"{'':>11s}  {delta2_total:>+6d}"
    )
    if cf2_total_actual_rth > 0:
        pct_increase2 = (cf2_total_cf_rth - cf2_total_actual_rth) / cf2_total_actual_rth * 100
        print(f"\n  >> RTH-only spending would yield {cf2_total_cf_rth} RTH touches "
              f"vs {cf2_total_actual_rth} actual ({pct_increase2:+.0f}%)")

    # ---- Detailed touch breakdown ----
    print(f"\n\n  --- Detailed Touch Log (zones consumed before RTH) ---\n")

    pre_rth_details = []
    for r in results:
        for t in r.get("_touch_details", []):
            if t["session"] in ("asia", "london", "pre_market"):
                ts_et = t["timestamp"].astimezone(ET)
                pre_rth_details.append({
                    "date": r["date"],
                    "time_et": ts_et.strftime("%H:%M:%S"),
                    "session": t["session"],
                    "zone_price": t["zone_price"],
                    "zone_side": t["zone_side"],
                    "direction": t["direction"],
                })

    if pre_rth_details:
        print(
            f"  {'Date':10s}  {'Time(ET)':8s}  {'Session':12s}  "
            f"{'Zone Price':>10s}  {'Side':6s}  {'Direction':5s}"
        )
        print(f"  {'-' * 60}")
        for d in pre_rth_details:
            print(
                f"  {d['date']:10s}  {d['time_et']:8s}  {d['session']:12s}  "
                f"{d['zone_price']:>10.2f}  {d['zone_side']:6s}  {d['direction']:5s}"
            )
    else:
        print("  (no pre-RTH touches found)")

    # ---- Summary recommendation ----
    print(f"\n\n{'=' * 100}")
    print("  SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 100}\n")

    print(f"  Current behavior:")
    print(f"    - {totals['zones']} total zones across {n} trading days")
    if totals["zones"] > 0:
        print(f"    - {totals['pre_rth']} ({totals['pre_rth']/totals['zones']*100:.0f}%) spent before RTH")
        print(f"    - {totals['rth_tch']} ({totals['rth_tch']/totals['zones']*100:.0f}%) touched during RTH")
        print(f"    - {totals['rth_sig']} RTH signals, {totals['rth_exec']} executable")
        print(f"    - {totals['all_sig']} total signals across all sessions")

    print(f"\n  Counterfactual A (per-session first-touch):")
    if cf_total_actual_rth > 0:
        print(f"    - Would increase RTH touches from {cf_total_actual_rth} to {cf_total_cf_rth} "
              f"({(cf_total_cf_rth - cf_total_actual_rth) / cf_total_actual_rth * 100:+.0f}%)")
    print(f"    - Pro: More RTH trade opportunities")
    print(f"    - Con: Same zone may generate conflicting signals across sessions")

    print(f"\n  Counterfactual B (RTH-only spending):")
    if cf2_total_actual_rth > 0:
        print(f"    - Would increase RTH touches from {cf2_total_actual_rth} to {cf2_total_cf_rth} "
              f"({(cf2_total_cf_rth - cf2_total_actual_rth) / cf2_total_actual_rth * 100:+.0f}%)")
    print(f"    - Pro: All zones available at RTH, more signals")
    print(f"    - Con: Non-RTH touches still provide information (price reacted)")
    print(f"    - Trade-off: Higher signal count but may include lower-quality setups")

    print()


# ---- Main ----

def main() -> None:
    parser = argparse.ArgumentParser(description="First-touch rule diagnostic")
    parser.add_argument("--start", type=str, default="2025-06-02")
    parser.add_argument("--end", type=str, default="2025-07-08")
    args = parser.parse_args()

    all_dates = get_available_dates()
    if len(all_dates) < 3:
        print("ERROR: Need at least 3 dates")
        sys.exit(1)

    # Model
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Select date range
    start_idx = next(
        (i for i, d in enumerate(all_dates) if d >= args.start), None,
    )
    end_idx = next(
        (i for i, d in enumerate(reversed(all_dates)) if d <= args.end), None,
    )
    if start_idx is None or end_idx is None:
        print(f"ERROR: Date range {args.start} to {args.end} not available")
        sys.exit(1)
    end_idx = len(all_dates) - 1 - end_idx
    selected_dates = all_dates[start_idx : end_idx + 1]

    print(f"\n{'=' * 100}")
    print("  FIRST-TOUCH RULE DIAGNOSTIC")
    print(f"{'=' * 100}")
    print(f"\n  Date range: {selected_dates[0]} to {selected_dates[-1]} ({len(selected_dates)} days)")

    # Components
    price_buffer = PriceBuffer(max_duration=timedelta(hours=72))
    feature_computer = FeatureComputer()
    model_mgr = ModelManager(Path("data/models"))
    version = model_mgr.upload_model(MODEL_PATH)
    model_mgr.activate_model(version["id"])

    conn = duckdb.connect()
    results: list[dict] = []

    print(f"\n  Processing...\n")

    for i, date_str in enumerate(selected_dates):
        t0 = time_mod.monotonic()
        result = analyze_day(price_buffer, date_str, conn, feature_computer, model_mgr)
        elapsed = time_mod.monotonic() - t0
        results.append(result)

        # Progress
        pre_rth = result["touched_before_rth"]
        rth = result["touched_rth"]
        zones = result["total_zones"]
        sys.stdout.write(
            f"\r  [{i+1:2d}/{len(selected_dates)}] {date_str}  "
            f"zones={zones}  pre-RTH={pre_rth}  RTH={rth}  "
            f"sigs={result['all_signals']}  [{elapsed:.1f}s]"
        )
        sys.stdout.flush()

    print()  # Newline after progress
    conn.close()

    # Evict buffer
    price_buffer.evict()

    # Print results
    print_results(results)

    # Save CSV (exclude internal detail fields)
    csv_rows = []
    for r in results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        csv_rows.append(row)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(csv_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
