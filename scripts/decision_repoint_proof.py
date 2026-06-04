"""Phase-5 Part-1 plumbing proof: engine decision layer == legacy CQL decision layer.

Runs BOTH paths on a sample of book-mid days and prints a per-day table proving the
``strategy_core`` engine repoint (``engine_decision``) reproduces the legacy
duplicate CQL decision output EXACTLY on the SAME bars + levels:

  * zones (set of (rep_price, side, names))
  * touches (count + (level_type, bar ts, direction, rep_price))
  * labels (class + max_mfe / max_mae)
  * the 6 model features (max abs diff, expect 0)

This is the anchor of the repoint: bars stay book-mid (``price_source="book_mid"``),
nothing about the strategy changes. The ONE known parity_harness exception is the
23:59-ET midnight-crossing interaction window-bound artifact; if a sample touch hits
it, it is reported as the known harness/window artifact, NOT a repoint bug.

Run:
    cd C:/Users/gonza/Documents/Claude-Quant-Lab
    python scripts/decision_repoint_proof.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as B
from alpha_lab.agents.data_infra.ml import engine_decision as E
from alpha_lab.agents.data_infra.ml.config import MLPipelineConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import (
    NO_RESOLUTION,
    label_touch_event,
)

DATA_DIR = Path(r"C:/Users/gonza/Documents/Trade-Dashboard/data/databento")
SYMBOL = "NQ"
_ET = "US/Eastern"

# Sample of 4a book-mid days present in the store. 2025-07-09 has a missing prior
# calendar day (07-08), exercising the gap; the others are contiguous.
SAMPLE_DAYS = ["2025-07-09", "2025-07-10", "2025-07-11", "2025-07-14", "2025-07-15"]
WARMUP_DAY = "2025-07-07"  # prior available trading day, for PDH/PDL carry

PROD_CONFIG = MLPipelineConfig(
    training_mode="dashboard_utility",
    instrument="NQ",
    tick_size=0.25,
    dashboard_utility=dict(
        bar_type="147t",
        interaction_window_minutes=5,
        approach_window_minutes=30,
        include_approach_features=True,
        tp_points=15.0,
        sl_points=30.0,
        trap_mfe_min=5.0,
        level_proximity_pts=0.5,
    ),
)
U = PROD_CONFIG.dashboard_utility

MODEL_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
    "app_avg_trade_size",
    "app_large_trade_vol_pct",
    "app_max_spread",
]


def _advance_carry(date_str, prev_ny, prev_asia, prev_london):
    return B._get_session_hl_for_date(
        DATA_DIR, SYMBOL, date_str, U, prev_ny, prev_asia, prev_london
    )


def _zone_key(z) -> tuple:
    """Engine Zone or legacy zone dict -> comparable key."""
    if isinstance(z, dict):
        return (round(float(z["representative_price"]), 6), z["side"], tuple(z["names"]))
    return (round(float(z.representative_price), 6), z.side.value, tuple(z.names))


def _touch_key_legacy(t) -> tuple:
    return (
        t["level_type"],
        E._to_utc_dt(pd.Timestamp(t["bar_ts"])).isoformat(),
        t["direction"],
        round(float(t["representative_price"]), 6),
    )


def _touch_key_engine(t) -> tuple:
    return (
        t.level_type,
        t.bar_ts_utc.isoformat(),
        t.direction.value,
        round(float(t.representative_price), 6),
    )


def _bookmid_bars(date_str):
    """Book-mid 147t bars (the cutover hardwired _build_bars_for_date to trade).

    This proof is a BOOK-MID engine==legacy proof; build book-mid bars directly so
    the trade-bar cutover does not turn it into a trade-vs-book mismatch.
    """
    from datetime import date as _date, timedelta as _td

    from alpha_lab.agents.data_infra.tick_store import TickStore

    td = _date.fromisoformat(date_str)
    prev = td - _td(days=1)
    s = pd.Timestamp(f"{prev} 18:00:00", tz="America/New_York").tz_convert("UTC")
    e = pd.Timestamp(f"{td} 18:00:00", tz="America/New_York").tz_convert("UTC")
    tc = int(U.bar_type[:-1])
    st = TickStore(DATA_DIR)
    try:
        for d in (prev, td):
            st.register_symbol_date(SYMBOL, d)
        return st.build_tick_bars(SYMBOL, s, e, tick_count=tc, price_source="book_mid")
    finally:
        st.close()


def run_day(date_str, prev_ny, prev_asia, prev_london) -> dict:
    bars = _bookmid_bars(date_str)
    if bars.empty:
        return {"day": date_str, "skip": "no bars"}
    bars_et = B._ensure_et_index(bars.copy())
    levels = B._compute_levels_for_date(
        bars_et, date_str, prev_ny, prev_asia, prev_london
    )

    td = date.fromisoformat(date_str)

    # ── zones ───────────────────────────────────────────────────────────────
    legacy_zones = B._build_zones(levels)
    eng_zones = E.build_zones(E.levels_to_engine(levels))
    zones_match = sorted(_zone_key(z) for z in legacy_zones) == sorted(
        _zone_key(z) for z in eng_zones
    )

    # ── touches (fresh zones; detection mutates 'touched') ──────────────────
    legacy_touches = B._detect_touches(bars_et, B._build_zones(levels))
    eng_touches = E.detect_touches(
        E.bars_et_to_engine(bars_et, td, E.BOOK_MID_TICK),
        E.build_zones(E.levels_to_engine(levels)),
        tick_size=E.BOOK_MID_TICK,
        trading_day=td,
    )
    legacy_tk = [_touch_key_legacy(t) for t in legacy_touches]
    eng_tk = [_touch_key_engine(t) for t in eng_touches]
    touches_match = legacy_tk == eng_tk

    # ── labels (per touch with resolvable forward window) ───────────────────
    rth_cutoff = pd.Timestamp(f"{date_str} 16:15:00", tz=_ET)
    labels_match = True
    label_mismatch = None
    for lt, et in zip(legacy_touches, eng_touches):
        forward = bars_et[(bars_et.index > lt["bar_ts"]) & (bars_et.index < rth_cutoff)]
        if forward.empty:
            continue
        legacy_lab = label_touch_event(lt, forward, U)
        eng_out = E.resolve_outcome(
            entry_points=float(et.representative_price),
            direction=et.direction,
            forward_bars=E.bars_et_to_engine(forward, td, E.BOOK_MID_TICK),
            tick_size=E.BOOK_MID_TICK,
            tp_points=U.tp_points,
            sl_points=U.sl_points,
            trap_mfe_min=U.trap_mfe_min,
        )
        ok = (
            legacy_lab["label"] == eng_out.label
            and legacy_lab["label_encoded"] == eng_out.label_encoded
            and abs(round(float(legacy_lab["max_mfe"]), 4) - round(float(eng_out.max_mfe), 4)) < 1e-9
            and abs(round(float(legacy_lab["max_mae"]), 4) - round(float(eng_out.max_mae), 4)) < 1e-9
        )
        if not ok and label_mismatch is None:
            label_mismatch = (
                f"legacy({legacy_lab['label']},mfe={legacy_lab['max_mfe']},mae={legacy_lab['max_mae']}) "
                f"vs eng({eng_out.label},mfe={eng_out.max_mfe},mae={eng_out.max_mae})"
            )
        labels_match = labels_match and ok

    # ── 6 features (max abs diff over all kept touches) ─────────────────────
    feat_max_abs = 0.0
    feat_worst = None
    kept = 0
    artifact_note = None
    for lt, et in zip(legacy_touches, eng_touches):
        legacy_int = B._compute_interaction_features(lt, DATA_DIR, SYMBOL, U)
        if legacy_int is None:
            continue
        eng_int = E.compute_interaction_features_engine(
            et, DATA_DIR, SYMBOL, U,
            price_source="book_mid", tick_size=E.BOOK_MID_TICK,
        )
        if eng_int is None:
            # legacy kept it, engine dropped (or vice versa) -> a real divergence
            feat_worst = f"interaction None mismatch @ {et.bar_ts_utc.isoformat()}"
            feat_max_abs = float("inf")
            continue
        legacy_app = B._compute_approach_features(lt, DATA_DIR, SYMBOL, U) or {}
        eng_app = E.compute_approach_features_engine(et, DATA_DIR, SYMBOL, U) or {}
        kept += 1

        merged_legacy = {**legacy_int, **legacy_app}
        merged_eng = {**eng_int, **eng_app}
        for f in MODEL_FEATURES:
            lv = merged_legacy.get(f)
            ev = merged_eng.get(f)
            if lv is None or ev is None:
                continue
            if np.isnan(lv) and np.isnan(ev):
                continue
            d = abs(float(lv) - float(ev))
            if d > feat_max_abs:
                feat_max_abs = d
                feat_worst = (
                    f"{f} @ {et.bar_ts_utc.isoformat()}: legacy={lv} eng={ev} (|d|={d})"
                )
                # Known harness/window artifact: a touch whose 5-min interaction
                # window crosses UTC midnight from a ~23:59-ET bar.
                ts_et = pd.Timestamp(et.bar_ts_utc).tz_convert(_ET)
                win_end = pd.Timestamp(et.bar_ts_utc) + timedelta(
                    minutes=U.interaction_window_minutes
                )
                if pd.Timestamp(et.bar_ts_utc).date() != win_end.date() and f.startswith("int_"):
                    artifact_note = (
                        f"{date_str}: {f} diff on a midnight-UTC-crossing interaction "
                        f"window (touch {ts_et.isoformat()}); known parity_harness window "
                        f"artifact, not a repoint bug."
                    )

    return {
        "day": date_str,
        "n_touches_legacy": len(legacy_touches),
        "n_touches_engine": len(eng_touches),
        "kept_feature_touches": kept,
        "zones_match": zones_match,
        "touches_match": touches_match,
        "labels_match": labels_match,
        "label_mismatch": label_mismatch,
        "feature_max_abs_diff": feat_max_abs,
        "feature_worst": feat_worst,
        "artifact_note": artifact_note,
    }


def main():
    prev_ny = prev_asia = prev_london = None
    # warm carry
    prev_ny, prev_asia, prev_london = _advance_carry(
        WARMUP_DAY, prev_ny, prev_asia, prev_london
    )

    rows = []
    for d in SAMPLE_DAYS:
        r = run_day(d, prev_ny, prev_asia, prev_london)
        rows.append(r)
        prev_ny, prev_asia, prev_london = _advance_carry(
            d, prev_ny, prev_asia, prev_london
        )
        print(
            f"{d}: touches L/E={r.get('n_touches_legacy')}/{r.get('n_touches_engine')} "
            f"kept={r.get('kept_feature_touches')} zones={r.get('zones_match')} "
            f"touches={r.get('touches_match')} labels={r.get('labels_match')} "
            f"feat_max_abs={r.get('feature_max_abs_diff')}",
            flush=True,
        )
        if r.get("label_mismatch"):
            print(f"    label mismatch: {r['label_mismatch']}", flush=True)
        if r.get("feature_worst") and r.get("feature_max_abs_diff", 0) > 1e-9:
            print(f"    feature worst: {r['feature_worst']}", flush=True)
        if r.get("artifact_note"):
            print(f"    ARTIFACT: {r['artifact_note']}", flush=True)

    print("\n=== Per-day proof table ===")
    print(
        f"{'day':12} {'tch L/E':9} {'kept':5} {'zones':6} {'touch':6} "
        f"{'label':6} {'feat_max_abs':12}"
    )
    all_clean = True
    for r in rows:
        if "skip" in r:
            print(f"{r['day']:12} SKIP {r['skip']}")
            continue
        clean = (
            r["zones_match"]
            and r["touches_match"]
            and r["labels_match"]
            and r["feature_max_abs_diff"] <= 1e-9
        )
        all_clean = all_clean and clean
        print(
            f"{r['day']:12} {r['n_touches_legacy']}/{r['n_touches_engine']:<7} "
            f"{r['kept_feature_touches']:<5} {str(r['zones_match']):6} "
            f"{str(r['touches_match']):6} {str(r['labels_match']):6} "
            f"{r['feature_max_abs_diff']:<12}"
        )
    print(f"\nALL CLEAN (engine == legacy EXACTLY): {all_clean}")


if __name__ == "__main__":
    main()
