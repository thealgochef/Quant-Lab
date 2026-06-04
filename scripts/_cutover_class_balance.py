"""Ad-hoc validation: NEW (trade-bar + honest-entry) vs OLD (book-mid level-entry)
class balance on a small sample. INFORMATIONAL ONLY — not committed logic.

NEW  = build_utility_dataset(use_engine=True) production default
       (trade bars 0.25, trade-print interaction features, decision-time honest outcome).
OLD  = engine book-mid mode (price_source="book_mid", tick_size=0.125, honest_entry=False)
       on book-mid bars == the legacy book-mid level-entry labeling.

Run:  PYTHONPATH=src python scripts/_cutover_class_balance.py
"""
from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.simplefilter("ignore")

from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as B
from alpha_lab.agents.data_infra.ml import engine_decision as E
from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig, MLPipelineConfig

DATA_DIR = Path(r"C:/Users/gonza/Documents/Trade-Dashboard/data/databento")
SYMBOL = "NQ"
SAMPLE = [
    "2026-02-09", "2026-02-10", "2026-02-11", "2026-02-12", "2026-02-13",
    "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
]  # ~10 trade days (16th is a holiday in the store; skipped if absent)


def _cfg() -> MLPipelineConfig:
    return MLPipelineConfig(
        training_mode="dashboard_utility",
        instrument="NQ",
        tick_size=0.25,
        dashboard_utility=DashboardUtilityConfig(
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


def _bookmid_bars(date_str, util_cfg):
    from alpha_lab.agents.data_infra.tick_store import TickStore

    td = date.fromisoformat(date_str)
    prev = td - timedelta(days=1)
    s = pd.Timestamp(f"{prev} 18:00:00", tz="America/New_York").tz_convert("UTC")
    e = pd.Timestamp(f"{td} 18:00:00", tz="America/New_York").tz_convert("UTC")
    tc = int(util_cfg.bar_type[:-1])
    st = TickStore(DATA_DIR)
    try:
        for d in (prev, td):
            st.register_symbol_date(SYMBOL, d)
        return st.build_tick_bars(SYMBOL, s, e, tick_count=tc, price_source="book_mid")
    finally:
        st.close()


def _balance(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    vc = df["label"].value_counts().to_dict()
    n = int(len(df))
    return {"n": n, **{k: int(v) for k, v in vc.items()},
            **{f"{k}_pct": round(100 * v / n, 1) for k, v in vc.items()}}


def main():
    cfg = _cfg()
    util = cfg.dashboard_utility
    dates = [d for d in SAMPLE if (DATA_DIR / SYMBOL / d).is_dir()]

    # NEW (production default: trade bars + trade-print features + honest entry)
    new_df = B.build_utility_dataset(dates, DATA_DIR, cfg, use_engine=True)

    # OLD (book-mid level-entry) via the engine's book-mid regression mode.
    prev_ny = prev_asia = prev_london = None
    # warm prior-day levels from the day before the first sample day
    warm = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    if (DATA_DIR / SYMBOL / warm).is_dir():
        prev_ny, prev_asia, prev_london = B._get_session_hl_for_date(
            DATA_DIR, SYMBOL, warm, util, prev_ny, prev_asia, prev_london
        )
    old_frames = []
    for ds in dates:
        bm = _bookmid_bars(ds, util)
        if bm.empty:
            prev_ny, prev_asia, prev_london = B._get_session_hl_for_date(
                DATA_DIR, SYMBOL, ds, util, prev_ny, prev_asia, prev_london)
            continue
        bars_et = B._ensure_et_index(bm.copy())
        levels = B._compute_levels_for_date(bars_et, ds, prev_ny, prev_asia, prev_london)
        df = E.process_single_date_engine(
            bars_et, levels, ds, DATA_DIR, SYMBOL, util,
            price_source="book_mid", tick_size=E.BOOK_MID_TICK, honest_entry=False,
        )
        if not df.empty:
            old_frames.append(df)
        prev_ny, prev_asia, prev_london = B._get_session_hl_for_date(
            DATA_DIR, SYMBOL, ds, util, prev_ny, prev_asia, prev_london)
    old_df = pd.concat(old_frames, ignore_index=True) if old_frames else pd.DataFrame()

    print("DATES:", dates)
    print("NEW (trade-bar + decision-time honest entry):", _balance(new_df))
    print("OLD (book-mid + level entry):                ", _balance(old_df))
    has_int = [c for c in ("int_time_beyond_level", "int_time_within_2pts", "int_absorption_ratio") if c in new_df.columns]
    print("NEW interaction feature cols present:", has_int)


if __name__ == "__main__":
    main()
