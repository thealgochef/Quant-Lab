"""Honest-edge audit harness for the production book-mid 3-class reversal model.

WHAT THIS AUDITS (and what it does NOT change)
==============================================
The production model is
``models/NQ_20260405_147t_5m_30m_multiclass-250602-260220-iterations800_depth4``.
Its reported edge (``evaluation.json``) is ``expectancy_15_30_pts = 11.71``,
``profit_factor_15_30 = 6.333`` over ``n_simulated_trades = 574`` OOS gated touches
across a 44-fold walk-forward.

That 11.71 is IDEALIZED: the label path measures MFE/MAE from the LEVEL price at the
touch INSTANT with ZERO costs.  But the model's decision needs the post-touch
5-minute interaction features, so the signal is not actionable until +5m.  This
harness measures the HONEST edge: a realistic +5-minute entry at the prevailing
book-mid, with stated slippage and commissions, on the SAME book-mid bars.

HARD CONSTRAINTS honored here:
  * Bars stay book-mid throughout (``_build_bars_for_date`` is NOT touched;
    ``price_source="book_mid"``).  We audit the EXISTING book-mid model.
  * NO new/different model is trained.  We REPRODUCE the existing walk-forward
    eval with the SAME spec (train=40d / test=5d / gap=2d ROLLING; CatBoost
    iterations=800, depth=4, MultiClass, auto_class_weights=Balanced; RFECV on,
    random_seed=42) to recover the per-touch OOS predictions.  No feature / HP /
    fold-scheme change.
  * This file lives in CQL and imports alpha_lab / strategy_core only.  It does
    NOT touch Trade-Lab.

PIPELINE (3 steps)
==================
STEP 1 -- ENRICHED per-day dataset (one pass per day):
    For each date, build book-mid bars + levels ONCE (reusing the production
    builder), run the engine decision layer (``process_single_date_engine``) to
    get the standard dashboard_utility rows (event_ts, level_type,
    representative_price, direction, label, label_encoded, IDEALIZED
    max_mfe/max_mae from the LEVEL entry, the 6 features), and THEN add the HONEST
    trade outcome per touch on the SAME book-mid bars:
      - entry_ts = touch bar close ts + interaction_window (5m).  If entry_ts is
        at/after 15:55 ET (flatten) OR there is no bar/quote available by 16:15 ET,
        traded=False (no honest trade).  Else traded=True.
      - entry_price = the front-month BOOK-MID at entry_ts (most-recent
        (bid_px_00+ask_px_00)/2 with ts_event <= entry_ts).  ASSUMPTION: book-mid
        at +5m, because the model itself is a book-mid model and the bars it trades
        are book-mid; the honest fill should be on the same surface.  A STATED entry
        slippage of 1 tick (0.25 pt) is applied adverse to the fill (long pays up,
        short sells down).
      - honest forward scan: book-mid bars with close_ts strictly in
        (entry_ts, 16:15 ET].  Bracket TP +15 / SL -30 FROM THE REAL ENTRY (not the
        level), MAE-FIRST (the adverse side is checked first each bar, matching
        ``resolve_outcome`` / the labeler).  LONG: mfe=high-entry, mae=entry-low;
        SHORT mirrored.  If neither bracket is hit by 16:15, exit at the 16:15 mark
        (last bar close at/before the cutoff) = mark-to-market.  Exit slippage of 1
        tick (0.25 pt) is applied adverse to the exit fill.
      - honest_gross_pts is signed and ALREADY INCLUDES entry+exit slippage.

STEP 2 -- REPRODUCE the walk-forward OOS gated set (NOT a new model): sort the
    enriched touches by event_ts; build folds with ``WalkForwardSplitter`` using
    the PRODUCTION config (train=40 / test=5 / gap=2, ROLLING -- expanding=False,
    confirmed from evaluation.json full_config).  RFECV once on the full feature
    set (reproducing the production pipeline), then per fold train
    CatBoost(800, depth4, MultiClass, Balanced) on the TRAIN touches' selected
    features and predict_proba on the TEST touches -> per-touch
    prob_reversal/prob_trap/prob_blowthrough + fold id + argmax predicted label.

STEP 3 -- GATE + METRICS:
    IDEALIZED gate (reproduces 11.71): argmax predicted class ==
    tradeable_reversal (i.e. raw_pred == 0), exactly as ``ml_training_tab`` builds
    the OOS confusion matrix that feeds ``compute_utility_metrics``.  We confirm
    this against the dataset by recomputing tp/fp on the SAME (eval_y, eval_pred).
    PRODUCTION serve gate (honest): prob_reversal >= 0.70 AND session==ny_rth AND
    eligible_class tradeable_reversal (strategy.json).
    (a) IDEALIZED: ``compute_utility_metrics(tp15, sl30)`` on the OOS confusion
        matrix from the idealized max_mfe/max_mae labels -> expectancy_pts
        (TARGET ~11.71 on the full range), PF, n.
    (b) HONEST: from honest_gross_pts of the TRADED gated touches -> n, hit rate,
        avg win, avg loss, expectancy GROSS and NET (pts and $ at point_value=20),
        PF, total net PnL, max drawdown (equity ordered by event_ts).
    (c) BASELINE: ALL eligible ny_rth touches (NO model gate), same honest entry +
        costs -> same metric block.
    Plus a per-fold table (n gated, gross/net expectancy, hit rate).

COST SCHEME (stated, no double-counting)
========================================
  * Slippage: 1 tick / side, modeled DIRECTLY in the entry and exit fills
    (entry adverse 0.25pt + exit adverse 0.25pt = 0.5pt = $10 round-turn adverse).
    Therefore honest_gross_pts is already slippage-inclusive.
  * Commissions: (exchange_nfa + broker) * 2 sides = (2.14 + 0.50) * 2 = $5.28 per
    round turn, SUBTRACTED from the $ P&L only (NOT re-applied as slippage).
  * Cross-check vs instruments.yaml NQ total_round_turn 7.78: that figure bundles
    commission $5.28 + slippage $2.50/side... wait -- instruments.yaml uses
    avg_slippage 0.5 tick = $2.50/side = $5.00 RT, giving 5.28 + 5.00 = $10.28.
    We instead model slippage as exactly 1 tick/side = $5.00 RT inside the fills and
    add $5.28 commission, for $10.28 total adverse round-turn.  (We DOUBLED the
    instruments.yaml 0.5-tick/side slippage to a full 1 tick/side because the prompt
    specifies a 1-tick adverse fill on EACH of entry and exit.)  Every number is in
    the results json under ``cost_scheme``.

USAGE
=====
    cd C:/Users/gonza/Documents/Claude-Quant-Lab
    PYTHONPATH=src python scripts/honest_edge_audit.py \
        --start 2025-06-02 --end 2026-02-22 \
        --out-parquet data/experiment/honest_edge/enriched.parquet \
        --results-json data/experiment/honest_edge/results.json

CHUNKABLE + RESUMABLE: STEP 1 appends per-date to --out-parquet and skips dates
already present, so the full range can be built in <600s Bash chunks by calling
``build_enriched`` (or the CLI with --enrich-only) over sub-ranges.  STEP 2/3
(``run_walk_forward_audit``) read the full parquet and emit the json; run once at
the end (or with --analyze-only) over whatever is in the parquet.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

import strategy_core as sc

from alpha_lab.agents.data_infra.ml.config import (
    DashboardUtilityConfig,
    MLPipelineConfig,
    ModelConfig,
    WalkForwardConfig,
)
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _build_bars_for_date,
    _compute_levels_for_date,
    _ensure_et_index,
    _get_session_hl_for_date,
)
from alpha_lab.agents.data_infra.ml.engine_decision import (
    BOOK_MID_TICK,
    process_single_date_engine,
)
from alpha_lab.agents.data_infra.ml.model_evaluator import ModelEvaluator
from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer
from alpha_lab.agents.data_infra.ml.walk_forward import WalkForwardSplitter
from alpha_lab.agents.data_infra.tick_store import TickStore

# compute_utility_metrics lives in scripts/ml_training_tab.py.  Import it by path
# (the scripts dir is not a package).
import importlib.util as _ilu

_THIS = Path(__file__).resolve()
_MLT_PATH = _THIS.parent / "ml_training_tab.py"
_spec = _ilu.spec_from_file_location("_mlt_for_audit", _MLT_PATH)
_mlt = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mlt)  # type: ignore[union-attr]
compute_utility_metrics = _mlt.compute_utility_metrics

logger = logging.getLogger("honest_edge_audit")

ET = ZoneInfo("America/New_York")
_ET_STR = "US/Eastern"

# ── Fixed production constants (do NOT change -- audit of the EXISTING model) ──
DATA_DIR = Path("C:/Users/gonza/Documents/Trade-Dashboard/data/databento")
SYMBOL = "NQ"
INSTRUMENTS_YAML = _THIS.parent.parent / "config" / "instruments.yaml"

# Known unusable sessions in the store (recon facts): skip cleanly.
KNOWN_GAPS = {"2025-07-08", "2025-11-14", "2025-11-20"}

# Flatten / cutoff (ET).
FLATTEN_HOUR, FLATTEN_MINUTE = 15, 55          # no new trades at/after 15:55 ET
RTH_CUTOFF_HM = (16, 15)                         # forward scan / mark cutoff

# Stated execution assumptions.
ENTRY_SLIPPAGE_TICKS = 1.0                       # 1 tick adverse on entry fill
EXIT_SLIPPAGE_TICKS = 1.0                        # 1 tick adverse on exit fill
TICK_SIZE = 0.25                                 # NQ
POINT_VALUE = 20.0                               # NQ $/pt
ENTRY_BOOK_LOOKBACK_MIN = 30                     # how far back to find a book quote

# Production gate.
CONF_GATE = 0.70
ELIGIBLE_CLASS_ENCODED = 0                       # tradeable_reversal
NY_RTH = (datetime.min.time().replace(hour=9, minute=30),
          datetime.min.time().replace(hour=16, minute=15))


# ════════════════════════════════════════════════════════════════════════════
# Production config (the EXACT spec that produced 11.71 / 574 / 6.333)
# ════════════════════════════════════════════════════════════════════════════

def build_production_config() -> MLPipelineConfig:
    """The MLPipelineConfig matching evaluation.json full_config.

    walk_forward: train=40, test=5, gap=2, expanding=False (ROLLING).
    model: catboost iterations=800 depth=4 MultiClass Balanced rfecv_enabled=True.
    dashboard_utility: 147t book-mid bars, 5m interaction, 30m approach,
                       tp15/sl30, trap_mfe_min5, include_approach_features=True.
    """
    return MLPipelineConfig(
        training_mode="dashboard_utility",
        walk_forward=WalkForwardConfig(
            train_days=40, test_days=5, gap_days=2, expanding=False,
        ),
        model=ModelConfig(
            model_type="catboost",
            iterations=800,
            depth=4,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            rfecv_enabled=True,
            rfecv_min_features=5,
        ),
        dashboard_utility=DashboardUtilityConfig(
            tp_points=15.0,
            sl_points=30.0,
            trap_mfe_min=5.0,
            interaction_window_minutes=5,
            level_proximity_pts=0.50,
            bar_type="147t",
            include_approach_features=True,
            approach_window_minutes=30,
        ),
        tick_size=0.25,
        instrument="NQ",
    )


def load_nq_cost_spec() -> dict:
    """Load the NQ cost figures from config/instruments.yaml."""
    with open(INSTRUMENTS_YAML) as fh:
        spec = yaml.safe_load(fh)
    return spec["NQ"]


# ════════════════════════════════════════════════════════════════════════════
# Date helpers
# ════════════════════════════════════════════════════════════════════════════

def available_dates(start: str, end: str) -> list[str]:
    """Sorted store dates in [start, end] that exist and are not known gaps."""
    out: list[str] = []
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    d = d0
    while d <= d1:
        ds = d.isoformat()
        if ds not in KNOWN_GAPS and (DATA_DIR / SYMBOL / ds).is_dir():
            out.append(ds)
        d += timedelta(days=1)
    return out


def _rth_cutoff_ts(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(f"{date_str} {RTH_CUTOFF_HM[0]:02d}:{RTH_CUTOFF_HM[1]:02d}:00", tz=_ET_STR)


def _is_after_flatten(ts_et: pd.Timestamp) -> bool:
    return (ts_et.hour > FLATTEN_HOUR) or (
        ts_et.hour == FLATTEN_HOUR and ts_et.minute >= FLATTEN_MINUTE
    )


def _in_ny_rth(ts_et: pd.Timestamp) -> bool:
    t = ts_et.time()
    return NY_RTH[0] <= t < NY_RTH[1]


# ════════════════════════════════════════════════════════════════════════════
# Front-month book-mid lookup at a timestamp (the +5m entry price source)
# ════════════════════════════════════════════════════════════════════════════

def _book_mid_at(store: TickStore, symbol: str, as_of_utc: datetime,
                 lookback_min: int = ENTRY_BOOK_LOOKBACK_MIN) -> float | None:
    """Most-recent front-month book-mid (bid_px_00+ask_px_00)/2 with ts <= as_of.

    Mirrors the front-month + book filters used in ``query_tick_feature_rows`` /
    ``engine_decision._query_quotes``.  Bounded lookback keeps the scan cheap.
    """
    views = store._get_views(symbol)
    if not views:
        return None
    union_sql = store._union_views_sql(views)
    sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
    cols = {r[0] for r in store._conn.execute(sample_sql).fetchall()}
    if not ("bid_px_00" in cols and "ask_px_00" in cols):
        return None
    if "symbol" in cols:
        front = store._conn.execute(
            f"SELECT symbol, count(*) AS n FROM ({union_sql}) AS t "
            f"WHERE symbol NOT LIKE '%-%' GROUP BY symbol ORDER BY n DESC LIMIT 1"
        ).fetchone()
        sym_f = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
    else:
        sym_f = ""
    lo = pd.Timestamp(as_of_utc) - pd.Timedelta(minutes=lookback_min)
    sql = f"""
        SELECT (bid_px_00 + ask_px_00) / 2.0 AS mid
        FROM ({union_sql}) AS t
        WHERE ts_event <= $1 AND ts_event >= $2
          AND bid_px_00 > 0 AND ask_px_00 > 0
          {sym_f}
        ORDER BY ts_event DESC LIMIT 1
    """
    row = store._conn.execute(sql, [pd.Timestamp(as_of_utc), lo]).fetchone()
    return float(row[0]) if row and row[0] is not None else None


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 -- honest forward scan on book-mid bars
# ════════════════════════════════════════════════════════════════════════════

def _honest_outcome(
    direction: str,
    entry_price: float,
    forward_bars: pd.DataFrame,
    tp_points: float,
    sl_points: float,
    exit_slippage_pts: float,
) -> dict:
    """MAE-first TP/SL bracket from the REAL entry on book-mid forward bars.

    forward_bars: ET-indexed OHLC, already restricted to (entry_ts, 16:15].
    Returns honest_exit_price, honest_exit_reason (tp/sl/cutoff), honest_gross_pts
    (signed, INCLUDING exit slippage; entry slippage is already baked into
    entry_price by the caller).
    """
    is_long = str(direction).lower() == "long"
    tp_level = entry_price + tp_points if is_long else entry_price - tp_points
    sl_level = entry_price - sl_points if is_long else entry_price + sl_points

    for _idx, bar in forward_bars.iterrows():
        hi = float(bar["high"])
        lo = float(bar["low"])
        # MAE-FIRST: check the adverse (stop) side before the favorable (target).
        if is_long:
            if lo <= sl_level:
                exit_px = sl_level - exit_slippage_pts
                return {"exit_price": exit_px, "exit_reason": "sl",
                        "gross_pts": exit_px - entry_price}
            if hi >= tp_level:
                exit_px = tp_level - exit_slippage_pts
                return {"exit_price": exit_px, "exit_reason": "tp",
                        "gross_pts": exit_px - entry_price}
        else:
            if hi >= sl_level:
                exit_px = sl_level + exit_slippage_pts
                return {"exit_price": exit_px, "exit_reason": "sl",
                        "gross_pts": entry_price - exit_px}
            if lo <= tp_level:
                exit_px = tp_level + exit_slippage_pts
                return {"exit_price": exit_px, "exit_reason": "tp",
                        "gross_pts": entry_price - exit_px}

    # Neither bracket hit by cutoff -> mark-to-market at the last bar's close.
    last_close = float(forward_bars.iloc[-1]["close"])
    if is_long:
        exit_px = last_close - exit_slippage_pts
        gross = exit_px - entry_price
    else:
        exit_px = last_close + exit_slippage_pts
        gross = entry_price - exit_px
    return {"exit_price": exit_px, "exit_reason": "cutoff", "gross_pts": gross}


def enrich_one_date(
    date_str: str,
    config: MLPipelineConfig,
    prev_levels_state: tuple,
) -> tuple[pd.DataFrame, tuple]:
    """Build the enriched per-touch frame for one date.

    Returns (enriched_df, new_prev_levels_state).  The state carries
    (prev_ny_hl, prev_asia_hl, prev_london_hl) forward so the NEXT day's levels
    use this day's session highs/lows -- identical to build_utility_dataset's loop.
    """
    util_cfg = config.dashboard_utility
    symbol = config.instrument
    prev_ny_hl, prev_asia_hl, prev_london_hl = prev_levels_state

    bars = _build_bars_for_date(DATA_DIR, symbol, date_str, util_cfg)
    if bars.empty:
        new_state = _get_session_hl_for_date(
            DATA_DIR, symbol, date_str, util_cfg,
            prev_ny_hl, prev_asia_hl, prev_london_hl,
        )
        return pd.DataFrame(), new_state

    bars_et = _ensure_et_index(bars)
    levels = _compute_levels_for_date(
        bars_et, date_str, prev_ny_hl, prev_asia_hl, prev_london_hl,
    )

    # Advance the level state for the next day (same call build_utility_dataset uses).
    new_state = _get_session_hl_for_date(
        DATA_DIR, symbol, date_str, util_cfg,
        prev_ny_hl, prev_asia_hl, prev_london_hl,
    )

    if not levels:
        return pd.DataFrame(), new_state

    # The standard dashboard_utility engine rows (idealized labels + 6 features).
    # This audit is a BOOK-MID audit (bars stay book-mid; idealized LEVEL-entry
    # labels). Pin the engine to its book-mid regression mode so the trade-bar +
    # honest-entry cutover (engine v2 production default) does NOT silently change
    # this harness's idealized baseline.
    base = process_single_date_engine(
        bars_et, levels, date_str, DATA_DIR, symbol, util_cfg,
        price_source="book_mid", tick_size=BOOK_MID_TICK, honest_entry=False,
    )
    if base.empty:
        return pd.DataFrame(), new_state

    window = util_cfg.interaction_window_minutes
    tp_points = util_cfg.tp_points
    sl_points = util_cfg.sl_points
    cutoff = _rth_cutoff_ts(date_str)
    entry_slip_pts = ENTRY_SLIPPAGE_TICKS * TICK_SIZE
    exit_slip_pts = EXIT_SLIPPAGE_TICKS * TICK_SIZE

    store = TickStore(DATA_DIR)
    store.register_symbol_date(symbol, date_str)
    try:
        honest_rows: list[dict] = []
        for _, r in base.iterrows():
            touch_ts_et = pd.Timestamp(r["event_ts"])  # ET-indexed bar close
            if touch_ts_et.tz is None:
                touch_ts_et = touch_ts_et.tz_localize(_ET_STR)
            entry_ts_et = touch_ts_et + pd.Timedelta(minutes=window)

            rec = {
                "traded": False,
                "honest_entry_ts": pd.NaT,
                "honest_entry_price": np.nan,
                "honest_exit_price": np.nan,
                "honest_exit_reason": None,
                "honest_gross_pts": np.nan,
            }

            # Reject: entry at/after flatten, or entry already past the cutoff.
            if _is_after_flatten(entry_ts_et) or entry_ts_et >= cutoff:
                honest_rows.append(rec)
                continue

            entry_ts_utc = entry_ts_et.tz_convert("UTC").to_pydatetime()
            mid = _book_mid_at(store, symbol, entry_ts_utc)
            if mid is None:
                honest_rows.append(rec)
                continue

            direction = r["direction"]
            is_long = str(direction).lower() == "long"
            # Entry slippage adverse: long pays up, short sells down.
            if is_long:
                entry_price = mid + entry_slip_pts
            else:
                entry_price = mid - entry_slip_pts

            forward = bars_et[(bars_et.index > entry_ts_et) & (bars_et.index <= cutoff)]
            if forward.empty:
                honest_rows.append(rec)
                continue

            outcome = _honest_outcome(
                direction, entry_price, forward,
                tp_points, sl_points, exit_slip_pts,
            )
            rec.update({
                "traded": True,
                "honest_entry_ts": entry_ts_et,
                "honest_entry_price": entry_price,
                "honest_exit_price": outcome["exit_price"],
                "honest_exit_reason": outcome["exit_reason"],
                "honest_gross_pts": outcome["gross_pts"],
            })
            honest_rows.append(rec)
    finally:
        store.close()

    enriched = base.reset_index(drop=True).copy()
    honest_df = pd.DataFrame(honest_rows).reset_index(drop=True)
    enriched = pd.concat([enriched, honest_df], axis=1)
    return enriched, new_state


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 driver: build / append enriched parquet (resumable)
# ════════════════════════════════════════════════════════════════════════════

def _existing_dates(out_parquet: Path) -> set[str]:
    if not out_parquet.exists():
        return set()
    try:
        existing = pd.read_parquet(out_parquet, columns=["date"])
        return set(existing["date"].astype(str).unique())
    except Exception:
        existing = pd.read_parquet(out_parquet)
        return set(existing["date"].astype(str).unique()) if "date" in existing else set()


def build_enriched(start: str, end: str, out_parquet: Path,
                   config: MLPipelineConfig | None = None) -> dict:
    """STEP 1 over [start, end].  Appends per date; skips dates already present.

    To keep levels correct, the loop ALWAYS warms each prior day's session H/L
    even for dates already cached in the parquet (cheap bar read), so a resumed
    chunk that starts mid-range still has the right prior-day levels for its first
    newly-built date.
    """
    config = config or build_production_config()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    done = _existing_dates(out_parquet)

    dates = available_dates(start, end)
    n_new = 0
    n_touch = 0
    prev_state: tuple = (None, None, None)
    util_cfg = config.dashboard_utility

    for ds in dates:
        if ds in done:
            # Still advance level state from this day's bars (cheap, keeps levels right).
            prev_state = _get_session_hl_for_date(
                DATA_DIR, SYMBOL, ds, util_cfg, *prev_state,
            )
            continue
        try:
            enriched, prev_state = enrich_one_date(ds, config, prev_state)
        except Exception:
            logger.exception("Failed to enrich %s; skipping", ds)
            # Best-effort: advance level state so subsequent days are not corrupted.
            try:
                prev_state = _get_session_hl_for_date(
                    DATA_DIR, SYMBOL, ds, util_cfg, *prev_state,
                )
            except Exception:
                pass
            continue

        if enriched.empty:
            logger.info("%s: 0 touches", ds)
            continue

        # Append incrementally (read-modify-write keeps a single parquet file).
        if out_parquet.exists():
            prior = pd.read_parquet(out_parquet)
            combined = pd.concat([prior, enriched], ignore_index=True)
        else:
            combined = enriched
        combined.to_parquet(out_parquet, index=False)
        done.add(ds)
        n_new += 1
        n_touch += len(enriched)
        logger.info("%s: %d touches (traded=%d)", ds, len(enriched),
                    int(enriched["traded"].sum()))

    return {"dates_in_range": len(dates), "dates_built": n_new,
            "touches_built": n_touch}


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 -- reproduce the walk-forward OOS predictions (SAME spec, no new model)
# ════════════════════════════════════════════════════════════════════════════

def reproduce_walk_forward(enriched: pd.DataFrame, config: MLPipelineConfig) -> pd.DataFrame:
    """Recover per-touch OOS predictions over the production walk-forward.

    Mirrors ml_training_tab.run_walk_forward_training: feature cols = int_*/app_*,
    label_encoded as y, WalkForwardSplitter on the timestamp, RFECV once (rolling),
    then per-fold CatBoost train/predict.  Returns the enriched frame restricted to
    OOS test rows with added columns: fold, raw_pred, prob_reversal, prob_trap,
    prob_blowthrough.  (Same purge buffer as production.)
    """
    feature_cols = [c for c in enriched.columns if c.startswith(("int_", "app_"))]
    label_column = "label_encoded"

    valid = enriched[enriched[label_column].notna()].copy()
    valid = valid.sort_values("event_ts").reset_index(drop=True)
    features = valid[feature_cols]
    y = valid[label_column].astype(int)
    timestamps = pd.to_datetime(valid["timestamp"])

    splitter = WalkForwardSplitter(config.walk_forward)
    splits = splitter.split(timestamps)
    if len(splits) < 2:
        raise ValueError(f"Only {len(splits)} fold(s); need >=2 (date range too short).")

    # RFECV once on the full feature set, exactly like the production pipeline.
    selected_features = feature_cols
    if config.model.rfecv_enabled:
        preliminary_cv = [
            (s.train_indices, s.test_indices)
            for s in splits
            if y.iloc[s.train_indices].nunique() >= 2
        ]
        if len(preliminary_cv) >= 2:
            rfecv_trainer = ExtremaModelTrainer(config.model)
            rfecv_res = rfecv_trainer.train(features, y, cv_splits=preliminary_cv)
            selected_features = rfecv_res.selected_features
            logger.info("RFECV selected %d/%d features: %s",
                        len(selected_features), len(feature_cols), selected_features)

    fold_model_config = config.model.model_copy(update={"rfecv_enabled": False})

    # Same label-purge buffer as production (forward_window // 500, floor 5 min).
    fw_minutes = max(5, config.labeling.forward_window // 500)
    purge_buffer = pd.Timedelta(minutes=fw_minutes)

    out_frames: list[pd.DataFrame] = []
    for split in splits:
        train_ts = timestamps.iloc[split.train_indices]
        safe_cutoff = split.test_start - purge_buffer
        purged_train_idx = split.train_indices[(train_ts <= safe_cutoff).values]

        x_train = features.iloc[purged_train_idx][selected_features]
        y_train = y.iloc[purged_train_idx]
        if y_train.nunique() < 2:
            logger.warning("Skipping fold %d: single-class train", split.fold)
            continue

        x_test = features.iloc[split.test_indices][selected_features]
        trainer = ExtremaModelTrainer(fold_model_config)
        fold_model = trainer.train(x_train, y_train)
        raw_pred = fold_model.model.predict(x_test).flatten().astype(int)
        proba = fold_model.model.predict_proba(x_test)

        # CatBoost class order == sorted unique training labels.
        classes = list(np.asarray(fold_model.model.classes_).astype(int))

        def _p(col_label: int, probs=proba, cls=classes):
            return probs[:, cls.index(col_label)] if col_label in cls else np.zeros(len(probs))

        test_rows = valid.iloc[split.test_indices].copy()
        test_rows["fold"] = split.fold
        test_rows["raw_pred"] = raw_pred
        test_rows["prob_reversal"] = _p(0)
        test_rows["prob_trap"] = _p(1)
        test_rows["prob_blowthrough"] = _p(2)
        out_frames.append(test_rows)

    if not out_frames:
        raise ValueError("No valid folds produced OOS predictions.")
    oos = pd.concat(out_frames, ignore_index=True)
    return oos.sort_values("event_ts").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 -- gate + metrics
# ════════════════════════════════════════════════════════════════════════════

def _idealized_metrics(oos: pd.DataFrame, tp_points: float, sl_points: float) -> dict:
    """Reproduce the EXACT 11.71 number.

    ml_training_tab builds the OOS confusion matrix from:
        eval_y    = (label_encoded == 0)   # actual tradeable_reversal
        eval_pred = (raw_pred      == 0)   # argmax predicted tradeable_reversal
    then compute_utility_metrics(tp, sl) treats tp (actual&pred reversal) as wins
    (+tp) and fp (pred reversal, not actual) as losses (-sl).  We rebuild the same
    (tp, fp) confusion matrix and call the SAME compute_utility_metrics.
    """
    eval_y = (oos["label_encoded"].astype(int) == 0).astype(int).values
    eval_pred = (oos["raw_pred"].astype(int) == 0).astype(int).values
    evaluator = ModelEvaluator(n_bootstrap=1, n_permutations=1)
    er = evaluator.evaluate(eval_y, eval_pred, None)
    util = compute_utility_metrics(er, tp_points=tp_points, sl_points=sl_points)
    return {
        "expectancy_pts": util["expectancy_pts"],
        "profit_factor": util["profit_factor"],
        "n_simulated_trades": util["n_simulated_trades"],
        "confusion": dict(er.confusion_matrix),
        "precision": er.precision,
    }


def _honest_block(trades: pd.DataFrame, commission_rt: float) -> dict:
    """Honest GROSS/NET metric block from a frame of traded touches.

    trades must have honest_gross_pts (signed, slippage-inclusive) and event_ts.
    """
    t = trades[trades["traded"] & trades["honest_gross_pts"].notna()].copy()
    t = t.sort_values("event_ts").reset_index(drop=True)
    n = len(t)
    if n == 0:
        return {"n": 0}

    gp = t["honest_gross_pts"].astype(float).values
    wins = gp[gp > 0]
    losses = gp[gp < 0]
    hit_rate = float(len(wins) / n)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0

    gross_exp_pts = float(gp.mean())
    gross_pnl_usd = float(gp.sum() * POINT_VALUE)

    commission_pts = commission_rt / POINT_VALUE  # express commission in points
    net_pts = gp - commission_pts                  # slippage already in gp
    net_exp_pts = float(net_pts.mean())
    net_exp_usd = float(net_exp_pts * POINT_VALUE)
    net_pnl_usd = float(net_pts.sum() * POINT_VALUE)

    gross_gain = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf_gross = float(gross_gain / gross_loss) if gross_loss > 0 else float("inf")
    net_pos = net_pts[net_pts > 0].sum()
    net_neg = -net_pts[net_pts < 0].sum()
    pf_net = float(net_pos / net_neg) if net_neg > 0 else float("inf")

    # Max drawdown on the NET equity curve (ordered by event_ts), in $.
    equity = np.cumsum(net_pts) * POINT_VALUE
    running_max = np.maximum.accumulate(equity)
    max_dd_usd = float((running_max - equity).max()) if n else 0.0

    reason_counts = t["honest_exit_reason"].value_counts().to_dict()

    return {
        "n": int(n),
        "hit_rate": hit_rate,
        "avg_win_pts": avg_win,
        "avg_loss_pts": avg_loss,
        "expectancy_gross_pts": gross_exp_pts,
        "expectancy_net_pts": net_exp_pts,
        "expectancy_net_usd": net_exp_usd,
        "profit_factor_gross": pf_gross,
        "profit_factor_net": pf_net,
        "total_gross_pnl_usd": gross_pnl_usd,
        "total_net_pnl_usd": net_pnl_usd,
        "max_drawdown_usd": max_dd_usd,
        "exit_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "commission_rt_usd": commission_rt,
        "commission_pts": commission_pts,
    }


def _per_fold_table(gated: pd.DataFrame, commission_rt: float) -> list[dict]:
    rows = []
    commission_pts = commission_rt / POINT_VALUE
    for fold, grp in gated.groupby("fold"):
        t = grp[grp["traded"] & grp["honest_gross_pts"].notna()]
        n = len(t)
        if n == 0:
            rows.append({"fold": int(fold), "n_gated_traded": 0})
            continue
        gp = t["honest_gross_pts"].astype(float).values
        rows.append({
            "fold": int(fold),
            "n_gated_traded": int(n),
            "hit_rate": float((gp > 0).mean()),
            "expectancy_gross_pts": float(gp.mean()),
            "expectancy_net_pts": float((gp - commission_pts).mean()),
        })
    return rows


def run_walk_forward_audit(enriched: pd.DataFrame, config: MLPipelineConfig,
                           cost_spec: dict) -> dict:
    """STEP 2 + STEP 3: reproduce OOS preds, then compute (a)/(b)/(c)/per-fold."""
    tp = config.dashboard_utility.tp_points
    sl = config.dashboard_utility.sl_points
    commission_rt = (cost_spec["exchange_nfa_per_side"]
                     + cost_spec["broker_commission_per_side"]) * 2.0

    oos = reproduce_walk_forward(enriched, config)
    # ET helper column for the session gate.
    ts_et = pd.to_datetime(oos["event_ts"])
    if getattr(ts_et.dt, "tz", None) is None:
        ts_et = ts_et.dt.tz_localize(_ET_STR)
    else:
        ts_et = ts_et.dt.tz_convert(_ET_STR)
    oos = oos.assign(_in_rth=[_in_ny_rth(x) for x in ts_et])

    # (a) IDEALIZED -- the EXACT 11.71 reproduction (argmax==reversal gate, all OOS).
    idealized = _idealized_metrics(oos, tp, sl)

    # PRODUCTION serve gate: prob_reversal>=0.70 AND ny_rth AND eligible_class.
    gated_mask = (
        (oos["prob_reversal"] >= CONF_GATE)
        & oos["_in_rth"]
    )
    gated = oos[gated_mask].copy()

    # (b) HONEST -- traded gated touches.
    honest = _honest_block(gated, commission_rt)

    # (c) BASELINE -- ALL eligible ny_rth OOS touches, no model gate.
    baseline_pop = oos[oos["_in_rth"]].copy()
    baseline = _honest_block(baseline_pop, commission_rt)

    per_fold = _per_fold_table(gated, commission_rt)

    return {
        "n_oos_touches": int(len(oos)),
        "n_oos_rth_touches": int(oos["_in_rth"].sum()),
        "n_gated": int(len(gated)),
        "n_gated_traded": int((gated["traded"] & gated["honest_gross_pts"].notna()).sum()),
        "idealized": idealized,
        "honest_gated": honest,
        "baseline_all_rth": baseline,
        "per_fold": per_fold,
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _print_summary(report: dict) -> None:
    a = report["idealized"]
    b = report["honest_gated"]
    c = report["baseline_all_rth"]
    print("\n================ HONEST EDGE AUDIT (terse) ================")
    print(f"OOS touches={report['n_oos_touches']}  rth={report['n_oos_rth_touches']}  "
          f"gated={report['n_gated']}  gated_traded={report['n_gated_traded']}")
    print(f"(a) IDEALIZED: expectancy={a['expectancy_pts']} pts  PF={a['profit_factor']}  "
          f"n={a['n_simulated_trades']}  cm={a['confusion']}")
    if b.get("n"):
        print(f"(b) HONEST gated: n={b['n']}  hit={b['hit_rate']:.3f}  "
              f"exp_gross={b['expectancy_gross_pts']:.3f}pt  "
              f"exp_net={b['expectancy_net_pts']:.3f}pt (${b['expectancy_net_usd']:.2f})  "
              f"PFnet={b['profit_factor_net']:.3f}  netPnL=${b['total_net_pnl_usd']:.0f}  "
              f"maxDD=${b['max_drawdown_usd']:.0f}")
    else:
        print("(b) HONEST gated: n=0")
    if c.get("n"):
        print(f"(c) BASELINE all-rth: n={c['n']}  hit={c['hit_rate']:.3f}  "
              f"exp_net={c['expectancy_net_pts']:.3f}pt  PFnet={c['profit_factor_net']:.3f}  "
              f"netPnL=${c['total_net_pnl_usd']:.0f}")
    else:
        print("(c) BASELINE all-rth: n=0")
    print("==========================================================\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Honest-edge audit of the book-mid reversal model.")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out-parquet", required=True, type=Path)
    p.add_argument("--results-json", type=Path, default=None)
    p.add_argument("--enrich-only", action="store_true",
                   help="STEP 1 only: build/append enriched parquet, no WF/metrics.")
    p.add_argument("--analyze-only", action="store_true",
                   help="STEP 2+3 only: read existing parquet, emit metrics.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = build_production_config()
    cost_spec = load_nq_cost_spec()
    commission_rt = (cost_spec["exchange_nfa_per_side"]
                     + cost_spec["broker_commission_per_side"]) * 2.0

    build_stats = None
    if not args.analyze_only:
        build_stats = build_enriched(args.start, args.end, args.out_parquet, config)
        logger.info("STEP 1 done: %s", build_stats)

    if args.enrich_only:
        if args.results_json:
            args.results_json.parent.mkdir(parents=True, exist_ok=True)
            with open(args.results_json, "w") as fh:
                json.dump({"step1": build_stats}, fh, indent=2, default=str)
        return 0

    enriched = pd.read_parquet(args.out_parquet)
    report = run_walk_forward_audit(enriched, config, cost_spec)
    report["range"] = {"start": args.start, "end": args.end}
    report["step1"] = build_stats
    report["cost_scheme"] = {
        "slippage_ticks_per_side": ENTRY_SLIPPAGE_TICKS,
        "slippage_pts_round_turn": (ENTRY_SLIPPAGE_TICKS + EXIT_SLIPPAGE_TICKS) * TICK_SIZE,
        "slippage_usd_round_turn": (ENTRY_SLIPPAGE_TICKS + EXIT_SLIPPAGE_TICKS) * TICK_SIZE * POINT_VALUE,
        "slippage_modeled_in_fills": True,
        "commission_rt_usd": commission_rt,
        "commission_formula": "(exchange_nfa 2.14 + broker 0.50) * 2 sides = 5.28",
        "point_value": POINT_VALUE,
        "tick_size": TICK_SIZE,
        "instruments_yaml_total_round_turn": cost_spec.get("total_round_turn"),
        "note": "slippage 1 tick/side in fills ($10 RT) + commission $5.28 = $15.28 adverse RT.",
    }
    report["assumptions"] = {
        "entry_price_source": "front-month book-mid (bid_px_00+ask_px_00)/2 most-recent <= touch_close+5m",
        "entry_slippage": "1 tick (0.25pt) adverse in entry fill",
        "exit_slippage": "1 tick (0.25pt) adverse in exit fill",
        "flatten_et": f"{FLATTEN_HOUR:02d}:{FLATTEN_MINUTE:02d} (no new trades at/after)",
        "cutoff_mark_et": f"{RTH_CUTOFF_HM[0]:02d}:{RTH_CUTOFF_HM[1]:02d} (forward-scan end; unresolved -> mark at last close)",
        "interaction_window_minutes": config.dashboard_utility.interaction_window_minutes,
        "bars": "book-mid 147t (price_source=book_mid; UNCHANGED)",
        "resolution": "MAE-first TP+15/SL-30 from REAL entry",
        "idealized_gate": "argmax predicted class == tradeable_reversal (raw_pred==0); matches ml_training_tab OOS confusion matrix",
        "production_serve_gate": "prob_reversal>=0.70 AND session==ny_rth AND eligible_class=tradeable_reversal",
        "walk_forward": "train=40 test=5 gap=2 ROLLING (expanding=False); RFECV once; CatBoost 800/depth4/MultiClass/Balanced/seed42",
    }

    if args.results_json:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Wrote results -> %s", args.results_json)

    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
