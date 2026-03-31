"""
Phase 4 — Order Flow Feature Engineering for Key Level Classifier.

Builds a feature matrix from raw MBP-10 tick data for each labeled touch event.
Uses DuckDB SQL push-down for approach window aggregation (avoids loading
~7.5M records per event into pandas) and pandas for the smaller interaction
window where level-relative computations are easier.

Two windows per event:
  - Approach: 90 min BEFORE event_ts (order flow leading into the level)
  - Interaction: 5 min AFTER event_ts inclusive (absorption behavior at level)

Temporal integrity: no data after event_ts + 5 min is ever accessed.

Output: data/experiment/feature_matrix.parquet
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.tick_store import TickStore
from alpha_lab.experiment.labeling import (
    AGGRESSIVE_BLOWTHROUGH,
    NO_RESOLUTION,
    TRADEABLE_REVERSAL,
    TRAP_REVERSAL,
    _classify_event_session,
    _primary_level_name,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

APPROACH_WINDOW_MINUTES = 90
INTERACTION_WINDOW_MINUTES = 5
LARGE_TRADE_THRESHOLD = 10  # contracts
LEVEL_PROXIMITY_PTS = 0.50  # ±0.50 pts for "at level"
SUB_WINDOW_MINUTES = 15  # last N min for trend features
CANCEL_BURST_SECONDS = 30

LABEL_ENCODING = {
    TRADEABLE_REVERSAL: 0,
    TRAP_REVERSAL: 1,
    AGGRESSIVE_BLOWTHROUGH: 2,
}

CAT_FEATURES = ["ctx_direction", "ctx_level_type", "ctx_session"]


# ── Static / Categorical Features ─────────────────────────────


def _extract_static_features(event: pd.Series | dict) -> dict:
    """Extract context/categorical features from event metadata.

    Returns 7 features: 3 string categoricals + 4 numeric.
    """
    event_ts = pd.Timestamp(event["event_ts"])
    level_names_json = event["level_names"]

    return {
        "ctx_direction": event["direction"],
        "ctx_level_type": _primary_level_name(level_names_json),
        "ctx_session": _classify_event_session(event_ts),
        "ctx_hour": event_ts.hour,
        "ctx_time_normalized": (event_ts.hour * 60 + event_ts.minute) / 1440.0,
        "ctx_day_of_week": event_ts.dayofweek,
        "ctx_approach_from_above": 1 if event["approach_direction"] == "from_above" else 0,
    }


# ── Approach Window (DuckDB SQL) ──────────────────────────────


def _query_approach_features(
    conn,
    union_sql: str,
    sym_filter: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    sub_window_minutes: int = SUB_WINDOW_MINUTES,
) -> dict[str, float]:
    """Compute 27 approach-window features via a single DuckDB aggregate query.

    Window: [start_utc, end_utc) — exclusive at end_utc (event timestamp).
    """
    sub_boundary = end_utc - pd.Timedelta(minutes=sub_window_minutes)

    sql = f"""
    WITH base AS (
        SELECT *,
               (CAST(bid_px_00 AS DOUBLE) + CAST(ask_px_00 AS DOUBLE)) / 2.0 AS mid
        FROM ({union_sql}) AS t
        WHERE ts_event >= $1 AND ts_event < $2
          AND bid_px_00 > 0 AND ask_px_00 > 0
          {sym_filter}
    ),
    trades AS (
        SELECT * FROM base WHERE action = 'T'
    ),
    -- Trade aggregates
    trade_agg AS (
        SELECT
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE side = 'A'), 0) AS buy_volume,
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE side = 'B'), 0) AS sell_volume,
            COALESCE(SUM(CAST(size AS DOUBLE)), 0) AS total_volume,
            COUNT(*) AS trade_count,
            COUNT(*) FILTER (WHERE size >= {LARGE_TRADE_THRESHOLD}) AS large_trade_count,
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE size >= {LARGE_TRADE_THRESHOLD}), 0) AS large_trade_volume,
            AVG(CAST(size AS DOUBLE)) AS avg_trade_size,
            APPROX_QUANTILE(CAST(size AS DOUBLE), 0.9) AS p90_trade_size,
            FIRST(price ORDER BY ts_event) AS first_price,
            LAST(price ORDER BY ts_event) AS last_price,
            MAX(price) AS max_price,
            MIN(price) AS min_price
        FROM trades
    ),
    -- Early/late sub-windows for trend
    early_trades AS (
        SELECT
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE side = 'A'), 0) AS early_buy,
            COALESCE(SUM(CAST(size AS DOUBLE)), 0) AS early_total,
            COALESCE(SUM(CAST(size AS DOUBLE)), 0) AS early_volume
        FROM trades
        WHERE ts_event < $3
    ),
    late_trades AS (
        SELECT
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE side = 'A'), 0) AS late_buy,
            COALESCE(SUM(CAST(size AS DOUBLE)), 0) AS late_total,
            COALESCE(SUM(CAST(size AS DOUBLE)), 0) AS late_volume,
            FIRST(price ORDER BY ts_event) AS late_first_price,
            LAST(price ORDER BY ts_event) AS late_last_price
        FROM trades
        WHERE ts_event >= $3
    ),
    -- Book aggregates
    book_agg AS (
        SELECT
            AVG(CAST(bid_sz_00 AS DOUBLE) / NULLIF(CAST(bid_sz_00 AS DOUBLE) + CAST(ask_sz_00 AS DOUBLE), 0))
                AS avg_tob_imbalance,
            AVG(CAST(bid_sz_00 AS DOUBLE) + CAST(bid_sz_01 AS DOUBLE) + CAST(bid_sz_02 AS DOUBLE)
                + CAST(bid_sz_03 AS DOUBLE) + CAST(bid_sz_04 AS DOUBLE)
                + CAST(ask_sz_00 AS DOUBLE) + CAST(ask_sz_01 AS DOUBLE) + CAST(ask_sz_02 AS DOUBLE)
                + CAST(ask_sz_03 AS DOUBLE) + CAST(ask_sz_04 AS DOUBLE))
                AS avg_top5_depth,
            AVG(CAST(ask_px_00 AS DOUBLE) - CAST(bid_px_00 AS DOUBLE))
                AS avg_spread,
            MAX(CAST(ask_px_00 AS DOUBLE) - CAST(bid_px_00 AS DOUBLE))
                AS max_spread,
            COUNT(*) FILTER (WHERE action = 'C') * 1.0
                / NULLIF(COUNT(*) FILTER (WHERE action != 'T'), 0)
                AS cancel_rate,
            AVG((CAST(bid_sz_00 AS DOUBLE) + CAST(bid_sz_01 AS DOUBLE) + CAST(bid_sz_02 AS DOUBLE)
                + CAST(bid_sz_03 AS DOUBLE) + CAST(bid_sz_04 AS DOUBLE))
                / NULLIF(
                    CAST(bid_sz_00 AS DOUBLE) + CAST(bid_sz_01 AS DOUBLE) + CAST(bid_sz_02 AS DOUBLE)
                    + CAST(bid_sz_03 AS DOUBLE) + CAST(bid_sz_04 AS DOUBLE)
                    + CAST(ask_sz_00 AS DOUBLE) + CAST(ask_sz_01 AS DOUBLE) + CAST(ask_sz_02 AS DOUBLE)
                    + CAST(ask_sz_03 AS DOUBLE) + CAST(ask_sz_04 AS DOUBLE), 0))
                AS avg_bid_depth_ratio,
            AVG((CAST(bid_sz_00 AS DOUBLE) + CAST(ask_sz_00 AS DOUBLE))
                / NULLIF(
                    CAST(bid_sz_00 AS DOUBLE) + CAST(bid_sz_01 AS DOUBLE) + CAST(bid_sz_02 AS DOUBLE)
                    + CAST(bid_sz_03 AS DOUBLE) + CAST(bid_sz_04 AS DOUBLE)
                    + CAST(ask_sz_00 AS DOUBLE) + CAST(ask_sz_01 AS DOUBLE) + CAST(ask_sz_02 AS DOUBLE)
                    + CAST(ask_sz_03 AS DOUBLE) + CAST(ask_sz_04 AS DOUBLE), 0))
                AS depth_concentration
        FROM base
    ),
    -- Book imbalance trend (early vs late)
    book_early AS (
        SELECT AVG(CAST(bid_sz_00 AS DOUBLE) / NULLIF(CAST(bid_sz_00 AS DOUBLE) + CAST(ask_sz_00 AS DOUBLE), 0))
            AS early_imbalance
        FROM base WHERE ts_event < $3
    ),
    book_late AS (
        SELECT AVG(CAST(bid_sz_00 AS DOUBLE) / NULLIF(CAST(bid_sz_00 AS DOUBLE) + CAST(ask_sz_00 AS DOUBLE), 0))
            AS late_imbalance
        FROM base WHERE ts_event >= $3
    ),
    -- Volatility: 5-min bucket returns (split window fn from aggregate)
    vol_buckets AS (
        SELECT
            time_bucket(INTERVAL '5 minutes', ts_event) AS bucket,
            LAST(price ORDER BY ts_event) AS close_price
        FROM trades
        GROUP BY bucket
    ),
    vol_returns AS (
        SELECT
            (close_price - LAG(close_price) OVER (ORDER BY bucket))
            / NULLIF(LAG(close_price) OVER (ORDER BY bucket), 0) AS ret
        FROM vol_buckets
    ),
    vol_full AS (
        SELECT STDDEV_SAMP(ret) AS volatility_full FROM vol_returns WHERE ret IS NOT NULL
    ),
    -- Volatility: 1-min bucket returns over last sub_window
    vol_recent_buckets AS (
        SELECT
            time_bucket(INTERVAL '1 minute', ts_event) AS bucket,
            LAST(price ORDER BY ts_event) AS close_price
        FROM trades
        WHERE ts_event >= $3
        GROUP BY bucket
    ),
    vol_recent_returns AS (
        SELECT
            (close_price - LAG(close_price) OVER (ORDER BY bucket))
            / NULLIF(LAG(close_price) OVER (ORDER BY bucket), 0) AS ret
        FROM vol_recent_buckets
    ),
    vol_recent AS (
        SELECT STDDEV_SAMP(ret) AS volatility_recent FROM vol_recent_returns WHERE ret IS NOT NULL
    ),
    -- Tick direction bias (split window fn from aggregate)
    trade_prices AS (
        SELECT price, LAG(price) OVER (ORDER BY ts_event) AS prev_price
        FROM trades
    ),
    tick_dirs AS (
        SELECT
            SUM(CASE WHEN price > prev_price THEN 1 ELSE 0 END) AS upticks,
            SUM(CASE WHEN price < prev_price THEN 1 ELSE 0 END) AS downticks
        FROM trade_prices
        WHERE prev_price IS NOT NULL
    )
    SELECT
        t.buy_volume, t.sell_volume, t.total_volume, t.trade_count,
        t.large_trade_count, t.large_trade_volume,
        t.avg_trade_size, t.p90_trade_size,
        t.first_price, t.last_price, t.max_price, t.min_price,
        e.early_buy, e.early_total, e.early_volume,
        l.late_buy, l.late_total, l.late_volume,
        l.late_first_price, l.late_last_price,
        b.avg_tob_imbalance, b.avg_top5_depth, b.avg_spread, b.max_spread,
        b.cancel_rate, b.avg_bid_depth_ratio, b.depth_concentration,
        be.early_imbalance, bl.late_imbalance,
        vf.volatility_full, vr.volatility_recent,
        td.upticks, td.downticks
    FROM trade_agg t, early_trades e, late_trades l, book_agg b,
         book_early be, book_late bl, vol_full vf, vol_recent vr, tick_dirs td
    """

    try:
        row = conn.execute(sql, [start_utc, end_utc, sub_boundary]).fetchone()
    except Exception as exc:
        logger.warning("Approach SQL failed for window %s - %s: %s", start_utc, end_utc, exc)
        return _empty_approach_features()

    if row is None:
        return _empty_approach_features()

    # Unpack row into named values
    (
        buy_vol, sell_vol, total_vol, trade_count,
        large_count, large_vol,
        avg_size, p90_size,
        first_price, last_price, max_price, min_price,
        early_buy, early_total, early_volume,
        late_buy, late_total, late_volume,
        late_first_price, late_last_price,
        avg_tob_imb, avg_top5, avg_spread, max_spread,
        cancel_rate, bid_depth_ratio, depth_conc,
        early_imb, late_imb,
        vol_full, vol_recent,
        upticks, downticks,
    ) = row

    # Derived features
    buy_sell_total = _safe_float(buy_vol) + _safe_float(sell_vol)
    buy_sell_ratio = _safe_float(buy_vol) / buy_sell_total if buy_sell_total > 0 else float("nan")
    large_vol_pct = _safe_float(large_vol) / _safe_float(total_vol) if _safe_float(total_vol) > 0 else float("nan")

    early_ratio = _safe_float(early_buy) / _safe_float(early_total) if _safe_float(early_total) > 0 else float("nan")
    late_ratio = _safe_float(late_buy) / _safe_float(late_total) if _safe_float(late_total) > 0 else float("nan")
    aggression_trend = (late_ratio - early_ratio) if not (math.isnan(late_ratio) or math.isnan(early_ratio)) else float("nan")

    early_vol = _safe_float(early_volume)
    late_vol_val = _safe_float(late_volume)
    # volume_acceleration: (late_vol/sub_window) / (early_vol/(90-sub_window))
    early_rate = early_vol / (APPROACH_WINDOW_MINUTES - SUB_WINDOW_MINUTES) if early_vol > 0 else 0
    late_rate = late_vol_val / SUB_WINDOW_MINUTES if late_vol_val > 0 else 0
    vol_accel = late_rate / early_rate if early_rate > 0 else float("nan")

    fp = _safe_float(first_price)
    lp = _safe_float(last_price)
    price_change = lp - fp if not (math.isnan(fp) or math.isnan(lp)) else float("nan")
    price_change_pct = price_change / fp if fp != 0 and not math.isnan(price_change) else float("nan")

    lfp = _safe_float(late_first_price)
    llp = _safe_float(late_last_price)
    price_vel_15m = llp - lfp if not (math.isnan(lfp) or math.isnan(llp)) else float("nan")

    mx = _safe_float(max_price)
    mn = _safe_float(min_price)
    price_range = mx - mn if not (math.isnan(mx) or math.isnan(mn)) else float("nan")

    up = _safe_float(upticks)
    dn = _safe_float(downticks)
    total_ticks = up + dn
    tick_bias = (up - dn) / total_ticks if total_ticks > 0 else float("nan")

    vol_f = _safe_float(vol_full)
    vol_r = _safe_float(vol_recent)
    vol_ratio = vol_r / vol_f if vol_f > 0 and not math.isnan(vol_f) else float("nan")

    imb_trend = _safe_float(late_imb) - _safe_float(early_imb) if not (
        math.isnan(_safe_float(late_imb)) or math.isnan(_safe_float(early_imb))
    ) else float("nan")

    return {
        "app_buy_volume": _safe_float(buy_vol),
        "app_sell_volume": _safe_float(sell_vol),
        "app_buy_sell_ratio": buy_sell_ratio,
        "app_large_trade_count": _safe_float(large_count),
        "app_large_trade_vol_pct": large_vol_pct,
        "app_aggression_trend": aggression_trend,
        "app_total_trade_volume": _safe_float(total_vol),
        "app_trade_count": _safe_float(trade_count),
        "app_volume_acceleration": vol_accel,
        "app_avg_trade_size": _safe_float(avg_size),
        "app_p90_trade_size": _safe_float(p90_size),
        "app_avg_tob_imbalance": _safe_float(avg_tob_imb),
        "app_avg_top5_depth": _safe_float(avg_top5),
        "app_avg_spread": _safe_float(avg_spread),
        "app_max_spread": _safe_float(max_spread),
        "app_cancel_rate": _safe_float(cancel_rate),
        "app_book_imbalance_trend": imb_trend,
        "app_avg_bid_depth_ratio": _safe_float(bid_depth_ratio),
        "app_depth_concentration": _safe_float(depth_conc),
        "app_price_change": price_change,
        "app_price_change_pct": price_change_pct,
        "app_tick_direction_bias": tick_bias,
        "app_price_velocity_15m": price_vel_15m,
        "app_price_range": price_range,
        "app_volatility_full": vol_f,
        "app_volatility_recent": vol_r,
        "app_volatility_ratio": vol_ratio,
    }


def _empty_approach_features() -> dict[str, float]:
    """Return NaN-filled approach features dict."""
    keys = [
        "app_buy_volume", "app_sell_volume", "app_buy_sell_ratio",
        "app_large_trade_count", "app_large_trade_vol_pct", "app_aggression_trend",
        "app_total_trade_volume", "app_trade_count", "app_volume_acceleration",
        "app_avg_trade_size", "app_p90_trade_size",
        "app_avg_tob_imbalance", "app_avg_top5_depth", "app_avg_spread",
        "app_max_spread", "app_cancel_rate", "app_book_imbalance_trend",
        "app_avg_bid_depth_ratio", "app_depth_concentration",
        "app_price_change", "app_price_change_pct", "app_tick_direction_bias",
        "app_price_velocity_15m", "app_price_range",
        "app_volatility_full", "app_volatility_recent", "app_volatility_ratio",
    ]
    return {k: float("nan") for k in keys}


# ── Interaction Window ─────────────────────────────────────────


def _query_interaction_features(
    conn,
    union_sql: str,
    sym_filter: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    representative_price: float,
    direction: str,
) -> dict[str, float]:
    """Compute 18 interaction-window features via DuckDB SQL + light pandas.

    Window: [start_utc, end_utc] — inclusive of touch bar.

    Level-relative features (absorption, sweep, tempo) use representative_price
    and direction to determine what counts as "at level" vs "through level".
    """
    level_low = representative_price - LEVEL_PROXIMITY_PTS
    level_high = representative_price + LEVEL_PROXIMITY_PTS
    burst_boundary = start_utc + pd.Timedelta(seconds=CANCEL_BURST_SECONDS)

    # For LONG: adverse = price below level; for SHORT: adverse = price above level
    if direction == "LONG":
        adverse_condition = f"price < {representative_price}"
    else:
        adverse_condition = f"price > {representative_price}"

    sql = f"""
    WITH base AS (
        SELECT *,
               (CAST(bid_px_00 AS DOUBLE) + CAST(ask_px_00 AS DOUBLE)) / 2.0 AS mid
        FROM ({union_sql}) AS t
        WHERE ts_event >= $1 AND ts_event <= $2
          AND bid_px_00 > 0 AND ask_px_00 > 0
          {sym_filter}
    ),
    trades AS (
        SELECT * FROM base WHERE action = 'T'
    ),
    trade_agg AS (
        SELECT
            COALESCE(SUM(CAST(size AS DOUBLE)), 0) AS total_volume,
            COUNT(*) AS trade_count,
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE side = 'A'), 0) AS buy_volume,
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE side = 'B'), 0) AS sell_volume,
            COUNT(*) FILTER (WHERE size >= {LARGE_TRADE_THRESHOLD}) AS large_count,
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE size >= {LARGE_TRADE_THRESHOLD}), 0) AS large_volume,
            MAX(CAST(size AS DOUBLE)) AS max_trade_size,
            AVG(CAST(size AS DOUBLE)) AS avg_trade_size,
            APPROX_QUANTILE(CAST(size AS DOUBLE), 0.9) AS p90_trade_size,
            -- Volume at level (within ±0.50 pts)
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE price >= {level_low} AND price <= {level_high}), 0)
                AS volume_at_level,
            -- Volume through level (adverse direction)
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE {adverse_condition}), 0)
                AS volume_through_level,
            -- Sweep volume: trades that move price through the level adversely
            COALESCE(SUM(CAST(size AS DOUBLE)) FILTER (WHERE {adverse_condition} AND size >= {LARGE_TRADE_THRESHOLD}), 0)
                AS sweep_volume,
            FIRST(price ORDER BY ts_event) AS first_price,
            LAST(price ORDER BY ts_event) AS last_price
        FROM trades
    ),
    book_agg AS (
        SELECT
            AVG(CAST(bid_sz_00 AS DOUBLE) / NULLIF(CAST(bid_sz_00 AS DOUBLE) + CAST(ask_sz_00 AS DOUBLE), 0))
                AS avg_tob_imbalance,
            AVG(CAST(ask_px_00 AS DOUBLE) - CAST(bid_px_00 AS DOUBLE))
                AS avg_spread,
            AVG(CAST(bid_sz_00 AS DOUBLE) + CAST(bid_sz_01 AS DOUBLE) + CAST(bid_sz_02 AS DOUBLE)
                + CAST(bid_sz_03 AS DOUBLE) + CAST(bid_sz_04 AS DOUBLE)
                + CAST(ask_sz_00 AS DOUBLE) + CAST(ask_sz_01 AS DOUBLE) + CAST(ask_sz_02 AS DOUBLE)
                + CAST(ask_sz_03 AS DOUBLE) + CAST(ask_sz_04 AS DOUBLE))
                AS avg_depth
        FROM base
    ),
    cancel_agg AS (
        SELECT
            COUNT(*) FILTER (WHERE action = 'C' AND ts_event < $3) AS cancel_burst,
            COUNT(*) FILTER (WHERE action = 'C') AS cancel_total
        FROM base
    ),
    -- Tempo features: time within 2 pts and time beyond level
    -- Use mid-price (book-based) with nanosecond deltas
    tempo_data AS (
        SELECT
            ts_event,
            mid,
            LEAD(ts_event) OVER (ORDER BY ts_event) AS next_ts,
            CASE WHEN ABS(mid - {representative_price}) <= 2.0 THEN 1 ELSE 0 END AS within_2pts,
            CASE WHEN {'mid < ' + str(representative_price) if direction == 'LONG' else 'mid > ' + str(representative_price)}
                THEN 1 ELSE 0 END AS beyond_level
        FROM base
    ),
    tempo_agg AS (
        SELECT
            SUM(CASE WHEN within_2pts = 1 AND next_ts IS NOT NULL
                THEN EPOCH(next_ts - ts_event) ELSE 0 END) AS time_within_2pts,
            SUM(CASE WHEN beyond_level = 1 AND next_ts IS NOT NULL
                THEN EPOCH(next_ts - ts_event) ELSE 0 END) AS time_beyond_level
        FROM tempo_data
    )
    SELECT
        ta.total_volume, ta.trade_count, ta.buy_volume, ta.sell_volume,
        ta.large_count, ta.large_volume, ta.max_trade_size,
        ta.avg_trade_size, ta.p90_trade_size,
        ta.volume_at_level, ta.volume_through_level, ta.sweep_volume,
        ta.first_price, ta.last_price,
        ba.avg_tob_imbalance, ba.avg_spread, ba.avg_depth,
        ca.cancel_burst, ca.cancel_total,
        te.time_within_2pts, te.time_beyond_level
    FROM trade_agg ta, book_agg ba, cancel_agg ca, tempo_agg te
    """

    try:
        row = conn.execute(sql, [start_utc, end_utc, burst_boundary]).fetchone()
    except Exception as exc:
        logger.warning("Interaction SQL failed for window %s - %s: %s", start_utc, end_utc, exc)
        return _empty_interaction_features()

    if row is None:
        return _empty_interaction_features()

    (
        total_vol, trade_count, buy_vol, sell_vol,
        large_count, large_vol, max_size,
        avg_size, p90_size,
        vol_at_level, vol_through, sweep_vol,
        first_price, last_price,
        avg_tob_imb, avg_spread, avg_depth,
        cancel_burst, cancel_total,
        time_within, time_beyond,
    ) = row

    total_v = _safe_float(total_vol)
    buy_v = _safe_float(buy_vol)
    sell_v = _safe_float(sell_vol)
    buy_sell_total = buy_v + sell_v
    buy_sell_ratio = buy_v / buy_sell_total if buy_sell_total > 0 else float("nan")

    large_v = _safe_float(large_vol)
    large_pct = large_v / total_v if total_v > 0 else float("nan")

    vol_at = _safe_float(vol_at_level)
    vol_thru = _safe_float(vol_through)
    absorption = vol_at / (vol_at + vol_thru) if (vol_at + vol_thru) > 0 else float("nan")

    cb = _safe_float(cancel_burst)
    ct = _safe_float(cancel_total)
    cancel_burst_ratio = cb / ct if ct > 0 else float("nan")

    fp = _safe_float(first_price)
    lp = _safe_float(last_price)
    int_displacement = abs(lp - fp) if not (math.isnan(fp) or math.isnan(lp)) else float("nan")

    return {
        "int_total_trade_volume": total_v,
        "int_trade_count": _safe_float(trade_count),
        "int_volume_at_level": vol_at,
        "int_volume_through_level": vol_thru,
        "int_absorption_ratio": absorption,
        "int_buy_sell_ratio": buy_sell_ratio,
        "int_large_trade_count": _safe_float(large_count),
        "int_large_trade_pct": large_pct,
        "int_max_trade_size": _safe_float(max_size),
        "int_sweep_volume": _safe_float(sweep_vol),
        "int_cancel_burst": cancel_burst_ratio,
        "int_avg_trade_size": _safe_float(avg_size),
        "int_p90_trade_size": _safe_float(p90_size),
        "int_avg_tob_imbalance": _safe_float(avg_tob_imb),
        "int_avg_spread": _safe_float(avg_spread),
        "int_avg_depth": _safe_float(avg_depth),
        "int_time_within_2pts": _safe_float(time_within),
        "int_time_beyond_level": _safe_float(time_beyond),
        # Store displacement for deceleration_ratio computation in cross-window
        "_int_displacement": int_displacement,
        "_int_first_price": fp,
        "_int_last_price": lp,
    }


def _empty_interaction_features() -> dict[str, float]:
    """Return NaN-filled interaction features dict."""
    keys = [
        "int_total_trade_volume", "int_trade_count",
        "int_volume_at_level", "int_volume_through_level", "int_absorption_ratio",
        "int_buy_sell_ratio",
        "int_large_trade_count", "int_large_trade_pct",
        "int_max_trade_size", "int_sweep_volume", "int_cancel_burst",
        "int_avg_trade_size", "int_p90_trade_size",
        "int_avg_tob_imbalance", "int_avg_spread", "int_avg_depth",
        "int_time_within_2pts", "int_time_beyond_level",
        "_int_displacement", "_int_first_price", "_int_last_price",
    ]
    return {k: float("nan") for k in keys}


# ── Cross-Window Features ─────────────────────────────────────


def _compute_cross_window_features(
    approach: dict[str, float],
    interaction: dict[str, float],
) -> dict[str, float]:
    """Compute 6 features that compare approach vs interaction windows."""
    # Aggression flip
    app_bsr = approach.get("app_buy_sell_ratio", float("nan"))
    int_bsr = interaction.get("int_buy_sell_ratio", float("nan"))
    aggression_flip = _safe_sub(int_bsr, app_bsr)

    # Book imbalance shift
    app_imb = approach.get("app_avg_tob_imbalance", float("nan"))
    int_imb = interaction.get("int_avg_tob_imbalance", float("nan"))
    book_shift = _safe_sub(int_imb, app_imb)

    # Spread widening
    app_spread = approach.get("app_avg_spread", float("nan"))
    int_spread = interaction.get("int_avg_spread", float("nan"))
    spread_wide = _safe_sub(int_spread, app_spread)

    # Depth change ratio
    app_depth = approach.get("app_avg_top5_depth", float("nan"))
    int_depth = interaction.get("int_avg_depth", float("nan"))
    depth_change = _safe_div(int_depth, app_depth)

    # Size comparison
    app_size = approach.get("app_avg_trade_size", float("nan"))
    int_size = interaction.get("int_avg_trade_size", float("nan"))
    size_vs = _safe_div(int_size, app_size)

    # Deceleration ratio: approach displacement rate / interaction displacement rate
    app_price_change = approach.get("app_price_change", float("nan"))
    int_disp = interaction.get("_int_displacement", float("nan"))
    app_rate = abs(app_price_change) / APPROACH_WINDOW_MINUTES if not math.isnan(app_price_change) else float("nan")
    int_rate = int_disp / INTERACTION_WINDOW_MINUTES if not math.isnan(int_disp) else float("nan")
    decel = _safe_div(app_rate, int_rate)

    return {
        "int_aggression_flip": aggression_flip,
        "int_book_imbalance_shift": book_shift,
        "int_spread_widening": spread_wide,
        "int_depth_change": depth_change,
        "int_size_vs_approach": size_vs,
        "int_deceleration_ratio": decel,
    }


# ── Utilities ──────────────────────────────────────────────────


def _safe_float(val) -> float:
    """Convert a DuckDB result value to float, handling None."""
    if val is None:
        return float("nan")
    try:
        f = float(val)
        return f
    except (TypeError, ValueError):
        return float("nan")


def _safe_sub(a: float, b: float) -> float:
    """a - b, returning NaN if either is NaN."""
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return a - b


def _safe_div(num: float, den: float) -> float:
    """num / den, returning NaN if den is 0 or either is NaN."""
    if math.isnan(num) or math.isnan(den) or den == 0:
        return float("nan")
    return num / den


# ── Main Builder ───────────────────────────────────────────────


class EventFeatureBuilder:
    """Builds feature matrix from labeled events using DuckDB-backed TickStore."""

    def __init__(self, data_dir: Path = Path("data/databento")) -> None:
        self._data_dir = data_dir

    def build_feature_matrix(
        self,
        labeled_events: pd.DataFrame,
        output_path: Path | None = Path("data/experiment/feature_matrix.parquet"),
        progress_fn: object = None,
    ) -> pd.DataFrame:
        """Build the complete feature matrix for all resolved events.

        Groups events by trading date and processes each group with a
        dedicated TickStore that only registers the 2-3 calendar dates
        needed for that group's approach + interaction windows. This avoids
        UNION ALL column mismatches across distant dates.

        Args:
            labeled_events: DataFrame from labeled_events.parquet (Phase 3).
            output_path: Where to write the feature matrix.
            progress_fn: Optional callback(pct: float, msg: str).

        Returns:
            DataFrame with 58 features + metadata columns per event.
        """
        # Filter out no_resolution
        resolved = labeled_events[labeled_events["label"] != NO_RESOLUTION].copy()
        n = len(resolved)
        logger.info("Building features for %d resolved events (excluded %d no_resolution)",
                     n, len(labeled_events) - n)

        feature_rows: list[dict] = []
        nan_event_indices: list[int] = []
        processed = 0

        # Group by trading date to share TickStore per group
        for date_str, group in resolved.groupby("date"):
            # Determine calendar dates needed for this group
            cal_dates = self._calendar_dates_for_group(group)

            # Create a fresh TickStore for this date group
            store = TickStore(self._data_dir)
            n_reg = 0
            for d in cal_dates:
                if store.register_symbol_date("NQ", d):
                    n_reg += 1

            if n_reg == 0:
                logger.warning("No tick data for date group %s, all features NaN", date_str)
                for _, event in group.iterrows():
                    features = {**_extract_static_features(event),
                                **_empty_approach_features(),
                                **_empty_interaction_features(),
                                **_compute_cross_window_features(
                                    _empty_approach_features(), _empty_interaction_features())}
                    feature_rows.append(features)
                    nan_event_indices.append(processed)
                    processed += 1
                store.close()
                continue

            # Get query context for this date group
            conn, union_sql, sym_filter = self._get_query_context(store)

            for _, event in group.iterrows():
                if progress_fn is not None:
                    progress_fn(processed / n, f"Extracting features {processed + 1}/{n}")

                features = self._extract_event_features(event, conn, union_sql, sym_filter)

                # Check for all-NaN numeric features
                numeric_vals = [v for k, v in features.items()
                               if isinstance(v, float) and not k.startswith("_")]
                if numeric_vals and all(math.isnan(v) for v in numeric_vals):
                    nan_event_indices.append(processed)
                    logger.warning(
                        "All-NaN features for event %d: date=%s zone=%s",
                        processed, event["date"], event["zone_id"],
                    )

                feature_rows.append(features)
                processed += 1

            store.close()

        if progress_fn is not None:
            progress_fn(1.0, "Done")

        # Check for systematic data access problem
        nan_pct = len(nan_event_indices) / n if n > 0 else 0
        if len(nan_event_indices) > 0:
            logger.warning(
                "%d/%d events (%.1f%%) have all-NaN features",
                len(nan_event_indices), n, 100 * nan_pct,
            )
        if len(nan_event_indices) >= 28:
            raise RuntimeError(
                f"Systematic data access problem: {len(nan_event_indices)}/{n} events "
                f"({100*nan_pct:.1f}%) have all-NaN features. Check TickStore registration."
            )

        # Build DataFrame
        feature_df = pd.DataFrame(feature_rows)

        # Drop internal columns (prefixed with _)
        internal_cols = [c for c in feature_df.columns if c.startswith("_")]
        feature_df = feature_df.drop(columns=internal_cols)

        # Add metadata
        feature_df["event_ts"] = resolved["event_ts"].values
        feature_df["date"] = resolved["date"].values
        feature_df["label"] = resolved["label"].values
        feature_df["label_encoded"] = resolved["label"].map(LABEL_ENCODING).values

        # Reorder: metadata first, then features
        meta_cols = ["event_ts", "date", "label", "label_encoded"]
        feature_cols = [c for c in feature_df.columns if c not in meta_cols]
        feature_df = feature_df[meta_cols + sorted(feature_cols)]

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            feature_df.to_parquet(output_path, index=False)
            logger.info("Wrote feature matrix to %s", output_path)

        # Print summary
        _print_feature_summary(feature_df, feature_cols)

        return feature_df

    def _calendar_dates_for_group(self, group: pd.DataFrame) -> list[str]:
        """Determine which calendar dates' Parquet files are needed.

        For a group of events on one trading date, we need the trading date
        itself plus ±1 day to cover cross-midnight approach windows.
        """
        from datetime import date as date_cls, timedelta as td

        dates_needed: set[str] = set()
        for _, event in group.iterrows():
            event_ts = pd.Timestamp(event["event_ts"])
            event_ts_utc = event_ts.tz_convert("UTC")
            approach_start = event_ts_utc - pd.Timedelta(minutes=APPROACH_WINDOW_MINUTES)
            interaction_end = event_ts_utc + pd.Timedelta(minutes=INTERACTION_WINDOW_MINUTES)

            # Calendar dates covered by these timestamps
            d_start = approach_start.date()
            d_end = interaction_end.date()
            current = d_start
            while current <= d_end:
                dates_needed.add(current.isoformat())
                current += td(days=1)

        return sorted(dates_needed)

    def _get_query_context(self, store: TickStore) -> tuple:
        """Extract DuckDB connection, union SQL, and front-month filter.

        Accesses TickStore internals for SQL push-down.
        """
        conn = store._conn
        views = store._get_views("NQ")
        if not views:
            raise RuntimeError("No NQ views registered — check data directory")
        union_sql = store._union_views_sql(views)

        # Detect symbol column and determine front-month filter
        sample_sql = (
            f"SELECT column_name FROM "
            f"(DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        )
        cols = {r[0] for r in conn.execute(sample_sql).fetchall()}
        has_symbol = "symbol" in cols

        if has_symbol:
            front = conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            sym_filter = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
        else:
            sym_filter = ""

        return conn, union_sql, sym_filter

    def _extract_event_features(
        self,
        event: pd.Series,
        conn,
        union_sql: str,
        sym_filter: str,
    ) -> dict:
        """Extract all features for a single event."""
        event_ts = pd.Timestamp(event["event_ts"])
        event_ts_utc = event_ts.tz_convert("UTC")

        approach_start = event_ts_utc - pd.Timedelta(minutes=APPROACH_WINDOW_MINUTES)
        interaction_end = event_ts_utc + pd.Timedelta(minutes=INTERACTION_WINDOW_MINUTES)

        # 1. Static features
        static = _extract_static_features(event)

        # 2. Approach window
        approach = _query_approach_features(
            conn, union_sql, sym_filter, approach_start, event_ts_utc,
        )

        # 3. Interaction window
        interaction = _query_interaction_features(
            conn, union_sql, sym_filter,
            event_ts_utc, interaction_end,
            event["representative_price"], event["direction"],
        )

        # 4. Cross-window
        cross = _compute_cross_window_features(approach, interaction)

        return {**static, **approach, **interaction, **cross}


def _print_feature_summary(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Print feature matrix summary."""
    print(f"\n{'=' * 70}")
    print("  PHASE 4 FEATURE ENGINEERING — SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Feature matrix shape: {df.shape}")
    print(f"  Features: {len(feature_cols)}")

    print(f"\n  Label distribution:")
    for label, cnt in df["label"].value_counts().sort_index().items():
        print(f"    {label:30s}  {cnt:4d}  ({100*cnt/len(df):.1f}%)")

    # NaN counts
    print(f"\n  NaN counts per feature (>0 only):")
    for col in sorted(feature_cols):
        if col in df.columns:
            nan_count = df[col].isna().sum() if df[col].dtype != object else 0
            if nan_count > 0:
                pct = 100 * nan_count / len(df)
                flag = " *** HIGH" if pct > 50 else ""
                print(f"    {col:40s}  {nan_count:4d}  ({pct:.1f}%){flag}")

    cat_cols = [c for c in feature_cols if c.startswith("ctx_") and df[c].dtype == object]
    if cat_cols:
        print(f"\n  Categorical features ({len(cat_cols)}):")
        for col in cat_cols:
            vals = df[col].unique()
            print(f"    {col}: {sorted(vals)}")

    print(f"\n  Sample rows (first 3):")
    sample_cols = ["label"] + feature_cols[:10]
    valid_cols = [c for c in sample_cols if c in df.columns]
    print(df[valid_cols].head(3).to_string())
    print()
