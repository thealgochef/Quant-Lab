"""
Self-contained dashboard-utility dataset builder.

Builds labeled feature matrices for the dashboard 3-class model directly
from raw tick data — no dependency on pre-built experiment artifacts.

Pipeline per date:
  1. Build bars (tick-count or 1m time bars) from tick data
  2. Compute key levels from prior-date session highs/lows
  3. Detect touch events (bar range intersects level, first-touch per zone)
  4. Label with configurable TP/SL (MAE-first, conservative)
  5. Compute 3 interaction features from raw ticks
  6. Optionally compute 27 approach features from pre-touch order flow

All parameters (bar_type, interaction window, approach window, TP/SL)
are configurable via DashboardUtilityConfig and included in the cache hash.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig, MLPipelineConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import (
    NO_RESOLUTION,
    label_touch_event,
)
from alpha_lab.agents.data_infra.tick_store import TickStore

logger = logging.getLogger(__name__)

# The 3 canonical dashboard features
DASHBOARD_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
]

# Session boundaries (Eastern Time)
_ET = "US/Eastern"
_ASIA_START = time(18, 0)
_ASIA_END = time(1, 0)
_LONDON_START = time(1, 0)
_LONDON_END = time(8, 0)
_NY_RTH_START = time(9, 30)
_NY_RTH_END = time(16, 15)

# Zone merge threshold (NQ points)
_ZONE_PROXIMITY = 3.0


def build_utility_dataset(
    dates: list[str],
    data_dir: Path,
    config: MLPipelineConfig,
    progress_fn=None,
) -> pd.DataFrame:
    """Build a labeled feature dataset for dashboard-utility training.

    Self-contained: builds bars, detects levels, finds touches, labels,
    and computes features directly from raw tick parquet files.

    Args:
        dates: List of date strings to process (e.g. ["2025-06-02", ...]).
        data_dir: Root databento data directory.
        config: Pipeline config (uses dashboard_utility sub-config).
        progress_fn: Optional callable(fraction, text) for progress.

    Returns:
        DataFrame with one row per labeled touch event.
    """
    util_cfg = config.dashboard_utility
    symbol = config.instrument
    cache_tag = config.dataset_config_hash()

    frames: list[pd.DataFrame] = []
    cached_count = 0

    # Track prior-day session highs/lows for PDH/PDL computation
    prev_ny_hl: tuple[float, float] | None = None
    prev_asia_hl: tuple[float, float] | None = None
    prev_london_hl: tuple[float, float] | None = None

    for i, date_str in enumerate(sorted(dates)):
        if progress_fn:
            progress_fn(i / len(dates), f"Processing {date_str} ({i + 1}/{len(dates)})...")

        cache_path = data_dir / symbol / date_str / f"ml_utility_{cache_tag}.parquet"

        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            cached_count += 1
            if not df.empty:
                frames.append(df)
            # Still need to compute session H/L for next day's levels
            _update_session_levels_from_cache(
                df, prev_ny_hl, prev_asia_hl, prev_london_hl,
            )
            # Read session levels from bars if we need them for next day
            prev_ny_hl, prev_asia_hl, prev_london_hl = _get_session_hl_for_date(
                data_dir, symbol, date_str, util_cfg,
                prev_ny_hl, prev_asia_hl, prev_london_hl,
            )
            continue

        # Build fresh for this date
        df = _process_single_date(
            date_str, data_dir, symbol, util_cfg,
            prev_ny_hl, prev_asia_hl, prev_london_hl,
        )

        # Update session levels for next day
        prev_ny_hl, prev_asia_hl, prev_london_hl = _get_session_hl_for_date(
            data_dir, symbol, date_str, util_cfg,
            prev_ny_hl, prev_asia_hl, prev_london_hl,
        )

        if not df.empty:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            frames.append(df)

    if progress_fn:
        progress_fn(1.0, "Done.")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)

    # Strip any non-live-computable approach features (may exist in old caches)
    from alpha_lab.agents.data_infra.ml.config import LIVE_APPROACH_FEATURES, LIVE_INTERACTION_FEATURES
    live_features = set(LIVE_INTERACTION_FEATURES + LIVE_APPROACH_FEATURES)
    drop_cols = [c for c in result.columns
                 if (c.startswith("app_") or c.startswith("int_"))
                 and c not in live_features]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    logger.info(
        "Utility dataset: %d labeled events from %d dates (%d cached)",
        len(result), len(dates), cached_count,
    )
    return result


def _get_session_hl_for_date(
    data_dir: Path,
    symbol: str,
    date_str: str,
    util_cfg: DashboardUtilityConfig,
    prev_ny_hl, prev_asia_hl, prev_london_hl,
):
    """Compute session highs/lows for a date (for next day's levels)."""
    bars = _build_bars_for_date(data_dir, symbol, date_str, util_cfg)
    if bars.empty:
        return prev_ny_hl, prev_asia_hl, prev_london_hl

    bars_et = _ensure_et_index(bars)
    ny = _slice_session(bars_et, "ny_rth")
    asia = _slice_session(bars_et, "asia")
    london = _slice_session(bars_et, "london")

    new_ny = _session_hl(ny) if not ny.empty else prev_ny_hl
    new_asia = _session_hl(asia) if not asia.empty else prev_asia_hl
    new_london = _session_hl(london) if not london.empty else prev_london_hl

    return new_ny, new_asia, new_london


def _update_session_levels_from_cache(df, prev_ny, prev_asia, prev_london):
    """No-op placeholder — session levels must be computed from bars."""
    pass


def _process_single_date(
    date_str: str,
    data_dir: Path,
    symbol: str,
    util_cfg: DashboardUtilityConfig,
    prev_ny_hl: tuple[float, float] | None,
    prev_asia_hl: tuple[float, float] | None,
    prev_london_hl: tuple[float, float] | None,
) -> pd.DataFrame:
    """Process a single date: bars -> levels -> touches -> label -> features."""

    # 1. Build bars
    bars = _build_bars_for_date(data_dir, symbol, date_str, util_cfg)
    if bars.empty:
        return pd.DataFrame()

    bars_et = _ensure_et_index(bars)

    # 2. Compute key levels available for this date
    levels = _compute_levels_for_date(
        bars_et, date_str, prev_ny_hl, prev_asia_hl, prev_london_hl,
    )
    if not levels:
        return pd.DataFrame()

    # 3. Build zones and detect touches
    zones = _build_zones(levels)
    touches = _detect_touches(bars_et, zones)
    if not touches:
        return pd.DataFrame()

    # 4. Label each touch and compute features
    rows: list[dict] = []
    for touch in touches:
        # Forward bars for MFE/MAE labeling
        forward = bars_et[bars_et.index > touch["bar_ts"]]
        # Limit to RTH close
        rth_cutoff = pd.Timestamp(f"{date_str} 16:15:00", tz=_ET)
        forward = forward[forward.index < rth_cutoff]

        if forward.empty:
            continue

        label_result = label_touch_event(touch, forward, util_cfg)
        if label_result["label"] == NO_RESOLUTION:
            continue

        # Compute interaction features from raw ticks
        features = _compute_interaction_features(
            touch, data_dir, symbol, util_cfg,
        )
        if features is None:
            continue

        row = {
            "event_ts": touch["bar_ts"],
            "date": date_str,
            "timestamp": touch["bar_ts"],
            "direction": touch["direction"],
            "representative_price": touch["representative_price"],
            "level_type": touch["level_type"],
            "label": label_result["label"],
            "label_encoded": label_result["label_encoded"],
            "max_mfe": label_result["max_mfe"],
            "max_mae": label_result["max_mae"],
        }
        row.update(features)

        # Optionally compute approach features
        if util_cfg.include_approach_features:
            approach = _compute_approach_features(
                touch, data_dir, symbol, util_cfg,
            )
            if approach:
                row.update(approach)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Bar Building ──────────────────────────────────────────────────


def _build_bars_for_date(
    data_dir: Path, symbol: str, date_str: str,
    util_cfg: DashboardUtilityConfig,
) -> pd.DataFrame:
    """Build bars for a single date using the configured bar_type."""
    td = date.fromisoformat(date_str)
    prev_day = td - timedelta(days=1)

    # Session spans 18:00 ET prev day to 18:00 ET current day
    # In UTC: 23:00 prev day to 23:00 current day (EST)
    start_utc = datetime(prev_day.year, prev_day.month, prev_day.day, 23, 0)
    end_utc = datetime(td.year, td.month, td.day, 23, 0)

    store = TickStore(data_dir)
    try:
        # Register both prev day and current day
        for d in [prev_day, td]:
            store.register_symbol_date(symbol, d)

        bar_type = util_cfg.bar_type
        if bar_type == "1m":
            # Check for cached session bars first
            cached = data_dir / symbol / date_str / "ohlcv_1m_session.parquet"
            if cached.exists():
                df = pd.read_parquet(cached)
                if not isinstance(df.index, pd.DatetimeIndex):
                    if "timestamp" in df.columns:
                        df = df.set_index("timestamp")
                return df
            df = store.build_bars_from_ticks(
                symbol, start_utc, end_utc, bar_size="1 minute",
            )
        elif bar_type.endswith("t"):
            tick_count = int(bar_type[:-1])
            df = store.build_tick_bars(
                symbol, start_utc, end_utc, tick_count=tick_count,
            )
        else:
            logger.warning("Unknown bar_type: %s, falling back to 987t", bar_type)
            df = store.build_tick_bars(symbol, start_utc, end_utc, tick_count=987)
    finally:
        store.close()

    return df


def _ensure_et_index(bars: pd.DataFrame) -> pd.DataFrame:
    """Convert bar index to US/Eastern timezone."""
    if bars.empty:
        return bars
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC").tz_convert(_ET)
    elif str(bars.index.tz) != _ET:
        bars.index = bars.index.tz_convert(_ET)
    return bars


# ── Level Computation ─────────────────────────────────────────────


def _session_hl(bars: pd.DataFrame) -> tuple[float, float]:
    return float(bars["high"].max()), float(bars["low"].min())


def _slice_session(bars: pd.DataFrame, session: str) -> pd.DataFrame:
    if bars.empty:
        return bars
    times = bars.index.time
    if session == "asia":
        mask = (times >= _ASIA_START) | (times < _ASIA_END)
    elif session == "london":
        mask = (times >= _LONDON_START) & (times < _LONDON_END)
    elif session == "ny_rth":
        mask = (times >= _NY_RTH_START) & (times < _NY_RTH_END)
    else:
        return pd.DataFrame(columns=bars.columns)
    return bars[mask]


def _compute_levels_for_date(
    bars_et: pd.DataFrame,
    date_str: str,
    prev_ny_hl: tuple[float, float] | None,
    prev_asia_hl: tuple[float, float] | None,
    prev_london_hl: tuple[float, float] | None,
) -> list[dict]:
    """Compute key levels available for this trading date."""
    levels = []

    # PDH/PDL from prior day's NY RTH
    if prev_ny_hl is not None:
        levels.append({"name": "PDH", "price": prev_ny_hl[0], "side": "HIGH"})
        levels.append({"name": "PDL", "price": prev_ny_hl[1], "side": "LOW"})

    # Asia levels (available after 01:00 ET)
    asia = _slice_session(bars_et, "asia")
    if not asia.empty:
        hl = _session_hl(asia)
        levels.append({"name": "asia_high", "price": hl[0], "side": "HIGH"})
        levels.append({"name": "asia_low", "price": hl[1], "side": "LOW"})

    # London levels (available after 08:00 ET)
    london = _slice_session(bars_et, "london")
    if not london.empty:
        hl = _session_hl(london)
        levels.append({"name": "london_high", "price": hl[0], "side": "HIGH"})
        levels.append({"name": "london_low", "price": hl[1], "side": "LOW"})

    return levels


# ── Zone Merging & Touch Detection ────────────────────────────────


def _build_zones(levels: list[dict]) -> list[dict]:
    """Merge levels within ZONE_PROXIMITY into zones."""
    if not levels:
        return []

    sorted_levels = sorted(levels, key=lambda l: l["price"])
    groups: list[list[dict]] = [[sorted_levels[0]]]

    for lvl in sorted_levels[1:]:
        if lvl["price"] - groups[-1][-1]["price"] <= _ZONE_PROXIMITY:
            groups[-1].append(lvl)
        else:
            groups.append([lvl])

    zones = []
    for group in groups:
        prices = [l["price"] for l in group]
        rep_price = sum(prices) / len(prices)
        names = [l["name"] for l in group]
        # Side by majority
        high_count = sum(1 for l in group if l["side"] == "HIGH")
        side = "HIGH" if high_count > len(group) / 2 else "LOW"
        zones.append({
            "representative_price": rep_price,
            "names": names,
            "side": side,
            "touched": False,
        })

    return zones


def _detect_touches(
    bars_et: pd.DataFrame, zones: list[dict],
) -> list[dict]:
    """Detect first-touch events: bar range intersects level price."""
    touches = []

    for bar_ts, bar in bars_et.iterrows():
        bar_low = bar["low"]
        bar_high = bar["high"]

        for zone in zones:
            if zone["touched"]:
                continue

            rep = zone["representative_price"]
            if bar_low <= rep <= bar_high:
                zone["touched"] = True
                direction = "LONG" if zone["side"] == "LOW" else "SHORT"
                touches.append({
                    "bar_ts": bar_ts,
                    "representative_price": rep,
                    "direction": direction,
                    "level_type": zone["names"][0],
                    "date": str(bar_ts.date()) if hasattr(bar_ts, "date") else "",
                })

    return touches


# ── Feature Computation ───────────────────────────────────────────


def _compute_interaction_features(
    touch: dict,
    data_dir: Path,
    symbol: str,
    config: DashboardUtilityConfig,
) -> dict[str, float] | None:
    """Compute 3 dashboard features from the tick interaction window."""
    event_ts = pd.Timestamp(touch["bar_ts"])
    rep_price = float(touch["representative_price"])
    direction = touch["direction"]
    window_minutes = config.interaction_window_minutes
    proximity = config.level_proximity_pts

    if event_ts.tz is not None:
        event_ts_utc = event_ts.tz_convert("UTC")
    else:
        event_ts_utc = event_ts.tz_localize("UTC")

    date_str = touch.get("date", "")
    if not date_str:
        date_str = str(event_ts.date())

    store = TickStore(data_dir)
    registered = store.register_symbol_date(symbol, date_str)
    if not registered:
        store.close()
        return None

    start = event_ts_utc.to_pydatetime()
    end = start + timedelta(minutes=window_minutes)

    try:
        ticks = store.query_tick_feature_rows(symbol, start, end)
    finally:
        store.close()

    if ticks.empty or len(ticks) < 5:
        return None

    if "price" not in ticks.columns:
        return None

    mid = ticks["price"].values
    ts = ticks["ts_event"].values

    # Tempo features
    time_beyond = 0.0
    time_within = 0.0

    for j in range(len(mid) - 1):
        delta = ts[j + 1] - ts[j]
        dt_sec = float(delta / np.timedelta64(1, "s"))
        if dt_sec < 0 or dt_sec > 600:
            continue

        m = float(mid[j])
        if abs(m - rep_price) <= 2.0:
            time_within += dt_sec

        if direction == "LONG" and m < rep_price:
            time_beyond += dt_sec
        elif direction == "SHORT" and m > rep_price:
            time_beyond += dt_sec

    # Absorption ratio
    if "size" in ticks.columns:
        prices = ticks["price"].values
        sizes = ticks["size"].values
        vol_at_level = 0.0
        vol_through = 0.0
        level_low = rep_price - proximity
        level_high = rep_price + proximity

        for j in range(len(prices)):
            p = float(prices[j])
            s = float(sizes[j])
            if level_low <= p <= level_high:
                vol_at_level += s
            elif direction == "LONG" and p < rep_price:
                vol_through += s
            elif direction == "SHORT" and p > rep_price:
                vol_through += s

        total = vol_at_level + vol_through
        absorption = vol_at_level / total if total > 0 else 0.0
    else:
        absorption = 0.0

    return {
        "int_time_beyond_level": round(time_beyond, 4),
        "int_time_within_2pts": round(time_within, 4),
        "int_absorption_ratio": round(min(1.0, max(0.0, absorption)), 6),
    }


def _compute_approach_features(
    touch: dict,
    data_dir: Path,
    symbol: str,
    config: DashboardUtilityConfig,
) -> dict[str, float] | None:
    """Compute live-computable approach features from pre-touch order flow.

    Queries all 27 features via DuckDB SQL, then filters to the 8 that
    can be computed from MBP-1 (top-of-book) in live trading. Features
    requiring deeper book levels or cancel events are excluded.
    """
    event_ts = pd.Timestamp(touch["bar_ts"])
    if event_ts.tz is not None:
        event_ts_utc = event_ts.tz_convert("UTC")
    else:
        event_ts_utc = event_ts.tz_localize("UTC")

    date_str = touch.get("date", str(event_ts.date()))
    approach_minutes = config.approach_window_minutes
    approach_start = event_ts_utc - pd.Timedelta(minutes=approach_minutes)

    store = TickStore(data_dir)
    registered = store.register_symbol_date(symbol, date_str)
    # Also register previous calendar day (approach window may cross midnight)
    prev_date = (date.fromisoformat(date_str) - timedelta(days=1)).isoformat()
    store.register_symbol_date(symbol, prev_date)

    if not registered:
        store.close()
        return None

    try:
        views = store._get_views(symbol)
        if not views:
            return None
        union_sql = store._union_views_sql(views)

        # Detect symbol filter
        sample_sql = (
            f"SELECT column_name FROM "
            f"(DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        )
        cols = {r[0] for r in store._conn.execute(sample_sql).fetchall()}
        has_symbol = "symbol" in cols

        if has_symbol:
            front = store._conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            sym_f = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
        else:
            sym_f = ""

        # Import and call the experiment approach feature query
        from alpha_lab.experiment.features import _query_approach_features
        from alpha_lab.agents.data_infra.ml.config import LIVE_APPROACH_FEATURES
        all_features = _query_approach_features(
            store._conn, union_sql, sym_f,
            approach_start, event_ts_utc,
        )
        # Keep only features computable from MBP-1 (top-of-book) in live trading
        return {k: v for k, v in all_features.items() if k in LIVE_APPROACH_FEATURES}
    except Exception as exc:
        logger.debug("Approach features failed for %s: %s", date_str, exc)
        return None
    finally:
        store.close()
