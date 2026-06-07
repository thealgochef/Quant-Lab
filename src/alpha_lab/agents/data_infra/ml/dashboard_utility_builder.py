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

# Session boundaries (Eastern Time) — SINGLE-SOURCED from the shared engine scheme
# (strategy_core.constants.RESEARCH_SESSION_SCHEME) so research slicing can never drift
# from the engine's classify_session. Engine v3 re-clock: asia 19:00->02:45 (crosses
# midnight), london 03:00->08:00, ny 09:00->17:00; 18:00 ET trading-day boundary.
from strategy_core.constants import (
    RESEARCH_SESSION_SCHEME as _SCHEME,
)
from strategy_core.constants import (
    TRADING_DAY_BOUNDARY as _TRADING_DAY_BOUNDARY,
)
from strategy_core.constants import (
    ZONE_PROXIMITY_PTS as _ZONE_PROXIMITY,
)

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

_ET = _SCHEME.timezone  # "US/Eastern"
_ASIA = _SCHEME.sessions["asia"]
_LONDON = _SCHEME.sessions["london"]
_NY = _SCHEME.sessions["ny"]


def build_utility_dataset(
    dates: list[str],
    data_dir: Path,
    config: MLPipelineConfig,
    progress_fn=None,
    *,
    use_engine: bool = True,
) -> pd.DataFrame:
    """Build a labeled feature dataset for dashboard-utility training.

    Self-contained: builds bars, detects levels, finds touches, labels,
    and computes features directly from raw tick parquet files.

    Args:
        dates: List of date strings to process (e.g. ["2025-06-02", ...]).
        data_dir: Root databento data directory.
        config: Pipeline config (uses dashboard_utility sub-config).
        progress_fn: Optional callable(fraction, text) for progress.
        use_engine: When True (default, phase-5 repoint), the DECISION LAYER
            (zones -> touches -> labels -> 6 features) is single-sourced onto the
            shared ``strategy_core`` engine via ``engine_decision``. When False,
            the legacy duplicate CQL decision code runs. Bars and levels are
            identical in both paths (book-mid, unchanged); only the decision layer
            differs, so the two paths are parity-diffable. The engine path is
            proven to reproduce the legacy/4a-canonical output EXACTLY (see
            ``tests/agents/test_decision_repoint_parity.py``).

    Returns:
        DataFrame with one row per labeled touch event.
    """
    util_cfg = config.dashboard_utility
    symbol = config.instrument
    cache_tag = config.dataset_config_hash()

    frames: list[pd.DataFrame] = []
    cached_count = 0

    # Track the prior FULL trading day's high/low for PDH/PDL (engine v3); asia/london
    # carry kept for signature stability (levels recompute them from the current day).
    prev_full_hl: tuple[float, float] | None = None
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
                df,
                prev_full_hl,
                prev_asia_hl,
                prev_london_hl,
            )
            # Read session levels from bars if we need them for next day
            prev_full_hl, prev_asia_hl, prev_london_hl = _get_session_hl_for_date(
                data_dir,
                symbol,
                date_str,
                util_cfg,
                prev_full_hl,
                prev_asia_hl,
                prev_london_hl,
            )
            continue

        # Build fresh for this date
        df = _process_single_date(
            date_str,
            data_dir,
            symbol,
            util_cfg,
            prev_full_hl,
            prev_asia_hl,
            prev_london_hl,
            use_engine=use_engine,
        )

        # Update session levels for next day
        prev_full_hl, prev_asia_hl, prev_london_hl = _get_session_hl_for_date(
            data_dir,
            symbol,
            date_str,
            util_cfg,
            prev_full_hl,
            prev_asia_hl,
            prev_london_hl,
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
    from alpha_lab.agents.data_infra.ml.config import (
        LIVE_APPROACH_FEATURES,
        LIVE_INTERACTION_FEATURES,
    )

    live_features = set(LIVE_INTERACTION_FEATURES + LIVE_APPROACH_FEATURES)
    drop_cols = [
        c
        for c in result.columns
        if (c.startswith("app_") or c.startswith("int_")) and c not in live_features
    ]
    if drop_cols:
        result = result.drop(columns=drop_cols)

    logger.info(
        "Utility dataset: %d labeled events from %d dates (%d cached)",
        len(result),
        len(dates),
        cached_count,
    )
    return result


def _get_session_hl_for_date(
    data_dir: Path,
    symbol: str,
    date_str: str,
    util_cfg: DashboardUtilityConfig,
    prev_full_hl,
    prev_asia_hl,
    prev_london_hl,
):
    """Compute this date's carry state for the NEXT day's levels.

    Engine v3 PDH/PDL change: the FIRST element is now the FULL prior-trading-day H/L
    (max-high / min-low over the entire [18:00, 18:00) ET window — the daily-candle
    extremes), which becomes the next day's PDH/PDL. (Was the prior NY-RTH slice.) The
    asia/london elements are retained for signature stability; ``_compute_levels_for_date``
    recomputes asia/london from the CURRENT day's bars, so they are not consumed.
    """
    bars = _build_bars_for_date(data_dir, symbol, date_str, util_cfg)
    if bars.empty:
        return prev_full_hl, prev_asia_hl, prev_london_hl

    bars_et = _ensure_et_index(bars)
    # FULL trading-day high/low (the entire 18:00->18:00 ET window) -> next day PDH/PDL.
    new_full = (float(bars_et["high"].max()), float(bars_et["low"].min()))
    asia = _slice_session(bars_et, "asia")
    london = _slice_session(bars_et, "london")

    new_asia = _session_hl(asia) if not asia.empty else prev_asia_hl
    new_london = _session_hl(london) if not london.empty else prev_london_hl

    return new_full, new_asia, new_london


def _update_session_levels_from_cache(df, prev_ny, prev_asia, prev_london):
    """No-op placeholder — session levels must be computed from bars."""
    pass


def _process_single_date(
    date_str: str,
    data_dir: Path,
    symbol: str,
    util_cfg: DashboardUtilityConfig,
    prev_full_hl: tuple[float, float] | None,
    prev_asia_hl: tuple[float, float] | None,
    prev_london_hl: tuple[float, float] | None,
    *,
    use_engine: bool = True,
) -> pd.DataFrame:
    """Process a single date: bars -> levels -> touches -> label -> features.

    The bars (TRADE-price after the Part-1 cutover) and the level set are computed
    here regardless of ``use_engine``. When ``use_engine`` is True (default) the
    decision/label/feature stage is single-sourced onto the shared ``strategy_core``
    engine (``engine_decision.process_single_date_engine``) in its PRODUCTION mode
    (trade-print interaction features at 0.25 + the decision-time honest outcome).
    When False the legacy duplicate CQL decision code below runs (book-mid-era
    level-entry labeling); the two are NO LONGER expected to match after the cutover
    (the engine path is trade-print + honest-entry by design), but both remain
    importable, and the book-mid-equivalence regression is exercised via
    ``process_single_date_engine(price_source="book_mid", tick_size=0.125,
    honest_entry=False)`` in the repoint-parity test.
    """

    # 1. Build bars (TRADE-price after the Part-1 cutover — identical input for both
    # paths; the legacy path's book-mid-era labeler now sees trade bars).
    bars = _build_bars_for_date(data_dir, symbol, date_str, util_cfg)
    if bars.empty:
        return pd.DataFrame()

    bars_et = _ensure_et_index(bars)

    # 2. Compute key levels available for this date (identical for both paths)
    levels = _compute_levels_for_date(
        bars_et,
        date_str,
        prev_full_hl,
        prev_asia_hl,
        prev_london_hl,
    )
    if not levels:
        return pd.DataFrame()

    # 2b. Phase-5 repoint: run the engine decision layer on the SAME bars+levels.
    if use_engine:
        from alpha_lab.agents.data_infra.ml.engine_decision import (
            process_single_date_engine,
        )

        return process_single_date_engine(
            bars_et,
            levels,
            date_str,
            data_dir,
            symbol,
            util_cfg,
        )

    # ── Legacy CQL decision path (kept intact for the parity diff) ──────────
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
            touch,
            data_dir,
            symbol,
            util_cfg,
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
                touch,
                data_dir,
                symbol,
                util_cfg,
            )
            if approach:
                row.update(approach)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Bar Building ──────────────────────────────────────────────────


def _build_bars_for_date(
    data_dir: Path,
    symbol: str,
    date_str: str,
    util_cfg: DashboardUtilityConfig,
) -> pd.DataFrame:
    """Build bars for a single date using the configured bar_type."""
    td = date.fromisoformat(date_str)
    prev_day = td - timedelta(days=1)

    # Session spans 18:00 ET (prev day) to 18:00 ET (current day), DST-aware.
    # Pass tz-aware UTC bounds so DuckDB compares them directly against the TIMESTAMPTZ
    # ts_event column -- NOT a naive datetime that DuckDB reinterprets in its session
    # timezone (the prior 23:00-CT window artifact). build_tick_bars partitions by the
    # same 18:00-ET trading day, so [prev 18:00 ET, cur 18:00 ET) yields this date's bars.
    start_utc = pd.Timestamp(f"{prev_day.isoformat()} 18:00:00", tz="America/New_York").tz_convert(
        "UTC"
    )
    end_utc = pd.Timestamp(f"{td.isoformat()} 18:00:00", tz="America/New_York").tz_convert("UTC")

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
                if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                return df
            df = store.build_bars_from_ticks(
                symbol,
                start_utc,
                end_utc,
                bar_size="1 minute",
            )
        elif bar_type.endswith("t"):
            tick_count = int(bar_type[:-1])
            # TRADE-BAR CUTOVER (Part 1): the decision/dashboard pipeline now builds
            # TRADE-PRICE tick bars (a tick is a trade print, action='T', OHLC on the
            # 0.25 grid) — the ratified production bar definition
            # (strategy_core.constants.BAR_PRICE_SOURCE="trade_price"). This supersedes
            # the phase-4c book-mid bars. The engine decision path is threaded with
            # tick_size=0.25 to match (engine_decision.process_single_date_engine).
            # DuckDB<->streaming trade-bar parity is proven in strategy-core/validation.
            df = store.build_tick_bars(
                symbol,
                start_utc,
                end_utc,
                tick_count=tick_count,
                price_source="trade",
            )
        else:
            logger.warning("Unknown bar_type: %s, falling back to 987t", bar_type)
            df = store.build_tick_bars(
                symbol,
                start_utc,
                end_utc,
                tick_count=987,
                price_source="trade",
            )
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
        # crosses midnight: t >= 19:00 OR t < 02:45 (engine SessionWindow.contains)
        mask = (times >= _ASIA.start) | (times < _ASIA.end)
    elif session == "london":
        mask = (times >= _LONDON.start) & (times < _LONDON.end)
    elif session == "ny":
        mask = (times >= _NY.start) & (times < _NY.end)
    else:
        return pd.DataFrame(columns=bars.columns)
    return bars[mask]


def _compute_levels_for_date(
    bars_et: pd.DataFrame,
    date_str: str,
    prev_full_hl: tuple[float, float] | None,
    prev_asia_hl: tuple[float, float] | None,
    prev_london_hl: tuple[float, float] | None,
) -> list[dict]:
    """Compute key levels available for this trading date.

    Engine v3: each level carries ``available_from`` — the UTC instant at/after which
    it may first be touched (its defining session's CLOSE) — so the engine's enforced
    look-ahead guard can gate detect_touches. PDH/PDL are the FULL prior trading day's
    high/low (``prev_full_hl``, the daily-candle extremes), available from the
    trading-day start (prior 18:00 ET). asia/london H/L come from THIS day's session
    slices and are available from the Asia close (02:45 ET) / London close (08:00 ET).
    Availability instants are single-sourced from the engine session scheme.
    """
    td = date.fromisoformat(date_str)
    prev_day = td - timedelta(days=1)

    def _avail(d: date, t: time) -> datetime:
        # ET wall-clock instant -> tz-aware UTC (DST-aware), matching how the engine
        # bars' close_ts_utc are produced, so the detect_touches gate compares like
        # for like. DST-safe: on a spring-forward day the 02:45 ET asia-close instant
        # does not exist (clocks jump 02:00->03:00), so localize a NAIVE wall-clock with
        # nonexistent="shift_forward" (the level becomes available at 03:00 ET that day —
        # 15 min later, conservative, never earlier). The 18:00/02:45/08:00 anchors are
        # never in the fall-back fold (01:00-02:00), so ambiguous is moot.
        naive = pd.Timestamp(f"{d.isoformat()} {t.strftime('%H:%M:%S')}")
        return (
            naive.tz_localize(_ET, nonexistent="shift_forward", ambiguous=False)
            .tz_convert("UTC")
            .to_pydatetime()
        )

    pdh_pdl_avail = _avail(prev_day, _TRADING_DAY_BOUNDARY)  # prior 18:00 ET (day start)
    asia_avail = _avail(td, _ASIA.end)  # 02:45 ET (Asia close)
    london_avail = _avail(td, _LONDON.end)  # 08:00 ET (London close)

    levels = []

    # PDH/PDL = FULL prior trading day's high/low; available from the trading-day start.
    if prev_full_hl is not None:
        levels.append(
            {
                "name": "PDH",
                "price": prev_full_hl[0],
                "side": "HIGH",
                "available_from": pdh_pdl_avail,
            }
        )
        levels.append(
            {
                "name": "PDL",
                "price": prev_full_hl[1],
                "side": "LOW",
                "available_from": pdh_pdl_avail,
            }
        )

    # Asia levels (available from the Asia close, 02:45 ET)
    asia = _slice_session(bars_et, "asia")
    if not asia.empty:
        hl = _session_hl(asia)
        levels.append(
            {"name": "asia_high", "price": hl[0], "side": "HIGH", "available_from": asia_avail}
        )
        levels.append(
            {"name": "asia_low", "price": hl[1], "side": "LOW", "available_from": asia_avail}
        )

    # London levels (available from the London close, 08:00 ET)
    london = _slice_session(bars_et, "london")
    if not london.empty:
        hl = _session_hl(london)
        levels.append(
            {"name": "london_high", "price": hl[0], "side": "HIGH", "available_from": london_avail}
        )
        levels.append(
            {"name": "london_low", "price": hl[1], "side": "LOW", "available_from": london_avail}
        )

    return levels


# ── Zone Merging & Touch Detection ────────────────────────────────


def _build_zones(levels: list[dict]) -> list[dict]:
    """Merge levels within ZONE_PROXIMITY into zones."""
    if not levels:
        return []

    sorted_levels = sorted(levels, key=lambda level: level["price"])
    groups: list[list[dict]] = [[sorted_levels[0]]]

    for lvl in sorted_levels[1:]:
        if lvl["price"] - groups[-1][-1]["price"] <= _ZONE_PROXIMITY:
            groups[-1].append(lvl)
        else:
            groups.append([lvl])

    zones = []
    for group in groups:
        prices = [level["price"] for level in group]
        rep_price = sum(prices) / len(prices)
        names = [level["name"] for level in group]
        # Side by majority
        high_count = sum(1 for level in group if level["side"] == "HIGH")
        side = "HIGH" if high_count > len(group) / 2 else "LOW"
        zones.append(
            {
                "representative_price": rep_price,
                "names": names,
                "side": side,
                "touched": False,
            }
        )

    return zones


def _detect_touches(
    bars_et: pd.DataFrame,
    zones: list[dict],
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
                touches.append(
                    {
                        "bar_ts": bar_ts,
                        "representative_price": rep,
                        "direction": direction,
                        "level_type": zone["names"][0],
                        "date": str(bar_ts.date()) if hasattr(bar_ts, "date") else "",
                    }
                )

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

        if direction == "LONG" and m < rep_price or direction == "SHORT" and m > rep_price:
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
            elif direction == "LONG" and p < rep_price or direction == "SHORT" and p > rep_price:
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
        sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
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
        from alpha_lab.agents.data_infra.ml.config import LIVE_APPROACH_FEATURES
        from alpha_lab.experiment.features import _query_approach_features

        all_features = _query_approach_features(
            store._conn,
            union_sql,
            sym_f,
            approach_start,
            event_ts_utc,
        )
        # Keep only features computable from MBP-1 (top-of-book) in live trading
        return {k: v for k, v in all_features.items() if k in LIVE_APPROACH_FEATURES}
    except Exception as exc:
        logger.debug("Approach features failed for %s: %s", date_str, exc)
        return None
    finally:
        store.close()
