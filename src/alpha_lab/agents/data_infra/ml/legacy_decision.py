"""LEGACY decision stages — imported by parity/regression tests ONLY.

W1 P4b: these stages left the production pipeline (``dashboard_utility_builder``
now drives the shared Strategy-Core runtime over the canonical day stream via
``engine_decision.process_single_date_stream``). They are preserved here, verbatim,
solely so the recorded book-mid repoint-parity proof
(``tests/agents/test_decision_repoint_parity.py``) stays executable against the
historical implementation. NOTHING in ``src/`` imports this module.

Moved from ``dashboard_utility_builder.py``:
  * ``_session_hl`` / ``_slice_session``      (bar-close session attribution)
  * ``_compute_levels_for_date``              (levels from final-bar slices)
  * ``_build_zones`` / ``_detect_touches``    (builder-local zone/touch twins)
  * ``_compute_interaction_features``         (DuckDB tick-window features)
  * ``_compute_approach_features``            (DuckDB approach features)
  * ``process_single_date_legacy``            (the ``use_engine=False`` branch)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from strategy_core.constants import (
    RESEARCH_SESSION_SCHEME as _SCHEME,
)
from strategy_core.constants import (
    TRADING_DAY_BOUNDARY as _TRADING_DAY_BOUNDARY,
)
from strategy_core.constants import (
    ZONE_PROXIMITY_PTS as _ZONE_PROXIMITY,
)

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _build_bars_for_date,
    _ensure_et_index,
)
from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import (
    NO_RESOLUTION,
    label_touch_event,
)
from alpha_lab.agents.data_infra.tick_store import TickStore

logger = logging.getLogger(__name__)

_ET = _SCHEME.timezone  # "US/Eastern"
_ASIA = _SCHEME.sessions["asia"]
_LONDON = _SCHEME.sessions["london"]
_NY = _SCHEME.sessions["ny"]


def _get_session_hl_for_date(
    data_dir: Path,
    symbol: str,
    date_str: str,
    util_cfg: DashboardUtilityConfig,
    prev_full_hl,
    prev_asia_hl,
    prev_london_hl,
):
    """LEGACY carry: (full H/L, asia H/L, london H/L) for the next day's levels."""
    bars = _build_bars_for_date(data_dir, symbol, date_str, util_cfg)
    if bars.empty:
        return prev_full_hl, prev_asia_hl, prev_london_hl

    bars_et = _ensure_et_index(bars)
    new_full = (float(bars_et["high"].max()), float(bars_et["low"].min()))
    asia = _slice_session(bars_et, "asia")
    london = _slice_session(bars_et, "london")

    new_asia = _session_hl(asia) if not asia.empty else prev_asia_hl
    new_london = _session_hl(london) if not london.empty else prev_london_hl

    return new_full, new_asia, new_london


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
    """Compute key levels available for this trading date (LEGACY batch form).

    Engine v3: each level carries ``available_from`` — the UTC instant at/after which
    it may first be touched (its defining session's CLOSE) — so the engine's enforced
    look-ahead guard can gate detect_touches. PDH/PDL are the FULL prior trading day's
    high/low (``prev_full_hl``, the daily-candle extremes), available from the
    trading-day start (prior 18:00 ET). asia/london H/L come from THIS day's session
    slices and are available from the Asia close (02:45 ET) / London close (08:00 ET).
    """
    td = date.fromisoformat(date_str)
    prev_day = td - timedelta(days=1)

    def _avail(d: date, t: time) -> datetime:
        # ET wall-clock instant -> tz-aware UTC (DST-aware). On a spring-forward day a
        # nonexistent instant shifts forward (conservative, never earlier).
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

    asia = _slice_session(bars_et, "asia")
    if not asia.empty:
        hl = _session_hl(asia)
        levels.append(
            {"name": "asia_high", "price": hl[0], "side": "HIGH", "available_from": asia_avail}
        )
        levels.append(
            {"name": "asia_low", "price": hl[1], "side": "LOW", "available_from": asia_avail}
        )

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
    """Compute live-computable approach features from pre-touch order flow."""
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

        from alpha_lab.agents.data_infra.ml.config import LIVE_APPROACH_FEATURES
        from alpha_lab.experiment.features import _query_approach_features

        all_features = _query_approach_features(
            store._conn,
            union_sql,
            sym_f,
            approach_start,
            event_ts_utc,
        )
        return {k: v for k, v in all_features.items() if k in LIVE_APPROACH_FEATURES}
    except Exception as exc:
        logger.debug("Approach features failed for %s: %s", date_str, exc)
        return None
    finally:
        store.close()


def process_single_date_legacy(
    date_str: str,
    data_dir: Path,
    symbol: str,
    util_cfg: DashboardUtilityConfig,
    prev_full_hl: tuple[float, float] | None,
    prev_asia_hl: tuple[float, float] | None,
    prev_london_hl: tuple[float, float] | None,
) -> pd.DataFrame:
    """The retired ``_process_single_date(use_engine=False)`` branch, verbatim."""

    bars = _build_bars_for_date(data_dir, symbol, date_str, util_cfg)
    if bars.empty:
        return pd.DataFrame()

    bars_et = _ensure_et_index(bars)

    levels = _compute_levels_for_date(
        bars_et,
        date_str,
        prev_full_hl,
        prev_asia_hl,
        prev_london_hl,
    )
    if not levels:
        return pd.DataFrame()

    zones = _build_zones(levels)
    touches = _detect_touches(bars_et, zones)
    if not touches:
        return pd.DataFrame()

    rows: list[dict] = []
    for touch in touches:
        forward = bars_et[bars_et.index > touch["bar_ts"]]
        rth_cutoff = pd.Timestamp(f"{date_str} 16:15:00", tz=_ET)
        forward = forward[forward.index < rth_cutoff]

        if forward.empty:
            continue

        label_result = label_touch_event(touch, forward, util_cfg)
        if label_result["label"] == NO_RESOLUTION:
            continue

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
