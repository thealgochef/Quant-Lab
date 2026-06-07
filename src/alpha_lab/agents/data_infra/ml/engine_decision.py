"""Engine-backed decision layer for the dashboard-utility training builder.

Phase 5 single-sourced the TRAINING decision layer onto the shared
``strategy_core`` engine instead of the duplicate CQL code in
``dashboard_utility_builder.py`` (``_build_zones`` / ``_detect_touches`` /
``label_touch_event`` / ``_compute_interaction_features`` / the approach
features). This module is the thin ADAPTER that converts CQL's bars + levels +
raw tick rows into the engine's neutral types and runs
``build_zones -> detect_touches -> resolve_outcome -> the 6 engine features``.

TRADE-BAR CUTOVER + HONEST-ENTRY RE-ANCHOR (Parts 1+2 — engine v2). The
PRODUCTION decision path is now TRADE bars + TRADE-PRINT interaction features +
the decision-time honest outcome:
  * Bars come from ``_build_bars_for_date`` at ``price_source="trade"`` (0.25 grid).
  * The decision grid is threaded ``tick_size = TRADE_TICK = 0.25`` (the contract
    constant ``strategy_core.constants.DEFAULT_TICK_SIZE``) through bars / touches /
    outcome / interaction. ``within_band`` stays 2.0 POINTS (a points constant).
  * ALL 3 interaction features re-source to TRADE PRINTS (action='T', 0.25 grid)
    via ``query_tick_feature_rows(price_source="trade")``.
  * The canonical OUTCOME re-anchors to the REALISTIC price at the DECISION instant
    (touch + decision_offset = interaction window), matching the Trade-Lab executor,
    with the forward window starting AFTER the decision (look-ahead closure). Touches
    whose decision_time is at/after the Strategy-Core flatten gate get NO tradeable outcome.

BOOK-MID MODE STAYS REACHABLE (regression): every adapter entry point is
parameterized by ``price_source`` / ``tick_size``. ``price_source="book_mid"`` +
``tick_size=BOOK_MID_TICK`` (0.125) + ``honest_entry=False`` reproduces the phase-5
book-mid level-entry decision layer EXACTLY, so the phase-5 engine==legacy parity
proof (``tests/agents/test_decision_repoint_parity.py``) stays valid against that
mode. Production defaults are trade/0.25/honest_entry=True.

Training labels EVERY touch with a resolvable forward window — the runtime gates
``ny_rth`` at inference, not training — so NO session gate is added to labeling
here (``classify_session`` is available for eligibility reporting). The ONE
execution rule folded into labeling is the flatten no-entry rule (Part 2), since a
touch the executor would never trade has no honest outcome.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import strategy_core as sc
from strategy_core import (
    Bar,
    CloseReason,
    HonestEntryDrop,
    Level,
    Quote,
    Side,
    Trade,
    app_avg_trade_size,
    app_large_trade_vol_pct,
    app_max_spread,
    build_zones,
    classify_session,
    detect_touches,
    int_absorption_ratio,
    int_time_beyond_level,
    int_time_within_2pts,
    make_bar_id,
    resolve_honest_outcome,
    resolve_outcome,
)
from strategy_core.constants import (
    DEFAULT_TICK_SIZE,
    RTH_END,
)

from alpha_lab.agents.data_infra.ml.config import (
    LIVE_APPROACH_FEATURES,
    DashboardUtilityConfig,
)
from alpha_lab.agents.data_infra.tick_store import TickStore

# Book-mid grid (phase-4c/4f legacy): lossless representation of book-mid bars and
# the bar-based decision layer (zones / touches / labels). round(price / 0.125)
# gives integer ticks with no loss because a top-of-book mid lands on the 0.125
# grid. KEPT REACHABLE for the phase-5 repoint-parity regression. (parity_harness_v2.CANON_TICK)
BOOK_MID_TICK = 0.125
# Real NQ trade-price grid: the production decision grid (trade bars + trade-print
# interaction features) AND the approach trades/quotes. == strategy_core DEFAULT_TICK_SIZE.
TRADE_TICK = 0.25
DECISION_OFFSET_MINUTES = sc.constants.DECISION_OFFSET_MINUTES
assert TRADE_TICK == DEFAULT_TICK_SIZE  # the contract/constant the trade path threads

_ET = "US/Eastern"


# ── Conversions: CQL representations -> engine neutral types ────────────────


def _to_utc_dt(ts) -> datetime:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


def _round_to_ticks(value: float, tick: float) -> int:
    return int(round(float(value) / tick))


def bars_et_to_engine(
    bars_et: pd.DataFrame, trading_day: date, tick_size: float = TRADE_TICK
) -> list[Bar]:
    """Engine ``Bar``s from an ET-indexed OHLCV frame at the given decision grid.

    Mirrors ``parity_harness_v2._engine_bars_from_et``: the decision stages depend
    only on HIGH/LOW (tie-break independent), so open/close come straight from the
    frame. ``open_ts_utc`` and ``close_ts_utc`` are both the bar's index (its close
    instant — ``build_tick_bars`` indexes by LAST(ts_event)); ``bar_id`` via
    ``make_bar_id``. ``tick_size`` defaults to the production TRADE grid (0.25); pass
    ``BOOK_MID_TICK`` (0.125) for the reachable book-mid regression.
    """
    out: list[Bar] = []
    for i, (idx, row) in enumerate(bars_et.iterrows()):
        close_utc = _to_utc_dt(idx)
        out.append(
            Bar(
                timeframe_ticks=0,  # unused by the decision layer
                trading_day=trading_day,
                bar_index=i,
                bar_id=make_bar_id(0, trading_day, i),
                open_ts_utc=close_utc,
                close_ts_utc=close_utc,
                open_ticks=_round_to_ticks(row["open"], tick_size),
                high_ticks=_round_to_ticks(row["high"], tick_size),
                low_ticks=_round_to_ticks(row["low"], tick_size),
                close_ticks=_round_to_ticks(row["close"], tick_size),
                volume=int(row["volume"]) if "volume" in row else 0,
                trade_count=0,
                is_complete=True,
                is_partial=False,
                close_reason=CloseReason.COMPLETE,
            )
        )
    return out


def levels_to_engine(levels: list[dict], *, with_availability: bool = False) -> list[Level]:
    """CQL level dicts -> engine ``Level``s (parity_harness_v2._canon_levels_to_engine).

    ``with_availability`` (engine v3): when True, carry each level's ``available_from``
    UTC instant (its defining session's close) so the engine ENFORCES the look-ahead
    guard in ``detect_touches`` (a level cannot be touched before it exists). When
    False (the default — the ungated book-mid regression and the direct comparator
    calls in the repoint-parity test), availability is dropped to ``None`` so the
    engine reproduces the pre-v3 no-guard behavior byte-for-byte.
    """
    return [
        Level(
            name=level["name"],
            price=float(level["price"]),
            side=Side(level["side"]),
            available_from=(
                _to_utc_dt(level["available_from"])
                if (with_availability and level.get("available_from") is not None)
                else None
            ),
        )
        for level in levels
    ]


# ── The engine decision path for a single date ──────────────────────────────


def process_single_date_engine(
    bars_et: pd.DataFrame,
    levels: list[dict],
    date_str: str,
    data_dir: Path,
    symbol: str,
    config: DashboardUtilityConfig,
    *,
    price_source: str = "trade",
    tick_size: float = TRADE_TICK,
    honest_entry: bool = True,
) -> pd.DataFrame:
    """Engine-backed twin of ``_process_single_date``'s decision/label/feature stage.

    Pipeline: ``build_zones -> detect_touches -> resolve_outcome -> the 6 engine
    features``. Produces the SAME dataset row schema the legacy path emits. Returns
    an empty frame when there are no levels / touches / rows.

    Mode (trade-bar cutover + honest entry — engine v2 production DEFAULT):
      * ``tick_size=TRADE_TICK`` (0.25) — bars / touches / outcome / interaction grid.
      * ``price_source="trade"`` — the 3 interaction features read TRADE PRINTS.
      * ``honest_entry=True`` — the canonical OUTCOME is anchored to the realistic
        TRADE price at the DECISION instant (touch + ``DECISION_OFFSET_MINUTES``);
        the forward window starts AFTER the decision (look-ahead closure); touches
        whose decision_time is at/after the flatten get NO tradeable outcome.

    Reachable book-mid regression (the phase-5 parity proof): pass
    ``price_source="book_mid", tick_size=BOOK_MID_TICK, honest_entry=False`` to
    reproduce the legacy book-mid level-entry decision layer EXACTLY.
    """
    if not levels:
        return pd.DataFrame()

    td = date.fromisoformat(date_str)
    eng_bars = bars_et_to_engine(bars_et, td, tick_size)
    # Engine v3 look-ahead guard: the production (honest) path carries per-level
    # availability so detect_touches gates a touch to bars closing at/after the level's
    # defining-session close. The book-mid regression (honest_entry=False) stays
    # UNGATED, reproducing the pre-v3 no-guard behavior the repoint-parity proof pins.
    eng_levels = levels_to_engine(levels, with_availability=honest_entry)

    zones = build_zones(eng_levels)
    touches = detect_touches(eng_bars, zones, tick_size=tick_size, trading_day=td)
    if not touches:
        return pd.DataFrame()

    rth_cutoff = pd.Timestamp(f"{date_str} 16:15:00", tz=_ET)
    # The decision offset EQUALS the interaction feature window (the decision cannot
    # fire until the post-touch interaction features exist). Single-source it from
    # config; ``DECISION_OFFSET_MINUTES`` is the canonical default (== the constant
    # DEFAULT_INTERACTION_WINDOW_MINUTES) that config defaults to. The feature window
    # uses the SAME value, so feature window [touch, touch+offset] and label window
    # (touch+offset, Strategy-Core RTH_END] never overlap regardless of the
    # configured window.
    window_minutes = config.interaction_window_minutes
    decision_offset = window_minutes

    # The decision-time entry price is a front-month TRADE-print point query; open a
    # single per-date store once for all touches (honest mode only).
    entry_store: TickStore | None = None
    if honest_entry:
        entry_store = TickStore(data_dir)
        entry_store.register_symbol_date(symbol, date_str)
    # Honest mode runs the SINGLE engine orchestration (decision_ts / flatten /
    # cutoff / entry lookup / forward selection / resolve_outcome) — this adapter
    # only INJECTS the trade-price accessor and the day's engine bars. The whole
    # day's engine bars are built once (the engine slices the forward window).
    day_eng_bars = bars_et_to_engine(bars_et, td, tick_size) if honest_entry else None

    def _trade_price_for(ts_utc: datetime) -> float | None:
        return _trade_price_at(entry_store, symbol, ts_utc)

    try:
        rows: list[dict] = []
        for touch in touches:
            bar_ts_et = pd.Timestamp(touch.bar_ts_utc).tz_convert(_ET)

            if honest_entry:
                # The honest decision-time outcome is the ONE engine orchestration
                # (strategy_core.resolve_honest_outcome): decision_ts = touch close +
                # decision_offset; DROP at/after the Strategy-Core flatten gate or cutoff;
                # entry = the injected realistic TRADE price at the decision instant;
                # forward = bars whose close is in (decision, Strategy-Core RTH_END); then the pure
                # resolve_outcome. A drop -> no tradeable outcome (skip the touch).
                result = resolve_honest_outcome(
                    touch,
                    day_eng_bars,
                    _trade_price_for,
                    tick_size=tick_size,
                    tp_points=config.tp_points,
                    sl_points=config.sl_points,
                    trap_mfe_min=config.trap_mfe_min,
                    decision_offset_minutes=decision_offset,
                )
                if isinstance(result, HonestEntryDrop):
                    continue
                outcome = result
            else:
                # Legacy / book-mid regression: entry = level price; forward bars
                # strictly AFTER the touch bar, truncated at 16:15 ET.
                entry_price = float(touch.representative_price)
                forward = bars_et[(bars_et.index > bar_ts_et) & (bars_et.index < rth_cutoff)]
                if forward.empty:
                    continue
                outcome = resolve_outcome(
                    entry_points=entry_price,
                    direction=touch.direction,
                    forward_bars=bars_et_to_engine(forward, td, tick_size),
                    tick_size=tick_size,
                    tp_points=config.tp_points,
                    sl_points=config.sl_points,
                    trap_mfe_min=config.trap_mfe_min,
                )

            if outcome.label == sc.constants.NO_RESOLUTION:
                continue

            features = compute_interaction_features_engine(
                touch,
                data_dir,
                symbol,
                config,
                price_source=price_source,
                tick_size=tick_size,
                window_minutes=window_minutes,
            )
            if features is None:
                continue

            session_info = classify_session(touch.bar_ts_utc)
            decision_time_et = pd.Timestamp(
                touch.bar_ts_utc + timedelta(minutes=decision_offset),
            ).tz_convert(_ET)
            label_window_end = (
                pd.Timestamp(f"{date_str} {RTH_END.strftime('%H:%M:%S')}", tz=_ET)
                if honest_entry
                else rth_cutoff
            )
            row = {
                "event_ts": bar_ts_et,
                "date": date_str,
                "timestamp": bar_ts_et,
                "session": session_info.session,
                "decision_time": decision_time_et,
                "label_window_end": label_window_end,
                "direction": touch.direction.value,
                "representative_price": touch.representative_price,
                "level_type": touch.level_type,
                "label": outcome.label,
                "label_encoded": outcome.label_encoded,
                "max_mfe": outcome.max_mfe,
                "max_mae": outcome.max_mae,
            }
            row.update(features)

            if config.include_approach_features:
                approach = compute_approach_features_engine(touch, data_dir, symbol, config)
                if approach:
                    row.update(approach)

            rows.append(row)
    finally:
        if entry_store is not None:
            entry_store.close()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _trade_price_at(
    store: TickStore,
    symbol: str,
    as_of_utc: datetime,
    lookback_min: int = 30,
) -> float | None:
    """Most-recent front-month TRADE print price with ts_event <= as_of (decision fill).

    Mirrors ``honest_edge_audit._book_mid_at`` but on TRADE PRINTS (action='T'), since
    the production path is now on trade bars: the honest fill is the realistic trade
    price prevailing at the decision instant. Bounded lookback keeps the scan cheap.
    Reuses the front-month + book-valid filters of ``_query_trades`` (action='T').
    """
    views = store._get_views(symbol)
    if not views:
        return None
    union_sql = store._union_views_sql(views)
    sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
    cols = {r[0] for r in store._conn.execute(sample_sql).fetchall()}
    if "price" not in cols or "action" not in cols:
        return None
    front = _front_symbol(store, union_sql)
    sym_f = f"AND symbol = '{front}'" if front else "AND symbol NOT LIKE '%-%'"
    has_book = "bid_px_00" in cols and "ask_px_00" in cols
    book_f = "AND bid_px_00 > 0 AND ask_px_00 > 0" if has_book else ""
    lo = pd.Timestamp(as_of_utc) - pd.Timedelta(minutes=lookback_min)
    sql = f"""
        SELECT price
        FROM ({union_sql}) AS t
        WHERE ts_event <= $1 AND ts_event >= $2
          AND action = 'T'
          AND price IS NOT NULL AND price > 0
          {book_f}
          {sym_f}
        ORDER BY ts_event DESC LIMIT 1
    """
    row = store._conn.execute(sql, [pd.Timestamp(as_of_utc), lo]).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def compute_interaction_features_engine(
    touch,
    data_dir: Path,
    symbol: str,
    config: DashboardUtilityConfig,
    *,
    price_source: str = "trade",
    tick_size: float = TRADE_TICK,
    window_minutes: int | None = None,
) -> dict[str, float] | None:
    """3 interaction features via the engine over the post-touch interaction window.

    TRADE-BAR CUTOVER (Part 1, production DEFAULT): all 3 features re-source to
    TRADE PRINTS (``price_source="trade"`` routes ``query_tick_feature_rows`` to the
    front-month action='T' trade ``price``) on the 0.25 grid (``tick_size``). This
    SUPERSEDES the book-mid feed (price = (bid+ask)/2 at 0.125) and is why the engine
    interaction features now DIFFER from the legacy book-mid path BY DESIGN.

    Reachable book-mid regression (the phase-5 parity proof): pass
    ``price_source="book_mid", tick_size=BOOK_MID_TICK`` to feed the BOOK rows at
    0.125 exactly as before.

    CRITICAL (parity_harness_v2 Stage E): register ONLY the touch's own ET date in a
    per-date single-day store, so the window never leaks rows from the next day's
    file across a UTC midnight. The ``< 5``-row drop (builder:493) is reproduced so
    the kept-touch set is well-defined. The window is the INTERACTION window
    [touch, touch+window_minutes) — UNCHANGED by the honest-entry re-anchor (the
    re-anchor only moves the OUTCOME's entry/forward window, not the feature window).
    """
    event_ts_utc = touch.bar_ts_utc
    rep_price = float(touch.representative_price)
    if window_minutes is None:
        window_minutes = config.interaction_window_minutes

    # The canonical builder keys the store off the touch's OWN calendar date, where
    # that date is the ET-INDEXED bar_ts date (``_detect_touches`` set
    # ``touch["date"] = str(bar_ts.date())`` over an ET-indexed frame, and
    # ``_compute_interaction_features`` reads it). Derive the SAME ET date here.
    ds = str(pd.Timestamp(event_ts_utc).tz_convert(_ET).date())

    store = TickStore(data_dir)
    registered = store.register_symbol_date(symbol, ds)
    if not registered:
        store.close()
        return None

    start = event_ts_utc
    end = start + timedelta(minutes=window_minutes)
    try:
        ticks = store.query_tick_feature_rows(symbol, start, end, price_source=price_source)
    finally:
        store.close()

    # Legacy early-returns: < 5 rows or no price column -> drop the touch.
    if ticks.empty or len(ticks) < 5:
        return None
    if "price" not in ticks.columns:
        return None

    has_size = "size" in ticks.columns
    eng_trades: list[Trade] = []
    for r in ticks.itertuples(index=False):
        eng_trades.append(
            Trade(
                event_ts_utc=_to_utc_dt(r.ts_event),
                price_ticks=_round_to_ticks(r.price, tick_size),
                size=int(r.size) if has_size else 1,
            )
        )

    direction = touch.direction
    beyond = int_time_beyond_level(eng_trades, rep_price, direction, tick_size)
    within = int_time_within_2pts(eng_trades, rep_price, tick_size)
    absorp = int_absorption_ratio(eng_trades, rep_price, direction, tick_size)

    return {
        "int_time_beyond_level": beyond,
        "int_time_within_2pts": within,
        "int_absorption_ratio": absorp,
    }


def compute_approach_features_engine(
    touch,
    data_dir: Path,
    symbol: str,
    config: DashboardUtilityConfig,
) -> dict[str, float] | None:
    """The 3 RUNTIME approach features via the engine (the live-computable subset).

    Reproduces parity_harness_v2 Stage G: feed real TRADE prints + L0 quotes over
    ``[touch - approach_window, touch)`` at the 0.25 trade grid into
    ``app_avg_trade_size`` / ``app_large_trade_vol_pct`` / ``app_max_spread``. Only
    the 3 features the engine implements as scalar functions are returned; the
    other LIVE_APPROACH_FEATURES (acceleration, imbalance, volatility) have no
    engine formula yet and are flagged in the report (Part 2 territory).
    """
    event_ts_utc = touch.bar_ts_utc
    approach_minutes = config.approach_window_minutes
    approach_start = event_ts_utc - timedelta(minutes=approach_minutes)

    # ET-indexed touch date (same as legacy ``_compute_approach_features``), plus
    # the prior calendar day so the approach window can cross midnight.
    et_date = pd.Timestamp(event_ts_utc).tz_convert(_ET).date()
    ds = et_date.isoformat()
    prev_ds = (et_date - timedelta(days=1)).isoformat()

    store = TickStore(data_dir)
    registered = store.register_symbol_date(symbol, ds)
    store.register_symbol_date(symbol, prev_ds)
    if not registered:
        store.close()
        return None

    try:
        trades_df = _query_trades(store, symbol, approach_start, event_ts_utc)
        quotes_df = _query_quotes(store, symbol, approach_start, event_ts_utc)
    finally:
        store.close()

    eng_trades = [
        Trade(
            event_ts_utc=_to_utc_dt(r.ts_event),
            price_ticks=_round_to_ticks(r.price, TRADE_TICK),
            size=int(r.size),
        )
        for r in trades_df.itertuples(index=False)
    ]
    eng_quotes = [
        Quote(
            event_ts_utc=_to_utc_dt(r.ts_event),
            bid_price_ticks=_round_to_ticks(r.bid_px_00, TRADE_TICK),
            ask_price_ticks=_round_to_ticks(r.ask_px_00, TRADE_TICK),
        )
        for r in quotes_df.itertuples(index=False)
    ]

    out = {
        "app_avg_trade_size": app_avg_trade_size(eng_trades),
        "app_large_trade_vol_pct": app_large_trade_vol_pct(eng_trades),
        "app_max_spread": app_max_spread(eng_quotes, TRADE_TICK),
    }
    # Keep only LIVE_APPROACH_FEATURES (the 3 engine-backed ones are a subset).
    return {k: v for k, v in out.items() if k in LIVE_APPROACH_FEATURES}


# ── DuckDB trade / quote extraction (front-month aware), per parity harness ──


def _front_symbol(store: TickStore, union_sql: str) -> str | None:
    sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
    cols = {r[0] for r in store._conn.execute(sample_sql).fetchall()}
    if "symbol" not in cols:
        return None
    front = store._conn.execute(
        f"SELECT symbol, count(*) AS n FROM ({union_sql}) AS t "
        f"WHERE symbol NOT LIKE '%-%' GROUP BY symbol ORDER BY n DESC LIMIT 1"
    ).fetchone()
    return front[0] if front else None


def _query_trades(store: TickStore, symbol, start_utc, end_utc) -> pd.DataFrame:
    """Front-month TRADE prints over ``[start, end)`` (approach window is end-exclusive)."""
    views = store._get_views(symbol)
    if not views:
        return pd.DataFrame(columns=["ts_event", "price", "size"])
    union_sql = store._union_views_sql(views)
    front = _front_symbol(store, union_sql)
    sym_f = f"AND symbol = '{front}'" if front else "AND symbol NOT LIKE '%-%'"
    sql = f"""
        SELECT ts_event, price, size
        FROM ({union_sql}) AS t
        WHERE ts_event >= $1 AND ts_event < $2
          AND bid_px_00 > 0 AND ask_px_00 > 0
          AND action = 'T'
          {sym_f}
        ORDER BY ts_event ASC
    """
    return store._conn.execute(sql, [pd.Timestamp(start_utc), pd.Timestamp(end_utc)]).fetchdf()


def _query_quotes(store: TickStore, symbol, start_utc, end_utc) -> pd.DataFrame:
    """Front-month L0 quotes over ``[start, end)`` for the max-spread feature."""
    views = store._get_views(symbol)
    if not views:
        return pd.DataFrame(columns=["ts_event", "bid_px_00", "ask_px_00"])
    union_sql = store._union_views_sql(views)
    front = _front_symbol(store, union_sql)
    sym_f = f"AND symbol = '{front}'" if front else "AND symbol NOT LIKE '%-%'"
    sql = f"""
        SELECT ts_event, bid_px_00, ask_px_00
        FROM ({union_sql}) AS t
        WHERE ts_event >= $1 AND ts_event < $2
          AND bid_px_00 > 0 AND ask_px_00 > 0
          {sym_f}
        ORDER BY ts_event ASC
    """
    return store._conn.execute(sql, [pd.Timestamp(start_utc), pd.Timestamp(end_utc)]).fetchdf()
