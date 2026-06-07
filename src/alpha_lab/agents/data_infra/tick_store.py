"""
DuckDB-backed tick data store for backtesting and replay.

Provides time-bounded queries over date/symbol-partitioned Parquet files
with strict look-ahead bias prevention.  All query methods enforce a hard
``end`` boundary — no data beyond that timestamp is ever accessible.

Usage::

    store = TickStore(Path("data/databento"))
    store.register_date_range("NQ", date(2026, 2, 18), date(2026, 2, 20))
    df = store.query_ticks("NQ", start, end)            # strict [start, end]
    for batch in store.replay("NQ", start, end, step):   # chronological
        process(batch)
    store.close()
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Default Parquet sub-path pattern: {data_dir}/{symbol}/{date}/mbp10.parquet
_TICK_FILENAME = "mbp10.parquet"
_OHLCV_FILENAME_PATTERN = "ohlcv_{tf}.parquet"

# Tick filenames to search for, in priority order
_TICK_FILENAMES = ["mbp10.parquet", "mbp1.parquet", "trades.parquet"]

# ── Shared tick-bar spec (phase 4c/4d: parity-locked with strategy_core) ───────
# 18:00 ET (DST-aware) trading-day boundary as SQL. ts_event is a TIMESTAMPTZ (UTC);
# AT TIME ZONE 'America/New_York' yields the DST-correct NY wall clock, +6h rolls the
# date exactly at 18:00 ET, and CAST AS DATE labels the trading day. This matches
# strategy_core.decisions.sessions.trading_day_for (local.time() >= 18:00 -> next day)
# instant-for-instant. Replaces the prior bug where a NAIVE python datetime bound was
# cast against ts_event using the DuckDB session TimeZone (America/Chicago) -> a 23:00
# CT window artifact.
_SQL_TRADING_DAY = "CAST((ts_event AT TIME ZONE 'America/New_York') + INTERVAL 6 HOUR AS DATE)"
# Bar-price source (phase 4d): production tick bars are TRADE-PRICE tick bars -- a "tick"
# is a TRADE print (action='T'), not a book event. ``price_source='trade'`` filters to
# trades and builds OHLC from the trade ``price`` (the NQ trade grid is 0.25, lossless).
# This supersedes the phase-4c book-mid bars (``price_source='book_mid'``: every book-valid
# event, OHLC from (bid+ask)/2 on the 0.125 grid), which is kept reachable for comparison
# and the documented reversal path. ``query_tick_feature_rows`` (the decision-layer feature
# stream) is intentionally NOT touched here -- 4d switches only the BAR definition.
#
# Intrinsic, reader-independent total order. ts_event + `sequence` is NOT a total order:
# databento emits MULTIPLE records under one (ts_event, sequence). For TRADE bars those are
# the prints of a single matching event -- a sweep that lifts/hits successive levels at
# MONOTONIC prices. Phase 4e orders them by SIDE-SIGNED price (+price for a BUY aggressor,
# -price for a SELL aggressor) then size, so ORDER BY ASC reproduces the true wire direction
# (buy sweeps print ascending as they lift asks, sell sweeps descending as they hit bids).
# Phase 4d's plain price-ASC key got buy sweeps right but REVERSED sell sweeps, diverging from
# Trade-Lab's wire-order bars on ~14% of bars; the side-signed key removes that. VERIFIED
# encoding (two independent empirical checks, 0 cross-contamination): databento side='B' = BUY
# (above mid, ascending), side='A' = SELL (below mid, descending). For book bars the same
# (ts_event,sequence) carried a 'T' plus its 'C'/'A'/'M' book consequences at DIFFERENT,
# non-monotonic mids, so that key appends the mid determinants (bid_px_00, ask_px_00). Either
# way every record whose bar-price/direction differs is deterministically ordered and any
# residual tie is bar-irrelevant; the key is composed of fields IN the data, so it is identical
# across DuckDB and pandas readers regardless of physical read order. The concrete order is
# built per-source in ``_tick_event_selection``.
#
# Phase 4f -- the DuckDB us/ns ordering seam (TRADE path only). DuckDB 1.4.4 reads a parquet
# ``timestamp[ns, tz=UTC]`` column as a microsecond TIMESTAMP WITH TIME ZONE (TIMESTAMPTZ is
# inherently us and cannot hold tz-aware ns: read_parquet AND an Arrow scan both truncate it).
# The streaming engine, fed the raw parquet via pandas, keeps ns. So for same-microsecond /
# distinct-nanosecond trades the order key ``ts_event, sequence, side_signed_price, size`` lost
# its leading tie-break under DuckDB, reordering ~4 bars/day on a ties-heavy day (07-11) -- a
# VOLUME-ONLY shuffle (same-price multi-fills move across a bucket boundary; OHLC unchanged).
# FIX (trade path only): read the trade columns via pyarrow and append a BIGINT nanosecond key
# ``ts_event_ns = cast(cast(ts_event, timestamp[ns,UTC]), int64())`` (the double-cast normalizes
# any input resolution to ns then to raw ns-since-epoch). Register that Arrow table as the DuckDB
# relation and make the ORDER key LEAD with ``ts_event_ns`` (BIGINT). This is ORDERING-ONLY: the
# us ``ts_event`` TIMESTAMPTZ is still used for the 18:00-ET trading_day cast, the [start,end)
# window filter, and the open_time/bar_time OUTPUT columns -- output schema/types/values are
# unchanged, so the supporting parity test and the production gate keep working. ``book_mid`` is
# left on the plain read_parquet view (us), byte-unchanged, so ``_build_bars_for_date`` and the
# pinned decision layer carry zero regression risk.


class TickStore:
    """DuckDB-backed query and replay layer over partitioned Parquet files.

    Look-ahead bias prevention contract
    ------------------------------------
    * All query methods accept ``end`` as a **hard wall** — no row with
      ``ts_event > end`` is ever returned.
    * ``replay()`` yields batches where each batch contains only data
      up to the current replay timestamp.
    * Results are always sorted by ``ts_event ASC``.
    """

    def __init__(
        self,
        data_dir: Path | str,
        read_only: bool = True,
        tick_filename: str | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._conn = duckdb.connect(database=":memory:", read_only=False)
        self._registered: dict[str, list[str]] = {}  # symbol -> [date_str, ...]
        self._read_only = read_only
        self._tick_filename = tick_filename  # None = auto-detect
        # Phase 4f: cache of ns-keyed Arrow relations registered with DuckDB for the
        # trade path (keyed by the registered date set), so repeated trade-bar/event
        # queries don't re-read the parquet. ``_arrow_ns_tables`` keeps a strong ref to
        # the Arrow tables so DuckDB's zero-copy registration stays valid.
        self._arrow_ns_relname: dict[str, str] = {}  # cache_key -> registered name
        self._arrow_ns_tables: dict[str, pa.Table] = {}  # cache_key -> Arrow table (keepalive)
        logger.info("TickStore opened (data_dir=%s)", self._data_dir)

    # ── Registration ──────────────────────────────────────────────

    def register_symbol_date(self, symbol: str, dt: date | str) -> bool:
        """Register a single Parquet file as a DuckDB view.

        Returns True if the file was found and registered.
        """
        date_str = dt.isoformat() if isinstance(dt, date) else str(dt)
        parquet_path = self._resolve_tick_path(symbol, date_str)

        if parquet_path is None:
            logger.debug("No tick parquet for %s/%s", symbol, date_str)
            return False

        view_name = self._view_name(symbol, date_str)
        self._conn.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_parquet('{parquet_path.as_posix()}')"
        )

        self._registered.setdefault(symbol, [])
        if date_str not in self._registered[symbol]:
            self._registered[symbol].append(date_str)

        logger.debug("Registered %s", view_name)
        return True

    def register_date_range(self, symbol: str, start: date, end: date) -> int:
        """Register all available dates in [start, end] for a symbol.

        Returns the number of dates successfully registered.
        """
        count = 0
        current = start
        while current <= end:
            if self.register_symbol_date(symbol, current):
                count += 1
            current += timedelta(days=1)
        logger.info(
            "Registered %d dates for %s (%s to %s)",
            count,
            symbol,
            start,
            end,
        )
        return count

    # ── Queries ───────────────────────────────────────────────────

    def query_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return ticks strictly within [start, end], ordered by ts_event.

        **No data after ``end`` is ever accessible.**
        """
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame()

        union_sql = self._union_views_sql(views)

        # Detect whether data has a symbol column (real Databento data does,
        # synthetic test data may not)
        sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        cols = {r[0] for r in self._conn.execute(sample_sql).fetchall()}
        has_symbol = "symbol" in cols

        # Keep only the most liquid (front-month) outright contract
        if has_symbol:
            front = self._conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            sym_filter = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
        else:
            sym_filter = ""

        sql = (
            f"SELECT * FROM ({union_sql}) AS t "
            f"WHERE ts_event >= $1 AND ts_event <= $2 "
            f"{sym_filter} "
            f"ORDER BY ts_event ASC"
        )
        return self._conn.execute(sql, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()

    def query_tick_prices(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return lean tick data (ts_event, price, size) within [start, end].

        For MBP order-book data, ``price`` is the top-of-book mid-price
        ``(bid_px_00 + ask_px_00) / 2``.  Only the front-month contract is
        included.  This is **~24x faster** than :meth:`query_ticks` because
        it fetches 3 columns instead of 72.
        """
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame(columns=["ts_event", "price", "size"])

        union_sql = self._union_views_sql(views)

        # Detect columns
        sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        cols = {r[0] for r in self._conn.execute(sample_sql).fetchall()}
        has_book = "bid_px_00" in cols and "ask_px_00" in cols
        has_symbol = "symbol" in cols

        # Front-month filter
        if has_symbol:
            front = self._conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            sym_f = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
        else:
            sym_f = ""

        if has_book:
            sql = f"""
                SELECT ts_event,
                       (bid_px_00 + ask_px_00) / 2.0 AS price,
                       size
                FROM ({union_sql}) AS t
                WHERE ts_event >= $1 AND ts_event <= $2
                  AND bid_px_00 > 0 AND ask_px_00 > 0
                  {sym_f}
                ORDER BY ts_event ASC
            """
        else:
            sql = f"""
                SELECT ts_event, price, size
                FROM ({union_sql}) AS t
                WHERE ts_event >= $1 AND ts_event <= $2
                  AND price IS NOT NULL AND price > 0
                  {sym_f}
                ORDER BY ts_event ASC
            """
        return self._conn.execute(sql, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()

    def query_tick_feature_rows(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        price_source: str = "book_mid",
    ) -> pd.DataFrame:
        """Return ML feature rows with MBP-10 depth columns when available.

        This query is intended for ML feature extraction where we need more
        than lean (ts_event, price, size) data, but still avoid loading every
        raw column from the parquet source.

        Args:
            price_source: ``"book_mid"`` (DEFAULT, legacy) sets ``price`` to the
                top-of-book mid ``(bid_px_00+ask_px_00)/2`` over book-valid events —
                the original behavior all existing callers rely on, preserved as the
                default. ``"trade"`` (trade-bar cutover, Part 1) sets ``price`` to
                the TRADE PRINT ``price`` filtered to ``action='T'`` — the front-month
                trade stream the engine's 3 interaction features now consume on the
                0.25 grid. Depth columns are still projected when present (the
                interaction features ignore them; the column shape is unchanged).
        """
        if price_source not in ("book_mid", "trade"):
            raise ValueError(
                f"unknown price_source {price_source!r}; expected 'book_mid' or 'trade'"
            )
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame(columns=["ts_event", "price", "size"])

        union_sql = self._union_views_sql(views)
        sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        cols = {r[0] for r in self._conn.execute(sample_sql).fetchall()}
        has_symbol = "symbol" in cols
        has_book = "bid_px_00" in cols and "ask_px_00" in cols
        has_size = "size" in cols

        if has_symbol:
            front = self._conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            sym_f = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
        else:
            sym_f = ""

        has_action = "action" in cols

        select_cols = ["ts_event"]
        if has_book:
            # price_source routes the ``price`` column: book_mid (legacy default) is
            # the top-of-book mid; trade routes to the TRADE PRINT price and filters
            # to action='T' (the front-month trade stream the engine's interaction
            # features now consume on the 0.25 grid).
            if price_source == "trade":
                select_cols.append("price")
                trade_filter = "AND action = 'T'" if has_action else ""
            else:
                select_cols.append("(bid_px_00 + ask_px_00) / 2.0 AS price")
                trade_filter = ""
            select_cols.append("size" if has_size else "1.0 AS size")
            # Support variable-depth schemas (e.g. mbp1 vs mbp10).
            depth_levels = [
                i
                for i in range(10)
                if (
                    f"bid_px_{i:02d}" in cols
                    and f"ask_px_{i:02d}" in cols
                    and f"bid_sz_{i:02d}" in cols
                    and f"ask_sz_{i:02d}" in cols
                )
            ]
            for i in depth_levels:
                select_cols.extend(
                    [
                        f"bid_px_{i:02d}",
                        f"ask_px_{i:02d}",
                        f"bid_sz_{i:02d}",
                        f"ask_sz_{i:02d}",
                    ]
                )
            sql = f"""
                SELECT {", ".join(select_cols)}
                FROM ({union_sql}) AS t
                WHERE ts_event >= $1 AND ts_event <= $2
                  AND bid_px_00 > 0 AND ask_px_00 > 0
                  {trade_filter}
                  {sym_f}
                ORDER BY ts_event ASC
            """
        else:
            # Fallback for trades-only schemas.
            select_cols.append("price")
            select_cols.append("size" if has_size else "1.0 AS size")
            sql = f"""
                SELECT {", ".join(select_cols)}
                FROM ({union_sql}) AS t
                WHERE ts_event >= $1 AND ts_event <= $2
                  AND price IS NOT NULL AND price > 0
                  {sym_f}
                ORDER BY ts_event ASC
            """

        return self._conn.execute(sql, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()

    def query_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return pre-computed OHLCV bars within [start, end].

        Reads from cached ``ohlcv_{tf}.parquet`` files.
        """
        frames: list[pd.DataFrame] = []
        dates = self._registered.get(symbol, [])

        for date_str in sorted(dates):
            path = self._data_dir / symbol / date_str / f"ohlcv_{timeframe}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()

        # Enforce time bounds
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        if hasattr(df.index, "tz") and df.index.tz is not None:
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize(df.index.tz)
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize(df.index.tz)
        return df[(df.index >= start_ts) & (df.index <= end_ts)]

    def build_bars_from_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        bar_size: str = "5 minutes",
    ) -> pd.DataFrame:
        """Aggregate ticks into OHLCV bars using DuckDB time_bucket.

        Args:
            symbol: Instrument symbol
            start: Start time (inclusive)
            end: End time (inclusive) — hard boundary
            bar_size: DuckDB interval string, e.g. "1 minute", "5 minutes"

        Returns:
            DataFrame with [open, high, low, close, volume] and
            DatetimeIndex.
        """
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        union_sql = self._union_views_sql(views)

        # Detect whether this is MBP (order-book) data with bid/ask columns.
        # If so, build OHLCV from top-of-book mid-price to avoid extreme
        # wicks caused by deep book levels (bid_px_09 / ask_px_09).
        sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        cols = {r[0] for r in self._conn.execute(sample_sql).fetchall()}
        has_book = "bid_px_00" in cols and "ask_px_00" in cols

        # Filter out calendar-spread symbols (e.g. "NQZ5-NQH6") and
        # back-month contracts whose different price levels corrupt OHLCV.
        # Keep only the most liquid (front-month) outright contract.
        has_symbol = "symbol" in cols
        if has_symbol:
            front = self._conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            symbol_filter = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
        else:
            symbol_filter = ""

        if has_book:
            # Mid-price from top-of-book gives clean OHLCV
            sql = f"""
                SELECT
                    time_bucket(INTERVAL '{bar_size}', ts_event) AS bar_time,
                    first(mid ORDER BY ts_event) AS open,
                    max(mid) AS high,
                    min(mid) AS low,
                    last(mid ORDER BY ts_event) AS close,
                    sum(size) AS volume
                FROM (
                    SELECT ts_event, size,
                           (bid_px_00 + ask_px_00) / 2.0 AS mid
                    FROM ({union_sql}) AS t
                    WHERE ts_event >= $1 AND ts_event <= $2
                      AND bid_px_00 > 0 AND ask_px_00 > 0
                      {symbol_filter}
                ) AS m
                GROUP BY bar_time
                ORDER BY bar_time ASC
            """
        else:
            # Fallback for trades-only data: use raw price
            sql = f"""
                SELECT
                    time_bucket(INTERVAL '{bar_size}', ts_event) AS bar_time,
                    first(price ORDER BY ts_event) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price ORDER BY ts_event) AS close,
                    sum(size) AS volume
                FROM ({union_sql}) AS t
                WHERE ts_event >= $1 AND ts_event <= $2
                  AND price IS NOT NULL AND price > 0
                  {symbol_filter}
                GROUP BY bar_time
                ORDER BY bar_time ASC
            """
        df = self._conn.execute(sql, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = df.set_index("bar_time")
        df.index.name = "timestamp"
        return df

    def build_tick_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tick_count: int = 987,
        *,
        price_source: str = "trade",
    ) -> pd.DataFrame:
        """Aggregate ticks into count-based OHLCV bars.

        Groups every ``tick_count`` ticks into one bar using
        ROW_NUMBER-based bucketing in DuckDB.

        Args:
            symbol: Instrument symbol
            start: Start time (inclusive)
            end: End time (inclusive) — hard boundary
            tick_count: Number of ticks per bar (e.g. 147, 987, 2000)
            price_source: ``"trade"`` (default, production) builds TRADE-PRICE tick
                bars -- a tick is a trade print (action='T'), OHLC from the trade
                ``price`` on the 0.25 grid. ``"book_mid"`` is the legacy phase-4c
                spec (a tick is any book-valid event, OHLC from (bid+ask)/2 on the
                0.125 grid), kept for comparison / the reversal path.

        Returns:
            DataFrame with [open, high, low, close, volume] and
            DatetimeIndex.
        """
        sel = self._tick_event_selection(symbol, price_source=price_source)
        if sel is None:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        union_sql, where_clause, price_expr, ev_extra, order_clause = sel

        # Bars are bucketed PER 18:00-ET trading day (PARTITION BY trading_day) so
        # bar_index restarts at each 18:00-ET open and Sunday-evening Globex groups
        # into the correct (Monday) trading day. Ordering is the deterministic,
        # reader-independent composite ``order_clause`` (ts_event, sequence + mid
        # determinants + size); ROW_NUMBER and every FIRST/LAST use it, so the bucket
        # boundaries and open/close are fully determined. The trailing <tick_count
        # bucket of a day is KEPT (no HAVING) as an incomplete bar -- matching the
        # streaming engine's END_OF_DAY partial, which is what makes the two builders
        # byte-identical.
        sql = f"""
            WITH ev AS (
                SELECT
                    ts_event,
                    "sequence",
                    size,
                    {ev_extra},
                    {price_expr} AS px,
                    {_SQL_TRADING_DAY} AS trading_day
                FROM ({union_sql}) AS t
                {where_clause}
            ),
            numbered AS (
                SELECT *,
                    CAST(ROW_NUMBER() OVER (
                        PARTITION BY trading_day ORDER BY {order_clause}
                    ) - 1 AS BIGINT) AS rn
                FROM ev
            )
            SELECT
                trading_day,
                (rn // {tick_count}) AS bar_index,
                FIRST(ts_event ORDER BY {order_clause}) AS open_time,
                LAST(ts_event ORDER BY {order_clause}) AS bar_time,
                FIRST(px ORDER BY {order_clause}) AS open,
                MAX(px) AS high,
                MIN(px) AS low,
                LAST(px ORDER BY {order_clause}) AS close,
                SUM(size) AS volume,
                COUNT(*) AS trade_count
            FROM numbered
            GROUP BY trading_day, bar_index
            ORDER BY trading_day, bar_index
        """
        df = self._conn.execute(sql, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()

        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df["is_complete"] = df["trade_count"] == tick_count
        df = df.set_index("bar_time")
        df.index.name = "timestamp"
        return df

    def _tick_event_selection(
        self, symbol: str, price_source: str = "trade"
    ) -> tuple[str, str, str, str, str] | None:
        """Resolve the SQL fragments shared by ``build_tick_bars`` and
        ``query_tick_events`` so both consume the IDENTICAL filtered, IDENTICALLY
        ORDERED event set: ``(union_sql, where_clause, price_expr, ev_extra_cols,
        order_clause)``.

        * ``price_source`` selects the bar definition (see the module spec note):
          ``"trade"`` (default, production) keeps only trade prints (``action='T'``)
          and prices them by the trade ``price``; ``"book_mid"`` (legacy phase-4c)
          keeps every book-valid event and prices it by ``(bid+ask)/2``.
        * ``where_clause`` is half-open ``[start, end)`` on bound params ``$1``/``$2``
          (pass tz-aware UTC 18:00-ET bounds).
        * ``order_clause`` is the deterministic, reader-independent total order:
          ``(ts_event, sequence, <bar-price determinants>, size)`` -- enough that every
          record whose bar price differs is deterministically ordered, so the two
          builders agree byte-for-byte regardless of physical read order.
        * ``ev_extra_cols`` are the extra raw columns the order needs carried into the
          bucketing CTE.

        Requires the intrinsic ``sequence`` column; raises if absent.
        """
        views = self._get_views(symbol)
        if not views:
            return None
        union_sql = self._union_views_sql(views)
        cols = {
            r[0]
            for r in self._conn.execute(
                f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
            ).fetchall()
        }
        # Phase 4f: only the TRADE path needs the ns ordering key. The columns it may
        # reference are ts_event (filter/output + ns key), sequence/size (order/agg),
        # symbol (front-month filter), action (trade filter), price (bar price), side
        # (side-signed order). Read just those that exist; ``_ns_keyed_relation`` adds
        # ts_event_ns. book_mid stays on the plain read_parquet view (us, unchanged).
        trade_ns_cols = {
            "ts_event",
            "sequence",
            "size",
            "symbol",
            "action",
            "price",
            "side",
        }
        if "sequence" not in cols:
            raise ValueError(
                "tick parquet lacks the intrinsic 'sequence' column required for "
                "reader-independent deterministic tick bars"
            )
        has_book = "bid_px_00" in cols and "ask_px_00" in cols
        has_symbol = "symbol" in cols
        has_action = "action" in cols
        has_price = "price" in cols
        has_side = "side" in cols

        if has_symbol:
            front = self._conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            sym_f = f"AND symbol = '{front[0]}'" if front else "AND symbol NOT LIKE '%-%'"
        else:
            sym_f = ""

        if price_source == "trade":
            # TRADE-PRICE tick bars: a tick is a trade print. Keep only action='T'
            # (case-insensitive; a trades-only file with no 'action' column is already
            # all trades) and build OHLC from the trade ``price``.
            if not has_price:
                raise ValueError(
                    "trade-price tick bars require a 'price' column on the tick parquet"
                )
            # Phase 4f: swap the plain read_parquet union (us-truncated ts_event) for an
            # ns-keyed Arrow relation so the order key keeps its nanosecond tie-break. The
            # relation carries every column the trade path uses PLUS BIGINT ts_event_ns;
            # ts_event itself is unchanged (still us TIMESTAMPTZ for the trading_day cast,
            # the window filter, and the open_time/bar_time outputs). The order key below
            # LEADS with ts_event_ns instead of ts_event -- ordering-only.
            ns_rel = self._ns_keyed_relation(symbol, trade_ns_cols)
            if ns_rel is not None:
                union_sql = f"SELECT * FROM {ns_rel}"
                ts_order_key = "ts_event_ns"
                ns_carry = "ts_event_ns, "
            else:
                # No resolvable parquet files (in-memory-only views): fall back to the
                # us-truncated ts_event order. Bars are still deterministic; this only
                # affects the rare same-us/distinct-ns tie that needs the real files.
                ts_order_key = "ts_event"
                ns_carry = ""
            price_expr = "price"
            trade_filt = "AND lower(CAST(action AS VARCHAR)) = 't'" if has_action else ""
            filt = f"{trade_filt} AND price IS NOT NULL AND price > 0"
            if has_side:
                # Phase 4e -- SIDE-SIGNED price order. Within a (ts_event, sequence) matching
                # event the trade prints are a monotonic sweep; ordering by raw price ASC
                # (phase 4d) reproduced wire order for BUY sweeps but REVERSED SELL sweeps
                # (~half), diverging from Trade-Lab's wire-order bars on ~14% of bars. Signing
                # the price by aggressor side makes ORDER BY ASC reproduce the true wire
                # direction (buys ascending, sells descending) for both.
                #
                # VERIFIED encoding (empirically pinned on real NQ front-month trades, two
                # independent checks agreeing with 0 cross-contamination): databento
                # ``side='B'`` is the BUY aggressor (lifts ascending asks; prints above mid)
                # and ``side='A'`` is the SELL aggressor (hits descending bids; prints below
                # mid). So +price for 'B', -price for everything else (sells; 'N' never occurs
                # on trades). Mixed-side (ts_event, sequence) groups (~20-32/day) fall back to
                # this same per-row rule deterministically.
                ev_extra = f"{ns_carry}price, side"
                signed_price = (
                    "CASE WHEN lower(CAST(side AS VARCHAR)) = 'b' THEN price ELSE -price END"
                )
                order_clause = f'{ts_order_key}, "sequence", {signed_price}, size'
            else:
                # No aggressor side available -> fall back to raw price (4d behavior).
                ev_extra = f"{ns_carry}price"
                order_clause = f'{ts_order_key}, "sequence", price, size'
        elif price_source == "book_mid":
            # Legacy phase-4c book-mid bars (reversal / comparison path).
            if has_book:
                price_expr = "(bid_px_00 + ask_px_00) / 2.0"
                filt = "AND bid_px_00 > 0 AND ask_px_00 > 0"
                ev_extra = "bid_px_00, ask_px_00"
                order_clause = 'ts_event, "sequence", bid_px_00, ask_px_00, size'
            else:
                price_expr = "price"
                filt = "AND price IS NOT NULL AND price > 0"
                ev_extra = "price"
                order_clause = 'ts_event, "sequence", price, size'
        else:
            raise ValueError(
                f"unknown price_source {price_source!r}; expected 'trade' or 'book_mid'"
            )

        where_clause = f"WHERE ts_event >= $1 AND ts_event < $2 {filt} {sym_f}"
        return union_sql, where_clause, price_expr, ev_extra, order_clause

    def query_tick_events(
        self, symbol: str, start: datetime, end: datetime, *, price_source: str = "trade"
    ) -> pd.DataFrame:
        """Return the EXACT ordered event stream ``build_tick_bars`` buckets.

        Columns ``[ts_event, price, size]`` over ``[start, end)``, filtered identically
        to ``build_tick_bars`` for the same ``price_source`` (default ``"trade"``:
        front-month trade prints; ``"book_mid"``: front-month book-valid events) and
        ordered by the SAME deterministic composite ``order_clause`` (for trades, the
        phase-4e SIDE-SIGNED price order). ``price`` is the bar price (the trade price,
        or the book mid). Feeding these prints to the streaming ``CandleEngine``
        reproduces ``build_tick_bars``' bars byte-for-byte; this is the hook the standing
        DuckDB<->streaming parity test uses.

        NOTE: the order is POSITIONAL -- the side-signed direction is already baked into
        the row order here, so ``side`` is intentionally NOT returned. Consumers must
        preserve this row order (do not re-sort) for the bars to match ``build_tick_bars``.
        """
        sel = self._tick_event_selection(symbol, price_source=price_source)
        if sel is None:
            return pd.DataFrame(columns=["ts_event", "price", "size"])
        union_sql, where_clause, price_expr, _ev_extra, order_clause = sel
        sql = f"""
            SELECT ts_event, {price_expr} AS price, size
            FROM ({union_sql}) AS t
            {where_clause}
            ORDER BY {order_clause}
        """
        return self._conn.execute(sql, [pd.Timestamp(start), pd.Timestamp(end)]).fetchdf()

    # ── Replay iterator ───────────────────────────────────────────

    def replay(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        step: timedelta = timedelta(minutes=1),
    ) -> Generator[pd.DataFrame]:
        """Yield tick batches in strict chronological order.

        Each yielded DataFrame contains only data in
        [current_time, current_time + step).  ``current_time`` advances
        by ``step`` on each iteration.

        **No future data is ever exposed within any single batch.**
        """
        current = start
        while current < end:
            batch_end = min(current + step, end)
            batch = self.query_ticks(symbol, current, batch_end)
            if not batch.empty:
                yield batch
            current = batch_end

    # ── Book snapshot ─────────────────────────────────────────────

    def get_book_snapshot(self, symbol: str, as_of: datetime) -> pd.DataFrame:
        """Return the order book state (10 levels) at exact timestamp.

        Finds the most recent tick record at or before ``as_of`` and
        extracts bid/ask levels from it.
        """
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame()

        union_sql = self._union_views_sql(views)
        sql = (
            f"SELECT * FROM ({union_sql}) AS t WHERE ts_event <= $1 ORDER BY ts_event DESC LIMIT 1"
        )
        df = self._conn.execute(sql, [pd.Timestamp(as_of)]).fetchdf()
        if df.empty:
            return pd.DataFrame()

        row = df.iloc[0]
        levels = []
        for i in range(10):
            bid_px_col = f"bid_px_{i:02d}"
            ask_px_col = f"ask_px_{i:02d}"
            bid_sz_col = f"bid_sz_{i:02d}"
            ask_sz_col = f"ask_sz_{i:02d}"
            level = {"depth": i}
            if bid_px_col in row.index:
                level["bid_px"] = row[bid_px_col]
            if ask_px_col in row.index:
                level["ask_px"] = row[ask_px_col]
            if bid_sz_col in row.index:
                level["bid_sz"] = row[bid_sz_col]
            if ask_sz_col in row.index:
                level["ask_sz"] = row[ask_sz_col]
            if len(level) > 1:
                levels.append(level)

        return pd.DataFrame(levels) if levels else df

    # ── Lifecycle ─────────────────────────────────────────────────

    def close(self) -> None:
        """Close DuckDB connection."""
        self._conn.close()
        logger.info("TickStore closed")

    # ── Private helpers ───────────────────────────────────────────

    def _resolve_tick_path(self, symbol: str, date_str: str) -> Path | None:
        """Find the tick parquet file for a given symbol/date.

        If tick_filename was set explicitly, use that. Otherwise search
        in priority order: mbp10 > mbp1 > trades.
        """
        date_dir = self._data_dir / symbol / date_str
        if self._tick_filename:
            p = date_dir / self._tick_filename
            return p if p.exists() else None
        for fname in _TICK_FILENAMES:
            p = date_dir / fname
            if p.exists():
                return p
        return None

    @staticmethod
    def _view_name(symbol: str, date_str: str) -> str:
        return f"ticks_{symbol}_{date_str.replace('-', '_')}"

    def _get_views(self, symbol: str) -> list[str]:
        dates = self._registered.get(symbol, [])
        return [self._view_name(symbol, d) for d in sorted(dates)]

    @staticmethod
    def _union_views_sql(views: list[str]) -> str:
        return " UNION ALL ".join(f"SELECT * FROM {v}" for v in views)

    # ── Phase 4f: ns-keyed Arrow relation for the TRADE-bar ordering ──────────
    def _ns_keyed_relation(self, symbol: str, needed_cols: set[str]) -> str | None:
        """Register (and cache) a DuckDB relation for ``symbol`` whose rows carry a
        BIGINT ``ts_event_ns`` ordering key, so the trade-bar order key keeps the
        nanosecond tie-break DuckDB's microsecond TIMESTAMPTZ would otherwise truncate.

        Reads only the columns the trade path references (``needed_cols`` ∩ schema)
        via pyarrow -- matching the footprint of the existing pandas/gate reads --
        then appends ``ts_event_ns = cast(cast(ts_event, ts[ns,UTC]), int64())``
        (the double-cast normalizes any input resolution to raw ns-since-epoch; a
        ``timestamp[us]`` fixture stays correct because its ns == us). The Arrow table
        is registered as a DuckDB relation and a strong ref is kept alive. Returns the
        relation name, or ``None`` if no parquet files are registered for ``symbol``.

        NOTE: ``ts_event`` is carried through UNCHANGED (still the native parquet
        TIMESTAMPTZ); only an EXTRA ordering column is added. The trading_day cast,
        the [start,end) window filter, and the open_time/bar_time outputs keep using
        ``ts_event`` at microsecond resolution -- the fix is ordering-only.
        """
        dates = sorted(self._registered.get(symbol, []))
        paths = [self._resolve_tick_path(symbol, d) for d in dates]
        paths = [p for p in paths if p is not None]
        if not paths:
            return None
        cache_key = "\n".join(p.as_posix() for p in paths)
        cached = self._arrow_ns_relname.get(cache_key)
        if cached is not None:
            return cached

        tables: list[pa.Table] = []
        for p in paths:
            schema_names = set(pq.read_schema(p).names)
            read_cols = [c for c in needed_cols if c in schema_names]
            if "ts_event" not in read_cols:
                read_cols.append("ts_event")
            t = pq.read_table(p, columns=read_cols)
            ns = pc.cast(
                pc.cast(t.column("ts_event"), pa.timestamp("ns", "UTC")),
                pa.int64(),
            )
            tables.append(t.append_column("ts_event_ns", ns))
        table = (
            tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote_options="default")
        )

        relname = f"arrow_ns_{symbol}_{abs(hash(cache_key)) & 0xFFFFFFFF:08x}"
        self._conn.register(relname, table)
        self._arrow_ns_tables[cache_key] = table  # keepalive for zero-copy scan
        self._arrow_ns_relname[cache_key] = relname
        logger.debug("Registered ns-keyed trade relation %s (%d rows)", relname, table.num_rows)
        return relname
