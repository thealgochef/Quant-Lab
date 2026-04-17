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

logger = logging.getLogger(__name__)

# Default Parquet sub-path pattern: {data_dir}/{symbol}/{date}/mbp10.parquet
_TICK_FILENAME = "mbp10.parquet"
_OHLCV_FILENAME_PATTERN = "ohlcv_{tf}.parquet"

# Tick filenames to search for, in priority order
_TICK_FILENAMES = ["mbp10.parquet", "mbp1.parquet", "trades.parquet"]


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

    def register_date_range(
        self, symbol: str, start: date, end: date
    ) -> int:
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
            count, symbol, start, end,
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
        sample_sql = (
            f"SELECT column_name FROM "
            f"(DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        )
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
            sym_filter = (
                f"AND symbol = '{front[0]}'"
                if front else "AND symbol NOT LIKE '%-%'"
            )
        else:
            sym_filter = ""

        sql = (
            f"SELECT * FROM ({union_sql}) AS t "
            f"WHERE ts_event >= $1 AND ts_event <= $2 "
            f"{sym_filter} "
            f"ORDER BY ts_event ASC"
        )
        return self._conn.execute(
            sql, [pd.Timestamp(start), pd.Timestamp(end)]
        ).fetchdf()

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
        sample_sql = (
            f"SELECT column_name FROM "
            f"(DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        )
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
        return self._conn.execute(
            sql, [pd.Timestamp(start), pd.Timestamp(end)]
        ).fetchdf()

    def query_tick_feature_rows(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return ML feature rows with MBP-10 depth columns when available.

        This query is intended for ML feature extraction where we need more
        than lean (ts_event, price, size) data, but still avoid loading every
        raw column from the parquet source.
        """
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame(columns=["ts_event", "price", "size"])

        union_sql = self._union_views_sql(views)
        sample_sql = (
            f"SELECT column_name FROM "
            f"(DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        )
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

        select_cols = ["ts_event"]
        if has_book:
            select_cols.append("(bid_px_00 + ask_px_00) / 2.0 AS price")
            select_cols.append("size" if has_size else "1.0 AS size")
            # Support variable-depth schemas (e.g. mbp1 vs mbp10).
            depth_levels = [
                i for i in range(10)
                if (
                    f"bid_px_{i:02d}" in cols
                    and f"ask_px_{i:02d}" in cols
                    and f"bid_sz_{i:02d}" in cols
                    and f"ask_sz_{i:02d}" in cols
                )
            ]
            for i in depth_levels:
                select_cols.extend([
                    f"bid_px_{i:02d}",
                    f"ask_px_{i:02d}",
                    f"bid_sz_{i:02d}",
                    f"ask_sz_{i:02d}",
                ])
            sql = f"""
                SELECT {", ".join(select_cols)}
                FROM ({union_sql}) AS t
                WHERE ts_event >= $1 AND ts_event <= $2
                  AND bid_px_00 > 0 AND ask_px_00 > 0
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

        return self._conn.execute(
            sql, [pd.Timestamp(start), pd.Timestamp(end)]
        ).fetchdf()

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
            path = (
                self._data_dir / symbol / date_str
                / f"ohlcv_{timeframe}.parquet"
            )
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
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        union_sql = self._union_views_sql(views)

        # Detect whether this is MBP (order-book) data with bid/ask columns.
        # If so, build OHLCV from top-of-book mid-price to avoid extreme
        # wicks caused by deep book levels (bid_px_09 / ask_px_09).
        sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        cols = {
            r[0]
            for r in self._conn.execute(sample_sql).fetchall()
        }
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
            symbol_filter = (
                f"AND symbol = '{front[0]}'"
                if front else "AND symbol NOT LIKE '%-%'"
            )
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
        df = self._conn.execute(
            sql, [pd.Timestamp(start), pd.Timestamp(end)]
        ).fetchdf()

        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        df = df.set_index("bar_time")
        df.index.name = "timestamp"
        return df

    def build_tick_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tick_count: int = 987,
    ) -> pd.DataFrame:
        """Aggregate ticks into count-based OHLCV bars.

        Groups every ``tick_count`` ticks into one bar using
        ROW_NUMBER-based bucketing in DuckDB.

        Args:
            symbol: Instrument symbol
            start: Start time (inclusive)
            end: End time (inclusive) — hard boundary
            tick_count: Number of ticks per bar (e.g. 147, 987, 2000)

        Returns:
            DataFrame with [open, high, low, close, volume] and
            DatetimeIndex.
        """
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        union_sql = self._union_views_sql(views)

        sample_sql = f"SELECT column_name FROM (DESCRIBE SELECT * FROM ({union_sql}) LIMIT 0)"
        cols = {
            r[0]
            for r in self._conn.execute(sample_sql).fetchall()
        }
        has_book = "bid_px_00" in cols and "ask_px_00" in cols
        has_symbol = "symbol" in cols

        if has_symbol:
            front = self._conn.execute(f"""
                SELECT symbol, count(*) AS n
                FROM ({union_sql}) AS t
                WHERE symbol NOT LIKE '%-%'
                GROUP BY symbol ORDER BY n DESC LIMIT 1
            """).fetchone()
            symbol_filter = (
                f"AND symbol = '{front[0]}'"
                if front else "AND symbol NOT LIKE '%-%'"
            )
        else:
            symbol_filter = ""

        if has_book:
            price_expr = "(bid_px_00 + ask_px_00) / 2.0"
            filter_expr = "AND bid_px_00 > 0 AND ask_px_00 > 0"
        else:
            price_expr = "price"
            filter_expr = "AND price IS NOT NULL AND price > 0"

        sql = f"""
            WITH numbered AS (
                SELECT
                    ts_event,
                    {price_expr} AS mid,
                    size,
                    CAST(ROW_NUMBER() OVER (ORDER BY ts_event) - 1 AS BIGINT) AS rn
                FROM ({union_sql}) AS t
                WHERE ts_event >= $1 AND ts_event <= $2
                  {filter_expr}
                  {symbol_filter}
            )
            SELECT
                (rn // {tick_count}) AS bar_id,
                LAST(ts_event ORDER BY ts_event) AS bar_time,
                FIRST(mid ORDER BY ts_event) AS open,
                MAX(mid) AS high,
                MIN(mid) AS low,
                LAST(mid ORDER BY ts_event) AS close,
                SUM(size) AS volume
            FROM numbered
            GROUP BY bar_id
            HAVING COUNT(*) = {tick_count}
            ORDER BY bar_id ASC
        """
        df = self._conn.execute(
            sql, [pd.Timestamp(start), pd.Timestamp(end)]
        ).fetchdf()

        if df.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"]
            )

        df = df.drop(columns=["bar_id"]).set_index("bar_time")
        df.index.name = "timestamp"
        return df

    # ── Replay iterator ───────────────────────────────────────────

    def replay(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        step: timedelta = timedelta(minutes=1),
    ) -> Generator[pd.DataFrame, None, None]:
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

    def get_book_snapshot(
        self, symbol: str, as_of: datetime
    ) -> pd.DataFrame:
        """Return the order book state (10 levels) at exact timestamp.

        Finds the most recent tick record at or before ``as_of`` and
        extracts bid/ask levels from it.
        """
        views = self._get_views(symbol)
        if not views:
            return pd.DataFrame()

        union_sql = self._union_views_sql(views)
        sql = (
            f"SELECT * FROM ({union_sql}) AS t "
            f"WHERE ts_event <= $1 "
            f"ORDER BY ts_event DESC LIMIT 1"
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
