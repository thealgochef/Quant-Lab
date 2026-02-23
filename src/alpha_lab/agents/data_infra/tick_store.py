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

    def __init__(self, data_dir: Path | str, read_only: bool = True) -> None:
        self._data_dir = Path(data_dir)
        self._conn = duckdb.connect(database=":memory:", read_only=False)
        self._registered: dict[str, list[str]] = {}  # symbol -> [date_str, ...]
        self._read_only = read_only
        logger.info("TickStore opened (data_dir=%s)", self._data_dir)

    # ── Registration ──────────────────────────────────────────────

    def register_symbol_date(self, symbol: str, dt: date | str) -> bool:
        """Register a single Parquet file as a DuckDB view.

        Returns True if the file was found and registered.
        """
        date_str = dt.isoformat() if isinstance(dt, date) else str(dt)
        parquet_path = self._data_dir / symbol / date_str / _TICK_FILENAME

        if not parquet_path.exists():
            logger.debug("No parquet at %s", parquet_path)
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
        sql = (
            f"SELECT * FROM ({union_sql}) AS t "
            f"WHERE ts_event >= $1 AND ts_event <= $2 "
            f"ORDER BY ts_event ASC"
        )
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

        # Use first() / last() with ORDER BY for correct OHLC
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

    @staticmethod
    def _view_name(symbol: str, date_str: str) -> str:
        return f"ticks_{symbol}_{date_str.replace('-', '_')}"

    def _get_views(self, symbol: str) -> list[str]:
        dates = self._registered.get(symbol, [])
        return [self._view_name(symbol, d) for d in sorted(dates)]

    @staticmethod
    def _union_views_sql(views: list[str]) -> str:
        return " UNION ALL ".join(f"SELECT * FROM {v}" for v in views)
