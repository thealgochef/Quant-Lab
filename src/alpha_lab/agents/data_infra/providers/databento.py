"""
Databento data provider for CME futures (NQ/ES).

Fetches L2 (MBP-10) tick data and OHLCV bars via the databento Python client.
Stores data as date/symbol-partitioned Parquet files for efficient replay.
Auto-detects front-month contract using CME quarterly cycle.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from alpha_lab.agents.data_infra.providers.base import DataProvider
from alpha_lab.core.enums import Timeframe

logger = logging.getLogger(__name__)

# Default data directory (project_root/data/databento/)
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[5] / "data" / "databento"

# CME quarterly expiry months and their letter codes
_QUARTER_MONTHS: dict[int, str] = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Databento schema for OHLCV bars at different resolutions
_OHLCV_SCHEMAS: dict[str, str] = {
    Timeframe.M1: "ohlcv-1m",
    Timeframe.H1: "ohlcv-1h",
    Timeframe.D1: "ohlcv-1d",
}


class DatabentDataProvider(DataProvider):
    """Databento data provider for CME futures with L2 book depth."""

    def __init__(
        self,
        api_key: str | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("DATABENTO_API_KEY", "")
        self._data_dir = data_dir or _DEFAULT_DATA_DIR
        self._client = None  # db.Historical instance

    # ── Connection ────────────────────────────────────────────────

    def connect(self) -> None:
        """Create Databento Historical client and ensure data directory exists."""
        if not self._api_key:
            msg = "DATABENTO_API_KEY not set (pass api_key= or set env var)"
            raise ValueError(msg)

        import databento as db

        self._client = db.Historical(self._api_key)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Connected to Databento (Historical)")

    def disconnect(self) -> None:
        """Clear client reference."""
        self._client = None
        logger.info("Disconnected from Databento")

    # ── Properties ────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "databento"

    @property
    def supported_symbols(self) -> list[str]:
        return ["NQ", "ES"]

    # ── Front-month ticker resolution ─────────────────────────────

    @staticmethod
    def resolve_front_month_ticker(
        base_symbol: str,
        as_of: datetime | None = None,
    ) -> str:
        """Auto-detect the front-month CME futures contract ticker.

        Uses same quarterly cycle logic as PolygonDataProvider.

        Args:
            base_symbol: Base instrument name, e.g. "NQ" or "ES"
            as_of: Reference date (defaults to now)

        Returns:
            Futures ticker like "NQH6" (NQ March 2026)
        """
        now = as_of or datetime.now()
        year = now.year
        month = now.month
        quarter_months = sorted(_QUARTER_MONTHS.keys())

        front_month = None
        front_year = year
        for idx, qm in enumerate(quarter_months):
            if qm > month:
                front_month = qm
                break
            if qm == month:
                if now.day < 15:
                    front_month = qm
                else:
                    if idx + 1 < len(quarter_months):
                        front_month = quarter_months[idx + 1]
                    else:
                        front_month = quarter_months[0]
                        front_year = year + 1
                break

        if front_month is None:
            front_month = 3
            front_year = year + 1

        month_code = _QUARTER_MONTHS[front_month]
        year_digit = str(front_year)[-1]
        return f"{base_symbol}{month_code}{year_digit}"

    # ── Tick data (MBP-10) ────────────────────────────────────────

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch MBP-10 tick data, caching each day as Parquet.

        Args:
            symbol: Base symbol ("NQ" or "ES")
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with MBP-10 columns, ordered by ts_event.
        """
        frames: list[pd.DataFrame] = []
        current = start.date() if isinstance(start, datetime) else start
        end_date = end.date() if isinstance(end, datetime) else end

        while current <= end_date:
            df = self._fetch_and_cache_ticks(symbol, current)
            if not df.empty:
                frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)

        # Filter to exact time range
        if "ts_event" in result.columns:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize("UTC")
            if end_ts.tz is None:
                end_ts = end_ts.tz_localize("UTC")
            result = result[
                (result["ts_event"] >= start_ts) & (result["ts_event"] <= end_ts)
            ]

        return result.reset_index(drop=True)

    def _fetch_and_cache_ticks(self, symbol: str, date) -> pd.DataFrame:
        """Fetch a single day of MBP-10 data, caching as Parquet.

        Returns cached data if the Parquet file already exists.
        """
        parquet_path = self._parquet_path(symbol, date, "mbp10")

        if parquet_path.exists():
            logger.info("Cache hit (ticks): %s", parquet_path.name)
            return pd.read_parquet(parquet_path)

        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)

        date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)
        next_date = date + timedelta(days=1)
        # Clamp end to ~1 hour before now to avoid data_end_after_available
        from datetime import UTC

        now_utc = datetime.now(UTC) - timedelta(hours=1)
        if next_date > now_utc.date():
            next_date_str = now_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        else:
            next_date_str = (
                next_date.isoformat()
                if hasattr(next_date, "isoformat")
                else str(next_date)
            )

        try:
            data = self._client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=f"{symbol}.FUT",
                stype_in="parent",
                schema="mbp-10",
                start=date_str,
                end=next_date_str,
            )

            df = data.to_df()
            if df.empty:
                logger.info("No tick data for %s on %s", symbol, date_str)
                return pd.DataFrame()

            # Ensure directory exists and write Parquet
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(parquet_path)
            logger.info(
                "Cached %d ticks to %s", len(df), parquet_path.name
            )
            return df

        except Exception:
            logger.exception(
                "Failed to fetch ticks for %s on %s", symbol, date_str
            )
            return pd.DataFrame()

    # ── Trade ticks (lightweight) ─────────────────────────────────

    def get_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch trade-only tick data using the ``trades`` schema.

        Much lighter than MBP-10 (~100x smaller).  Returns a DataFrame
        with columns [price, size, timestamp] suitable for
        ``aggregate_tick_bars()``.
        """
        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)

        # Build cache path
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        cache_path = (
            self._data_dir / symbol / f"trades_{start_str}_{end_str}.parquet"
        )

        if cache_path.exists():
            logger.info("Cache hit (trades): %s", cache_path.name)
            return pd.read_parquet(cache_path)

        from datetime import UTC as _UTC

        # Clamp end to avoid data_end_after_available error
        safe_end = min(end, datetime.now(_UTC) - timedelta(hours=1))
        try:
            data = self._client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=f"{symbol}.FUT",
                stype_in="parent",
                schema="trades",
                start=start.isoformat(),
                end=safe_end.isoformat(),
            )
            df = data.to_df()
        except Exception:
            logger.exception("Failed to fetch trades for %s", symbol)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Filter to front-month contract only (parent symbol returns
        # all contracts including spreads and back-months)
        if "symbol" in df.columns:
            front = self.resolve_front_month_ticker(symbol, as_of=start)
            if front in df["symbol"].values:
                df = df[df["symbol"] == front]
                logger.info(
                    "Filtered trades to front-month %s (%d ticks)",
                    front, len(df),
                )

        # Normalize to standard columns
        result = pd.DataFrame({
            "price": df["price"].values,
            "size": df["size"].values,
            "timestamp": (
                pd.to_datetime(df["ts_event"]).values
                if "ts_event" in df.columns
                else df.index.values
            ),
        })

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
        logger.info("Cached %d trades to %s", len(result), cache_path.name)
        return result

    # ── OHLCV bars ────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Databento.

        For 1m bars: fetches directly via ohlcv-1m schema.
        For other time-based timeframes: fetches 1m and lets caller resample.
        For tick-based timeframes: raises NotImplementedError (use get_ticks
        + aggregate_tick_bars instead).
        """
        if timeframe in (Timeframe.TICK_987, Timeframe.TICK_2000):
            msg = (
                "Tick-based timeframes not directly supported. "
                "Use get_ticks() + aggregate_tick_bars() instead."
            )
            raise NotImplementedError(msg)

        # Use 1m as the base fetch resolution
        schema = _OHLCV_SCHEMAS.get(timeframe, "ohlcv-1m")

        cache_path = self._parquet_path(
            symbol,
            start.date(),
            f"ohlcv_{timeframe.value}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}",
        )

        if cache_path.exists():
            logger.info("Cache hit (OHLCV): %s", cache_path.name)
            return pd.read_parquet(cache_path)

        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)

        try:
            data = self._client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=f"{symbol}.FUT",
                stype_in="parent",
                schema=schema,
                start=start.isoformat(),
                end=end.isoformat(),
            )

            df = data.to_df()
            if df.empty:
                return pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"],
                    index=pd.DatetimeIndex([], name="ts_event", tz="UTC"),
                )

            # Filter to front-month contract only (parent symbol returns
            # all contracts including spreads and back-months)
            if "symbol" in df.columns:
                front = self.resolve_front_month_ticker(symbol, as_of=start)
                if front in df["symbol"].values:
                    df = df[df["symbol"] == front]
                    logger.info(
                        "Filtered to front-month %s (%d bars)", front, len(df),
                    )

            # Normalize column names to match project convention
            col_map = {}
            for col in df.columns:
                lower = col.lower()
                if lower in ("open", "high", "low", "close", "volume"):
                    col_map[col] = lower
            if col_map:
                df = df.rename(columns=col_map)

            # Ensure we have the standard columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    df[col] = 0.0

            result = df[["open", "high", "low", "close", "volume"]].copy()

            # Set ts_event as index if available
            if "ts_event" in df.columns:
                result.index = pd.DatetimeIndex(df["ts_event"], name="timestamp")
            elif hasattr(df.index, "tz"):
                result.index.name = "timestamp"

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(cache_path)
            logger.info("Cached %d OHLCV bars to %s", len(result), cache_path.name)
            return result

        except Exception:
            logger.exception("Failed to fetch OHLCV for %s", symbol)
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], name="timestamp"),
            )

    def get_daily_settlement(self, symbol: str, date: datetime) -> float:
        """Get settlement price for a given date."""
        df = self.get_ohlcv(symbol, Timeframe.D1, date, date + timedelta(days=1))
        if df.empty:
            msg = f"No daily data for {symbol} on {date.date()}"
            raise ValueError(msg)
        return float(df["close"].iloc[-1])

    # ── Helpers ────────────────────────────────────────────────────

    def _parquet_path(self, symbol: str, date, schema: str) -> Path:
        """Build path: data/databento/{symbol}/{date}/{schema}.parquet"""
        date_str = date.isoformat() if hasattr(date, "isoformat") else str(date)
        return self._data_dir / symbol / date_str / f"{schema}.parquet"
