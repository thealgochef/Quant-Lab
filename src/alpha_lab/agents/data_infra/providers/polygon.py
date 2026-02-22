"""
Polygon.io data provider for CME futures (NQ/ES).

Fetches OHLCV bars via the polygon-api-client library.
Auto-detects front-month contract ticker based on CME quarterly cycle.
Caches fetched data as Parquet files.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from polygon import RESTClient

from alpha_lab.agents.data_infra.providers.base import DataProvider
from alpha_lab.core.enums import Timeframe

logger = logging.getLogger(__name__)

# CME quarterly expiry months and their letter codes
_QUARTER_MONTHS: dict[int, str] = {3: "H", 6: "M", 9: "U", 12: "Z"}

# Default cache directory (project_root/data/cache/)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[5] / "data" / "cache"

# Map Timeframe enum to Polygon resolution string
_TIMEFRAME_TO_RESOLUTION: dict[str, str] = {
    Timeframe.M1: "1min",
    Timeframe.M5: "5min",
    Timeframe.M15: "15min",
    Timeframe.M30: "30min",
    Timeframe.H1: "1hour",
    Timeframe.H4: "4hour",
    Timeframe.D1: "1day",
}


class PolygonDataProvider(DataProvider):
    """Polygon.io data provider for CME futures."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._client: RESTClient | None = None

    # ── Connection ────────────────────────────────────────────────

    def connect(self) -> None:
        if not self._api_key:
            msg = "POLYGON_API_KEY not set (pass api_key= or set env var)"
            raise ValueError(msg)
        self._client = RESTClient(api_key=self._api_key)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Connected to Polygon.io")

    def disconnect(self) -> None:
        self._client = None
        logger.info("Disconnected from Polygon.io")

    # ── Properties ────────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "polygon"

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

        Args:
            base_symbol: Base instrument name, e.g. "NQ" or "ES"
            as_of: Reference date (defaults to now)

        Returns:
            Futures ticker like "NQH6" (NQ March 2026)
        """
        now = as_of or datetime.now()
        year = now.year
        month = now.month
        quarter_months = sorted(_QUARTER_MONTHS.keys())  # [3, 6, 9, 12]

        front_month = None
        front_year = year
        for idx, qm in enumerate(quarter_months):
            if qm > month:
                front_month = qm
                break
            if qm == month:
                if now.day < 15:
                    # Still in the expiry month, before rollover
                    front_month = qm
                else:
                    # Past rollover: advance to next quarterly month
                    if idx + 1 < len(quarter_months):
                        front_month = quarter_months[idx + 1]
                    else:
                        front_month = quarter_months[0]  # March
                        front_year = year + 1
                break

        if front_month is None:
            # Past all quarterly months this year -> roll to March next year
            front_month = 3
            front_year = year + 1

        month_code = _QUARTER_MONTHS[front_month]
        year_digit = str(front_year)[-1]
        return f"{base_symbol}{month_code}{year_digit}"

    # ── Data retrieval ────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Polygon futures aggregates.

        Args:
            symbol: Base symbol ("NQ" or "ES") — auto-resolves to front month
            timeframe: Target timeframe (time-based only, not tick)
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with DatetimeIndex (US/Eastern) and columns
            [open, high, low, close, volume]
        """
        if timeframe in (Timeframe.TICK_987, Timeframe.TICK_2000):
            msg = "Tick-based timeframes not supported by Polygon provider"
            raise NotImplementedError(msg)

        ticker = self.resolve_front_month_ticker(symbol, as_of=start)

        # For timeframes Polygon doesn't directly support (3m, 10m),
        # we fetch 1m and let the caller resample
        resolution = _TIMEFRAME_TO_RESOLUTION.get(timeframe)
        if resolution is None:
            msg = f"Fetch 1m bars and resample to {timeframe} via aggregation module"
            raise ValueError(msg)

        # Check cache
        cache_path = self._cache_path(ticker, timeframe, start, end)
        if cache_path.exists():
            logger.info("Cache hit: %s", cache_path.name)
            return pd.read_parquet(cache_path)

        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)

        rows: list[dict] = []
        start_ts = pd.Timestamp(start, tz="US/Eastern")
        end_ts = pd.Timestamp(end, tz="US/Eastern") + pd.Timedelta(days=1)

        for agg in self._client.list_futures_aggregates(
            ticker,
            resolution,
            params={
                "resolution": resolution,
                "sort": "window_start.asc",
                "limit": 5000,
            },
        ):
            ts = pd.Timestamp(agg.window_start / 1e9, unit="s", tz="US/Eastern")
            if ts < start_ts:
                continue
            if ts >= end_ts:
                break
            rows.append({
                "timestamp": ts,
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume or 0,
            })

        if not rows:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], name="timestamp", tz="US/Eastern"),
            )

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df.to_parquet(cache_path)
        logger.info("Cached %d bars to %s", len(df), cache_path.name)
        return df

    def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        msg = "Tick data not supported by Polygon provider (time bars only)"
        raise NotImplementedError(msg)

    def get_daily_settlement(self, symbol: str, date: datetime) -> float:
        df = self.get_ohlcv(symbol, Timeframe.D1, date, date)
        if df.empty:
            msg = f"No daily data for {symbol} on {date.date()}"
            raise ValueError(msg)
        return float(df["close"].iloc[-1])

    # ── Cache helper ──────────────────────────────────────────────

    def _cache_path(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> Path:
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")
        return self._cache_dir / f"{ticker}_{timeframe.value}_{start_str}_{end_str}.parquet"
