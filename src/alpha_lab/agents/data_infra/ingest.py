"""
Ingestion runner for Databento market data.

Orchestrates historical backfill and live streaming of L2 (MBP-10)
tick data into date/symbol-partitioned Parquet files.

Usage::

    from alpha_lab.agents.data_infra.providers.databento import DatabentDataProvider
    provider = DatabentDataProvider()
    provider.connect()
    runner = IngestionRunner(provider, Path("data/databento"))
    runner.ingest_historical("NQ", date(2026, 1, 1), date(2026, 2, 20))
    runner.backfill("ES", days=30)
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default data directory
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "data" / "databento"


class IngestionRunner:
    """Orchestrates historical and live data ingestion from Databento."""

    def __init__(
        self,
        provider,  # DatabentDataProvider instance
        data_dir: Path | str | None = None,
    ) -> None:
        self._provider = provider
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    def ingest_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        schema: str = "mbp-10",
    ) -> dict:
        """Fetch historical tick data day-by-day and save as Parquet.

        Skips dates that already have cached Parquet files.

        Args:
            symbol: Base symbol ("NQ" or "ES")
            start_date: First date to fetch
            end_date: Last date to fetch (inclusive)
            schema: Databento schema (default "mbp-10")

        Returns:
            Summary dict with counts of fetched, cached, and failed dates.
        """
        summary = {"fetched": 0, "cached": 0, "failed": 0, "total_ticks": 0}
        current = start_date

        while current <= end_date:
            parquet_path = (
                self._data_dir / symbol / current.isoformat() / "mbp10.parquet"
            )

            if parquet_path.exists():
                summary["cached"] += 1
                logger.debug("Already cached: %s %s", symbol, current)
            else:
                try:
                    start_dt = datetime.combine(current, datetime.min.time())
                    end_dt = datetime.combine(
                        current + timedelta(days=1), datetime.min.time()
                    )
                    df = self._provider.get_ticks(symbol, start_dt, end_dt)
                    if not df.empty:
                        summary["fetched"] += 1
                        summary["total_ticks"] += len(df)
                    else:
                        summary["cached"] += 1  # No data for this date
                except Exception:
                    logger.exception(
                        "Failed to ingest %s %s", symbol, current
                    )
                    summary["failed"] += 1

            current += timedelta(days=1)

        logger.info(
            "Historical ingestion complete for %s: %s", symbol, summary
        )
        return summary

    def backfill(self, symbol: str, days: int = 252) -> dict:
        """Convenience method: backfill the last N trading days.

        Args:
            symbol: Base symbol
            days: Number of calendar days to look back

        Returns:
            Summary dict from ingest_historical.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        return self.ingest_historical(symbol, start_date, end_date)

    def ingest_live(
        self,
        symbol: str,
        callback=None,
        max_records: int | None = None,
    ) -> dict:
        """Stream live MBP-10 data, buffering to daily Parquet files.

        Uses the Databento Live client to subscribe to real-time data.
        Records are accumulated in a buffer and flushed to Parquet
        when the trading date changes or when ``max_records`` is reached.

        Args:
            symbol: Base symbol ("NQ" or "ES")
            callback: Optional callback(record) for each tick
            max_records: Stop after N records (None = run until interrupted)

        Returns:
            Summary dict with record count.
        """
        import databento as db

        api_key = os.environ.get("DATABENTO_API_KEY", "")
        if not api_key:
            msg = "DATABENTO_API_KEY not set for live ingestion"
            raise ValueError(msg)

        # Resolve front-month ticker
        from alpha_lab.agents.data_infra.providers.databento import (
            DatabentDataProvider,
        )

        ticker = DatabentDataProvider.resolve_front_month_ticker(symbol)

        live = db.Live(key=api_key)
        live.subscribe(
            dataset="GLBX.MDP3",
            schema="mbp-10",
            stype_in="raw_symbol",
            symbols=[ticker],
        )

        buffer: list[dict] = []
        current_date: date | None = None
        count = 0

        try:
            for record in live:
                count += 1

                if callback is not None:
                    callback(record)

                # Extract timestamp and convert to date
                ts_event = getattr(record, "ts_event", None)
                if ts_event is not None:
                    record_date = pd.Timestamp(
                        ts_event, unit="ns"
                    ).date()
                else:
                    record_date = date.today()

                # Flush buffer on date change
                if current_date is not None and record_date != current_date:
                    self._flush_buffer(symbol, current_date, buffer)
                    buffer = []

                current_date = record_date

                # Accumulate record as dict
                if hasattr(record, "__dict__"):
                    buffer.append({"ts_event": ts_event, "raw": str(record)})

                if max_records is not None and count >= max_records:
                    break

        except KeyboardInterrupt:
            logger.info("Live ingestion interrupted by user")
        finally:
            if buffer and current_date is not None:
                self._flush_buffer(symbol, current_date, buffer)

        logger.info("Live ingestion: %d records processed", count)
        return {"records": count}

    def _flush_buffer(
        self, symbol: str, dt: date, buffer: list[dict]
    ) -> None:
        """Write buffered records to a Parquet file."""
        if not buffer:
            return

        parquet_path = (
            self._data_dir / symbol / dt.isoformat() / "mbp10_live.parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(buffer)
        df.to_parquet(parquet_path)
        logger.info(
            "Flushed %d live records to %s", len(buffer), parquet_path
        )
