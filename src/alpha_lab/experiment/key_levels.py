"""
Phase 1 — Key Level Computation for Order Flow Hypothesis Test.

Computes pre-session key levels (PDH, PDL, Asia/London/NY session highs
and lows) for all trading days in the NQ dataset.

Session boundaries (all Eastern Time):
    Asia:    18:00 - 01:00  (spans two calendar dates)
    London:  01:00 - 08:00
    NY RTH:  09:30 - 16:15

Output: data/experiment/key_levels.parquet
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from alpha_lab.agents.data_infra.tick_store import TickStore

logger = logging.getLogger(__name__)

# Eastern Time zone
_ET = "US/Eastern"

# NQ contract roll: NQZ5 → NQH6 on December 15, 2025
_ROLL_DATE = date(2025, 12, 15)

# Session boundaries (Eastern Time)
_ASIA_START = time(18, 0)   # Previous calendar day
_ASIA_END = time(1, 0)      # Current calendar day
_LONDON_START = time(1, 0)
_LONDON_END = time(8, 0)
_NY_RTH_START = time(9, 30)
_NY_RTH_END = time(16, 15)

# Tick file names to look for (priority order)
_TICK_FILENAMES = ["mbp10.parquet", "mbp1.parquet", "trades.parquet"]


def discover_trading_dates(data_dir: Path, symbol: str) -> list[date]:
    """Scan data directory for dates with tick data, return weekdays only."""
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return []

    dates: list[date] = []
    for d in sorted(symbol_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            dt = date.fromisoformat(d.name)
        except ValueError:
            continue
        # Must have at least one tick file
        if not any((d / f).exists() for f in _TICK_FILENAMES):
            continue
        # Weekdays only (Mon=0 through Fri=4)
        if dt.weekday() <= 4:
            dates.append(dt)

    return sorted(dates)


def discover_all_calendar_dates(data_dir: Path, symbol: str) -> list[date]:
    """Scan for ALL calendar dates with tick data (including weekends).

    Needed because Sunday evening GLOBEX data contributes to Monday's
    Asia session.
    """
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return []

    dates: list[date] = []
    for d in sorted(symbol_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            dt = date.fromisoformat(d.name)
        except ValueError:
            continue
        if any((d / f).exists() for f in _TICK_FILENAMES):
            dates.append(dt)

    return sorted(dates)


def get_front_month_symbol(dt: date) -> str:
    """Return the front-month NQ contract symbol for a given date."""
    if dt < _ROLL_DATE:
        return "NQZ5"
    return "NQH6"


def build_session_bars(
    data_dir: Path,
    symbol: str,
    trading_date: date,
    all_calendar_dates: list[date] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build 1m bars spanning the full CME session for one trading day.

    The session runs from 18:00 ET (previous calendar day) to 18:00 ET
    (current day).  We register both the previous and current calendar
    date in the TickStore so the query covers the overnight crossover.

    Returns DataFrame with US/Eastern DatetimeIndex and OHLCV columns.
    """
    cache_path = data_dir / symbol / str(trading_date) / "ohlcv_1m_session.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        if not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
            if df.index.tz is None:
                df.index = df.index.tz_localize(_ET)
            elif str(df.index.tz) != _ET:
                df.index = df.index.tz_convert(_ET)
            return df

    # Compute UTC range for the full session:
    # 18:00 ET prev day = 23:00 UTC prev day (during EST)
    prev_day = trading_date - timedelta(days=1)
    start_utc = datetime(prev_day.year, prev_day.month, prev_day.day, 23, 0)
    end_utc = datetime(trading_date.year, trading_date.month, trading_date.day, 23, 0)

    # Determine which calendar dates to register
    # Previous day's parquet covers evening ticks (23:00 UTC = 18:00 ET)
    # Current day's parquet covers daytime ticks
    dates_to_register = [prev_day, trading_date]

    # Also check if the next calendar date has data that overlaps
    # (not typically needed but handles edge cases)
    if all_calendar_dates is not None:
        cal_set = set(all_calendar_dates)
        dates_to_register = [d for d in dates_to_register if d in cal_set]
    else:
        # Filter to dates that actually have data
        dates_to_register = [
            d for d in dates_to_register
            if any((data_dir / symbol / str(d) / f).exists() for f in _TICK_FILENAMES)
        ]

    if not dates_to_register:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    store = TickStore(data_dir)
    try:
        for d in dates_to_register:
            store.register_symbol_date(symbol, d)

        df = store.build_bars_from_ticks(
            symbol, start_utc, end_utc, bar_size="1 minute",
        )
    finally:
        store.close()

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Convert to Eastern Time
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert(_ET)
    else:
        df.index = df.index.tz_convert(_ET)
    df.index.name = "timestamp"

    # Cache for subsequent runs
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

    return df


def classify_bar_session(et_time: time) -> str | None:
    """Classify a time-of-day (Eastern) into a session name.

    Returns:
        "asia", "london", "ny_rth", or None (gap period).
    """
    # Asia: 18:00 - 01:00 (cross-midnight)
    if et_time >= _ASIA_START or et_time < _ASIA_END:
        return "asia"
    # London: 01:00 - 08:00
    if _LONDON_START <= et_time < _LONDON_END:
        return "london"
    # NY RTH: 09:30 - 16:15
    if _NY_RTH_START <= et_time < _NY_RTH_END:
        return "ny_rth"
    return None


def _slice_session(bars: pd.DataFrame, session: str) -> pd.DataFrame:
    """Extract bars belonging to a specific session."""
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


def _session_high_low(bars: pd.DataFrame) -> tuple[float, float] | None:
    """Extract (high, low) from a session slice. Returns None if empty."""
    if bars.empty:
        return None
    return float(bars["high"].max()), float(bars["low"].min())


def compute_key_levels(
    data_dir: Path,
    symbol: str = "NQ",
    progress_fn: object = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Compute all key levels for all trading days.

    Args:
        data_dir: Path to the Databento data directory.
        symbol: Instrument symbol (default "NQ").
        progress_fn: Optional callback(pct: float, msg: str) for progress.
        use_cache: Whether to use/write cached session bar files.

    Returns:
        DataFrame with columns [date, level_name, level_price, available_from].
    """
    all_cal_dates = discover_all_calendar_dates(data_dir, symbol)
    trading_dates = discover_trading_dates(data_dir, symbol)

    if not trading_dates:
        logger.warning("No trading dates found in %s/%s", data_dir, symbol)
        return pd.DataFrame(
            columns=["date", "level_name", "level_price", "available_from"],
        )

    logger.info(
        "Computing key levels for %d trading days (%s to %s)",
        len(trading_dates), trading_dates[0], trading_dates[-1],
    )

    # Build session bars for all trading dates and store them
    session_bars: dict[date, pd.DataFrame] = {}
    n = len(trading_dates)

    for i, td in enumerate(trading_dates):
        if progress_fn is not None:
            progress_fn(i / n, f"Building bars for {td} ({i + 1}/{n})")

        bars = build_session_bars(
            data_dir, symbol, td,
            all_calendar_dates=all_cal_dates,
            use_cache=use_cache,
        )
        session_bars[td] = bars

    if progress_fn is not None:
        progress_fn(1.0, "Computing levels...")

    # Compute levels for each trading day
    rows: list[dict] = []
    prev_ny_hl: tuple[float, float] | None = None

    for td in trading_dates:
        bars = session_bars.get(td, pd.DataFrame())
        date_str = td.isoformat()

        # Slice sessions
        asia_bars = _slice_session(bars, "asia")
        london_bars = _slice_session(bars, "london")
        ny_bars = _slice_session(bars, "ny_rth")

        # Asia levels
        asia_hl = _session_high_low(asia_bars)
        if asia_hl is not None:
            available = f"{date_str}T01:00:00-05:00"
            rows.append({
                "date": date_str,
                "level_name": "asia_high",
                "level_price": asia_hl[0],
                "available_from": available,
            })
            rows.append({
                "date": date_str,
                "level_name": "asia_low",
                "level_price": asia_hl[1],
                "available_from": available,
            })

        # London levels
        london_hl = _session_high_low(london_bars)
        if london_hl is not None:
            available = f"{date_str}T08:00:00-05:00"
            rows.append({
                "date": date_str,
                "level_name": "london_high",
                "level_price": london_hl[0],
                "available_from": available,
            })
            rows.append({
                "date": date_str,
                "level_name": "london_low",
                "level_price": london_hl[1],
                "available_from": available,
            })

        # PDH/PDL from PREVIOUS day's NY RTH
        if prev_ny_hl is not None:
            available = f"{date_str}T09:30:00-05:00"
            rows.append({
                "date": date_str,
                "level_name": "PDH",
                "level_price": prev_ny_hl[0],
                "available_from": available,
            })
            rows.append({
                "date": date_str,
                "level_name": "PDL",
                "level_price": prev_ny_hl[1],
                "available_from": available,
            })

        # Update prev_ny for next day
        ny_hl = _session_high_low(ny_bars)
        if ny_hl is not None:
            prev_ny_hl = ny_hl

    df = pd.DataFrame(rows)
    logger.info("Computed %d key levels across %d trading days", len(df), len(trading_dates))
    return df


def spot_check(levels_df: pd.DataFrame, dates: list[str]) -> None:
    """Print formatted levels table for the given dates."""
    for date_str in dates:
        subset = levels_df[levels_df["date"] == date_str].sort_values("level_name")
        print(f"\n{'=' * 60}")
        print(f"  Key Levels for {date_str}")
        print(f"{'=' * 60}")
        if subset.empty:
            print("  (no levels found)")
            continue
        for _, row in subset.iterrows():
            print(
                f"  {row['level_name']:>14s}  "
                f"{row['level_price']:>10.2f}  "
                f"available from {row['available_from']}"
            )
    print()
