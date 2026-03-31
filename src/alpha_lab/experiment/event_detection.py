"""
Phase 2 — Event Detection for Order Flow Hypothesis Test.

Detects first-touch events when price reaches pre-computed key levels
(PDH, PDL, Asia/London session highs/lows) from Phase 1.

Touch definition: a 1m bar's range intersects the level price:
    bar.low <= level_price <= bar.high

Rules:
    - First touch only per level per day
    - available_from enforcement (no look-ahead bias)
    - Proximity zone merging (levels within 3 pts treated as single zone)
    - Daily reset

Output: data/experiment/events.parquet
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_ET = "US/Eastern"
PROXIMITY_THRESHOLD = 3.0  # NQ points

# Level type → trade direction mapping
_HIGH_LEVELS = {"PDH", "asia_high", "london_high"}
_LOW_LEVELS = {"PDL", "asia_low", "london_low"}


@dataclass
class Zone:
    """A proximity-merged group of one or more key levels."""

    zone_id: str
    level_names: list[str]
    level_prices: dict[str, float]
    representative_price: float
    available_from: pd.Timestamp
    touched: bool = field(default=False)


def build_zones(
    day_levels: pd.DataFrame,
    date_str: str,
    threshold: float = PROXIMITY_THRESHOLD,
) -> list[Zone]:
    """Group levels within ``threshold`` points into merged zones.

    Uses greedy single-linkage on sorted prices: if consecutive sorted
    levels are within ``threshold``, they merge into one zone.

    Args:
        day_levels: DataFrame with columns [level_name, level_price, available_from]
                    filtered to a single trading day.
        date_str: Trading date string (e.g. "2026-02-20").
        threshold: Maximum price distance for merging (default 3.0 NQ points).

    Returns:
        List of Zone objects, each containing one or more merged levels.
    """
    if day_levels.empty:
        return []

    sorted_levels = day_levels.sort_values("level_price").reset_index(drop=True)

    # Greedy single-linkage clustering
    groups: list[list[int]] = [[0]]
    for i in range(1, len(sorted_levels)):
        prev_price = sorted_levels.loc[groups[-1][-1], "level_price"]
        curr_price = sorted_levels.loc[i, "level_price"]
        if curr_price - prev_price <= threshold:
            groups[-1].append(i)
        else:
            groups.append([i])

    zones: list[Zone] = []
    for group_indices in groups:
        rows = sorted_levels.iloc[group_indices]
        names = sorted(rows["level_name"].tolist())
        prices = {r["level_name"]: r["level_price"] for _, r in rows.iterrows()}
        rep_price = rows["level_price"].mean()
        avail_from = max(pd.Timestamp(af) for af in rows["available_from"])

        if len(names) == 1:
            zid = f"{date_str}_{names[0]}"
        else:
            zid = f"{date_str}_{'+'.join(names)}"

        zones.append(Zone(
            zone_id=zid,
            level_names=names,
            level_prices=prices,
            representative_price=rep_price,
            available_from=avail_from,
        ))

    return zones


def load_session_bars(data_dir: Path, date_str: str) -> pd.DataFrame:
    """Load cached 1m session bars for a single trading day.

    Args:
        data_dir: Root data directory (e.g. Path("data/databento")).
        date_str: Trading date string (e.g. "2026-02-20").

    Returns:
        DataFrame with DatetimeIndex (US/Eastern) and OHLCV columns.
    """
    bar_path = data_dir / "NQ" / date_str / "ohlcv_1m_session.parquet"
    if not bar_path.exists():
        logger.warning("No session bars for %s at %s", date_str, bar_path)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.read_parquet(bar_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")

    if df.index.tz is None:
        df.index = df.index.tz_localize(_ET)
    elif str(df.index.tz) != _ET:
        df.index = df.index.tz_convert(_ET)

    return df


def compute_approach_direction(
    bars: pd.DataFrame,
    touch_idx: int,
    level_price: float,
) -> str:
    """Determine whether price approached the level from above or below.

    Uses previous bar's close. If touch is at bar index 0, uses the
    touch bar's open instead.
    """
    if touch_idx > 0:
        ref_price = bars.iloc[touch_idx - 1]["close"]
    else:
        ref_price = bars.iloc[touch_idx]["open"]

    return "from_above" if ref_price > level_price else "from_below"


def _determine_direction(zone: Zone, approach_direction: str) -> str:
    """Determine trade direction (LONG/SHORT) from level types.

    - All LOW levels → LONG (expecting bounce up)
    - All HIGH levels → SHORT (expecting reversal down)
    - Mixed zone → use approach_direction as tiebreaker
    """
    has_high = any(n in _HIGH_LEVELS for n in zone.level_names)
    has_low = any(n in _LOW_LEVELS for n in zone.level_names)

    if has_low and not has_high:
        return "LONG"
    if has_high and not has_low:
        return "SHORT"
    # Mixed zone: from_above → LONG (price falling to support), from_below → SHORT
    return "LONG" if approach_direction == "from_above" else "SHORT"


def detect_touches_single_day(
    bars: pd.DataFrame,
    zones: list[Zone],
    date_str: str,
) -> list[dict]:
    """Scan 1m bars and detect first-touch events for each zone.

    Emits one event row per zone touch (not per constituent level).

    Args:
        bars: Session bars with DatetimeIndex in US/Eastern.
        zones: Pre-built zones for this day (from build_zones).
        date_str: Trading date string.

    Returns:
        List of event dicts, one per zone touch.
    """
    if bars.empty or not zones:
        return []

    events: list[dict] = []

    for bar_pos in range(len(bars)):
        bar = bars.iloc[bar_pos]
        bar_ts = bars.index[bar_pos]

        for zone in zones:
            if zone.touched:
                continue

            # available_from enforcement
            if bar_ts < zone.available_from:
                continue

            # Touch detection: bar range intersects representative price
            if bar["low"] <= zone.representative_price <= bar["high"]:
                zone.touched = True
                approach = compute_approach_direction(
                    bars, bar_pos, zone.representative_price,
                )
                direction = _determine_direction(zone, approach)

                events.append({
                    "date": date_str,
                    "event_ts": bar_ts,
                    "level_names": json.dumps(zone.level_names),
                    "level_prices": json.dumps(zone.level_prices),
                    "representative_price": zone.representative_price,
                    "touch_price": bar["close"],
                    "approach_direction": approach,
                    "direction": direction,
                    "bar_open": bar["open"],
                    "bar_high": bar["high"],
                    "bar_low": bar["low"],
                    "bar_close": bar["close"],
                    "bar_volume": bar["volume"],
                    "zone_id": zone.zone_id,
                })

        # Early termination
        if all(z.touched for z in zones):
            break

    return events


def detect_all_events(
    levels_path: Path = Path("data/experiment/key_levels.parquet"),
    data_dir: Path = Path("data/databento"),
    output_path: Path | None = Path("data/experiment/events.parquet"),
    proximity_threshold: float = PROXIMITY_THRESHOLD,
    progress_fn: object = None,
) -> pd.DataFrame:
    """Detect all touch events across all trading days.

    Args:
        levels_path: Path to Phase 1 key_levels.parquet.
        data_dir: Root path to databento bar data.
        output_path: Where to write events.parquet (None to skip writing).
        proximity_threshold: Max distance for zone merging (default 3.0).
        progress_fn: Optional callback(pct: float, msg: str) for progress.

    Returns:
        DataFrame with all touch events.
    """
    levels_df = pd.read_parquet(levels_path)
    dates = sorted(levels_df["date"].unique())
    n = len(dates)

    logger.info("Detecting touch events for %d trading days", n)

    all_events: list[dict] = []
    stats = {"days": 0, "levels": 0, "touches": 0, "zones_merged": 0}

    for i, date_str in enumerate(dates):
        if progress_fn is not None:
            progress_fn(i / n, f"Scanning {date_str} ({i + 1}/{n})")

        day_levels = levels_df[levels_df["date"] == date_str]
        stats["levels"] += len(day_levels)

        zones = build_zones(day_levels, date_str, proximity_threshold)
        stats["zones_merged"] += sum(1 for z in zones if len(z.level_names) > 1)

        bars = load_session_bars(data_dir, date_str)
        if bars.empty:
            logger.warning("No bars for %s, skipping", date_str)
            continue

        day_events = detect_touches_single_day(bars, zones, date_str)
        all_events.extend(day_events)
        stats["touches"] += len(day_events)
        stats["days"] += 1

    if progress_fn is not None:
        progress_fn(1.0, "Done")

    _COLUMNS = [
        "date", "event_ts", "level_names", "level_prices",
        "representative_price", "touch_price", "approach_direction",
        "direction", "bar_open", "bar_high", "bar_low", "bar_close",
        "bar_volume", "zone_id",
    ]

    if not all_events:
        df = pd.DataFrame(columns=_COLUMNS)
    else:
        df = pd.DataFrame(all_events)
        df["event_ts"] = pd.to_datetime(df["event_ts"], utc=False)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Wrote %d events to %s", len(df), output_path)

    logger.info(
        "Event detection complete: %d days, %d levels, %d touch events, "
        "%d merged zones",
        stats["days"], stats["levels"], stats["touches"], stats["zones_merged"],
    )

    return df


def spot_check_events(events_df: pd.DataFrame, dates: list[str]) -> None:
    """Print formatted events table for given dates."""
    for date_str in dates:
        subset = events_df[events_df["date"] == date_str].sort_values("event_ts")
        print(f"\n{'=' * 90}")
        print(f"  Touch Events for {date_str}")
        print(f"{'=' * 90}")
        if subset.empty:
            print("  (no events)")
            continue
        for _, row in subset.iterrows():
            ts = pd.Timestamp(row["event_ts"]).strftime("%H:%M")
            names = json.loads(row["level_names"])
            names_str = "+".join(names)
            print(
                f"  {ts}  {names_str:>25s}  "
                f"level={row['representative_price']:>10.2f}  "
                f"close={row['touch_price']:>10.2f}  "
                f"{row['direction']:>5s}  "
                f"{row['approach_direction']:>12s}  "
                f"zone={row['zone_id']}"
            )
    print()
