"""
Phase 3 — Event Labeling for Order Flow Hypothesis Test.

Labels each touch event (from Phase 2) by tracking forward price excursion
using 1m bar high/low to compute MFE (max favorable excursion) and MAE
(max adverse excursion) from the representative_price entry point.

Label logic:
    - tradeable_reversal:      MFE >= 10 pts before MAE >= 37.5 pts
    - trap_reversal:           MAE >= 37.5 (stopped out) AND MFE >= 5 pts
    - aggressive_blowthrough:  MAE >= 37.5 (stopped out) AND MFE < 5 pts
    - no_resolution:           Neither threshold hit by end of RTH (16:15 ET)

Entry price = representative_price (the level itself), NOT bar close.
Forward window = event_ts to 16:15 ET of the effective trading date.

Output: data/experiment/labeled_events.parquet
"""

from __future__ import annotations

import json
import logging
from datetime import time
from pathlib import Path

import pandas as pd

from alpha_lab.experiment.event_detection import load_session_bars

logger = logging.getLogger(__name__)

_ET = "US/Eastern"

# Labeling thresholds (NQ points)
MFE_TARGET = 25.0
MAE_STOP = 37.5
TRAP_MFE_MIN = 5.0

# Forward window cutoff
RTH_END = time(16, 15)

# Labels
TRADEABLE_REVERSAL = "tradeable_reversal"
TRAP_REVERSAL = "trap_reversal"
AGGRESSIVE_BLOWTHROUGH = "aggressive_blowthrough"
NO_RESOLUTION = "no_resolution"


def _get_forward_bars(
    event: dict | pd.Series,
    bars: pd.DataFrame,
    next_day_bars: pd.DataFrame | None,
    trading_dates: list[str],
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Get forward bars AFTER event_ts to 16:15 ET of the effective trading date.

    The touch bar itself is excluded — MFE/MAE tracking starts from the
    NEXT bar after the event, since the touch bar contains the level
    interaction itself, not reversal behavior.

    For events after 16:15 ET (evening Asia session touches), the forward
    window extends into the next trading day's bars to reach that day's
    16:15 ET RTH close.

    Returns:
        (forward_bars, rth_cutoff) tuple.
    """
    event_ts = pd.Timestamp(event["event_ts"])
    trading_date_str = event["date"]

    # Determine if event is past RTH close of its assigned trading date
    rth_cutoff = pd.Timestamp(f"{trading_date_str} 16:15:00", tz=_ET)

    if event_ts < rth_cutoff:
        # Normal case: event is before RTH close, scan same-day bars
        # Use > (not >=) to exclude the touch bar itself
        fwd = bars[(bars.index > event_ts) & (bars.index < rth_cutoff)]
        return fwd, rth_cutoff

    # Late event (18:00+ ET): scan into next trading day's bars
    date_idx = trading_dates.index(trading_date_str) if trading_date_str in trading_dates else -1
    if date_idx < 0 or date_idx + 1 >= len(trading_dates):
        # No next trading date available
        return pd.DataFrame(columns=bars.columns), rth_cutoff

    next_date_str = trading_dates[date_idx + 1]
    next_rth_cutoff = pd.Timestamp(f"{next_date_str} 16:15:00", tz=_ET)

    if next_day_bars is not None and not next_day_bars.empty:
        fwd = next_day_bars[
            (next_day_bars.index > event_ts) & (next_day_bars.index < next_rth_cutoff)
        ]
        return fwd, next_rth_cutoff

    return pd.DataFrame(columns=bars.columns), next_rth_cutoff


def label_single_event(
    event: dict | pd.Series,
    forward_bars: pd.DataFrame,
) -> dict:
    """Label a single touch event by scanning forward bars for MFE/MAE.

    Entry price is ``representative_price`` (the level itself).
    Adverse extreme is checked first each bar (conservative ordering).

    Args:
        event: One row from events.parquet.
        forward_bars: 1m bars AFTER event_ts to RTH close (touch bar excluded).

    Returns:
        Dict with label, max_mfe, max_mae, resolution_ts, bars_to_resolution.
    """
    entry_price = event["representative_price"]
    direction = event["direction"]

    max_mfe = 0.0
    max_mae = 0.0
    label = NO_RESOLUTION
    resolution_ts = pd.NaT
    bars_to_resolution = -1

    for i, (bar_ts, bar) in enumerate(forward_bars.iterrows()):
        if direction == "LONG":
            bar_mfe = bar["high"] - entry_price
            bar_mae = entry_price - bar["low"]
        else:  # SHORT
            bar_mfe = entry_price - bar["low"]
            bar_mae = bar["high"] - entry_price

        max_mfe = max(max_mfe, bar_mfe)
        max_mae = max(max_mae, bar_mae)

        # Check adverse first (conservative: assume stop triggers first)
        if max_mae >= MAE_STOP:
            if max_mfe >= TRAP_MFE_MIN:
                label = TRAP_REVERSAL
            else:
                label = AGGRESSIVE_BLOWTHROUGH
            resolution_ts = bar_ts
            bars_to_resolution = i
            break

        # Then check favorable
        if max_mfe >= MFE_TARGET:
            label = TRADEABLE_REVERSAL
            resolution_ts = bar_ts
            bars_to_resolution = i
            break

    return {
        "label": label,
        "max_mfe": round(max_mfe, 4),
        "max_mae": round(max_mae, 4),
        "resolution_ts": resolution_ts,
        "bars_to_resolution": bars_to_resolution,
    }


def label_all_events(
    events_path: Path = Path("data/experiment/events.parquet"),
    data_dir: Path = Path("data/databento"),
    output_path: Path | None = Path("data/experiment/labeled_events.parquet"),
    progress_fn: object = None,
) -> pd.DataFrame:
    """Label all touch events with MFE/MAE-based classification.

    For events after 16:15 ET, automatically loads the next trading day's
    bars to extend the forward window.

    Args:
        events_path: Path to Phase 2 events.parquet.
        data_dir: Root path to databento bar data.
        output_path: Where to write labeled_events.parquet.
        progress_fn: Optional callback(pct: float, msg: str).

    Returns:
        DataFrame with all event columns plus label, max_mfe, max_mae,
        resolution_ts, bars_to_resolution.
    """
    events_df = pd.read_parquet(events_path)
    n = len(events_df)
    trading_dates = sorted(events_df["date"].unique().tolist())

    logger.info("Labeling %d events across %d trading days", n, len(trading_dates))

    label_results: list[dict] = []

    # Cache bars by date to avoid reloading
    bars_cache: dict[str, pd.DataFrame] = {}

    def _get_bars(date_str: str) -> pd.DataFrame:
        if date_str not in bars_cache:
            bars_cache[date_str] = load_session_bars(data_dir, date_str)
        return bars_cache[date_str]

    for i, (_, event) in enumerate(events_df.iterrows()):
        if progress_fn is not None:
            progress_fn(i / n, f"Labeling event {i + 1}/{n}")

        date_str = event["date"]
        bars = _get_bars(date_str)

        # Find next trading day's bars for late events
        date_idx = trading_dates.index(date_str)
        next_day_bars = None
        if date_idx + 1 < len(trading_dates):
            next_day_bars = _get_bars(trading_dates[date_idx + 1])

        fwd_bars, _ = _get_forward_bars(
            event, bars, next_day_bars, trading_dates,
        )

        result = label_single_event(event, fwd_bars)
        label_results.append(result)

    if progress_fn is not None:
        progress_fn(1.0, "Done")

    # Merge label columns into events
    label_df = pd.DataFrame(label_results)
    labeled = pd.concat([events_df.reset_index(drop=True), label_df], axis=1)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labeled.to_parquet(output_path, index=False)
        logger.info("Wrote %d labeled events to %s", len(labeled), output_path)

    return labeled


def _classify_event_session(event_ts: pd.Timestamp) -> str:
    """Classify event into session based on time of day."""
    t = event_ts.time()
    if t >= time(18, 0) or t < time(1, 0):
        return "Asia"
    if time(1, 0) <= t < time(8, 0):
        return "London"
    if time(8, 0) <= t < time(9, 30):
        return "Pre-market"
    if time(9, 30) <= t < time(16, 15):
        return "NY RTH"
    return "Post-market"


def _primary_level_name(level_names_json: str) -> str:
    """Extract primary level name from JSON list for grouping."""
    names = json.loads(level_names_json)
    # Prefer the single name; for merged zones, use the first alphabetically
    return names[0]


def print_distribution_summary(df: pd.DataFrame) -> None:
    """Print full label distribution breakdown."""
    total = len(df)
    resolved = df[df["label"] != NO_RESOLUTION]
    unresolved = df[df["label"] == NO_RESOLUTION]

    print(f"\n{'=' * 70}")
    print("  PHASE 3 EVENT LABELING — DISTRIBUTION SUMMARY")
    print(f"{'=' * 70}")

    # Overall counts
    print(f"\n  Total events:    {total}")
    print(f"  Resolved:        {len(resolved)} ({100*len(resolved)/total:.1f}%)")
    print(f"  No resolution:   {len(unresolved)} ({100*len(unresolved)/total:.1f}%)")

    print(f"\n  --- Label Distribution (all events) ---")
    for label, cnt in df["label"].value_counts().sort_index().items():
        print(f"    {label:30s}  {cnt:4d}  ({100*cnt/total:.1f}%)")

    # By direction
    print(f"\n  --- By Direction ---")
    for direction in sorted(df["direction"].unique()):
        subset = df[df["direction"] == direction]
        print(f"\n    {direction} ({len(subset)} events):")
        for label, cnt in subset["label"].value_counts().sort_index().items():
            print(f"      {label:30s}  {cnt:4d}  ({100*cnt/len(subset):.1f}%)")

    # By session
    print(f"\n  --- By Session ---")
    df = df.copy()
    df["_session"] = df["event_ts"].apply(
        lambda x: _classify_event_session(pd.Timestamp(x))
    )
    for session in ["Asia", "London", "Pre-market", "NY RTH", "Post-market"]:
        subset = df[df["_session"] == session]
        if subset.empty:
            continue
        print(f"\n    {session} ({len(subset)} events):")
        for label, cnt in subset["label"].value_counts().sort_index().items():
            print(f"      {label:30s}  {cnt:4d}  ({100*cnt/len(subset):.1f}%)")

    # By level type
    print(f"\n  --- By Level Type ---")
    df["_level"] = df["level_names"].apply(_primary_level_name)
    for level in sorted(df["_level"].unique()):
        subset = df[df["_level"] == level]
        print(f"\n    {level} ({len(subset)} events):")
        for label, cnt in subset["label"].value_counts().sort_index().items():
            print(f"      {label:30s}  {cnt:4d}  ({100*cnt/len(subset):.1f}%)")

    # MFE/MAE stats per resolved class
    print(f"\n  --- MFE/MAE Statistics (resolved events only) ---")
    for label in [TRADEABLE_REVERSAL, TRAP_REVERSAL, AGGRESSIVE_BLOWTHROUGH]:
        subset = resolved[resolved["label"] == label]
        if subset.empty:
            continue
        print(f"\n    {label} (n={len(subset)}):")
        print(f"      MFE: mean={subset['max_mfe'].mean():.2f}  "
              f"median={subset['max_mfe'].median():.2f}  "
              f"std={subset['max_mfe'].std():.2f}")
        print(f"      MAE: mean={subset['max_mae'].mean():.2f}  "
              f"median={subset['max_mae'].median():.2f}  "
              f"std={subset['max_mae'].std():.2f}")
        print(f"      Bars to resolution: mean={subset['bars_to_resolution'].mean():.1f}  "
              f"median={subset['bars_to_resolution'].median():.1f}")

    print()
