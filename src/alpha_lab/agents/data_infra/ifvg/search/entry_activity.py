"""Entry chronology diagnostics over an explicit evaluation calendar.

These research outputs do not change strategy gates or trigger entries. Callers
supply every evaluated date, including zero-entry dates, and untrimmed trades
for adjacent boundaries. Stored resolution days are never used as entry days.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


def entry_trading_day(timestamp, *, timezone="America/New_York", boundary="18:00") -> str:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        raise ValueError("entry timestamp must be timezone aware")
    local = ts.tz_convert(timezone)
    day = local.date() + timedelta(days=int(local.time() >= time.fromisoformat(boundary)))
    return day.isoformat()


def day_start(day: str, *, timezone="America/New_York", boundary="18:00") -> pd.Timestamp:
    previous = date.fromisoformat(day) - timedelta(days=1)
    return pd.Timestamp(
        datetime.combine(previous, time.fromisoformat(boundary), ZoneInfo(timezone))
    ).tz_convert("UTC")


def entry_activity(
    trades: pd.DataFrame,
    evaluation_dates: list[str],
    *,
    cutoff_utc,
    timezone="America/New_York",
    boundary="18:00",
) -> dict:
    """Return daily counts, all sorted gaps, adjacent trade boundaries and summary.

    Trade grain is one actual execution (including an unresolved final position).
    Required fields: trade_id, entry_ts_utc, resolution_ts_utc (nullable). Leading
    and trailing gaps are censored relative to the explicit evaluation calendar.
    Elapsed intervals are independently measured between actual entries; a long
    open position is not labeled flat. Missing resolution yields a null flat gap.
    """
    dates = list(evaluation_dates)
    if not dates or dates != sorted(set(dates)):
        raise ValueError("evaluation dates must be nonempty, ordered and unique")
    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone aware")
    required = ["trade_id", "entry_ts_utc", "resolution_ts_utc"]
    frame = trades.reindex(columns=required).copy() if trades.empty else trades[required].copy()
    if frame.trade_id.isna().any() or frame.trade_id.duplicated().any():
        raise ValueError("trade IDs must be nonnull and unique")
    for column in ("entry_ts_utc", "resolution_ts_utc"):
        # Reject naive values before pandas can silently assume UTC.
        if any(pd.Timestamp(value).tzinfo is None for value in frame[column].dropna()):
            raise ValueError(f"{column} must be timezone aware")
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.entry_ts_utc.isna().any() or (frame.entry_ts_utc > cutoff).any():
        raise ValueError("missing entry or entry beyond cutoff")
    if (frame.resolution_ts_utc < frame.entry_ts_utc).any():
        raise ValueError("resolution precedes entry")
    if (frame.resolution_ts_utc.dropna() > cutoff).any():
        raise ValueError("resolution beyond cutoff")
    frame = frame.sort_values(["entry_ts_utc", "trade_id"], kind="stable")
    frame["entry_day"] = frame.entry_ts_utc.map(
        lambda ts: entry_trading_day(ts, timezone=timezone, boundary=boundary)
    ).astype("string")
    rows = list(frame.to_dict("records"))
    elapsed = []
    for previous, following in zip(rows, rows[1:], strict=False):
        exit_ts = previous["resolution_ts_utc"]
        if pd.notna(exit_ts) and exit_ts > following["entry_ts_utc"]:
            raise ValueError("overlapping actual positions")
        elapsed.append(
            {
                "previous_trade_id": previous["trade_id"],
                "next_trade_id": following["trade_id"],
                "previous_entry_utc": previous["entry_ts_utc"],
                "previous_exit_utc": exit_ts,
                "next_entry_utc": following["entry_ts_utc"],
                "entry_to_entry_seconds": (
                    following["entry_ts_utc"] - previous["entry_ts_utc"]
                ).total_seconds(),
                "flat_seconds": (following["entry_ts_utc"] - exit_ts).total_seconds()
                if pd.notna(exit_ts)
                else None,
            }
        )
    counts = frame.entry_day.value_counts()
    daily = pd.DataFrame(
        {"evaluation_date": dates, "actual_entries": [int(counts.get(d, 0)) for d in dates]}
    )
    gaps = []
    start = None
    for i in range(len(dates) + 1):
        zero = i < len(dates) and daily.actual_entries.iloc[i] == 0
        if zero and start is None:
            start = i
        if not zero and start is not None:
            end = i - 1
            lower = day_start(dates[start], timezone=timezone, boundary=boundary)
            upper = day_start(
                (date.fromisoformat(dates[end]) + timedelta(days=1)).isoformat(),
                timezone=timezone,
                boundary=boundary,
            )
            upper = min(upper, cutoff)
            before = frame.loc[frame.entry_ts_utc < lower]
            after = frame.loc[frame.entry_ts_utc >= upper]
            prior = before.iloc[-1] if len(before) else None
            nxt = after.iloc[0] if len(after) else None
            gaps.append(
                {
                    "first_evaluation_date": dates[start],
                    "last_evaluation_date": dates[end],
                    "evaluation_dates_without_entry": end - start + 1,
                    "cohort_start_utc": lower,
                    "cohort_end_exclusive_utc": upper,
                    "leading_censored": start == 0,
                    "trailing_censored": end == len(dates) - 1,
                    "previous_trade_id": prior.trade_id if prior is not None else None,
                    "previous_entry_utc": prior.entry_ts_utc if prior is not None else None,
                    "previous_exit_utc": prior.resolution_ts_utc if prior is not None else None,
                    "next_trade_id": nxt.trade_id if nxt is not None else None,
                    "next_entry_utc": nxt.entry_ts_utc if nxt is not None else None,
                }
            )
            start = None
    gaps.sort(
        key=lambda row: (-row["evaluation_dates_without_entry"], row["first_evaluation_date"])
    )
    maximum = gaps[0]["evaluation_dates_without_entry"] if gaps else 0
    return {
        "daily": daily,
        "gaps": pd.DataFrame(gaps),
        "elapsed": pd.DataFrame(elapsed),
        "summary": {
            "evaluation_dates": len(dates),
            "active_entry_dates": int((daily.actual_entries > 0).sum()),
            "zero_entry_dates": int((daily.actual_entries == 0).sum()),
            "longest_zero_entry_dates": maximum,
            "tied_longest_intervals": [
                row for row in gaps if row["evaluation_dates_without_entry"] == maximum
            ],
            "entry_rows_outside_evaluation_dates": int((~frame.entry_day.isin(dates)).sum()),
        },
    }


def entry_activity_payload(trades, evaluation_dates, **kwargs):
    """JSON-safe immutable research output, with explicit measurement contract."""
    result = entry_activity(trades, evaluation_dates, **kwargs)
    payload = {
        key: json.loads(result[key].to_json(orient="records", date_format="iso"))
        for key in ("daily", "gaps", "elapsed")
    }
    summary = {k: v for k, v in result["summary"].items() if k != "tied_longest_intervals"}
    summary["tied_longest_intervals"] = [
        row
        for row in payload["gaps"]
        if row["evaluation_dates_without_entry"] == summary["longest_zero_entry_dates"]
    ]
    return {
        "schema_version": 1,
        "summary": summary,
        **payload,
        "contract": {
            "date_grain": "actual entry timestamp mapped to trading day",
            "evaluation_dates": list(evaluation_dates),
            "cutoff_utc": str(kwargs["cutoff_utc"]),
            "timezone": kwargs.get("timezone", "America/New_York"),
            "boundary": kwargs.get("boundary", "18:00"),
            "gates_changed": False,
            "warmup_in_daily": False,
            "warmup_preserved_for_adjacent_boundaries": True,
        },
    }
