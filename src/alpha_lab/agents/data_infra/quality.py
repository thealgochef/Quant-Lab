"""
Data quality validation per architecture spec Section 3.1.

Checks:
- No gaps > 2 minutes during RTH (flag if found)
- Volume > 0 for all RTH bars
- High >= max(Open, Close) and Low <= min(Open, Close) for all bars
- Timestamp monotonically increasing
- Cross-timeframe consistency: 1D bar ~ aggregation of 1H bars
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from alpha_lab.core.contracts import QualityReport


def validate_no_gaps(bars: pd.DataFrame, max_gap_seconds: int = 120) -> list[dict]:
    """Check for gaps > max_gap_seconds during RTH.

    Returns list of gap details: [{timestamp, duration_sec, severity}].
    Severity: "minor" if < 5 min, "major" if >= 5 min.
    """
    rth = bars[bars["session_type"] == "RTH"] if "session_type" in bars.columns else bars

    if len(rth) < 2:
        return []

    diffs = rth.index.to_series().diff().dt.total_seconds().iloc[1:]
    mask = diffs > max_gap_seconds

    return [
        {
            "timestamp": str(diffs.index[i]),
            "duration_sec": secs,
            "severity": "minor" if secs < 300 else "major",
        }
        for i, secs in enumerate(diffs)
        if mask.iloc[i]
    ]


def validate_volume(bars: pd.DataFrame) -> int:
    """Count zero-volume bars during RTH."""
    rth = bars[bars["session_type"] == "RTH"] if "session_type" in bars.columns else bars

    if rth.empty:
        return 0
    return int((rth["volume"] <= 0).sum())


def validate_ohlc(bars: pd.DataFrame) -> int:
    """Validate High >= max(O,C) and Low <= min(O,C). Returns violation count."""
    if bars.empty:
        return 0
    high_ok = bars["high"] >= bars[["open", "close"]].max(axis=1)
    low_ok = bars["low"] <= bars[["open", "close"]].min(axis=1)
    return int((~(high_ok & low_ok)).sum())


def validate_timestamps(bars: pd.DataFrame) -> bool:
    """Check that the DatetimeIndex is monotonically increasing."""
    if bars.empty:
        return True
    return bool(bars.index.is_monotonic_increasing)


def validate_cross_timeframe(bars_dict: dict[str, pd.DataFrame]) -> int:
    """Verify 1D bars match aggregated 1H bars. Returns mismatch count."""
    if "1D" not in bars_dict or "1H" not in bars_dict:
        return 0

    daily = bars_dict["1D"]
    hourly = bars_dict["1H"]
    if daily.empty or hourly.empty:
        return 0

    mismatches = 0
    tol = 0.01

    for day_ts, day_row in daily.iterrows():
        day_date = day_ts.date() if hasattr(day_ts, "date") else day_ts
        day_hourly = hourly[hourly.index.date == day_date]

        if day_hourly.empty:
            mismatches += 1
            continue

        agg_open = day_hourly["open"].iloc[0]
        agg_high = day_hourly["high"].max()
        agg_low = day_hourly["low"].min()
        agg_close = day_hourly["close"].iloc[-1]

        if (
            abs(day_row["open"] - agg_open) > tol
            or abs(day_row["high"] - agg_high) > tol
            or abs(day_row["low"] - agg_low) > tol
            or abs(day_row["close"] - agg_close) > tol
        ):
            mismatches += 1

    return mismatches


def run_quality_checks(
    bars_dict: dict[str, pd.DataFrame],
    instrument: str,
) -> QualityReport:
    """Run the full quality validation suite.

    Uses 1m bars as the primary validation target for gap and volume checks.
    Returns a QualityReport with pass/fail status.
    """
    bars_1m = bars_dict.get("1m", pd.DataFrame())
    total_bars = sum(len(df) for df in bars_dict.values())

    gaps_detail = validate_no_gaps(bars_1m) if not bars_1m.empty else []
    gaps_found = len(gaps_detail)
    volume_zeros = validate_volume(bars_1m) if not bars_1m.empty else 0
    ohlc_violations = sum(validate_ohlc(df) for df in bars_dict.values())
    timestamps_ok = all(validate_timestamps(df) for df in bars_dict.values())
    cross_tf = validate_cross_timeframe(bars_dict)

    # Timestamp coverage: ratio of actual to expected 1m RTH bars
    # Expected ~405 bars per RTH session (09:30-16:15 = 405 minutes)
    timestamp_coverage = 1.0
    if not bars_1m.empty and "session_type" in bars_1m.columns:
        rth = bars_1m[bars_1m["session_type"] == "RTH"]
        if not rth.empty:
            unique_days = len(set(rth.index.date))
            expected = unique_days * 405
            timestamp_coverage = min(1.0, len(rth) / expected) if expected > 0 else 1.0

    passed = (
        gaps_found == 0
        and volume_zeros == 0
        and ohlc_violations == 0
        and timestamps_ok
        and cross_tf == 0
    )

    return QualityReport(
        passed=passed,
        total_bars=total_bars,
        gaps_found=gaps_found,
        gaps_detail=gaps_detail,
        volume_zeros=volume_zeros,
        ohlc_violations=ohlc_violations,
        cross_tf_mismatches=cross_tf,
        timestamp_coverage=timestamp_coverage,
        report_generated_at=datetime.now(UTC).isoformat(),
    )
