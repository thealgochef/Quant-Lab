"""Tests for Phase 2 — Event Detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment.event_detection import (
    Zone,
    build_zones,
    compute_approach_direction,
    detect_all_events,
    detect_touches_single_day,
    _determine_direction,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_levels_df(
    date_str: str,
    levels: list[tuple[str, float, str]],
) -> pd.DataFrame:
    """Build a levels DataFrame for one day.

    Args:
        date_str: Trading date.
        levels: List of (level_name, level_price, available_from) tuples.
    """
    rows = [
        {
            "date": date_str,
            "level_name": name,
            "level_price": price,
            "available_from": avail,
        }
        for name, price, avail in levels
    ]
    return pd.DataFrame(rows)


def _make_controlled_bars(
    date_str: str,
    closes: list[float],
    start_hour: int = 9,
    start_minute: int = 30,
    bar_range: float = 1.0,
) -> pd.DataFrame:
    """Create bars with controlled close prices for deterministic testing.

    Each close in ``closes`` becomes a 1m bar. The bar's high/low are
    close +/- bar_range, and open = previous close (or first close).
    """
    n = len(closes)
    idx = pd.date_range(
        f"{date_str} {start_hour:02d}:{start_minute:02d}",
        periods=n,
        freq="1min",
        tz="US/Eastern",
    )
    closes_arr = np.array(closes)
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]

    return pd.DataFrame(
        {
            "open": opens,
            "high": closes_arr + bar_range,
            "low": closes_arr - bar_range,
            "close": closes_arr,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


# ── TestBuildZones ─────────────────────────────────────────────


class TestBuildZones:
    def test_no_merging_all_separate(self):
        levels = _make_levels_df("2026-02-20", [
            ("PDH", 22100.0, "2026-02-20T09:30:00-05:00"),
            ("PDL", 21900.0, "2026-02-20T09:30:00-05:00"),
            ("asia_high", 22050.0, "2026-02-20T01:00:00-05:00"),
            ("asia_low", 21950.0, "2026-02-20T01:00:00-05:00"),
            ("london_high", 22020.0, "2026-02-20T08:00:00-05:00"),
            ("london_low", 21980.0, "2026-02-20T08:00:00-05:00"),
        ])
        zones = build_zones(levels, "2026-02-20")
        assert len(zones) == 6
        for z in zones:
            assert len(z.level_names) == 1

    def test_two_levels_merge(self):
        levels = _make_levels_df("2026-02-20", [
            ("PDH", 22000.0, "2026-02-20T09:30:00-05:00"),
            ("london_high", 22002.5, "2026-02-20T08:00:00-05:00"),
        ])
        zones = build_zones(levels, "2026-02-20")
        assert len(zones) == 1
        z = zones[0]
        assert sorted(z.level_names) == ["PDH", "london_high"]
        assert "PDH+london_high" in z.zone_id

    def test_three_way_chain_merge(self):
        levels = _make_levels_df("2026-02-20", [
            ("asia_low", 100.0, "2026-02-20T01:00:00-05:00"),
            ("london_low", 102.0, "2026-02-20T08:00:00-05:00"),
            ("PDL", 104.0, "2026-02-20T09:30:00-05:00"),
        ])
        zones = build_zones(levels, "2026-02-20", threshold=3.0)
        # A-B dist=2, B-C dist=2, both <= threshold → single chain
        assert len(zones) == 1
        assert len(zones[0].level_names) == 3

    def test_available_from_uses_max(self):
        levels = _make_levels_df("2026-02-20", [
            ("PDH", 22000.0, "2026-02-20T09:30:00-05:00"),
            ("london_high", 22001.0, "2026-02-20T08:00:00-05:00"),
        ])
        zones = build_zones(levels, "2026-02-20")
        assert len(zones) == 1
        # 09:30 > 08:00, so available_from should be 09:30
        assert zones[0].available_from.hour == 9
        assert zones[0].available_from.minute == 30

    def test_representative_price_is_mean(self):
        levels = _make_levels_df("2026-02-20", [
            ("PDH", 100.0, "2026-02-20T09:30:00-05:00"),
            ("london_high", 102.0, "2026-02-20T08:00:00-05:00"),
        ])
        zones = build_zones(levels, "2026-02-20")
        assert zones[0].representative_price == pytest.approx(101.0)

    def test_identical_prices_merge(self):
        levels = _make_levels_df("2026-02-20", [
            ("london_high", 22000.0, "2026-02-20T08:00:00-05:00"),
            ("london_low", 22000.0, "2026-02-20T08:00:00-05:00"),
        ])
        zones = build_zones(levels, "2026-02-20")
        assert len(zones) == 1
        assert len(zones[0].level_names) == 2

    def test_empty_levels(self):
        empty = pd.DataFrame(columns=["date", "level_name", "level_price", "available_from"])
        zones = build_zones(empty, "2026-02-20")
        assert zones == []


# ── TestComputeApproachDirection ───────────────────────────────


class TestComputeApproachDirection:
    def test_from_below(self):
        bars = _make_controlled_bars("2026-02-20", [21990.0, 21995.0, 22000.0])
        # Bar at index 2 touches 22000; prev close = 21995 < 22000
        assert compute_approach_direction(bars, 2, 22000.0) == "from_below"

    def test_from_above(self):
        bars = _make_controlled_bars("2026-02-20", [22010.0, 22005.0, 22000.0])
        # Bar at index 2 touches 22000; prev close = 22005 > 22000
        assert compute_approach_direction(bars, 2, 22000.0) == "from_above"

    def test_first_bar_uses_open(self):
        bars = _make_controlled_bars("2026-02-20", [22010.0, 22000.0])
        # Bar at index 0: no previous bar, uses bar's own open
        # open for first bar = close[0] = 22010 > 22000
        assert compute_approach_direction(bars, 0, 22000.0) == "from_above"


# ── TestDetermineDirection ─────────────────────────────────────


class TestDetermineDirection:
    def test_low_level_is_long(self):
        zone = Zone("z", ["PDL"], {"PDL": 100.0}, 100.0, pd.Timestamp.now())
        assert _determine_direction(zone, "from_above") == "LONG"
        assert _determine_direction(zone, "from_below") == "LONG"

    def test_high_level_is_short(self):
        zone = Zone("z", ["PDH"], {"PDH": 100.0}, 100.0, pd.Timestamp.now())
        assert _determine_direction(zone, "from_above") == "SHORT"
        assert _determine_direction(zone, "from_below") == "SHORT"

    def test_mixed_zone_uses_approach(self):
        zone = Zone(
            "z", ["PDH", "PDL"], {"PDH": 100.0, "PDL": 100.0},
            100.0, pd.Timestamp.now(),
        )
        assert _determine_direction(zone, "from_above") == "LONG"
        assert _determine_direction(zone, "from_below") == "SHORT"


# ── TestDetectTouchesSingleDay ─────────────────────────────────


class TestDetectTouchesSingleDay:
    def test_basic_single_touch(self):
        bars = _make_controlled_bars("2026-02-20", [
            21990.0, 21995.0, 22000.0, 22005.0,
        ])
        zone = Zone(
            zone_id="2026-02-20_PDH",
            level_names=["PDH"],
            level_prices={"PDH": 22000.5},
            representative_price=22000.5,
            available_from=pd.Timestamp("2026-02-20 09:30:00", tz="US/Eastern"),
        )
        events = detect_touches_single_day(bars, [zone], "2026-02-20")
        assert len(events) == 1
        assert events[0]["direction"] == "SHORT"
        assert events[0]["zone_id"] == "2026-02-20_PDH"

    def test_first_touch_only(self):
        # Price crosses 22000 at bar 2, then again at bar 5
        bars = _make_controlled_bars("2026-02-20", [
            21990.0, 21995.0, 22000.0, 22010.0, 21995.0, 22000.0,
        ])
        zone = Zone(
            zone_id="2026-02-20_PDL",
            level_names=["PDL"],
            level_prices={"PDL": 22000.5},
            representative_price=22000.5,
            available_from=pd.Timestamp("2026-02-20 09:30:00", tz="US/Eastern"),
        )
        events = detect_touches_single_day(bars, [zone], "2026-02-20")
        assert len(events) == 1
        # Should be the first touch (bar index 2, time 09:32)
        ts = pd.Timestamp(events[0]["event_ts"])
        assert ts.minute == 32

    def test_available_from_enforcement(self):
        # Bars start at 08:00, level available at 09:30
        bars = _make_controlled_bars(
            "2026-02-20",
            [22000.0, 22000.0, 22000.0] * 40,  # 120 bars from 08:00
            start_hour=8,
            start_minute=0,
        )
        zone = Zone(
            zone_id="2026-02-20_PDH",
            level_names=["PDH"],
            level_prices={"PDH": 22000.5},
            representative_price=22000.5,
            available_from=pd.Timestamp("2026-02-20 09:30:00", tz="US/Eastern"),
        )
        events = detect_touches_single_day(bars, [zone], "2026-02-20")
        assert len(events) == 1
        ts = pd.Timestamp(events[0]["event_ts"])
        assert ts.hour == 9 and ts.minute == 30

    def test_no_touch(self):
        # Price stays at 21000, level at 22000
        bars = _make_controlled_bars(
            "2026-02-20", [21000.0, 21001.0, 21002.0],
            bar_range=0.5,
        )
        zone = Zone(
            zone_id="2026-02-20_PDH",
            level_names=["PDH"],
            level_prices={"PDH": 22000.0},
            representative_price=22000.0,
            available_from=pd.Timestamp("2026-02-20 09:30:00", tz="US/Eastern"),
        )
        events = detect_touches_single_day(bars, [zone], "2026-02-20")
        assert len(events) == 0

    def test_merged_zone_one_event_row(self):
        bars = _make_controlled_bars("2026-02-20", [
            21990.0, 21995.0, 22000.0,
        ])
        zone = Zone(
            zone_id="2026-02-20_PDH+london_high",
            level_names=["PDH", "london_high"],
            level_prices={"PDH": 22000.0, "london_high": 22001.0},
            representative_price=22000.5,
            available_from=pd.Timestamp("2026-02-20 09:30:00", tz="US/Eastern"),
        )
        events = detect_touches_single_day(bars, [zone], "2026-02-20")
        # ONE event row for the merged zone
        assert len(events) == 1
        names = json.loads(events[0]["level_names"])
        assert sorted(names) == ["PDH", "london_high"]

    def test_bar_fields_captured(self):
        bars = _make_controlled_bars("2026-02-20", [22000.0], bar_range=2.0)
        zone = Zone(
            zone_id="2026-02-20_PDH",
            level_names=["PDH"],
            level_prices={"PDH": 22001.0},
            representative_price=22001.0,
            available_from=pd.Timestamp("2026-02-20 09:30:00", tz="US/Eastern"),
        )
        events = detect_touches_single_day(bars, [zone], "2026-02-20")
        assert len(events) == 1
        e = events[0]
        assert e["bar_open"] == pytest.approx(22000.0)
        assert e["bar_high"] == pytest.approx(22002.0)
        assert e["bar_low"] == pytest.approx(21998.0)
        assert e["bar_close"] == pytest.approx(22000.0)
        assert e["bar_volume"] == pytest.approx(1000.0)


# ── TestDetectAllEvents (Integration) ──────────────────────────


class TestDetectAllEventsIntegration:
    def test_with_real_data_smoke(self, tmp_path):
        """Run on real data if available, verify output schema and rules."""
        levels_path = Path("data/experiment/key_levels.parquet")
        data_dir = Path("data/databento")
        if not levels_path.exists() or not data_dir.exists():
            pytest.skip("Real data not available")

        output = tmp_path / "events.parquet"
        df = detect_all_events(levels_path, data_dir, output)

        assert not df.empty, "Should detect at least some events"
        assert len(df) > 100, f"Expected >100 events, got {len(df)}"

        # Check columns
        expected_cols = {
            "date", "event_ts", "level_names", "level_prices",
            "representative_price", "touch_price", "approach_direction",
            "direction", "bar_open", "bar_high", "bar_low", "bar_close",
            "bar_volume", "zone_id",
        }
        assert set(df.columns) == expected_cols

        # No duplicate zone touches per day
        assert not df.duplicated(subset=["date", "zone_id"]).any(), \
            "Duplicate (date, zone_id) found — first-touch-only violation"

        # Direction values
        assert set(df["direction"].unique()) <= {"LONG", "SHORT"}

        # Approach direction values
        assert set(df["approach_direction"].unique()) <= {"from_above", "from_below"}

        # Output parquet written
        assert output.exists()

    def test_parquet_roundtrip(self, tmp_path):
        """Write and read back, verify schema preservation."""
        levels_path = Path("data/experiment/key_levels.parquet")
        data_dir = Path("data/databento")
        if not levels_path.exists() or not data_dir.exists():
            pytest.skip("Real data not available")

        output = tmp_path / "events.parquet"
        df_orig = detect_all_events(levels_path, data_dir, output)
        df_read = pd.read_parquet(output)

        assert list(df_orig.columns) == list(df_read.columns)
        assert len(df_orig) == len(df_read)
