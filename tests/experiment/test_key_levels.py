"""Tests for Phase 1 — Key Level Computation."""

from __future__ import annotations

from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment.key_levels import (
    _session_high_low,
    _slice_session,
    classify_bar_session,
    compute_key_levels,
    discover_trading_dates,
    get_front_month_symbol,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_bars_et(
    date_str: str,
    start_hour: int,
    start_minute: int,
    n_bars: int,
    base_price: float = 22000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic 1m OHLCV bars in US/Eastern timezone."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(
        f"{date_str} {start_hour:02d}:{start_minute:02d}",
        periods=n_bars,
        freq="1min",
        tz="US/Eastern",
    )
    close = base_price + rng.standard_normal(n_bars).cumsum() * 0.5
    high = close + np.abs(rng.standard_normal(n_bars)) * 0.5
    low = close - np.abs(rng.standard_normal(n_bars)) * 0.5
    op = close + rng.standard_normal(n_bars) * 0.3

    return pd.DataFrame(
        {
            "open": op,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(100, 5000, n_bars).astype(float),
        },
        index=idx,
    )


def _make_full_session_bars(trading_date_str: str, seed: int = 42) -> pd.DataFrame:
    """Create synthetic bars spanning a full CME session.

    Covers: Asia (prev day 18:00 ET - 01:00 ET), London (01:00-08:00 ET),
    gap (08:00-09:30), NY RTH (09:30-16:15 ET), gap (16:15-18:00).
    """
    d = pd.Timestamp(trading_date_str).date()
    prev_day = d - timedelta(days=1)

    frames = []
    # Asia evening portion: prev day 18:00-23:59
    frames.append(_make_bars_et(
        str(prev_day), 18, 0, 360, base_price=21900, seed=seed,
    ))
    # Asia morning portion: current day 00:00-00:59
    frames.append(_make_bars_et(
        str(d), 0, 0, 60, base_price=21950, seed=seed + 1,
    ))
    # London: 01:00-07:59
    frames.append(_make_bars_et(
        str(d), 1, 0, 420, base_price=22000, seed=seed + 2,
    ))
    # NY RTH: 09:30-16:14
    frames.append(_make_bars_et(
        str(d), 9, 30, 405, base_price=22100, seed=seed + 3,
    ))

    return pd.concat(frames).sort_index()


# ── Tests ──────────────────────────────────────────────────────


class TestClassifyBarSession:
    def test_asia_evening(self):
        assert classify_bar_session(time(18, 0)) == "asia"
        assert classify_bar_session(time(20, 30)) == "asia"
        assert classify_bar_session(time(23, 59)) == "asia"

    def test_asia_morning(self):
        assert classify_bar_session(time(0, 0)) == "asia"
        assert classify_bar_session(time(0, 30)) == "asia"
        assert classify_bar_session(time(0, 59)) == "asia"

    def test_london(self):
        assert classify_bar_session(time(1, 0)) == "london"
        assert classify_bar_session(time(5, 0)) == "london"
        assert classify_bar_session(time(7, 59)) == "london"

    def test_london_end_boundary(self):
        # 08:00 is NOT london (it's a gap)
        assert classify_bar_session(time(8, 0)) is None

    def test_ny_rth(self):
        assert classify_bar_session(time(9, 30)) == "ny_rth"
        assert classify_bar_session(time(12, 0)) == "ny_rth"
        assert classify_bar_session(time(16, 14)) == "ny_rth"

    def test_ny_rth_end_boundary(self):
        # 16:15 is NOT RTH (it's post-market gap)
        assert classify_bar_session(time(16, 15)) is None

    def test_gap_pre_market(self):
        assert classify_bar_session(time(8, 30)) is None
        assert classify_bar_session(time(9, 0)) is None
        assert classify_bar_session(time(9, 29)) is None

    def test_gap_post_market(self):
        assert classify_bar_session(time(16, 15)) is None
        assert classify_bar_session(time(17, 0)) is None
        assert classify_bar_session(time(17, 59)) is None


class TestSliceSession:
    def test_asia_cross_midnight(self):
        bars = _make_full_session_bars("2026-02-20")
        asia = _slice_session(bars, "asia")
        assert not asia.empty
        # Asia should include bars from both prev day evening and current morning
        times = asia.index.time
        has_evening = any(t >= time(18, 0) for t in times)
        has_morning = any(t < time(1, 0) for t in times)
        assert has_evening, "Asia should include evening bars"
        assert has_morning, "Asia should include pre-01:00 bars"

    def test_london_boundaries(self):
        bars = _make_full_session_bars("2026-02-20")
        london = _slice_session(bars, "london")
        assert not london.empty
        times = london.index.time
        assert all(time(1, 0) <= t < time(8, 0) for t in times)

    def test_ny_rth_boundaries(self):
        bars = _make_full_session_bars("2026-02-20")
        ny = _slice_session(bars, "ny_rth")
        assert not ny.empty
        times = ny.index.time
        assert all(time(9, 30) <= t < time(16, 15) for t in times)

    def test_empty_bars(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert _slice_session(empty, "asia").empty
        assert _slice_session(empty, "london").empty
        assert _slice_session(empty, "ny_rth").empty


class TestSessionHighLow:
    def test_extracts_high_low(self):
        bars = _make_full_session_bars("2026-02-20")
        ny = _slice_session(bars, "ny_rth")
        result = _session_high_low(ny)
        assert result is not None
        high, low = result
        assert high == float(ny["high"].max())
        assert low == float(ny["low"].min())
        assert high > low

    def test_empty_returns_none(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert _session_high_low(empty) is None


class TestGetFrontMonthSymbol:
    def test_pre_roll_nqz5(self):
        assert get_front_month_symbol(date(2025, 11, 17)) == "NQZ5"
        assert get_front_month_symbol(date(2025, 12, 14)) == "NQZ5"

    def test_post_roll_nqh6(self):
        assert get_front_month_symbol(date(2025, 12, 15)) == "NQH6"
        assert get_front_month_symbol(date(2026, 2, 20)) == "NQH6"


class TestDiscoverTradingDates:
    def test_finds_weekdays(self, tmp_path):
        """Only weekday directories with tick files are returned."""
        sym_dir = tmp_path / "NQ"
        # Create a Monday
        mon = date(2026, 2, 16)  # Monday
        (sym_dir / str(mon)).mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(
            sym_dir / str(mon) / "mbp10.parquet",
        )
        # Create a Saturday (should be filtered out)
        sat = date(2026, 2, 14)
        (sym_dir / str(sat)).mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(
            sym_dir / str(sat) / "mbp10.parquet",
        )
        # Create a dir without tick files (should be filtered out)
        no_data = date(2026, 2, 17)
        (sym_dir / str(no_data)).mkdir(parents=True)

        result = discover_trading_dates(tmp_path, "NQ")
        assert result == [mon]

    def test_empty_dir(self, tmp_path):
        assert discover_trading_dates(tmp_path, "NQ") == []


class TestComputeKeyLevels:
    def test_synthetic_two_days(self, tmp_path):
        """Compute levels from synthetic data for 2 consecutive trading days."""
        sym_dir = tmp_path / "NQ"

        # Create 3 calendar dates of tick data (we need prev day for session bars)
        dates = [date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20)]
        for d in dates:
            out = sym_dir / str(d)
            out.mkdir(parents=True)
            # Create synthetic tick data spanning full CME day
            # 00:00 UTC to 22:00 UTC covers all ET sessions
            # (19:00 ET prev day through 17:00 ET current day)
            rng = np.random.default_rng(d.toordinal())
            n = 8000
            ts = pd.date_range(
                f"{d} 00:00", periods=n, freq="10s", tz="UTC",
            )
            prices = 22000 + rng.standard_normal(n).cumsum() * 0.25
            ticks = pd.DataFrame({
                "ts_event": ts,
                "price": prices,
                "size": rng.integers(1, 50, n),
            })
            ticks.to_parquet(out / "mbp10.parquet")

        levels = compute_key_levels(
            tmp_path, "NQ", use_cache=False,
        )

        assert isinstance(levels, pd.DataFrame)
        assert not levels.empty
        assert set(levels.columns) == {
            "date", "level_name", "level_price", "available_from",
        }

        # Should have levels for Wed Feb 19 and Thu Feb 20
        level_dates = sorted(levels["date"].unique())
        assert len(level_dates) >= 2

        # First trading day should NOT have PDH/PDL (no prior RTH data)
        first_day = levels[levels["date"] == level_dates[0]]
        assert "PDH" not in first_day["level_name"].values

        # Second trading day should have PDH/PDL
        if len(level_dates) > 1:
            second_day = levels[levels["date"] == level_dates[1]]
            level_names = set(second_day["level_name"].values)
            assert "PDH" in level_names
            assert "PDL" in level_names

    def test_available_from_timestamps(self, tmp_path):
        """Verify each level type has the correct available_from format."""
        sym_dir = tmp_path / "NQ"
        dates = [date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20)]
        for d in dates:
            out = sym_dir / str(d)
            out.mkdir(parents=True)
            rng = np.random.default_rng(d.toordinal())
            n = 8000
            ts = pd.date_range(f"{d} 00:00", periods=n, freq="10s", tz="UTC")
            prices = 22000 + rng.standard_normal(n).cumsum() * 0.25
            pd.DataFrame({
                "ts_event": ts, "price": prices,
                "size": rng.integers(1, 50, n),
            }).to_parquet(out / "mbp10.parquet")

        levels = compute_key_levels(tmp_path, "NQ", use_cache=False)

        for _, row in levels.iterrows():
            ts = pd.Timestamp(row["available_from"])
            assert ts.tzinfo is not None, "available_from must be tz-aware"

            name = row["level_name"]
            if name.startswith("asia"):
                assert "T01:00:00" in row["available_from"]
            elif name.startswith("london"):
                assert "T08:00:00" in row["available_from"]
            elif name in ("PDH", "PDL"):
                assert "T09:30:00" in row["available_from"]
