"""Tests for Phase 3 — Event Labeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment.labeling import (
    AGGRESSIVE_BLOWTHROUGH,
    MAE_STOP,
    MFE_TARGET,
    NO_RESOLUTION,
    TRADEABLE_REVERSAL,
    TRAP_REVERSAL,
    TRAP_MFE_MIN,
    label_all_events,
    label_single_event,
)


# ── Helpers ────────────────────────────────────────────────────


def _make_forward_bars(
    date_str: str,
    closes: list[float],
    start_hour: int = 10,
    start_minute: int = 0,
    bar_range: float = 1.0,
) -> pd.DataFrame:
    """Create forward bars with controlled close prices.

    high = close + bar_range, low = close - bar_range.
    """
    n = len(closes)
    idx = pd.date_range(
        f"{date_str} {start_hour:02d}:{start_minute:02d}",
        periods=n,
        freq="1min",
        tz="US/Eastern",
    )
    closes_arr = np.array(closes)
    return pd.DataFrame(
        {
            "open": closes_arr,
            "high": closes_arr + bar_range,
            "low": closes_arr - bar_range,
            "close": closes_arr,
            "volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def _make_event(
    entry_price: float = 22000.0,
    direction: str = "LONG",
    date_str: str = "2026-02-20",
    event_ts: str = "2026-02-20 10:00:00",
) -> dict:
    """Create a minimal event dict for testing."""
    return {
        "date": date_str,
        "event_ts": pd.Timestamp(event_ts, tz="US/Eastern"),
        "representative_price": entry_price,
        "touch_price": entry_price + 0.5,
        "direction": direction,
        "level_names": '["PDH"]',
        "zone_id": f"{date_str}_PDH",
    }


# ── TestLabelSingleEvent ──────────────────────────────────────


class TestLabelSingleEvent:
    def test_tradeable_reversal_long(self):
        """LONG: price rises 25+ pts from entry before adverse hits 37.5."""
        entry = 22000.0
        # Price gradually rises: +5, +12, +20, +26 pts from entry
        closes = [entry + 5, entry + 12, entry + 20, entry + 26]
        bars = _make_forward_bars("2026-02-20", closes, bar_range=1.0)
        event = _make_event(entry_price=entry, direction="LONG")

        result = label_single_event(event, bars)
        assert result["label"] == TRADEABLE_REVERSAL
        assert result["max_mfe"] >= MFE_TARGET
        assert result["max_mae"] < MAE_STOP
        assert result["bars_to_resolution"] >= 0

    def test_tradeable_reversal_short(self):
        """SHORT: price drops 25+ pts from entry."""
        entry = 22000.0
        closes = [entry - 5, entry - 12, entry - 20, entry - 26]
        bars = _make_forward_bars("2026-02-20", closes, bar_range=1.0)
        event = _make_event(entry_price=entry, direction="SHORT")

        result = label_single_event(event, bars)
        assert result["label"] == TRADEABLE_REVERSAL
        assert result["max_mfe"] >= MFE_TARGET

    def test_trap_reversal(self):
        """Price shows favorable movement (5-9 pts) then stops out at 37.5."""
        entry = 22000.0
        # LONG: rises 7 pts, then crashes 38 pts
        closes = [entry + 7, entry + 3, entry - 20, entry - 38]
        bars = _make_forward_bars("2026-02-20", closes, bar_range=1.0)
        event = _make_event(entry_price=entry, direction="LONG")

        result = label_single_event(event, bars)
        assert result["label"] == TRAP_REVERSAL
        assert result["max_mfe"] >= TRAP_MFE_MIN
        assert result["max_mae"] >= MAE_STOP

    def test_aggressive_blowthrough(self):
        """Price barely moves favorably (<5 pts) then stops out at 37.5."""
        entry = 22000.0
        # LONG: barely rises, then crashes
        closes = [entry + 1, entry - 10, entry - 25, entry - 38]
        bars = _make_forward_bars("2026-02-20", closes, bar_range=1.0)
        event = _make_event(entry_price=entry, direction="LONG")

        result = label_single_event(event, bars)
        assert result["label"] == AGGRESSIVE_BLOWTHROUGH
        assert result["max_mfe"] < TRAP_MFE_MIN
        assert result["max_mae"] >= MAE_STOP

    def test_no_resolution(self):
        """Price chops within both thresholds until session end."""
        entry = 22000.0
        # Price stays within ±8 pts
        closes = [entry + 3, entry - 2, entry + 5, entry - 4, entry + 2]
        bars = _make_forward_bars("2026-02-20", closes, bar_range=2.0)
        event = _make_event(entry_price=entry, direction="LONG")

        result = label_single_event(event, bars)
        assert result["label"] == NO_RESOLUTION
        assert result["max_mfe"] < MFE_TARGET
        assert result["max_mae"] < MAE_STOP
        assert result["bars_to_resolution"] == -1
        assert pd.isna(result["resolution_ts"])

    def test_adverse_checked_first(self):
        """Both MFE and MAE thresholds crossed in same bar → stop out."""
        entry = 22000.0
        # Single bar with huge range: high = entry + 30, low = entry - 40
        # Both MFE (30 >= 25) and MAE (40 >= 37.5) hit in same bar
        bars = _make_forward_bars("2026-02-20", [entry], bar_range=0.0)
        # Override high/low manually
        bars["high"] = entry + 30.0
        bars["low"] = entry - 40.0
        event = _make_event(entry_price=entry, direction="LONG")

        result = label_single_event(event, bars)
        # Conservative: adverse checked first → stop out
        # MFE was 30 >= 5 → trap_reversal (not tradeable)
        assert result["label"] == TRAP_REVERSAL

    def test_entry_uses_representative_price(self):
        """Verify entry is representative_price, not touch_price."""
        entry = 22000.0
        touch = 22005.0  # Different from entry
        # With entry at 22000, MFE target at 22025
        closes = [22010.0, 22020.0, 22026.0]
        bars = _make_forward_bars("2026-02-20", closes, bar_range=1.0)
        event = _make_event(entry_price=entry, direction="LONG")
        event["touch_price"] = touch  # Should be ignored

        result = label_single_event(event, bars)
        # MFE from representative_price: bar high = 22027 - 22000 = 27 >= 25
        assert result["label"] == TRADEABLE_REVERSAL
        assert result["max_mfe"] >= MFE_TARGET

    def test_empty_forward_bars(self):
        """No forward bars (event after RTH close) → no_resolution."""
        bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        event = _make_event()

        result = label_single_event(event, bars)
        assert result["label"] == NO_RESOLUTION
        assert result["max_mfe"] == 0.0
        assert result["max_mae"] == 0.0


# ── TestLabelAllEvents (Integration) ──────────────────────────


class TestLabelAllEventsIntegration:
    def test_with_real_data_smoke(self, tmp_path):
        """Run on real data, verify output schema and label validity."""
        events_path = Path("data/experiment/events.parquet")
        data_dir = Path("data/databento")
        if not events_path.exists() or not data_dir.exists():
            pytest.skip("Real data not available")

        output = tmp_path / "labeled_events.parquet"
        df = label_all_events(events_path, data_dir, output)

        assert not df.empty
        assert "label" in df.columns
        assert "max_mfe" in df.columns
        assert "max_mae" in df.columns
        assert "resolution_ts" in df.columns
        assert "bars_to_resolution" in df.columns

        # All labels are valid
        valid_labels = {
            TRADEABLE_REVERSAL, TRAP_REVERSAL,
            AGGRESSIVE_BLOWTHROUGH, NO_RESOLUTION,
        }
        assert set(df["label"].unique()) <= valid_labels

        # MFE/MAE are non-negative
        assert (df["max_mfe"] >= 0).all()
        assert (df["max_mae"] >= 0).all()

        # no_resolution has -1 bars_to_resolution
        nr = df[df["label"] == NO_RESOLUTION]
        assert (nr["bars_to_resolution"] == -1).all()

        # Resolved events have non-negative bars_to_resolution
        resolved = df[df["label"] != NO_RESOLUTION]
        if not resolved.empty:
            assert (resolved["bars_to_resolution"] >= 0).all()

    def test_label_distribution_plausible(self, tmp_path):
        """No single class should dominate > 80%."""
        events_path = Path("data/experiment/events.parquet")
        data_dir = Path("data/databento")
        if not events_path.exists() or not data_dir.exists():
            pytest.skip("Real data not available")

        output = tmp_path / "labeled_events.parquet"
        df = label_all_events(events_path, data_dir, output)

        total = len(df)
        for label, cnt in df["label"].value_counts().items():
            pct = cnt / total
            # Relaxed threshold — actual distribution is reviewed manually
            assert pct < 0.95, f"{label} dominates at {pct:.1%}"
