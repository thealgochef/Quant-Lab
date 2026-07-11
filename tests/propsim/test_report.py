"""Report layer: NaN-poisoned pools must degrade, never crash (PS-1)."""

from __future__ import annotations

from datetime import UTC, date, datetime

from alpha_lab.propsim import Ruleset, TradePath
from alpha_lab.propsim.loaders import LoadedTrades
from alpha_lab.propsim.report import build_report, format_human

_RULESET = Ruleset(
    starting_balance=50_000.0,
    profit_target=3_000.0,
    trail_amount=2_000.0,
    trail_style="eod_floor_realtime_breach",
    trail_locks_at_start=True,
    dll_amount=1_000.0,
    dll_soft=True,
    consistency_pct=50.0,
    min_days=None,
    point_value=20.0,
)


def test_format_human_survives_nan_points_pool():
    """A NaN points value (bypassing loader guards) sanitizes to None in the
    pool stats; format_human must omit the stats line instead of crashing."""
    nan_trade = TradePath(
        day=date(2026, 1, 5),
        entry_ts=datetime(2026, 1, 5, 14, tzinfo=UTC),
        points_optimistic=float("nan"),
        points_conservative=float("nan"),
        mfe_pts=None,
        mae_pts=None,
        resolution=None,
    )
    loaded = LoadedTrades(
        trades=[nan_trade],
        source="executions",
        excursions_available=False,
        degradation_reason="test",
        notes=[],
    )
    report = build_report(
        loaded, "topstep_50k", _RULESET, n_runs=5, seed=42, max_days=10
    )
    assert report["pool"]["optimistic"]["mean_points"] is None
    text = format_human(report)
    assert isinstance(text, str)
    assert "win rate" not in text
