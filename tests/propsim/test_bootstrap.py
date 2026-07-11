"""Bootstrap determinism + degenerate-pool sanity."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from alpha_lab.propsim import Ruleset, TradePath, group_by_day, run_bootstrap


def _ruleset(**overrides) -> Ruleset:
    base = {
        "starting_balance": 50_000.0,
        "profit_target": 3_000.0,
        "trail_amount": 2_000.0,
        "trail_style": "eod_floor_realtime_breach",
        "trail_locks_at_start": True,
        "dll_amount": None,
        "dll_hard": False,
        "consistency_pct": None,
        "min_days": None,
        "max_eval_days": None,
        "point_value": 20.0,
    }
    base.update(overrides)
    return Ruleset(**base)


def _pool(day_points: list[list[float]]) -> list:
    trades = []
    start = date(2026, 1, 5)
    for i, points_list in enumerate(day_points):
        day = start + timedelta(days=i)
        for j, points in enumerate(points_list):
            trades.append(
                TradePath(
                    day=day,
                    entry_ts=datetime(day.year, day.month, day.day, 14, j, tzinfo=UTC),
                    points_optimistic=points,
                    points_conservative=points,
                    mfe_pts=abs(points) + 2.0,
                    mae_pts=abs(points) / 2.0,
                    resolution=None,
                )
            )
    return group_by_day(trades)


def test_bootstrap_is_deterministic_under_the_seed():
    pool = _pool([[15.0, -15.0], [30.0], [-25.0, 10.0], [50.0], [-40.0]])
    rs = _ruleset()
    kwargs = {
        "column": "optimistic",
        "breach_mode": "unrealized_adverse_first",
        "n_runs": 400,
        "max_days": 300,
    }
    first = run_bootstrap(pool, rs, seed=42, **kwargs)
    second = run_bootstrap(pool, rs, seed=42, **kwargs)
    assert first == second
    different = run_bootstrap(pool, rs, seed=43, **kwargs)
    assert different != first


def test_always_pass_pool():
    """A single +200pt day (= +$4,000) passes on day 1, every run."""
    pool = _pool([[200.0]])
    summary = run_bootstrap(
        pool,
        _ruleset(),
        column="optimistic",
        breach_mode="realized_only",
        n_runs=200,
        seed=42,
    )
    assert summary.p_pass == 1.0
    assert summary.p_bust == 0.0
    assert summary.days_to_pass_median == 1.0
    assert summary.pass_ci95_low <= summary.p_pass <= summary.pass_ci95_high


def test_always_bust_pool():
    """A single -200pt day (= -$4,000) busts the 2k trail on day 1, every run."""
    pool = _pool([[-200.0]])
    summary = run_bootstrap(
        pool,
        _ruleset(),
        column="optimistic",
        breach_mode="realized_only",
        n_runs=200,
        seed=42,
    )
    assert summary.p_bust == 1.0
    assert summary.p_pass == 0.0
    assert summary.days_to_bust_median == 1.0
    assert summary.bust_reasons == {"trailing_floor": 200}


def test_max_days_guard_yields_incomplete_runs():
    """A zero-P&L pool can never pass nor bust: every run hits max_days."""
    pool = _pool([[0.0]])
    summary = run_bootstrap(
        pool,
        _ruleset(),
        column="optimistic",
        breach_mode="realized_only",
        n_runs=50,
        seed=42,
        max_days=25,
    )
    assert summary.p_incomplete == 1.0
    assert summary.p_expired == 0.0


def test_ruleset_expiry_yields_expired_runs_not_incomplete():
    """With a max_eval_days budget the zero-P&L pool EXPIRES every run —
    distinct from the max_days runaway guard's incomplete."""
    pool = _pool([[0.0]])
    summary = run_bootstrap(
        pool,
        _ruleset(max_eval_days=10),
        column="optimistic",
        breach_mode="realized_only",
        n_runs=50,
        seed=42,
        max_days=25,
    )
    assert summary.p_expired == 1.0
    assert summary.p_incomplete == 0.0
    assert summary.p_pass + summary.p_bust + summary.p_expired + summary.p_incomplete == 1.0


def test_empty_pool_raises():
    with pytest.raises(ValueError, match="at least one trading day"):
        run_bootstrap([], _ruleset(), column="optimistic",
                      breach_mode="realized_only", n_runs=10, seed=42)
