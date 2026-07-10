"""Oracle tests: hand-built day sequences with hand-computed outcomes."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from alpha_lab.propsim import EvaluationWalk, Ruleset, TradePath, walk_days


def _trade(
    day_str: str,
    points: float,
    *,
    hour: int = 14,
    minute: int = 0,
    mae: float | None = None,
    mfe: float | None = None,
) -> TradePath:
    day = date.fromisoformat(day_str)
    return TradePath(
        day=day,
        entry_ts=datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC),
        points_optimistic=points,
        points_conservative=points,
        mfe_pts=mfe,
        mae_pts=mae,
        resolution=None,
    )


def _days(*day_trades: list[TradePath]) -> list[tuple[date, list[TradePath]]]:
    start = date(2026, 1, 5)
    return [(start + timedelta(days=i), trades) for i, trades in enumerate(day_trades)]


def _ruleset(**overrides) -> Ruleset:
    base = {
        "starting_balance": 50_000.0,
        "profit_target": 3_000.0,
        "trail_amount": 2_000.0,
        "trail_style": "eod_floor_realtime_breach",
        "trail_locks_at_start": True,
        "dll_amount": 1_000.0,
        "dll_soft": True,
        "consistency_pct": 50.0,
        "min_days": None,
        "point_value": 20.0,
    }
    base.update(overrides)
    return Ruleset(**base)


def test_floor_locks_at_starting_balance():
    """Once EOD peaks exceed start + trail, the floor caps AT the start — never above."""
    rs = _ruleset(dll_amount=None, consistency_pct=None, profit_target=1e9)
    walk = EvaluationWalk(rs, column="optimistic", breach_mode="realized_only")
    assert walk.floor == 48_000.0  # day 1: start - trail
    walk.play_day([_trade("2026-01-05", 100.0)])  # EOD 52_000
    assert walk.floor == 50_000.0  # min(52k - 2k, start) -> locked at start
    walk.play_day([_trade("2026-01-06", 100.0)])  # EOD 54_000
    assert walk.floor == 50_000.0  # never above the starting balance


def test_eod_floor_never_moves_down():
    """The floor ratchets on EOD highs and holds through losing days."""
    rs = _ruleset(
        dll_amount=None, consistency_pct=None, profit_target=1e9,
        trail_locks_at_start=False,
    )
    walk = EvaluationWalk(rs, column="optimistic", breach_mode="realized_only")
    floors = [walk.floor]
    walk.play_day([_trade("2026-01-05", 100.0)])  # EOD 52_000 -> floor 50_000
    floors.append(walk.floor)
    walk.play_day([_trade("2026-01-06", -40.0)])  # EOD 51_200 -> peak still 52k
    floors.append(walk.floor)
    assert floors == [48_000.0, 50_000.0, 50_000.0]


def test_mae_excursion_busts_unrealized_but_realized_survives_same_sequence():
    """THE SAME day: a +10pt winner whose path first dipped 120pts (= $2,400)."""
    rs = _ruleset(dll_amount=None, consistency_pct=None, profit_target=1e9)
    day_blocks = _days([_trade("2026-01-05", 10.0, mae=120.0, mfe=12.0)])

    realized = walk_days(day_blocks, rs, column="optimistic", breach_mode="realized_only")
    assert realized.verdict == "incomplete"
    assert realized.final_balance == 50_200.0

    unrealized = walk_days(
        day_blocks, rs, column="optimistic", breach_mode="unrealized_adverse_first"
    )
    assert unrealized.verdict == "bust"
    assert unrealized.bust_reason == "trailing_floor"
    assert unrealized.days_to_outcome == 1
    assert unrealized.min_floor_distance == pytest.approx(-400.0)  # 47.6k vs 48k


def test_dll_soft_halt_skips_remaining_trades_without_busting():
    """Realized: the halting close stands; remaining trades are skipped; no bust."""
    rs = _ruleset(trail_amount=10_000.0, consistency_pct=None, profit_target=1e9)
    walk = EvaluationWalk(rs, column="optimistic", breach_mode="realized_only")
    verdict = walk.play_day(
        [
            _trade("2026-01-05", -30.0),  # -600, running -600
            _trade("2026-01-05", -30.0),  # -600, running -1200 <= -1000 -> halt
            _trade("2026-01-05", 50.0),  # skipped
        ]
    )
    assert verdict is None
    result = walk.result()
    assert result.verdict == "incomplete"
    assert result.final_balance == 48_800.0  # both closes stand, third skipped
    assert result.skipped_trades == 1
    assert result.halted_days == 1
    # The next day trades again.
    walk.play_day([_trade("2026-01-06", 50.0)])
    assert walk.result().final_balance == 49_800.0


def test_dll_soft_excursion_force_closes_at_the_dll_level():
    """Unrealized: the adverse leg touches the DLL barrier first -> flat AT it."""
    rs = _ruleset(trail_amount=10_000.0, consistency_pct=None, profit_target=1e9)
    walk = EvaluationWalk(rs, column="optimistic", breach_mode="unrealized_adverse_first")
    verdict = walk.play_day(
        [
            _trade("2026-01-05", 25.0, mae=60.0, mfe=30.0),  # dip -1200 crosses -1000
            _trade("2026-01-05", 50.0),  # skipped: day halted
        ]
    )
    assert verdict is None
    result = walk.result()
    assert result.verdict == "incomplete"
    # Force-closed AT the DLL level: the +25pt settle never happens.
    assert result.final_balance == 49_000.0
    assert result.skipped_trades == 1
    assert result.halted_days == 1


def test_dll_hard_is_a_bust():
    rs = _ruleset(dll_soft=False, trail_amount=10_000.0, consistency_pct=None,
                  profit_target=1e9)
    result = walk_days(
        _days([_trade("2026-01-05", -60.0)]),  # -1200 <= -1000 at the close
        rs,
        column="optimistic",
        breach_mode="realized_only",
    )
    assert result.verdict == "bust"
    assert result.bust_reason == "daily_loss_limit"


def test_floor_beats_dll_when_both_barriers_cross_on_the_adverse_leg():
    """If the floor sits ABOVE the DLL level, the falling path hits it first: bust."""
    rs = _ruleset(trail_amount=500.0, consistency_pct=None, profit_target=1e9)
    # floor = 49_500; dll level = 49_000; adverse dip to 48_800 crosses both.
    result = walk_days(
        _days([_trade("2026-01-05", 10.0, mae=60.0)]),
        rs,
        column="optimistic",
        breach_mode="unrealized_adverse_first",
    )
    assert result.verdict == "bust"
    assert result.bust_reason == "trailing_floor"


def test_realized_bust_on_the_trailing_floor_uses_lte():
    """Equity EXACTLY at the floor is a breach."""
    rs = _ruleset(dll_amount=None, consistency_pct=None, profit_target=1e9)
    result = walk_days(
        _days([_trade("2026-01-05", -50.0)], [_trade("2026-01-06", -50.0)]),
        rs,
        column="optimistic",
        breach_mode="realized_only",
    )
    # 50k -> 49k (floor 48k holds) -> 48k == floor -> bust on day 2.
    assert result.verdict == "bust"
    assert result.bust_reason == "trailing_floor"
    assert result.days_to_outcome == 2
    assert result.min_floor_distance == 0.0


def test_consistency_blocks_pass_until_later_days_dilute():
    rs = _ruleset(dll_amount=None)
    walk = EvaluationWalk(rs, column="optimistic", breach_mode="realized_only")
    assert walk.play_day([_trade("2026-01-05", 100.0)]) is None  # +2000; total 2000
    assert walk.play_day([_trade("2026-01-06", 45.0)]) is None  # +900; total 2900 < target
    # Total 3100 >= 3000 BUT best day 2000 > 0.5 * 3100 -> blocked.
    assert walk.play_day([_trade("2026-01-07", 10.0)]) is None
    # Total 3600; best 2000 > 1800 -> still blocked.
    assert walk.play_day([_trade("2026-01-08", 25.0)]) is None
    # Total 4100; best 2000 <= 2050 -> PASS.
    assert walk.play_day([_trade("2026-01-09", 25.0)]) == "pass"
    result = walk.result()
    assert result.verdict == "pass"
    assert result.days_to_outcome == 5
    assert result.best_day_ratio == pytest.approx(2000.0 / 4100.0)


def test_min_days_delays_the_pass():
    rs = _ruleset(dll_amount=None, consistency_pct=None, min_days=3)
    walk = EvaluationWalk(rs, column="optimistic", breach_mode="realized_only")
    assert walk.play_day([_trade("2026-01-05", 200.0)]) is None  # +4000 >= target
    assert walk.play_day([_trade("2026-01-06", 0.0)]) is None
    assert walk.play_day([_trade("2026-01-07", 0.0)]) == "pass"
    assert walk.result().days_to_outcome == 3


def test_conservative_column_is_selectable():
    rs = _ruleset(dll_amount=None, consistency_pct=None, profit_target=1e9)
    trade = TradePath(
        day=date(2026, 1, 5),
        entry_ts=datetime(2026, 1, 5, 14, tzinfo=UTC),
        points_optimistic=15.0,
        points_conservative=14.75,
        mfe_pts=None,
        mae_pts=None,
        resolution="tp_hit",
    )
    optimistic = walk_days(
        [(trade.day, [trade])], rs, column="optimistic", breach_mode="realized_only"
    )
    conservative = walk_days(
        [(trade.day, [trade])], rs, column="conservative", breach_mode="realized_only"
    )
    assert optimistic.final_balance == 50_300.0
    assert conservative.final_balance == 50_295.0


def test_trades_without_excursions_are_realized_only_in_unrealized_mode():
    rs = _ruleset(dll_amount=None, consistency_pct=None, profit_target=1e9)
    result = walk_days(
        _days([_trade("2026-01-05", -50.0, mae=None)]),
        rs,
        column="optimistic",
        breach_mode="unrealized_adverse_first",
    )
    assert result.verdict == "incomplete"
    assert result.final_balance == 49_000.0


def test_empty_walk_reports_no_trades():
    rs = _ruleset()
    result = walk_days([], rs, column="optimistic", breach_mode="realized_only")
    assert result.verdict == "no_trades"
    assert result.days_to_outcome is None
    assert result.min_floor_distance is None


def test_engine_validates_inputs():
    rs = _ruleset()
    with pytest.raises(ValueError, match="Unknown fill column"):
        EvaluationWalk(rs, column="middle", breach_mode="realized_only")
    with pytest.raises(ValueError, match="Unknown breach mode"):
        EvaluationWalk(rs, column="optimistic", breach_mode="psychic")
    with pytest.raises(ValueError, match="Unsupported trail_style"):
        EvaluationWalk(
            _ruleset(trail_style="intraday_trail"),
            column="optimistic",
            breach_mode="realized_only",
        )


def test_terminated_walk_refuses_more_days():
    rs = _ruleset(dll_amount=None, consistency_pct=None)
    walk = EvaluationWalk(rs, column="optimistic", breach_mode="realized_only")
    assert walk.play_day([_trade("2026-01-05", 200.0)]) == "pass"
    with pytest.raises(RuntimeError, match="already terminated"):
        walk.play_day([_trade("2026-01-06", 1.0)])
