"""Scale-out exit at the account level: 10 micros, 5 exit at the target, 5 held.

Micro Nasdaq-100: $0.50 per tick per contract; cost $0.514 per contract per fill
(owner choice, September 23, 2026), kept exact in tenths of a cent.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from alpha_lab.propsim.funded.position_walk import (
    MinuteObservations,
    fill_cost_cents,
    minute_approximation,
    open_position,
    walk_minute,
)
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES
from tests.propsim.funded.pair_builders import BASE, SyntheticDay, flat, run_pair, weekdays

TPT = FIRM_PROFILES["takeprofittrader"]
MICRO = 50  # cents per tick per contract
MILLS = 514  # $0.514 per contract per fill
DAYS = weekdays(date(2026, 3, 2), 6)


def _obs(prices, *, continuous=None, fidelity="ordered_trade_prints"):
    n = len(prices)
    return MinuteObservations(
        open_ns=0, close_ns=60, close_ticks=prices[-1],
        ts_ns=np.arange(n, dtype=np.int64) + 1, price_ticks=np.array(prices, dtype=np.int64),
        continuous=np.zeros(n, dtype=bool) if continuous is None else np.array(continuous),
        fidelity=fidelity)


def _open(*, balance=0, floor=-200_000, stop=BASE - 100, target=BASE + 100):
    return open_position(
        profile=TPT, trade_ref="t", direction="long", entry_ns=0, entry_ticks=BASE,
        stop_ticks=stop, target_ticks=target, quantity=10, tick_value_cents=MICRO,
        cost_per_side_cents=0, balance_cents=balance, floor_cents=floor, peak_cents=0,
        cost_per_contract_mills=MILLS, scale_out=True)


def test_exact_micro_costs():
    assert fill_cost_cents(10, MILLS) == 514 and fill_cost_cents(5, MILLS) == 257
    with pytest.raises(ValueError):
        fill_cost_cents(3, MILLS)  # $1.542 is not a whole number of cents


def test_half_at_target_then_breakeven_on_a_later_minute():
    pos, failure = _open()
    assert failure is None and pos.balance_cents == -514
    assert walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 60, BASE + 100]),
                       deadline_minute=False) is None
    # 5 micros x 100 ticks x $0.50 = $250 gross, less $2.57
    assert pos.scaled and pos.stop_ticks == BASE and pos.remaining_quantity == 5
    assert pos.balance_cents == -514 + 25_000 - 257
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 30, BASE - 2]),
                        deadline_minute=False)
    assert exit_.kind == "breakeven_stop" and exit_.fill_ticks == BASE - 2  # stop-market
    assert exit_.scale_out_ticks == BASE + 100 and exit_.scale_out_quantity == 5
    assert not exit_.scaled_in_exit_minute
    assert exit_.gross_pnl_cents == 25_000 + (-2 * MICRO * 5)
    assert exit_.balance_after_cents == -514 + 25_000 - 257 - 500 - 257


def test_breakeven_inside_the_scale_out_minute_is_flagged():
    pos, _ = _open()
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 100, BASE + 20, BASE]),
                        deadline_minute=False)
    assert exit_.kind == "breakeven_stop" and exit_.fill_ticks == BASE
    assert exit_.scaled_in_exit_minute


def test_stop_before_target_exits_all_ten():
    pos, _ = _open()
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE - 101, BASE + 200]),
                        deadline_minute=False)
    assert exit_.kind == "stop" and exit_.fill_ticks == BASE - 101
    assert exit_.gross_pnl_cents == -101 * MICRO * 10 and exit_.scale_out_quantity == 0


def test_held_half_runs_to_the_deadline():
    pos, _ = _open()
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 100]), deadline_minute=False)
    assert walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 400]), deadline_minute=False) \
        is None  # no second target: the rest is held
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 450, BASE + 480]),
                        deadline_minute=True)
    assert exit_.kind == "scheduled_close" and exit_.fill_ticks == BASE + 480
    assert exit_.gross_pnl_cents == 100 * MICRO * 5 + 480 * MICRO * 5


def test_loss_limit_on_the_remaining_half_after_the_scale_out():
    # TakeProfitTrader floor locked at $0 with $10 of room on a scaled position
    pos, _ = _open(balance=1_514, floor=0, stop=BASE - 900, target=BASE + 10)
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 10]), deadline_minute=False)
    # balance 1,514 - 514 + 5*10*50 - 257 = 3,243; break-even stop at BASE; the account
    # fails only if equity reaches 0, which the break-even stop prevents first
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE - 1]), deadline_minute=False)
    assert exit_.kind == "breakeven_stop" and not exit_.account_failed


def test_approximated_minute_scales_on_the_continuous_leg():
    pos, _ = _open()
    obs = minute_approximation(0, 60, BASE, BASE + 150, BASE - 10, BASE + 5, sign=1)
    # losing side first: BASE -> BASE-10 -> BASE+150 (target at +100) -> close +5
    assert walk_minute(profile=TPT, pos=pos, obs=obs, deadline_minute=False) is None
    assert pos.scaled and pos.scale_out_ticks == BASE + 100


def test_pair_engine_with_the_scripted_scale_out_strategy():
    # day 1: target minute, then the price returns to entry inside the SAME minute
    same = flat(6)
    same[1] = [BASE + 100, BASE + 30, BASE]
    # day 2: target minute, break-even touched on a LATER minute
    later = flat(6)
    later[1] = [BASE + 100]
    later[2] = [BASE + 10, BASE - 1]
    days = [SyntheticDay(DAYS[0], same, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[1], later, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[2], flat(6))]
    run, ledger, _ = run_pair(days, quantity=10, tick_value_cents=MICRO,
                              cost_per_side_cents=0, cost_per_contract_mills=MILLS,
                              scale_out=True)
    first, second = ledger.trades
    assert first["exit_kind"] == second["exit_kind"] == "breakeven_stop"
    assert first["strategy_recorded_exit_kind"] == "breakeven_stop_within_the_scale_out_minute"
    assert second["strategy_recorded_exit_kind"] == "breakeven_stop"
    assert first["scale_out_quantity"] == 5 and first["final_exit_quantity"] == 5
    assert first["costs_cents"] == 514 + 257 + 257
    assert run.driver.forced_flat == 0  # not an account liquidation
    assert len(ledger.accounts) == 1


def test_crossing_print_counts_for_the_takeprofittrader_peak_after_the_scale_out():
    # review round 2, B4: the print that gaps through the target stays in the path
    pos, _ = _open(stop=BASE - 900, target=BASE + 100)
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 140]), deadline_minute=False)
    # 5 remaining micros marked at +140 ticks raise the peak (not only the +100 fill)
    expected_peak = -514 + 25_000 - 257 + 140 * MICRO * 5
    assert pos.peak_cents == expected_peak


def test_scaled_trade_row_keeps_the_initial_stop_and_counts_breakeven_gaps():
    later = flat(6)
    later[1] = [BASE + 100]
    later[2] = [BASE + 10, BASE - 3]  # break-even stop gaps 3 ticks through the entry
    days = [SyntheticDay(DAYS[0], later, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[1], flat(6))]
    _run, ledger, _ = run_pair(days, quantity=10, tick_value_cents=MICRO,
                               cost_per_side_cents=0, cost_per_contract_mills=MILLS,
                               scale_out=True)
    (trade,) = ledger.trades
    assert trade["stop_ticks"] == BASE - 100 and trade["final_stop_ticks"] == BASE
    assert trade["exit_kind"] == "breakeven_stop" and trade["exit_ticks"] == BASE - 3
