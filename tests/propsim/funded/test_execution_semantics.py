"""Funded execution policy ``ordered_prints_stop_market_v2`` with the half exit.

Synthetic checks of the documented funded-account semantics (SPEC.md A11). They
are separate from Strategy-Core's no-account semantics, where the break-even stop
is checked only from the candle after the half exit.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_lab.propsim.funded.position_walk import (
    MinuteObservations,
    open_position,
    walk_minute,
)
from alpha_lab.propsim.funded.price_evidence import load_print_day
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES
from tests.propsim.funded.pair_builders import BASE, SyntheticDay, flat, run_pair, weekdays

TPT = FIRM_PROFILES["takeprofittrader"]
MICRO = 50
MILLS = 514
DAYS = weekdays(date(2026, 3, 2), 4)


def _obs(prices, ts=None):
    n = len(prices)
    return MinuteObservations(
        open_ns=0, close_ns=60, close_ticks=prices[-1],
        ts_ns=np.array(ts if ts is not None else range(1, n + 1), dtype=np.int64),
        price_ticks=np.array(prices, dtype=np.int64), continuous=np.zeros(n, dtype=bool),
        fidelity="ordered_trade_prints")


def _open(*, balance=0, floor=-200_000, stop=BASE - 100, target=BASE + 100):
    return open_position(
        profile=TPT, trade_ref="t", direction="long", entry_ns=0, entry_ticks=BASE,
        stop_ticks=stop, target_ticks=target, quantity=10, tick_value_cents=MICRO,
        cost_per_side_cents=0, balance_cents=balance, floor_cents=floor, peak_cents=0,
        cost_per_contract_mills=MILLS, scale_out=True)


def test_same_timestamp_prints_follow_their_recorded_order():
    # one matching event: the target print and a print through the entry share a timestamp
    pos, _ = _open()
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 100, BASE - 1], ts=[5, 5]),
                        deadline_minute=False)
    assert exit_.kind == "breakeven_stop" and exit_.fill_ticks == BASE - 1
    assert exit_.scaled_in_exit_minute
    # the reverse recorded order: the dip does not reach the original stop, then the half
    pos, _ = _open()
    assert walk_minute(profile=TPT, pos=pos, obs=_obs([BASE - 1, BASE + 100], ts=[5, 5]),
                       deadline_minute=False) is None
    assert pos.scaled and pos.remaining_quantity == 5


def test_print_files_order_by_timestamp_then_sequence_then_file_row(tmp_path):
    folder = tmp_path / "2026-03-02"
    folder.mkdir()
    ts = [2_000, 1_000, 1_000, 1_000]
    table = pa.table({
        "ts_event": pa.array(ts, pa.timestamp("ns", "UTC")),
        "action": ["T"] * 4, "price": [100.0, 101.0, 100.5, 100.25],
        "instrument_id": [7] * 4, "symbol": ["NQH6"] * 4,
        "sequence": [9, 5, 4, 5]})
    pq.write_table(table, folder / "mbp1.parquet")
    day = load_print_day(tmp_path, date(2026, 3, 2))
    # sequence 4 first; the two sequence-5 prints keep their file order; then ts 2,000
    assert list(day.ts_ns) == [1_000, 1_000, 1_000, 2_000]
    assert list(day.ticks) == [402, 404, 401, 400]


def test_partial_fee_remaining_mark_and_intraday_floor():
    pos, _ = _open()
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 100]), deadline_minute=False)
    # half realized: 5 x 100 ticks x $0.50 less 5 x $0.514 on top of the $5.14 entry cost
    assert pos.balance_cents == -514 + 25_000 - 257
    assert pos.value_cents == 5 * MICRO and pos.exit_cost_cents == 257
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 300]), deadline_minute=False)
    # the 5 remaining contracts are marked; the TakeProfitTrader floor follows the peak
    peak = -514 + 25_000 - 257 + 300 * 5 * MICRO
    assert pos.peak_cents == peak and pos.floor_cents == peak - 200_000


def test_account_breach_on_the_remaining_half_before_the_breakeven_stop():
    # a -$1,900 account: the remainder's rise raises the floor above the entry-price equity
    pos, _ = _open(balance=-190_000)
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 100]), deadline_minute=False)
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 900]), deadline_minute=False)
    realized = -190_000 - 514 + 25_000 - 257
    floor = realized + 900 * 5 * MICRO - 200_000
    assert pos.floor_cents == floor
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 50, BASE - 5]),
                        deadline_minute=False)
    assert exit_.kind == "account_failure" and exit_.fill_ticks == BASE + 50
    assert exit_.account_failed and exit_.scale_out_quantity == 5


def test_breakeven_stop_gaps_through_the_entry_and_fills_at_the_print():
    pos, _ = _open()
    walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 100]), deadline_minute=False)
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 4, BASE - 6]),
                        deadline_minute=False)
    assert exit_.kind == "breakeven_stop" and exit_.fill_ticks == BASE - 6


def _big_half_day():
    # the half alone realizes 5 x 1,100 ticks x $0.50 = $2,750: above $2,100 + $500
    minutes = flat(6, BASE + 1100)
    minutes[0] = [BASE]
    return [SyntheticDay(DAYS[0], minutes, {0: ("long", BASE - 1100, BASE + 1100)}),
            SyntheticDay(DAYS[1], flat(6))]


def test_realized_half_with_an_open_remainder_never_locks_a_payout():
    _run, ledger, _ = run_pair(_big_half_day(), quantity=10, tick_value_cents=MICRO,
                               cost_per_side_cents=0, cost_per_contract_mills=MILLS,
                               scale_out=True)
    (trade,) = ledger.trades
    assert trade["scale_out_ns"] < trade["exit_ns"]
    secured = [e for e in ledger.account_events if e.get("status_after") == "secured"]
    # eligibility only once the whole position is flat, at the final exit
    assert [e["ts_ns"] for e in secured] == [trade["exit_ns"]]
    assert all(e["ts_ns"] >= trade["exit_ns"] for e in ledger.payout_events)
    requested = [p for p in ledger.payout_events if p["event"] == "requested"]
    assert len(requested) == 1 and requested[0]["balance_after_cents"] == 210_000
