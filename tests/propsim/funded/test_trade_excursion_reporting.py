"""Trade price-path evidence is separate from account equity and unchanged economics."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from alpha_lab.propsim.funded.campaign import machine_time
from alpha_lab.propsim.funded.full_range_reporting import _funded_trade
from alpha_lab.propsim.funded.pair_engine import PairRun, run_days
from alpha_lab.propsim.funded.position_walk import (
    MinuteObservations,
    OpenPosition,
    minute_approximation,
    open_position,
    walk_minute,
)
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES
from tests.propsim.funded.pair_builders import (
    BASE,
    ScriptedDriver,
    ScriptedPrints,
    SyntheticDay,
    build,
    flat,
    ledger_for,
    run_pair,
    weekdays,
)

MFFU = FIRM_PROFILES["myfundedfutures"]


def _observations(prices: list[int], start_ns: int = 1) -> MinuteObservations:
    ts = np.arange(start_ns, start_ns + len(prices), dtype=np.int64)
    return MinuteObservations(
        open_ns=start_ns - 1, close_ns=start_ns + 59,
        close_ticks=prices[-1], ts_ns=ts,
        price_ticks=np.array(prices, dtype=np.int64),
        continuous=np.zeros(len(prices), dtype=bool), fidelity="ordered_trade_prints",
    )


def _open_position() -> OpenPosition:
    pos, failure = open_position(
        profile=MFFU, trade_ref="path", direction="long", entry_ns=0,
        entry_ticks=BASE, stop_ticks=BASE - 100, target_ticks=BASE + 100,
        quantity=10, tick_value_cents=50, cost_per_side_cents=0,
        balance_cents=0, floor_cents=-200_000, peak_cents=0,
        cost_per_contract_mills=514, scale_out=True,
    )
    assert failure is None
    return pos


def test_ordered_print_jump_is_capped_at_target_before_partial_then_seen_by_remainder():
    pos = _open_position()
    assert walk_minute(profile=MFFU, pos=pos,
                       obs=_observations([BASE + 60, BASE + 140]),
                       deadline_minute=False) is None
    assert pos.pre_target_price_max_ticks == BASE + 100
    assert pos.pre_target_price_max_ns == 2
    assert pos.post_target_price_max_ticks == BASE + 140
    assert pos.post_target_price_max_ns == 2
    result = walk_minute(profile=MFFU, pos=pos,
                         obs=_observations([BASE + 40, BASE - 2], 61),
                         deadline_minute=False)
    assert result is not None and result.kind == "breakeven_stop"
    assert (pos.price_min_ticks, pos.price_min_ns) == (BASE - 2, 62)
    assert (pos.price_max_ticks, pos.price_max_ns) == (BASE + 140, 2)
    assert (pos.post_target_price_min_ticks, pos.post_target_price_min_ns) == (BASE - 2, 62)
    assert pos.min_equity_cents != pos.price_min_ticks
    assert pos.to_json() == OpenPosition.from_json(pos.to_json()).to_json()


def test_whole_target_jump_never_claims_unheld_price_and_approximation_is_labeled():
    pos, _ = open_position(
        profile=MFFU, trade_ref="whole", direction="long", entry_ns=0,
        entry_ticks=BASE, stop_ticks=BASE - 100, target_ticks=BASE + 100,
        quantity=10, tick_value_cents=50, cost_per_side_cents=0,
        balance_cents=0, floor_cents=-200_000, peak_cents=0,
        cost_per_contract_mills=514,
    )
    result = walk_minute(profile=MFFU, pos=pos,
                         obs=_observations([BASE + 140, BASE + 300]),
                         deadline_minute=False)
    assert result is not None and result.fill_ticks == BASE + 100
    assert pos.price_max_ticks == BASE + 100
    assert pos.post_target_price_max_ticks is None
    approx = _open_position()
    obs = minute_approximation(0, 60, BASE, BASE + 150, BASE - 10, BASE + 5, 1)
    assert walk_minute(profile=MFFU, pos=approx, obs=obs, deadline_minute=False) is None
    assert approx.pre_target_price_max_ticks == BASE + 100
    assert approx.post_target_price_max_ticks == BASE + 150
    assert approx.price_min_ticks == BASE - 10
    assert approx.minutes_approximated == 1


def test_short_stop_gap_uses_executable_print_and_ignores_later_prices():
    pos, _ = open_position(
        profile=MFFU, trade_ref="short-gap", direction="short", entry_ns=0,
        entry_ticks=BASE, stop_ticks=BASE + 100, target_ticks=BASE - 100,
        quantity=10, tick_value_cents=50, cost_per_side_cents=0,
        balance_cents=0, floor_cents=-200_000, peak_cents=0,
        cost_per_contract_mills=514,
    )
    result = walk_minute(profile=MFFU, pos=pos,
                         obs=_observations([BASE - 20, BASE + 103, BASE - 500]),
                         deadline_minute=False)
    assert result is not None and result.kind == "stop" and result.fill_ticks == BASE + 103
    assert (pos.price_min_ticks, pos.price_min_ns) == (BASE - 20, 1)
    assert (pos.price_max_ticks, pos.price_max_ns) == (BASE + 103, 2)


def test_original_stop_r_and_price_fields_survive_funded_trade_reporting():
    day = weekdays(date(2026, 3, 2), 2)
    minute = flat(6)
    minute[1] = [BASE + 140]
    minute[2] = [BASE + 40, BASE - 3]
    _run, ledger, _ = run_pair(
        [SyntheticDay(day[0], minute, {0: ("long", BASE - 100, BASE + 100)}),
         SyntheticDay(day[1], flat(6))],
        firm_key="myfundedfutures", quantity=10, tick_value_cents=50,
        cost_per_side_cents=0, cost_per_contract_mills=514, scale_out=True,
    )
    (trade,) = ledger.trades
    assert trade["initial_risk_cents"] == 50_000
    assert trade["gross_pnl_cents"] == 25_000 - 3 * 50 * 5
    assert trade["costs_cents"] == 514 + 257 + 257
    assert trade["gross_r"] == round(trade["gross_pnl_cents"] / 50_000, 8)
    assert trade["net_r"] == round(trade["net_pnl_cents"] / 50_000, 8)
    assert trade["pre_target_price_max_ticks"] == BASE + 100
    assert trade["post_target_price_max_ticks"] == BASE + 140
    assert trade["favorable_excursion_ticks"] == 140
    assert trade["adverse_excursion_ticks"] == 3
    row = {
        **trade, "configuration": "MCB001", "firm_key": "myfundedfutures",
        "entry_utc": machine_time(trade["entry_ns"]),
        "exit_utc": machine_time(trade["exit_ns"]),
        "gross_pnl_usd": trade["gross_pnl_cents"] / 100,
        "costs_usd": trade["costs_cents"] / 100,
        "net_pnl_usd": trade["net_pnl_cents"] / 100,
    }
    size = {"MCB001": {"instrument": "micro", "tick_value_cents": 50,
                       "cost_per_contract_mills": 514}}
    reported = _funded_trade(row, size)
    assert reported["gross_r"] == trade["gross_r"]
    assert reported["net_r"] == trade["net_r"]
    assert reported["price_excursion_status"] == "available"
    assert reported["price_max_utc"] == machine_time(trade["price_max_ns"])
    assert reported["htf_zone_id"] is None
    old = dict(row)
    for key in list(old):
        if key.startswith(("price_", "pre_target_price_", "post_target_price_")):
            old.pop(key)
    reused = _funded_trade(old, size, reused=True)
    assert reused["price_excursion_status"] == "unavailable_legacy_reuse"
    assert reused["price_max_ticks"] is None
    with pytest.raises(ValueError, match="complete price path"):
        _funded_trade(old, size)


def test_old_mid_position_checkpoint_cannot_claim_complete_excursion():
    old = _open_position().to_json()
    for key in list(old):
        if key.startswith(("price_", "pre_target_price_", "post_target_price_")):
            old.pop(key)
    restored = OpenPosition.from_json(old)
    assert restored.price_excursion_status == "unavailable_checkpoint_history"


def test_entry_zone_id_passes_pair_engine_into_saved_funded_trade():
    class ZoneDriver(ScriptedDriver):
        def step(self, bar, gate):
            outcome = super().step(bar, gate)
            if outcome.entry is None:
                return outcome
            return replace(outcome, entry=replace(outcome.entry, htf_zone_id="zone-123"))

    day = weekdays(date(2026, 3, 2), 2)
    minutes = flat(6)
    minutes[1] = [BASE + 100]
    inputs, schedule, start_ns, cutoff_ns = build([
        SyntheticDay(day[0], minutes, {0: ("long", BASE - 100, BASE + 100)}),
        SyntheticDay(day[1], flat(6)),
    ])
    ledger = ledger_for("myfundedfutures", schedule, start_ns, cutoff_ns)
    run = PairRun(ledger.pair_id, ZoneDriver(), ledger)
    prints = ScriptedPrints()
    run_days([run], inputs, lambda _day: (lambda: prints))
    ledger.finish()
    assert ledger.trades[0]["htf_zone_id"] == "zone-123"
