"""Payout race — full version: the adapter that feeds resampled trades to the funded ledger.

Synthetic histories come from the real pair engine and ledger driven by the
scripted strategy of ``tests/propsim/funded/pair_builders.py``; their trade rows
are converted to the saved-result format and replayed through the adapter,
which must reproduce the engine's money exactly. The saved funded variation
study is checked the same way (skipped where it is not on this computer).
Nothing here writes a store.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab import firm_race as fr
from alpha_lab.agents.data_infra.ifvg.presentation.lab import resampling as rs
from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600, iso_utc, to_ns
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES
from tests.propsim.funded.pair_builders import BASE, SyntheticDay, flat, run_pair, weekdays

DAYS = weekdays(date(2026, 3, 2), 14)
MICRO, MILLS = 50, 514

REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
SAVED = (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file()


# ── synthetic histories through the real engine ───────────────────────────


def winner(day: date, gain: int = 600) -> SyntheticDay:
    minutes = flat(6)
    minutes[1] = [BASE - 40, BASE + gain // 2, BASE + gain]
    return SyntheticDay(day, minutes, {0: ("long", BASE - 100, BASE + gain)})


def loser(day: date, drop: int = 450, stop: int = 900) -> SyntheticDay:
    minutes = flat(6)
    minutes[1] = [BASE + 60, BASE - drop // 2, BASE - drop]
    minutes[2] = [BASE]
    return SyntheticDay(day, minutes, {0: ("long", BASE - stop, BASE + 2000)})


def up_then_down(day: date) -> SyntheticDay:
    """Rises 300 ticks first, then falls 250: the best point comes before the worst."""

    minutes = flat(6)
    minutes[1] = [BASE + 300]
    minutes[2] = [BASE - 250]
    minutes[3] = [BASE - 20]
    return SyntheticDay(day, minutes, {0: ("long", BASE - 900, BASE + 2000)})


def quiet(day: date, signal: bool = False) -> SyntheticDay:
    return SyntheticDay(day, flat(6), {0: ("long", BASE - 100, BASE + 100)} if signal else {})


def saved_rows(ledger) -> list[dict]:
    """The ledger's trade rows in the saved-result format (times ISO, money in dollars)."""

    rows = []
    for trade in ledger.trades:
        row = dict(trade)
        for name in ("entry_ns", "exit_ns", "min_equity_ns"):
            row[name[:-3] + "_utc"] = iso_utc(row.pop(name))
        for key in [k for k in row if k.endswith("_cents")]:
            row[key[:-6] + "_usd"] = row.pop(key) / 100
        rows.append(row)
    return rows


def minute_index(inputs) -> fr.MinuteIndex:
    bars = [bar for day in inputs for bar in day.bars_by_tf[60]]
    return fr.MinuteIndex(
        open_ns=np.array([to_ns(b.logical_open_ts_utc) for b in bars], dtype=np.int64),
        high_ticks=np.array([b.high_ticks for b in bars], dtype=np.int64),
        low_ticks=np.array([b.low_ticks for b in bars], dtype=np.int64))


def adapter_for(ledger, built, firm_key, *, quantity=1, tick=500, mills=5140, scale_out=False):
    inputs, schedule, start_ns, cutoff_ns = built
    profile = FIRM_PROFILES[firm_key]
    rules = fr.FirmRules(
        pair_id=ledger.pair_id, configuration="CFG", firm_key=firm_key,
        firm_name=profile.firm_name, profile=profile, processing=TWO_BUSINESS_DAYS_FED_1600,
        quantity=quantity, tick_value_cents=tick, cost_per_contract_mills=mills,
        scale_out=scale_out, trading_days=schedule, start_ns=start_ns, cutoff_ns=cutoff_ns)
    rows = saved_rows(ledger)
    shapes = tuple(fr.trade_shape(r, i, tick_value_cents=tick, mills=mills,
                                  minutes=minute_index(inputs)) for i, r in enumerate(rows))
    return rules, fr.slots_from_rows(rows), shapes, rows


def assert_same_money(original, replayed) -> None:
    assert replayed.receipts == original.receipts
    assert replayed.costs == original.costs
    assert len(replayed.accounts) == len(original.accounts)
    assert ([t["net_pnl_cents"] for t in replayed.trades]
            == [t["net_pnl_cents"] for t in original.trades])
    assert ([t["account_failed"] for t in replayed.trades]
            == [t["account_failed"] for t in original.trades])
    assert (sum(1 for e in replayed.payout_events if e["event"] == "received")
            == sum(1 for e in original.payout_events if e["event"] == "received"))


HISTORY = [winner(DAYS[0]), quiet(DAYS[1], signal=True), quiet(DAYS[2]),
           loser(DAYS[3], drop=420, stop=600), up_then_down(DAYS[4]), loser(DAYS[5]),
           winner(DAYS[6], gain=900), quiet(DAYS[7]), loser(DAYS[8], drop=800),
           winner(DAYS[9], gain=200), quiet(DAYS[10])]


@pytest.mark.parametrize("firm_key", ["takeprofittrader", "myfundedfutures"])
def test_the_original_order_reproduces_the_engine_exactly(firm_key):
    _run, ledger, built = run_pair(HISTORY, firm_key=firm_key)
    assert ledger.receipts > 0 and any(a.status == "failed" for a in ledger.accounts)
    rules, slots, shapes, _rows = adapter_for(ledger, built, firm_key)
    replayed = fr.replay(rules, slots, shapes, range(len(shapes)))
    assert_same_money(ledger, replayed)
    outcome = fr.ledger_outcome(replayed)
    assert outcome.net_cash_cents == ledger.net_cash


def test_half_exit_trades_reproduce_the_engine_exactly():
    scaled = flat(6)
    scaled[1] = [BASE + 50, BASE + 100, BASE + 300]
    for index in range(2, 6):
        scaled[index] = [BASE + 250]
    back_to_entry = flat(6)
    back_to_entry[1] = [BASE - 30, BASE + 100, BASE + 180]
    back_to_entry[2] = [BASE + 60, BASE - 2]
    stopped = flat(6)
    stopped[1] = [BASE + 60, BASE - 120]
    days = [SyntheticDay(DAYS[0], scaled, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[1], back_to_entry, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[2], stopped, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[3], flat(6))]
    for firm_key in ("takeprofittrader", "myfundedfutures"):
        _run, ledger, built = run_pair(days, firm_key=firm_key, quantity=10,
                                       tick_value_cents=MICRO, cost_per_side_cents=0,
                                       cost_per_contract_mills=MILLS, scale_out=True)
        kinds = [t["exit_kind"] for t in ledger.trades]
        assert "scheduled_close" in kinds and "breakeven_stop" in kinds
        rules, slots, shapes, _rows = adapter_for(ledger, built, firm_key, quantity=10,
                                                  tick=MICRO, mills=MILLS, scale_out=True)
        assert shapes[0].scaled and BASE + 100 in shapes[0].path_ticks
        replayed = fr.replay(rules, slots, shapes, range(len(shapes)))
        assert_same_money(ledger, replayed)


def test_the_stored_minutes_pin_which_came_first():
    down_then_up = flat(6)
    down_then_up[1] = [BASE - 250]
    down_then_up[2] = [BASE + 300]
    down_then_up[3] = [BASE + 20]
    days = [up_then_down(DAYS[0]),
            SyntheticDay(DAYS[1], down_then_up, {0: ("long", BASE - 900, BASE + 2000)}),
            quiet(DAYS[2])]
    _run, ledger, built = run_pair(days, firm_key="myfundedfutures")
    first, second = saved_rows(ledger)
    minutes = minute_index(built[0])
    for row, best_first in ((first, True), (second, False)):
        blind = fr.trade_shape(row, 0, tick_value_cents=500, mills=5140)
        seen = fr.trade_shape(row, 0, tick_value_cents=500, mills=5140, minutes=minutes)
        # without the minutes the order is not pinned: best first, a fixed convention
        # (correction A3: not a proven best or worst case)
        assert not blind.order_pinned and blind.path_ticks == (BASE + 300, BASE - 250,
                                                               row["exit_ticks"])
        assert seen.order_pinned
        assert seen.path_ticks[0] == (BASE + 300 if best_first else BASE - 250)
    rules, slots, shapes, _rows = adapter_for(ledger, built, "myfundedfutures")
    assert_same_money(ledger, fr.replay(rules, slots, shapes, range(len(shapes))))


def test_a_stop_exit_at_the_low_puts_the_best_point_first_and_keeps_the_fill():
    _run, ledger, _built = run_pair([loser(DAYS[0], drop=300, stop=300), quiet(DAYS[1])])
    (row,) = saved_rows(ledger)
    shape = fr.trade_shape(row, 0, tick_value_cents=500, mills=5140)
    assert row["exit_kind"] == "stop"
    assert shape.order_pinned and shape.path_ticks == (BASE + 60, row["exit_ticks"])
    # the order's stop and target never trigger by themselves
    assert shape.stop_ticks < min(BASE, BASE + 60) and shape.target_ticks > BASE + 60


# ── resampled orders ──────────────────────────────────────────────────────


def test_the_full_version_uses_the_fixed_floor_race_orders():
    values = [float(v) for v in np.linspace(-500, 900, 40)]
    small = fr.race_orders(len(values), paths=300, seed=7)
    large = fr.race_orders(len(values), paths=2_000, seed=7)
    assert (small == large[:300]).all()
    draws = rs.resample_paths(values, paths=300, length=fr.RACE_DRAW_LENGTH,
                              method="blocks", seed=7)
    assert (np.asarray(values)[small] == draws[:, :len(values)]).all()


@pytest.mark.parametrize("method", ["blocks", "shuffle"])
def test_more_than_200_trades_share_the_first_200_of_every_chart_path(method):
    """Review fix: a 250-trade pair used to share only its first path with the chart."""

    values = np.random.default_rng(5).normal(40.0, 400.0, 250)
    orders = fr.race_orders(len(values), paths=40, seed=20260923, method=method)
    assert orders.shape == (40, 250)
    assert orders.min() >= 0 and orders.max() < 250
    chart = rs.resample_paths(values, paths=40, length=fr.RACE_DRAW_LENGTH, method=method,
                              seed=20260923)
    # every path's first 200 trades are exactly the fixed-floor race's draw
    assert (values[orders[:, :fr.RACE_DRAW_LENGTH]] == chart).all()
    # the other 50 trade slots continue from the derived seed, fixed and reproducible
    tail = rs.resample_paths(values, paths=40, length=50, method=method,
                             seed=fr.extension_seed(20260923))
    assert (values[orders[:, fr.RACE_DRAW_LENGTH:]] == tail).all()
    assert (fr.race_orders(250, paths=40, seed=20260923, method=method) == orders).all()
    more = fr.race_orders(250, paths=400, seed=20260923, method=method)
    assert (more[:40] == orders).all()  # "the first N of the same resampled draws"
    assert fr.extension_seed(20260923) != 20260923
    assert fr.shared_trades(250) == 200 and fr.shared_trades(114) == 114


def test_full_race_is_reproducible_and_a_new_seed_changes_it():
    _run, ledger, built = run_pair(HISTORY)
    rules, slots, shapes, _rows = adapter_for(ledger, built, "takeprofittrader")
    first = fr.full_race(rules, slots, shapes, paths=60, seed=11)
    again = fr.full_race(rules, slots, shapes, paths=60, seed=11)
    other = fr.full_race(rules, slots, shapes, paths=60, seed=12)
    assert replace(first, seconds=0) == replace(again, seconds=0)
    assert replace(first, seconds=0) != replace(other, seconds=0)
    assert first.paid_share + first.died_share + first.still_going_share == pytest.approx(1)
    assert first.accounts_bought >= first.paths
    seen = []
    fr.full_race(rules, slots, shapes, paths=5, seed=3,
                 progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (5, 5)


# ── the saved funded variation study (read only) ──────────────────────────


@pytest.fixture(scope="module")
def saved():
    if not SAVED:
        pytest.skip("the saved funded variation study is not on this computer")
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        open_funded_study,
    )

    study = open_funded_study(STORE, RESULT_ID)
    source = fr.open_source(study.plan)
    if source is None:
        pytest.skip("the verified strategy package is not on this computer")
    frame = market.load_index_minutes(market.study_package_root(study.plan),
                                      cutoff_utc=study.result["period"]["cutoff_utc"])
    return study, source, fr.MinuteIndex.from_frame(frame)


@pytest.mark.parametrize("firm_key", ["takeprofittrader", "myfundedfutures"])
def test_the_saved_leader_replays_exactly(saved, firm_key):
    study, source, minutes = saved
    check = fr.validate_original_order(study, LEADER, firm_key, source=source, minutes=minutes)
    assert check.exact, check.differences
    assert check.trades == check.trades_matching == 114
    assert check.refused_slots == 0 and check.unpinned_shapes == 0
    summary = study.summary(LEADER, firm_key)
    assert check.replayed["net_cash_cents"] == summary["net_cash_earned_cents"]


def test_rules_come_from_the_saved_terms(saved):
    study, source, _minutes = saved
    rules = fr.firm_rules(study, LEADER, "takeprofittrader", source=source)
    assert (rules.quantity, rules.tick_value_cents, rules.cost_per_contract_mills) == (10, 50, 514)
    assert rules.scale_out and rules.loss_allowance == 2000 and rules.trigger == 2600
    assert rules.profile.acquisition_cost_cents == 10_200
    assert len(rules.trading_days) == 107
    assert rules.trading_days[-1].trading_day == "2026-06-10"


def test_the_saved_leader_full_race_is_stable(saved):
    study, source, minutes = saved
    rules, slots, shapes = fr.build_inputs(study, LEADER, "takeprofittrader", source=source,
                                           minutes=minutes)
    race = fr.full_race(rules, slots, shapes, paths=40, seed=20260923)
    again = fr.full_race(rules, slots, shapes, paths=40, seed=20260923)
    assert replace(race, seconds=0) == replace(again, seconds=0)
    assert 0 < race.paid_share < 1 and race.slots == 114


def test_every_saved_configuration_replays_exactly(saved):
    """All 64 configurations at both firms: net cash, payouts received, payout count, accounts,
    account costs, the number of trades, and each trade's net result and account-loss flag in
    order."""

    study, source, minutes = saved
    checked = 0
    for configuration in study.configurations:
        for firm_key in ("takeprofittrader", "myfundedfutures"):
            summary = study.summary(configuration, firm_key)
            if not summary or summary.get("status") != "Completed":
                continue
            check = fr.validate_original_order(study, configuration, firm_key, source=source,
                                               minutes=minutes)
            assert check.exact, (configuration, firm_key, check.differences,
                                 check.trades_matching, check.trades)
            checked += 1
    assert checked == 128
