"""The account seams against the real, unmodified Strategy-Core (pinned commit).

Uses the verified September 18 daily-close study (read only) and its trusted
cached day artifacts for the control configuration's first days: ten warmup
days plus the first evaluation days. Skipped when that local archive is absent.
"""

from __future__ import annotations

from datetime import date

import pytest

from alpha_lab.propsim.funded.comparison_source import (
    core_trade_key,
    discover_comparison_sources,
    iter_days,
    reference_trades,
    resolve_configuration,
)
from alpha_lab.propsim.funded.pair_ledger import BLOCK_REASONS
from alpha_lab.propsim.funded.print_minutes import bar_window_ns
from alpha_lab.propsim.funded.strategy_driver import CoreStrategyDriver

CONTROL = "S0_D160_W1_P0"
EVAL_DAYS = 6


@pytest.fixture(scope="module")
def control():
    sources = [s for s in discover_comparison_sources() if CONTROL in s.by_name]
    if not sources:
        pytest.skip("the verified daily-close study archive is not available locally")
    source = sources[0]
    config = source.by_name[CONTROL]
    section, cfg = resolve_configuration(config.axis_value_ids)
    days = []
    for day in iter_days(source, config, cfg=cfg):
        days.append(day)
        if sum(d.is_evaluation for d in days) == EVAL_DAYS:
            break
    return source, section, cfg, days


def _drive(section, cfg, days, *, gate_for=None, force_flat_first=False):
    driver = CoreStrategyDriver(section, tick_size=cfg.tick_size)
    trades, entries, refused = [], [], 0
    driver.also_blocked = 0
    flatten_next = False
    flattened = False
    for day in days:
        for bar in driver.begin_day(day.bars_by_tf, day.levels_for):
            if flatten_next and driver.in_position:
                out = driver.step(bar, None)
                if not out.core_exits:
                    driver.force_flat()
                flatten_next = False
                trades.extend(out.core_exits)
                continue
            gate = gate_for(day, bar) if gate_for else None
            out = driver.step(bar, gate)
            trades.extend(out.core_exits)
            driver.also_blocked += out.also_blocked_by_strategy
            if out.refused:
                refused += 1
                driver.discard_refused_setup()
            if out.entry is not None:
                entries.append((day.trading_day, bar_window_ns(bar)[1], out.entry))
                if force_flat_first and day.is_evaluation and not flattened:
                    flatten_next = flattened = True
        trades.extend(driver.end_day(date.fromisoformat(day.trading_day)))
    return driver, trades, entries, refused


def _keys(records):
    return [core_trade_key({"entry_ts_utc": r.entry_ts_utc, "entry_ticks": int(r.entry_ticks),
                            "stop_ticks": int(r.stop_ticks), "resolution": str(r.resolution),
                            "exit_ticks": int(r.exit_ticks)}) for r in records]


def test_no_account_driver_reproduces_the_saved_study_on_these_days(control):
    source, section, cfg, days = control
    _driver, trades, entries, _ = _drive(section, cfg, days)
    last = days[-1].trading_day
    expected = [k for k in reference_trades(source, CONTROL) if k[0][:10] <= last]
    # trades whose entry falls on the last included day are compared too (daily close)
    assert _keys(trades) == expected[: len(trades)]
    assert len(trades) == len(expected) and len(trades) >= 2
    assert all(e[2].direction == "long" for e in entries)


def test_payout_refusal_blocks_every_evaluation_entry_and_discards_the_setup(control):
    _source, section, cfg, days = control
    refusal = BLOCK_REASONS["processing"]
    _driver, _trades, entries, refused = _drive(
        section, cfg, days, gate_for=lambda day, _bar: refusal if day.is_evaluation else None)
    assert not [e for e in entries if e[0] >= days[-EVAL_DAYS].trading_day]
    assert refused >= 1  # at least one real entry signal was refused and disarmed


def test_account_liquidation_clears_the_strategy_position_and_it_continues(control):
    _source, section, cfg, days = control
    baseline_driver, baseline_trades, baseline_entries, _ = _drive(section, cfg, days)
    driver, trades, entries, _ = _drive(section, cfg, days, force_flat_first=True)
    assert driver.forced_flat == 1
    first_eval = next(e for e in entries if e[0] >= days[-EVAL_DAYS].trading_day)
    # the forced trade never resolves in the strategy's own records
    assert first_eval[2].trade_id not in {str(r.trade_id) for r in trades}
    # the strategy keeps running and can enter again after the liquidation
    assert any(e[1] > first_eval[1] for e in entries)
    assert baseline_driver.forced_flat == 0 and len(baseline_trades) >= len(trades)


def test_driver_checkpoint_resumes_to_the_same_seed(control):
    _source, section, cfg, days = control
    straight, _t, _e, _r = _drive(section, cfg, days)
    first = CoreStrategyDriver(section, tick_size=cfg.tick_size)
    half = len(days) // 2
    for day in days[:half]:
        for bar in first.begin_day(day.bars_by_tf, day.levels_for):
            first.step(bar, None)
        first.end_day(date.fromisoformat(day.trading_day))
    resumed = CoreStrategyDriver(section, tick_size=cfg.tick_size)
    resumed.restore(first.checkpoint())
    for day in days[half:]:
        for bar in resumed.begin_day(day.bars_by_tf, day.levels_for):
            resumed.step(bar, None)
        resumed.end_day(date.fromisoformat(day.trading_day))
    assert resumed.seed_hash() == straight.seed_hash()



def test_candidates_core_blocks_itself_never_count_as_refusals_or_discard(control):
    """Review finding B1: during a payout pause, a candidate Core blocks for its own
    reasons (an unratified retest, outside the entry hours) is not a refused entry."""
    _source, section, cfg, days = control
    refusal = BLOCK_REASONS["processing"]
    driver, _trades, _entries, refused = _drive(
        section, cfg, days, gate_for=lambda day, _bar: refusal if day.is_evaluation else None)
    assert driver.also_blocked >= 1  # such candidates occur on these real days
    assert driver.discarded_setups == refused  # only sole-reason refusals discard
    # with no pause the same days produce entries where the pause refused them
    _d, _t, baseline_entries, _ = _drive(section, cfg, days)
    evaluation_entries = [e for e in baseline_entries if e[0] >= days[-EVAL_DAYS].trading_day]
    assert refused >= 1 and refused <= len(evaluation_entries) + driver.also_blocked
