"""The stop-difference summary uses the contracts actually closed at the final stop.

Regression for the independent review of result 5fa65149: the summary used the
entry quantity (10 micros) after 5 had already exited, and skipped break-even
stops inside the half-exit minute. The correction is summary-only.
"""

from __future__ import annotations

import copy
from datetime import date

from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
    comparison_headline_figures,
)
from alpha_lab.propsim.funded.comparison_result import (
    STOP_DIFFERENCE_CORRECTION_ID,
    apply_reporting_corrections,
    stop_fill_difference_cents,
)
from tests.propsim.funded.comparison_fixture import comparison_fixture_result
from tests.propsim.funded.pair_builders import BASE, SyntheticDay, flat, run_pair, weekdays

MICRO = 50
DAYS = weekdays(date(2026, 3, 2), 4)


def _trade(**kw):
    row = {"exit_kind": "breakeven_stop", "direction": "long", "quantity": 10,
           "stop_ticks": 102_154, "final_stop_ticks": 102_254, "final_exit_quantity": 5,
           "exit_ticks": 102_253}
    row.update(kw)
    return row


def test_reviewed_example_counts_the_five_remaining_micros():
    # entry-price stop 102,254 ticks, fill 102,253, five micros left: 5 x $0.50
    assert stop_fill_difference_cents(_trade(), MICRO) == 250


def test_better_fills_and_other_exits_count_zero_and_short_is_mirrored():
    assert stop_fill_difference_cents(_trade(exit_ticks=102_255), MICRO) == 0
    assert stop_fill_difference_cents(_trade(exit_kind="scheduled_close"), MICRO) == 0
    assert stop_fill_difference_cents(
        _trade(direction="short", final_stop_ticks=100, exit_ticks=103), MICRO) == 750


def test_rows_saved_before_the_half_exit_use_their_only_stop_and_quantity():
    row = {"exit_kind": "stop", "direction": "long", "quantity": 1, "stop_ticks": 103_577,
           "exit_ticks": 103_576}
    assert stop_fill_difference_cents(row, 500) == 500


def test_same_minute_breakeven_stops_are_counted_by_the_engine_summary():
    same = flat(6)
    same[1] = [BASE + 100, BASE + 30, BASE - 2]  # half, then through the entry, same minute
    later = flat(6)
    later[1] = [BASE + 100]
    later[2] = [BASE + 10, BASE - 3]
    days = [SyntheticDay(DAYS[0], same, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[1], later, {0: ("long", BASE - 100, BASE + 100)}),
            SyntheticDay(DAYS[2], flat(6))]
    _run, ledger, _ = run_pair(days, quantity=10, tick_value_cents=MICRO,
                               cost_per_side_cents=0, cost_per_contract_mills=514,
                               scale_out=True)
    first, second = ledger.trades
    assert first["strategy_recorded_exit_ticks"] is None  # the old summary skipped it
    assert [stop_fill_difference_cents(t, MICRO) for t in ledger.trades] == [
        2 * 5 * MICRO, 3 * 5 * MICRO]


def _corrupted(result):
    """A saved result whose stop summary used the entry quantity (the old defect)."""

    broken = copy.deepcopy(result)
    pair = next(k for k, s in broken["summaries_cents"].items() if s["status"] == "Completed")
    broken["summaries_cents"][pair]["stop_slippage_cents"] += 1_234
    broken["summaries_cents"][pair]["stop_exits_filled_worse_than_stop"] += 1
    return broken, pair


def test_correction_changes_only_the_stop_fields_and_never_money():
    saved = comparison_fixture_result()
    assert apply_reporting_corrections(saved) is saved  # a correct result is untouched
    broken, pair = _corrupted(saved)
    frozen = copy.deepcopy(broken)
    fixed = apply_reporting_corrections(broken)
    assert broken == frozen  # the loaded (saved) object is never modified
    assert fixed["summaries_cents"][pair] == saved["summaries_cents"][pair]
    assert comparison_headline_figures(fixed) == comparison_headline_figures(broken)
    for key in ("net_cash_earned_cents", "payouts_received_cents", "account_costs_cents"):
        assert all(fixed["summaries_cents"][k][key] == broken["summaries_cents"][k][key]
                   for k in broken["summaries_cents"] if key in broken["summaries_cents"][k])
    unchanged = [t for t in broken["tables"] if t != "pair_results"]
    assert all(fixed["tables"][t] == broken["tables"][t] for t in unchanged)
    (record,) = fixed["reporting_corrections"]
    assert record["correction_id"] == STOP_DIFFERENCE_CORRECTION_ID
    assert record["pairs_changed"] == 1 and record["changes"][0]["pair_id"] == pair
    row = next(r for r in fixed["tables"]["pair_results"] if r["pair_id"] == pair)
    assert row["stop_slippage_usd"] == fixed["summaries_cents"][pair][
        "stop_slippage_cents"] / 100
    # idempotent: a corrected copy needs no second correction
    assert apply_reporting_corrections(fixed) is fixed


def test_saved_daily_close_wording_is_corrected_without_touching_money():
    from alpha_lab.propsim.funded.comparison_result import DAILY_CLOSE_WORDING_CORRECTION_ID

    saved = copy.deepcopy(comparison_fixture_result())
    config = saved["tables"]["configurations"][0]
    config["settings"] = [*config["settings"], {
        "setting": "Daily close", "value": "All positions closed by 3:55 PM Chicago (earlier "
        "on shortened days); no overnight or weekend holding"}]
    fixed = apply_reporting_corrections(saved)
    (record,) = fixed["reporting_corrections"]
    assert record["correction_id"] == DAILY_CLOSE_WORDING_CORRECTION_ID
    value = [s["value"] for s in fixed["tables"]["configurations"][0]["settings"]
             if s["setting"] == "Daily close"][0]
    assert "no overnight" not in value and "past midnight inside one trading day" in value
    assert "3:55 PM Chicago" in value
    assert fixed["summaries_cents"] == saved["summaries_cents"]
