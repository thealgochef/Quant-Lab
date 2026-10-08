"""Repair-contract C01-C03/C12-C14: shared identity and exact saved fills."""

import pytest

from alpha_lab.propsim.funded.reporting_accounts import account_token, resolve_account
from alpha_lab.propsim.funded.reporting_legs import allocate_entry_fee, trade_legs


def _trade(quantity=6, partial=3, rate=514):
    from alpha_lab.propsim.funded.position_walk import fill_cost_cents

    costs = sum(fill_cost_cents(q, rate) for q in (quantity, partial, quantity - partial))
    gross = partial * 20 * 50 + (quantity - partial) * 40 * 50
    return {"quantity": quantity, "scale_out_quantity": partial,
            "final_exit_quantity": quantity - partial, "entry_ticks": 100,
            "scale_out_ticks": 120 if partial else None, "exit_ticks": 140,
            "direction": "long", "tick_value_cents": 50,
            "cost_per_contract_mills": rate, "costs_cents": costs,
            "net_pnl_cents": gross - costs, "exit_kind": "scheduled_close"}


@pytest.mark.parametrize("quantity,partial,total", [(6, 3, 616), (10, 5, 1028), (5, 2, 516)])
def test_actual_fill_rounding_and_conserving_leg_allocation(quantity, partial, total):
    rate = 515 if quantity == 5 else 514
    row = _trade(quantity, partial, rate)
    legs = trade_legs(row)
    assert sum((legs.entry_fee_cents, legs.first_exit_fee_cents,
                legs.final_exit_fee_cents)) == total
    assert legs.first_entry_fee_cents + legs.remaining_entry_fee_cents == legs.entry_fee_cents
    assert legs.first_cents + legs.remaining_cents == row["net_pnl_cents"]
    assert legs.had_partial and legs.outcome == "partial_to_scheduled_close"


def test_allocation_floor_then_residual_does_not_repost_cost():
    assert allocate_entry_fee(257, 5, 2) == (102, 155)
    assert allocate_entry_fee(308, 6, 3) == (154, 154)


def test_whole_deadline_is_not_a_remaining_half():
    legs = trade_legs(_trade(6, 0))
    assert not legs.had_partial
    assert legs.first_cents == 0
    assert legs.remaining_cents == legs.net_cents
    assert legs.outcome == "no_partial"


def test_tampered_posted_fee_or_net_fails_attribution():
    row = _trade()
    row["entry_fee_cents"] = 309
    with pytest.raises(ValueError, match="posted entry_fee"):
        trade_legs(row)
    row.pop("entry_fee_cents")
    row["net_pnl_cents"] += 1
    with pytest.raises(ValueError, match="whole-trade net"):
        trade_legs(row)


def test_scoped_accounts_do_not_collide_across_result_variant_or_firm():
    row = {"configuration": "MCB001", "firm_key": "myfundedfutures", "account_number": 1}
    keys = {resolve_account(row, result_id="R").key,
            resolve_account(row, result_id="S").key,
            resolve_account({**row, "configuration": "MCB025"}, result_id="R").key,
            resolve_account({**row, "firm_key": "takeprofittrader"}, result_id="R").key}
    assert len(keys) == 4


def test_legacy_alias_conflict_is_visible():
    assert account_token({"account_id": "01", "account_number": 1}) == "1"
    assert account_token({"account_id": "legacy-A"}) == "legacy-A"
    assert account_token({}) is None
    with pytest.raises(ValueError, match="account_identity_conflict"):
        account_token({"account_id": "A", "account_number": "B"})
    with pytest.raises(ValueError, match="empty"):
        account_token({"account_number": " "})


def test_held_presenter_retains_two_distinct_deadline_populations():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (
        held_legs_from_rows,
    )

    rows = [_trade(), _trade(6, 0)]
    value = held_legs_from_rows(rows, tick_value_cents=50, mills=514, half_exit=True)
    assert value.reconciled
    assert value.partial_count == value.whole_deadline_count == 1
    assert value.partial_remainders_cents == trade_legs(rows[0]).remaining_cents
    assert value.whole_deadline_cents == rows[1]["net_pnl_cents"]
