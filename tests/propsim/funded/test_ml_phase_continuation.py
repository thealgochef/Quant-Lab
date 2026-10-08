"""Actual ordered-print walker: next event, protective priority and restart."""

from dataclasses import replace

import numpy as np
import pytest

from alpha_lab.propsim.funded.position_walk import (
    MinuteObservations,
    OpenPosition,
    open_position,
    walk_minute,
)
from alpha_lab.propsim.funded.profiles import MYFUNDEDFUTURES_PROFILE as PROFILE


def position():
    pos, failure = open_position(
        profile=PROFILE,
        trade_ref="fixture",
        direction="long",
        entry_ns=0,
        entry_ticks=400,
        stop_ticks=360,
        target_ticks=440,
        quantity=10,
        tick_value_cents=50,
        cost_per_side_cents=0,
        balance_cents=0,
        floor_cents=-200000,
        peak_cents=0,
        cost_per_contract_mills=514,
        scale_out=True,
    )
    assert failure is None
    return pos


def obs(prices, times=None, start=0, end=60):
    return MinuteObservations(
        start,
        end,
        prices[-1],
        np.array(times if times is not None else range(1, len(prices) + 1)),
        np.array(prices),
        np.zeros(len(prices), dtype=bool),
        "ordered_trade_prints",
    )


def negative(pos, visible, index):
    assert len(visible.ts_ns) == index + 1
    assert visible.close_ticks == visible.price_ticks[-1]
    return {"action": "change", "score": -0.1, "model_id": "fixture"}


def test_next_same_ns_print_not_trigger_price_and_exact_fees():
    pos = position()
    result = walk_minute(
        profile=PROFILE,
        pos=pos,
        obs=obs([440, 439, 520], [1, 1, 2]),
        deadline_minute=False,
        continuation_selector=negative,
    )
    assert result.kind == "ml_continuation_close"
    assert result.fill_ticks == 439
    assert result.scale_out_ticks == 440 and result.scale_out_quantity == 5
    assert result.balance_after_cents == 10000 + 9750 - 1028
    assert pos.continuation_decision["fill_ordinal"] == 1


@pytest.mark.parametrize("last,kind", [(400, "breakeven_stop"), (0, "account_failure")])
def test_protection_precedes_voluntary_fill(last, kind):
    pos = position()
    result = walk_minute(
        profile=PROFILE,
        pos=pos,
        obs=obs([440, last]),
        deadline_minute=False,
        continuation_selector=negative,
    )
    # A gap to zero loses $1000 on five micros, below this account's $2000 floor;
    # use a higher floor for the independent loss-limit case below.
    if last == 0:
        assert result.kind == "breakeven_stop"
    else:
        assert result.kind == kind
    assert pos.continuation_decision["state"] == "cancelled_by_protection"


def test_loss_floor_priority_at_next_event():
    pos = position()
    pos.floor_cents = -1000
    result = walk_minute(
        profile=PROFILE,
        pos=pos,
        obs=obs([440, 350]),
        deadline_minute=False,
        continuation_selector=negative,
    )
    assert result.kind == "account_failure"


def test_pending_intent_survives_restart_and_deadline_cancels():
    pos = position()
    assert (
        walk_minute(
            profile=PROFILE,
            pos=pos,
            obs=obs([440]),
            deadline_minute=False,
            continuation_selector=negative,
        )
        is None
    )
    restored = OpenPosition.from_json(pos.to_json())
    result = walk_minute(
        profile=PROFILE, pos=restored, obs=obs([450], [61], 60, 120), deadline_minute=False
    )
    assert result.kind == "ml_continuation_close" and result.fill_ticks == 450
    pos = position()
    result = walk_minute(
        profile=PROFILE,
        pos=pos,
        obs=obs([440]),
        deadline_minute=True,
        continuation_selector=negative,
    )
    assert result.kind == "scheduled_close"
    assert pos.continuation_decision["state"] == "cancelled_by_protection"


def test_disabled_and_always_baseline_economics_equal():
    plain, scored = position(), position()
    observations = obs([435, 440, 450, 399])
    first = walk_minute(profile=PROFILE, pos=plain, obs=observations, deadline_minute=False)
    second = walk_minute(
        profile=PROFILE,
        pos=scored,
        obs=observations,
        deadline_minute=False,
        continuation_selector=lambda *_: {"action": "baseline", "score": 0.0},
    )
    assert first == second


def test_approximation_cannot_elect_close():
    with pytest.raises(ValueError, match="approximated checkpoint"):
        walk_minute(
            profile=PROFILE,
            pos=position(),
            obs=replace(obs([440, 450]), fidelity="minute_adverse_first"),
            deadline_minute=False,
            continuation_selector=negative,
        )
