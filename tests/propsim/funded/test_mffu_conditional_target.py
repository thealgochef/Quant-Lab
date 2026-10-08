"""Causal first-target decisions preserve the funded execution order."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_lab.propsim.funded.position_walk import (
    MinuteObservations,
    OpenPosition,
    TargetDecision,
    open_position,
    walk_minute,
)
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

MFFU = FIRM_PROFILES["myfundedfutures"]
ENTRY = 80_000


def _open(*, balance=0, floor=-200_000):
    return open_position(
        profile=MFFU, trade_ref="conditional", direction="long", entry_ns=0,
        entry_ticks=ENTRY, stop_ticks=ENTRY - 100, target_ticks=ENTRY + 100,
        quantity=10, tick_value_cents=50, cost_per_side_cents=0,
        balance_cents=balance, floor_cents=floor, peak_cents=0,
        cost_per_contract_mills=514, target_decision_needed=True,
    )


def _minute(prices, times=None):
    times = list(range(1, len(prices) + 1)) if times is None else times
    return MinuteObservations(
        open_ns=0, close_ns=60, close_ticks=prices[-1],
        ts_ns=np.array(times, dtype=np.int64),
        price_ticks=np.array(prices, dtype=np.int64),
        continuous=np.zeros(len(prices), dtype=bool),
        fidelity="ordered_trade_prints",
    )


def test_whole_branch_uses_the_first_target_observation_once():
    pos, failure = _open()
    assert failure is None
    called = []

    def select(ts_ns):
        called.append(ts_ns)
        return TargetDecision("whole", {"report_id": "report-after-release"})

    exit_ = walk_minute(
        profile=MFFU, pos=pos,
        obs=_minute([ENTRY + 60, ENTRY + 120, ENTRY + 200], [1, 2, 3]),
        deadline_minute=False, target_selector=select,
    )
    assert exit_ is not None and exit_.kind == "target"
    assert exit_.ts_ns == 2 and exit_.fill_ticks == ENTRY + 100
    assert exit_.scale_out_quantity == 0
    assert called == [2]
    assert pos.target_decision == {
        "action": "whole", "context": {"report_id": "report-after-release"},
        "decision_ns": 2, "target_observation_ns": 2,
        "source_fidelity": "ordered_trade_prints",
    }


def test_partial_branch_is_frozen_through_checkpoint_and_later_stop():
    pos, _ = _open()
    first = walk_minute(
        profile=MFFU, pos=pos, obs=_minute([ENTRY + 100]),
        deadline_minute=False,
        target_selector=lambda ts: TargetDecision("partial", {"report_id": "negative"}),
    )
    assert first is None and pos.scaled and pos.remaining_quantity == 5
    resumed = OpenPosition.from_json(pos.to_json())
    assert resumed.target_decision == pos.target_decision
    exit_ = walk_minute(
        profile=MFFU, pos=resumed, obs=_minute([ENTRY - 1]),
        deadline_minute=False,
        target_selector=lambda _ts: pytest.fail("first-target branch ran twice"),
    )
    assert exit_ is not None and exit_.kind == "breakeven_stop"
    assert exit_.scale_out_quantity == 5


def test_floor_or_stop_precedes_context_branch():
    for balance, floor, price in (
        (0, -200_000, ENTRY - 101),
        (10_514, 10_000, ENTRY - 1),
    ):
        pos, failure = _open(balance=balance, floor=floor)
        assert failure is None
        exit_ = walk_minute(
            profile=MFFU, pos=pos, obs=_minute([price, ENTRY + 100]),
            deadline_minute=False,
            target_selector=lambda _ts: pytest.fail("context read after earlier exit"),
        )
        assert exit_ is not None and exit_.kind in {"stop", "account_failure"}
        assert pos.target_decision is None


def test_conditional_target_cannot_fall_back_to_whole_without_selector():
    pos, _ = _open()
    with pytest.raises(ValueError, match="as-of decision selector"):
        walk_minute(
            profile=MFFU, pos=pos, obs=_minute([ENTRY + 100]),
            deadline_minute=False,
        )


def test_approximated_target_uses_context_from_minute_open():
    pos, _ = _open()
    obs = MinuteObservations(
        open_ns=100, close_ns=160, close_ticks=ENTRY + 100,
        ts_ns=np.array([100, 130], dtype=np.int64),
        price_ticks=np.array([ENTRY, ENTRY + 100], dtype=np.int64),
        continuous=np.array([False, True]), fidelity="minute_adverse_first",
    )
    selected = []

    def select(ts_ns):
        selected.append(ts_ns)
        return TargetDecision("whole", {"report_id": "known_at_open"})

    exit_ = walk_minute(
        profile=MFFU, pos=pos, obs=obs, deadline_minute=False,
        target_selector=select,
    )
    assert exit_ is not None and exit_.kind == "target"
    assert selected == [100]
    assert pos.target_decision["target_observation_ns"] == 130


@pytest.mark.parametrize(("action", "exit_kind", "half"), [
    ("whole", "target", False), ("partial", "scheduled_close", True),
])
def test_first_target_on_deadline_minute_finishes_once(action, exit_kind, half):
    pos, failure = _open()
    assert failure is None
    observed = []

    def select(ts_ns):
        observed.append(ts_ns)
        return TargetDecision(action, {"report_id": "selected_at_target"})

    exit_ = walk_minute(
        profile=MFFU, pos=pos, obs=_minute([ENTRY + 100]),
        deadline_minute=True, target_selector=select,
    )
    assert exit_ is not None and exit_.kind == exit_kind
    assert observed == [1]
    assert bool(exit_.scale_out_quantity) is half
    assert pos.entry_cost_cents + pos.partial_cost_cents + pos.exit_cost_cents == 1028
    assert exit_.gross_pnl_cents == 100 * 10 * 50
