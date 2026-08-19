"""Risk-family sizing + skip reasons; withdrawal behaviors + identity (§16.4)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.propsim.calendar import DayCountBasis, DurationRule
from alpha_lab.propsim.risk import (
    FIXED_ONE_NQ_RISK_POLICY,
    PostLossRule,
    PropRiskPolicyPayload,
    RiskPolicyFamily,
    size_position,
)
from alpha_lab.propsim.withdrawal import (
    REQUEST_MAX_AT_ELIGIBILITY,
    WithdrawalPolicyPayload,
    plan_withdrawal_request,
)

_BASE = dict(
    risk_points=10.0,  # 40 ticks
    start_buffer=2_000.0,
    current_buffer=2_000.0,
    day_realized_pnl=0.0,
)


def _policy(**overrides) -> PropRiskPolicyPayload:
    return FIXED_ONE_NQ_RISK_POLICY.model_copy(update=overrides)


def test_every_family_sizes_or_skips_with_exact_reasons() -> None:
    # fixed_dollar: $600 budget / ($10pt × $20) = 3 contracts
    fixed_dollar = _policy(
        family=RiskPolicyFamily.FIXED_DOLLAR, initial_risk_dollars=600.0, fixed_contracts=None
    )
    decision = size_position(fixed_dollar, **_BASE)
    assert (decision.contracts, decision.instrument) == (3, "NQ")
    # pct_start_buffer: 10% × 2000 = $200 → 1 contract
    pct_start = _policy(
        family=RiskPolicyFamily.PCT_START_BUFFER, risk_pct=10.0, fixed_contracts=None
    )
    assert size_position(pct_start, **_BASE).contracts == 1
    # pct_current_buffer scales with the LIVE buffer
    pct_current = _policy(
        family=RiskPolicyFamily.PCT_CURRENT_BUFFER, risk_pct=50.0, fixed_contracts=None
    )
    assert size_position(pct_current, **{**_BASE, "current_buffer": 800.0}).contracts == 2
    # fixed_nq / fixed_mnq / custom_contracts
    assert size_position(_policy(fixed_contracts=2), **_BASE).contracts == 2
    micro = _policy(family=RiskPolicyFamily.FIXED_MNQ, fixed_contracts=4)
    decision = size_position(micro, **_BASE)
    assert (decision.contracts, decision.instrument, decision.point_value) == (4, "MNQ", 2.0)
    custom = _policy(family=RiskPolicyFamily.CUSTOM_CONTRACTS, fixed_contracts=3)
    assert size_position(custom, **_BASE).contracts == 3
    # nq_mnq_adaptive: budget below one NQ falls to micros; below one micro skips
    adaptive = _policy(
        family=RiskPolicyFamily.NQ_MNQ_ADAPTIVE,
        initial_risk_dollars=90.0,
        fixed_contracts=None,
    )
    decision = size_position(adaptive, **_BASE)
    assert (decision.instrument, decision.contracts) == ("MNQ", 4)  # 90/(10×2)=4.5
    tiny = _policy(
        family=RiskPolicyFamily.NQ_MNQ_ADAPTIVE,
        initial_risk_dollars=10.0,
        fixed_contracts=None,
    )
    decision = size_position(tiny, **_BASE)
    assert decision.skipped and decision.skip_reason == "budget_below_one_micro"


def test_skip_reasons_are_typed_and_fail_closed() -> None:
    no_budget = _policy(
        family=RiskPolicyFamily.FIXED_DOLLAR, initial_risk_dollars=None, fixed_contracts=None
    )
    assert size_position(no_budget, **_BASE).skip_reason == "no_risk_budget"
    small = _policy(
        family=RiskPolicyFamily.FIXED_DOLLAR, initial_risk_dollars=100.0, fixed_contracts=None
    )
    assert (
        size_position(small, **_BASE).skip_reason == "min_contract_exceeds_budget"
    )
    forced = small.model_copy(update={"skip_if_min_contract_exceeds_budget": False})
    assert size_position(forced, **_BASE).contracts == 1
    stopped = _policy(daily_stop_dollars=500.0)
    assert (
        size_position(stopped, **{**_BASE, "day_realized_pnl": -600.0}).skip_reason
        == "daily_stop_dollars_reached"
    )
    assert (
        size_position(_policy(), **{**_BASE, "risk_points": 0.0}).skip_reason
        == "nonpositive_risk_points"
    )
    guarded = _policy(min_remaining_buffer_dollars=1_900.0)
    assert (
        size_position(guarded, **_BASE).skip_reason == "min_remaining_buffer"
    )
    capped = _policy(max_current_buffer_usage_pct=5.0)  # 5% × 2000 = $100 < $200
    assert size_position(capped, **_BASE).skip_reason == "buffer_usage_cap"


def test_post_loss_adjustment_and_micro_scaling_and_caps() -> None:
    adjusted = _policy(
        family=RiskPolicyFamily.FIXED_DOLLAR,
        initial_risk_dollars=600.0,
        fixed_contracts=None,
        post_loss_adjustment=PostLossRule(
            consecutive_losses=2, risk_multiplier=0.5, recovery_wins=2
        ),
    )
    # after two consecutive losses the budget halves: 300 → 1 contract
    decision = size_position(adjusted, **_BASE, consecutive_losses=2)
    assert decision.contracts == 1 and decision.risk_budget_dollars == 300.0
    # recovery restores the full budget
    decision = size_position(
        adjusted, **_BASE, consecutive_losses=2, recovery_wins_since=2
    )
    assert decision.contracts == 3
    # post-payout derisk
    derisk = adjusted.model_copy(
        update={"post_loss_adjustment": None, "post_payout_derisk_factor": 0.5}
    )
    assert size_position(derisk, **_BASE, payouts_taken=1).contracts == 1
    # firm micro-scaling table gates by CURRENT buffer; phase cap applies last
    decision = size_position(
        _policy(fixed_contracts=5),
        **{**_BASE, "current_buffer": 1_500.0},
        micro_scaling_table=((1_000.0, 2), (2_000.0, 5)),
    )
    assert decision.contracts == 2
    decision = size_position(
        _policy(fixed_contracts=5),
        **{**_BASE, "current_buffer": 500.0},
        micro_scaling_table=((1_000.0, 2),),
    )
    assert decision.skip_reason == "micro_scaling_zero_allowance"
    assert (
        size_position(_policy(fixed_contracts=5), **_BASE, phase_max_contracts=3).contracts
        == 3
    )


def test_withdrawal_behaviors_and_identity_separation() -> None:
    """P0-15/16: behaviors differ; two policy payloads → two identities."""

    assert (
        plan_withdrawal_request(
            REQUEST_MAX_AT_ELIGIBILITY,
            payout_available=1_500.0,
            current_buffer=2_500.0,
            elapsed_since_last_payout=None,
        )
        == 1_500.0
    )
    retain = WithdrawalPolicyPayload(
        policy_key="retain_1k_v1",
        behavior="retain_minimum_buffer",
        minimum_buffer_retained=1_000.0,
        cadence=None,
        partial_amount=None,
        post_payout_derisk_ref=None,
    )
    assert (
        plan_withdrawal_request(
            retain,
            payout_available=1_500.0,
            current_buffer=2_200.0,
            elapsed_since_last_payout=None,
        )
        == 1_200.0
    )
    cadence = WithdrawalPolicyPayload(
        policy_key="every_10_days_v1",
        behavior="fixed_cadence",
        minimum_buffer_retained=None,
        cadence=DurationRule(count=10, basis=DayCountBasis.TRADING_DAY),
        partial_amount=None,
        post_payout_derisk_ref=None,
    )
    assert (
        plan_withdrawal_request(
            cadence,
            payout_available=1_500.0,
            current_buffer=2_500.0,
            elapsed_since_last_payout=5,
        )
        == 0.0
    )
    assert (
        plan_withdrawal_request(
            cadence,
            payout_available=1_500.0,
            current_buffer=2_500.0,
            elapsed_since_last_payout=10,
        )
        == 1_500.0
    )
    partial = WithdrawalPolicyPayload(
        policy_key="partial_500_v1",
        behavior="partial_fixed_amount",
        minimum_buffer_retained=None,
        cadence=None,
        partial_amount=500.0,
        post_payout_derisk_ref=None,
    )
    assert (
        plan_withdrawal_request(
            partial,
            payout_available=1_500.0,
            current_buffer=2_500.0,
            elapsed_since_last_payout=None,
        )
        == 500.0
    )
    # identity: the two payloads hash differently, so account-policy sets and
    # therefore simulation identities differ (P0-15/P0-16)
    assert canonical_contract_sha256(retain) != canonical_contract_sha256(partial)
    with pytest.raises(Exception, match="minimum_buffer_retained"):
        WithdrawalPolicyPayload(
            policy_key="broken",
            behavior="retain_minimum_buffer",
            minimum_buffer_retained=None,
            cadence=None,
            partial_amount=None,
            post_payout_derisk_ref=None,
        )


def test_remaining_skip_reasons_are_reachable_and_exact() -> None:
    """§16.4 'every risk-family sizing + skip reasons' — the last three."""

    # daily_stop_r_reached: day pnl at or below -r × budget
    stop_r = _policy(
        family=RiskPolicyFamily.FIXED_DOLLAR,
        initial_risk_dollars=400.0,
        fixed_contracts=None,
        daily_stop_r=2.0,
    )
    decision = size_position(stop_r, **{**_BASE, "day_realized_pnl": -900.0})
    assert decision.skipped and decision.skip_reason == "daily_stop_r_reached"
    # zero_contracts_configured: a fixed family configured at zero contracts
    zero = _policy(fixed_contracts=0)
    decision = size_position(zero, **_BASE)
    assert decision.skipped and decision.skip_reason == "zero_contracts_configured"
    # capped_to_zero: sized fine, then the phase cap removes every contract
    capped = _policy(fixed_contracts=2)
    decision = size_position(capped, **{**_BASE, "phase_max_contracts": 0})
    assert decision.skipped and decision.skip_reason == "capped_to_zero"


def test_max_allowed_each_period_requests_the_full_available() -> None:
    policy = WithdrawalPolicyPayload(
        policy_key="max_each_period_v1",
        behavior="max_allowed_each_period",
        minimum_buffer_retained=None,
        cadence=None,
        partial_amount=None,
        post_payout_derisk_ref=None,
    )
    assert (
        plan_withdrawal_request(
            policy,
            payout_available=1_234.0,
            current_buffer=9_999.0,
            elapsed_since_last_payout=7,
        )
        == 1_234.0
    )


def test_withdrawal_policy_change_changes_payout_stream_and_simulation_id() -> None:
    """P0-15: two runs differing ONLY in the withdrawal policy produce
    different simulation identities AND different payout streams."""

    from datetime import date

    from alpha_lab.propsim.account import AccountPolicySetPayload, AccountTrade, AccountWalk
    from alpha_lab.propsim.firm_contracts import SYNTHETIC_FIXTURE_FIRM

    quick_eval = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"profit_target": 300.0, "min_days": None, "consistency_pct": None}
    )
    payout = SYNTHETIC_FIXTURE_FIRM.payout.model_copy(
        update={
            "waiting_period": DurationRule(count=1, basis=DayCountBasis.TRADING_DAY),
            "min_between_payouts": DurationRule(count=1, basis=DayCountBasis.TRADING_DAY),
            "min_winning_days": None,
            "min_payout": None,
        }
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={"evaluation": quick_eval, "payout": payout}
    )
    partial = WithdrawalPolicyPayload(
        policy_key="partial_100_v1",
        behavior="partial_fixed_amount",
        minimum_buffer_retained=None,
        cadence=None,
        partial_amount=100.0,
        post_payout_derisk_ref=None,
    )
    streams = {}
    policy_set_ids = {}
    for label, withdrawal in (("max", REQUEST_MAX_AT_ELIGIBILITY), ("partial", partial)):
        policy_set = AccountPolicySetPayload(
            firm_contract_id=canonical_contract_sha256(firm),
            risk_policy_id=canonical_contract_sha256(FIXED_ONE_NQ_RISK_POLICY),
            withdrawal_policy_id=canonical_contract_sha256(withdrawal),
            replacement_policy="none",
            max_replacements=0,
            clock_policy_id="historical_calendar_clock_v1",
        )
        policy_set_ids[label] = canonical_contract_sha256(policy_set)
        walk = AccountWalk(
            firm=firm,
            firm_contract_id=policy_set.firm_contract_id,
            risk_policy=FIXED_ONE_NQ_RISK_POLICY,
            withdrawal_policy=withdrawal,
            policy_set=policy_set,
            path_instance_id="path-00000-p015",
            enable_fees=False,
        )
        days = ("2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16")
        for day in days:
            walk.play_day(
                date.fromisoformat(day),
                [
                    AccountTrade(
                        day=date.fromisoformat(day),
                        entry_ts_utc=f"{day}T14:00:00+00:00",
                        resolution_ts_utc=f"{day}T14:05:00+00:00",
                        points=25.0,
                        risk_points=10.0,
                        mfe_pts=None,
                        mae_pts=None,
                        trade_id=f"trade-{day}",
                    )
                ],
            )
        streams[label] = tuple(
            (event.event_ts_utc, event.payload.trader_amount)
            for event in walk.result().events
            if event.event_type == "payout"
        )
    # a withdrawal-only change → different policy-set id (→ simulation id) …
    assert policy_set_ids["max"] != policy_set_ids["partial"]
    # … and a genuinely different payout stream
    assert streams["max"] != streams["partial"]
    assert streams["max"] and streams["partial"]
    assert streams["partial"][0][1] < streams["max"][0][1]
