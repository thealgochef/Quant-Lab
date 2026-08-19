"""Event-based prop metric builders (CS §5.6) — the walk→vector bridge."""

from __future__ import annotations

from datetime import date

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.propsim.account import (
    AccountPolicySetPayload,
    AccountTrade,
    AccountWalk,
)
from alpha_lab.propsim.calendar import DayCountBasis, DurationRule
from alpha_lab.propsim.firm_contracts import SYNTHETIC_FIXTURE_FIRM
from alpha_lab.propsim.prop_metrics import (
    build_evaluation_fitness,
    build_payout_reliability_vector,
)
from alpha_lab.propsim.risk import FIXED_ONE_NQ_RISK_POLICY
from alpha_lab.propsim.withdrawal import REQUEST_MAX_AT_ELIGIBILITY

_POLICY_SET = AccountPolicySetPayload(
    firm_contract_id=canonical_contract_sha256(SYNTHETIC_FIXTURE_FIRM),
    risk_policy_id=canonical_contract_sha256(FIXED_ONE_NQ_RISK_POLICY),
    withdrawal_policy_id=canonical_contract_sha256(REQUEST_MAX_AT_ELIGIBILITY),
    replacement_policy="none",
    max_replacements=0,
    clock_policy_id="historical_calendar_clock_v1",
)


def _trade(day: str, points: float, *, mae: float | None = None) -> AccountTrade:
    return AccountTrade(
        day=date.fromisoformat(day),
        entry_ts_utc=f"{day}T14:00:00+00:00",
        resolution_ts_utc=f"{day}T14:05:00+00:00",
        points=points,
        risk_points=10.0,
        mfe_pts=None,
        mae_pts=mae,
        trade_id=f"trade-{day}",
    )


def _quick_firm(**payout_overrides):
    quick_eval = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"profit_target": 300.0, "min_days": None, "consistency_pct": None}
    )
    payout = SYNTHETIC_FIXTURE_FIRM.payout.model_copy(
        update={
            "waiting_period": DurationRule(count=1, basis=DayCountBasis.TRADING_DAY),
            "min_between_payouts": DurationRule(count=1, basis=DayCountBasis.TRADING_DAY),
            "min_winning_days": None,
            "min_payout": None,
            **payout_overrides,
        }
    )
    return SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={"evaluation": quick_eval, "payout": payout}
    )


def _walk(firm):
    return AccountWalk(
        firm=firm,
        firm_contract_id=canonical_contract_sha256(firm),
        risk_policy=FIXED_ONE_NQ_RISK_POLICY,
        withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
        policy_set=_POLICY_SET,
        path_instance_id="path-00000-metrics",
        enable_fees=False,
    )


def test_pass_then_funded_breach_still_counts_as_an_evaluation_pass() -> None:
    """The F9 repro: verdict-string heuristics undercounted passes; the
    event stream is the record."""

    firm = _quick_firm()
    walk = _walk(firm)
    walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 20.0)])  # passes → funded
    # funded: one loss straight through the funded trail → breach
    verdict = walk.play_day(date(2026, 1, 14), [_trade("2026-01-14", -120.0)])
    assert verdict == "breached_out"
    result = walk.result()
    fitness = build_evaluation_fitness([result])
    assert fitness.pass_probability == 1.0  # the pass HAPPENED
    assert fitness.median_days_to_pass == pytest.approx(1.0)
    # the breach occurred in the FUNDED phase, not evaluation
    assert fitness.breach_probability == 0.0
    assert fitness.expiration_probability == 0.0


def test_evaluation_breach_and_expiry_are_phase_scoped() -> None:
    firm = _quick_firm()
    breach_walk = _walk(firm)
    breach_walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", -120.0)])
    breached = breach_walk.result()
    assert breached.verdict == "breached_out"
    expiring = SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={
            "account_expiration": DurationRule(count=1, basis=DayCountBasis.TRADING_DAY)
        }
    )
    expire_walk = _walk(expiring)
    expire_walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 5.0)])
    assert expire_walk.play_day(date(2026, 1, 14), [_trade("2026-01-14", 5.0)]) == "expired"
    expired = expire_walk.result()
    fitness = build_evaluation_fitness([breached, expired])
    assert fitness.pass_probability == 0.0
    assert fitness.breach_probability == pytest.approx(0.5)
    assert fitness.expiration_probability == pytest.approx(0.5)
    assert fitness.median_days_to_pass is None


def test_reliability_vector_counts_payouts_and_windows_breaches() -> None:
    firm = _quick_firm()
    walk = _walk(firm)
    walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 20.0)])  # → funded
    for day in ("2026-01-14", "2026-01-15", "2026-01-16"):
        walk.play_day(date.fromisoformat(day), [_trade(day, 25.0)])
    result = walk.result()
    payouts = [e for e in result.events if e.event_type == "payout"]
    assert payouts, "the fixture must actually pay out"
    vector = build_payout_reliability_vector([result])
    assert vector.first_payout_probability_30d == 1.0
    assert vector.first_payout_probability_60d == 1.0
    assert vector.breach_probability_90d == 0.0
    assert vector.expected_net_payout_90d > 0.0
    # a breach beyond the horizon is NOT a 90d breach: shrink the horizon
    # below the breach day and confirm the window excludes it
    breach_walk = _walk(firm)
    breach_walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 5.0)])
    breach_walk.play_day(date(2026, 1, 14), [_trade("2026-01-14", 5.0)])
    verdict = breach_walk.play_day(date(2026, 1, 15), [_trade("2026-01-15", -120.0)])
    assert verdict == "breached_out"
    breached = breach_walk.result()
    windowed = build_payout_reliability_vector([breached], horizon_days_90=2)
    assert windowed.breach_probability_90d == 0.0  # breach landed on day 3
    full = build_payout_reliability_vector([breached])
    assert full.breach_probability_90d == 1.0
