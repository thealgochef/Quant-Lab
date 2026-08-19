"""Rule-by-rule AccountWalk fixtures (§16.4) + EvaluationWalk parity + ordering.

One deterministic hand-built stream per rule; the walk's event stream is the
single total order under ``prop_account_event_order_v1`` with exact source
links; the evaluation phase at one fixed contract with fees disabled
reproduces the untouched :class:`EvaluationWalk` exactly.
"""

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
from alpha_lab.propsim.engine import EvaluationWalk
from alpha_lab.propsim.firm_contracts import (
    SYNTHETIC_FIXTURE_FIRM,
    PhaseRules,
)
from alpha_lab.propsim.models import Ruleset, TradePath
from alpha_lab.propsim.risk import FIXED_ONE_NQ_RISK_POLICY
from alpha_lab.propsim.withdrawal import REQUEST_MAX_AT_ELIGIBILITY

_POLICY_SET = AccountPolicySetPayload(
    firm_contract_id="a" * 64,
    risk_policy_id=canonical_contract_sha256(FIXED_ONE_NQ_RISK_POLICY),
    withdrawal_policy_id=canonical_contract_sha256(REQUEST_MAX_AT_ELIGIBILITY),
    replacement_policy="none",
    max_replacements=0,
    clock_policy_id="historical_calendar_clock_v1",
)


def _trade(
    day: str,
    points: float,
    *,
    slot: int = 0,
    mfe: float | None = None,
    mae: float | None = None,
    risk: float = 10.0,
) -> AccountTrade:
    return AccountTrade(
        day=date.fromisoformat(day),
        entry_ts_utc=f"{day}T1{4 + slot}:00:00+00:00",
        resolution_ts_utc=f"{day}T1{4 + slot}:05:00+00:00",
        points=points,
        risk_points=risk,
        mfe_pts=mfe,
        mae_pts=mae,
        trade_id=f"trade-{day}-{slot}",
        decision_id=f"dec-{day}-{slot}",
        candidate_id=f"cand-{day}-{slot}",
        setup_id=f"setup-{day}-{slot}",
    )


def _walk(firm=SYNTHETIC_FIXTURE_FIRM, *, policy_set=_POLICY_SET, **kwargs) -> AccountWalk:
    return AccountWalk(
        firm=firm,
        firm_contract_id=policy_set.firm_contract_id,
        risk_policy=kwargs.pop("risk_policy", FIXED_ONE_NQ_RISK_POLICY),
        withdrawal_policy=kwargs.pop("withdrawal_policy", REQUEST_MAX_AT_ELIGIBILITY),
        policy_set=policy_set,
        path_instance_id="path-00000-test",
        **kwargs,
    )


def _days(*blocks: tuple[str, list[AccountTrade]]):
    return [(date.fromisoformat(day), trades) for day, trades in blocks]


# ── EvaluationWalk ≡ AccountWalk one-contract parity (§16.4) ────────────────


@pytest.mark.parametrize("breach_mode", ["realized_only", "unrealized_adverse_first"])
@pytest.mark.parametrize(
    "trail_style", ["eod_floor_realtime_breach", "intraday_peak_trail", "static_floor"]
)
def test_account_walk_reproduces_evaluation_walk_exactly(
    trail_style: str, breach_mode: str
) -> None:
    ruleset = Ruleset(
        starting_balance=50_000.0,
        profit_target=3_000.0,
        trail_amount=2_000.0,
        trail_style=trail_style,
        trail_locks_at_start=True,
        dll_amount=1_000.0,
        dll_hard=False,
        consistency_pct=50.0,
        min_days=2,
        max_eval_days=6,
        point_value=20.0,
    )
    phase = PhaseRules(
        starting_balance=ruleset.starting_balance,
        profit_target=ruleset.profit_target,
        trail_amount=ruleset.trail_amount,
        trail_style=trail_style,  # type: ignore[arg-type]
        trail_locks_at_start=True,
        dll_amount=ruleset.dll_amount,
        dll_hard=False,
        consistency_pct=ruleset.consistency_pct,
        min_days=DurationRule(count=2, basis=DayCountBasis.TRADING_DAY),
        max_eval_days=DurationRule(count=6, basis=DayCountBasis.TRADING_DAY),
        unrealized_equity_counts_for_breach=breach_mode != "realized_only",
        breach_observation_policy="event_updates_only",
        max_contracts=1,
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={"evaluation": phase, "funded": None, "payout": None}
    )
    scripts = {
        "2026-01-13": [(90.0, 20.0, 30.0), (-40.0, 5.0, 45.0)],
        "2026-01-14": [(60.0, 70.0, 10.0)],
        "2026-01-15": [(-55.0, 10.0, 60.0), (25.0, 30.0, 8.0)],
        "2026-01-16": [(80.0, 85.0, 12.0)],
        "2026-01-19": [(-30.0, 6.0, 95.0)],
        "2026-01-20": [(45.0, 50.0, 20.0)],
        "2026-01-21": [(10.0, 12.0, 4.0)],
    }
    engine_walk = EvaluationWalk(ruleset, column="optimistic", breach_mode=breach_mode)
    account_walk = _walk(firm, breach_mode=breach_mode, enable_fees=False)
    for day, spec in scripts.items():
        engine_trades = [
            TradePath(
                day=date.fromisoformat(day),
                entry_ts=__import__("datetime").datetime.fromisoformat(
                    f"{day}T1{4 + slot}:00:00+00:00"
                ),
                points_optimistic=points,
                points_conservative=points,
                mfe_pts=mfe,
                mae_pts=mae,
                resolution=None,
            )
            for slot, (points, mfe, mae) in enumerate(spec)
        ]
        account_trades = [
            _trade(day, points, slot=slot, mfe=mfe, mae=mae)
            for slot, (points, mfe, mae) in enumerate(spec)
        ]
        engine_verdict = engine_walk.play_day(engine_trades)
        account_verdict = account_walk.play_day(date.fromisoformat(day), account_trades)
        if engine_verdict is not None:
            break
        assert account_verdict in (None, "funded", "evaluation_passed")
        if account_verdict is not None:
            break
    engine_result = engine_walk.result()
    account_result = account_walk.result()
    assert account_walk.state.balance == pytest.approx(engine_result.final_balance)
    verdict_map = {
        "pass": ("evaluation_passed", "retired"),
        "bust": ("breached_out",),
        "expired": ("expired",),
        "incomplete": ("evaluation_alive",),
    }
    assert account_result.verdict in verdict_map[engine_result.verdict]
    if engine_result.verdict == "bust":
        assert account_walk.state.breach_reason == engine_result.bust_reason


# ── trailing styles + static drawdown + DLL hard/soft ───────────────────────


def test_soft_dll_halts_the_day_and_hard_dll_busts() -> None:
    firm = SYNTHETIC_FIXTURE_FIRM  # soft DLL $1,000
    walk = _walk(firm, breach_mode="realized_only", enable_fees=False)
    verdict = walk.play_day(
        date(2026, 1, 13),
        [
            _trade("2026-01-13", -55.0, slot=0),  # -$1,100 < -DLL → halt
            _trade("2026-01-13", 100.0, slot=1),  # skipped (halted)
        ],
    )
    assert verdict is None
    halts = [event for event in walk.result().events if event.event_type == "daily_halt"]
    assert len(halts) == 1
    assert halts[0].payload.halt_reason == "daily_loss_limit_soft"
    assert walk.state.balance == pytest.approx(50_000.0 - 1_100.0)

    hard_phase = firm.evaluation.model_copy(update={"dll_hard": True})
    hard_firm = firm.model_copy(update={"evaluation": hard_phase})
    walk = _walk(hard_firm, breach_mode="realized_only", enable_fees=False)
    verdict = walk.play_day(
        date(2026, 1, 13), [_trade("2026-01-13", -55.0, slot=0)]
    )
    assert verdict == "breached_out"
    breaches = [e for e in walk.result().events if e.event_type == "breach"]
    assert breaches[0].payload.breach_reason == "daily_loss_limit"
    # breach carries its exact source-trade link (§16.4 "breach with linked trade")
    assert breaches[0].source_trade_id == "trade-2026-01-13-0"
    assert breaches[0].source_setup_id == "setup-2026-01-13-0"


def test_static_floor_never_ratchets_and_eod_style_does() -> None:
    static_phase = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"trail_style": "static_floor", "dll_amount": None}
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(update={"evaluation": static_phase})
    walk = _walk(firm, breach_mode="realized_only", enable_fees=False)
    walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 100.0)])
    assert walk.state.drawdown_floor == pytest.approx(48_000.0)  # start − trail, fixed
    ratchets = [
        e for e in walk.result().events if e.event_type == "threshold_ratchet"
    ]
    assert ratchets == []

    eod_firm = SYNTHETIC_FIXTURE_FIRM
    walk = _walk(eod_firm, breach_mode="realized_only", enable_fees=False)
    walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 100.0)])  # +$2,000
    ratchets = [
        e for e in walk.result().events if e.event_type == "threshold_ratchet"
    ]
    assert len(ratchets) == 1
    assert ratchets[0].payload.new_floor == pytest.approx(50_000.0)  # locked at start


# ── funded lifecycle: payouts, fees, post-payout rules, replacement ─────────


def _pass_then(walk: AccountWalk, funded_days: list[tuple[str, list[AccountTrade]]]):
    """Drive the synthetic firm through its evaluation, then the funded days."""

    passing = [
        ("2026-01-13", [_trade("2026-01-13", 80.0)]),   # +1600
        ("2026-01-14", [_trade("2026-01-14", 80.0, slot=1)]),  # +1600 → target hit d2
    ]
    for day, trades in passing:
        verdict = walk.play_day(date.fromisoformat(day), trades)
        if verdict is not None and verdict != "funded":
            raise AssertionError(f"evaluation did not pass cleanly: {verdict}")
    assert walk.state.phase.value == "funded"
    for day, trades in funded_days:
        verdict = walk.play_day(date.fromisoformat(day), trades)
        if verdict is not None:
            return verdict
    return None


def test_payout_eligibility_cap_split_and_fee_kinds() -> None:
    walk = _walk(SYNTHETIC_FIXTURE_FIRM, breach_mode="realized_only")
    _pass_then(
        walk,
        [
            ("2026-01-15", [_trade("2026-01-15", 60.0)]),  # +1200 (winning)
            ("2026-01-16", [_trade("2026-01-16", 60.0)]),  # +1200 (winning)
            ("2026-01-19", [_trade("2026-01-19", 60.0)]),  # waiting satisfied → payout
        ],
    )
    result = walk.result()
    payouts = [e for e in result.events if e.event_type == "payout"]
    assert len(payouts) == 1
    payout = payouts[0].payload
    # available profit above the funded base is capped at $2,000/period
    assert payout.approved_amount == pytest.approx(2_000.0)
    assert payout.trader_amount == pytest.approx(1_800.0)  # 90% split
    assert payout.firm_amount == pytest.approx(200.0)
    # fee kinds: evaluation at open + activation at funding (recurring later)
    fee_kinds = [e.payload.fee_kind for e in result.events if e.event_type == "fee"]
    assert fee_kinds[:2] == ["evaluation", "activation"]
    assert walk.result().total_trader_payouts == pytest.approx(1_800.0)


def test_all_three_post_payout_threshold_rules() -> None:
    # the funded phase must NOT lock its trail at start, or "unchanged" and
    # "locked_at_starting_balance" coincide by construction
    unlocked_funded = SYNTHETIC_FIXTURE_FIRM.funded.model_copy(
        update={"trail_locks_at_start": False}
    )
    outcomes = {}
    for rule in (
        "threshold_unchanged",
        "threshold_resets_to_balance_minus_trail",
        "locked_at_starting_balance",
    ):
        payout = SYNTHETIC_FIXTURE_FIRM.payout.model_copy(
            update={"post_payout_buffer_rule": rule}
        )
        firm = SYNTHETIC_FIXTURE_FIRM.model_copy(
            update={"payout": payout, "funded": unlocked_funded}
        )
        walk = _walk(firm, breach_mode="realized_only", enable_fees=False)
        _pass_then(
            walk,
            [
                ("2026-01-15", [_trade("2026-01-15", 60.0)]),
                ("2026-01-16", [_trade("2026-01-16", 60.0)]),
                ("2026-01-19", [_trade("2026-01-19", 60.0)]),
            ],
        )
        outcomes[rule] = walk.state.drawdown_floor
    # the three rules produce three distinct floors
    assert len({round(value, 2) for value in outcomes.values()}) == 3
    assert outcomes["locked_at_starting_balance"] == pytest.approx(50_000.0)


def test_min_days_winning_days_and_consistency_denominator() -> None:
    # min_days blocks a day-1 target hit; consistency uses TOTAL profit
    phase = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"consistency_pct": 50.0}
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={"evaluation": phase, "funded": None, "payout": None}
    )
    walk = _walk(firm, breach_mode="realized_only", enable_fees=False)
    assert walk.play_day(
        date(2026, 1, 13), [_trade("2026-01-13", 160.0)]
    ) is None  # target hit but min_days=2 blocks
    # day 2 dilutes: best day 3200 of total 3300 > 50% → consistency blocks
    assert walk.play_day(date(2026, 1, 14), [_trade("2026-01-14", 5.0)]) is None
    assert walk.state.consistency_ok is False
    # still blocked: best 3200 > 50% × 6300 (the denominator is TOTAL profit)
    assert walk.play_day(
        date(2026, 1, 15), [_trade("2026-01-15", 150.0)]
    ) is None
    verdict = walk.play_day(date(2026, 1, 16), [_trade("2026-01-16", 30.0)])
    assert verdict == "evaluation_passed"

    # winning-day minimum gates the payout side
    payout = SYNTHETIC_FIXTURE_FIRM.payout.model_copy(
        update={
            "min_winning_days": DurationRule(count=3, basis=DayCountBasis.WINNING_DAY),
            "waiting_period": DurationRule(count=1, basis=DayCountBasis.TRADING_DAY),
        }
    )
    gated = SYNTHETIC_FIXTURE_FIRM.model_copy(update={"payout": payout})
    walk = _walk(gated, breach_mode="realized_only", enable_fees=False)
    _pass_then(
        walk,
        [
            ("2026-01-15", [_trade("2026-01-15", 60.0)]),
            ("2026-01-16", [_trade("2026-01-16", -5.0)]),  # not winning
        ],
    )
    assert not [e for e in walk.result().events if e.event_type == "payout"]


def test_replacement_carries_ordinal_and_reset_fee() -> None:
    policy_set = _POLICY_SET.model_copy(
        update={"replacement_policy": "auto_replace_up_to_n", "max_replacements": 1}
    )
    hard_phase = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"dll_hard": True}
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(update={"evaluation": hard_phase})
    walk = _walk(firm, policy_set=policy_set, breach_mode="realized_only")
    verdict = walk.play_day(
        date(2026, 1, 13), [_trade("2026-01-13", -55.0)]
    )
    assert verdict == "replaced"
    result = walk.result()
    replacements = [e for e in result.events if e.event_type == "replacement"]
    assert len(replacements) == 1
    assert replacements[0].payload.prior_account_id.startswith("acct-000-")
    assert replacements[0].payload.replacement_account_id.startswith("acct-001-")
    assert replacements[0].payload.reset_fee == pytest.approx(80.0)
    fee_kinds = [e.payload.fee_kind for e in result.events if e.event_type == "fee"]
    assert fee_kinds.count("reset") == 1
    assert fee_kinds.count("evaluation") == 2  # both accounts opened
    # the replacement account is live and the walk continues
    assert walk.state.phase.value == "evaluation"
    assert walk.result().accounts_used == 2
    # a second breach exhausts max_replacements → terminal
    verdict = walk.play_day(date(2026, 1, 14), [_trade("2026-01-14", -55.0)])
    assert verdict == "breached_out"


def test_event_stream_is_one_strict_total_order_with_exact_links() -> None:
    """P0-13: DLL touch + ratchet + payout eligibility + fees in ONE ordered
    stream; ordinals strictly increase; same-day ties keep the deterministic
    emission precedence; every source link resolves."""

    walk = _walk(SYNTHETIC_FIXTURE_FIRM, breach_mode="realized_only")
    _pass_then(
        walk,
        [
            ("2026-01-15", [_trade("2026-01-15", -55.0), _trade("2026-01-15", 60.0, slot=1)]),
            ("2026-01-16", [_trade("2026-01-16", 60.0)]),
            ("2026-01-19", [_trade("2026-01-19", 60.0)]),
            ("2026-01-20", [_trade("2026-01-20", 60.0)]),
        ],
    )
    events = walk.result().events
    ordinals = [event.event_ordinal for event in events]
    assert ordinals == sorted(ordinals) and len(set(ordinals)) == len(ordinals)
    assert {event.event_order_policy_id for event in events} == {
        "prop_account_event_order_v1"
    }
    kinds = {event.event_type for event in events}
    assert {
        "fee",
        "equity_update",
        "daily_halt",
        "phase_transition",
        "threshold_ratchet",
        "payout",
    } <= kinds
    # deterministic rerun → identical event ids in identical order
    walk_2 = _walk(SYNTHETIC_FIXTURE_FIRM, breach_mode="realized_only")
    _pass_then(
        walk_2,
        [
            ("2026-01-15", [_trade("2026-01-15", -55.0), _trade("2026-01-15", 60.0, slot=1)]),
            ("2026-01-16", [_trade("2026-01-16", 60.0)]),
            ("2026-01-19", [_trade("2026-01-19", 60.0)]),
            ("2026-01-20", [_trade("2026-01-20", 60.0)]),
        ],
    )
    assert [event.event_id for event in events] == [
        event.event_id for event in walk_2.result().events
    ]
    # every trade-linked event carries the full source tuple
    linked = [event for event in events if event.source_trade_id is not None]
    assert linked and all(
        event.source_decision_id and event.source_candidate_id and event.source_setup_id
        for event in linked
    )


def test_cited_path_events_must_exist_in_the_bundle() -> None:
    """CS §5.3: a path event cited by an account event must exist by exact id."""

    trade = AccountTrade(
        day=date(2026, 1, 13),
        entry_ts_utc="2026-01-13T14:00:00+00:00",
        resolution_ts_utc="2026-01-13T14:05:00+00:00",
        points=10.0,
        risk_points=10.0,
        mfe_pts=None,
        mae_pts=None,
        trade_id="trade-x",
        exit_path_event_id="f" * 64,  # not in the bundle
    )
    walk = _walk(
        SYNTHETIC_FIXTURE_FIRM,
        breach_mode="realized_only",
        enable_fees=False,
        bundle_event_ids=frozenset({"a" * 64}),
    )
    with pytest.raises(ValueError, match="unknown path event"):
        walk.play_day(date(2026, 1, 13), [trade])


# ── R3 review-driven rule fixtures (recurring fees, calendars, refusals) ────


def _quick_funded_firm(**payout_overrides):
    """A firm whose evaluation passes on one +$400 day, for funded fixtures."""

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


def test_recurring_fee_charges_on_its_trading_day_period() -> None:
    """Rule row: recurring fees are actually charged, on the period boundary."""

    fees = SYNTHETIC_FIXTURE_FIRM.fees.model_copy(
        update={"recurring_period": DurationRule(count=2, basis=DayCountBasis.TRADING_DAY)}
    )
    firm = _quick_funded_firm().model_copy(update={"fees": fees})
    walk = _walk(firm, enable_fees=True)
    days = ("2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-20")
    for day in days:
        walk.play_day(date.fromisoformat(day), [_trade(day, 20.0)])
    fee_kinds = [
        event.payload.fee_kind
        for event in walk.result().events
        if event.event_type == "fee"
    ]
    # eval fee at open, activation at the funded transition, then a $50
    # recurring fee every 2 funded trading days (4 funded days -> 2 fees)
    assert fee_kinds.count("evaluation") == 1
    assert fee_kinds.count("activation") == 1
    assert fee_kinds.count("recurring") == 2


def test_calendar_month_recurring_fee_works_under_the_historical_clock() -> None:
    """P0-14: a calendar basis is ROUTED through the typed clock when the
    active clock represents it -- never silently dropped."""

    fees = SYNTHETIC_FIXTURE_FIRM.fees.model_copy(
        update={"recurring_period": DurationRule(count=1, basis=DayCountBasis.CALENDAR_MONTH)}
    )
    firm = _quick_funded_firm().model_copy(update={"fees": fees})
    walk = _walk(firm, enable_fees=True)
    # pass in January, stay funded into February and March
    for day in ("2026-01-13", "2026-01-14", "2026-02-16", "2026-03-16"):
        walk.play_day(date.fromisoformat(day), [_trade(day, 20.0)])
    recurring = [
        event
        for event in walk.result().events
        if event.event_type == "fee" and event.payload.fee_kind == "recurring"
    ]
    assert len(recurring) == 2  # one per elapsed calendar month since funding


def test_calendar_basis_rules_fail_closed_under_the_bootstrap_clock() -> None:
    """P0-14 at the WALK level: construction refuses, never guesses."""

    from alpha_lab.propsim.calendar import (
        BOOTSTRAP_CLOCK_POLICY,
        UnsupportedCalendarRuleError,
    )

    fees = SYNTHETIC_FIXTURE_FIRM.fees.model_copy(
        update={"recurring_period": DurationRule(count=1, basis=DayCountBasis.CALENDAR_MONTH)}
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(update={"fees": fees})
    with pytest.raises(UnsupportedCalendarRuleError, match="fees.recurring_period"):
        _walk(firm, clock_policy=BOOTSTRAP_CLOCK_POLICY)
    # a calendar-month withdrawal cadence is refused the same way
    from alpha_lab.propsim.withdrawal import WithdrawalPolicyPayload

    monthly = WithdrawalPolicyPayload(
        policy_key="monthly_cadence_v1",
        behavior="fixed_cadence",
        minimum_buffer_retained=None,
        cadence=DurationRule(count=1, basis=DayCountBasis.CALENDAR_MONTH),
        partial_amount=None,
        post_payout_derisk_ref=None,
    )
    with pytest.raises(UnsupportedCalendarRuleError, match="withdrawal.cadence"):
        _walk(
            SYNTHETIC_FIXTURE_FIRM,
            withdrawal_policy=monthly,
            clock_policy=BOOTSTRAP_CLOCK_POLICY,
        )


def test_account_expiration_expires_the_account() -> None:
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={"account_expiration": DurationRule(count=2, basis=DayCountBasis.TRADING_DAY)}
    )
    walk = _walk(firm, enable_fees=False)
    assert walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 5.0)]) is None
    assert walk.play_day(date(2026, 1, 14), [_trade("2026-01-14", 5.0)]) is None
    verdict = walk.play_day(date(2026, 1, 15), [_trade("2026-01-15", 5.0)])
    assert verdict == "expired"
    transitions = [
        event
        for event in walk.result().events
        if event.event_type == "phase_transition"
    ]
    assert transitions[-1].payload.reason == "account_expiration"


def test_funded_consistency_rule_gates_payout_eligibility() -> None:
    funded = SYNTHETIC_FIXTURE_FIRM.funded.model_copy(
        update={"consistency_pct": 50.0}
    )
    firm = _quick_funded_firm().model_copy(update={"funded": funded})
    walk = _walk(firm, enable_fees=False)
    walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 20.0)])  # -> funded
    # one huge funded day: best day is ~100% of funded profit -> not eligible
    walk.play_day(date(2026, 1, 14), [_trade("2026-01-14", 100.0)])
    walk.play_day(date(2026, 1, 15), [_trade("2026-01-15", 5.0)])
    assert walk.state.consistency_ok is False
    assert walk.state.payout_eligible is False
    assert not [
        e for e in walk.result().events if e.event_type == "payout"
    ]


def test_payout_processing_delay_is_refused_not_silently_instant() -> None:
    from alpha_lab.propsim.account import UnsupportedFirmRuleError

    firm = _quick_funded_firm(
        payout_processing=DurationRule(count=3, basis=DayCountBasis.TRADING_DAY)
    )
    with pytest.raises(UnsupportedFirmRuleError, match="payout_processing"):
        _walk(firm)


def test_firm_requiring_unrealized_observation_refuses_realized_only() -> None:
    from alpha_lab.propsim.account import UnsupportedFirmRuleError

    unrealized_phase = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"unrealized_equity_counts_for_breach": True}
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(update={"evaluation": unrealized_phase})
    with pytest.raises(UnsupportedFirmRuleError, match="under-observation"):
        _walk(firm, breach_mode="realized_only")
    # the conservative unrealized scenario is permitted (labeled by its mode)
    walk = _walk(firm, breach_mode="unrealized_adverse_first")
    assert walk.state.phase.value == "evaluation"


def test_intraday_trail_raises_emit_ratchet_events_with_trade_links() -> None:
    intraday = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"trail_style": "intraday_peak_trail", "dll_amount": None}
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(update={"evaluation": intraday})
    walk = _walk(firm, breach_mode="unrealized_adverse_first", enable_fees=False)
    walk.play_day(
        date(2026, 1, 13),
        [_trade("2026-01-13", 100.0, mfe=120.0, mae=5.0)],
    )
    ratchets = [
        event
        for event in walk.result().events
        if event.event_type == "threshold_ratchet"
    ]
    assert ratchets, "an intraday floor raise must emit a threshold_ratchet"
    # the mid-day raise carries its source trade and a mid-day timestamp
    assert ratchets[0].source_trade_id == "trade-2026-01-13-0"
    assert "T23:59:59" not in ratchets[0].event_ts_utc
    floors = [event.payload.new_floor for event in ratchets]
    assert floors == sorted(floors)


def test_initial_account_open_timestamp_is_never_a_sentinel() -> None:
    walk = _walk(SYNTHETIC_FIXTURE_FIRM, enable_fees=True)
    # the evaluation fee is DEFERRED to the first played day (no fake ts)
    assert walk.result().events == ()
    walk.play_day(date(2026, 1, 13), [_trade("2026-01-13", 5.0)])
    first = walk.result().events[0]
    assert first.event_type == "fee" and first.payload.fee_kind == "evaluation"
    assert first.event_ts_utc == "2026-01-13T00:00:00+00:00"
    for event in walk.result().events:
        date.fromisoformat(event.event_ts_utc[:10])  # every ts parses
