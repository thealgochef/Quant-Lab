"""Single-account configuration comparison: account sequence, payouts, execution.

Deterministic synthetic days (``pair_builders``) drive the real pair engine and
ledger with a scripted strategy that obeys Strategy-Core's one-slot rules.
TakeProfitTrader numbers: one mini = $5 per tick, $5.14 per fill.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from alpha_lab.propsim.funded.comparison_result import (
    build_comparison_result,
    validate_comparison,
)
from alpha_lab.propsim.funded.pair_engine import PairRun, run_days
from alpha_lab.propsim.funded.position_walk import (
    MinuteObservations,
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

TPT = FIRM_PROFILES["takeprofittrader"]
MFF = FIRM_PROFILES["myfundedfutures"]
DAYS = weekdays(date(2026, 3, 2), 12)  # Monday March 2, 2026 onward (no Fed holidays)


def winner(day: date, *, gain: int = 600) -> SyntheticDay:
    """Enter long at BASE; the next minute reaches the target (+gain ticks)."""
    minutes = flat(6)
    minutes[1] = [BASE + gain // 2, BASE + gain]
    return SyntheticDay(day, minutes, {0: ("long", BASE - 100, BASE + gain)})


def loser(day: date, *, drop: int = 450, stop: int = 900) -> SyntheticDay:
    """Enter long at BASE; prices fall ``drop`` ticks, far from the stop."""
    minutes = flat(6)
    minutes[1] = [BASE - drop // 2, BASE - drop]
    minutes[2] = [BASE]
    return SyntheticDay(day, minutes, {0: ("long", BASE - stop, BASE + 2000)})


def quiet(day: date, *, signal: bool = False) -> SyntheticDay:
    return SyntheticDay(day, flat(6), {0: ("long", BASE - 100, BASE + 100)} if signal else {})


# ── account sequence ───────────────────────────────────────────────────────


def test_profitable_payout_then_failure_keeps_the_receipt_and_charges_one_replacement():
    days = [winner(DAYS[0]), quiet(DAYS[1], signal=True), quiet(DAYS[2]),
            loser(DAYS[3], drop=420, stop=600), winner(DAYS[4], gain=100)]
    run, ledger, _ = run_pair(days)
    first, second = ledger.accounts
    # 600 ticks x $5 = $3,000 less $10.28 costs; surplus above $2,100 = $889.72
    request = next(r for r in ledger.payout_events if r["event"] == "requested")
    assert request["gross_cents"] == 88_972
    received = next(r for r in ledger.payout_events if r["event"] == "received")
    assert received["trader_cents"] == 71_178  # 80% of $889.72, rounded half up
    assert first.status == "failed" and first.payouts_received == 1
    assert ledger.receipts == 71_178  # kept after the failure
    assert ledger.costs == 2 * TPT.acquisition_cost_cents
    assert second.created_ns == first.failed_ns and second.replaces == first.account_id
    # the replacement's first trade comes from a later signal, never the failure event
    second_trades = [t for t in ledger.trades if t["account_id"] == second.account_id]
    assert second_trades and all(t["entry_ns"] > second.created_ns for t in second_trades)
    # the loss-limit breach (floor locked at $0) came before the strategy stop
    failed_trade = next(t for t in ledger.trades if t["account_failed"])
    assert failed_trade["exit_kind"] == "account_failure"
    assert failed_trade["exit_ticks"] == BASE - 420
    assert run.driver.forced_flat == 1
    assert failed_trade["strategy_recorded_exit_kind"] == "ended_by_account_liquidation"


def test_more_than_five_replacements_without_any_credit_stop():
    days = [loser(d) for d in DAYS[:7]] + [quiet(DAYS[7])]
    _run, ledger, _ = run_pair(days)
    assert len(ledger.accounts) == 8  # seven failures, each replaced at once
    assert ledger.costs == 8 * 10_200
    assert [a.status for a in ledger.accounts] == ["failed"] * 7 + ["ready"]
    purchases = [r for r in ledger.cash_ledger if r["kind"] == "account_purchase"]
    assert len(purchases) == 8 and {r["amount_cents"] for r in purchases} == {10_200}
    assert ledger.net_cash == -81_600
    assert ledger.max_shortfall == 81_600


def test_myfundedfutures_replacements_cost_125_each():
    days = [loser(d) for d in DAYS[:6]] + [quiet(DAYS[6])]
    _run, ledger, _ = run_pair(days, firm_key="myfundedfutures")
    assert len(ledger.accounts) == 7 and ledger.costs == 7 * 12_500


def test_a_large_receipt_never_expands_to_a_second_account():
    days = [winner(DAYS[0], gain=1400)] + [quiet(d) for d in DAYS[1:5]]
    _run, ledger, _ = run_pair(days)
    assert ledger.receipts > 0
    assert len(ledger.accounts) == 1
    assert [r["kind"] for r in ledger.cash_ledger] == ["account_purchase", "payout_received"]


def test_processing_account_is_not_replaced_and_refuses_entries():
    days = [winner(DAYS[0]), quiet(DAYS[1], signal=True), quiet(DAYS[2], signal=True),
            quiet(DAYS[3])]
    run, ledger, _ = run_pair(days)
    account = ledger.accounts[0]
    assert len(ledger.accounts) == 1
    # Tuesday's signal arrives while processing (paid Wednesday 4:00 PM Chicago);
    # Wednesday's morning signal is also before the payment time
    assert account.blocked_entries == {"account_payout_processing": 2}
    assert run.driver.refusals == ["account_payout_processing"] * 2
    assert run.driver.discarded_setups == 2
    assert account.trades == 1
    received = next(r for r in ledger.payout_events if r["event"] == "received")
    assert received["received_ns"] == received["due_ns"]


def test_secured_account_stops_trading_for_the_rest_of_the_day():
    day = winner(DAYS[0])
    day.signals[3] = ("long", BASE - 100, BASE + 100)  # later the same day
    run, ledger, _ = run_pair([day, quiet(DAYS[1])])
    assert ledger.accounts[0].blocked_entries == {"account_payout_protection": 1}
    assert ledger.accounts[0].trades == 1


def test_zero_payout_configuration_reports_only_its_cost():
    days = [winner(d, gain=40) for d in DAYS[:3]]
    _run, ledger, _ = run_pair(days)
    assert ledger.receipts == 0 and ledger.costs == 10_200 and ledger.net_cash == -10_200
    assert not [r for r in ledger.payout_events if r["event"] == "requested"]


def test_payout_pending_at_the_cutoff_is_not_received():
    _run, ledger, _ = run_pair([winner(DAYS[0])])
    assert [r["event"] for r in ledger.payout_events] == ["eligibility_secured", "requested"]
    assert ledger.receipts == 0
    assert ledger.accounts[0].status == "processing"


def test_myfundedfutures_updates_its_floor_at_the_close_before_the_request():
    _run, ledger, _ = run_pair([winner(DAYS[0]), quiet(DAYS[1])], firm_key="myfundedfutures")
    account = ledger.accounts[0]
    floor_move = next(r for r in ledger.boundary_evidence if r["check"] == "floor_moved")
    request = next(r for r in ledger.payout_events if r["event"] == "requested")
    assert floor_move["new_floor_cents"] == 10_000  # locked at +$100
    assert floor_move["ts_ns"] == request["ts_ns"] and floor_move["seq"] < request["seq"]
    assert account.balance == 210_000 and account.floor == 10_000


def test_two_pairs_share_the_market_but_never_money_or_state():
    days = [winner(DAYS[0]), loser(DAYS[1], drop=450), quiet(DAYS[2], signal=True)]
    inputs, schedule, start_ns, cutoff_ns = build(days)
    runs = [PairRun(f"CFG|{k}", ScriptedDriver(),
                    ledger_for(k, schedule, start_ns, cutoff_ns, pair_id=f"CFG|{k}"))
            for k in ("takeprofittrader", "myfundedfutures")]
    prints = ScriptedPrints()
    run_days(runs, inputs, lambda _d: (lambda: prints))
    for run in runs:
        run.ledger.finish()
        for table in (run.ledger.cash_ledger, run.ledger.trades, run.ledger.payout_events,
                      run.ledger.account_events):
            assert {row["pair_id"] for row in table} == {run.pair_id}
    assert runs[0].driver is not runs[1].driver
    assert runs[0].ledger.costs == 10_200 and runs[1].ledger.costs == 12_500


# ── execution on ordered prints ───────────────────────────────────────────


def _obs(prices, *, fidelity="ordered_trade_prints", continuous=None):
    n = len(prices)
    return MinuteObservations(
        open_ns=0, close_ns=60, close_ticks=prices[-1],
        ts_ns=np.arange(n, dtype=np.int64) + 1, price_ticks=np.array(prices, dtype=np.int64),
        continuous=np.zeros(n, dtype=bool) if continuous is None else np.array(continuous),
        fidelity=fidelity)


def _open(profile=TPT, *, balance=0, floor=-200_000, peak=0, stop=BASE - 100,
          target=BASE + 100):
    pos, failure = open_position(
        profile=profile, trade_ref="t", direction="long", entry_ns=0, entry_ticks=BASE,
        stop_ticks=stop, target_ticks=target, quantity=1, tick_value_cents=500,
        cost_per_side_cents=514, balance_cents=balance, floor_cents=floor, peak_cents=peak)
    return pos, failure


def test_stop_gap_fills_at_the_first_print_through_the_stop_not_the_stop_price():
    pos, _ = _open()
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE - 50, BASE - 106]),
                        deadline_minute=False)
    assert exit_.kind == "stop" and exit_.fill_ticks == BASE - 106
    assert exit_.gross_pnl_cents == -106 * 500
    assert exit_.balance_after_cents == -514 - 53_000 - 514


def test_target_fills_at_the_target_even_when_a_print_is_better():
    pos, _ = _open()
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 40, BASE + 130]),
                        deadline_minute=False)
    assert exit_.kind == "target" and exit_.fill_ticks == BASE + 100


def test_loss_limit_breach_on_a_print_comes_before_the_stop():
    pos, _ = _open(balance=10_000, floor=0, peak=210_000, stop=BASE - 600)  # floor locked
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE - 10, BASE - 25, BASE - 700]),
                        deadline_minute=False)
    # equity 9,486 + (-25 x 500) = -3,014 <= 0 at the second print
    assert exit_.kind == "account_failure" and exit_.fill_ticks == BASE - 25
    assert exit_.failure_stage == "open_position"


def test_exit_cost_can_lose_the_account_after_the_fill():
    # TakeProfitTrader floor locked at 0; after the $5.14 entry cost the balance is $503.00,
    # the stop fill leaves $3.00 above the floor and the $5.14 exit cost takes it below
    pos, _ = _open(balance=50_814, floor=0, peak=300_000, stop=BASE - 100)
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE - 100]), deadline_minute=False)
    assert exit_.kind == "stop" and exit_.account_failed
    assert exit_.failure_stage == "after_exit_cost"


def test_entry_cost_alone_can_lose_the_account():
    # locked +$100 floor, strictly-below rule: $105.14 - $5.14 = $100.00 survives
    _pos, failure = _open(profile=MFF, balance=10_514, floor=10_000)
    assert failure is None
    _pos, failure = _open(profile=MFF, balance=10_513, floor=10_000)
    assert failure is not None and failure.failure_stage == "at_entry_cost"


def test_myfundedfutures_locked_floor_is_strictly_below_and_equality_survives():
    # equity exactly +$100 at the locked floor does not fail; one cent lower does
    pos, _ = _open(profile=MFF, balance=60_514, floor=10_000, stop=BASE - 500)
    assert walk_minute(profile=MFF, pos=pos, obs=_obs([BASE - 100]),
                       deadline_minute=False) is None
    pos, _ = _open(profile=MFF, balance=60_513, floor=10_000, stop=BASE - 500)
    exit_ = walk_minute(profile=MFF, pos=pos, obs=_obs([BASE - 100]), deadline_minute=False)
    assert exit_.kind == "account_failure"


def test_takeprofittrader_floor_rises_with_open_equity_then_fails_on_the_pullback():
    pos, _ = _open(balance=0, floor=-200_000, stop=BASE - 900, target=BASE + 900)
    assert walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 300]), deadline_minute=False) is None
    assert pos.floor_cents == -514 + 150_000 - 200_000  # peak minus $2,000
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE - 100]), deadline_minute=False)
    assert exit_.kind == "account_failure" and exit_.fill_ticks == BASE - 100


def test_deadline_minute_closes_at_its_last_trade():
    pos, _ = _open(stop=BASE - 500, target=BASE + 500)
    exit_ = walk_minute(profile=TPT, pos=pos, obs=_obs([BASE + 5, BASE + 12]),
                        deadline_minute=True)
    assert exit_.kind == "scheduled_close" and exit_.fill_ticks == BASE + 12


def test_minute_approximation_is_labeled_and_fills_the_stop_on_a_continuous_leg():
    pos, _ = _open(stop=BASE - 20)
    obs = minute_approximation(0, 60, BASE, BASE + 10, BASE - 30, BASE - 5, sign=1)
    exit_ = walk_minute(profile=TPT, pos=pos, obs=obs, deadline_minute=False)
    assert exit_.kind == "stop" and exit_.fill_ticks == BASE - 20 and exit_.approximate
    pos, _ = _open(stop=BASE - 20)
    gapped = minute_approximation(0, 60, BASE - 40, BASE - 10, BASE - 45, BASE - 20, sign=1)
    exit_ = walk_minute(profile=TPT, pos=pos, obs=gapped, deadline_minute=False)
    assert exit_.fill_ticks == BASE - 40  # the candle opened through the stop


# ── checkpoint, validation ────────────────────────────────────────────────


def test_resumed_run_equals_the_straight_run():
    days = [winner(DAYS[0]), quiet(DAYS[1], signal=True), loser(DAYS[2], drop=420, stop=600),
            loser(DAYS[3]), winner(DAYS[4]), quiet(DAYS[5]), quiet(DAYS[6])]
    inputs, schedule, start_ns, cutoff_ns = build(days)
    prints = ScriptedPrints()

    def fresh():
        return PairRun("CFG|takeprofittrader", ScriptedDriver(),
                       ledger_for("takeprofittrader", schedule, start_ns, cutoff_ns))

    straight = fresh()
    saved = {}

    def remember(index, _day):
        if index == 3:
            saved["state"] = json.loads(json.dumps(straight.to_state()))

    run_days([straight], inputs, lambda _d: (lambda: prints), on_day_start=remember)
    straight.ledger.finish()
    resumed = fresh()
    resumed.driver.restore(saved["state"]["driver"])
    resumed.ledger.restore(saved["state"]["ledger"])
    run_days([resumed], inputs[3:], lambda _d: (lambda: prints))
    resumed.ledger.finish()
    a = json.dumps(json.loads(json.dumps(straight.ledger.snapshot())), sort_keys=True)
    b = json.dumps(json.loads(json.dumps(resumed.ledger.snapshot())), sort_keys=True)
    assert a == b
    assert len(straight.ledger.accounts) >= 2


def _output(runs, name="CFG"):
    return {
        "configuration": name, "display_name": "Synthetic configuration", "axes": {},
        "settings_plain": [{"setting": "Entry hours", "value": "synthetic"}],
        "reference": {"equivalent": True, "saved_study_trades": 0, "replayed_trades": 0},
        "resumed": {"takeprofittrader": {"identical": True}},
        "prints": {"minutes_checked": 1, "minutes_rebuilt_exactly": 1, "missing_utc_days": [],
                   "files": []},
        "pairs": {run.ledger.profile.firm_key: {
            "pair_id": run.pair_id, "ledger": json.loads(json.dumps(run.ledger.snapshot())),
            "forced_flat": run.driver.forced_flat, "trades_not_in_reference": 0}
            for run in runs},
    }


def test_result_validation_passes_and_detects_tampering():
    days = [winner(DAYS[0]), quiet(DAYS[1]), quiet(DAYS[2]), loser(DAYS[3], drop=420, stop=600),
            winner(DAYS[4], gain=100)]
    run, _ledger, (_inputs, schedule, start_ns, cutoff_ns) = run_pair(days)
    output = _output([run])
    kwargs = dict(context={}, failures=[], profiles=(TPT,), trading_days=schedule,
                  start_ns=start_ns, cutoff_ns=cutoff_ns, settings={"tick_value_cents": 500})
    result = build_comparison_result(outputs=[output], **kwargs)
    report = validate_comparison(result, [output], (TPT,), cutoff_ns)
    assert report["passed"], report
    summary = result["summaries_cents"]["CFG|takeprofittrader"]
    assert summary["accounts_purchased"] == 2 and summary["accounts_lost_after_a_payout"] == 1
    assert summary["net_cash_earned_cents"] == 71_178 - 20_400
    tampered = json.loads(json.dumps(output))
    tampered["pairs"]["takeprofittrader"]["ledger"]["cash_ledger"][0]["amount_cents"] = 0
    bad = validate_comparison(result, [tampered], (TPT,), cutoff_ns)
    assert not bad["passed"]


@pytest.mark.parametrize("firm_key", ["takeprofittrader", "myfundedfutures"])
def test_failure_bar_never_starts_a_replacement_trade(firm_key):
    day = loser(DAYS[0], drop=450)
    day.signals[1] = ("long", BASE - 900, BASE + 900)  # the failure candle itself
    day.signals[3] = ("long", BASE - 900, BASE + 900)
    run, ledger, _ = run_pair([day, quiet(DAYS[1])], firm_key=firm_key)
    second = ledger.accounts[1]
    trades = [t for t in ledger.trades if t["account_id"] == second.account_id]
    assert len(trades) == 1 and trades[0]["entry_ns"] > second.created_ns
