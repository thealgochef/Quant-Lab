"""Funded-payout engine: the specification's exact synthetic cases (section 11).

All prices and dates are SYNTHETIC fixtures (costs are zero unless a test says
otherwise). They exercise the engine's rules; they are not historical evidence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from alpha_lab.propsim.funded.campaign import (
    FirmCampaign,
    SettingsError,
    resumed_equivalence,
    run_campaign,
    validate_inputs,
)
from alpha_lab.propsim.funded.clock import (
    ELAPSED_48_HOURS,
    TWO_BUSINESS_DAYS_FED_1600,
    processing_due_ns,
    to_ns,
)
from alpha_lab.propsim.funded.instance import FirmInstance, split_gross
from alpha_lab.propsim.funded.paths import minute_scenario_path
from alpha_lab.propsim.funded.positions import AccountMarks, simulate_position
from alpha_lab.propsim.funded.profiles import (
    MYFUNDEDFUTURES_PROFILE as MFF,
)
from alpha_lab.propsim.funded.profiles import (
    TAKEPROFITTRADER_PROFILE as TPT,
)
from alpha_lab.propsim.funded.result import build_result, validate_result
from tests.propsim.funded.builders import BASE, execution, make_inputs, path, ticks_for

START = "2026-01-12T23:00:00Z"  # Monday 5:00 PM CST, first trading day Jan 13
WEEK = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]


def _one(ex, pts, **kw):
    kw.setdefault("days", WEEK)
    kw.setdefault("start", START)
    kw.setdefault("cutoff", "2026-01-16T22:00:00Z")
    return make_inputs([ex], [path(ex, pts)], **kw)


def _account(instance, number=1):
    return instance.accounts[number - 1]


# ── fresh funded start ────────────────────────────────────────────────────


def test_fresh_funded_start_has_no_cushion_and_one_paid_acquisition_each():
    out = run_campaign(make_inputs([], [], days=WEEK, start=START,
                                   cutoff="2026-01-16T22:00:00Z"))
    for key, cost in (("takeprofittrader", 10_200), ("myfundedfutures", 12_500)):
        firm = out[key]
        assert len(firm.accounts) == 5
        assert all(a.balance == 0 and a.floor == -200_000 for a in firm.accounts)
        assert all(a.funding == "initial_credit" for a in firm.accounts)
        assert firm.credits == 0
        assert firm.acquisition_costs == 5 * cost
        purchases = [r for r in firm.cash_ledger if r["kind"] == "account_purchase"]
        assert [r["amount_cents"] for r in purchases] == [cost] * 5
        assert {r["funding"] for r in purchases} == {"monthly_credit"}
        # no evaluation phase, no evaluation/activation/reset fee exists
        assert {r["kind"] for r in firm.cash_ledger} == {"account_purchase"}


# ── TakeProfitTrader intraday threshold ──────────────────────────────────


def test_tpt_open_equity_51500_then_49500_fails_and_recovery_cannot_pay():
    ex = execution("t1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=3_000, day="2026-01-13")
    pts = [("2026-01-13T15:10:00Z", 1_500), ("2026-01-13T15:20:00Z", -500),
           ("2026-01-13T15:30:00Z", 3_000)]
    firm = run_campaign(_one(ex, pts, profiles=(TPT,)))["takeprofittrader"]
    account = _account(firm)
    assert account.status == "failed"
    failure = [r for r in firm.boundary_evidence if r["check"] == "account_failure"][0]
    assert failure["floor_cents"] == -50_000  # 51,500 peak - 2,000 = 49,500
    assert failure["equity_cents"] == -50_000
    assert failure["ts_ns"] == to_ns("2026-01-13T15:20:00Z")
    assert account.balance == -50_000  # liquidated at the observed print
    assert firm.receipts == 0 and not firm.payout_events


def test_tpt_floor_stops_rising_at_the_starting_balance():
    ex = execution("t1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=2_400, day="2026-01-13")
    pts = [("2026-01-13T15:10:00Z", 2_500), ("2026-01-13T15:20:00Z", 2_450)]
    firm = run_campaign(_one(ex, pts, profiles=(TPT,)))["takeprofittrader"]
    account = _account(firm)
    assert account.floor == 0 and account.peak == 250_000
    # a higher peak later never raises the floor above zero
    ex2 = execution("t2", entry="2026-01-14T15:00:00Z", exit="2026-01-14T16:00:00Z",
                    move_usd=-2_400, day="2026-01-14", reason="stop", stop_usd=2_400)
    pts2 = [("2026-01-14T15:10:00Z", -2_395)]
    inputs = make_inputs([ex, ex2], [path(ex, pts), path(ex2, pts2)], days=WEEK,
                         start=START, cutoff="2026-01-16T22:00:00Z", profiles=(TPT,))
    firm = run_campaign(inputs)["takeprofittrader"]
    account = _account(firm)
    assert account.status == "failed"  # equity 0 touches the locked zero floor
    failure = [r for r in firm.boundary_evidence if r["check"] == "account_failure"][0]
    assert failure["floor_cents"] == 0 and failure["equity_cents"] == 0
    assert failure["comparator"] == "at_or_below"


# ── MyFundedFutures end-of-day ratchet with open-position enforcement ────


def test_mff_open_gain_and_closed_profit_do_not_ratchet_until_session_close():
    ex = execution("m1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=1_000, day="2026-01-13")
    pts = [("2026-01-13T15:10:00Z", 1_500)]
    ex2 = execution("m2", entry="2026-01-14T15:00:00Z", exit="2026-01-14T16:00:00Z",
                    move_usd=-3_000, day="2026-01-14", reason="stop", stop_usd=3_000)
    pts2 = [("2026-01-14T15:10:00Z", -1_900), ("2026-01-14T15:20:00Z", -2_005)]
    inputs = make_inputs([ex, ex2], [path(ex, pts), path(ex2, pts2)], days=WEEK,
                         start=START, cutoff="2026-01-16T22:00:00Z", profiles=(MFF,))
    firm = run_campaign(inputs)["myfundedfutures"]
    moves = [r for r in firm.boundary_evidence
             if r["check"] == "floor_moved" and r["account_id"].endswith("001")]
    # exactly one floor move: at the Jan 13 4:00 PM CST close, to -1,000
    assert [(r["ts_ns"], r["prior_floor_cents"], r["new_floor_cents"]) for r in moves] == [
        (to_ns("2026-01-13T22:00:00Z"), -200_000, -100_000)
    ]
    exit_row = [r for r in firm.account_events
                if r["event"] == "in_trade->ready" and r["account_id"].endswith("001")][0]
    assert exit_row["floor_cents"] == -200_000  # unchanged right after the exit
    failure = [r for r in firm.boundary_evidence if r["check"] == "account_failure"][0]
    # 1,000 - 2,005 = -1,005 is below the -1,000 floor while the trade is open
    assert failure["equity_cents"] == -100_500 and failure["floor_cents"] == -100_000
    assert failure["ts_ns"] == to_ns("2026-01-14T15:20:00Z")


def test_mff_prior_loss_leaves_less_room_and_never_resets_downward():
    ex = execution("m1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=-1_500, day="2026-01-13", reason="stop", stop_usd=1_500)
    ex2 = execution("m2", entry="2026-01-14T15:00:00Z", exit="2026-01-14T16:00:00Z",
                    move_usd=-1_000, day="2026-01-14", reason="stop", stop_usd=1_000)
    pts2 = [("2026-01-14T15:10:00Z", -500)]
    inputs = make_inputs([ex, ex2], [path(ex, []), path(ex2, pts2)], days=WEEK,
                         start=START, cutoff="2026-01-16T22:00:00Z", profiles=(MFF,))
    firm = run_campaign(inputs)["myfundedfutures"]
    account = _account(firm)
    # no new closing high: the floor stays -2,000 (not a fresh daily budget)
    assert not [r for r in firm.boundary_evidence if r["check"] == "floor_moved"]
    assert account.status == "failed"
    failure = [r for r in firm.boundary_evidence if r["check"] == "account_failure"][0]
    assert failure["equity_cents"] == -200_000  # -1,500 then -500 more: at the floor


@pytest.mark.parametrize(
    ("balance", "floor", "breached"),
    [
        (210_000, 10_000, False),  # locked: equity exactly +$100 is NOT below
        (209_999, 10_000, True),  # one cent below the locked floor fails
        (0, -200_000, True),  # before lock: equality fails (owner-approved assumption)
        (1, -200_000, False),  # one cent above the pre-lock floor survives
    ],
)
def test_mff_comparators_equality_and_one_cent(balance, floor, breached):
    ex = execution("b1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=-2_000, day="2026-01-13", reason="stop", stop_usd=2_000)
    outcome = simulate_position(
        profile=MFF, marks=AccountMarks(balance, floor, 0), execution=ex,
        path=path(ex, [("2026-01-13T15:30:00Z", -2_000)], touch=False),
        quantity=1, tick_value_cents=500, cost_per_side_cents=0,
    )
    assert outcome.failed is breached
    if breached:
        assert outcome.failure_comparator == ("below" if floor == 10_000 else "at_or_below")


# ── payout eligibility, request and processing ──────────────────────────


@pytest.mark.parametrize(
    ("profit", "gross", "tpt_cash", "mff_cash"),
    [(2_600, 500, 400, 450), (2_650, 550, 440, 495), (5_500, 3_400, 2_720, 3_060)],
)
def test_payout_amounts_retain_2100_and_split_after_request(profit, gross, tpt_cash, mff_cash):
    ex = execution("p1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=profit, day="2026-01-13")
    out = run_campaign(_one(ex, [("2026-01-13T15:30:00Z", 100)]))
    for key, cash in (("takeprofittrader", tpt_cash), ("myfundedfutures", mff_cash)):
        firm = out[key]
        requested = [r for r in firm.payout_events if r["event"] == "requested"]
        received = [r for r in firm.payout_events if r["event"] == "received"]
        assert len(requested) == 5 and len(received) == 5
        for r in requested:
            assert r["gross_cents"] == gross * 100
            assert r["balance_after_cents"] == 210_000
            assert r["requested_ns"] == to_ns("2026-01-13T22:00:00Z")  # day end, not intraday
            assert r["trader_cents"] + r["firm_share_cents"] == r["gross_cents"]
        for r in received:
            assert r["trader_cents"] == cash * 100
            assert r["received_ns"] == to_ns("2026-01-15T22:00:00Z")  # 48 h after request
        assert firm.receipts == 5 * cash * 100
        secured = [r for r in firm.payout_events if r["event"] == "eligibility_secured"]
        assert all(r["ts_ns"] == to_ns("2026-01-13T16:00:00Z") for r in secured)


def test_2599_99_is_not_eligible_and_2600_is():
    firm = FirmInstance(profile=TPT, processing=ELAPSED_48_HOURS, quantity=1,
                        tick_value_cents=500, cost_per_side_cents=0,
                        start_ns=to_ns(START))
    firm.start()
    first, second = firm.accounts[0], firm.accounts[1]
    first.balance, second.balance = 259_999, 260_000
    firm._check_secure(first, to_ns("2026-01-13T16:00:00Z"), "2026-01-13")
    firm._check_secure(second, to_ns("2026-01-13T16:00:00Z"), "2026-01-13")
    assert first.status == "ready" and second.status == "secured"
    assert split_gross(50_000, 80) == (40_000, 10_000)
    assert split_gross(55_001, 90) == (49_501, 5_500)  # half-up, parts sum exactly


def test_open_equity_touching_2600_is_not_realized_eligibility():
    ex = execution("p1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=2_000, day="2026-01-13", reason="scheduled_close")
    firm = run_campaign(_one(ex, [("2026-01-13T15:30:00Z", 2_700)],
                             profiles=(MFF,)))["myfundedfutures"]
    account = _account(firm)
    assert account.balance == 200_000  # the trade kept its own exit; no early close
    assert not [r for r in firm.payout_events]


def test_qualifying_exit_blocks_a_same_cursor_entry_and_waiting_entries():
    ex = execution("p1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=2_650, day="2026-01-13")
    same = execution("p2", entry="2026-01-13T16:00:00Z", exit="2026-01-13T17:00:00Z",
                     move_usd=500, day="2026-01-13")
    during = execution("p3", entry="2026-01-14T15:00:00Z", exit="2026-01-14T16:00:00Z",
                       move_usd=500, day="2026-01-14")
    after = execution("p4", entry="2026-01-16T15:00:00Z", exit="2026-01-16T16:00:00Z",
                      move_usd=500, day="2026-01-16")
    exes = [ex, same, during, after]
    inputs = make_inputs(exes, [path(e, []) for e in exes], days=WEEK, start=START,
                         cutoff="2026-01-16T22:00:00Z")
    for firm in run_campaign(inputs).values():
        account = _account(firm)
        taken = [t["trade_id"] for t in firm.trades if t["account_id"] == account.account_id]
        assert taken == ["p1", "p4"]  # p2 same cursor, p3 during processing
        assert account.blocked_signals == {"payout_protection": 1, "payout_processing": 1}
        blocked = [r for r in firm.boundary_evidence
                   if r["check"] == "entry_blocked_by_payout_policy"]
        assert len(blocked) == 10


def test_mff_session_close_floor_update_precedes_the_withdrawal_debit():
    ex = execution("p1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=2_650, day="2026-01-13")
    firm = run_campaign(_one(ex, [], profiles=(MFF,)))["myfundedfutures"]
    move = [r for r in firm.boundary_evidence if r["check"] == "floor_moved"][0]
    request = [r for r in firm.payout_events if r["event"] == "requested"][0]
    assert move["new_floor_cents"] == 10_000  # min(100, 2,650 - 2,000)
    assert move["seq"] < request["seq"]
    assert request["floor_cents"] == 10_000 and request["balance_after_cents"] == 210_000
    assert _account(firm).floor == 10_000  # withdrawal never lowers the floor


def test_request_at_run_cutoff_stays_processing_not_received():
    ex = execution("p1", entry="2026-01-16T15:00:00Z", exit="2026-01-16T16:00:00Z",
                   move_usd=3_000, day="2026-01-16")
    inputs = _one(ex, [], processing=TWO_BUSINESS_DAYS_FED_1600)
    out = run_campaign(inputs)
    result = build_result(inputs, out, run_identity={}, price_evidence={})
    for key in out:
        summary = result["summaries_cents"][key]
        assert summary["payouts_received_cents"] == 0
        assert summary["payouts_processing_at_cutoff"] == 5
        assert summary["pending_gross_at_cutoff_cents"] == 5 * 90_000
    events = {r["event"] for r in result["tables"]["payout_events"]}
    assert "processing_not_received_at_cutoff" in events


def test_secured_before_cutoff_without_a_day_end_is_secured_not_requested():
    ex = execution("p1", entry="2026-01-16T15:00:00Z", exit="2026-01-16T16:00:00Z",
                   move_usd=3_000, day="2026-01-16")
    inputs = _one(ex, [], cutoff="2026-01-16T18:00:00Z")
    out = run_campaign(inputs)
    result = build_result(inputs, out, run_identity={}, price_evidence={})
    assert result["summaries_cents"]["takeprofittrader"]["secured_not_requested_at_cutoff"] == 5
    assert result["summaries_cents"]["takeprofittrader"]["gross_withdrawals_requested_cents"] == 0


# ── processing clocks ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("requested", "elapsed", "business"),
    [
        # Monday: both clocks can give the same Wednesday
        ("2026-01-12T22:00:00Z", "2026-01-14T22:00:00Z", "2026-01-14T22:00:00Z"),
        # Friday: 48 h ends Sunday; business days end Tuesday
        ("2026-01-09T22:00:00Z", "2026-01-11T22:00:00Z", "2026-01-13T22:00:00Z"),
        # Friday before the MLK holiday (Monday Jan 19): business days end Wednesday
        ("2026-01-16T22:00:00Z", "2026-01-18T22:00:00Z", "2026-01-21T22:00:00Z"),
        # daylight saving starts Sunday Mar 8: elapsed time is absolute
        ("2026-03-06T22:00:00Z", "2026-03-08T22:00:00Z", "2026-03-10T21:00:00Z"),
    ],
)
def test_processing_clocks_both_modes(requested, elapsed, business):
    assert processing_due_ns(to_ns(requested), ELAPSED_48_HOURS) == to_ns(elapsed)
    assert processing_due_ns(to_ns(requested), TWO_BUSINESS_DAYS_FED_1600) == to_ns(business)


def test_processing_completes_on_calendar_time_without_price_events_and_credits_once():
    ex = execution("p1", entry="2026-01-16T15:00:00Z", exit="2026-01-16T16:00:00Z",
                   move_usd=3_000, day="2026-01-16")
    after = execution("p2", entry="2026-01-21T15:00:00Z", exit="2026-01-21T16:00:00Z",
                      move_usd=100, day="2026-01-21")
    days = ["2026-01-16", "2026-01-20", "2026-01-21"]
    inputs = make_inputs([ex, after], [path(ex, []), path(after, [])], days=days,
                         start="2026-01-15T23:00:00Z", cutoff="2026-01-21T22:00:00Z",
                         processing=ELAPSED_48_HOURS, profiles=(TPT,))
    campaign = FirmCampaign(inputs, TPT)
    campaign.run()
    firm = campaign.instance
    received = [r for r in firm.payout_events if r["event"] == "received"]
    assert {r["received_ns"] for r in received} == {to_ns("2026-01-18T22:00:00Z")}  # Sunday
    assert len(received) == 5 and firm.receipts == 5 * 72_000
    # a repeated completion is idempotent
    account = _account(firm)
    firm.processing_complete(to_ns("2026-01-19T00:00:00Z"), account.account_id,
                             f"{account.account_id}-payout-90000")
    assert firm.receipts == 5 * 72_000
    # resumed with the same strategy after the pause: the next NEW signal
    assert [t["trade_id"] for t in firm.trades if t["account_id"] == account.account_id] == [
        "p1", "p2"]


def test_checkpoint_resume_preserves_pending_payouts_without_duplicates():
    ex = execution("p1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=3_000, day="2026-01-13")
    ex2 = execution("p2", entry="2026-01-16T15:00:00Z", exit="2026-01-16T16:00:00Z",
                    move_usd=-500, day="2026-01-16", reason="stop", stop_usd=500)
    inputs = make_inputs([ex, ex2], [path(ex, []), path(ex2, [])], days=WEEK, start=START,
                         cutoff="2026-01-16T22:00:00Z", processing=TWO_BUSINESS_DAYS_FED_1600)
    for profile in inputs.profiles:
        first = FirmCampaign(inputs, profile)
        # stop right after the Jan 13 request (pending, locked, due Jan 15)
        while True:
            first.run(stop_after_events=first.processed + 1)
            if any(r["event"] == "requested" for r in first.instance.payout_events):
                break
        saved = first.checkpoint()
        state = json.loads(saved)
        pending = [a["pending_payout"] for a in state["instance"]["accounts"]]
        assert all(p and p["due_ns"] == to_ns("2026-01-15T22:00:00Z") for p in pending)
        resumed = FirmCampaign.resume(inputs, profile, saved)
        resumed.run()
        straight = FirmCampaign(inputs, profile)
        straight.run()
        assert resumed.instance.snapshot() == straight.instance.snapshot()
        assert len([r for r in resumed.instance.payout_events if r["event"] == "requested"]) == 5
    report = resumed_equivalence(inputs)
    assert all(item["identical"] for item in report.values())


# ── money isolation, credits, replacement and growth ────────────────────


def test_large_payout_then_failure_keeps_the_receipt_and_charges_only_bought_replacements():
    win = execution("w", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                    move_usd=5_500, day="2026-01-13")
    loss = execution("l", entry="2026-01-16T15:00:00Z", exit="2026-01-16T16:00:00Z",
                     move_usd=-2_500, day="2026-01-16", reason="stop", stop_usd=2_500)
    inputs = make_inputs([win, loss], [path(win, []), path(loss, [])],
                         days=WEEK, start=START, cutoff="2026-02-02T22:00:00Z",
                         profiles=(TPT,))
    firm = run_campaign(inputs)["takeprofittrader"]
    assert all(a.status == "failed" for a in firm.accounts[:5])
    assert all(a.payouts_received == 1 for a in firm.accounts[:5])
    assert firm.receipts == 5 * 272_000
    # the Jan 15 receipts fund one growth block from the wallet (13,600 >= 2,040);
    # those five fresh accounts also take the Jan 16 loss and fail unpaid
    growth = [a for a in firm.accounts if a.funding == "growth_wallet"]
    assert len(growth) == 5 and all(a.status == "failed" for a in growth)
    # no credits in January: ten vacancies wait; February's five credits buy
    # five replacements (stable vacancy order) and five slots keep waiting
    replacements = [a for a in firm.accounts if a.funding == "replacement_credit"]
    assert len(replacements) == 5
    assert {a.created_ns for a in replacements} == {to_ns("2026-02-01T06:00:00Z")}
    assert [a.replaces for a in replacements] == [f"takeprofittrader-00{i}" for i in range(1, 6)]
    assert len(firm.vacancies) == 5
    assert firm.acquisition_costs == 15 * 10_200
    result = build_result(inputs, {"takeprofittrader": firm}, run_identity={},
                          price_evidence={})
    summary = result["summaries_cents"]["takeprofittrader"]
    assert summary["accounts_lost_after_a_payout"] == 5
    assert summary["accounts_lost_before_first_payout"] == 5
    assert summary["acquisition_costs_from_wallet_cents"] == 51_000
    assert summary["net_cash_earned_cents"] == 5 * 272_000 - 15 * 10_200
    assert summary["payout_wallet_at_end_cents"] == 5 * 272_000 - 51_000


def test_all_five_fail_together_and_wait_for_credits_despite_the_other_firm():
    loss = execution("l", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                     move_usd=-2_000, day="2026-01-13", reason="stop", stop_usd=2_000)
    other = execution("o", entry="2026-01-14T15:00:00Z", exit="2026-01-14T16:00:00Z",
                      move_usd=100, day="2026-01-14")
    inputs = make_inputs([loss, other], [path(loss, []), path(other, [])], days=WEEK,
                         start=START, cutoff="2026-01-16T22:00:00Z")
    out = run_campaign(inputs)
    tpt = out["takeprofittrader"]
    assert len(tpt.accounts) == 5 and len(tpt.vacancies) == 5 and tpt.credits == 0
    assert tpt.missed_for_credit == 5  # five empty slots x one later signal
    # MyFundedFutures' floor is -2,000 exactly: -2,000 at-or-below fails too
    assert all(a.status == "failed" for a in out["myfundedfutures"].accounts)


def test_monthly_grant_once_and_unused_credits_carry():
    inputs = make_inputs([], [], days=WEEK, start=START, cutoff="2026-03-02T22:00:00Z")
    firm = run_campaign(inputs)["myfundedfutures"]
    grants = [r for r in firm.credit_events if r["kind"] == "monthly_grant"]
    assert [r["grant_id"] for r in grants] == ["2026-01", "2026-02", "2026-03"]
    assert firm.credits == 10  # January's five used; Feb + Mar carried unused
    firm.monthly_grant(to_ns("2026-03-01T06:00:00Z"), "2026-03")  # re-delivery
    assert firm.credits == 10


@pytest.mark.parametrize(
    ("profile", "wallet", "buys", "left"),
    [(TPT, 203_999, False, 203_999), (TPT, 204_000, True, 153_000),
     (MFF, 249_999, False, 249_999), (MFF, 250_000, True, 187_500)],
)
def test_growth_threshold_uses_the_received_wallet(profile, wallet, buys, left):
    firm = FirmInstance(profile=profile, processing=ELAPSED_48_HOURS, quantity=1,
                        tick_value_cents=500, cost_per_side_cents=0, start_ns=to_ns(START))
    firm.start()
    firm.receipts = wallet
    firm.wallet = wallet
    firm.review_growth(to_ns("2026-01-20T22:00:00Z"), "2026-01-20")
    assert firm.wallet == left
    assert (firm.capacity == 10) is buys
    if buys:
        # the wallet purchase enters costs ONCE; net cash is not reduced twice
        assert firm.net_cash == wallet - 2 * profile.cost_of_next_block_cents
        assert firm.wallet == wallet - profile.cost_of_next_block_cents
        # the remaining wallet cannot immediately fund the next same-price block
        firm.review_growth(to_ns("2026-01-21T22:00:00Z"), "2026-01-21")
        assert firm.capacity == 10


def test_growth_requires_positive_net_cash_and_caps_at_twenty_one_block_per_day():
    firm = FirmInstance(profile=TPT, processing=ELAPSED_48_HOURS, quantity=1,
                        tick_value_cents=500, cost_per_side_cents=0, start_ns=to_ns(START))
    firm.start()
    firm.receipts = firm.wallet = 40_000  # wallet below threshold
    firm.review_growth(to_ns("2026-01-20T22:00:00Z"), "2026-01-20")
    assert firm.capacity == 5
    firm.receipts = firm.wallet = 10_000_000
    firm.review_growth(to_ns("2026-01-21T22:00:00Z"), "2026-01-21")
    firm.review_growth(to_ns("2026-01-21T22:00:00Z"), "2026-01-21")  # same day: one only
    assert firm.capacity == 10
    for day in ("2026-01-22", "2026-01-23", "2026-01-26"):
        firm.review_growth(to_ns(f"{day}T22:00:00Z"), day)
    assert firm.capacity == 20
    decisions = [g["decision"] for g in firm.growth_events]
    assert decisions[-1] == "declined" and "20-account" in firm.growth_events[-1]["reason"]
    assert decisions.count("not_reviewed") == 1
    # negative net cash blocks growth even with a large wallet
    other = FirmInstance(profile=TPT, processing=ELAPSED_48_HOURS, quantity=1,
                         tick_value_cents=500, cost_per_side_cents=0, start_ns=to_ns(START))
    other.start()
    other.wallet, other.receipts = 300_000, 40_000  # net = 40,000 - 51,000 < 0
    other.review_growth(to_ns("2026-01-20T22:00:00Z"), "2026-01-20")
    assert other.capacity == 5


def test_pending_payout_cannot_fund_growth_before_receipt():
    ex = execution("p1", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=5_500, day="2026-01-13")
    firm = run_campaign(_one(ex, [], profiles=(TPT,)))["takeprofittrader"]
    request = [r for r in firm.payout_events if r["event"] == "requested"][0]
    growth = firm.growth_events
    assert growth and all(g["ts_ns"] > request["ts_ns"] for g in growth)
    assert growth[0]["ts_ns"] == to_ns("2026-01-15T22:00:00Z")  # the receipt time
    assert growth[0]["wallet_before_cents"] == 5 * 272_000
    assert growth[0]["decision"] == "purchased" and firm.capacity == 10


def test_pause_is_per_account_and_occupies_its_slot():
    firm = FirmInstance(profile=TPT, processing=ELAPSED_48_HOURS, quantity=1,
                        tick_value_cents=500, cost_per_side_cents=0, start_ns=to_ns(START))
    firm.start()
    firm.accounts[0].status = "processing"
    ex = execution("s", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=100, day="2026-01-13")
    scheduled = []
    firm.on_signal(to_ns(ex.entry_ts_utc), ex, path(ex, []),
                   lambda *args: scheduled.append(args))
    assert len(scheduled) == 4  # the other four trade independently
    assert firm.accounts[0].blocked_signals == {"payout_processing": 1}
    assert not firm.vacancies and len(firm.accounts) == 5  # no replacement, no extra


# ── sizes, gap-through, stale entries and approximations ────────────────


def test_firm_size_limits_are_enforced_never_clipped():
    ex = execution("s", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=100, day="2026-01-13")
    ok = _one(ex, [], quantity=3)
    validate_inputs(ok)
    with pytest.raises(SettingsError, match="MyFundedFutures"):
        validate_inputs(_one(ex, [], quantity=4))
    validate_inputs(_one(ex, [], quantity=6, profiles=(TPT,)))
    with pytest.raises(SettingsError, match="TakeProfitTrader"):
        validate_inputs(_one(ex, [], quantity=7, profiles=(TPT,)))
    validate_inputs(_one(ex, [], quantity=30, instrument="micro"))  # 30 micros = 3 minis
    with pytest.raises(SettingsError):
        validate_inputs(_one(ex, [], quantity=31, instrument="micro"))


def test_holding_through_a_session_close_is_refused():
    ex = execution("h", entry="2026-01-13T21:00:00Z", exit="2026-01-14T15:00:00Z",
                   move_usd=100, day="2026-01-13")
    with pytest.raises(SettingsError, match="daily-close"):
        validate_inputs(_one(ex, []))


def test_gap_through_fills_at_the_print_and_no_later_target_or_stale_entry():
    ex = execution("g", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                   move_usd=3_000, day="2026-01-13")
    pts = [("2026-01-13T15:10:00Z", -1_000), ("2026-01-13T15:11:00Z", -2_600)]
    nxt = execution("n", entry="2026-01-14T15:00:00Z", exit="2026-01-14T16:00:00Z",
                    move_usd=100, day="2026-01-14")
    inputs = make_inputs([ex, nxt], [path(ex, pts), path(nxt, [])], days=WEEK,
                         start=START, cutoff="2026-02-02T22:00:00Z", profiles=(TPT,))
    firm = run_campaign(inputs)["takeprofittrader"]
    rows = [t for t in firm.trades if t["trade_id"] == "g"]
    assert len(rows) == 5  # one exit per account, no double exit/cost
    assert all(t["exit_kind"] == "account_failure" for t in rows)
    assert all(t["exit_ticks"] == BASE - ticks_for(2_600) for t in rows)  # print, not floor
    assert all(t["net_pnl_cents"] == -260_000 for t in rows)
    # replacements (February) never inherit the old entry or the Jan 14 signal
    for account in firm.accounts[5:]:
        assert account.trades == 0


def test_minute_candle_order_changes_survival_and_is_labeled_approximate():
    ex = execution("a", entry="2026-01-13T15:00:00Z", exit="2026-01-13T15:03:00Z",
                   move_usd=2_500, day="2026-01-13", stop_usd=2_500)
    base = ex.entry_ticks
    t0 = to_ns("2026-01-13T15:00:00Z")
    minute = 60 * 10**9
    # candle 1 range: +1,500 / -1,000 (order unknown); candle 2 resolves at target
    bars = [
        (t0, t0 + minute, base, base + ticks_for(1_500), base - ticks_for(1_000), base),
        (t0 + minute, t0 + 2 * minute, base, base + ticks_for(2_500), base - 20, base),
    ]
    adverse = minute_scenario_path(ex, bars, order="adverse_first")
    favorable = minute_scenario_path(ex, bars, order="favorable_first")
    outcomes = {}
    for label, p in (("adverse", adverse), ("favorable", favorable)):
        outcomes[label] = simulate_position(
            profile=TPT, marks=AccountMarks(0, -200_000, 0), execution=ex, path=p,
            quantity=1, tick_value_cents=500, cost_per_side_cents=0)
        assert p.is_approximate
    # losing side first: -1,000 vs floor -2,000 survives; winning side first
    # raises the floor to -500, then -1,000 breaches at the floor price
    assert not outcomes["adverse"].failed
    assert outcomes["favorable"].failed
    assert outcomes["favorable"].exit_ticks == base - ticks_for(500)
    assert "approximation" in outcomes["favorable"].failure_fill_basis
    assert isinstance(adverse.ts_ns, np.ndarray)


def test_validation_reconciles_money_and_isolation():
    win = execution("w", entry="2026-01-13T15:00:00Z", exit="2026-01-13T16:00:00Z",
                    move_usd=5_500, day="2026-01-13")
    loss = execution("l", entry="2026-01-16T15:00:00Z", exit="2026-01-16T16:00:00Z",
                     move_usd=-2_500, day="2026-01-16", reason="stop", stop_usd=2_500)
    inputs = make_inputs([win, loss], [path(win, []), path(loss, [])], days=WEEK,
                         start=START, cutoff="2026-02-03T22:00:00Z",
                         cost_per_side_cents=514)
    out = run_campaign(inputs)
    result = build_result(inputs, out, run_identity={}, price_evidence={})
    report = validate_result(result, out, resumed_equivalence(inputs))
    assert report["passed"], report
    tpt, mff = (result["summaries_cents"][k] for k in ("takeprofittrader", "myfundedfutures"))
    assert tpt["acquisition_costs_cents"] % 10_200 == 0
    assert mff["acquisition_costs_cents"] % 12_500 == 0
    # costs post at both fills: a +5,500 move nets 5,489.72 before the request
    request = [r for r in out["takeprofittrader"].payout_events if r["event"] == "requested"][0]
    assert request["gross_cents"] == 550_000 - 1_028 - 210_000


def test_ready_time_splits_market_closed_from_waiting_for_a_signal():
    inputs = make_inputs([], [], days=WEEK, start=START, cutoff="2026-01-16T22:00:00Z",
                         profiles=(TPT,))
    out = run_campaign(inputs)
    summary = build_result(inputs, out, run_identity={}, price_evidence={})[
        "summaries_cents"]["takeprofittrader"]
    # four sessions: each day 21:55Z deadline to 23:00Z reopen is closed/locked
    # (Jan 13-15 fully inside the window; Jan 16's lock starts 5 minutes before the cutoff)
    closed_hours = 5 * (3 * 65 + 5) / 60
    assert summary["hours_ready_market_closed_or_locked"] == round(closed_hours, 2)
    total = 5 * (to_ns("2026-01-16T22:00:00Z") - to_ns(START)) / 3.6e12
    assert summary["hours_ready_market_open_no_new_signal"] == round(total - closed_hours, 2)
