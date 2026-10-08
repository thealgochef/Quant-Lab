"""Analytical corrections A2, A3, A4, A5 and A9: risk models, the firm ledger, Summary findings.

Synthetic marked-profit paths and synthetic races run through the REAL funded
ledger (:class:`alpha_lab.propsim.funded.pair_ledger.PairLedger`) via
``firm_race.replay`` / ``_observations``; the ledger is never changed. Firm terms
are the built-in profiles, which equal the saved TakeProfitTrader and
MyFundedFutures terms of export v4 (``firm_rules.json``; checked below when that
export is on this computer). Costs are zero for clarity. Nothing here saves,
approves or launches anything.

Every synthetic case is a module-level constant (``id``, ``inputs``, ``expected``)
so the evidence file can import it and record the actual outputs.
"""

from __future__ import annotations

import html
import inspect
import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _path in (REPO / "scripts", REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ifvg_lab_cache  # noqa: E402
import ifvg_lab_detail_risk as risk  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.lab import firm_race as fr  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_measures as fm  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.lab import resampling as rs  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (  # noqa: E402
    study_from_result,
)
from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600, iso_utc  # noqa: E402
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES  # noqa: E402
from tests.propsim.funded.pair_builders import (  # noqa: E402
    BASE,
    SyntheticDay,
    build,
    flat,
    weekdays,
)

TPT, MFF = "takeprofittrader", "myfundedfutures"
TICK_CENTS = 500  # one E-mini tick: $5, so every dollar amount below is whole ticks
EXPORT = REPO / "reports/funded_comparison/funded_comparison_5fa65149843484b1_export_v4"
STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
SAVED = (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file()


# ── synthetic cases (imported by the evidence file) ───────────────────────

#: A2: a closed-profit fall of $2,500 that does not end an account whose floor has locked
A2_LOCKED_FLOOR = {
    "id": "A2-locked-floor-negative",
    "inputs": {"marked_path_usd": [0, 6000, 3500], "firms": [TPT, MFF], "costs_usd": 0},
    "expected": {"closed_profit_fall_usd": 2500.0, "counted_as_2000_fall": True,
                 "failed": {TPT: False, MFF: False},
                 "takeprofittrader_floor_after_trade_usd": 0.0,
                 "myfundedfutures_floor_after_session_close_usd": 100.0},
}
#: A2: an account lost inside an open trade whose later closing result is a gain
A2_OPEN_POSITION = {
    "id": "A2-open-position-failure",
    "inputs": {"marked_path_usd": [0, -2100, 500], "firms": [TPT, MFF], "costs_usd": 0},
    "expected": {"closed_result_usd": 500.0, "closed_profit_fall_usd": 0.0,
                 "failed": {TPT: True, MFF: True}, "recorded_trade_net_usd": -2000.0},
}
#: A3: two marked paths that compress to the same four points but differ at the ledger
A3_COUNTEREXAMPLE = {
    "id": "A3-compression-counterexample",
    "inputs": {"A_marked_path_usd": [0, -1000, 2500, -100, 4000, 3000],
               "B_marked_path_usd": [0, -1000, 2500, 2000, 4000, 3000],
               "firm": TPT, "floor_lock_usd": 0, "costs_usd": 0},
    "expected": {"compressed_A": [0, -1000, 4000, 3000], "compressed_B": [0, -1000, 4000, 3000],
                 "full_A_failed": True, "full_B_failed": False,
                 "compressed_time_order_failed": {"A": False, "B": False},
                 "compressed_best_first_failed": {"A": True, "B": True}},
}
#: A4: denominators of a tiny synthetic race (5 trading days, one slot a day)
A4_DENOMINATORS = {
    "id": "A4-denominators",
    "inputs": {"firm": TPT, "days": ["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05",
                                     "2026-03-06"],
               "shapes_usd": {"W": 3000, "L": -2500, "S": 10},
               "paths": [["W", "S", "S", "L", "W"], ["L", "W", "S", "S", "S"],
                         ["W", "S", "S", "L", "L"]]},
    "expected": {"accounts_purchased": 7, "failed_accounts": 4, "payouts_by_failed": 2,
                 "payouts_before_death": 0.5, "open_accounts": 3, "payouts_by_open": 1,
                 "unresolved_requests": 1, "unresolved_trader_cents": 72_000,
                 "received_cents_total": 216_000, "costs_cents_total": 71_400,
                 "cash_per_account": (216_000 - 71_400) / 7 / 100,
                 "first_account_endpoints": ["received", "failed", "received"]},
}
#: A4: the pooled ratio can sit above or below the mean of per-path ratios
A4_POOLED_VS_MEAN = [
    {"id": "A4-pooled-above-mean",
     "inputs": {"paths": [{"net_usd": 5000, "accounts": 5}, {"net_usd": -100, "accounts": 1}]},
     "expected": {"pooled_usd": 4900 / 6, "mean_of_ratios_usd": 450.0}},
    {"id": "A4-pooled-below-mean",
     "inputs": {"paths": [{"net_usd": 5000, "accounts": 1}, {"net_usd": -100, "accounts": 5}]},
     "expected": {"pooled_usd": 4900 / 6, "mean_of_ratios_usd": 2490.0}},
]
#: A4: clocks of the first account (eligibility, request, receipt) on 8 trading days
A4_CLOCKS = {
    "id": "A4-clocks",
    "inputs": {"firm": TPT, "days": 8, "first_day": "2026-03-02",
               "received_path": ["S", "S", "W", "S", "S", "S", "S", "S"],
               "unresolved_path_5_days": ["S", "S", "S", "S", "W"]},
    "expected": {"trades_to_eligibility": 3, "trades_to_request": 3, "trades_to_receipt": 3,
                 "days_strictly_increase": True, "unresolved_first_receipt": None,
                 "unresolved_requests": 1, "unresolved_trader_cents": 75_200},
}
#: A5: both percentile sentences count ties as half
A5_TIES = {
    "id": "A5-ties-count-half",
    "inputs": {"values": [100.0, 200.0, 300.0, 300.0, 400.0], "actual": 300.0},
    "expected": {"share_below": 0.6, "text": "60%"},
}
#: A9: legs with an odd quantity and an entry cost that cannot be halved in whole cents
A9_LEGS = {
    "id": "A9-odd-quantity-legs",
    "inputs": {"tick_value_cents": 50, "mills": 1030, "rows": [
        {"seq": 1, "direction": "long", "quantity": 5, "scale_out_quantity": 2,
         "final_exit_quantity": 3, "entry_ticks": 1000, "scale_out_ticks": 1010,
         "exit_ticks": 1030, "net_pnl_usd": 44.70, "costs_usd": 10.30},
        {"seq": 2, "direction": "short", "quantity": 5, "scale_out_quantity": 2,
         "final_exit_quantity": 3, "entry_ticks": 2000, "scale_out_ticks": 1990,
         "exit_ticks": 2005, "net_pnl_usd": -7.80, "costs_usd": 10.30},
        {"seq": 3, "direction": "long", "quantity": 5, "scale_out_quantity": 0,
         "final_exit_quantity": 5, "entry_ticks": 3000, "scale_out_ticks": None,
         "exit_ticks": 3004, "net_pnl_usd": -0.30, "costs_usd": 10.30}]},
    "expected": {"entry_split_cents": [206, 309], "first_cents": [588, 588, 0],
                 "rest_cents": [3882, -1368, -30], "first_halves_cents": 1176,
                 "remainders_cents": 2484, "whole_cents": 3660, "reconciled": True,
                 "floor_split_5_2_514": [102, 155]},
}
#: A9 frozen reference: the saved leader at TakeProfitTrader (skipped when absent)
A9_FROZEN = {
    "id": "A9-frozen-leader-reference",
    "inputs": {"result_id": RESULT_ID, "configuration": LEADER, "firm": TPT},
    "expected": {"count": 13, "first_halves_cents": 449_068, "remainders_cents": 4_027_818,
                 "whole_cents": 4_476_886, "reconciled": True, "half_exit": True},
}
CASES = [A2_LOCKED_FLOOR, A2_OPEN_POSITION, A3_COUNTEREXAMPLE, A4_DENOMINATORS,
         *A4_POOLED_VS_MEAN, A4_CLOCKS, A5_TIES, A9_LEGS, A9_FROZEN]


# ── synthetic ledger inputs ───────────────────────────────────────────────


def _rules(firm_key: str, days: list[date]) -> tuple[fr.FirmRules, tuple[fr.Slot, ...]]:
    """Ledger rules at zero cost on ``days`` (9:00–9:06 AM Chicago), one slot a day."""

    _inputs, schedule, start_ns, cutoff_ns = build([SyntheticDay(d, flat(6)) for d in days])
    profile = FIRM_PROFILES[firm_key]
    rules = fr.FirmRules(
        pair_id=f"CFG|{firm_key}", configuration="CFG", firm_key=firm_key,
        firm_name=profile.firm_name, profile=profile, processing=TWO_BUSINESS_DAYS_FED_1600,
        quantity=1, tick_value_cents=TICK_CENTS, cost_per_contract_mills=0, scale_out=False,
        trading_days=schedule, start_ns=start_ns, cutoff_ns=cutoff_ns)
    minute = 60_000_000_000
    slots = tuple(fr.Slot(entry_ns=d.deadline_ns - 4 * minute, exit_ns=d.deadline_ns,
                          trading_day=d.trading_day) for d in schedule)
    return rules, slots


def _ticks(usd: float) -> int:
    return BASE + int(round(usd * 100)) // TICK_CENTS


def _shape(marks_usd: list[float], index: int = 0) -> fr.TradeShape:
    """A long trade marked at these profits after its $0 entry (the last is the exit)."""

    ticks = [_ticks(m) for m in marks_usd]
    return fr.TradeShape(index=index, direction="long", entry_ticks=BASE, exit_ticks=ticks[-1],
                         stop_ticks=min(BASE, *ticks) - 1, target_ticks=max(BASE, *ticks) + 1,
                         path_ticks=tuple(ticks), net_cents=int(round(marks_usd[-1] * 100)),
                         scaled=False, order_pinned=True, exact=True)


def _one_trade(firm_key: str, marks_usd: list[float]):
    rules, slots = _rules(firm_key, [date(2026, 3, 2)])
    return fr.replay(rules, slots, [_shape(marks_usd)], [0])


def compress(marked: list[float]) -> tuple[float, float, float, float]:
    """(entry, lowest, highest, exit): what the adapter keeps of a trade's path."""

    return (marked[0], min(marked), max(marked), marked[-1])


def compress_with_times(marked: list[float]) -> tuple:
    """The compressed tuple with the positions of the low AND the high."""

    return (marked[0], (min(marked), marked.index(min(marked))),
            (max(marked), marked.index(max(marked))), marked[-1])


# ── A2: a closed-profit drawdown is not an account failure ────────────────


def test_a2_locked_floor_fall_is_counted_but_does_not_fail_the_account():
    """A2: $0 → +$6,000 → +$3,500 counts as a $2,500 closed-profit fall, yet no account fails.

    The drawdown measure (running high minus value, as ``drawdown_growth`` computes)
    counts it against $2,000; the ledger with TakeProfitTrader's terms locks the
    floor at $0 and with MyFundedFutures' terms moves it only at the session close,
    so neither fails.
    """

    case = A2_LOCKED_FLOOR
    path = np.asarray(case["inputs"]["marked_path_usd"], dtype=float)
    fall = float(rs.running_fall(path)[-1])
    assert fall == case["expected"]["closed_profit_fall_usd"] and fall >= 2000.0
    for firm_key in (TPT, MFF):
        ledger = _one_trade(firm_key, [6000, 3500])
        assert ledger.accounts[0].failed_ns is None, firm_key
        assert not ledger.trades[0]["account_failed"]
        assert ledger.trades[0]["net_pnl_cents"] == 350_000
    tpt = _one_trade(TPT, [6000, 3500])
    assert tpt.trades[0]["floor_after_cents"] == FIRM_PROFILES[TPT].floor_lock_cents == 0
    mff = _one_trade(MFF, [6000, 3500])
    assert mff.accounts[0].floor == FIRM_PROFILES[MFF].floor_lock_cents == 10_000


def test_a2_open_position_failure_is_hidden_from_the_closed_result():
    """A2: $0 → −$2,100 → +$500 closes at +$500 with no fall, but the ledger fails it inside.

    Looking only at the later closing result misses the loss inside the open trade.
    """

    closed = np.asarray([0.0, 500.0])
    assert float(rs.running_fall(closed)[-1]) == 0.0
    for firm_key in (TPT, MFF):
        ledger = _one_trade(firm_key, [-2100, 500])
        assert ledger.accounts[0].failed_ns is not None, firm_key
        assert ledger.trades[0]["account_failed"] is True
        assert ledger.trades[0]["net_pnl_cents"] == -200_000  # liquidated at the floor


# ── A3: the compressed path cannot tell A from B ──────────────────────────


def test_a3_counterexample_compressed_paths_are_identical_but_full_paths_differ():
    """A3: A fails on the −$100 reversal, B survives; both compress to the same four points.

    A: $0 → −$1,000 → +$2,500 → −$100 → +$4,000 → +$3,000;
    B: $0 → −$1,000 → +$2,500 → +$2,000 → +$4,000 → +$3,000 (TakeProfitTrader, lock $0).
    Adding the high's timestamp does not change the compressed tuple: the low and the
    high sit at the same positions in both paths. The compressed paths in time order
    both survive; under the adapter's unpinned best-first convention both fail — so
    neither placement is a bound on the full-path outcome.
    """

    a, b = A3_COUNTEREXAMPLE["inputs"]["A_marked_path_usd"], \
        A3_COUNTEREXAMPLE["inputs"]["B_marked_path_usd"]
    assert compress(a) == compress(b) == (0, -1000, 4000, 3000)
    assert compress_with_times(a) == compress_with_times(b)  # the high's time adds nothing

    def failed(marks: list[float]) -> bool:
        return _one_trade(TPT, marks).accounts[0].failed_ns is not None

    full_a, full_b = failed(a[1:]), failed(b[1:])
    assert (full_a, full_b) == (True, False)
    _entry, low, high, end = compress(a)
    in_time_order = failed([low, high, end])
    best_first = failed([high, low, end])
    assert in_time_order is False and in_time_order != full_a
    assert best_first is True and best_first != full_b
    # the adapter's saved-row compression gives the best-first placement when unpinned
    row = {"direction": "long", "entry_ticks": BASE, "exit_ticks": _ticks(3000), "quantity": 1,
           "balance_before_usd": 0.0, "min_equity_usd": -1000.0, "max_equity_usd": 4000.0,
           "entry_utc": "2026-03-02T15:02:00Z", "exit_utc": "2026-03-02T15:06:00Z",
           "min_equity_utc": "2026-03-02T15:03:00Z", "net_pnl_usd": 3000.0,
           "exit_kind": "scheduled_close"}
    shape = fr.trade_shape(row, 0, tick_value_cents=TICK_CENTS, mills=0)
    assert shape.path_ticks == (_ticks(4000), _ticks(-1000), _ticks(3000))
    assert not shape.order_pinned


def test_a3_model_identity_and_limitations_are_exported():
    """A3: the model id and its four limitations travel with every result."""

    assert fr.MODEL_ID == "conditional_firm_ledger_resampling_v1"
    assert len(fr.LIMITATIONS) == 4
    text = " ".join(fr.LIMITATIONS)
    for words in ("Source selection", "Fixed slots", "Shortened liquidation",
                  "Compressed intratrade path"):
        assert words in text
    source = Path(fr.__file__).read_text(encoding="utf-8")
    assert "stricter choice" not in source and "labelled approximate" not in source


# ── A4: population definitions and payout clocks ──────────────────────────


def _race_paths(rules, slots, names_by_path):
    shapes = {"W": _shape([3000], 0), "L": _shape([-2500], 1), "S": _shape([10], 2)}
    ordered = [shapes["W"], shapes["L"], shapes["S"]]
    index = {"W": 0, "L": 1, "S": 2}
    return [fr.path_outcome(fr.replay(rules, slots, ordered, [index[n] for n in names]), i)
            for i, names in enumerate(names_by_path)]


def _five_days() -> list[date]:
    return weekdays(date(2026, 3, 2), 5)


def test_a4_denominators_are_exported_and_unresolved_money_is_not_received():
    """A4: failed-account average, open accounts, unresolved requests and the pooled ratio.

    Path 1: account 1 is paid then fails; account 2 is still processing a request at
    the cutoff. Path 2: account 1 fails unpaid; account 2 is open with one payout.
    Path 3: account 1 is paid then fails; account 2 fails unpaid; account 3 is open.
    """

    case = A4_DENOMINATORS
    rules, slots = _rules(TPT, _five_days())
    outcomes = _race_paths(rules, slots, case["inputs"]["paths"])
    race = fr.race_from_outcomes(rules, outcomes, paths=3, seed=0, method="blocks",
                                 slots=len(slots), horizon=fr.horizon_days(rules, slots))
    expected = case["expected"]
    assert [o.first_account_endpoint for o in outcomes] == expected["first_account_endpoints"]
    assert race.accounts_bought == expected["accounts_purchased"]
    assert race.failed_accounts == race.lost_accounts == expected["failed_accounts"]
    assert race.payouts_by_failed == expected["payouts_by_failed"]
    assert race.payouts_before_death == race.payouts_by_failed / race.failed_accounts == 0.5
    assert (race.open_accounts, race.payouts_by_open) == (3, 1)
    assert (race.unresolved_requests, race.unresolved_trader_cents) == (1, 72_000)
    assert race.received_cents_total == expected["received_cents_total"]
    assert race.costs_cents_total == expected["costs_cents_total"]
    # an unresolved request is never received cash
    assert race.received_cents_total == sum(o.received_cents for o in outcomes)
    assert outcomes[0].received_cents == 72_000 and outcomes[0].requests_unresolved == 1
    assert race.cash_per_account == pytest.approx(expected["cash_per_account"])
    assert race.cash_per_account == pytest.approx(
        (race.received_cents_total - race.costs_cents_total) / race.accounts_bought / 100)
    assert (race.horizon_first_day, race.horizon_last_day) == ("2026-03-02", "2026-03-06")
    assert race.model_id == fr.MODEL_ID
    # the screen quotes the numerator, the denominator and the separate populations
    detail = risk.payouts_detail(replace(race, paths=3))
    assert detail == (
        "2 payouts received by the 4 accounts that failed before the March 6, 2026 cutoff, "
        "across 3 resampled paths. Not a lifetime expectation. At the cutoff 3 accounts were "
        "still open (they had received 1 payout), and 1 payout request totalling $720.00 "
        "after the split was still processing — not counted as received.")
    assert risk.payouts_value(race) == "0.50 payouts"
    cash = risk.cash_detail(race, rules)
    assert cash == (
        "All paths' payouts received after the 80% split ($2,160.00) minus all account "
        "purchases ($714.00: 7 accounts at $102), divided by all 7 accounts purchased. "
        "Requested-but-unreceived payouts and money left in open accounts count as $0 "
        "received. This pools every path; it is not an average of each path's own ratio.")
    # the same definitions hold for a real resampled run
    full = fr.full_race(rules, slots, [_shape([3000], 0), _shape([-2500], 1),
                                       _shape([10], 2)], paths=30, seed=5)
    if full.failed_accounts:
        assert full.payouts_before_death == full.payouts_by_failed / full.failed_accounts
    assert full.received_cents_total - full.costs_cents_total == sum(
        o.net_cash_cents for o in full.outcomes)
    assert full.accounts_bought == sum(o.accounts_purchased for o in full.outcomes)


def _outcome(path: int, net_usd: float, accounts: int) -> fr.PathOutcome:
    costs = accounts * 10_200
    return fr.PathOutcome(
        path=path, net_cash_cents=int(net_usd * 100), received_cents=int(net_usd * 100) + costs,
        costs_cents=costs, accounts_purchased=accounts, accounts_failed=accounts - 1,
        accounts_open_at_cutoff=1, payouts_received=0, payouts_received_by_failed=0,
        payouts_received_by_open=0, requests_unresolved=0, unresolved_trader_cents=0,
        first_account_endpoint="open", first_trades_to_eligibility=None,
        first_trades_to_request=None, first_trades_to_receipt=None,
        first_trades_to_failure=None, first_days_to_eligibility=None,
        first_days_to_request=None, first_days_to_receipt=None, first_days_to_failure=None)


def test_a4_detail_says_plainly_when_no_request_is_still_processing():
    """A4: with nothing pending at the cutoff the detail says so (never "0 … totalling $0.00")."""

    rules, _slots = _rules(TPT, _five_days())
    outcomes = [replace(_outcome(0, 500.0, 2), payouts_received_by_failed=1, payouts_received=1)]
    race = replace(fr.race_from_outcomes(rules, outcomes, paths=1, seed=0, method="blocks",
                                         slots=5), paths=1)
    assert race.unresolved_requests == 0
    detail = risk.payouts_detail(race)
    assert detail.endswith("and no payout request was still processing.")
    assert "totalling $0.00" not in detail


@pytest.mark.parametrize("case", A4_POOLED_VS_MEAN, ids=lambda c: c["id"])
def test_a4_pooled_ratio_is_not_bounded_by_the_mean_of_ratios(case):
    """A4: pooled net cash per purchased account can be above OR below the mean of ratios.

    +$5,000 over 5 accounts and −$100 over 1 give pooled $816.67 > mean $450; swapping
    the account counts gives pooled $816.67 < mean $2,490. Neither direction is
    guaranteed, so the screen claims neither.
    """

    rules, _slots = _rules(TPT, _five_days())
    paths = case["inputs"]["paths"]
    outcomes = [_outcome(i, p["net_usd"], p["accounts"]) for i, p in enumerate(paths)]
    race = fr.race_from_outcomes(rules, outcomes, paths=len(paths), seed=0, method="blocks",
                                 slots=5)
    mean = float(np.mean([p["net_usd"] / p["accounts"] for p in paths]))
    assert race.cash_per_account == pytest.approx(case["expected"]["pooled_usd"])
    assert mean == pytest.approx(case["expected"]["mean_of_ratios_usd"])
    assert race.cash_per_account != pytest.approx(mean)
    assert "lower than" not in risk.cash_detail(race, rules)
    assert "higher than" not in risk.cash_detail(race, rules)


def test_a4_clocks_separate_eligibility_request_and_receipt():
    """A4: trades to eligibility = request = receipt = k, while the days differ.

    Entries are refused while the account is secured or processing, so no trade
    closes between eligibility and receipt; the request is at that trading day's
    end and the receipt after the saved two-business-day clock. A request still
    processing at the cutoff has no receipt clock and is counted unresolved.
    """

    rules, slots = _rules(TPT, weekdays(date(2026, 3, 2), 8))
    (outcome,) = _race_paths(rules, slots, [A4_CLOCKS["inputs"]["received_path"]])
    k = A4_CLOCKS["expected"]["trades_to_eligibility"]
    assert (outcome.first_trades_to_eligibility, outcome.first_trades_to_request,
            outcome.first_trades_to_receipt) == (k, k, k)
    assert (outcome.first_days_to_eligibility < outcome.first_days_to_request
            < outcome.first_days_to_receipt)
    # Wednesday 9:06 AM eligibility, 9:11 AM request, Friday 4:00 PM receipt (from 8:59 AM Mon)
    assert outcome.first_days_to_receipt - outcome.first_days_to_request == pytest.approx(
        2 + (16 * 60 - (9 * 60 + 11)) / (24 * 60))
    assert outcome.first_account_endpoint == "received" and outcome.requests_unresolved == 0
    five, five_slots = _rules(TPT, _five_days())
    (pending,) = _race_paths(five, five_slots, [A4_CLOCKS["inputs"]["unresolved_path_5_days"]])
    assert pending.first_trades_to_request == 5 and pending.first_trades_to_receipt is None
    assert pending.first_days_to_receipt is None and pending.first_account_endpoint == "open"
    assert (pending.requests_unresolved, pending.unresolved_trader_cents) == (1, 75_200)
    assert pending.received_cents == 0


def _fake_study(rules: fr.FirmRules, slots) -> object:
    profile = rules.profile.model_dump(mode="json")
    result = {
        "settings": {"firm_profiles": [profile],
                     "processing_clock": rules.processing.model_dump(mode="json")},
        "period": {"cutoff_utc": iso_utc(rules.cutoff_ns)},
        "summaries_cents": {},
        "tables": {"trades": [{"pair_id": rules.pair_id, "configuration": "CFG",
                               "firm_key": rules.firm_key, "entry_utc": iso_utc(s.entry_ns),
                               "seq": i} for i, s in enumerate(slots)]},
    }
    return study_from_result(result, result_id="r" * 64)


def test_a4_greedy_policy_and_key_ignore_the_typed_boundaries():
    """A4: the ledger run and its cache key never depend on the fixed diagnostic's boundaries."""

    rules, slots = _rules(TPT, _five_days())
    shapes = [_shape([3000], 0), _shape([-2500], 1), _shape([10], 2)]
    first = fr.full_race(rules, slots, shapes, paths=20, seed=9)
    again = fr.full_race(rules, slots, shapes, paths=20, seed=9)
    assert replace(first, seconds=0) == replace(again, seconds=0)
    for name in ("loss_limit", "trigger", "lower", "upper", "boundaries"):
        assert name not in inspect.signature(fr.full_race).parameters
        assert name not in inspect.signature(ifvg_lab_cache.firm_race_key).parameters
    ctx = SimpleNamespace(study=_fake_study(rules, slots), store_root="store",
                          result_id="r" * 64, configuration="CFG", firm_key=TPT)
    keys = []
    for lower, upper in ((-2000.0, 2600.0), (-1500.0, 3100.0)):
        values = [float(s.net_cents) / 100 for s in shapes] * 4
        rs.payout_race(values, loss_limit=lower, trigger=upper, paths=200, seed=9)
        keys.append(risk.ledger_key(ctx, rules, slots, seed=9, paths=20))
    assert keys[0] == keys[1] and keys[0] is not None
    assert not {-2000.0, 2600.0, -1500.0, 3100.0} & {v for v in keys[0]
                                                      if isinstance(v, float | int)}
    assert keys[0][4] == fr.MODEL_ID and keys[0][5] == "blocks"


# ── A3: the cache never returns another model's result ────────────────────


def test_a3_cache_refuses_another_model_id_or_terms_digest(monkeypatch):
    """A3: ``default_firm_race`` returns only this model's result under the saved terms."""

    rules, slots = _rules(TPT, _five_days())
    study = _fake_study(rules, slots)
    monkeypatch.setattr(ifvg_lab_cache, "_study", lambda *_args: study)
    slots_n, cutoff_ns, digest = ifvg_lab_cache.saved_race_binding(study, "CFG", TPT)
    assert (slots_n, cutoff_ns) == (len(slots), rules.cutoff_ns)
    shapes = [_shape([3000], 0), _shape([-2500], 1), _shape([10], 2)]
    race = fr.full_race(rules, slots, shapes, paths=5, seed=fm.DEFAULT_SEED)
    right = ifvg_lab_cache.firm_race_key("store", "r" * 64, "CFG", TPT, method="blocks",
                                         seed=fm.DEFAULT_SEED, paths=fr.DEFAULT_FULL_PATHS,
                                         slots=slots_n, cutoff_ns=cutoff_ns, terms_digest=digest)
    old_model = (*right[:4], "payout_race_full_version_v0", *right[5:])
    other_terms = (*right[:-1], fr.terms_digest({"loss_allowance_cents": 150_000},
                                                 rules.processing.model_dump(mode="json")))
    for store in ({old_model: race}, {other_terms: race}, {right: replace(race, model_id="x")}):
        monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda s=store: s)
        assert ifvg_lab_cache.default_firm_race("store", "r" * 64, "CFG", TPT) is None
    monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda: {right: race})
    assert ifvg_lab_cache.default_firm_race("store", "r" * 64, "CFG", TPT) is race
    # another clock gives another digest
    assert fr.terms_digest({"a": 1}, {"basis": "two_business_days"}) != fr.terms_digest(
        {"a": 1}, {"basis": "elapsed_48_hours"})


# ── A5: descriptive percentiles, never luck ───────────────────────────────


def test_a5_risk_module_has_no_luck_classification_and_no_reordering_words():
    """A5: no "lucky"/"unlucky"/"middle of the pile", no ``pile_side``, no reordering claims."""

    source = Path(risk.__file__).read_text(encoding="utf-8")
    assert "lucky" not in source.lower() and "middle of the pile" not in source
    assert "pile_side" not in source and not hasattr(risk, "pile_side")
    for words in ("reordered runs", "reorderings", "other orders of the same",
                  "Shuffle every trade", "0.35 <=", "<= 0.65", "> 0.65"):
        assert words not in source, words
    assert rs.METHODS == {"blocks": "Keep streaks together", "shuffle": "Draw single trades"}


def _race(values, *, slots=114, method="blocks") -> fr.FullRace:
    return fr.FullRace(
        firm_key=TPT, firm_name="TakeProfitTrader", paths=len(values), seed=fm.DEFAULT_SEED,
        method=method, slots=slots, paid_share=0.58, died_share=0.42, still_going_share=0.0,
        typical_to_payout=8.0, typical_to_limit=10.0, payouts_before_death=1.84,
        lost_accounts=0, cash_per_account=5_328.0, accounts_bought=0, net_cash_bad=0.0,
        net_cash_typical=0.0, net_cash_good=0.0, net_cash_values=tuple(values),
        horizon_first_day="2026-01-13", horizon_last_day="2026-06-10")


def test_a5_both_percentile_sentences_count_ties_half_and_state_the_model():
    """A5: one ties-count-half rule for both sentences, with replacement and the horizon."""

    values, actual = A5_TIES["inputs"]["values"], A5_TIES["inputs"]["actual"]
    assert rs.share_below_ties_half(values, actual) == A5_TIES["expected"]["share_below"]
    fan = rs.EquityFan(method="blocks", paths=5, seed=1, trades=114, end_values=tuple(values))
    assert risk.end_caption(fan, actual) == (
        "In trading profit, the recorded result (+$300) is above 60% of the 5 resampled "
        "paths' ends (ties count half). Each path draws the recorded trades with replacement "
        "(blocks of 10), so path totals vary; a pure reordering of the same trades would "
        "always end at the recorded total. The percentile describes where the recorded result "
        "sits under this sampling model; it does not measure luck.")
    one_at_a_time = rs.EquityFan(method="shuffle", paths=5, seed=1, trades=114,
                                 end_values=tuple(values))
    assert "(one at a time)" in risk.end_caption(one_at_a_time, actual)
    race = _race(values)
    assert race.share_below(actual) == 0.6
    assert risk.firm_standing(race, 30_000) == (
        "Under this conditional model, the recorded net cash ($300.00) is above 60% of the 5 "
        "resampled paths' net cash (ties count half). Horizon: the study's 114 trade slots, "
        "January 13 – June 10, 2026; paths draw recorded trades with replacement in blocks of "
        "10. The percentile describes where the recorded result sits; it does not measure "
        "luck.")


def test_a5_verdict_shows_not_supported_when_both_edge_checks_fail():
    """A5: both edge checks failing is "Not supported" (orange), never hidden."""

    common = dict(integrity_ok=True, integrity_text="", edge_text="e", worst_drawdown=None,
                  loss_limit=None, lost_before_payout=0, account_text="", trading_days=300,
                  has_unseen_result=True, sample_text="")
    edge = fm.verdict(range95_low=-1.0, deflated=0.1, **common)[1]
    assert (edge.status, edge.tone) == ("Not supported", "orange")
    missing = fm.verdict(range95_low=None, deflated=None, **common)[1]
    assert missing.status == "Not available" and missing.status is not None


# ── A9: first halves and remainders of the deadline trades ────────────────


def test_a9_legs_reconcile_with_an_odd_quantity_and_uneven_entry_cost():
    """A9: the entry fill is allocated by quantity in exact tenths of a cent; legs reconcile.

    Quantity 5 split 2 + 3 at $1.03 per contract per fill: the $5.15 entry cost can't
    be halved in whole cents; the half gets 206 cents and the rest 309. A trade
    without a half fill has first = 0 and remainder = the whole trade.
    """

    case = A9_LEGS
    assert fm.split_entry_cost(5, 2, 1030) == tuple(case["expected"]["entry_split_cents"])
    assert fm.split_entry_cost(5, 2, 514) == tuple(case["expected"]["floor_split_5_2_514"])
    assert sum(fm.split_entry_cost(5, 2, 514)) == 5 * 514 // 10
    legs = fm.held_legs_from_rows(case["inputs"]["rows"], tick_value_cents=50, mills=1030,
                                  half_exit=True)
    assert [leg.first_cents for leg in legs.per_trade] == case["expected"]["first_cents"]
    assert [leg.rest_cents for leg in legs.per_trade] == case["expected"]["rest_cents"]
    assert all(leg.first_cents + leg.rest_cents == leg.recorded_net_cents
               for leg in legs.per_trade)
    assert (legs.first_halves_cents, legs.remainders_cents, legs.whole_cents) == (
        1176, 2484, 3660)
    assert legs.reconciled and legs.half_exit and legs.count == 3


def test_a9_unreconciled_legs_fall_back_to_whole_trade_wording():
    """A9: legs that don't reconcile are never attributed; the whole-trade label is used."""

    rows = [dict(r) for r in A9_LEGS["inputs"]["rows"]]
    rows[0]["net_pnl_usd"] = 44.71  # one cent off the recorded ticks and costs
    legs = fm.held_legs_from_rows(rows, tick_value_cents=50, mills=1030, half_exit=True)
    assert not legs.reconciled
    fired = fm.findings(largest_account_share=None, five_largest_share=None,
                        died_first_share=None, total_profit=10.0, beta_r2=None, held=legs)
    (held,) = fired
    assert held.title == "Trades held to the daily deadline carry the profit"
    assert held.text == ("Total profit from 3 trades held to the 3:55 PM deadline: $37, more "
                         "than the configuration's total trading profit ($10).")
    reconciled = fm.held_legs_from_rows(A9_LEGS["inputs"]["rows"], tick_value_cents=50,
                                        mills=1030, half_exit=True)
    # the held halves (+$24.84) carry the $10 profit on their own
    (halves,) = fm.findings(largest_account_share=None, five_largest_share=None,
                            died_first_share=None, total_profit=10.0, beta_r2=None,
                            held=reconciled)
    assert halves.title == "Held halves carry the profit"
    # the held halves don't carry a $30 profit: no attribution to the remainders
    assert fm.findings(largest_account_share=None, five_largest_share=None,
                       died_first_share=None, total_profit=30.0, beta_r2=None,
                       held=reconciled) == []


@pytest.mark.skipif(not SAVED, reason="the saved funded variation study is not on this computer")
def test_a9_frozen_reference_on_the_saved_leader():
    """A9: the leader's 13 deadline trades: $4,490.68 + $40,278.18 = $44,768.86 (reference)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study

    study = open_funded_study(STORE, RESULT_ID)
    legs = fm.held_to_deadline_legs(study, LEADER, TPT)
    expected = A9_FROZEN["expected"]
    assert (legs.count, legs.first_halves_cents, legs.remainders_cents, legs.whole_cents) == (
        expected["count"], expected["first_halves_cents"], expected["remainders_cents"],
        expected["whole_cents"])
    assert legs.reconciled and legs.half_exit
    assert fm.held_to_close(study, LEADER, TPT) == (13, 44768.86)  # unchanged stored figure
    (finding,) = fm.findings(largest_account_share=None, five_largest_share=None,
                             died_first_share=None, total_profit=37730.58, beta_r2=None,
                             held=legs)
    assert finding.title == "Held halves carry the profit"
    assert finding.text == (
        "The held halves of 13 trades kept to the daily deadline made $40,278, more than the "
        "configuration's total trading profit ($37,731). Their first halves, closed at the "
        "target, made $4,491 (whole trades $44,769).")
    assert finding.next_step == "test a capped hold on the second half."


# ── the saved terms the tests use ─────────────────────────────────────────


@pytest.mark.skipif(not (EXPORT / "firm_rules.json").is_file(),
                    reason="the export v4 review folder is not on this computer")
def test_the_built_in_profiles_equal_the_saved_export_terms():
    """A2/A3: the firm terms used above equal the saved TakeProfitTrader/MyFundedFutures terms."""

    from alpha_lab.propsim.funded.profiles import FundedFirmProfile

    saved = json.loads((EXPORT / "firm_rules.json").read_text(encoding="utf-8"))
    for profile in saved["firm_profiles"]:
        assert FundedFirmProfile.model_validate(profile) == FIRM_PROFILES[profile["firm_key"]]


def test_the_owner_facing_ledger_texts_are_exact():
    """A3/A4: heading, limitations, button, placeholder and approximation texts."""

    assert risk.ledger_heading("TakeProfitTrader") == (
        "Conditional resampling of recorded trades with TakeProfitTrader's ledger rules")
    assert risk.ledger_heading("MyFundedFutures") == (
        "Conditional resampling of recorded trades with MyFundedFutures' ledger rules")
    assert risk.run_label("TakeProfitTrader") == (
        "Run conditional resampling with TakeProfitTrader's rules")
    assert risk.not_run_text("TakeProfitTrader") == (
        "Not run yet · choose Run conditional resampling with TakeProfitTrader's rules")
    assert risk.PAYOUTS_LABEL == (
        "Average payouts among accounts that failed within the tested horizon")
    assert risk.CASH_LABEL == "Pooled net cash per purchased account"
    assert risk.limitations_line().startswith("Conditional on the recorded trades, not an exact "
                                              "fresh-account model at either firm")
    tpt = SimpleNamespace(firm_name="TakeProfitTrader", profile=FIRM_PROFILES[TPT])
    mff = SimpleNamespace(firm_name="MyFundedFutures", profile=FIRM_PROFILES[MFF])
    assert "fails once the floor has locked at $0" in risk.approximation_text(tpt)
    assert risk.approximation_text(mff).endswith("so these results are conditional too, not "
                                                 "exact.")
    table = html.unescape(str(risk.full_comparison(
        _race([1.0] * 10), None, 3_078_188,
        SimpleNamespace(firm_name="TakeProfitTrader", profile=FIRM_PROFILES[TPT],
                        processing=TWO_BUSINESS_DAYS_FED_1600))))
    for row in risk.COMPARISON_ROWS:
        assert row in table
    assert "after the saved two-business-day processing clock" in table


# ── A2: the drawdown and boundary charts say what they measure ────────────


def test_a2_drawdown_and_boundary_wording_is_exact():
    """A2: a sampled closed-profit drawdown and fixed boundaries, never account failure."""

    g = rs.DrawdownGrowth(method="blocks", paths=20_000, seed=1, limit=2000.0,
                          typical=(0.0,) * 101, worse_1_in_4=(0.0,) * 101,
                          worse_1_in_20=(0.0,) * 101,
                          touched_share=tuple([0.0] * 100 + [0.94]))
    sentence = risk.growth_sentence(g, "TakeProfitTrader", risk.lock_text(
        FIRM_PROFILES[TPT].model_dump()))
    assert sentence == (
        "By 100 resampled trades, 94% of paths have had their cumulative closed profit fall at "
        "least $2,000 below a previous high. This counts closed trade results with no "
        "withdrawals and no account floor, so it is not the share of funded accounts that "
        "fail: TakeProfitTrader's floor can lock (at $0), so a $2,000 fall from a higher peak "
        "need not end an account, and a loss inside an open trade can end an account even when "
        "the closed result recovers.")
    assert "locks the gains in" not in sentence and "withdrawing early" not in sentence
    assert risk.lock_text(FIRM_PROFILES[MFF].model_dump()) == "at +$100"
    race = rs.PayoutRace(paths=20_000, seed=1, method="blocks", loss_limit=-2000.0,
                         trigger=2600.0, max_trades=200, paid_share=0.78, died_share=0.22,
                         still_going_share=0.0, typical_to_payout=8.0, typical_to_limit=10.0)
    assert risk.race_caption(race) == (
        "Of 20,000 resampled paths of up to 200 trades (blocks of 10 recorded trades drawn "
        "with replacement), 78% reached cumulative closed profit of +$2,600 before −$2,000 and "
        "22% fell to −$2,000 first. This diagnostic counts closed trade results against fixed "
        "boundaries only: it leaves out the firm's trailing and locking floor, losses inside "
        "open trades, payout requests, processing and receipts, so it is not a payout or "
        "account-failure model.")
    table = html.unescape(str(risk.growth_table(rs.drawdown_growth([100.0, -50.0] * 30,
                                                                   paths=200, seed=1))))
    for row in ("After…", "Largest fall, typical · 1 in 20",
                "Closed profit, 5th · median · 95th percentile",
                "Fell $2,000 from a previous high"):
        assert row in table
    left, right = risk.growth_figures(g)
    assert any(a.text == "<b>$2,000 fall</b>" for a in left.layout.annotations)
    assert "have fallen $2,000 from a previous high" in right.data[0].hovertemplate
    assert risk.fan_title(20_000, 114) == "20,000 resampled paths of 114 trades"
    tpt_terms = FIRM_PROFILES[TPT].model_dump()
    assert risk.lower_help("TakeProfitTrader", tpt_terms) == (
        "A resampled path stops when its cumulative closed profit falls to this amount. The "
        "default is TakeProfitTrader's saved $2,000 loss allowance, used here as a fixed "
        "closed-profit boundary, not as the firm's floor.")
    assert risk.upper_help("TakeProfitTrader", tpt_terms) == (
        "A resampled path stops when its cumulative closed profit reaches this amount. The "
        "default is TakeProfitTrader's saved $2,100 cushion plus its $500 minimum request; "
        "reaching it here is not a payout.")


# ── A5/A11: the Summary's and the overview's wording and shared findings ──


def test_a5_summary_sharpe_tiles_note_and_verdict_wording():
    """A5: probabilistic and deflated Sharpe ratios explained by their series and adjustment."""

    import ifvg_lab_detail_summary as summary

    sharpe = fm.SharpeConfidence(daily_sharpe=0.25, skew=3.1602, kurtosis=13.8101, days=107,
                                 above_zero=0.9995, tested=64, benchmark_annualized=3.1499,
                                 deflated=0.8104)
    assert summary.sharpe_note(sharpe) == (
        "How the Sharpe scores are measured: both use this configuration's 107 daily funded "
        "results (days without a trade count as $0), treat the days as independent draws with "
        # review R1: a confidence score comparing the observed ratio, not an event's chance
        "the observed skew (3.16) and kurtosis (13.81), and report a normal-approximation "
        "confidence score (Φ of a z-score, like one minus a one-sided p-value) comparing the "
        "observed daily Sharpe ratio with a benchmark — 0 for the probabilistic "
        "Sharpe ratio, and the expected best of the 64 configurations compared in this study at "
        "this firm (3.15 annualized) for the deflated one. Neither is a probability that the "
        "edge is genuine or that payouts will continue, and the 64-configuration adjustment "
        "doesn't account for research done before this study.")
    ctx = SimpleNamespace(row=SimpleNamespace(sharpe=4.01, sortino=18.82),
                          study=SimpleNamespace(strategy_measures=lambda _c: {}),
                          configuration="CFG")
    bundle = SimpleNamespace(sharpe=sharpe, funded_profit_factor=2.7, hold=None,
                             tie=SimpleNamespace(beta=0.067, r_squared=0.112, days=106))
    top, bottom = summary._measure_tiles(ctx, bundle)
    tiles = html.unescape(str(top) + str(bottom))
    assert "Probabilistic Sharpe ratio" in tiles and "&gt; 0.99" not in tiles
    assert "> 0.99" in tiles and "Deflated for 64 configurations compared: 0.81" in tiles
    assert "R² 0.11: a straight-line fit explains 11% of daily variation" in tiles
    assert "Chance the true Sharpe" not in tiles
    below = SimpleNamespace(**{**bundle.__dict__, "sharpe": replace(sharpe, above_zero=0.9712)})
    assert "0.97" in html.unescape(str(summary._measure_tiles(ctx, below)[0]))
    study = SimpleNamespace(result={"price_evidence": {"position_minutes_checked": 10,
                                                       "position_minutes_rebuilt_exactly_from"
                                                       "_prints": 10},
                                    "validation": {"passed": True}})
    vctx = SimpleNamespace(study=study, row=SimpleNamespace(worst_drawdown=3661.0,
                                                            lost_before_payout=4, trades=114))
    vbundle = SimpleNamespace(trades_approximated=0, ranges=None, sharpe=None, loss_limit=2000.0,
                              lost_first_accounts=4, trading_days=107, strategy_trades=154)
    account = summary._verdict_parts(vctx, vbundle)[2].text
    assert account == ("Worst closed-profit drawdown of the trade path across accounts $3,661 "
                       "(each account's loss allowance: $2,000). The first 4 accounts were lost "
                       "before any payout.")
    source = Path(summary.__file__).read_text(encoding="utf-8")
    assert "Chance the true Sharpe is above zero\"" not in source


def test_a11_overview_and_summary_share_one_findings_helper(monkeypatch):
    """A11: the overview's finding count uses exactly what the Summary passes."""

    import ifvg_lab_detail_summary as summary
    import ifvg_lab_funded

    monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda: {})
    legs = fm.held_legs_from_rows(A9_LEGS["inputs"]["rows"], tick_value_cents=50, mills=1030,
                                  half_exit=True)
    bundle = SimpleNamespace(
        concentration=SimpleNamespace(largest_account_share=0.92,
                                      five_largest_trades_share=0.74),
        race=SimpleNamespace(died_share=0.22, loss_limit=-2000.0, trigger=2600.0, paths=20_000),
        tie=SimpleNamespace(r_squared=0.11, beta=0.07, days=106), held_legs=legs,
        held_count=3, held_net=36.60, total_profit=10.0, beta_days=106)
    target = {"store_root": "store", "result_id": "r" * 64}
    overview = ifvg_lab_funded._findings(bundle, target, "CFG", TPT, "TakeProfitTrader")
    shared = ifvg_lab_cache.pair_findings(bundle, "store", "r" * 64, "CFG", TPT,
                                          "TakeProfitTrader")
    assert overview == shared and len(shared) == 4
    ctx = SimpleNamespace(store_root="store", result_id="r" * 64, configuration="CFG",
                          firm_key=TPT, firm="TakeProfitTrader")
    assert "Findings · 4" in html.unescape(str(summary._findings_card(ctx, bundle)))
    source = Path(ifvg_lab_funded.__file__).read_text(encoding="utf-8")
    assert "chance the leader's edge" not in source  # correction A5


# ── the Summary tab and the overview on the saved study (read only) ──────


def _render(scripts: str, screen: str):
    import sys as _sys

    _sys.path.insert(0, scripts)
    import streamlit as st
    from ifvg_lab_funded import render_funded_detail, render_funded_results

    (render_funded_detail if screen == "detail" else render_funded_results)(st, {})


def _saved_app(screen: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_render, default_timeout=300,
                               kwargs={"scripts": str(REPO / "scripts"), "screen": screen})
    at.session_state["ifvg_lab_v1_funded_target"] = {
        "result_id": RESULT_ID, "store_root": str(STORE), "app": "ifsm",
        "study_key": RESULT_ID, "name": "Funded variation study", "status": "Completed"}
    at.session_state["funded_comparison_v1_selected_context"] = {
        RESULT_ID: {"firm_key": TPT, "configuration": LEADER, "tab": "Summary"}}
    return at


def _page(at) -> str:
    return html.unescape("\n".join(str(getattr(e.proto, "body", "")) for e in at.get("html")))


@pytest.mark.skipif(not SAVED, reason="the saved funded variation study is not on this computer")
def test_saved_summary_and_overview_show_the_corrected_wording(monkeypatch):
    """A2/A3/A5/A9/A11: the saved leader's Summary and the overview, before and after a run.

    Before a conditional run both screens quote the fixed-boundary diagnostic; once
    the default-draw result is in the process store (the Risk tab's key, rebuilt
    from the saved result) both quote the conditional model. Nothing is run here.
    """

    import ifvg_lab_ui

    store: dict = {}
    monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda: store)
    detail = _saved_app("detail").run()
    assert not detail.exception, detail.exception
    page = _page(detail)
    assert "Probabilistic Sharpe ratio" in page and "Chance the true Sharpe" not in page
    assert "Deflated for 64 configurations compared: 0.81" in page
    assert "How the Sharpe scores are measured: both use this configuration's 107 daily" in page
    assert ("Worst closed-profit drawdown of the trade path across accounts $3,661 (each "
            "account's loss allowance: $2,000).") in page
    assert "Early losses in the fixed-boundary diagnostic" in page
    assert "The held halves of 13 trades kept to the daily deadline made $40,278" in page
    count = int(page.split("Findings · ", 1)[1][0])
    overview = _saved_app("overview").run()
    assert not overview.exception, overview.exception
    first = _page(overview)
    assert f"{count} findings" in first or (count == 1 and "1 finding" in first)
    assert ("Deflated Sharpe ratio after comparing 64 configurations in this study at this "
            "firm: the probabilistic Sharpe ratio against the expected best of 64 (daily "
            "results, independence assumed). Not a probability of a genuine edge or of future "
            "payouts, and it doesn't account for research done before this study.") in first
    # a conditional run at the default draw, as the Risk tab stores it
    study = ifvg_lab_ui.funded_study(str(STORE), RESULT_ID)
    slots, cutoff_ns, digest = ifvg_lab_cache.saved_race_binding(study, LEADER, TPT)
    key = ifvg_lab_cache.firm_race_key(str(STORE), RESULT_ID, LEADER, TPT, method="blocks",
                                       seed=fm.DEFAULT_SEED, paths=fr.DEFAULT_FULL_PATHS,
                                       slots=slots, cutoff_ns=cutoff_ns, terms_digest=digest)
    store[key] = replace(_race([1.0] * 1_000), paths=1_000)
    after = _page(_saved_app("detail").run())
    assert "Early account failures (conditional model)" in after
    assert "(1,000 paths, to June 10, 2026), 42% of first accounts failed" in after
    assert int(after.split("Findings · ", 1)[1][0]) == count
    assert f"{count} finding" in _page(_saved_app("overview").run())


# ── review R1 follow-ups ──────────────────────────────────────────────────


def test_a11_an_unavailable_edge_check_is_never_counted_as_a_failure():
    """A11 (review R1): each edge check passes, fails or is unavailable.

    One passing check with the other unavailable is "Partly checked" (neutral), not
    "Weak"; one failing check with the other unavailable is "Weak", never "Not
    supported" (which says both checks failed).
    """

    common = dict(integrity_ok=True, integrity_text="", edge_text="e", worst_drawdown=None,
                  loss_limit=None, lost_before_payout=0, account_text="", trading_days=300,
                  has_unseen_result=True, sample_text="")

    def edge(low, deflated):
        part = fm.verdict(range95_low=low, deflated=deflated, **common)[1]
        return part.status, part.tone

    assert edge(150.0, None) == ("Partly checked", "neutral")
    assert edge(None, 0.8) == ("Partly checked", "neutral")
    assert edge(-5.0, None) == ("Weak", "orange")
    assert edge(None, 0.2) == ("Weak", "orange")
    assert edge(150.0, 0.8) == ("Holds", "blue")
    assert edge(150.0, 0.2) == ("Weak", "orange")
    assert edge(-5.0, 0.2) == ("Not supported", "orange")
    assert edge(None, None) == ("Not available", "neutral")


def test_a3_risk_example_quotes_both_marked_paths_and_says_miss_or_create():
    """A3 (review R1): the on-screen counterexample is the tested A/B pair, quoted in full."""

    tpt = SimpleNamespace(firm_name="TakeProfitTrader", profile=FIRM_PROFILES[TPT])
    assert risk.approximation_text(tpt) == (
        "For TakeProfitTrader, whose floor trails the highest equity inside a trade, the "
        "compressed path can miss or create an account loss: a trade marked $0 → −$1,000 → "
        "+$2,500 → −$100 → +$4,000 → +$3,000 fails once the floor has locked at $0, and one "
        "marked $0 → −$1,000 → +$2,500 → +$2,000 → +$4,000 → +$3,000 survives, yet both "
        "compress to the same four points ($0, −$1,000, +$4,000, +$3,000).")
    a, b = A3_COUNTEREXAMPLE["inputs"]["A_marked_path_usd"], \
        A3_COUNTEREXAMPLE["inputs"]["B_marked_path_usd"]
    assert compress(a) == compress(b) == (0, -1000, 4000, 3000)


def test_a3_ledger_key_uses_the_saved_cutoff_and_refuses_a_mismatch():
    """A3 (review R1): the Risk tab keys its run by the saved result's cutoff and slots, the
    same inputs the Summary rebuilds; a run whose inputs differ is never cached or quoted."""

    rules, slots = _rules(TPT, _five_days())
    ctx = SimpleNamespace(study=_fake_study(rules, slots), store_root="store",
                          result_id="r" * 64, configuration="CFG", firm_key=TPT)
    key = risk.ledger_key(ctx, rules, slots, seed=9, paths=20)
    slots_n, cutoff_ns, digest = ifvg_lab_cache.saved_race_binding(ctx.study, "CFG", TPT)
    assert key == ifvg_lab_cache.firm_race_key(
        "store", "r" * 64, "CFG", TPT, method="blocks", seed=9, paths=20, slots=slots_n,
        cutoff_ns=cutoff_ns, terms_digest=digest)
    shifted = replace(rules, cutoff_ns=rules.cutoff_ns + 60_000_000_000)
    assert risk.ledger_key(ctx, shifted, slots, seed=9, paths=20) is None
    assert risk.ledger_key(ctx, rules, slots[:-1], seed=9, paths=20) is None


def test_a11_edge_text_names_the_check_that_was_not_made():
    """A11 (review R1): with no deflated Sharpe ratio the Edge text says that check wasn't made."""

    import ifvg_lab_detail_summary as summary

    ranges = fm.BootstrapRange(mean=330.0, count=114, paths=20_000, seed=1,
                               ranges={68: (200.0, 450.0), 90: (130.0, 550.0),
                                       95: (100.0, 600.0)})
    sharpe = fm.SharpeConfidence(daily_sharpe=0.25, skew=3.16, kurtosis=13.8, days=107,
                                 above_zero=0.99, tested=1)
    bundle = SimpleNamespace(ranges=ranges, sharpe=sharpe, loss_limit=2000.0,
                             lost_first_accounts=0, trades_approximated=0, trading_days=107,
                             strategy_trades=None)
    ctx = SimpleNamespace(study=SimpleNamespace(result={}),
                          row=SimpleNamespace(worst_drawdown=1000.0, lost_before_payout=0,
                                              trades=114))
    edge = summary._verdict_parts(ctx, bundle)[1]
    assert (edge.status, edge.tone) == ("Partly checked", "neutral")
    assert edge.text.endswith(
        "The deflated Sharpe ratio isn't available for this comparison (for example fewer than "
        "two completed configurations, or daily results that don't vary), so that check wasn't "
        "made.")
