"""CALCULATIONS.md reference values for the IFVG Lab redesign.

Reference case: funded variation study ``5fa65149843484b1`` (export v4),
configuration ``S1-T1-H14-P1-L-SO`` at TakeProfitTrader, January 13 – June 10,
2026 (107 trading days). "Exact" values match to the cent or the shown
decimals; resampled values use the stated tolerance.

These tests read the saved result, plan and verified strategy package on this
computer (read only). They are skipped where those local records are absent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
TPT, MFF = "takeprofittrader", "myfundedfutures"

pytestmark = pytest.mark.skipif(
    not (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file(),
    reason="the saved funded variation study is not on this computer")


@pytest.fixture(scope="module")
def study():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study

    return open_funded_study(STORE, RESULT_ID)


@pytest.fixture(scope="module")
def minutes(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market

    root = market.study_package_root(study.plan)
    if root is None:
        pytest.skip("the verified strategy package is not on this computer")
    return market.load_index_minutes(root, cutoff_utc=study.result["period"]["cutoff_utc"])


def _within(value, reference, tolerance):
    return abs(value - reference) <= abs(reference) * tolerance


# ── inputs ────────────────────────────────────────────────────────────────


def test_inputs(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        daily_results,
        ordered_trades,
        trade_path,
    )

    assert len(ordered_trades(study, LEADER, TPT)) == 114
    assert len(daily_results(study, LEADER, TPT)) == 107 == len(study.calendar)
    assert study.calendar[0] == "2026-01-13" and study.calendar[-1] == "2026-06-10"
    assert trade_path(study, LEADER, TPT)[0] == 0.0
    assert round(trade_path(study, LEADER, TPT)[-1], 2) == 37730.58


# ── funded results table and headline tiles ───────────────────────────────


def test_leader_row_exact(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import funded_row

    row = funded_row(study, LEADER, TPT)
    assert row.rank == 1
    assert row.net_cash_cents == 3_078_188
    assert row.received_cents == 3_139_388 and row.payouts == 13
    assert row.costs_cents == 61_200 and row.accounts == 6
    assert f"{row.cash_per_dollar:.2f}" == "50.30"
    assert row.largest_payout_cents == 561_066 and row.median_payout_cents == 221_733
    assert round(row.worst_drawdown) == 3661
    assert f"{row.sharpe:.2f}" == "4.01" and f"{row.sortino:.2f}" == "18.82"
    assert row.net_r == 73.45 and row.win_rate_pct == 58.4


@pytest.mark.parametrize(("rank", "sharpe", "sortino", "drawdown"), [
    (2, "3.51", "13.85", 6268), (3, "2.98", "9.79", 7379), (4, "4.00", "16.69", 2659)])
def test_other_leaders_takeprofittrader(study, rank, sharpe, sortino, drawdown):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import ranking

    row = ranking(study, TPT)[rank - 1]
    assert row.rank == rank
    assert (f"{row.sharpe:.2f}", f"{row.sortino:.2f}", round(row.worst_drawdown)) == (
        sharpe, sortino, drawdown)


def test_myfundedfutures_leader(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import ranking

    row = ranking(study, MFF)[0]
    assert row.net_cash_cents == 3_481_811
    assert (f"{row.sharpe:.2f}", f"{row.sortino:.2f}", round(row.worst_drawdown)) == (
        "4.05", "19.84", 3461)


# ── checks on the leader ──────────────────────────────────────────────────


@pytest.mark.parametrize(("firm", "leader", "runner"), [
    (TPT, 2_517_122, 2_171_832), (MFF, 2_850_612, 2_487_065)])
def test_drop_the_largest_payout(study, firm, leader, runner):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (
        drop_largest_payout,
    )

    check = drop_largest_payout(study, firm)
    assert check.passed is True
    assert (check.leader_after_cents, check.runner_up_after_cents) == (leader, runner)


def test_result_per_trade_range(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (
        bootstrap_mean_ranges,
    )

    values = [t["net_pnl_usd"] for t in ordered_trades(study, LEADER, TPT)]
    first = bootstrap_mean_ranges(values)
    assert round(first.mean, 2) == 330.97
    references = {95: (95, 601), 68: (201, 458)}
    for level, (low, high) in references.items():
        assert _within(first.ranges[level][0], low, 0.05), level
        assert _within(first.ranges[level][1], high, 0.05), level
    # 90%: high end within 5%; the low end is recorded in DATA_GAPS/decision log
    # (this draw gives about $134 against the mock's $126)
    assert _within(first.ranges[90][1], 551, 0.05)
    assert 120 <= first.ranges[90][0] <= 140
    assert bootstrap_mean_ranges(values) == first  # fixed seed, identical on reload


def test_sharpe_confidence_and_deflation(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import deflated_sharpe

    result = deflated_sharpe(study, LEADER, TPT)
    assert f"{result.skew:.2f}" == "3.16" and f"{result.kurtosis:.2f}" == "13.81"
    assert result.above_zero > 0.99
    assert result.tested == 64
    assert f"{result.benchmark_annualized:.2f}" == "3.15"
    assert f"{result.deflated:.2f}" == "0.81"


def test_quality_gates(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import quality_gates

    thresholds = {"min_executed_trades": 60, "min_independent_days": 50,
                  "min_net_expectancy_r": 0.0, "min_profit_factor": 1.1,
                  "max_drawdown_r": 15.0, "max_time_under_water_days": 3,
                  "max_top_day_pnl_share": 0.4, "min_session_stability_score": 0.5,
                  "min_time_block_sign_consistency": 0.6, "max_top_setup_pnl_share": 0.25}
    rows = {r.gate: r for r in quality_gates(study, LEADER, thresholds,
                                             funded_days_with_trade=63)}
    assert (rows["Trades"].value, rows["Trades"].passed) == ("154", True)
    assert (rows["Result per trade"].value, rows["Result per trade"].passed) == ("0.48R", True)
    assert (rows["Profit factor"].value, rows["Profit factor"].passed) == ("2.13", True)
    assert (rows["Drawdown"].value, rows["Drawdown"].passed) == ("10.57R", True)
    under = rows["Trading days under water"]
    assert (under.value, under.passed, under.needs_decision) == ("40", False, True)
    # not stored by the strategy replay: shown as "Not in export", never a stand-in
    days = rows["Days with a trade"]
    assert days.value is None and days.note == "63 on funded trades"
    assert rows["Best day's share of profit"].value is None


# ── summary: concentration, verdict, findings, tie to the index ───────────


def test_concentration(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import concentration

    c = concentration(study, LEADER, TPT)
    assert f"{c.largest_account_share * 100:.1f}" == "91.6"
    assert f"{c.five_largest_trades_share * 100:.1f}" == "74.3"
    assert c.best_month == "2026-04" and f"{c.best_month_share * 100:.1f}" == "55.6"
    assert f"{c.largest_payout_share * 100:.1f}" == "17.9"
    assert f"{c.best_day_share * 100:.1f}" == "14.1"


def test_held_trades_and_findings(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (
        concentration,
        findings,
        held_to_close,
        held_to_deadline_legs,
    )

    count, net = held_to_close(study, LEADER, TPT)
    assert count == 13 and round(net) == 44769
    # correction A9: the 13 deadline trades split into first halves and held remainders
    legs = held_to_deadline_legs(study, LEADER, TPT)
    assert (legs.first_halves_cents, legs.remainders_cents, legs.whole_cents) == (
        449_068, 4_027_818, 4_476_886)
    assert legs.reconciled and legs.half_exit and legs.count == 13
    c = concentration(study, LEADER, TPT)
    fired = findings(largest_account_share=c.largest_account_share,
                     five_largest_share=c.five_largest_trades_share, died_first_share=0.216,
                     race_basis="flat", race_detail="20,000 resampled paths",
                     boundaries=(-2000.0, 2600.0), firm="TakeProfitTrader", held=legs,
                     total_profit=37730.58, beta_r2=0.11, beta_days=106)
    # corrections A2 (fixed-boundary diagnostic), A9 (held halves), A5 (association only)
    assert [f.title for f in fired] == ["Concentrated result",
                                       "Early losses in the fixed-boundary diagnostic",
                                       "Held halves carry the profit",
                                       "Low daily linear association with the index"]
    assert "92%" in fired[0].text and "74%" in fired[0].text
    assert "$40,278" in fired[2].text and "(whole trades $44,769)" in fired[2].text
    assert fired[3].text == (
        "A straight-line fit of daily results on the daily E-mini change explains about 11% of "
        "their day-to-day variation (R² 0.11, 106 days). This is an observation about daily "
        "co-movement; it doesn't show that the result was independent of market direction.")


def test_verdict_rules():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import verdict

    parts = verdict(integrity_ok=True, integrity_text="", range95_low=95.0, deflated=0.81,
                    edge_text="", worst_drawdown=3661, loss_limit=2000, lost_before_payout=4,
                    account_text="", trading_days=107, has_unseen_result=False, sample_text="")
    assert [(p.part, p.status) for p in parts] == [
        ("Data integrity", "Pass"), ("Edge", "Holds"), ("Account risk", "Watch"),
        ("Sample", "Limited")]
    weak = verdict(integrity_ok=False, integrity_text="", range95_low=-1.0, deflated=0.9,
                   edge_text="", worst_drawdown=100, loss_limit=2000, lost_before_payout=0,
                   account_text="", trading_days=300, has_unseen_result=True, sample_text="")
    assert [p.status for p in weak] == ["Fail", "Weak", "OK", "Adequate"]
    hidden = verdict(integrity_ok=True, integrity_text="", range95_low=-1.0, deflated=0.1,
                     edge_text="", worst_drawdown=None, loss_limit=None, lost_before_payout=0,
                     account_text="", trading_days=300, has_unseen_result=True, sample_text="")
    # correction A5: the weakest assessment is shown, never hidden
    assert (hidden[1].status, hidden[1].tone) == ("Not supported", "orange")


def test_index_tie_and_buy_and_hold(study, minutes):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.format import chicago_long
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import daily_results

    tie = market.index_tie(daily_results(study, LEADER, TPT), market.daily_closes(minutes))
    assert f"{tie.beta:.3f}" == "0.067" and f"{tie.r_squared:.3f}" == "0.112"
    hold = market.buy_and_hold(minutes, study.calendar)
    # Correction A10 (September 25, 2026): profit and the first $2,000 fall now share one
    # entry instant, the first trading day's 5:00 PM open. The superseded reference
    # (+$51,475 close to close, fall at 9:03 AM measured from a different start) mixed
    # two entries; the old close-to-close figure is still reproducible from the closes.
    assert hold.version == "buy_and_hold_one_emini_first_open_v2"
    assert chicago_long(hold.entry_utc) == "January 12, 2026, 5:00 PM"
    assert chicago_long(hold.exit_utc) == "June 10, 2026, 4:00 PM"
    assert (hold.entry_price, hold.exit_price) == (25953.5, 28472.0)
    assert hold.result == 50370.0
    # the first one-minute close 100 points below the running high (the entry price
    # counts as the first high), stamped at the bar's close
    assert chicago_long(hold.breach_utc) == "January 12, 2026, 7:41 PM"
    assert hold.breach_utc > hold.entry_utc
    closes = market.daily_closes(minutes)
    assert (closes[study.calendar[-1]] - closes[study.calendar[0]]) * 20 == 51475.0


# ── trades tab ────────────────────────────────────────────────────────────


def test_trades_tab(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import trade_stats
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    trades = ordered_trades(study, LEADER, TPT)
    counts = [c for _, c, _, _ in trade_stats.result_distribution(trades)]
    assert counts == [5, 17, 15, 10, 32, 15, 6, 1, 3, 6, 4]
    col = trade_stats.performance_column(trades)
    assert round(col.gross_profit, 2) == 59978.74 and round(col.gross_loss, 2) == -22248.16
    assert round(col.average_win, 2) == 895.21 and round(col.average_loss, 2) == -473.37
    assert round(col.largest_win, 2) == 7349.72 and round(col.largest_loss, 2) == -950.28
    assert (col.longest_winning, col.longest_losing) == (6, 6)
    dd = col.drawdowns
    assert dd.count == 12 and f"{dd.average_length:.1f}" == "7.8"
    assert round(dd.average_depth) == 1207 and dd.longest_length == 45
    assert trade_stats.performance_column(trades, "short").trades == 0
    losers_calm, losers, winners_calm, winners = trade_stats.excursion_counts(
        trade_stats.excursions(trades))
    assert (losers, winners, winners_calm) == (47, 67, 46)
    # CALCULATIONS.md's definition gives 36 (the mock shows 35: one loser peaked
    # at $249.86 after its $5.14 entry cost; decision log)
    assert losers_calm == 36


# ── risk and simulation ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def leader_values(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    return [t["net_pnl_usd"] for t in ordered_trades(study, LEADER, TPT)]


def test_payout_race_fixed_floor(leader_values):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import DEFAULT_SEED
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.resampling import payout_race

    race = payout_race(leader_values, seed=DEFAULT_SEED)
    assert abs(race.paid_share - 0.78) <= 0.02 and abs(race.died_share - 0.22) <= 0.02
    assert abs(race.typical_to_payout - 8) <= 1 and abs(race.typical_to_limit - 10) <= 1
    assert payout_race(leader_values, seed=DEFAULT_SEED) == race


@pytest.mark.parametrize(("method", "bad", "typical", "good", "dd"), [
    ("blocks", 14245, 38668, 66189, 6965), ("shuffle", 15058, 36958, 62916, 6854)])
def test_resampled_equity(leader_values, method, bad, typical, good, dd):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import DEFAULT_SEED
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.resampling import equity_fan

    fan = equity_fan(leader_values, method=method, seed=DEFAULT_SEED)
    assert _within(fan.bad_end, bad, 0.05) and _within(fan.typical_end, typical, 0.05)
    assert _within(fan.good_end, good, 0.05) and _within(fan.bad_case_drawdown, dd, 0.05)
    # "finished below $0" is about 0.2% (0.3% with streaks kept together here)
    assert fan.below_zero_share <= 0.005
    if method == "blocks":
        assert (fan.streak_typical, fan.streak_bad) == (6, 8)


def test_drawdown_growth(leader_values):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import DEFAULT_SEED
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.resampling import drawdown_growth

    growth = drawdown_growth(leader_values, seed=DEFAULT_SEED)
    for trades, typical, touched in ((10, 1111, 0.14), (20, 1547, 0.37), (40, 2487, 0.65),
                                     (100, 3474, 0.94)):
        assert _within(growth.typical[trades], typical, 0.05), trades
        assert abs(growth.touched_share[trades] - touched) <= 0.02, trades


# ── market conditions ─────────────────────────────────────────────────────


def test_market_conditions(study, minutes):
    from collections import Counter

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        daily_results,
        ordered_trades,
    )

    labels = market.condition_labels(market.daily_closes(minutes), study.calendar)
    assert labels.first_labeled == "2026-01-16"
    assert Counter(labels.labels.values()) == {
        "Rising · quiet": 39, "Falling · volatile": 34, "Rising · volatile": 18,
        "Falling · quiet": 13, "Not enough history": 3}
    trades = ordered_trades(study, LEADER, TPT)
    cards = {c.label: (c.trades, round(c.net)) for c in market.condition_cards(labels, trades)}
    assert cards == {"Rising · quiet": (41, 13546), "Rising · volatile": (14, 15844),
                     "Falling · quiet": (15, 56), "Falling · volatile": (39, 8949),
                     "Not enough history": (5, -664)}
    exit_cards = {c.label: (c.trades, round(c.net))
                  for c in market.condition_cards(labels, trades, by="exit")}
    assert exit_cards == cards  # every trade closes inside its entry trading day here
    falling = [s for s in market.stretches(labels, daily_results(study, LEADER, TPT))
               if s.label.startswith("Falling")]
    longest = max(falling, key=lambda s: s.days)
    assert (longest.first_day, longest.last_day, longest.days, round(longest.net)) == (
        "2026-03-18", "2026-04-07", 15, 10536)
    table = market.transition_table(labels)
    assert table["Rising · quiet"] == {"Rising · quiet": 32, "Rising · volatile": 3,
                                       "Falling · quiet": 2, "Falling · volatile": 2}
    measure = market.entry_measure(minutes, trades, measure="volatility")
    assert len(measure.x) == 114 and f"{measure.correlation:.2f}" == "0.04"
    assert measure.clear_link is False


# ── pure formatting ───────────────────────────────────────────────────────


def test_formats():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as f

    assert f.money_cents(3_078_188) == "$30,781.88"
    assert f.money(-215.28) == "−$215.28" and f.money(6254.72, signed=True) == "+$6,254.72"
    assert f.money_short(30781.88) == "$30.8k" and f.money_short(-2190) == "−$2.2k"
    assert f.chicago_long("2026-04-13T00:07:00Z") == "April 12, 2026, 7:07 PM"
    assert f.chicago_short("2026-04-13T00:07:00Z") == "Apr 12, 7:07 PM"
    assert f.date_range("2026-01-13", "2026-06-10") == "January 13 – June 10, 2026"
    assert f.points(99884, from_ticks=True) == "24,971.00"
    assert f.percent(0.584, decimals=1) == "58.4%"
    assert math.isclose(float(f.ticks_to_points(4)), 1.0)
