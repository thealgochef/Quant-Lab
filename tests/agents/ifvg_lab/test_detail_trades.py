"""Configuration detail · Trades tab (mock 06).

Pure builders are checked on hand-made trade rows and, where the saved funded
variation study is on this computer, on the reference case (configuration
``S1-T1-H14-P1-L-SO`` at TakeProfitTrader, 114 funded trades; read only).
The click on a trade row is checked through the tab's own action handler and
through a headless run of the detail screen: it must set the Trade review
target to that exact trade and move the workspace to Trade review. Nothing
here saves, approves or launches anything.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import ifvg_lab_detail_trades as trades_tab  # noqa: E402

STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
TPT = "takeprofittrader"
HAVE_STUDY = (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file()
needs_study = pytest.mark.skipif(not HAVE_STUDY,
                                 reason="the saved funded variation study is not on this computer")


def _text(markup) -> str:
    """Visible text of built HTML (tags removed, whitespace collapsed)."""

    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(markup))).strip()


def _trade(seq, net, *, account=1, direction="long", exit_kind="stop", low=None, high=None,
           scale=None, entry="2026-01-13T04:31:00Z", exit_="2026-01-13T04:41:57.695030175Z"):
    before = 50_000.0
    return {
        "seq": seq, "account_number": account, "direction": direction, "quantity": 10,
        "entry_ticks": 103_535, "stop_ticks": 103_499, "exit_ticks": 103_499,
        "scale_out_ticks": scale, "scale_out_quantity": 5 if scale else 0,
        "final_exit_quantity": 5 if scale else 10, "exit_kind": exit_kind,
        "net_pnl_usd": net, "balance_before_usd": before,
        "min_equity_usd": before - (low if low is not None else 10.0),
        "max_equity_usd": before + (high if high is not None else 10.0),
        "entry_utc": entry, "exit_utc": exit_, "trading_day": "2026-01-13",
    }


# ── pure builders on hand-made rows ───────────────────────────────────────


def test_distribution_card_counts_colors_and_caption():
    rows = ([_trade(i, -100.0) for i in range(3)] + [_trade(10 + i, 300.0) for i in range(4)]
            + [_trade(20, 2500.0), _trade(21, 5000.0)])
    card = str(trades_tab.distribution_card(rows))
    text = _text(card)
    assert text.startswith("Trade results, 9 funded trades")
    for label in ("under −$750", "−$750", "−$500", "−$250", "$0", "$250", "$500", "$750",
                  "$1k–2k", "$2k–4k", "over $4k"):
        assert label in text
    # losing bins are orange, winning bins blue (theme variables, never a literal); the
    # tallest bar is 160 px
    assert (card.count("background:var(--lab-orange)") == 4
            and card.count("background:var(--lab-blue)") == 7)
    assert "height:160px" in card
    assert "4 trades from $250 up to $500" in card and "1 trade made $4,000 or more" in card
    assert trades_tab.distribution_caption(rows) == (
        "Losses stay small and bunched: the worst of 3 losers is −$100. Two trades above "
        "$2,000 make the long right tail.")


def test_distribution_caption_other_shapes():
    assert trades_tab.distribution_caption([]) == "No funded trades at this firm."
    assert trades_tab.distribution_caption([_trade(1, 50.0)]) == (
        "No trade lost money. No trade made more than $2,000; the best made $50.")
    large = [_trade(1, -1500.0), _trade(2, 900.0)]
    assert trades_tab.distribution_caption(large) == (
        "Losses run large: the worst of 1 loser is −$1,500, against a best trade of $900. "
        "No trade made more than $2,000; the best made $900.")
    assert trades_tab.distribution_caption([_trade(1, -1500.0), _trade(2, 4000.0)]).startswith(
        "Losses stay small next to the wins: the worst of 1 loser is −$1,500.")


def test_performance_card_long_only_leaves_short_empty():
    rows = [_trade(1, 400.0), _trade(2, -200.0), _trade(3, 100.0)]
    card = trades_tab.performance_card(rows, [("Direction", "Long only")], "TakeProfitTrader")
    text = _text(card)
    assert "Trades · win rate 3 · 66.7% 3 · 66.7% —" in text
    assert "Gross profit $500.00 $500.00 —" in text
    assert "Largest win · largest loss $400 · −$200 $400 · −$200 —" in text
    assert "Short column stays empty on long-only configurations." in text


def test_performance_card_both_directions_fills_short():
    rows = [_trade(1, 400.0), _trade(2, -200.0, direction="short"),
            _trade(3, 300.0, direction="short")]
    text = _text(trades_tab.performance_card(rows, [("Direction", "Long and short")], "X"))
    assert "Trades · win rate 3 · 66.7% 1 · 100.0% 2 · 50.0%" in text
    assert "stays empty" not in text
    # a direction with no trades at this firm says so instead of showing zeros
    only_long = trades_tab.performance_card([_trade(1, 400.0)],
                                            [("Direction", "Long and short")], "MyFundedFutures")
    assert "No short trades were taken at MyFundedFutures." in _text(only_long)
    # no losing trade: the largest loss is a dash, never a winning trade
    assert "Largest win · largest loss $400 · — $400 · —" in _text(only_long)


def test_excursion_sentence_and_ticks():
    rows = [_trade(1, -200.0, high=100.0), _trade(2, -200.0, high=400.0),
            _trade(3, 500.0, low=50.0), _trade(4, 500.0, low=600.0),
            _trade(5, -300.0, high=10.0, exit_kind="account_failure")]
    pairs, left_out = trades_tab.excursion_pairs(rows + [{**_trade(6, 1.0),
                                                          "min_equity_usd": None}])
    assert left_out == 1 and len(pairs) == 5
    assert trades_tab.excursion_sentence(pairs) == (
        "2 of 3 losers were never more than $250 in profit before they closed. "
        "1 of 2 winners went less than $250 against you first.")
    at_stop = [p for p in pairs if p[1]["exit_kind"] == "stop"]
    assert trades_tab.excursion_sentence(at_stop).split(".")[0].endswith("before the stop")
    values, text = trades_tab.money_ticks(-960.0, 7350.0)
    assert values[0] <= -960 and values[-1] >= 7350 and "$0" in text and "$2k" in text
    assert trades_tab.money_ticks(0.0, 960.0, most=4)[1] == ["$0", "$250", "$500", "$750",
                                                            "$1k"]


def test_excursion_figures_split_winners_and_losers():
    rows = [_trade(1, -200.0, high=100.0), _trade(2, 500.0, low=50.0)]
    pairs, _ = trades_tab.excursion_pairs(rows)
    worst = trades_tab.excursion_figure(pairs, "worst")
    best = trades_tab.excursion_figure(pairs, "best")
    assert [t.name for t in worst.data] == ["Winner", "Loser"]
    assert list(worst.data[0].x) == [pytest.approx(50.0)]
    assert list(best.data[1].x) == [pytest.approx(100.0)]
    assert worst.data[0].marker.color == "#1D4E89" and worst.data[1].marker.color == "#A34A12"
    assert worst.layout.xaxis.title.text and worst.layout.yaxis.title.text


_COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{6}\b|rgba?\(")


def test_html_has_no_color_literal_and_figures_follow_the_dark_theme():
    """Theme contract: inline styles use ``var(--lab-…)``; charts read the active palette."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    rows = [_trade(1, -200.0, high=100.0), _trade(2, 500.0, low=50.0),
            _trade(3, 477.22, exit_kind="account_failure")]
    built = [trades_tab.distribution_card(rows),
             trades_tab.performance_card(rows, [("Direction", "Long only")], "X"),
             trades_tab.trades_card(rows, show_all=False),
             trades_tab._excursion_header(), trades_tab._MUTED_DASH]
    for markup in built:
        assert not _COLOR_LITERAL.search(str(markup)), str(markup)[:300]
    assert "color:var(--lab-orange-dark)" in str(built[2])  # the account-loss-limit reason
    assert "color:var(--lab-muted)" in str(built[4])
    pairs, _ = trades_tab.excursion_pairs(rows)
    theme.set_theme_resolver(lambda: "dark")
    try:
        fig = trades_tab.excursion_figure(pairs, "worst")
    finally:
        theme.set_theme_resolver(None)
    assert fig.data[0].marker.color == theme.DARK_COLORS["blue"]
    assert fig.data[1].marker.color == theme.DARK_COLORS["orange"]
    assert fig.layout.shapes[0].line.color == theme.DARK_COLORS["rule"]  # the zero line
    assert fig.layout.plot_bgcolor == theme.DARK_COLORS["chart_ground"]
    # with the resolver removed the figure is light again (nothing was kept at import time)
    light = trades_tab.excursion_figure(pairs, "worst")
    assert light.data[0].marker.color == theme.COLORS["blue"]
    assert light.layout.shapes[0].line.color == theme.COLORS["rule"]


def test_trades_card_rows_tint_actions_and_toggle():
    rows = [_trade(i, -100.0 if i % 2 else 200.0, account=1 + i // 5) for i in range(1, 12)]
    rows[4] = _trade(5, 477.22, exit_kind="account_failure", scale=103_600)
    rows[6] = _trade(7, -385.28, account=2, exit_kind="account_failure")
    card = str(trades_tab.trades_card(rows, show_all=False))
    assert card.count('data-action="trade:') == 8
    assert "Showing 8 of 11 · click any trade to open it in Trade review" in _text(card)
    assert 'data-action="showall"' in card and "Show all" in card
    # only the winning trade ended by the account's loss limit is tinted
    assert card.count('<tr class="risk clickable"') == 1
    assert 'data-action="trade:5|1"' in card.split('<tr class="risk clickable"')[1][:200]
    assert "5 at 25,900.00 · 5 at 25,874.75" in _text(card)
    assert "Account loss limit" in _text(card)
    assert "Jan 12, 10:31 PM" in _text(card) and "Jan 12, 10:41 PM" in _text(card)
    everything = str(trades_tab.trades_card(rows, show_all=True))
    assert everything.count('data-action="trade:') == 11
    assert "Showing all 11" in _text(everything) and "Show the first 8" in everything
    few = str(trades_tab.trades_card(rows[:3], show_all=False))
    assert "Showing all 3" in _text(few) and "showall" not in few


def test_action_handler_opens_that_exact_trade_in_trade_review():
    import ifvg_lab_nav as nav

    reruns: list[int] = []
    st_module = SimpleNamespace(session_state={}, rerun=lambda: reruns.append(1))
    target = {"result_id": RESULT_ID, "store_root": "isolated", "app": "ifsm",
              "study_key": RESULT_ID, "name": "Funded variation study"}
    ctx = SimpleNamespace(target=target, configuration=LEADER, firm_key=TPT, context={})
    rows = [_trade(4, -190.28), _trade(18, 477.22, exit_kind="account_failure")]
    trades_tab.handle_action("trade:18|1", ctx, rows, st_module)
    assert st_module.session_state[nav.REVIEW_TARGET] == {
        "source": "Funded trades", "result_id": RESULT_ID, "configuration": LEADER,
        "firm_key": TPT, "account_number": 1, "trade_seq": 18, "back_tab": "Trades"}
    assert st_module.session_state[nav.NAV] == "Trade review"
    assert st_module.session_state[nav.FUNDED_TARGET] == target
    shared = st_module.session_state[nav.CONTEXT_KEY][RESULT_ID]
    assert shared["account"] == {"pair": f"{LEADER}|{TPT}", "number": 1}
    assert reruns == [1]


def test_action_handler_toggles_and_ignores_unknown_trades():
    st_module = SimpleNamespace(session_state={}, rerun=lambda: None)
    ctx = SimpleNamespace(target={"result_id": "r"}, configuration="c", firm_key="f",
                          context={})
    rows = [_trade(4, -190.28)]
    trades_tab.handle_action("showall", ctx, rows, st_module)
    assert ctx.context[trades_tab.SHOW_ALL] is True
    trades_tab.handle_action("showall", ctx, rows, st_module)
    assert ctx.context[trades_tab.SHOW_ALL] is False
    for action in (None, "", "trade:999|1", "trade:x|1", "detail:c|f"):
        trades_tab.handle_action(action, ctx, rows, st_module)
    assert "ifvg_lab_v1_review_target" not in st_module.session_state


# ── the reference case (saved study, read only) ───────────────────────────


@pytest.fixture(scope="module")
def study():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study

    return open_funded_study(STORE, RESULT_ID)


@pytest.fixture(scope="module")
def leader_trades(study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    return ordered_trades(study, LEADER, TPT)


@needs_study
def test_reference_distribution_and_performance(study, leader_trades):
    card = trades_tab.distribution_card(leader_trades)
    assert _text(card).startswith("Trade results, 114 funded trades")
    assert trades_tab.distribution_caption(leader_trades) == (
        "Losses stay small and bunched: the worst of 47 losers is −$950. Ten trades above "
        "$2,000 make the long right tail.")
    text = _text(trades_tab.performance_card(leader_trades, study.settings(LEADER),
                                             "TakeProfitTrader"))
    for line in ("Trades · win rate 114 · 58.8% 114 · 58.8% —",
                 "Gross profit $59,978.74 $59,978.74 —",
                 "Gross loss −$22,248.16 −$22,248.16 —",
                 "Average win · average loss $895 · −$473 $895 · −$473 —",
                 "Largest win · largest loss $7,350 · −$950 $7,350 · −$950 —",
                 "Longest winning · losing streak 6 · 6 6 · 6 —",
                 "Drawdowns · average length 12 · 7.8 trades 12 · 7.8 trades —",
                 "Average drawdown · worst $1,207 · $3,661 $1,207 · $3,661 —",
                 "Longest drawdown 45 trades 45 trades —",
                 "Short column stays empty on long-only configurations."):
        assert line in text, line


@needs_study
def test_reference_excursions(leader_trades):
    pairs, left_out = trades_tab.excursion_pairs(leader_trades)
    assert left_out == 0 and len(pairs) == 114
    # CALCULATIONS.md's definition gives 36 of 47 (the mock shows 35); four losers
    # closed at the account's loss limit, not the stop
    assert trades_tab.excursion_sentence(pairs) == (
        "36 of 47 losers were never more than $250 in profit before they closed. "
        "46 of 67 winners went less than $250 against you first.")


@needs_study
def test_reference_trade_list_matches_the_mock_rows(leader_trades):
    card = trades_tab.trades_card(leader_trades, show_all=False)
    text = _text(card)
    assert "Showing 8 of 114 · click any trade to open it in Trade review" in text
    assert ("1 Jan 12, 10:31 PM Jan 12, 10:41 PM 25,883.75 25,874.75 10 at 25,874.75 Stop "
            "−$190.28") in text
    assert ("1 Jan 15, 2:22 AM Jan 15, 1:56 PM 25,723.75 25,704.00 5 at 25,743.50 · 5 at "
            "25,752.75 Account loss limit +$477.22") in text
    assert ("2 Jan 16, 3:03 AM Jan 16, 5:58 AM 25,827.50 25,805.75 5 at 25,849.25 · 5 at "
            "25,827.50 Break-even stop +$207.22") in text
    assert ("2 Jan 19, 8:31 PM Jan 19, 10:52 PM 25,415.00 25,387.75 10 at 25,387.75 Stop "
            "−$555.28") in text
    html = str(card)
    assert html.count('<tr class="risk clickable"') == 1
    assert 'data-action="trade:18|1"' in html.split('<tr class="risk clickable"')[1][:200]
    everything = str(trades_tab.trades_card(leader_trades, show_all=True))
    assert everything.count('data-action="trade:') == 114
    # all five account-loss-limit exits are named; only the winning one is tinted
    assert _text(everything).count("Account loss limit") == 5
    assert everything.count('<tr class="risk clickable"') == 1


@needs_study
def test_reference_more_keeps_every_earlier_column(study):
    rows = trades_tab.earlier_rows(study, LEADER, TPT)
    assert len(rows) == 114
    assert list(rows[0]) == [
        "Trade", "Trading day", "Account", "Direction", "Quantity", "Entry (Chicago)",
        "Exit (Chicago)", "Entry price (points)", "Stop price (points)", "Exit price (points)",
        "Points gained or lost", "Result after costs", "Initial risk", "Exit reason",
        "Prices from", "Stop filled worse than the stop"]
    fifth = rows[4]
    assert fifth["Trading day"] == "January 15, 2026"
    assert fifth["Entry (Chicago)"] == "January 15, 2026, 2:22 AM"
    assert fifth["Exit (Chicago)"] == "January 15, 2026, 1:56 PM"
    assert fifth["Points gained or lost"] == "5 at +19.75; 5 at +29.00"
    assert fifth["Initial risk"] == "$395.00" and fifth["Result after costs"] == "+$477.22"


# ── headless run of the detail screen on the Trades tab ───────────────────


def _detail_app(scripts: str):
    import sys as _sys

    _sys.path.insert(0, scripts)
    import streamlit as st
    from ifvg_lab_funded import render_funded_detail

    render_funded_detail(st, {})


def _app(monkeypatch, clicks: list[str] | None = None):
    import ifvg_lab_ui
    from streamlit.testing.v1 import AppTest

    pending = list(clicks or [])

    def fake_clickable(markup, *, key, st_module=None):
        if key.startswith("trades_") and pending:
            return pending.pop(0)
        return None

    monkeypatch.setattr(ifvg_lab_ui, "clickable", fake_clickable)
    at = AppTest.from_function(_detail_app, default_timeout=120,
                               kwargs={"scripts": str(REPO / "scripts")})
    at.session_state["ifvg_lab_v1_funded_target"] = {
        "result_id": RESULT_ID, "store_root": str(STORE), "app": "ifsm",
        "study_key": RESULT_ID, "name": "Funded variation study", "status": "Completed"}
    at.session_state["funded_comparison_v1_selected_context"] = {
        RESULT_ID: {"firm_key": TPT, "configuration": LEADER, "tab": "Trades"}}
    return at


@needs_study
def test_detail_screen_renders_the_trades_tab(monkeypatch):
    at = _app(monkeypatch).run()
    assert not at.exception, at.exception
    html = "\n".join(str(getattr(e.proto, "body", "")) for e in at.get("html"))
    assert "Trade results, 114 funded trades" in html
    assert "Performance summary" in html
    assert "Account equity range during each trade" in html
    assert "36 of 47 losers" in html
    assert len(at.get("plotly_chart")) == 2
    assert at.expander and at.expander[0].label == "More: every column of the earlier trade table"
    assert len(at.dataframe[0].value) == 114


@needs_study
def test_clicking_a_trade_row_opens_trade_review(monkeypatch):
    at = _app(monkeypatch, clicks=["trade:18|1"]).run()
    assert not at.exception, at.exception
    assert at.session_state["ifvg_lab_v1_review_target"] == {
        "source": "Funded trades", "result_id": RESULT_ID, "configuration": LEADER,
        "firm_key": TPT, "account_number": 1, "trade_seq": 18, "back_tab": "Trades"}
    assert at.session_state["ifvg_workspace_destination"] == "Trade review"
