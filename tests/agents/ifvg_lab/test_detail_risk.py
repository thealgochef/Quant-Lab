"""Configuration detail · Risk and simulation (mocks 05, 05b).

Pure helpers of ``scripts/ifvg_lab_detail_risk.py`` (money input, seeds, labels,
bins, sentences, figures) and a headless smoke test of the tab on the saved
funded variation study (read only; skipped where it is not on this computer):
the fixed seed gives identical numbers on every run, "Run again" draws a new
seed and says which, and "Back to the fixed draw" returns. Nothing here saves
anything; the full version is not run in the smoke test.
"""

from __future__ import annotations

import html
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
for _path in (REPO / "scripts", REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ifvg_lab_detail_risk as risk  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.lab import resampling as rs  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (  # noqa: E402
    DEFAULT_SEED,
)

#: a hard-coded color (``#RRGGBB`` or ``rgba(…)``): none may reach the screen's HTML
COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{6}\b|rgba?\(")

STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
SAVED = (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file()

VALUES = [float(v) for v in np.random.default_rng(5).normal(330, 900, 114).round(2)]


# ── inputs, seeds and wording ─────────────────────────────────────────────


@pytest.mark.parametrize(("text", "value"), [
    ("−$2,000", -2000.0), ("-2000", -2000.0), ("$2,000", 2000.0), ("+$2,600", 2600.0),
    (" 2600 ", 2600.0), ("2,600.50", 2600.5), ("abc", None), ("", None), (None, None),
    ("nan", None)])
def test_parse_money(text, value):
    assert risk.parse_money(text) == value


def test_new_seed_is_never_the_fixed_or_the_last_one_and_is_announced():
    seeds = {risk.new_seed(12345) for _ in range(200)}
    assert DEFAULT_SEED not in seeds and 12345 not in seeds
    assert all(10_000 <= s < 100_000 for s in seeds)
    seed = risk.new_seed()
    assert risk.seed_note(seed) == (
        f"Run again used a new random draw (seed {seed}). The fixed draw returns when you "
        "choose Back to the fixed draw.")
    assert risk.seed_note(DEFAULT_SEED) is None


def test_race_defaults_come_from_the_firm_terms():
    terms = {"loss_allowance_cents": 200_000, "retained_cushion_cents": 210_000,
             "minimum_gross_request_cents": 50_000}
    assert risk.race_defaults(terms) == (-2000.0, 2600.0)
    assert risk.race_defaults({**terms, "loss_allowance_cents": 150_000}) == (-1500.0, 2600.0)
    assert risk.race_defaults(None) == (None, None)


def test_explanation_names_the_trade_count():
    # correction A5: resampling with replacement, not reordering
    assert risk.explanation(114) == (
        "Every chart here resamples this configuration's own 114 recorded funded trades with "
        "replacement — in blocks of 10 consecutive trades, or one trade at a time — thousands "
        "of times. A path can repeat some trades and leave out others, so its total differs "
        "from the recorded total. These charts describe the recorded trades under that "
        "sampling model; they are not a forecast.")


def test_spread_labels_keep_a_minimum_gap():
    placed = risk.spread_labels([100.0, 101.0, 102.0, 500.0], 10.0)
    ordered = sorted(placed)
    assert all(b - a >= 10.0 - 1e-9 for a, b in zip(ordered, ordered[1:], strict=False))
    assert placed[3] > placed[0]


def test_money_ticks_are_round_and_readable():
    ticks, labels = risk.money_ticks(-5_000, 68_000)
    assert ticks[0] == 0 and all(t % 5_000 == 0 for t in ticks)
    assert labels[0] == "$0" and labels[1].endswith("k")
    ticks, labels = risk.money_ticks(-6_000, 1_000)
    assert "−$5k" in labels or "−$4k" in labels


def test_end_bins_are_aligned_on_zero_and_sum_to_one():
    bins = risk.end_bins([-7_000, -1, 0, 4_999, 5_000, 12_345])
    assert [left for left, _ in bins] == [-10_000, -5_000, 0, 5_000, 10_000]
    assert sum(share for _, share in bins) == pytest.approx(1.0)
    assert dict(bins)[0] == pytest.approx(2 / 6)


def test_methods_reading_close_and_apart():
    fan = rs.equity_fan(VALUES, method="blocks", paths=2_000, seed=1)
    near = rs.EquityFan(**{**fan.__dict__, "method": "shuffle",
                           "bad_end": fan.bad_end + 0.05 * abs(fan.typical_end)})
    assert risk.methods_reading({"blocks": fan, "shuffle": near}).startswith(
        "The two methods land close together")
    apart = rs.EquityFan(**{**fan.__dict__, "method": "shuffle",
                            "bad_end": fan.bad_end + 0.2 * abs(fan.typical_end) + 1_000})
    assert risk.methods_reading({"blocks": fan, "shuffle": apart}).startswith(
        "The two methods split apart")


def test_streak_and_end_captions_use_the_numbers():
    dist = rs.StreakDistribution(shares={5: 0.3, 6: 0.4, 7: 0.2, 8: 0.1}, typical=6, bad=8)
    assert risk.streak_caption(dist, 6) == ("Typical worst run is 6 losses, which is exactly "
                                            "what happened. Plan for 8 or more one time in "
                                            "twenty (orange).")
    assert "the actual order's worst run was 4 losses" in risk.streak_caption(dist, 4)
    fan = rs.equity_fan(VALUES, method="blocks", paths=2_000, seed=3)
    caption = risk.end_caption(fan, fan.typical_end)
    assert "is above 50%" in caption or "is above 49%" in caption or "is above 51%" in caption
    # correction A5: a percentile is described under its sampling model, never as luck
    assert "(ties count half)" in caption and "it does not measure luck." in caption


def test_figures_build_on_resampled_values():
    race = rs.payout_race(VALUES, paths=2_000, seed=2)
    fan = rs.equity_fan(VALUES, method="blocks", paths=2_000, seed=2)
    growth = rs.drawdown_growth(VALUES, paths=2_000, seed=2)
    path = list(np.cumsum([0.0, *VALUES]))
    assert len(risk.race_figure(race).data) == 3
    assert len(risk.fan_figure(fan, path).data) == 4 + 10 + 2
    assert risk.end_figure(fan, path[-1]).data[0].type == "bar"
    assert risk.streak_figure(rs.streak_distribution(fan.streaks)).data[0].type == "bar"
    left, right = risk.growth_figures(growth)
    assert len(left.data) == 3 and len(right.data) == 2
    # correction A2: a closed-profit fall and fixed boundaries, not account losses
    assert "Fell $2,000 from a previous high" in risk.growth_table(growth)
    assert risk.race_caption(race).startswith("Of 2,000 resampled paths of up to 200 trades")


def test_fixed_seed_is_reproducible_and_a_new_seed_differs():
    first = rs.payout_race(VALUES, paths=5_000, seed=DEFAULT_SEED)
    again = rs.payout_race(VALUES, paths=5_000, seed=DEFAULT_SEED)
    other = rs.payout_race(VALUES, paths=5_000, seed=risk.new_seed())
    assert first == again
    assert first.paid_by_trade != other.paid_by_trade


def test_a_missing_package_is_looked_for_again(monkeypatch):
    """Review fix: the firm-rules inputs are not cached while the package is unavailable."""

    import uuid
    from types import SimpleNamespace

    import ifvg_lab_cache
    import ifvg_lab_ui

    fr = risk.fr
    found: dict[str, object] = {"source": None, "frame": None}

    def open_source(_plan):
        if found["source"] == "broken":
            raise OSError("unreadable")
        return found["source"]

    monkeypatch.setattr(fr, "open_source", open_source)
    monkeypatch.setattr(ifvg_lab_cache, "index_minutes", lambda *_args: found["frame"])
    monkeypatch.setattr(ifvg_lab_ui, "funded_study", lambda *_args: SimpleNamespace(plan=None))
    monkeypatch.setattr(fr.MinuteIndex, "from_frame", classmethod(lambda cls, frame: frame))
    monkeypatch.setattr(fr, "build_inputs",
                        lambda *_args, source, minutes: ("rules", source, minutes))
    monkeypatch.setattr(fr, "check_original", lambda *_args: "validation")
    key = (f"cache-test-{uuid.uuid4().hex}", "result", "cfg", "takeprofittrader")
    assert risk._full_inputs(*key) == (
        None, "the study's verified strategy package is not on this computer")
    found["source"] = "broken"
    assert risk._full_inputs(*key) == (
        None, "the study's strategy package failed its check (OSError)")
    found["source"] = "source"  # restored; its bars not readable yet
    inputs, reason = risk._full_inputs(*key)
    assert inputs is None and "one-minute E-mini bars" in reason
    found["frame"] = "bars"
    assert risk._full_inputs(*key) == (("rules", "source", "bars", "validation"), None)
    found["source"] = None  # once built, kept for the process
    assert risk._full_inputs(*key)[0] == ("rules", "source", "bars", "validation")


def _full(slots: int):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import firm_race as fr

    return fr.FullRace(firm_key="takeprofittrader", firm_name="TakeProfitTrader", paths=1_000,
                       seed=DEFAULT_SEED, method="blocks", slots=slots, paid_share=0.5,
                       died_share=0.3, still_going_share=0.2, typical_to_payout=12.0,
                       typical_to_limit=9.0, payouts_before_death=1.2, lost_accounts=800,
                       cash_per_account=450.0, accounts_bought=1_900, net_cash_bad=-500.0,
                       net_cash_typical=9_000.0, net_cash_good=20_000.0, seconds=12.0)


def _rules():
    from types import SimpleNamespace

    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    return SimpleNamespace(firm_name="TakeProfitTrader",
                           profile=FIRM_PROFILES["takeprofittrader"],
                           processing=TWO_BUSINESS_DAYS_FED_1600)


def test_full_version_wording_says_which_trades_are_the_charts_draw():
    """Review fix: past 200 trade slots only the first 200 trades are the chart's runs."""

    rules = _rules()
    short = html.unescape(str(risk.full_comparison(_full(114), None, 3078188, rules)))
    # correction A3/A5: resampled draws of the conditional ledger model, not reordered runs
    assert ("Ledger rules: the first 1,000 of the same resampled draws, seed "
            f"{DEFAULT_SEED}, 12 seconds.") in short
    assert risk.beyond_chart(114) == "" and risk.beyond_chart(200) == ""
    long = html.unescape(str(risk.full_comparison(_full(250), None, 3078188, rules)))
    assert ("the first 1,000 of the same resampled draws (their first 200 trades; the other 50 "
            "trade slots continue with a further fixed draw), seed") in long


# ── theme: colors come from the active palette, never from the module ─────


def test_inline_html_names_theme_variables_and_no_color_literal():
    """Every color the tab writes into HTML is a ``var(--lab-…)`` (light and dark share it)."""

    from dataclasses import replace

    growth = rs.drawdown_growth(VALUES, paths=500, seed=2)
    rules = _rules()
    # net cash values give the firm-rules table its "standing" sentence
    race = replace(_full(114), net_cash_values=tuple([9_800.0] * 100 + [23_100.0] * 700
                                                     + [38_100.0] * 200))
    pieces = {
        "css": risk.RISK_CSS,
        "fan_legend": str(risk.fan_legend()),
        "growth_legend": str(risk.growth_legend()),
        "growth_table": str(risk.growth_table(growth)),
        "stat_block": str(risk.stat_block("Reached +$2,600 first", "61%",
                                          border=theme.css_var("blue"),
                                          color=theme.css_var("blue_dark"),
                                          sub="to the upper · to the lower")),
        "soft_box": str(risk.soft_box("Label", "value", detail="more")),
        "soft_box_placeholder": str(risk.soft_box("Label", "[waiting]", placeholder=True)),
        "title_row": str(risk.title_row("Title", "right")),
        "line": str(risk.line("A sentence")),
        "full_comparison": str(risk.full_comparison(race, None, 3078188, rules)),
    }
    literals = {name: COLOR_LITERAL.findall(text) for name, text in pieces.items()}
    assert not any(literals.values()), literals
    assert "var(--lab-body-2)" in pieces["fan_legend"]
    assert "border-top:1px solid var(--lab-sample-line)" in pieces["fan_legend"]
    assert "border-top:3px dashed var(--lab-ink)" in pieces["fan_legend"]
    assert "background:var(--lab-blue-line)" in pieces["growth_legend"]
    assert 'style="color:var(--lab-orange-dark)"' in pieces["growth_table"]
    assert "border-left:4px solid var(--lab-blue)" in pieces["stat_block"]
    assert "color:var(--lab-muted)" in pieces["stat_block"]
    assert "background:var(--lab-soft-panel)" in pieces["soft_box"]
    assert "color:var(--lab-ink)" in pieces["soft_box"]
    assert "color:var(--lab-muted)" in pieces["soft_box_placeholder"]
    assert "color:var(--lab-muted)" in pieces["title_row"]
    assert "color:var(--lab-body)" in pieces["line"]
    assert "color:var(--lab-body)" in pieces["full_comparison"]  # the standing sentence
    # the module holds palette keys, never palette values
    assert (risk.STILL_GOING, risk.SAMPLE_LINE, risk.ZERO_LINE) == (
        "light_rule", "sample_line", "zero_line")


def test_figures_take_the_dark_palette_when_the_theme_is_dark():
    """The same builders draw the dark palette once the theme resolver says dark."""

    race = rs.payout_race(VALUES, paths=2_000, seed=2)
    fan = rs.equity_fan(VALUES, method="blocks", paths=2_000, seed=2)
    growth = rs.drawdown_growth(VALUES, paths=2_000, seed=2)
    path = list(np.cumsum([0.0, *VALUES]))
    light, dark = theme.COLORS, theme.DARK_COLORS
    assert risk.race_figure(race).layout.plot_bgcolor == light["light_rule"]
    assert risk.fan_figure(fan, path).data[-2].line.color == light["blue"]
    theme.set_theme_resolver(lambda: "dark")
    try:
        race_fig = risk.race_figure(race)
        fan_fig = risk.fan_figure(fan, path)
        end_fig = risk.end_figure(fan, path[-1])
        streak_fig = risk.streak_figure(rs.streak_distribution(fan.streaks))
        left_fig, right_fig = risk.growth_figures(growth)
    finally:
        theme.set_theme_resolver(None)
    # the payout race: stacked fills, the "still going" ground and the panel-tinted grid
    assert [trace.fillcolor for trace in race_fig.data] == [
        dark["blue"], dark["light_rule"], dark["orange"]]
    assert race_fig.layout.plot_bgcolor == dark["light_rule"]
    assert race_fig.layout.yaxis.gridcolor == theme.rgba("panel", 0.35, "dark")
    checkpoint = next(a for a in race_fig.layout.annotations if "After 10 trades" in a.text)
    assert checkpoint.bgcolor == dark["panel"] and checkpoint.font.color == dark["body"]
    # correction A2: the diagnostic's labels name boundaries, not payouts
    paid_label = next(a for a in race_fig.layout.annotations if a.text == "<b>Upper first</b>")
    assert paid_label.font.color == dark["on_ink"]
    # the fan: bands, sample paths, typical path, what happened, the zero line
    assert fan_fig.data[1].fillcolor == dark["blue_band"]
    assert fan_fig.data[3].fillcolor == dark["blue_mid"]
    assert fan_fig.data[4].line.color == theme.rgba("sample_line", 0.55, "dark")
    assert fan_fig.data[-2].line.color == dark["blue"]
    assert fan_fig.data[-1].line.color == dark["ink"]
    assert fan_fig.layout.shapes[0].line.color == dark["zero_line"]
    happened = fan_fig.layout.annotations[-1]
    assert happened.bgcolor == dark["ink"] and happened.font.color == dark["on_ink"]
    # the distributions and the drawdown growth
    assert set(end_fig.data[0].marker.color) <= {dark["orange"], dark["blue_mid"]}
    assert set(streak_fig.data[0].marker.color) <= {dark["blue"], dark["orange"], dark["blue_mid"]}
    assert streak_fig.data[0].textfont.color == dark["body_2"]
    assert left_fig.layout.shapes[0].fillcolor == dark["orange_light"]
    assert left_fig.data[-1].line.color == dark["blue"]
    assert right_fig.data[0].fillcolor == dark["orange_light"]
    assert right_fig.data[1].marker.color == dark["ink"]
    # with the resolver gone the light palette is back
    assert risk.race_figure(race).layout.plot_bgcolor == light["light_rule"]


# ── the tab in a headless app (saved study, read only) ────────────────────


APP = f'''
import sys
from pathlib import Path
REPO = Path(r"{REPO}")
for path in (REPO / "scripts", REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import streamlit as st
from ifvg_lab_cache import ranking
from ifvg_lab_funded import DetailContext
from ifvg_lab_nav import app_roots, funded_context
from ifvg_lab_ui import funded_study
from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import configuration_name
import ifvg_lab_detail_risk as risk
store = str(REPO / "data/ifsm_ui_replication/search/v1")
target = {{"result_id": "{RESULT_ID}", "store_root": store, "app": "ifsm",
          "study_key": "{RESULT_ID}", "name": "Funded variation study", "status": "Completed"}}
study = funded_study(store, "{RESULT_ID}")
row = next(r for r in ranking(store, "{RESULT_ID}", "takeprofittrader")
           if r.configuration == "{LEADER}")
ctx = DetailContext(target=target, study=study, store_root=store, result_id="{RESULT_ID}",
                    configuration="{LEADER}", firm_key="takeprofittrader",
                    firm="TakeProfitTrader",
                    name=configuration_name(study.settings("{LEADER}"), "{LEADER}"), row=row,
                    roots=app_roots(REPO)["ifsm"], context=funded_context("{RESULT_ID}", st))
risk.header_right(st, ctx)
risk.render(st, ctx)
'''


def _texts(at) -> str:
    parts = [str(getattr(e, "body", "") or "") for e in at.get("html")]
    parts += [str(getattr(e, "value", "") or "") for e in at.markdown]
    return html.unescape("\n".join(parts))


def _run_tab(seed: int | None = None):
    from streamlit.testing.v1 import AppTest

    script = APP
    if seed is not None:  # what "Run again" stores for this configuration and firm
        script = script.replace(
            "risk.header_right(st, ctx)",
            f"ctx.context['risk_seeds'] = {{'{LEADER}|takeprofittrader': {seed}}}\n"
            "risk.header_right(st, ctx)")
    at = AppTest.from_string(script, default_timeout=240)
    started = time.monotonic()
    at.run()
    return at, time.monotonic() - started


@pytest.mark.skipif(not SAVED, reason="the saved funded variation study is not on this computer")
def test_tab_renders_with_the_fixed_seed_and_a_new_seed_says_so():
    at, first = _run_tab()
    assert not at.exception, at.exception
    text = _texts(at)
    # corrections A2, A3, A5: resampled paths, fixed boundaries, a conditional ledger model
    assert "resamples this configuration's own 114 recorded funded trades" in text
    assert f"Fixed draw: seed {DEFAULT_SEED}, 20,000 paths per chart" in text
    assert "20,000 resampled paths of 114 trades" in text
    assert "Fixed closed-profit boundaries: which is crossed first?" in text
    assert ("Conditional resampling of recorded trades with TakeProfitTrader's ledger rules"
            in text)
    assert risk.limitations_line() in text  # shown before any run
    assert "Checked first: the saved order, replayed through the same simulator" in text
    assert "Not run yet · choose Run conditional resampling with TakeProfitTrader's rules" in text
    assert "Sampled closed-profit drawdown from a previous high" in text
    assert "lucky" not in text.lower() and "reordered runs" not in text
    loss = next(w for w in at.text_input if w.label == "Lower boundary")
    trigger = next(w for w in at.text_input if w.label == "Upper boundary")
    assert (loss.value, trigger.value) == ("−$2,000", "+$2,600")
    assert any(b.label == "Run again" for b in at.button)
    assert not any(b.label == "Back to the fixed draw" for b in at.button)
    # the fixed draw gives identical numbers on every run (and opens from the cache)
    again, cached = _run_tab()
    assert _texts(again) == text
    assert cached < 3.0, f"cached render took {cached:.1f} s (first {first:.1f} s)"
    # a "Run again" seed is announced with the way back, and changes the numbers
    other, _ = _run_tab(seed=48213)
    changed = _texts(other)
    assert ("Run again used a new random draw (seed 48213). The fixed draw returns when you "
            "choose Back to the fixed draw.") in changed
    assert any(b.label == "Back to the fixed draw" for b in other.button)
    assert changed != text


# ── closeout review follow-up: the historical-order sentence names what it compares ──

_SAVED = {"net_cash_cents": 3_078_188, "received_cents": 3_139_388, "payouts": 13,
          "accounts": 6, "costs_cents": 61_200}


def _validation(**changes):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import firm_race as fr

    fields = {"firm_key": "takeprofittrader", "firm_name": "TakeProfitTrader", "trades": 114,
              "saved": dict(_SAVED), "replayed": dict(_SAVED), "trades_matching": 114,
              "refused_slots": 0, "inexact_shapes": 0, "unpinned_shapes": 0}
    return fr.Validation(**{**fields, **changes})


def test_historical_order_success_names_the_compared_fields_and_what_it_does_not_compare():
    text = risk.validation_text(_validation())
    assert text.startswith("Checked first: the saved order, replayed through the same simulator")
    assert "identical" not in text
    for words in ("net cash ($30,781.88)", "payouts received ($31,393.88)", "payout count (13)",
                  "accounts bought (6)", "account costs ($612.00) exactly",
                  "the same number of trades (114), each with the same net result and "
                  "account-loss flag in the saved order",
                  "It doesn't compare fill times, prices or quantities, which account took each "
                  "trade, payout or failure times, or which setup each trade came from."):
        assert words in text, words


def test_historical_order_success_shows_the_compared_values_and_reads_for_one_trade():
    one = {**_SAVED, "net_cash_cents": 0}
    text = risk.validation_text(_validation(trades=1, trades_matching=1, saved=one,
                                            replayed=dict(one)))
    assert "net cash ($0.00)" in text  # the compared saved value, never a blank
    assert "the same number of trades (1), each with the same net result" in text
    assert "trade'" not in text


def test_historical_order_failure_names_money_in_dollars_and_the_per_trade_fields():
    replayed = {**_SAVED, "net_cash_cents": 3_000_000}
    text = risk.validation_text(_validation(replayed=replayed, trades_matching=113,
                                            replayed_trades=114))
    assert "identical" not in text
    assert "net cash $30,781.88 saved vs $30,000.00 replayed" in text
    assert "113 of 114 trades have the same net result and account-loss flag" in text


def test_historical_order_failure_with_a_different_trade_count_says_so():
    text = risk.validation_text(_validation(trades_matching=0, replayed_trades=113))
    assert "the replay produced 113 trades against 114 saved" in text
    assert "0 of 114 trades" not in text
