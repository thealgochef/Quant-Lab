"""Fixes from the owner's review of the redesign (docs/ifvg-redesign-fixes, F2–F11).

Each test fails without its fix. Pure helpers and headless screens only; nothing
here saves, approves or launches anything.
"""

from __future__ import annotations

import html
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[3]
for _path in (REPO / "scripts", REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ifvg_lab_cache  # noqa: E402
import ifvg_lab_detail_risk as risk  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.presentation.lab import firm_race as fr  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (  # noqa: E402
    DEFAULT_SEED,
    findings,
)


def _full(net_values=(), died=0.42) -> fr.FullRace:
    return fr.FullRace(firm_key="takeprofittrader", firm_name="TakeProfitTrader", paths=1_000,
                       seed=DEFAULT_SEED, method="blocks", slots=114, paid_share=1 - died,
                       died_share=died, still_going_share=0.0, typical_to_payout=6.0,
                       typical_to_limit=6.0, payouts_before_death=1.84, lost_accounts=3_395,
                       cash_per_account=5_328.0, accounts_bought=4_395, net_cash_bad=9_800.0,
                       net_cash_typical=23_100.0, net_cash_good=38_100.0, seconds=8.0,
                       net_cash_values=tuple(net_values), horizon_first_day="2026-01-13",
                       horizon_last_day="2026-06-10")


# ── F2: the early-losses finding names its model (corrections A2, A3) ─────


def _bundle(died=0.22, limit=2_000.0):
    race = SimpleNamespace(died_share=died, loss_limit=-limit, trigger=2_600.0, paths=20_000)
    return SimpleNamespace(race=race, loss_limit=limit)


EARLY_TITLES = ("Early account failures (conditional model)",
                "Early losses in the fixed-boundary diagnostic")


def _finding(race: ifvg_lab_cache.EarlyLossRace):
    fired = findings(largest_account_share=None, five_largest_share=None,
                     died_first_share=race.died_share, total_profit=1.0, beta_r2=None,
                     race_basis=race.basis, race_detail=race.detail,
                     boundaries=race.boundaries, firm="TakeProfitTrader")
    return next((f for f in fired if f.title in EARLY_TITLES), None)


def _saved(monkeypatch):
    """A tiny saved result for the pair, so the Summary can rebuild the Risk tab's key."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import study_from_result
    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    result = {"settings": {"firm_profiles": [FIRM_PROFILES["takeprofittrader"]
                                             .model_dump(mode="json")],
                           "processing_clock": TWO_BUSINESS_DAYS_FED_1600.model_dump(
                               mode="json")},
              "period": {"cutoff_utc": "2026-06-10T21:00:00Z"}, "summaries_cents": {},
              "tables": {"trades": [{"pair_id": "cfg|takeprofittrader", "configuration": "cfg",
                                     "firm_key": "takeprofittrader",
                                     "entry_utc": f"2026-01-{13 + i}T15:00:00Z", "seq": i}
                                    for i in range(3)]}}
    study = study_from_result(result, result_id="result")
    monkeypatch.setattr(ifvg_lab_cache, "_study", lambda *_args: study)
    slots, cutoff_ns, digest = ifvg_lab_cache.saved_race_binding(study, "cfg",
                                                                 "takeprofittrader")
    return ifvg_lab_cache.firm_race_key(
        "store", "result", "cfg", "takeprofittrader", method="blocks", seed=DEFAULT_SEED,
        paths=fr.DEFAULT_FULL_PATHS, slots=slots, cutoff_ns=cutoff_ns, terms_digest=digest)


def test_f2_flat_race_is_named_and_points_to_the_firm_rules(monkeypatch):
    monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda: {})
    race = ifvg_lab_cache.early_loss_race(_bundle(), "store", "result", "cfg",
                                          "takeprofittrader", "TakeProfitTrader")
    assert race.basis == "flat" and race.died_share == 0.22
    assert race.boundaries == (-2_000.0, 2_600.0)
    finding = _finding(race)
    # correction A2: a fixed closed-profit diagnostic, not accounts hitting a loss limit
    assert finding.title == "Early losses in the fixed-boundary diagnostic"
    assert finding.text == (
        "In the fixed closed-profit diagnostic (−$2,000 / +$2,600, 20,000 resampled paths), 22% "
        "of paths fell to −$2,000 before reaching +$2,600. It uses closed trade results only, "
        "not TakeProfitTrader's account rules. Run conditional resampling with "
        "TakeProfitTrader's rules under Risk and simulation for the ledger version.")
    assert finding.next_step == ("run the ledger version under Risk and simulation before "
                                 "drawing account conclusions.")


def test_f2_firm_rules_race_replaces_the_flat_figure_when_it_exists(monkeypatch):
    key = _saved(monkeypatch)
    store = {key: _full(died=0.42)}
    monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda: store)

    def never(*_args, **_kwargs):  # the Summary never runs the ledger itself
        raise AssertionError("the Summary must not run conditional resampling")

    monkeypatch.setattr(fr, "full_race", never)
    race = ifvg_lab_cache.early_loss_race(_bundle(), "store", "result", "cfg",
                                          "takeprofittrader", "TakeProfitTrader")
    assert race.basis == "firm" and race.model_id == fr.MODEL_ID
    assert race.detail == "1,000 paths, to June 10, 2026"
    finding = _finding(race)
    # correction A3: a conditional model, not an exact fresh-account probability
    assert finding.title == "Early account failures (conditional model)"
    assert finding.text == (
        "In conditional resampling of the recorded trades with TakeProfitTrader's ledger rules "
        "(1,000 paths, to June 10, 2026), 42% of first accounts failed before any payout was "
        # review R1: the source-selection limitation is named too
        "received. It reuses the recorded trades (already shaped by the historical accounts' "
        "entry selection and skipped opportunities) in fixed slots, with shortened results and "
        "compressed intratrade paths, so it is not an exact fresh-account probability.")
    assert finding.next_step == "see its limits under Risk and simulation before relying on it."
    # after a reload the process store still holds it: the same figure is quoted
    again = ifvg_lab_cache.early_loss_race(_bundle(), "store", "result", "cfg",
                                           "takeprofittrader", "TakeProfitTrader")
    assert again == race
    # another seed or path count is not the default draw: the flat figure stays
    other = {key[:6] + (DEFAULT_SEED, 2_000) + key[8:]: _full(died=0.42)}
    monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda: other)
    assert ifvg_lab_cache.early_loss_race(_bundle(), "store", "result", "cfg",
                                          "takeprofittrader", "TakeProfitTrader").basis == "flat"


def test_f2_severity_rule_uses_the_figure_shown(monkeypatch):
    key = _saved(monkeypatch)
    monkeypatch.setattr(ifvg_lab_cache, "firm_race_results", lambda: {key: _full(died=0.10)})
    race = ifvg_lab_cache.early_loss_race(_bundle(died=0.22), "store", "result", "cfg",
                                          "takeprofittrader", "TakeProfitTrader")
    assert race.basis == "firm"
    assert _finding(race) is None  # 10% under the firm's rules: below the 15% rule


def test_f2_risk_tab_and_summary_share_one_store():
    assert risk._full_results() is ifvg_lab_cache.firm_race_results()


# ── F3: where the recorded cash result sits (superseded by correction A5) ─


def test_f3_share_below_counts_ties_half():
    race = _full(net_values=[100.0, 200.0, 300.0, 300.0, 400.0])
    assert race.share_below(300.0) == pytest.approx((2 + 1) / 5)
    assert race.share_below(50.0) == 0.0
    assert _full().share_below(1.0) is None


def test_f3_pile_classification_is_gone():
    # correction A5: no lucky/unlucky classification and no 35%/65% thresholds
    assert not hasattr(risk, "pile_side")


def test_f3_firm_table_states_the_actual_results_standing():
    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    values = [9_800.0] * 100 + [23_100.0] * 700 + [38_100.0] * 200
    race = _full(net_values=values)
    rules = SimpleNamespace(firm_name="TakeProfitTrader",
                            profile=FIRM_PROFILES["takeprofittrader"],
                            processing=TWO_BUSINESS_DAYS_FED_1600)
    text = html.unescape(str(risk.full_comparison(race, None, 3_078_188, rules)))
    # correction A5: a descriptive percentile with its model and horizon, never luck
    assert ("Under this conditional model, the recorded net cash ($30,781.88) is above 80% of "
            "the 1,000 resampled paths' net cash (ties count half). Horizon: the study's 114 "
            "trade slots, January 13 – June 10, 2026; paths draw recorded trades with "
            "replacement in blocks of 10. The percentile describes where the recorded result "
            "sits; it does not measure luck.") in text


def test_f3_trading_profit_caption_names_what_it_measures():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import resampling as rs

    fan = rs.equity_fan([100.0, -50.0, 200.0, -80.0] * 30, method="blocks", paths=2_000,
                        seed=3)
    caption = risk.end_caption(fan, fan.typical_end)
    assert caption.startswith("In trading profit, the recorded result")
    # correction A5: the percentile describes a sampling model; it does not measure luck
    assert caption.endswith("it does not measure luck.")
    assert "wasn't a lucky ordering" not in caption


def test_f3_full_race_keeps_every_runs_net_cash():
    race = replace(_full(), net_cash_values=(1.0, 2.0))
    assert race == _full()  # the per-run values never change equality or the saved figures


# ── F8: the study name in every breadcrumb ────────────────────────────────


def test_f8_breadcrumb_never_falls_back_to_a_generic_label():
    import ifvg_lab_funded
    import ifvg_lab_trade_review

    named = {"name": "Funded variation study — 64 configurations around original three windows"}
    assert ifvg_lab_funded.study_title(named) == "Funded variation study"
    for unreadable in ({"name": ""}, {"name": None}, {}):
        assert ifvg_lab_funded.study_title(unreadable) == "Study name not readable"
        assert ifvg_lab_trade_review._study_label(unreadable) == "Study name not readable"
    assert ifvg_lab_trade_review._study_label(named) == "Funded variation study"


def test_f8_a_link_takes_the_name_from_the_saved_run(monkeypatch):
    """A deep link (reload, shared link) opens the result under its saved run's name."""

    import ifvg_lab_nav

    result_id = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
    saved = {"result_id": result_id, "store_root": "store", "app": "ifsm",
             "study_key": "plan", "name": "Funded variation study — 64 configurations",
             "status": "Completed"}
    monkeypatch.setattr(ifvg_lab_nav, "_saved_target", lambda *_args: dict(saved))
    st_module = SimpleNamespace(session_state={})
    ifvg_lab_nav._apply_link({"view": "detail", "app": "ifsm", "result": result_id,
                              "firm": "takeprofittrader", "config": "S1", "tab": "risk"},
                             {"repo_root": REPO}, st_module)
    assert st_module.session_state[ifvg_lab_nav.FUNDED_TARGET]["name"] == saved["name"]


# ── F4: one draft, one configuration count ────────────────────────────────


def _comparison_draft(selections, gap_rules=(), triggers=(), gates=None):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs

    redesign = {"baseline": "named", "gap_rules": list(gap_rules),
                "withdrawal_triggers_usd": list(triggers)}
    if gates:
        redesign["gates"] = gates
    return SimpleNamespace(
        mode_id=fs.COMPARISON_MODE, draft_id="d1", updated_at_utc="2026-09-24T08:24:36+00:00",
        steps={"review": {
            "funded_comparison": {
                "cost_per_side_cents": 514, "firm_keys": ["takeprofittrader", "myfundedfutures"],
                "instrument": "mini", "plan_kind": "variations",
                "processing": "two_business_days", "quantity": 1,
                "source_run_id": "a0f66422ef9f6b3393ad21d0b9740ebc0f23684cfe3463139a90743141512fd7",
                "variation": {"base": "S0_D80_W1_P1", "half_cost_mills": 514, "half_quantity": 10,
                              "selections": selections, "whole_cost_mills": 5140,
                              "whole_quantity": 1}},
            fs.REDESIGN_KEY: redesign}})


#: the draft "Funded configuration comparison — September 24, 2026" (isolated copy)
SELECTIONS_36 = {
    "enable_shorts": ["enable_shorts.false"],
    "enabled_entry_sessions": ["enabled_entry_sessions.asia-london-ny",
                               "enabled_entry_sessions.all_open_market_v1"],
    "exit_policy": ["exit_policy.fixed_target_v1",
                    "exit_policy.scale_out_half_breakeven_hold_to_close_v1"],
    "htf_timeframes": ["htf_timeframes.1H-4H"],
    "parent_timeframes": ["parent_timeframes.1m-3m-5m-10m-15m-30m"],
    "tp_r_multiple": ["tp_r_multiple.1.0", "tp_r_multiple.3.0"],
}
GAPS = ["htf_gap_invalidation_policy.own_timeframe_close_v1",
        "htf_gap_invalidation_policy.execution_wick_full_fill_v1"]


def test_f4_plan_count_multiplies_gap_rules_and_triggers_and_names_the_engine():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs

    draft = _comparison_draft(SELECTIONS_36, GAPS, [500, 1000, 2000])
    check = SimpleNamespace(configuration_count=6, count_is_exact=True,
                            needs_half_exit_engine=False)
    count = fs.plan_count(check, draft)
    assert (count.total, count.text()) == (36, "36 configurations on the half-exit engine")
    # the same draft checked on the pinned engine: same count, same words
    pinned = SimpleNamespace(configuration_count=6, count_is_exact=True,
                             needs_half_exit_engine=True)
    assert fs.plan_count(pinned, draft).text() == count.text()
    assert fs.blocked_engine_lead(fs.plan_count(pinned, draft), True, True) == (
        "This study contains 36 configurations on the half-exit engine and needs the version "
        "that supports half exits.")
    # whole-position only: no engine words (both engines count it the same way)
    whole = {k: v for k, v in SELECTIONS_36.items() if k != "exit_policy"}
    four = SimpleNamespace(configuration_count=4, count_is_exact=True,
                           needs_half_exit_engine=False)
    assert fs.plan_count(four, _comparison_draft(whole, GAPS, [500, 1000, 2000])).text() == (
        "24 configurations")
    assert fs.plan_count(four, _comparison_draft(whole)).text() == "4 configurations"


def test_f4_my_studies_uses_the_shared_count_and_says_approval_is_blocked():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import library as lib

    draft = _comparison_draft(SELECTIONS_36, GAPS, [500, 1000, 2000])
    pinned = SimpleNamespace(configuration_count=6, count_is_exact=True,
                             needs_half_exit_engine=True, problems=("needs half",))
    assert lib.draft_summary(draft, pinned) == (
        "36 configurations on the half-exit engine · both firms")
    blocked = fs.approval_blocked_note([fs.Blocker("gap_rule", "g"),
                                        fs.Blocker("withdrawal_trigger", "t")])
    assert blocked == "Approval blocked: 2 settings need engine support."
    assert lib.comparison_draft_note(pinned, blocked) == (
        "Needs the half-exit engine to run. Approval blocked: 2 settings need engine support. "
        "Your saved settings are unchanged.", "orange")
    fine = SimpleNamespace(needs_half_exit_engine=False, problems=())
    assert lib.comparison_draft_note(fine, blocked) == (
        "Approval blocked: 2 settings need engine support. Your saved settings are unchanged.",
        "orange")
    assert fs.approval_blocked_note([fs.Blocker("dates", "d")]) == (
        "Approval blocked: 1 setting needs a change.")
    assert fs.approval_blocked_note([]) is None


def _source():
    from alpha_lab.propsim.funded.comparison_source import discover_comparison_sources

    try:
        return next((s for s in discover_comparison_sources() if "S0_D80_W1_P1" in s.by_name),
                    None)
    except Exception:
        return None


def test_f4_the_real_check_counts_the_draft_as_setup_and_review_do():
    """The verified source (when on this computer): the saved draft counts 36 on any engine."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs
    from alpha_lab.propsim.funded.comparison_draft import (
        check_saved_comparison,
        saved_settings,
        saved_study_selections,
    )

    source = _source()
    if source is None:
        pytest.skip("the verified strategy study is not on this computer")
    draft = _comparison_draft(SELECTIONS_36, GAPS, [500, 1000, 2000])
    draft.steps["review"]["funded_comparison"]["source_run_id"] = source.package.run_id
    check = check_saved_comparison(saved_settings(draft), saved_study_selections(draft),
                                   {source.package.run_id: source})
    assert fs.plan_count(check, draft).text() == "36 configurations on the half-exit engine"
    blockers = fs.saved_draft_blockers(draft, source, repo_root=REPO)
    assert fs.approval_blocked_note(blockers) == (
        "Approval blocked: 2 settings need engine support.")


# ── F5: point in time leaves out later trades and later accounts ──────────


def test_f5_point_in_time_lists_only_what_existed_at_the_moment():
    import pandas as pd

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_panels as rp

    at = pd.Timestamp
    opened = {1: at("2026-01-12T23:00:00Z"), 2: at("2026-02-10T16:00:00Z"),
              3: at("2026-04-12T23:30:00Z"), 4: at("2026-05-01T14:00:00Z")}
    moment = at("2026-04-13T00:10:00Z")  # Apr 12, 7:10 PM Chicago
    assert rp.known_accounts([1, 2, 3, 4], opened, moment, current=3) == [1, 2, 3]
    # C24: even the reviewed account cannot be listed before its saved creation.
    assert rp.known_accounts([1, 2, 5], opened, at("2026-01-13T00:00:00Z"), current=2) == [1]
    views = [SimpleNamespace(seq=n, entry_utc=at(t)) for n, t in (
        (1, "2026-04-12T20:00:00Z"), (2, "2026-04-13T00:07:00Z"), (3, "2026-04-14T15:00:00Z"))]
    assert [v.seq for v in rp.known_trades(views, moment, current=2)] == [1, 2]
    # a moment before the entry (a setup step) keeps the reviewed trade itself
    assert [v.seq for v in rp.known_trades(views, at("2026-04-12T22:18:00Z"), 2)] == [1, 2]


def test_f5_context_line_leaves_out_the_total_only_in_point_in_time():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_panels as rp

    parts = ["All open-market hours · Long only · Half at 1R", "1R", "TakeProfitTrader"]
    assert rp.context_line(parts, 6, 77, 114, point_in_time=True).endswith(
        "TakeProfitTrader · Account 6 · trade 77 at this firm")
    assert rp.context_line(parts, 6, 77, 114, point_in_time=False).endswith(
        "Account 6 · trade 77 of 114 at this firm")


# ── F7: the earlier explanations are readable on the redesigned screens ──


def test_f7_full_firm_terms_match_the_earlier_caption():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    assert fs.firm_terms_sentence(FIRM_PROFILES["takeprofittrader"]) == (
        "TakeProfitTrader: $102 per account, 80% trader share, up to 6 minis or equivalent, "
        "$2,000 loss allowance, keeps $2,100 after each payout, $500 minimum gross request. "
        "Owner-defined simulation terms.")
    assert fs.firm_terms_sentence(FIRM_PROFILES["myfundedfutures"]).startswith(
        "MyFundedFutures: $125 per account, 90% trader share, up to 3 minis or equivalent")


def test_f7_every_earlier_fact_note_is_kept_in_full_on_payouts():
    import ifvg_lab_detail_payouts as payouts
    import ifvg_lab_ui

    store = REPO / "data/ifsm_ui_replication/search/v1"
    result = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
    if not (store / "funded_comparison_results" / result / "result.json").is_file():
        pytest.skip("the saved funded variation study is not on this computer")
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        present_pair_detail,
    )

    study = ifvg_lab_ui.funded_study(str(store), result)
    view = payouts.build_payouts_view(study, "S1-T1-H14-P1-L-SO", "takeprofittrader")
    detail = present_pair_detail(study.result, "S1-T1-H14-P1-L-SO", "takeprofittrader")
    earlier = [f.note for f in (*detail.headline, *detail.facts) if f.note]
    assert earlier and [note for _l, _v, note in view.fact_notes] == earlier
    panel = html.unescape(str(payouts.fact_notes_panel(view)))
    assert all(note.replace("**", "") in panel for note in earlier)
    assert "February 11, 2026, 10:27 AM" in panel and "CST" not in panel


# ── F9: the shared shell has no framework red, words in names, one rail ──


def test_f9_theme_recolors_the_framework_red_widgets():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import RAIL_LINK_CSS, WORKSPACE_CSS

    for selector in ('[data-testid="stProgress"] [role="progressbar"] > div > div > div',
                     'label[data-baseweb="radio"]:has(input:checked) > div:first-child',
                     'label[data-baseweb="checkbox"]:has(input:checked) > span',
                     '[data-testid="stSlider"] [role="slider"]', '[data-baseweb="tag"]',
                     '[data-baseweb="tab-highlight"]'):
        assert selector in WORKSPACE_CSS, selector
    assert "FF4B4B" not in WORKSPACE_CSS.upper()
    assert RAIL_LINK_CSS in WORKSPACE_CSS
    assert "font-weight: 400 !important" in RAIL_LINK_CSS
    assert "white-space: normal !important" in RAIL_LINK_CSS


def test_f9_the_rail_ships_its_own_link_style(monkeypatch):
    import ifvg_lab_ui

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import RAIL_LINK_CSS

    html_out: list[str] = []

    class _Box:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake = SimpleNamespace(
        sidebar=_Box(), container=lambda **_k: _Box(), html=html_out.append,
        button=lambda *_a, **_k: False, page_link=lambda *_a, **_k: None)
    monkeypatch.setattr(ifvg_lab_ui, "_OTHER_PAGES", [SimpleNamespace(title="ML Training")])
    ifvg_lab_ui.rail("My studies", fake)
    assert any(RAIL_LINK_CSS in out for out in html_out)


def test_f9_new_default_study_names_use_the_date_in_words():
    from datetime import date

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs
    from alpha_lab.agents.data_infra.ifvg.study_drafts import date_in_words, proposed_draft_name

    assert date_in_words("2026-09-24") == "September 24, 2026"
    assert date_in_words(date(2026, 9, 4)) == "September 4, 2026"
    assert proposed_draft_name("Evaluate", baseline_profile_name=None, day="2026-09-24") == (
        "Evaluate — September 24, 2026")
    assert fs.new_study_name(date(2026, 9, 24)).endswith("September 24, 2026")
    wizard = (REPO / "scripts/ifvg_research_wizard.py").read_text(encoding="utf-8")
    assert "{date.today()}" not in wizard and wizard.count("date_in_words(date.today())") == 3


# ── F10: wording and number formats ──────────────────────────────────────


def test_f10_system_texts_lose_their_code_names():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.format import display_words

    saved = ("Strategy-Core runs from a separate branch that adds the scale-out exit; "
             "fixed-target configurations behave exactly as in the pinned Core (checked against "
             "the saved study's trades).")
    assert display_words(saved) == (
        "The strategy engine runs from a separate branch that adds the scale-out exit; "
        "fixed-target configurations behave exactly as in the default engine (checked against "
        "the saved study's trades).")
    assert display_words("Micro positions are priced on the E-mini Nasdaq-100 (NQ) recorded "
                         "trades.") == ("Micro positions are priced on the E-mini Nasdaq-100 "
                                        "recorded trades.")
    assert display_words("It requires the Strategy-Core version that supports partial exits.") \
        == "It requires the engine version that supports partial exits."
    for text in (saved, "1 x Micro E-mini Nasdaq-100 (MNQ) per trade", "pinned Core"):
        out = display_words(text)
        assert "Strategy-Core" not in out and "(NQ)" not in out and "(MNQ)" not in out
        assert "Core" not in out


def test_f10_saved_limitations_are_shown_without_code_names():
    import ifvg_lab_detail_settings as settings
    import ifvg_lab_ui

    store = REPO / "data/ifsm_ui_replication/search/v1"
    result = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
    if not (store / "funded_comparison_results" / result / "result.json").is_file():
        pytest.skip("the saved funded variation study is not on this computer")
    study = ifvg_lab_ui.funded_study(str(store), result)
    view = settings.build_settings_view(study, "S1-T1-H14-P1-L-SO", "takeprofittrader",
                                        "TakeProfitTrader", REPO)
    shown = " ".join([*view.limitations, *(c.description for c in view.corrections),
                      *(text for _kind, text in view.verification)])
    assert "Strategy-Core" not in shown and "(NQ)" not in shown and "pinned Core" not in shown
    # owner decision texts stay word for word
    saved = [str(d.get("decision") or "") for d in study.result.get("owner_decisions") or []]
    assert [decision for _d, _s, decision, _st, _a in view.decisions] == saved


def test_f10_the_drop_growth_profit_row_uses_one_money_format():
    import re

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import resampling as rs
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.format import money_k

    assert [money_k(v) for v in (-731, 30_000, 59_512, -2_400, 1_800)] == [
        "−$0.7k", "$30.0k", "$59.5k", "−$2.4k", "$1.8k"]
    values = [330.0, -450.0, 900.0, -120.0, 40.0] * 30
    growth = rs.drawdown_growth(values, paths=500, seed=1)
    table = html.unescape(str(risk.growth_table(growth)))
    # correction A2: the row is cumulative closed profit at the 5th/median/95th percentile
    row = (table.split("Closed profit, 5th · median · 95th percentile", 1)[1]
           .split("</tr>", 1)[0])
    amounts = re.findall(r"[−-]?\$[\d,.]+k?", row)
    assert amounts and all(re.fullmatch(r"−?\$\d[\d,]*\.\dk", a) for a in amounts), amounts


def test_f10_ranking_names_get_a_wide_column():
    import ifvg_lab_funded

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import WORKSPACE_CSS

    source = (REPO / "scripts/ifvg_lab_funded.py").read_text(encoding="utf-8")
    assert 'h.Column("config", "Configuration", width="37%")' in source
    assert ".lab-ranking .lab-table td.num" in WORKSPACE_CSS
    assert ifvg_lab_funded._ranking_table  # the table is wrapped in .lab-ranking
    assert '<div class="lab-ranking">' in source



def test_f10_the_strategy_measures_note_is_worded_for_the_screen():
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        STRATEGY_METRICS_NOTE,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.format import display_words

    assert "Strategy-Core" in STRATEGY_METRICS_NOTE  # the shared text is unchanged
    assert "the strategy engine's candle rules" in display_words(STRATEGY_METRICS_NOTE)
    source = (REPO / "scripts/ifvg_lab_detail_settings.py").read_text(encoding="utf-8")
    assert "note = fmt.display_words(STRATEGY_METRICS_NOTE)" in source
