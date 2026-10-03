"""New funded comparison and Review and approve (redesign P7b, mocks 10, 10b, 11, 11b).

Pure helpers run everywhere. The screen checks use the real verified daily-close
study's configurations (read only; skipped when that archive is absent) and
temporary stores only. Nothing here approves or launches: every approval and
launch function is replaced by one that fails the test if it is ever called.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_setup as fs  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: E402
    SEARCH_AXIS_REGISTRY_V1,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft, save_draft  # noqa: E402

HALF_EXIT_ENGINE = "exit_policy" in SEARCH_AXIS_REGISTRY_V1
S0 = "enabled_entry_sessions.asia-london-ny"
ALL_HOURS = "enabled_entry_sessions.all_open_market_v1"
S0_IDS = {
    "enabled_entry_sessions": S0,
    "holding_policy": "holding_policy.scheduled_daily_close_v1",
    "htf_gap_invalidation_policy": fs.GAP_CLOSE,
    "opposing_min_gap_ticks": "opposing_min_gap_ticks.1",
    "opposing_parent_distance_ticks_max": "opposing_parent_distance_ticks_max.80",
    "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m",
}
TRIGGER_TEXT = ("Comparing withdrawal triggers needs the funded simulator to accept a trigger per "
                "plan; today it uses one ($500 above the retained $2,100). Your saved settings "
                "haven't been changed.")
PACKAGE_GATES = {"min_executed_trades": 60, "min_independent_days": 50,
                 "min_profit_factor": 1.1, "max_drawdown_r": 15.0,
                 "max_time_under_water_days": 3, "max_top_day_pnl_share": 0.4}
DAYS = tuple((date(2026, 1, 13) + timedelta(days=n)).isoformat() for n in range(107))
VARIATION_SELECTIONS = {  # the saved 64-configuration variation study (repair R1)
    "enable_shorts": ["enable_shorts.false", "enable_shorts.true"],
    "enabled_entry_sessions": [S0, ALL_HOURS],
    "exit_policy": [fs.WHOLE_EXIT, fs.HALF_EXIT],
    "htf_timeframes": ["htf_timeframes.1H-4H", "htf_timeframes.1H"],
    "parent_timeframes": ["parent_timeframes.1m-3m-5m-10m-15m-30m",
                          "parent_timeframes.1m-5m-10m-15m-30m"],
    "tp_r_multiple": ["tp_r_multiple.1.0", "tp_r_multiple.2.0", "tp_r_multiple.3.0"],
}


def _choices(**overrides) -> fs.SetupChoices:
    base = dict(source_run_id="run", base="S0_D80_W1_P1",
                selections={"enabled_entry_sessions": (S0,),
                            "tp_r_multiple": ("tp_r_multiple.1.0",),
                            fs.GAP_AXIS: (fs.GAP_CLOSE,), fs.TRIGGER_AXIS: ("500",),
                            "enable_shorts": ("enable_shorts.false",)},
                firm_keys=("takeprofittrader", "myfundedfutures"))
    base.update(overrides)
    return fs.SetupChoices(**base)


# ── pure helpers ──────────────────────────────────────────────────────────


def test_chip_labels_are_readable_values_never_keys():
    assert fs.value_label("enabled_entry_sessions", S0) == (
        "Original three windows: 3:00 PM–12:45 AM, 1:00–6:00 AM, 7:00 AM–1:00 PM")
    assert fs.value_label("tp_r_multiple", "tp_r_multiple.3.0") == "3× the stop distance"
    assert fs.short_label("tp_r_multiple", "tp_r_multiple.1.0") == "1R"
    assert fs.value_label(fs.GAP_AXIS, fs.GAP_CLOSE) == (
        "A candle on its own chart closes through it")
    assert fs.value_label(fs.GAP_AXIS, fs.GAP_WICK) == "A one-minute wick reaches the far edge"
    assert fs.value_label("exit_policy", fs.HALF_EXIT) == (
        "Half at the target, rest held to break-even or 3:55 PM")
    assert fs.value_label("parent_timeframes", "parent_timeframes.1m-3m-5m-10m-15m-30m") == (
        "1, 3, 5, 10, 15 and 30 minutes")
    assert fs.value_label("htf_timeframes", "htf_timeframes.1H-4H") == "1-hour and 4-hour"
    assert fs.value_label(fs.TRIGGER_AXIS, "1000") == "$1,000"
    assert fs.axis_title(fs.TRIGGER_AXIS) == "Withdraw when the surplus above $2,100 reaches"


def test_firm_chips_read_price_and_share_from_the_saved_profiles():
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    for key, profile in FIRM_PROFILES.items():
        label = fs.firm_chip_label(key)
        assert label.startswith(profile.firm_name)
        assert f"${profile.acquisition_cost_cents // 100:,} per account" in label
        assert f"{profile.trader_share_pct}% share" in label
    assert fs.today_trigger_usd() == 500  # each firm's minimum gross request today


def test_named_baseline_text_is_built_from_saved_values():
    assert fs.baseline_description("S0_D80_W1_P1", S0_IDS) == (
        "S0_D80_W1_P1 — original three windows, opposing distance 80 ticks, 1-tick minimum, "
        "one-minute parents, flat by 3:55 PM")
    assert fs.baseline_caption(S0_IDS) == "Flat by 3:55 PM"


def test_legacy_warning_uses_the_repair_r7_wording(monkeypatch):
    from alpha_lab.agents.data_infra.ifvg import named_baselines

    first, rest = fs.legacy_warning_parts()
    assert first == "This baseline doesn't follow the 3:55 PM close."
    assert rest.startswith("This baseline holds positions across the daily close and weekends")
    assert rest.endswith("It's here so older studies still open exactly as saved.")
    monkeypatch.setattr(named_baselines, "legacy_baseline_warning", lambda _section: None)
    assert fs.legacy_warning_parts(object()) is None  # no warning when R7 gives none


def test_gate_text_parses_exactly_and_refuses_bad_input():
    assert fs.parse_gate("max_drawdown_r", "15R") == (15.0, None)
    assert fs.parse_gate("max_best_day_share", "40%") == (0.4, None)
    assert fs.parse_gate("min_days", "2,050") == (2050, None)
    assert fs.parse_gate("min_trades", "1.5")[1].startswith("Enter a whole number")
    assert fs.parse_gate("min_profit_factor", "abc")[1] == "Enter a number, like 1.1."
    assert fs.parse_gate("max_best_day_share", "140%")[1].startswith("Enter a percentage")
    assert fs.parse_gate("drop_largest_payout", "Information only") == ("Information only", None)
    assert fs.format_gate("min_days", 2050) == "2,050"
    assert fs.format_gate("max_best_day_share", 0.4) == "40%"
    assert fs.format_gate("max_drawdown_r", 15.0) == "15R"


def test_gate_defaults_come_from_the_source_studys_saved_thresholds():
    defaults = fs.gate_defaults(PACKAGE_GATES)
    assert defaults == {"min_trades": 60, "min_days": 50, "min_profit_factor": 1.1,
                        "max_drawdown_r": 15.0, "max_days_under_water": 3,
                        "max_best_day_share": 0.4, "drop_largest_payout": "Pass/fail"}


def test_impossible_threshold_is_kept_as_saved_and_flagged_never_trimmed():
    defaults = fs.gate_defaults(PACKAGE_GATES)
    rows = {r.spec.key: r for r in fs.gate_rows({"min_days": 2050}, defaults,
                                                trading_days=DAYS)}
    row = rows["min_days"]
    assert row.status == "flag" and row.text == "2,050"  # the saved value, not a clamp
    assert row.message == ("Can't be met: only 107 trading days are selected. Kept as saved, not "
                           "trimmed — change it or the dates.")
    ok = {r.spec.key: r for r in fs.gate_rows({}, defaults, trading_days=DAYS)}
    assert ok["min_days"].message == "OK against 107 trading days"
    assert ok["min_trades"].message == "OK"
    blockers = [b.key for b in fs.gate_blockers(list(rows.values()))]
    assert "gate_min_days" in blockers and "gates_changed" in blockers
    assert fs.gate_blockers(list(ok.values())) == []


def test_under_water_limit_is_marked_as_the_owners_decision_never_chosen():
    defaults = fs.gate_defaults(PACKAGE_GATES)
    row = next(r for r in fs.gate_rows({}, defaults, trading_days=DAYS)
               if r.spec.key == "max_days_under_water")
    assert row.status == "decision" and row.text == "3"
    # correction A11: a neutral pending decision, no "the data points to 18–20" advice
    assert row.message == ("Needs your decision. The saved limit is 3 trading days under water; "
                           "it stays as saved, and this page doesn't propose another value.")
    assert "18" not in row.message and "20" not in row.message
    assert not row.changed  # shown as saved; the page never substitutes another value


def test_typed_invalid_value_is_shown_but_nothing_is_saved():
    defaults = fs.gate_defaults(PACKAGE_GATES)
    rows = fs.gate_rows({}, defaults, trading_days=DAYS, typed={"min_trades": "sixty"})
    row = next(r for r in rows if r.spec.key == "min_trades")
    assert row.status == "invalid" and row.message.endswith("Nothing was saved.")
    assert any(b.key == "gate_min_trades" for b in fs.gate_blockers(rows))


def test_chip_actions_add_remove_and_never_empty_a_setting():
    options = {"enabled_entry_sessions": (S0, ALL_HOURS), fs.TRIGGER_AXIS: ("500", "1000", "2000"),
               fs.FIRM_AXIS: ("takeprofittrader", "myfundedfutures")}
    choices = _choices()
    same, adding, changed = fs.apply_action(choices, f"remove|enabled_entry_sessions|{S0}",
                                            options=options)
    assert same is choices and not changed  # the last value has no remove button
    _, adding, changed = fs.apply_action(choices, "add|enabled_entry_sessions", options=options)
    assert adding == "enabled_entry_sessions" and not changed
    picked, adding, changed = fs.apply_action(
        choices, f"pick|enabled_entry_sessions|{ALL_HOURS}", options=options,
        adding="enabled_entry_sessions")
    assert changed and picked.values("enabled_entry_sessions") == (S0, ALL_HOURS)
    assert adding is None  # nothing left to add
    unknown, _, changed = fs.apply_action(choices, "pick|enabled_entry_sessions|made.up",
                                          options=options)
    assert not changed and unknown is choices
    more, _, _ = fs.apply_action(choices, "pick|withdrawal_trigger_usd|2000", options=options)
    more, _, _ = fs.apply_action(more, "pick|withdrawal_trigger_usd|1000", options=options)
    assert more.triggers == (500, 1000, 2000)  # kept in the offered order
    firm, _, changed = fs.apply_action(choices, "remove|firm|myfundedfutures", options=options)
    assert changed and firm.firm_keys == ("takeprofittrader",)


def test_new_settings_block_approval_with_the_missing_capability_named():
    assert fs.setup_blockers(_choices(), base_ids=S0_IDS) == []
    triggers = _choices(selections={**_choices().selections,
                                    fs.TRIGGER_AXIS: ("500", "1000", "2000")})
    (blocker,) = fs.setup_blockers(triggers, base_ids=S0_IDS)
    assert blocker.key == "withdrawal_trigger" and blocker.text == TRIGGER_TEXT
    gaps = _choices(selections={**_choices().selections, fs.GAP_AXIS: (fs.GAP_CLOSE, fs.GAP_WICK)})
    (blocker,) = fs.setup_blockers(gaps, base_ids=S0_IDS)
    assert blocker.key == "gap_rule" and "a candle on its own chart closes through it" in (
        blocker.text) and blocker.text.endswith("Your saved settings haven't been changed.")
    legacy = fs.setup_blockers(_choices(baseline=fs.LEGACY), base_ids=S0_IDS)
    assert [b.key for b in legacy] == ["baseline"]
    assert "isn't a configuration of the verified strategy study" in legacy[0].text


def test_dates_other_than_the_source_studys_block_approval_with_a_sentence():
    class Resolved:
        problems = ()
        trading_days = DAYS[:5]

    (blocker,) = fs.date_blockers(Resolved(), DAYS)
    assert blocker.text.startswith("A funded comparison replays its source study's saved dates "
                                   "(January 13 – April 29, 2026, 107 trading days).")
    assert "(January 13 – January 17, 2026, 5 trading days) are kept in this draft" in blocker.text
    Resolved.trading_days = DAYS
    assert fs.date_blockers(Resolved(), DAYS) == []


def test_earliest_start_and_the_new_study_name_are_plain_words():
    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import human_name
    from alpha_lab.agents.data_infra.ifvg.research_period import (
        EARLIEST_LOCAL_MARKET_DATE,
        WARMUP_STORE_DAYS,
        earliest_evidence_day,
    )

    assert fs.earliest_start_sentence(earliest_evidence_day(), EARLIEST_LOCAL_MARKET_DATE,
                                      WARMUP_STORE_DAYS) == (
        "Earliest start: December 14, 2021. Stored market data begins December 2, 2021, and "
        "every study first replays 10 warmup days before its first evaluated day.")
    name = fs.new_study_name(date(2026, 9, 24))
    assert name == "Funded configuration comparison — September 24, 2026"
    assert human_name(name, "") == name  # accepted by the draft-name rule


def test_plan_lines_and_the_approval_sentence():
    choices = _choices(selections={"enabled_entry_sessions": (S0, ALL_HOURS),
                                   "tp_r_multiple": ("tp_r_multiple.1.0", "tp_r_multiple.3.0"),
                                   fs.GAP_AXIS: (fs.GAP_CLOSE, fs.GAP_WICK),
                                   "exit_policy": (fs.WHOLE_EXIT, fs.HALF_EXIT),
                                   fs.TRIGGER_AXIS: ("500", "1000", "2000"),
                                   "enable_shorts": ("enable_shorts.false",)})
    assert fs.plan_lines(choices) == ["2 entry hours", "× 2 targets", "× 2 gap rules",
                                      "× 2 exits", "× 3 withdrawal triggers"]
    assert fs.approval_sentence(48, ["takeprofittrader", "myfundedfutures"]) == (
        "I approve running this exact plan: 48 configurations, both firms, the dates above. "
        "No live trading.")
    assert fs.blocked_engine_lead(48, True, True) == (
        "This study contains 48 configurations and needs the version that supports half exits.")
    skipped = [({"exit_policy": fs.HALF_EXIT}, "the half exit is taken at 1R")] * 2
    assert fs.skipped_sentence(skipped, 6) == (
        "12 combinations left out: the half exit is taken at 1R.")


def test_review_rows_show_only_the_settings_that_differ():
    choices = _choices(selections={**_choices().selections,
                                   fs.TRIGGER_AXIS: ("500", "1000")})
    strategy = [{"enabled_entry_sessions": S0, "tp_r_multiple": t, "enable_shorts":
                 "enable_shorts.false", "htf_timeframes": "htf_timeframes.1H-4H",
                 "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m",
                 "exit_policy": fs.WHOLE_EXIT} for t in ("tp_r_multiple.3.0", "tp_r_multiple.1.0")]
    options = {"tp_r_multiple": ("tp_r_multiple.1.0", "tp_r_multiple.3.0"),
               fs.TRIGGER_AXIS: ("500", "1000", "2000")}
    rows = fs.review_rows(strategy, choices, options)
    assert len(rows) == 4
    assert [(r["tp_r_multiple"], r[fs.TRIGGER_AXIS]) for r in rows] == [
        ("tp_r_multiple.1.0", "500"), ("tp_r_multiple.1.0", "1000"),
        ("tp_r_multiple.3.0", "500"), ("tp_r_multiple.3.0", "1000")]
    assert fs.varying_axes(rows) == ["tp_r_multiple", fs.TRIGGER_AXIS]
    same = fs.same_for_every(rows, S0_IDS, choices)
    assert "Long only" in same and "1-hour and 4-hour gaps" in same
    assert "1, 3, 5, 10, 15 and 30-minute supporting charts" in same
    assert "Opposing distance 80 ticks" in same and "Smallest opposing gap 1 tick" in same
    assert "Flat by 3:55 PM" in same


def test_new_settings_are_saved_under_their_own_key_and_the_plan_keys_stay_known():
    from alpha_lab.propsim.funded.comparison_draft import _KNOWN_SETTINGS

    draft = new_draft(fs.COMPARISON_MODE, display_name="Round trip")
    draft.steps = {"review": {
        "funded_comparison": fs.new_variation_settings("run", "S0_D80_W1_P1")}}
    offered = {"enabled_entry_sessions": (S0, ALL_HOURS), "tp_r_multiple": ("tp_r_multiple.1.0",),
               "enable_shorts": ("enable_shorts.false",)}
    choices = _choices(selections={**_choices().selections,
                                   "enabled_entry_sessions": (S0, ALL_HOURS),
                                   fs.TRIGGER_AXIS: ("500", "2000"),
                                   "exit_policy": (fs.WHOLE_EXIT, fs.HALF_EXIT)},
                       start="2026-02-02", end="2026-05-29", gates={"min_days": 2050})
    fs.write_choices(draft, choices, offered=offered)
    funded = draft.steps["review"]["funded_comparison"]
    assert set(funded) <= _KNOWN_SETTINGS  # the saved-draft check never sees a new key
    assert "exit_policy" not in funded["variation"]["selections"]  # not offered → never written
    assert funded["variation"]["selections"]["enabled_entry_sessions"] == [S0, ALL_HOURS]
    assert draft.steps["review"][fs.REDESIGN_KEY] == {
        "baseline": "named", "gap_rules": [fs.GAP_CLOSE], "withdrawal_triggers_usd": [500, 2000],
        "dates": {"start": "2026-02-02", "end": "2026-05-29"}, "gates": {"min_days": 2050}}
    back = fs.choices_from_draft(draft, base_ids=S0_IDS, offered=offered)
    assert back.triggers == (500, 2000) and back.gates == {"min_days": 2050}
    assert (back.start, back.end) == ("2026-02-02", "2026-05-29")
    assert back.values("enabled_entry_sessions") == (S0, ALL_HOURS)


# ── the screens (real verified study, temporary stores) ──────────────────


def _source():
    from alpha_lab.propsim.funded.comparison_source import discover_comparison_sources

    return next((s for s in discover_comparison_sources() if "S0_D80_W1_P1" in s.by_name), None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    source = _source()
    if source is None:
        pytest.skip("the verified daily-close study archive is not available locally")
    import ifvg_funded_comparison_study as screen
    import ifvg_workspace

    roots = {
        "repo_root": Path(__file__).resolve().parents[3], "store_root": tmp_path / "store",
        "store_roots": {"research": tmp_path / "store"},
        "draft_root": tmp_path / "drafts", "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines",
        "funded_comparison_state_root": tmp_path / "comparison_jobs",
        "reports_root": tmp_path / "reports",
    }
    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    called: list[str] = []

    def forbidden(name):
        def fail(*args, **kwargs):
            called.append(name)
            raise AssertionError(f"{name} must never run from these screens in a test")

        return fail

    for name in ("_spawn", "_freeze_and_launch", "record_owner_approval", "save_plan"):
        monkeypatch.setattr(screen, name, forbidden(name))
    return {"roots": roots, "source": source, "called": called}


def _app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_workspace(st, roots=ifvg_workspace._TEST_ROOTS)


def _open(screen: str, draft_id: str | None = None):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_app, default_timeout=240)
    at.session_state["ifvg_workspace_destination"] = "New study"
    at.session_state["ifvg_workspace_screen"] = screen
    if draft_id:
        at.session_state["ifvg_study_v1_draft_id"] = draft_id
    at.run()
    assert not at.exception, at.exception
    return at


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "warning", "info", "success", "error"):
        parts += [str(e.value) for e in at.get(kind)]
    for element in at.get("html"):
        body = getattr(getattr(element, "proto", None), "body", "") or ""
        body = re.sub(r"<style>.*?</style>", " ", body, flags=re.S)
        parts.append(re.sub(r"<[^>]+>", " ", body))
    return re.sub(r"\s+", " ", html.unescape("\n".join(parts)))


def _variation_draft(source, selections=None, redesign=None):
    draft = new_draft(fs.COMPARISON_MODE, display_name="Screen check variation study")
    draft.steps = {
        "objective": {"mode_id": fs.COMPARISON_MODE, "question_id": "funded_configuration_cash"},
        "search_space": {"axis_selections": {}, "mode_id": fs.COMPARISON_MODE},
        "review": {"funded_comparison": {
            "cost_per_side_cents": 514, "firm_keys": ["takeprofittrader", "myfundedfutures"],
            "instrument": "mini", "plan_kind": "variations", "processing": "two_business_days",
            "quantity": 1, "source_run_id": source.package.run_id,
            "variation": {"base": "S0_D80_W1_P1", "half_cost_mills": 514, "half_quantity": 10,
                          "selections": selections or VARIATION_SELECTIONS,
                          "whole_cost_mills": 5140, "whole_quantity": 1}}},
    }
    if redesign is not None:
        draft.steps["review"][fs.REDESIGN_KEY] = redesign
    return draft


def test_check_saved_comparison_ignores_the_redesign_settings(env):
    from alpha_lab.propsim.funded.comparison_draft import check_saved_comparison, saved_settings

    sources = {env["source"].package.run_id: env["source"]}
    whole = {k: v for k, v in VARIATION_SELECTIONS.items() if k != "exit_policy"}
    plain = _variation_draft(env["source"], whole)
    extra = _variation_draft(env["source"], whole, redesign={
        "baseline": "legacy", "gap_rules": [fs.GAP_CLOSE, fs.GAP_WICK],
        "withdrawal_triggers_usd": [500, 1000], "gates": {"min_days": 2050},
        "dates": {"start": "2025-06-16", "end": "2026-06-10"}})
    first = check_saved_comparison(saved_settings(plain), {}, sources)
    second = check_saved_comparison(saved_settings(extra), {}, sources)
    assert first == second and second.runnable


def test_new_page_writes_nothing_and_offers_the_earlier_study_types(env):
    at = _open("new_funded")
    assert not env["roots"]["draft_root"].exists()  # opening never saves
    assert at.selectbox(key="ifvg_lab_v1_setup_baseline").value == fs.NAMED
    assert "Default for new studies. Built from that configuration's saved identity, not " \
           "retyped." in _text(at)
    at.button(key="ifvg_lab_v1_other_types").click().run()
    assert not at.exception, at.exception
    assert at.session_state["ifvg_workspace_screen"] == "new"
    assert any(r.label == "What would you like to research?" for r in at.radio)
    assert not env["roots"]["draft_root"].exists() and env["called"] == []


def test_save_draft_creates_it_and_reopening_never_rewrites_it(env):
    at = _open("new_funded")
    at.selectbox(key="ifvg_lab_v1_setup_baseline").set_value(fs.LEGACY).run()
    assert "This baseline doesn't follow the 3:55 PM close." in _text(at)
    assert not env["roots"]["draft_root"].exists()  # an unsaved page keeps edits in the session
    at.button(key="ifvg_lab_v1_setup_save").click().run()
    assert not at.exception, at.exception
    (path,) = env["roots"]["draft_root"].glob("*/draft.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    review = saved["steps"]["review"]
    assert review["funded_comparison"]["plan_kind"] == "variations"
    assert review["funded_comparison"]["variation"]["base"] == "S0_D80_W1_P1"
    assert review[fs.REDESIGN_KEY]["baseline"] == fs.LEGACY
    before = _digest(path)
    again = _open("new_funded", path.parent.name)
    again.run()
    assert _digest(path) == before  # opening and refreshing never write
    assert again.selectbox(key="ifvg_lab_v1_setup_baseline").value == fs.LEGACY
    review_page = _open("approve_funded", path.parent.name)
    review_page.run()
    assert _digest(path) == before
    assert "isn't a configuration of the verified strategy study" in _text(review_page)
    assert review_page.button(key="ifvg_lab_v1_approve_run").disabled
    assert env["called"] == []


def test_an_edit_on_a_saved_draft_saves_only_that_edit(env):
    path = save_draft(env["roots"]["draft_root"], _variation_draft(
        env["source"], {"enabled_entry_sessions": [S0]}))
    at = _open("new_funded", path.parent.name)
    assert _digest(path) == _digest(path)
    before = json.loads(path.read_text(encoding="utf-8"))
    at.selectbox(key="ifvg_lab_v1_setup_baseline").set_value(fs.LEGACY).run()
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["steps"]["review"][fs.REDESIGN_KEY]["baseline"] == fs.LEGACY
    assert after["steps"]["review"]["funded_comparison"]["variation"]["selections"][
        "enabled_entry_sessions"] == before["steps"]["review"]["funded_comparison"][
        "variation"]["selections"]["enabled_entry_sessions"]
    assert env["called"] == []


@pytest.mark.skipif(HALF_EXIT_ENGINE, reason="needs the pinned engine (no exit-rule setting)")
def test_half_exit_draft_opens_read_only_and_blocked_and_is_never_rewritten(env):
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"]))
    before = _digest(path)
    setup = _open("new_funded", path.parent.name)
    setup.run()
    assert _digest(path) == before
    warning = " ".join(str(w.value) for w in setup.warning)
    # the same count and engine words as My studies and Review and approve (fix F4)
    assert ("This study contains 64 configurations on the half-exit engine and needs the "
            "version that supports half exits.") in warning
    assert "Strategy-Core" not in warning
    keys = {getattr(w, "key", None) for w in (*setup.button, *setup.selectbox, *setup.date_input)}
    assert not {"ifvg_lab_v1_setup_continue", "ifvg_lab_v1_setup_save",
                "ifvg_lab_v1_setup_baseline"} & keys
    assert "ifvg_lab_v1_other_types" in keys
    review = _open("approve_funded", path.parent.name)
    review.run()
    assert _digest(path) == before
    text = _text(review)
    assert ("This study contains 64 configurations on the half-exit engine and needs the "
            "version that supports half exits.") in text
    assert "Strategy-Core" not in text
    assert "Your saved settings haven't been changed. Approval and launch are off until the " \
           "study is opened with that version. The plan below is shown exactly as saved." in text
    assert review.checkbox(key="ifvg_lab_v1_approve_agree").disabled
    assert review.button(key="ifvg_lab_v1_approve_run").disabled
    assert all(t.disabled for t in review.text_input)  # never saved from this engine
    assert "Showing 8 of 64" in text
    assert env["called"] == []


def test_review_saves_an_edited_check_as_typed_and_flags_it(env):
    whole = {k: v for k, v in VARIATION_SELECTIONS.items() if k != "exit_policy"}
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"], whole))
    at = _open("approve_funded", path.parent.name)
    assert at.text_input(key="ifvg_lab_v1_approve_gate_min_days").value == "50"
    assert "OK against 107 trading days" in _text(at)
    # correction A11: a neutral pending decision, no "the data points to 18–20" advice
    assert "Needs your decision. The saved limit is 3 trading days under water; it stays as " \
           "saved, and this page doesn't propose another value." in _text(at)
    assert "18–20" not in _text(at)
    at.text_input(key="ifvg_lab_v1_approve_gate_min_days").set_value("2050").run()
    assert not at.exception, at.exception
    saved = json.loads(path.read_text(encoding="utf-8"))["steps"]["review"][fs.REDESIGN_KEY]
    assert saved["gates"] == {"min_days": 2050}  # kept exactly as typed
    assert "Can't be met: only 107 trading days are selected. Kept as saved, not trimmed — " \
           "change it or the dates." in _text(at)
    assert at.button(key="ifvg_lab_v1_approve_run").disabled
    before = _digest(path)
    at.text_input(key="ifvg_lab_v1_approve_gate_min_trades").set_value("sixty").run()
    assert _digest(path) == before  # an invalid value is never saved
    assert "Nothing was saved." in _text(at)
    assert env["called"] == []


def test_a_draft_saved_in_another_window_resets_the_review_page(env):
    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft

    whole = {k: v for k, v in VARIATION_SELECTIONS.items() if k != "exit_policy"}
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"], whole))
    at = _open("approve_funded", path.parent.name)
    other = load_draft(env["roots"]["draft_root"], path.parent.name)
    other.steps["review"][fs.REDESIGN_KEY] = {"gates": {"min_trades": 70}}
    save_draft(env["roots"]["draft_root"], other)
    saved_bytes = path.read_bytes()
    at.run()
    assert "This draft was saved from another window; showing its saved settings." in _text(at)
    assert at.text_input(key="ifvg_lab_v1_approve_gate_min_trades").value == "70"
    assert path.read_bytes() == saved_bytes  # the other window's save is kept
    assert env["called"] == []


def test_a_stale_page_never_reaches_approval_or_launch(env):
    import streamlit as st
    from ifvg_lab_new_funded import REVIEW, _record_and_run

    whole = {k: v for k, v in VARIATION_SELECTIONS.items() if k != "exit_policy"}
    draft = _variation_draft(env["source"], whole)
    save_draft(env["roots"]["draft_root"], draft)

    class State:
        pass

    state = State()
    state.draft = draft
    state.envelope = object()
    st.session_state[REVIEW + "marker"] = [draft.draft_id, "a digest of an older save"]
    errors: list[str] = []

    class Fake:
        session_state = st.session_state

        def error(self, text):
            errors.append(text)

    _record_and_run(Fake(), env["roots"], state)
    assert errors and "out of date" in errors[0] and "Nothing was approved or launched" in errors[0]
    assert env["called"] == []


def _wired(env, monkeypatch):
    """The gated path with every writer replaced by a recorder (no store, nothing launched).

    ``_freeze_and_launch`` is replaced by one that only asks the REAL, read-only
    ``dispatch_problem`` whether a launch would be allowed; ``_spawn`` stays forbidden.
    """

    import ifvg_funded_comparison_study as screen

    from alpha_lab.propsim.funded import core_identity

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(screen, "save_plan", lambda store, envelope: calls.append(
        ("save_plan", envelope.funded_comparison_plan_id)))
    monkeypatch.setattr(screen, "record_owner_approval", lambda store, plan_id, **kw: calls.append(
        ("record_owner_approval", plan_id)))

    def launch(st_module, draft, envelope, roots):
        calls.append(("_freeze_and_launch", screen.dispatch_problem(roots, draft.draft_id,
                                                                    envelope)))

    monkeypatch.setattr(screen, "_freeze_and_launch", launch)
    return calls


def _fake_page(draft_id: str, digest: str):
    import streamlit as st
    from ifvg_lab_new_funded import REVIEW

    st.session_state[REVIEW + "marker"] = [draft_id, digest]
    errors: list[str] = []

    class Fake:
        session_state = st.session_state

        def error(self, text):
            errors.append(text)

    return Fake(), errors


def test_record_and_run_goes_only_through_the_existing_gated_path(env, monkeypatch):
    from ifvg_lab_new_funded import _record_and_run, _review_state

    calls = _wired(env, monkeypatch)
    whole = {"enabled_entry_sessions": [S0, ALL_HOURS]}
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"], whole))
    draft_id = path.parent.name
    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft

    sources = {env["source"].package.run_id: env["source"]}
    state = _review_state(env["roots"], load_draft(env["roots"]["draft_root"], draft_id), sources)
    assert state.envelope is not None and not state.build_problem
    page, errors = _fake_page(draft_id, _digest(path))
    _record_and_run(page, env["roots"], state)
    plan_id = state.envelope.funded_comparison_plan_id
    assert errors == []
    assert [name for name, _ in calls] == ["save_plan", "record_owner_approval",
                                           "_freeze_and_launch"]
    assert calls[0][1] == calls[1][1] == plan_id  # the exact plan shown is the one approved
    # the approval recorder stored nothing, so the real launch gate still refuses
    assert calls[2][1] == "This exact plan has no recorded approval. Nothing was launched."
    assert env["called"] == []  # _spawn never ran


def test_record_and_run_refuses_a_saved_setting_no_plan_can_carry(env, monkeypatch):
    from ifvg_lab_new_funded import _record_and_run, _review_state

    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft

    calls = _wired(env, monkeypatch)
    whole = {"enabled_entry_sessions": [S0, ALL_HOURS]}
    path = save_draft(env["roots"]["draft_root"], _variation_draft(
        env["source"], whole, redesign={"withdrawal_triggers_usd": [500, 1000]}))
    draft_id = path.parent.name
    sources = {env["source"].package.run_id: env["source"]}
    state = _review_state(env["roots"], load_draft(env["roots"]["draft_root"], draft_id), sources)
    before = _digest(path)
    page, errors = _fake_page(draft_id, before)
    _record_and_run(page, env["roots"], state)
    assert errors and TRIGGER_TEXT in errors[0] and errors[0].endswith(
        "Nothing was approved or launched.")
    assert calls == [] and env["called"] == []
    assert _digest(path) == before


def test_review_without_a_saved_draft_says_so(env):
    at = _open("approve_funded")
    assert "Nothing to review yet" in _text(at)
    assert not env["roots"]["draft_root"].exists()


def test_the_saved_64_configurations_are_listed_without_the_half_exit_engine(env):
    rows = fs.saved_strategy_rows(env["source"], {"base": "S0_D80_W1_P1", "selections":
                                                  VARIATION_SELECTIONS})
    assert len(rows) == 64
    assert sum(r["exit_policy"] == fs.HALF_EXIT for r in rows) == 16
    assert {r["tp_r_multiple"] for r in rows if r["exit_policy"] == fs.HALF_EXIT} == {
        "tp_r_multiple.1.0"}


def test_a_study_kind_draft_continued_here_opens_on_the_earlier_configurator(env):
    from alpha_lab.propsim.funded.comparison_study import axis_choices

    draft = new_draft(fs.COMPARISON_MODE, display_name="Study draft")
    draft.steps = {
        "objective": {"mode_id": fs.COMPARISON_MODE, "question_id": "funded_configuration_cash"},
        "search_space": {"mode_id": fs.COMPARISON_MODE,
                         "axis_selections": axis_choices(env["source"])},
        "review": {"funded_comparison": {
            "source_run_id": env["source"].package.run_id,
            "firm_keys": ["takeprofittrader", "myfundedfutures"], "instrument": "mini",
            "quantity": 1, "cost_per_side_cents": 514, "processing": "two_business_days"}},
    }
    path = save_draft(env["roots"]["draft_root"], draft)
    before = _digest(path)
    at = _open("new_funded", draft.draft_id)  # where My studies' "Continue draft" goes
    at.run()
    assert _digest(path) == before
    assert "It opens on the earlier configurator below" in _text(at)
    assert at.number_input(key="ifvg_fcmp_quantity").value == 1
    assert env["called"] == []


FAKE_CORE = {"root": "research-core", "base_commit": "7c7111e398c083cf8e966e2e0c5aac8a41cc12c0",
             "branch": "funded-scale-out-exit", "patch_sha256": "1" * 64}


def test_approval_opens_only_with_nothing_blocking_and_the_owners_tick(env, monkeypatch):
    from alpha_lab.propsim.funded import core_identity

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    whole = {"enabled_entry_sessions": [S0, ALL_HOURS],
             "tp_r_multiple": ["tp_r_multiple.1.0", "tp_r_multiple.3.0"]}
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"], whole))
    at = _open("approve_funded", path.parent.name)
    text = _text(at)
    assert "Approval is off" not in text
    assert at.checkbox(key="ifvg_lab_v1_approve_agree").label == (
        "I approve running this exact plan: 4 configurations, both firms, the dates above. "
        "No live trading.")
    assert not at.checkbox(key="ifvg_lab_v1_approve_agree").disabled
    assert at.button(key="ifvg_lab_v1_approve_run").disabled  # not before the owner's tick
    at.checkbox(key="ifvg_lab_v1_approve_agree").check().run()
    assert not at.button(key="ifvg_lab_v1_approve_run").disabled  # enabled, never clicked here
    # a withdrawal-trigger comparison saved in the same draft closes it again
    other = _variation_draft(env["source"], whole, redesign={
        "withdrawal_triggers_usd": [500, 1000]})
    path2 = save_draft(env["roots"]["draft_root"], other)
    blocked = _open("approve_funded", path2.parent.name)
    assert TRIGGER_TEXT in _text(blocked)
    assert blocked.checkbox(key="ifvg_lab_v1_approve_agree").disabled
    assert blocked.button(key="ifvg_lab_v1_approve_run").disabled
    assert env["called"] == []


@pytest.mark.skipif(not HALF_EXIT_ENGINE, reason="needs the research engine with the half exit")
def test_research_engine_opens_the_half_exit_draft_editable_and_reviews_all_64(env):
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"]))
    before = _digest(path)
    setup = _open("new_funded", path.parent.name)
    setup.run()
    assert _digest(path) == before  # editable, but opening still writes nothing
    text = _text(setup)
    assert "64 configurations" in text and "128 separate results" in text
    assert "32 combinations left out" in text
    assert "Needs the engine version that supports half exits. Checked at approval and again " \
           "at launch." in text
    assert not setup.button(key="ifvg_lab_v1_setup_continue").disabled
    review = _open("approve_funded", path.parent.name)
    assert _digest(path) == before
    assert "needs the version that supports half exits" not in _text(review)
    assert "Showing 8 of 64" in _text(review)
    assert review.checkbox(key="ifvg_lab_v1_approve_agree").label.startswith(
        "I approve running this exact plan: 64 configurations, both firms")
    assert review.button(key="ifvg_lab_v1_approve_run").disabled  # never clicked here
    assert env["called"] == []


def test_a_whole_position_only_exit_is_not_written_so_both_engines_can_edit_it(env):
    from alpha_lab.propsim.funded.comparison_draft import (
        check_saved_comparison,
        saved_settings,
        variation_selections,
    )

    draft = _variation_draft(env["source"], {"enabled_entry_sessions": [S0]})
    offered = {"enabled_entry_sessions": (S0, ALL_HOURS),
               "exit_policy": (fs.WHOLE_EXIT, fs.HALF_EXIT)}  # as the research engine offers
    run_id = env["source"].package.run_id
    choices = _choices(source_run_id=run_id,
                       selections={**_choices().selections, "exit_policy": (fs.WHOLE_EXIT,)})
    fs.write_choices(draft, choices, offered=offered)
    saved = saved_settings(draft)
    assert "exit_policy" not in saved["variation"]["selections"]
    sources = {env["source"].package.run_id: env["source"]}
    assert check_saved_comparison(saved, {}, sources).runnable  # this (pinned) engine edits it
    if HALF_EXIT_ENGINE:  # the research engine still builds the whole-position exit
        assert variation_selections(env["source"], saved["variation"])["exit_policy"] == [
            fs.WHOLE_EXIT]
    half = _choices(source_run_id=run_id, selections={**_choices().selections,
                                "exit_policy": (fs.WHOLE_EXIT, fs.HALF_EXIT)})
    fs.write_choices(draft, half, offered=offered)
    assert saved_settings(draft)["variation"]["selections"]["exit_policy"] == [
        fs.WHOLE_EXIT, fs.HALF_EXIT]


# ── review findings: the earlier configurator, session keys, draft links ──


DATES_LEAD = "A funded comparison replays its source study's saved dates"


def _fake_source():
    """A source whose saved dates are the calendar's trading days (every day file present)."""

    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.research_period import resolve_research_range

    days = resolve_research_range("2026-01-13", "2026-06-10",
                                  day_has_data=lambda _day: True).trading_days
    return SimpleNamespace(package=SimpleNamespace(run_id="run", root=None),
                           by_name={"S0_D80_W1_P1": SimpleNamespace(axis_value_ids=S0_IDS)},
                           evaluation_dates=days)


def test_saved_draft_blockers_read_only_the_redesign_key_of_the_saved_draft():
    source = _fake_source()

    def blockers(redesign=None, named_problem=None):
        draft = _variation_draft(source, {"enabled_entry_sessions": [S0]}, redesign=redesign)
        return fs.saved_draft_blockers(draft, source, day_has_data=lambda _day: True,
                                       named_problem=named_problem)

    assert blockers() == []  # no redesign key: exactly as before
    assert blockers({}) == []  # the key with today's behavior
    assert blockers({"withdrawal_triggers_usd": [500]}) == []
    assert [(b.key, b.text) for b in blockers({"withdrawal_triggers_usd": [500, 1000]})] == [
        ("withdrawal_trigger", TRIGGER_TEXT)]
    (dates,) = blockers({"dates": {"start": "2026-03-02", "end": "2026-06-10"}})
    assert dates.key == "dates" and dates.text.startswith(DATES_LEAD)
    assert [b.key for b in blockers({"baseline": fs.LEGACY})] == ["baseline"]
    assert [b.key for b in blockers({"gap_rules": [fs.GAP_CLOSE, fs.GAP_WICK]})] == ["gap_rule"]
    assert [b.key for b in blockers({"gates": {"min_trades": 70}})] == ["gates_changed"]
    assert [b.text for b in blockers({}, named_problem="Not offered here.")] == [
        "Not offered here."]
    # a saved value that can't be read refuses, never allows
    assert [b.text for b in blockers({"withdrawal_triggers_usd": ["lots"]})] == [
        fs.UNREADABLE_TEXT]


def test_saved_draft_blockers_are_the_review_pages_list(env, monkeypatch):
    from ifvg_lab_new_funded import _review_blockers, _review_state

    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft
    from alpha_lab.propsim.funded import core_identity

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    roots, source = env["roots"], env["source"]
    path = save_draft(roots["draft_root"], _variation_draft(
        source, {"enabled_entry_sessions": [S0, ALL_HOURS]}, redesign={
            "withdrawal_triggers_usd": [500, 1000, 2000], "gates": {"min_trades": 70},
            "dates": {"start": "2026-03-02", "end": "2026-06-10"}}))
    draft = load_draft(roots["draft_root"], path.parent.name)
    state = _review_state(roots, draft, {source.package.run_id: source})
    assert state.build_problem is None
    rows = fs.gate_rows(state.choices.gates, state.gate_defaults,
                        trading_days=state.resolved.trading_days,
                        warmup_days=state.resolved.warmup_dates)
    shown = [b.text for b in _review_blockers(state, rows)]
    saved = [b.text for b in fs.saved_draft_blockers(
        draft, source, repo_root=roots["repo_root"], named_problem=fs.named_baseline()[1])]
    assert saved == shown
    assert TRIGGER_TEXT in saved and any(t.startswith(DATES_LEAD) for t in saved)
    assert any(t.startswith("Changed pass/fail checks") for t in saved)


def _warning(at) -> str:
    return " ".join(str(w.value) for w in at.warning).replace("\\$", "$")


def test_the_earlier_configurator_refuses_approval_and_run_for_blocking_saved_settings(
        env, monkeypatch):
    """Review finding: Setup → More → the earlier configurator bypassed Review's blockers."""

    from alpha_lab.propsim.funded import core_identity

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    whole = {"enabled_entry_sessions": [S0, ALL_HOURS]}
    cases = (({"withdrawal_triggers_usd": [500, 1000, 2000]}, TRIGGER_TEXT),
             ({"dates": {"start": "2026-03-02", "end": "2026-06-10"}}, DATES_LEAD))
    for redesign, expected in cases:
        path = save_draft(env["roots"]["draft_root"],
                          _variation_draft(env["source"], whole, redesign=redesign))
        before = _digest(path)
        at = _open("new", path.parent.name)  # where "Open the earlier configurator" goes
        assert "2 configurations around" in " ".join(str(m.value) for m in at.markdown)
        warning = _warning(at)
        assert "Approval and running are off until these are settled" in warning
        assert expected in warning
        assert at.checkbox(key="ifvg_fcmp_agree").disabled
        assert at.button(key="ifvg_fcmp_approve").disabled
        assert at.button(key="ifvg_fcmp_run").disabled
        assert _digest(path) == before  # opening writes nothing
    # the same plan saved without the redesign key: unchanged (approval offered, not clicked)
    plain = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"], whole))
    at = _open("new", plain.parent.name)
    assert "Approval and running are off" not in _warning(at)
    assert not at.checkbox(key="ifvg_fcmp_agree").disabled
    at.checkbox(key="ifvg_fcmp_agree").check().run()
    assert not at.button(key="ifvg_fcmp_approve").disabled
    assert env["called"] == []


def test_the_earlier_configurator_drops_a_click_that_arrives_while_blocked(env, monkeypatch):
    """Even a click from an old page (drawn before the block was saved) records and runs nothing."""

    import ifvg_funded_comparison_study as screen

    from alpha_lab.propsim.funded import core_identity
    from alpha_lab.propsim.funded.comparison_draft import rebuild_saved_plan, saved_settings

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    monkeypatch.setattr(screen, "find_approval", lambda store, plan_id: None)
    draft = _variation_draft(env["source"], {"enabled_entry_sessions": [S0, ALL_HOURS]},
                             redesign={"withdrawal_triggers_usd": [500, 1000]})
    save_draft(env["roots"]["draft_root"], draft)
    envelope, _, _ = rebuild_saved_plan(env["source"], saved_settings(draft), {})
    shown: list[tuple[str, str, bool]] = []

    class Page:  # every control reports a click
        session_state: dict = {}

        def info(self, text):
            shown.append(("info", text, False))

        def warning(self, text):
            shown.append(("warning", text, False))

        def success(self, text):
            shown.append(("success", text, False))

        def checkbox(self, label, **kwargs):
            shown.append(("checkbox", label, kwargs.get("disabled", False)))
            return True

        def button(self, label, **kwargs):
            shown.append(("button", label, kwargs.get("disabled", False)))
            return True

        def rerun(self):
            raise AssertionError("nothing was recorded, so nothing reruns")

    screen._approve_and_run(Page(), draft, envelope, env["roots"], "scope",
                            source=env["source"])
    assert ("button", "Record my approval", True) in shown
    assert ("button", "Run funded comparison", True) in shown
    # the reasons are shown, with dollar signs escaped for Streamlit's text
    assert any(kind == "warning" and TRIGGER_TEXT.replace("$", "\\$") in text
               for kind, text, _ in shown)
    assert env["called"] == []  # save_plan, record_owner_approval, _freeze_and_launch


def test_the_launch_check_refuses_a_draft_whose_saved_settings_block_approval(env, monkeypatch):
    """dispatch_problem: the identical plan (and an approval of it) is refused for such a draft."""

    import ifvg_funded_comparison_study as screen

    from alpha_lab.propsim.funded import core_identity
    from alpha_lab.propsim.funded.comparison_draft import rebuild_saved_plan, saved_settings

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    # an approval of the exact plan is on record (read only; nothing is written)
    monkeypatch.setattr(screen, "find_approval", lambda store, plan_id: object())
    whole = {"enabled_entry_sessions": [S0, ALL_HOURS]}
    plain = _variation_draft(env["source"], whole)
    save_draft(env["roots"]["draft_root"], plain)
    envelope, _, _ = rebuild_saved_plan(env["source"], saved_settings(plain), {})
    assert screen.dispatch_problem(env["roots"], plain.draft_id, envelope) is None  # unchanged
    for redesign, expected in (({"withdrawal_triggers_usd": [500, 1000]}, TRIGGER_TEXT),
                               ({"dates": {"start": "2026-03-02", "end": "2026-06-10"}},
                                DATES_LEAD)):
        draft = _variation_draft(env["source"], whole, redesign=redesign)
        save_draft(env["roots"]["draft_root"], draft)
        problem = screen.dispatch_problem(env["roots"], draft.draft_id, envelope)
        assert problem.startswith("This saved draft can't be launched until these are settled: ")
        assert expected in problem and problem.endswith("Nothing was launched.")
    assert env["called"] == []


TRADE_REVIEW_STATE = {"ifvg_lab_v1_review_back": {"result_id": "r", "tab": "Trades"},
                      "ifvg_lab_v1_review_source_value": "Funded trades",
                      "ifvg_lab_v1_review_current": {"trade": 380}}


def _app_or_trade_review():
    """The workspace; or, when asked, one run of what Trade review stores on its own run."""

    import ifvg_workspace
    import streamlit as st

    stored = st.session_state.pop("test_trade_review_stores", None)
    if stored:  # Trade review's run draws no Review and approve widget
        st.session_state.update(stored)
    else:
        ifvg_workspace.render_workspace(st, roots=ifvg_workspace._TEST_ROOTS)


def test_review_and_approve_leaves_trade_reviews_session_keys_alone(env):
    """Review finding: both pages used the session prefix ifvg_lab_v1_review_.

    Trade review's stored back link then pressed Review and approve's "Back to setup"
    by itself, and opening another draft deleted Trade review's state.
    """

    from streamlit.testing.v1 import AppTest

    whole = {"enabled_entry_sessions": [S0, ALL_HOURS]}
    first = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"], whole))
    second = save_draft(env["roots"]["draft_root"], _variation_draft(
        env["source"], {"enabled_entry_sessions": [S0]}))
    at = AppTest.from_function(_app_or_trade_review, default_timeout=240)
    at.session_state["ifvg_workspace_destination"] = "New study"
    at.session_state["ifvg_workspace_screen"] = "approve_funded"
    at.session_state["ifvg_study_v1_draft_id"] = first.parent.name
    at.run()  # Review and approve
    assert not at.exception, at.exception
    at.session_state["test_trade_review_stores"] = dict(TRADE_REVIEW_STATE)
    at.run()  # Trade review's run
    at.run()  # back on Review and approve, same draft
    assert not at.exception, at.exception
    assert at.session_state["ifvg_workspace_screen"] == "approve_funded"  # not moved to setup
    at.session_state["ifvg_study_v1_draft_id"] = second.parent.name  # another draft
    at.run()
    assert not at.exception, at.exception
    assert at.session_state["ifvg_workspace_screen"] == "approve_funded"
    for key, value in TRADE_REVIEW_STATE.items():
        assert at.session_state[key] == value  # Trade review's state kept
    assert env["called"] == []


def test_a_draft_link_this_application_doesnt_have_says_so_and_opens_nothing(env):
    from ifvg_lab_new_funded import UNKNOWN_DRAFT_NOTE
    from streamlit.testing.v1 import AppTest

    known = save_draft(env["roots"]["draft_root"], _variation_draft(
        env["source"], {"enabled_entry_sessions": [S0]})).parent.name

    def opened(screen, draft=None):
        at = AppTest.from_function(_app, default_timeout=240)
        at.session_state["ifvg_workspace_destination"] = "New study"
        at.session_state["ifvg_workspace_screen"] = screen
        if draft is not None:
            at.query_params["draft"] = draft
        at.run()
        assert not at.exception, at.exception
        return at

    for screen in ("new_funded", "approve_funded"):
        for unknown in ("0" * 32, "not-a-draft"):
            at = opened(screen, unknown)
            assert UNKNOWN_DRAFT_NOTE in _text(at)
            assert "ifvg_study_v1_draft_id" not in at.session_state
        at = opened(screen, known)
        assert at.session_state["ifvg_study_v1_draft_id"] == known
        assert UNKNOWN_DRAFT_NOTE not in _text(at)
        at = opened(screen)  # no draft in the page address: nothing opened, no note
        assert "ifvg_study_v1_draft_id" not in at.session_state
        assert UNKNOWN_DRAFT_NOTE not in _text(at)
    assert env["called"] == []


# ── fix F11: both approval paths accept and refuse the same saved drafts ──


def _review_accepts(env, draft, sources) -> bool:
    """Review and approve: the approval is offered only with no blocker (never clicked)."""

    import ifvg_lab_new_funded as nf

    state = nf._review_state(env["roots"], draft, sources)
    if not state.check.runnable or state.envelope is None:
        return False
    days = state.resolved.trading_days if state.resolved is not None else ()
    warmup = state.resolved.warmup_dates if state.resolved is not None else ()
    rows = fs.gate_rows(state.choices.gates, state.gate_defaults, trading_days=days,
                        warmup_days=warmup)
    return not nf._review_blockers(state, rows)


def _earlier_accepts(env, draft, sources) -> bool:
    """The earlier configurator: "Record my approval" is enabled once the box is ticked."""

    import ifvg_funded_comparison_study as screen

    from alpha_lab.propsim.funded.comparison_draft import (
        check_saved_comparison,
        rebuild_saved_plan,
        saved_settings,
    )

    if not check_saved_comparison(saved_settings(draft), {}, sources).runnable:
        return False  # it opens read only: no approval control at all
    try:
        envelope, _skipped, _unavailable = rebuild_saved_plan(env["source"],
                                                              saved_settings(draft), {})
    except Exception:
        return False
    if envelope is None:
        return False
    shown: list[tuple[str, str, bool]] = []

    class Page:
        session_state: dict = {}

        def info(self, text):
            shown.append(("info", text, False))

        def warning(self, text):
            shown.append(("warning", text, False))

        def success(self, text):
            shown.append(("success", text, False))

        def checkbox(self, label, **kwargs):
            shown.append(("checkbox", label, kwargs.get("disabled", False)))
            return True

        def button(self, label, **kwargs):  # never clicked
            shown.append(("button", label, kwargs.get("disabled", False)))
            return False

    screen._approve_and_run(Page(), draft, envelope, env["roots"], "scope",
                            source=env["source"])
    return ("button", "Record my approval", False) in shown


def approval_case_specs(own_gap: str, other_gap: str, today: int | None
                        ) -> dict[str, tuple[dict, dict | None, list[str] | None]]:
    """The ten saved drafts both approval paths are checked on (fix F11; the A11 matrix).

    ``name -> (variation selections, New funded comparison settings or None, firms or None)``;
    ``None`` firms keep the draft's two known firms. ``own_gap`` is the starting
    configuration's own gap rule and ``today`` today's withdrawal trigger.
    """

    hours = {"enabled_entry_sessions": [S0, ALL_HOURS]}
    same = {"baseline": "named", "gap_rules": [own_gap], "withdrawal_triggers_usd": [today]}
    return {
        "no New funded comparison settings": (hours, None, None),
        "its settings at today's behavior": (hours, dict(same), None),
        "other withdrawal triggers": (hours, {**same, "withdrawal_triggers_usd": [today, 1000]},
                                      None),
        "another gap rule": (hours, {**same, "gap_rules": [own_gap, other_gap]}, None),
        "other dates": (hours, {**same, "dates": {"start": "2026-02-02", "end": "2026-06-10"}},
                        None),
        "a changed pass/fail check": (hours, {**same, "gates": {"min_trades": 75}}, None),
        "an impossible pass/fail check": (hours, {**same, "gates": {"min_days": 2050}}, None),
        "the legacy baseline": (hours, {**same, "baseline": "legacy"}, None),
        "the half exit": (VARIATION_SELECTIONS, dict(same), None),
        "an unknown firm": (hours, dict(same), ["takeprofittrader", "nofirm"]),
    }


def approval_case_draft(source, selections, redesign, firm_keys=None):
    """One of the ten drafts, built exactly as the F11 test saves it (never saved here)."""

    draft = _variation_draft(source, selections, redesign=redesign)
    if firm_keys is not None:
        draft.steps["review"]["funded_comparison"]["firm_keys"] = list(firm_keys)
    return draft


def test_both_approval_paths_accept_and_refuse_the_same_drafts(env, monkeypatch):
    import ifvg_funded_comparison_study as screen

    from alpha_lab.propsim.funded import core_identity

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    monkeypatch.setattr(screen, "find_approval", lambda store, plan_id: None)
    source = env["source"]
    sources = {source.package.run_id: source}
    base_ids = dict(source.by_name["S0_D80_W1_P1"].axis_value_ids)
    own_gap = fs.base_value(fs.GAP_AXIS, base_ids)
    other_gap = fs.GAP_WICK if own_gap != fs.GAP_WICK else fs.GAP_CLOSE
    today = fs.today_trigger_usd()
    results = {}
    for name, (selections, redesign, firm_keys) in approval_case_specs(
            own_gap, other_gap, today).items():
        draft = approval_case_draft(source, selections, redesign, firm_keys)
        save_draft(env["roots"]["draft_root"], draft)
        results[name] = (_review_accepts(env, draft, sources),
                         _earlier_accepts(env, draft, sources))
    assert len(results) == 10
    differ = {name: pair for name, pair in results.items() if pair[0] != pair[1]}
    assert differ == {}, differ
    accepted = {name for name, (review, _earlier) in results.items() if review}
    expected = {"no New funded comparison settings", "its settings at today's behavior"}
    if HALF_EXIT_ENGINE:
        expected.add("the half exit")
    assert accepted == expected
    assert env["called"] == []  # nothing saved, approved or launched


# ── theme: every color goes through the two-theme palette ────────────────


COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{3,8}\b|\brgba?\(|\bhsla?\(")
FONT_VARS = {"--lab-sans", "--lab-serif", "--lab-mono"}


def _html_bodies(at) -> list[str]:
    return [getattr(getattr(e, "proto", None), "body", "") or "" for e in at.get("html")]


def test_the_pages_stylesheet_names_palette_variables_defined_for_both_themes():
    import ifvg_lab_new_funded as nf

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    css = nf._CSS + " " + nf._icon_css()
    assert not COLOR_LITERAL.search(css), COLOR_LITERAL.search(css).group(0)
    used = set(re.findall(r"var\((--lab-[a-z0-9-]+)\)", css)) - FONT_VARS
    assert {"--lab-ink", "--lab-panel", "--lab-orange", "--lab-on-ink"} <= used
    light, dark = theme.theme_css_vars("light"), theme.theme_css_vars("dark")
    for name in sorted(used):
        assert f"{name}:" in light and f"{name}:" in dark, name
    # the light palette is the mocks' one; the dark one declares another value for each name
    for name in ("--lab-ink", "--lab-panel", "--lab-orange", "--lab-on-ink"):
        light_value = re.search(rf"{name}:([^;]+);", light).group(1)
        dark_value = re.search(rf"{name}:([^;]+);", dark).group(1)
        assert light_value != dark_value, name
    # the current step is ink with the text drawn for an ink fill, readable on either theme
    assert ".fs-step.on{background:var(--lab-ink);color:var(--lab-on-ink) !important;" in nf._CSS
    # the alert icon is a mask filled from the palette's orange; its stroke is only the shape
    assert "background-color:var(--lab-orange)" in nf._CSS
    assert "mask:var(--fs-icon) no-repeat center / contain" in nf._CSS
    icon_css = nf._icon_css()
    assert '.fs-icon.lock{--fs-icon:url("data:image/svg+xml,' in icon_css
    assert '.fs-icon.triangle{--fs-icon:url("data:image/svg+xml,' in icon_css
    assert "A34A12" not in icon_css.upper() and "background-image" not in icon_css
    with_dark = theme.set_theme_resolver
    try:
        with_dark(lambda: "dark")
        # the stylesheet is the same text on both themes: only the variables' values change
        assert nf._CSS + " " + nf._icon_css() == css
        assert theme.palette()["orange"] == theme.DARK_COLORS["orange"]
    finally:
        with_dark(None)


def test_the_rendered_pages_inline_styles_carry_palette_variables_never_values(env):
    import ifvg_lab_new_funded as nf

    at = _open("new_funded")
    own = [b for b in _html_bodies(at)
           if ".fs-title{" in b or "fs-plan-lines" in b
           or nf.NAMED_NOTE.split(".")[0] in b]  # the note (its apostrophe is escaped)
    assert len(own) >= 3, [b[:80] for b in own]
    joined = "\n".join(own)
    assert not COLOR_LITERAL.search(joined), COLOR_LITERAL.search(joined).group(0)
    assert 'style="font-size:14px;color:var(--lab-body-2);' in joined  # the baseline note
    assert "border-top:1px solid var(--lab-light-rule)" in joined  # the plan panel's divider
    whole = {k: v for k, v in VARIATION_SELECTIONS.items() if k != "exit_policy"}
    path = save_draft(env["roots"]["draft_root"], _variation_draft(env["source"], whole))
    at = _open("approve_funded", path.parent.name)
    flagged = [b for b in _html_bodies(at) if "border:2px solid" in b]
    assert flagged and all("var(--lab-orange) !important" in b for b in flagged)
    assert not any(COLOR_LITERAL.search(b) for b in flagged)
    assert env["called"] == []
