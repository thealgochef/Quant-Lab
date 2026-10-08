"""My studies (mock 01): rows from BOTH applications' stores, two action labels only,
leader net cash per firm (never added together) and the earlier-method tag.

Every store here is a ``tmp_path`` copy; nothing is launched and no real store is read
or written.
"""

from __future__ import annotations

import contextlib
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.lab import library as lib  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import (  # noqa: E402
    StudySummary,
    load_studies,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft, save_draft  # noqa: E402

TPT, MFF = "takeprofittrader", "myfundedfutures"


def test_full_range_plan_names_partial_and_whole_position_families():
    plan = SimpleNamespace(
        plan_schema="ifsm_correct_config_full_range_plan_v1",
        configurations=tuple(SimpleNamespace(exit_policy=(
            "scale_out_half_breakeven_hold_to_close_v1" if index < 2 else "fixed_target_v1"))
            for index in range(6)),
    )
    text = lib.plan_tested_line(plan)
    assert "6 configurations" in text
    assert "2 half-exit and 4 whole-position exits" in text
    assert "ten micros" in text and "one mini" in text


def _summary(configuration, firm_key, firm, net, rank, status="Completed"):
    return {"configuration": configuration, "firm_key": firm_key, "firm": firm,
            "status": status, "net_cash_earned_cents": net, "rank_within_firm": rank,
            "pair_id": f"{configuration}|{firm_key}"}


def _comparison_result():
    """Two configurations at two firms; the leader differs by firm."""

    rows = [_summary("A", TPT, "TakeProfitTrader", 3078188, 1),
            _summary("B", TPT, "TakeProfitTrader", 2100000, 2),
            _summary("A", MFF, "MyFundedFutures", 1500000, 2),
            _summary("B", MFF, "MyFundedFutures", 3481811, 1)]
    return {"summaries_cents": {r["pair_id"]: r for r in rows},
            "settings": {"firm_profiles": [{"firm_key": TPT, "firm_name": "TakeProfitTrader"},
                                           {"firm_key": MFF, "firm_name": "MyFundedFutures"}]},
            "tables": {"configurations": [{"configuration": "A"}, {"configuration": "B"}]}}


def _study(key, kind, *, status="Completed", state=None, name="Study", archived=False,
           draft=None, dates="January 13, 2026 – June 10, 2026", scope="research",
           updated="2026-09-23T17:04:41+00:00"):
    return StudySummary(key=key, kind=kind, name=name, question="A question?", dates=dates,
                        status=status, scope=scope, updated=updated, archived=archived,
                        draft=draft, state=state)


@pytest.fixture
def repo(tmp_path):
    from ifvg_lab_nav import app_roots

    both = app_roots(tmp_path)
    for roots in both.values():
        Path(roots["draft_root"]).mkdir(parents=True, exist_ok=True)
    return tmp_path, both


# ── leaders per firm ──────────────────────────────────────────────────────


def test_leader_is_the_rank_one_net_cash_at_each_firm_never_a_total():
    leaders = lib.funded_leaders(_comparison_result())
    assert [(x.firm_key, x.firm, x.net_cash_cents, x.configuration) for x in leaders] == [
        (TPT, "TakeProfitTrader", 3078188, "A"), (MFF, "MyFundedFutures", 3481811, "B")]
    assert all(x.net_cash_cents != 3078188 + 3481811 for x in leaders)


def test_a_firm_without_a_completed_configuration_has_no_value():
    result = _comparison_result()
    for row in result["summaries_cents"].values():
        if row["firm_key"] == MFF:
            row["status"] = "Failed"
    leaders = {x.firm_key: x for x in lib.funded_leaders(result)}
    assert leaders[TPT].net_cash_cents == 3078188
    assert leaders[MFF].net_cash_cents is None


def test_earlier_five_account_result_gives_one_value_per_firm_in_saved_order():
    result = {"summaries_cents": {
        MFF: {"firm": "MyFundedFutures", "net_cash_earned_cents": 7611060},
        TPT: {"firm": "TakeProfitTrader", "net_cash_earned_cents": 6806390}},
        "tables": {"instance_results": [{"firm_key": TPT}, {"firm_key": MFF}]}}
    leaders = lib.earlier_leaders(result)
    assert [(x.firm, x.net_cash_cents) for x in leaders] == [
        ("TakeProfitTrader", 6806390), ("MyFundedFutures", 7611060)]


# ── rows, tags and the two action labels ──────────────────────────────────


def test_earlier_method_rows_are_tagged_and_grey_candidates():
    earlier = _study("p0", "funded", state={"result_id": "r0", "plan_id": "p0",
                                            "completed_at_utc": "2026-09-23T03:33:48+00:00"},
                     name="Funded payout simulation — 2026-09-22")
    row = lib.study_row(earlier, "ifsm", current="ifsm",
                        leaders=lib.earlier_leaders({"summaries_cents": {
                            TPT: {"firm": "TakeProfitTrader", "net_cash_earned_cents": 1}}}))
    assert row.earlier_method and row.kind_label == lib.EARLIER_METHOD
    assert row.tab == "funded" and row.action == lib.OPEN_RESULTS
    assert row.name == "Funded payout simulation"
    assert row.status_date == "September 22, 2026"  # 3:33 AM UTC is the evening before in Chicago
    assert row.tested.startswith("Five accounts copying one signal")
    current = _study("p1", "funded_comparison", state={"result_id": "r1", "plan_id": "p1"},
                     name="Funded variation study — 64 configurations")
    assert not lib.study_row(current, "ifsm", current="ifsm").earlier_method


def test_only_two_action_labels_exist_and_each_row_goes_to_one_place():
    comparison = new_draft(lib.COMPARISON_MODE, display_name="Screen check")
    strategy = new_draft("single_configuration", display_name="Evaluate")
    studies = [
        _study("p1", "funded_comparison", state={"result_id": "r1", "plan_id": "p1"}),
        _study("p2", "funded_comparison", status="Running", state={"plan_id": "p2"}),
        _study("p0", "funded", state={"result_id": "r0"}),
        _study("s1", "search"), _study("c1", "context"), _study("x1", "pipeline"),
        _study(comparison.draft_id, "draft", status="Draft", draft=comparison),
        _study(strategy.draft_id, "draft", status="Draft", draft=strategy),
        _study("old", "search", archived=True),
    ]
    rows = [lib.study_row(s, "main", current="main") for s in studies]
    rows.append(lib.group_row({"group_id": "g1", "display_name": "Model", "status": "completed",
                               "cells": [{"label": "Parent timeout 240", "lane": "R5"}],
                               "study_spec": {"evaluation_start": "2026-02-23",
                                              "evaluation_end": "2026-06-10"}},
                              "main", current="main"))
    assert {r.action for r in rows} == {lib.OPEN_RESULTS, lib.CONTINUE_DRAFT}
    routes = {r.key: (r.action, r.route) for r in rows}
    assert routes["p1"] == (lib.OPEN_RESULTS, "funded_results")
    assert routes["p2"] == (lib.OPEN_RESULTS, "detail")  # progress on the study page
    assert routes["p0"] == (lib.OPEN_RESULTS, "detail")
    assert routes["s1"] == (lib.OPEN_RESULTS, "detail")
    assert routes["g1"] == (lib.OPEN_RESULTS, "research_group")
    assert routes[comparison.draft_id] == (lib.CONTINUE_DRAFT, "new_funded")
    assert routes[strategy.draft_id] == (lib.CONTINUE_DRAFT, "detail")
    by_key = {r.key: r for r in rows}
    assert by_key[comparison.draft_id].details_route == "detail"  # clone / rename / archive
    counts = lib.counts(rows)
    assert counts == {"funded": 3, "strategy": 3, "model": 1, "drafts": 2, "archived": 1}
    assert [r.key for r in lib.rows_in_tab(rows, "funded")] == [
        "p1", "p2", "p0", comparison.draft_id]
    assert [r.key for r in lib.rows_in_tab(rows, "archived")] == ["old"]
    assert lib.tab_label("funded", 3) == "Funded comparisons · 3"
    assert lib.tab_label("strategy", 3) == "Strategy studies"
    assert lib.tab_label("archived", 1) == "Archived"


def test_other_application_rows_open_funded_results_read_only_or_name_the_command():
    funded = _study("p1", "funded_comparison", state={"result_id": "r1", "plan_id": "p1"})
    search = _study("s1", "search")
    group = {"group_id": "g1", "display_name": "Model", "status": "completed", "cells": []}
    rows = [lib.study_row(funded, "ifsm", current="main"),
            lib.study_row(search, "ifsm", current="main"),
            lib.group_row(group, "ifsm", current="main")]
    assert rows[0].route == "funded_results" and not rows[0].elsewhere
    assert rows[1].route == "" and rows[1].elsewhere == (
        "Open it in the dedicated app: python scripts/run_ifsm_research_ui.py")
    assert rows[2].route == "" and "dedicated app" in rows[2].elsewhere
    back = lib.study_row(search, "main", current="ifsm")
    assert back.elsewhere == "Open it in the main app: streamlit run scripts/dashboard.py"


def test_draft_notes_come_from_real_checks():
    half = SimpleNamespace(needs_half_exit_engine=True, problems=("needs half",),
                           configuration_count=64, count_is_exact=True)
    assert lib.comparison_draft_note(half) == (
        "Needs the half-exit engine to run. Your saved settings are unchanged.", "orange")
    other = SimpleNamespace(needs_half_exit_engine=False, problems=("Unknown firm.",))
    assert lib.comparison_draft_note(other)[0].startswith("Can't run in this application: "
                                                          "Unknown firm.")
    fine = SimpleNamespace(needs_half_exit_engine=False, problems=())
    assert lib.comparison_draft_note(fine)[1] == ""
    assert lib.draft_note("Draft", "research", "Dates not selected") == (
        "Dates not chosen yet.", "")
    assert lib.draft_note("Draft", "unresolved", "Dates not selected")[1] == "orange"
    note, _ = lib.draft_note("Draft", "research", "2026-01-13 to 2026-06-10 · 107 days")
    assert note == "January 13 – June 10, 2026 · 107 trading days. Not run yet."
    draft = new_draft(lib.COMPARISON_MODE, display_name="Screen check")
    assert lib.draft_summary(draft, half) == ("64 configurations on the half-exit engine · both "
                                              "firms")


def test_dates_search_and_per_firm_sorting():
    assert lib.readable_dates("January 13, 2026 – June 10, 2026") == "January 13 – June 10, 2026"
    assert lib.readable_dates("2026-01-13") == "January 13, 2026"
    assert lib.readable_dates("Dates not selected") == "Dates not chosen yet"
    one = lib.study_row(_study("p1", "funded_comparison", state={"result_id": "r1"},
                               name="Alpha", updated="2026-09-22T00:00:00+00:00"),
                        "ifsm", current="ifsm", leaders=(
                            lib.FirmLeader(TPT, "TakeProfitTrader", 100),
                            lib.FirmLeader(MFF, "MyFundedFutures", 900)))
    two = lib.study_row(_study("p2", "funded_comparison", state={"result_id": "r2"},
                               name="Beta", updated="2026-09-23T00:00:00+00:00"),
                        "ifsm", current="ifsm", leaders=(
                            lib.FirmLeader(TPT, "TakeProfitTrader", 500),
                            lib.FirmLeader(MFF, "MyFundedFutures", 200)))
    earlier = lib.study_row(_study("p0", "funded", state={"result_id": "r0"}, name="Gamma"),
                            "ifsm", current="ifsm", leaders=(
                                lib.FirmLeader(TPT, "TakeProfitTrader", 99999),))
    firms = lib.firm_columns([one, two, earlier])
    assert firms == ((TPT, "TakeProfitTrader"), (MFF, "MyFundedFutures"))
    options = lib.sort_options("funded", firms)
    assert "Highest leader net cash, MyFundedFutures" in options
    by_tpt = lib.sort_rows([one, two, earlier], "Highest leader net cash, TakeProfitTrader",
                           firms)
    assert [r.name for r in by_tpt] == ["Beta", "Alpha", "Gamma"]  # earlier method last
    by_mff = lib.sort_rows([one, two, earlier], "Highest leader net cash, MyFundedFutures",
                           firms)
    assert [r.name for r in by_mff] == ["Alpha", "Beta", "Gamma"]
    assert [r.name for r in lib.sort_rows([one, two], "Newest first")] == ["Beta", "Alpha"]
    assert [r.name for r in lib.filter_rows([one, two, earlier], "gamma")] == ["Gamma"]
    assert lib.filter_rows([one, two], "", ["Running"]) == []
    shown, pages = lib.page([one] * 23, 3)
    assert pages == 3 and len(shown) == 3


# ── both stores, through the Streamlit layer ──────────────────────────────


def test_rows_come_from_both_applications_stores(repo, monkeypatch):
    import ifvg_lab_library

    tmp, both = repo
    monkeypatch.setattr(ifvg_lab_library, "_comparison_sources", lambda: {})
    ifvg_lab_library._other_app_studies.clear()
    from ifvg_study_tab import TASK_CARDS, start_draft_from_card

    card = next(c for c in TASK_CARDS if c.card_id == "evaluate_one")
    mine = start_draft_from_card(card, both["main"])
    mine.display_name = "Main app draft"
    save_draft(Path(both["main"]["draft_root"]), mine)
    theirs = start_draft_from_card(card, both["ifsm"])
    theirs.display_name = "Dedicated app draft"
    save_draft(Path(both["ifsm"]["draft_root"]), theirs)
    funded = new_draft(lib.COMPARISON_MODE, display_name="Funded draft")
    funded.steps["review"] = {"funded_comparison": {"source_run_id": "missing-run"}}
    save_draft(Path(both["ifsm"]["draft_root"]), funded)
    studies, issues = load_studies(both["main"])
    assert not issues
    rows, by_app, notices = ifvg_lab_library.library_rows(studies, both["main"])
    assert not notices
    assert set(by_app) == {"main", "ifsm"}
    apps = {r.name: r.app for r in rows}
    assert apps == {"Main app draft": "main", "Dedicated app draft": "ifsm",
                    "Funded draft": "ifsm"}
    by_name = {r.name: r for r in rows}
    assert by_name["Main app draft"].route == "detail"
    assert by_name["Dedicated app draft"].route == ""
    assert "python scripts/run_ifsm_research_ui.py" in by_name["Dedicated app draft"].elsewhere
    assert by_name["Funded draft"].funded_draft
    # from the dedicated application the same stores list the other way round
    studies, _ = load_studies(both["ifsm"])
    rows, _, _ = ifvg_lab_library.library_rows(studies, both["ifsm"])
    by_name = {r.name: r for r in rows}
    assert by_name["Main app draft"].app == "main" and by_name["Main app draft"].route == ""
    completed = ifvg_lab_library._complete(by_name["Funded draft"], by_app, "ifsm")
    assert completed.route == "new_funded"
    assert completed.note.startswith("Can't run in this application: The completed strategy "
                                     "study it uses")


def test_draft_note_follows_the_verified_studies_found_now(repo, monkeypatch):
    """Review fix: the draft check was cached by draft and save time only, so a strategy
    study appearing later kept the old "missing study" note until a restart."""

    import ifvg_lab_library

    from alpha_lab.propsim.funded import comparison_draft

    tmp, both = repo
    found: dict[str, object] = {}
    monkeypatch.setattr(ifvg_lab_library, "_comparison_sources", lambda: dict(found))

    def check(_settings, _selections, sources):
        missing = () if "run-a" in sources else ("The completed strategy study it uses is "
                                                 "not on this computer.",)
        return SimpleNamespace(needs_half_exit_engine=False, problems=missing,
                               configuration_count=64, count_is_exact=True)

    monkeypatch.setattr(comparison_draft, "check_saved_comparison", check)
    funded = new_draft(lib.COMPARISON_MODE, display_name="Funded draft")
    funded.steps["review"] = {"funded_comparison": {"source_run_id": "run-a"}}
    save_draft(Path(both["ifsm"]["draft_root"]), funded)
    studies, _ = load_studies(both["ifsm"])
    rows, by_app, _ = ifvg_lab_library.library_rows(studies, both["ifsm"])
    row = next(r for r in rows if r.name == "Funded draft")
    before = ifvg_lab_library._complete(row, by_app, "ifsm")
    assert before.note.startswith("Can't run in this application: The completed strategy")
    found["run-a"] = SimpleNamespace(package=SimpleNamespace(run_id="run-a",
                                                             manifest_sha256="1" * 64))
    after = ifvg_lab_library._complete(row, by_app, "ifsm")  # same draft, same save time
    assert not after.note.startswith("Can't run"), after.note
    assert ifvg_lab_library.sources_signature(found) == (("run-a", "1" * 64),)


def test_open_results_from_the_other_application_targets_its_store(repo):
    import ifvg_lab_library
    from ifvg_lab_nav import FUNDED_TARGET, SCREEN

    tmp, both = repo
    funded = _study("p1", "funded_comparison", state={"result_id": "r1", "plan_id": "p1"},
                    name="Funded variation study")
    row = lib.study_row(funded, "ifsm", current="main")

    class RerunRequestedError(Exception):
        pass

    def rerun():
        raise RerunRequestedError

    fake = SimpleNamespace(session_state={}, rerun=rerun)
    with pytest.raises(RerunRequestedError):
        ifvg_lab_library._open(fake, row, {"main": both["main"], "ifsm": both["ifsm"]})
    target = fake.session_state[FUNDED_TARGET]
    assert target["app"] == "ifsm" and target["result_id"] == "r1"
    assert target["store_root"] == str(both["ifsm"]["store_root"])
    assert fake.session_state[SCREEN] == "funded"


def _library_app():
    import ifvg_lab_library
    import streamlit as st

    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import load_studies

    roots = ifvg_lab_library._TEST_ROOTS
    studies, _ = load_studies(roots)
    ifvg_lab_library.render_library(st, studies, roots)


def test_library_page_tabs_and_actions_apptest(repo, monkeypatch):
    import ifvg_lab_library
    from ifvg_study_tab import TASK_CARDS, start_draft_from_card
    from ifvg_ui_common import STATE_PREFIX
    from streamlit.testing.v1 import AppTest

    tmp, both = repo
    monkeypatch.setattr(ifvg_lab_library, "_TEST_ROOTS", both["ifsm"], raising=False)
    monkeypatch.setattr(ifvg_lab_library, "_comparison_sources", lambda: {})
    ifvg_lab_library._other_app_studies.clear()
    card = next(c for c in TASK_CARDS if c.card_id == "evaluate_one")
    strategy = start_draft_from_card(card, both["ifsm"])
    strategy.display_name = "Evaluate draft"
    save_draft(Path(both["ifsm"]["draft_root"]), strategy)
    funded = new_draft(lib.COMPARISON_MODE, display_name="Funded draft")
    save_draft(Path(both["ifsm"]["draft_root"]), funded)
    elsewhere = start_draft_from_card(card, both["main"])
    elsewhere.display_name = "Main app draft"
    save_draft(Path(both["main"]["draft_root"]), elsewhere)
    at = AppTest.from_function(_library_app, default_timeout=60).run()
    assert not at.exception
    labels = {b.key: b.label for b in at.button}
    assert labels["ifvg_workspace_new"] == "New study"
    assert labels["ifvg_lab_v1_library_tab_funded"] == "Funded comparisons · 0"
    assert labels["ifvg_lab_v1_library_tab_drafts"] == "Drafts · 3"
    assert labels[f"ifvg_open_draft_{funded.draft_id}"] == lib.CONTINUE_DRAFT
    assert f"ifvg_open_draft_{strategy.draft_id}" not in labels  # only the chosen tab renders
    at.button(key="ifvg_lab_v1_library_tab_drafts").click().run()
    assert not at.exception
    actions = [b.label for b in at.button if str(b.key).startswith("ifvg_open_")]
    assert set(actions) == {lib.CONTINUE_DRAFT}
    assert f"ifvg_open_draft_{elsewhere.draft_id}" not in {b.key for b in at.button}
    text = " ".join(str(e.proto.body) for e in at.get("html"))
    assert "streamlit run scripts/dashboard.py" in text  # the main app's draft names its app
    assert "Dates not chosen yet." in text
    at.button(key=f"ifvg_open_draft_{strategy.draft_id}").click().run()
    assert not at.exception
    assert at.session_state["ifvg_workspace_screen"] == "detail"
    assert at.session_state["ifvg_workspace_selected_study"] == strategy.draft_id
    at.button(key="ifvg_lab_v1_library_tab_funded").click().run()
    at.button(key=f"ifvg_open_draft_{funded.draft_id}").click().run()
    assert not at.exception
    assert at.session_state["ifvg_workspace_screen"] == "new_funded"
    assert at.session_state["ifvg_workspace_destination"] == "New study"
    assert at.session_state[f"{STATE_PREFIX}draft_id"] == funded.draft_id
    assert not list(Path(both["ifsm"]["store_root"]).parent.glob("**/approvals/*"))


def test_search_and_status_filter_apptest(repo, monkeypatch):
    import ifvg_lab_library
    from ifvg_study_tab import TASK_CARDS, start_draft_from_card
    from streamlit.testing.v1 import AppTest

    tmp, both = repo
    monkeypatch.setattr(ifvg_lab_library, "_TEST_ROOTS", both["main"], raising=False)
    ifvg_lab_library._other_app_studies.clear()
    card = next(c for c in TASK_CARDS if c.card_id == "evaluate_one")
    for index in range(12):
        draft = start_draft_from_card(card, both["main"])
        draft.display_name = f"Draft number {index:02d}"
        save_draft(Path(both["main"]["draft_root"]), draft)
    at = AppTest.from_function(_library_app, default_timeout=60).run()
    at.button(key="ifvg_lab_v1_library_tab_drafts").click().run()
    assert not at.exception
    shown = [b for b in at.button if str(b.key).startswith("ifvg_open_draft_")]
    assert len(shown) == lib.PAGE_SIZE  # paged
    at.number_input(key="ifvg_lab_v1_library_page_drafts").set_value(2).run()
    assert len([b for b in at.button if str(b.key).startswith("ifvg_open_draft_")]) == 2
    at.text_input(key="ifvg_lab_v1_library_search").input("number 07").run()
    assert not at.exception
    names = [b.label for b in at.button if str(b.key).startswith("ifvg_details_")]
    assert names == ["Draft number 07"]
    at.text_input(key="ifvg_lab_v1_library_search").input("").run()
    status = at.multiselect(key="ifvg_lab_v1_library_status_drafts")
    assert status.options == ["Draft"]  # only statuses present in this tab
    status.set_value(["Draft"]).run()
    assert not at.exception
    # the search narrowed the list to one page, so the pager went back to page 1
    assert len([b for b in at.button if str(b.key).startswith("ifvg_open_draft_")]) == 10


def test_archived_rows_only_under_archived():
    archived = replace(lib.study_row(_study("s1", "search"), "main", current="main"),
                       archived=True)
    live = lib.study_row(_study("s2", "search"), "main", current="main")
    assert lib.rows_in_tab([archived, live], "strategy") == [live]
    assert lib.rows_in_tab([archived, live], "archived") == [archived]


# ── theme: palette variables only, defined for both themes ────────────────

_COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{3,8}\b|rgba?\(|:\s*(?:white|black)\b")


def test_library_css_and_chosen_tab_style_carry_no_color_literal():
    """Theme contract: the page stylesheet and the chosen-tab style name palette variables
    only, each one a key of BOTH palettes, so the dark theme gets its own value."""

    import ifvg_lab_library

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    emitted: list[str] = []
    fake = SimpleNamespace(session_state={ifvg_lab_library._TAB: "drafts"}, html=emitted.append,
                           container=lambda **_kw: contextlib.nullcontext(),
                           button=lambda *_a, **_kw: None)
    assert ifvg_lab_library._tabs(fake, {t: 0 for t in lib.TABS}) == "drafts"
    text = ifvg_lab_library._CSS + "".join(emitted)
    assert emitted and "tab_drafts button" in emitted[0]
    assert not _COLOR_LITERAL.search(text), _COLOR_LITERAL.search(text).group(0)
    used = set(re.findall(r"var\(--lab-([a-z0-9-]+)\)", text)) - {"mono", "sans", "serif"}
    keys = {name.replace("-", "_") for name in used}
    assert {"ink", "panel", "rule", "body_2", "muted"} <= keys
    assert keys <= set(theme.COLORS) and keys <= set(theme.DARK_COLORS)
    theme.set_theme_resolver(lambda: "dark")
    try:
        dark = theme.palette()
        assert all(dark[k] == theme.DARK_COLORS[k] != theme.COLORS[k] for k in keys)
        assert theme.css_var("ink") in emitted[0]  # a variable name, never a value
    finally:
        theme.set_theme_resolver(None)
    assert theme.palette()["ink"] == theme.COLORS["ink"]
