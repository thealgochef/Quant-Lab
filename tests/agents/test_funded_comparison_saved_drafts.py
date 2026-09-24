"""Saved funded-comparison drafts are never rewritten by opening them (repair R1).

A saved draft may hold values the running application cannot represent — the
half-exit selection under the pinned Strategy-Core, an unknown value, a missing
source study. Such a draft is shown as saved, read-only, with its real
configuration count; approval and launch are unavailable and its bytes never
change. A compatible draft is not rewritten by opening or refreshing either,
while a real edit still saves. The launch rebuilds the plan from the draft on
disk and refuses a stale page, a stale approval or an unapproved plan.

Uses the real verified daily-close study's configurations (read only; skipped
when that archive is absent) and temporary stores only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: E402
    SEARCH_AXIS_REGISTRY_V1,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft, save_draft  # noqa: E402
from alpha_lab.propsim.funded.comparison_source import discover_comparison_sources  # noqa: E402

HALF_EXIT_ENGINE = "exit_policy" in SEARCH_AXIS_REGISTRY_V1
VARIATION_SELECTIONS = {  # the saved 64-configuration variation study
    "enable_shorts": ["enable_shorts.false", "enable_shorts.true"],
    "enabled_entry_sessions": ["enabled_entry_sessions.asia-london-ny",
                               "enabled_entry_sessions.all_open_market_v1"],
    "exit_policy": ["exit_policy.fixed_target_v1",
                    "exit_policy.scale_out_half_breakeven_hold_to_close_v1"],
    "htf_timeframes": ["htf_timeframes.1H-4H", "htf_timeframes.1H"],
    "parent_timeframes": ["parent_timeframes.1m-3m-5m-10m-15m-30m",
                          "parent_timeframes.1m-5m-10m-15m-30m"],
    "tp_r_multiple": ["tp_r_multiple.1.0", "tp_r_multiple.2.0", "tp_r_multiple.3.0"],
}


def _source():
    return next((s for s in discover_comparison_sources() if "S0_D80_W1_P1" in s.by_name), None)


@pytest.fixture
def env(tmp_path, monkeypatch):
    if _source() is None:
        pytest.skip("the verified daily-close study archive is not available locally")
    import ifvg_funded_comparison_study
    import ifvg_workspace

    roots = {
        "repo_root": tmp_path, "store_root": tmp_path / "store",
        "store_roots": {"research": tmp_path / "store"},
        "draft_root": tmp_path / "drafts", "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines",
        "funded_state_root": tmp_path / "funded_jobs",
        "funded_comparison_state_root": tmp_path / "comparison_jobs",
        "reports_root": tmp_path / "reports",
    }
    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    spawned: list[list[str]] = []
    monkeypatch.setattr(ifvg_funded_comparison_study, "_spawn",
                        lambda cmd: spawned.append(cmd) or 1)
    return {"roots": roots, "spawned": spawned, "source": _source()}


def _app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_workspace(st, roots=ifvg_workspace._TEST_ROOTS)


def _variation_draft(source, *, selections=None, **overrides):
    draft = new_draft("funded_configuration_comparison",
                      display_name="Screen check — reopen the completed variation study")
    draft.steps = {
        "objective": {"mode_id": "funded_configuration_comparison",
                      "question_id": "funded_configuration_cash"},
        "search_space": {"axis_selections": {}, "mode_id": "funded_configuration_comparison"},
        "review": {"funded_comparison": {
            "cost_per_side_cents": 514, "firm_keys": ["takeprofittrader", "myfundedfutures"],
            "instrument": "mini", "plan_kind": "variations", "processing": "two_business_days",
            "quantity": 1, "source_run_id": source.package.run_id,
            "variation": {"base": "S0_D80_W1_P1", "half_cost_mills": 514, "half_quantity": 10,
                          "selections": selections or VARIATION_SELECTIONS,
                          "whole_cost_mills": 5140, "whole_quantity": 1},
            **overrides}},
    }
    return draft


def _study_draft(source, selections, **overrides):
    """A study-kind draft saved without ``plan_kind`` (as older drafts are)."""

    draft = new_draft("funded_configuration_comparison", display_name="Study draft")
    draft.steps = {
        "objective": {"mode_id": "funded_configuration_comparison",
                      "question_id": "funded_configuration_cash"},
        "search_space": {"mode_id": "funded_configuration_comparison",
                         "axis_selections": selections},
        "review": {"funded_comparison": {
            "source_run_id": source.package.run_id,
            "firm_keys": ["takeprofittrader", "myfundedfutures"], "instrument": "mini",
            "quantity": 1, "cost_per_side_cents": 514, "processing": "two_business_days",
            **overrides}},
    }
    return draft


def _all_study_values(source):
    from alpha_lab.propsim.funded.comparison_study import axis_choices

    return axis_choices(source)


def _save(roots, draft) -> Path:
    return save_draft(Path(roots["draft_root"]), draft)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _open(draft_id: str) -> AppTest:
    at = AppTest.from_function(_app, default_timeout=240).run()
    buttons = [b for b in at.button if b.label == "Continue"
               and b.key == f"ifvg_open_draft_{draft_id}"]
    (buttons[0] if buttons else next(b for b in at.button if b.label == "Continue")
     ).click().run()
    assert not at.exception, at.exception
    return at


def _text(at: AppTest) -> str:
    parts = [str(e.value) for group in (at.markdown, at.caption, at.warning, at.info,
                                        at.success, at.error) for e in group]
    return "\n".join(parts)


@pytest.mark.skipif(HALF_EXIT_ENGINE, reason="needs the pinned engine (no exit-rule setting)")
def test_pinned_engine_shows_the_saved_64_configurations_read_only_and_never_rewrites(env):
    path = _save(env["roots"], _variation_draft(env["source"]))
    before = _digest(path)
    at = _open(path.parent.name)
    assert _digest(path) == before  # opening did not rewrite the draft
    at.run()
    assert _digest(path) == before  # nor did a refresh
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["steps"]["review"]["funded_comparison"]["variation"]["selections"] == (
        VARIATION_SELECTIONS)
    warning = " ".join(str(w.value) for w in at.warning)
    assert "This study contains 64 configurations" in warning
    assert "partial exits" in warning and "have not been changed" in warning
    assert "48" not in _text(at)  # the reduced plan is never shown
    keys = {getattr(w, "key", None) for w in (*at.button, *at.checkbox, *at.multiselect)}
    assert not {"ifvg_fcmp_approve", "ifvg_fcmp_run", "ifvg_fcmp_agree"} & keys
    assert not any(str(k).startswith("ifvg_fcmp_var_") for k in keys)
    table = at.table[0].value
    assert "Half at 1R, rest at break-even to the close" in table.to_string()
    assert env["spawned"] == []


@pytest.mark.skipif(not HALF_EXIT_ENGINE, reason="needs the research engine with the half exit")
def test_research_engine_rebuilds_all_64_configurations_and_both_firms(env):
    from alpha_lab.propsim.funded.comparison_draft import (
        check_saved_comparison,
        rebuild_saved_plan,
        saved_settings,
        saved_study_selections,
    )

    draft = _variation_draft(env["source"])
    sources = {env["source"].package.run_id: env["source"]}
    check = check_saved_comparison(saved_settings(draft), saved_study_selections(draft), sources)
    assert check.runnable and check.configuration_count == 64 and check.count_is_exact
    envelope, skipped, _ = rebuild_saved_plan(env["source"], saved_settings(draft), {})
    plan = envelope.payload
    assert len(plan.variants) == 64 and len(skipped) == 32
    assert sum(v.exit_policy != "fixed_target_v1" for v in plan.variants) == 16
    assert [p.firm_key for p in plan.firm_profiles] == ["takeprofittrader", "myfundedfutures"]
    path = _save(env["roots"], draft)
    before = _digest(path)
    at = _open(path.parent.name)
    assert _digest(path) == before
    assert "64 configurations around" in _text(at) and "128 separate results" in _text(at)


def test_engine_independent_count_matches_the_half_exit_rule(env):
    from alpha_lab.propsim.funded.comparison_draft import check_saved_comparison, saved_settings

    draft = _variation_draft(env["source"])
    check = check_saved_comparison(saved_settings(draft), {},
                                   {env["source"].package.run_id: env["source"]})
    # 2 x 3 x 2 x 2 x 2 x 2 = 96 combinations; 32 half exits away from 1R are not run
    assert check.configuration_count == 64 and check.count_is_exact
    assert check.runnable is HALF_EXIT_ENGINE
    assert check.needs_half_exit_engine is (not HALF_EXIT_ENGINE)


def test_matching_saved_plan_is_found_only_for_the_exact_settings(env, tmp_path):
    from alpha_lab.propsim.funded.comparison_draft import check_saved_comparison, saved_settings
    from alpha_lab.propsim.funded.comparison_study import (
        build_variation_plan,
        save_plan,
        variation_variants,
    )

    source = env["source"]
    fake_core = {"base_commit": "7c7111e398c083cf8e966e2e0c5aac8a41cc12c0",
                 "branch": "funded-scale-out-exit", "patch_sha256": "1" * 64}
    few = {"tp_r_multiple": ["tp_r_multiple.1.0", "tp_r_multiple.2.0"]}
    variants, _ = variation_variants(
        source, "S0_D80_W1_P1", few,
        whole={"instrument": "mini", "quantity": 1, "cost_per_contract_mills": 5140},
        scale_out={"instrument": "micro", "quantity": 10, "cost_per_contract_mills": 514})
    store = tmp_path / "plans"
    plan_id = save_plan(store, build_variation_plan(source, "S0_D80_W1_P1", variants,
                                                    firm_keys=["takeprofittrader"],
                                                    core_source=fake_core))
    sources = {source.package.run_id: source}
    exact = _variation_draft(source, selections=few, firm_keys=["takeprofittrader"])
    other_size = _variation_draft(source, selections=few, firm_keys=["takeprofittrader"])
    other_size.steps["review"]["funded_comparison"]["variation"]["whole_quantity"] = 2
    other_firms = _variation_draft(source, selections=few)
    assert check_saved_comparison(saved_settings(exact), {}, sources,
                                  store_root=store).matching_plan_ids == (plan_id,)
    for draft in (other_size, other_firms):
        assert check_saved_comparison(saved_settings(draft), {}, sources,
                                      store_root=store).matching_plan_ids == ()


@pytest.mark.parametrize("case", ["unknown_value", "missing_source", "unknown_axis",
                                  "unknown_base", "unknown_firm"])
def test_unrepresentable_saved_settings_are_kept_and_block_editing(env, case):
    source = env["source"]
    if case == "unknown_value":
        values = _all_study_values(source)
        values["opposing_parent_distance_ticks_max"] = [
            *values["opposing_parent_distance_ticks_max"], "opposing_parent_distance_ticks_max.999"]
        draft = _study_draft(source, values)
    elif case == "missing_source":
        draft = _study_draft(source, _all_study_values(source), source_run_id="f" * 64)
    elif case == "unknown_axis":
        draft = _variation_draft(source, selections={
            **VARIATION_SELECTIONS, "htf_selection_max_per_timeframe": [
                "htf_selection_max_per_timeframe.2"]})
    elif case == "unknown_base":
        draft = _variation_draft(source)
        draft.steps["review"]["funded_comparison"]["variation"]["base"] = "S9_D999_W9_P9"
    else:
        draft = _study_draft(source, _all_study_values(source),
                             firm_keys=["takeprofittrader", "some_other_firm"])
    path = _save(env["roots"], draft)
    before = _digest(path)
    at = _open(path.parent.name)
    at.run()
    assert _digest(path) == before
    assert "Your saved settings have not been changed" in " ".join(str(w.value)
                                                                   for w in at.warning)
    keys = {getattr(w, "key", None) for w in (*at.button, *at.checkbox, *at.selectbox)}
    assert not {"ifvg_fcmp_approve", "ifvg_fcmp_run", "ifvg_fcmp_source"} & keys
    assert env["spawned"] == []


def test_opening_a_compatible_draft_never_normalizes_it_but_an_edit_saves(env):
    source = env["source"]
    path = _save(env["roots"], _study_draft(source, _all_study_values(source)))
    before = _digest(path)
    at = _open(path.parent.name)
    at.run()
    assert _digest(path) == before  # no plan_kind or defaults written by opening
    assert "32 configurations" in _text(at)
    at.number_input(key="ifvg_fcmp_quantity").set_value(2).run()
    saved = json.loads(path.read_text(encoding="utf-8"))["steps"]["review"]["funded_comparison"]
    assert saved["quantity"] == 2 and saved["plan_kind"] == "study"


def test_selected_chips_show_each_value_not_the_setting_name(env):
    source = env["source"]
    path = _save(env["roots"], _study_draft(source, _all_study_values(source)))
    at = _open(path.parent.name)
    captions = [str(c.value) for c in at.caption]
    for ms in at.multiselect:
        if not str(ms.key).startswith("ifvg_fcmp_axis_"):
            continue
        shown = [ms.format_func(v) for v in ms.value]
        assert len(set(shown)) == len(shown) >= 2, shown
        assert not any(label.startswith(ms.label) for label in shown), shown
        assert not any(label.startswith("Opposing") or label.startswith("One-minute opp")
                       or label.startswith("Parent timeframe") for label in shown), shown
    distance = next(c for c in captions if c.startswith("Selected:") and "160 ticks" in c)
    assert "80 ticks" in distance
    assert any("Original three windows - 3:00 PM to 12:45 AM" in c for c in captions)


def test_a_stale_page_or_approval_never_reaches_the_worker(env):
    import ifvg_funded_comparison_study as screen

    from alpha_lab.propsim.funded.comparison_draft import rebuild_saved_plan, saved_settings
    from alpha_lab.propsim.funded.comparison_study import record_owner_approval, save_plan

    source = env["source"]
    roots = env["roots"]
    values = _all_study_values(source)
    draft = _study_draft(source, values, plan_kind="study")
    path = _save(roots, draft)
    shown, _, _ = rebuild_saved_plan(source, saved_settings(draft), values)
    store = Path(roots["store_root"])
    save_plan(store, shown)
    # unapproved: refused
    assert "no recorded approval" in screen.dispatch_problem(roots, draft.draft_id, shown)
    record_owner_approval(store, shown.funded_comparison_plan_id, approved_on="2026-09-23",
                          channel="study_screen",
                          statement="Synthetic approval of this exact test plan.",
                          scope="32 configurations, test store only")
    assert screen.dispatch_problem(roots, draft.draft_id, shown) is None
    # the draft changes in another window: the approved plan on the old page is refused
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["steps"]["review"]["funded_comparison"]["quantity"] = 2
    path.write_text(json.dumps(changed), encoding="utf-8")
    assert "out of date" in screen.dispatch_problem(roots, draft.draft_id, shown)
    # and the page, refreshed, shows the new plan without the old approval
    at = _open(draft.draft_id)
    assert at.number_input(key="ifvg_fcmp_quantity").value == 2
    assert not any("Approved on" in str(s.value) for s in at.success)
    assert at.button(key="ifvg_fcmp_run").disabled
    assert env["spawned"] == []


def test_job_start_refuses_a_plan_without_its_owner_approval(env, tmp_path, monkeypatch):
    import ifvg_funded_comparison_job as job

    launched: list = []

    def never(*args, **kwargs):  # a real detached worker must never start from a test
        launched.append(args)
        raise AssertionError("a worker process was started")

    monkeypatch.setattr(job.subprocess, "Popen", never)

    from alpha_lab.propsim.funded.comparison_draft import rebuild_saved_plan, saved_settings
    from alpha_lab.propsim.funded.comparison_study import save_plan

    source = env["source"]
    values = _all_study_values(source)
    envelope, _, _ = rebuild_saved_plan(source, saved_settings(_study_draft(source, values)),
                                        values)
    store = tmp_path / "store"
    plan_id = save_plan(store, envelope)
    state_root = tmp_path / "states"
    assert job.main(["start", "--plan-id", plan_id, "--store-root", str(store),
                     "--state-root", str(state_root), "--reports-root",
                     str(tmp_path / "r")]) == 1
    assert not (state_root / plan_id).exists()  # nothing queued, no worker started
    assert launched == []


def test_a_draft_without_its_saved_study_is_refused_not_crashed(env):
    """Review finding: a draft saved before any strategy study existed."""

    import ifvg_funded_comparison_study as screen

    from alpha_lab.propsim.funded.comparison_draft import rebuild_saved_plan, saved_settings

    source = env["source"]
    values = _all_study_values(source)
    draft = _study_draft(source, values, plan_kind="study")
    shown, _, _ = rebuild_saved_plan(source, saved_settings(draft), values)
    del draft.steps["review"]["funded_comparison"]["source_run_id"]
    _save(env["roots"], draft)
    problem = screen.dispatch_problem(env["roots"], draft.draft_id, shown)
    assert "does not name a completed strategy study" in problem
    assert env["spawned"] == []


def test_a_draft_saved_in_another_window_resets_the_page_and_its_approval(env):
    """Review finding: the stale-widget reset and "Record my approval" on a stale page."""

    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft
    from alpha_lab.propsim.funded.comparison_runner import APPROVAL_STORE

    source = env["source"]
    roots = env["roots"]
    draft = _study_draft(source, _all_study_values(source), plan_kind="study")
    path = _save(roots, draft)
    at = _open(draft.draft_id)
    at.checkbox(key="ifvg_fcmp_agree").check().run()
    assert not at.button(key="ifvg_fcmp_approve").disabled
    # another window saves a change to the same draft
    other = load_draft(Path(roots["draft_root"]), draft.draft_id)
    other.steps["review"]["funded_comparison"]["quantity"] = 2
    _save(roots, other)
    saved_bytes = path.read_bytes()
    at.button(key="ifvg_fcmp_approve").click().run()  # clicked on the stale page
    assert not at.exception, at.exception
    assert "saved from another window" in _text(at)
    assert at.number_input(key="ifvg_fcmp_quantity").value == 2
    assert at.checkbox(key="ifvg_fcmp_agree").value is False
    approvals = Path(roots["store_root"]) / APPROVAL_STORE
    assert not approvals.exists() or not any(approvals.iterdir())  # no approval recorded
    assert path.read_bytes() == saved_bytes  # the other window's save is kept
    assert env["spawned"] == []
