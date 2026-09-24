"""Repairs R7 (named S0_D80_W1_P1 baseline) and R8 (extended research period).

Uses the verified daily-close study package when present (read only; the
named-baseline tests skip without it) and temporary draft roots only.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from alpha_lab.agents.data_infra.ifvg.development_access import (  # noqa: E402
    FROZEN_WARMUP_DATES,
    DevelopmentReplayPolicy,
)
from alpha_lab.agents.data_infra.ifvg.named_baselines import (  # noqa: E402
    LEGACY_PROFILE_NAME,
    OWNER_BASELINE_SECTION_HASH,
    NamedBaselineUnavailableError,
    legacy_baseline_warning,
    owner_selected_baseline,
)
from alpha_lab.agents.data_infra.ifvg.research_period import (  # noqa: E402
    earliest_evidence_day,
    resolve_research_range,
    warmup_dates_for,
)

apptest = pytest.importorskip("streamlit.testing.v1")


def _named():
    try:
        return owner_selected_baseline()
    except NamedBaselineUnavailableError:
        pytest.skip("the verified daily-close study package is not available locally")


def _section(profile, fixed=None):
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import resolve_axis_overrides
    from alpha_lab.agents.data_infra.ifvg.search.identities import canonicalize_section

    overrides = resolve_axis_overrides(dict(fixed)) if fixed else {}
    config = {"profile_name": profile, **({"section_overrides": dict(overrides)}
                                          if overrides else {})}
    return canonicalize_section(resolve_profile_config(config).section)


# ── R7: the named baseline ───────────────────────────────────────────────────


def test_named_baseline_matches_the_saved_configuration_field_for_field():
    from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

    from alpha_lab.propsim.funded.comparison_source import discover_comparison_sources

    named = _named()
    saved = next(s for s in discover_comparison_sources()
                 if s.package.run_id == named.source_run_id).by_name["S0_D80_W1_P1"]
    assert dict(named.axis_value_ids) == saved.axis_value_ids  # read, never retyped
    section = _section(named.profile_name, dict(named.axis_value_ids))
    assert ifvg_profile_hash(section) == OWNER_BASELINE_SECTION_HASH
    assert section.holding_policy == "scheduled_daily_close_v1"
    assert section.htf_gap_invalidation_policy == "own_timeframe_close_v1"
    assert section.htf_selection_max_per_timeframe == 2
    assert section.opposing_parent_distance_ticks_max == 80
    assert section.opposing_min_gap_ticks == 1
    assert section.parent_retest_timeout_1m_bars == 240
    assert section.opposing_timeout_1m_bars == 90
    assert 60 in section.parent_timeframes_seconds if hasattr(
        section, "parent_timeframes_seconds") else True
    assert legacy_baseline_warning(section) is None


def test_the_legacy_baseline_is_unchanged_and_warned():
    from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

    legacy = _section(LEGACY_PROFILE_NAME)
    assert ifvg_profile_hash(legacy).startswith("e0f31873")  # same identity as before
    assert legacy_baseline_warning(legacy) == (
        "This baseline holds positions across the daily close and weekends and has no "
        "retest time limit. It does not follow the mandatory 3:55 PM Chicago close.")


def test_new_evaluate_studies_start_on_the_named_baseline_other_types_unchanged(tmp_path):
    import ifvg_study_tab as tab

    named = _named()
    by_id = {card.card_id: card for card in tab.TASK_CARDS}
    evaluate = tab.start_draft_from_card(by_id["evaluate_one"], {})
    assert evaluate.steps["baseline"] == {
        "baseline_profile_name": LEGACY_PROFILE_NAME,
        "fixed_axis_value_ids": dict(named.axis_value_ids),
        "named_baseline_id": "S0_D80_W1_P1",
    }
    compare = tab.start_draft_from_card(by_id["compare_with_baseline"], {})
    assert "baseline" not in compare.steps


def test_the_developer_header_start_also_uses_the_named_baseline(monkeypatch, tmp_path):
    """Review finding: the wizard's own "Start new draft" button (developer header)."""

    import ifvg_study_wizard as wizard

    from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft
    from tests.agents.test_ifsm_replication_ui import _roots, _wizard_app

    named = _named()
    roots = _roots(monkeypatch, tmp_path)
    at = apptest.AppTest.from_function(_wizard_app, default_timeout=120).run()
    at.selectbox(key=f"{wizard._W}new_purpose").set_value("Development Research")
    at.selectbox(key=f"{wizard._W}new_mode").set_value("Single Configuration")
    at.button(key=f"{wizard._W}new_draft").click().run()
    assert not at.exception, at.exception
    next(b for b in at.button if b.label == "Save Draft").click().run()
    assert not at.exception, at.exception
    saved = load_draft(roots["drafts"], at.session_state[f"{wizard.STATE_PREFIX}draft_id"])
    assert saved.step_payload("objective")["question_id"] == "evaluate_one_configuration"
    assert saved.step_payload("baseline")["fixed_axis_value_ids"] == dict(named.axis_value_ids)
    assert saved.step_payload("baseline")["named_baseline_id"] == "S0_D80_W1_P1"


def _workspace(monkeypatch, tmp_path):
    import ifvg_workspace

    from alpha_lab.agents.data_infra.ifvg.presentation import workspace_mode

    roots = {"store_root": tmp_path / "store",
             "store_roots": {"research": tmp_path / "store", "test": tmp_path / "test"},
             "draft_root": tmp_path / "drafts", "state_root": tmp_path / "jobs",
             "pipeline_state_root": tmp_path / "pipelines", "repo_root": tmp_path}
    monkeypatch.setattr(ifvg_workspace, "_R78_ROOTS", roots, raising=False)
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", False)
    return roots


def _app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_workspace(st, roots=ifvg_workspace._R78_ROOTS)


def _saved_legacy_draft(roots, *, step, **steps):
    import ifvg_study_tab as tab

    from alpha_lab.agents.data_infra.ifvg.study_drafts import save_draft

    card = next(c for c in tab.TASK_CARDS if c.card_id == "evaluate_one")
    draft = tab.start_draft_from_card(card, roots)
    # the shape 17 saved drafts have: no fixed-settings key, a hash saved by the
    # strategy engine before the September 22 upgrade
    draft.steps["baseline"] = {"baseline_blocked_reason": None,
                               "baseline_profile_name": LEGACY_PROFILE_NAME,
                               "baseline_section_config_hash": "f3bce50c" + "0" * 56}
    draft.steps.update(steps)
    draft.current_step_key = step
    draft.step_index = {"baseline": 1, "validation": 3}[step]  # as saved drafts record it
    path = save_draft(Path(roots["draft_root"]), draft)
    return draft, path


def _open(draft):
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.session_state["ifvg_workspace_screen"] = "new"
    from ifvg_ui_common import STATE_PREFIX

    at.session_state[f"{STATE_PREFIX}draft_id"] = draft.draft_id
    return at.run()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_an_existing_legacy_draft_reopens_unchanged_and_warned(monkeypatch, tmp_path):
    roots = _workspace(monkeypatch, tmp_path)
    draft, path = _saved_legacy_draft(roots, step="baseline")
    before = _digest(path)
    at = _open(draft)
    assert not at.exception, at.exception
    at.run()  # refresh
    assert _digest(path) == before
    warnings = " ".join(str(w.value) for w in at.warning)
    assert "does not follow the mandatory 3:55 PM Chicago close" in warnings
    assert "earlier version of the strategy engine" in warnings
    # only the owner's explicit action updates the saved baseline
    next(b for b in at.button
         if b.label == "Update the saved baseline to the current engine").click().run()
    saved = json.loads(path.read_text(encoding="utf-8"))["steps"]["baseline"]
    assert saved["baseline_section_config_hash"].startswith("e0f31873")
    assert "fixed_axis_value_ids" not in saved and "replication_recipe_id" not in saved


def test_a_saved_configuration_without_the_gap_rule_reopens_unchanged(monkeypatch, tmp_path):
    """The main application's Configuration step, on the shape of draft f3099b0b…."""

    roots = _workspace(monkeypatch, tmp_path)
    fixed = {"opposing_timeout_1m_bars": "opposing_timeout_1m_bars.90",
             "parent_htf_distance_ticks_max": "parent_htf_distance_ticks_max.160",
             "parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.240"}
    draft, path = _saved_legacy_draft(roots, step="baseline", baseline={
        "baseline_blocked_reason": None, "baseline_profile_name": LEGACY_PROFILE_NAME,
        "baseline_section_config_hash": "5e8dfb22" + "0" * 56,
        "fixed_axis_value_ids": fixed})
    before = _digest(path)
    at = _open(draft)
    assert not at.exception, at.exception
    at.run()  # refresh
    assert _digest(path) == before


def test_review_states_a_holding_rule_other_than_the_daily_close():
    from ifvg_research_wizard import holding_statement

    from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft

    draft = new_draft("single_configuration")
    draft.steps["objective"] = {"question_id": "evaluate_one_configuration"}
    draft.steps["baseline"] = {"baseline_profile_name": LEGACY_PROFILE_NAME}
    assert "not the mandatory 3:55 PM Chicago daily close" in holding_statement(draft, None)
    draft.steps["baseline"]["fixed_axis_value_ids"] = {
        "holding_policy": "holding_policy.scheduled_daily_close_v1"}
    assert holding_statement(draft, None) is None


def test_the_legacy_warning_remains_when_saved_settings_cannot_be_read():
    from ifvg_research_wizard import _legacy_warning

    shown: list[str] = []

    class Screen:
        def warning(self, text):
            shown.append(text)

    _legacy_warning(Screen(), LEGACY_PROFILE_NAME, {"holding_policy": None})
    assert len(shown) == 1
    assert "does not follow the mandatory 3:55 PM Chicago close" in shown[0]
    assert "describes the baseline alone" in shown[0]


def test_a_search_varying_the_holding_rule_counts_only_the_open_configurations():
    """Review finding: the daily-close value is a composite setting without a payload."""

    from ifvg_research_wizard import holding_statement

    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1
    from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft

    close = "holding_policy.scheduled_daily_close_v1"
    if close not in SEARCH_AXIS_REGISTRY_V1["holding_policy"].registered_values:
        pytest.skip("this engine does not offer the daily-close holding rule")
    draft = new_draft("strategy_search")
    draft.steps["objective"] = {"question_id": "search_configurations"}
    draft.steps["baseline"] = {"baseline_profile_name": LEGACY_PROFILE_NAME}
    draft.steps["search_space"] = {"axis_selections": {"holding_policy": [close]}}
    assert holding_statement(draft, None).startswith("1 of 2 configurations")
    draft.steps["search_space"]["axis_selections"]["opposing_timeout_1m_bars"] = [
        "opposing_timeout_1m_bars.90"]
    assert holding_statement(draft, None).startswith("2 of 4 configurations")
    # saved settings this engine cannot read (draft 708c133a… saved null values) are
    # never reported as closing daily
    evaluate = new_draft("single_configuration")
    evaluate.steps["objective"] = {"question_id": "evaluate_one_configuration"}
    evaluate.steps["baseline"] = {"baseline_profile_name": LEGACY_PROFILE_NAME,
                                  "fixed_axis_value_ids": {"holding_policy": None}}
    assert "could not be checked" in holding_statement(evaluate, None)


# ── R8: the extended research period ─────────────────────────────────────────


def test_existing_period_resolves_to_the_saved_107_days_and_frozen_warmup():
    resolved = resolve_research_range("2026-01-13", "2026-06-10")
    assert len(resolved.trading_days) == 107 and resolved.usable
    assert resolved.warmup_dates == FROZEN_WARMUP_DATES == warmup_dates_for("2026-02-02")


def test_june_11_onward_is_refused_and_the_earliest_start_is_reported():
    assert resolve_research_range("2026-01-13", "2026-06-11").problems
    assert resolve_research_range("2026-06-11", "2026-06-12").problems
    assert earliest_evidence_day() == "2021-12-14"  # ten stored warmup days from 2021-12-02
    assert resolve_research_range("2021-12-13", "2021-12-20").problems
    early = resolve_research_range("2021-12-14", "2021-12-31")
    assert early.usable and early.warmup_dates[0] == "2021-12-02"
    assert ("2021-12-24", early.excluded[0][1]) == early.excluded[0]  # Christmas observed


def test_days_without_stored_data_are_left_out_with_their_reason():
    stored = {"2025-06-0" + str(d) for d in range(1, 10)} | {
        f"2025-06-{d}" for d in range(10, 31)}
    stored.discard("2025-06-17")
    resolved = resolve_research_range("2025-06-13", "2025-06-20",
                                      day_has_data=lambda day: day in stored)
    left_out = dict(resolved.excluded)
    assert left_out["2025-06-17"] == "no stored market data for 2025-06-17"
    assert left_out["2025-06-18"] == (
        "no stored market data for the previous evening (2025-06-17) that opens this trading day")
    assert "2025-06-17" not in resolved.trading_days
    assert any("cannot run" in w for w in resolved.warnings)  # inputs exist only for 2026


def test_worker_boundary_accepts_the_extended_window_and_still_protects_june_11():
    extended = (*warmup_dates_for("2025-06-13"), "2025-06-13", "2025-06-16")
    DevelopmentReplayPolicy(extended)  # accepted
    with pytest.raises(PermissionError):
        DevelopmentReplayPolicy((*extended, "2026-06-11"))
    with pytest.raises(ValueError):
        DevelopmentReplayPolicy(("2025-06-01", *extended[1:]))  # wrong warmup
    DevelopmentReplayPolicy((*FROZEN_WARMUP_DATES, "2026-01-13"))  # unchanged path
    with pytest.raises(PermissionError):
        DevelopmentReplayPolicy((*FROZEN_WARMUP_DATES, "2026-06-11"))


def test_the_date_step_keeps_a_saved_list_byte_identical(monkeypatch, tmp_path):
    roots = _workspace(monkeypatch, tmp_path)
    days = list(resolve_research_range("2026-01-13", "2026-06-10").trading_days)
    for saved_days in (days, ["2026-01-13", "2026-02-02", "2026-03-02"]):  # range and custom
        draft, path = _saved_legacy_draft(roots, step="validation", validation={
            "run_scope": "full_authorized_development", "evidence_class": "real",
            "real_dates": saved_days, "warmup_dates": list(FROZEN_WARMUP_DATES), "seed": 7,
            "worker_limit": 1})
        before = _digest(path)
        at = _open(draft)
        assert not at.exception, at.exception
        at.run()
        assert _digest(path) == before
        start = next(w for w in at.date_input if w.label == "Start date")
        assert start.value == date.fromisoformat(saved_days[0])


def test_a_saved_list_the_pickers_cannot_show_opens_unchanged(monkeypatch, tmp_path):
    """Review finding: a list ending in the protected period, or a line that is not a
    date, never crashes the step and is never rewritten by opening it."""

    roots = _workspace(monkeypatch, tmp_path)
    for saved_days in (["2026-06-09", "2026-06-12"], ["2026-01-13", "2026-13-40"]):
        draft, path = _saved_legacy_draft(roots, step="validation", validation={
            "run_scope": "full_authorized_development", "evidence_class": "real",
            "real_dates": saved_days, "warmup_dates": list(FROZEN_WARMUP_DATES), "seed": 7,
            "worker_limit": 1})
        before = _digest(path)
        at = _open(draft)
        assert not at.exception, at.exception
        at.run()
        assert _digest(path) == before
        assert any("kept exactly as saved" in str(w.value) for w in at.warning)
        assert next(w for w in at.date_input if w.label == "Start date").value is None


def test_a_changed_unusable_range_never_keeps_the_old_list():
    from ifvg_research_dates import render_research_dates

    class Picker:
        def __init__(self, chosen):
            self.chosen = chosen

        def date_input(self, label, **_):
            return self.chosen[label]

    class Screen:
        def __init__(self, chosen):
            self.shown, self.chosen = [], chosen

        def columns(self, n):
            return [Picker(self.chosen)] * n

        def write(self, *a, **k):
            pass

        caption = info = success = warning = code = table = write

        def error(self, text):
            self.shown.append(text)

        def expander(self, *_a, **_k):
            import contextlib

            return contextlib.nullcontext()

    payload = {"real_dates": ["2026-01-13", "2026-01-14"],
               "warmup_dates": list(FROZEN_WARMUP_DATES)}
    backwards = Screen({"Start date": date(2026, 3, 2), "End date": date(2026, 2, 2)})
    assert render_research_dates(backwards, payload, "k_", Path(".")) == ((), ())
    assert "The end date is before the start date." in backwards.shown
    unchanged = Screen({"Start date": date(2026, 1, 13), "End date": date(2026, 1, 14)})
    assert render_research_dates(unchanged, payload, "k_", Path("."))[0] == (
        "2026-01-13", "2026-01-14")


def test_the_not_prepared_note_follows_the_first_replayed_day():
    resolved = resolve_research_range("2026-01-02", "2026-01-30")
    assert resolved.warmup_dates[0] < "2026-01-01"
    assert any("first replays " + resolved.warmup_dates[0] in note
               for note in resolved.warnings)
    assert not resolve_research_range("2026-01-13", "2026-01-30").warnings


def test_saved_plan_date_policy_accepts_the_extended_window_and_protects_june_11():
    from alpha_lab.agents.data_infra.ifvg.search.charter import DatePolicy

    policy = "development_explicit_dates_before_path_v2"
    warmup = warmup_dates_for("2025-06-13")
    DatePolicy(replay_dates=(*warmup, "2025-06-13"), warmup_dates=warmup,
               access_policy_id=policy)
    DatePolicy(replay_dates=(*FROZEN_WARMUP_DATES, "2026-01-13"),
               warmup_dates=FROZEN_WARMUP_DATES, access_policy_id=policy)  # unchanged
    for bad in ((*warmup, "2025-06-13", "2026-06-11"), ("2021-11-30", *warmup[1:], "2025-06-13"),
                (*warmup[1:], "2025-06-13")):
        with pytest.raises(ValueError):
            DatePolicy(replay_dates=tuple(sorted(bad)), warmup_dates=warmup,
                       access_policy_id=policy)
    with pytest.raises(ValueError):  # the five-day verification fixture stays in 2026
        DatePolicy(replay_dates=("2025-06-13",), warmup_dates=(),
                   access_policy_id="verification_fixed_allowlist_max5_v1")


def test_validation_step_uses_the_extended_window_and_its_warmup():
    from alpha_lab.agents.data_infra.ifvg.study_presentation import validate_validation_step

    base = {"run_scope": "full_authorized_development", "evidence_class": "real", "seed": 7,
            "worker_limit": 1}
    ok = validate_validation_step({**base, "real_dates": ("2025-06-13", "2025-06-16"),
                                   "warmup_dates": warmup_dates_for("2025-06-13")})
    assert ok == {}
    wrong_warmup = validate_validation_step({**base, "real_dates": ("2025-06-13",),
                                             "warmup_dates": FROZEN_WARMUP_DATES})
    assert "warmup_dates" in wrong_warmup
    protected = validate_validation_step({**base, "real_dates": ("2026-06-11",),
                                          "warmup_dates": FROZEN_WARMUP_DATES})
    assert "protected" in protected["real_dates"]
    empty = validate_validation_step({**base, "real_dates": (),
                                      "warmup_dates": FROZEN_WARMUP_DATES})
    assert "2021-12-14" in empty["real_dates"]
    saved = validate_validation_step({**base, "real_dates": ("2026-01-13", "2026-01-14"),
                                      "warmup_dates": FROZEN_WARMUP_DATES})
    assert saved == {}  # every existing saved list validates exactly as before


def test_the_charter_uses_the_warmup_before_the_first_evidence_day(monkeypatch, tmp_path):
    import ifvg_study_tab as tab
    import ifvg_study_wizard as wizard

    roots = _workspace(monkeypatch, tmp_path)
    card = next(c for c in tab.TASK_CARDS if c.card_id == "evaluate_one")
    for first, days in (("2025-06-13", ["2025-06-13", "2025-06-16"]),
                        ("2026-01-13", ["2026-01-13", "2026-01-14"])):
        draft = tab.start_draft_from_card(card, roots)
        draft.steps["validation"].update(real_dates=days,
                                         warmup_dates=list(warmup_dates_for(first)), seed=7)
        policy = wizard._charter_fields(draft, roots)["date_policy"]
        assert policy.warmup_dates == warmup_dates_for(first)
        assert policy.replay_dates == (*warmup_dates_for(first), *days)
    assert warmup_dates_for("2026-01-13") == FROZEN_WARMUP_DATES  # 2026 charters unchanged
