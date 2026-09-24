"""Exact IFSM configuration UI, persistence and enumeration without replaying data.

Run against either the installed Core or a process-local repaired Core. Tests
requiring optional lifecycle fields skip when that engine does not expose them.
Every writable workspace and draft in this module is a temporary test fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("developer_presentation")
apptest = pytest.importorskip("streamlit.testing.v1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import ifvg_study_tab as study_tab  # noqa: E402
import ifvg_study_wizard as wizard  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.ifsm_replication import (  # noqa: E402
    catalog,
    draft_from_recipe,
    fixed_axis_values,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: E402
    SEARCH_AXIS_REGISTRY_V1,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: E402
    SearchCharterEnvelope,
    SearchCharterPayload,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: E402
    canonicalize_section,
)
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (  # noqa: E402
    enumerate_children,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import (  # noqa: E402
    load_draft,
    new_draft,
    save_draft,
)

BASELINE = "ifvg_v2_doc_default_fresh_static_1r"
NUMERIC_CHOICES = {
    "parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.60",
    "opposing_timeout_1m_bars": "opposing_timeout_1m_bars.90",
    "htf_registry_max_age_days": "htf_registry_max_age_days.15",
    "parent_reaction_window_parent_bars": "parent_reaction_window_parent_bars.20",
    "parent_htf_distance_ticks_max": "parent_htf_distance_ticks_max.160",
    "opposing_parent_distance_ticks_max": "opposing_parent_distance_ticks_max.160",
}
OPTIONAL_CHOICES = {
    "setup_timeout_1m_bars": "setup_timeout_1m_bars.180",
    "parent_replacement_policy": "parent_replacement_policy.preserve_selected",
    "parent_retest_depth_policy": "parent_retest_depth_policy.strictly_before_ce",
}


def _with_explicit_gap_policy(values=None):
    selected = dict(values or {})
    axis = "htf_gap_invalidation_policy"
    if axis in SEARCH_AXIS_REGISTRY_V1:
        selected.setdefault(axis, f"{axis}.execution_wick_full_fill_v1")
    return selected


def _named_start():
    """S0_D80_W1_P1's saved values and id, or nothing where its verified package is
    absent (a new Evaluate study then keeps the registered baseline, repair R7)."""

    from alpha_lab.agents.data_infra.ifvg.named_baselines import (
        NamedBaselineUnavailableError,
        owner_selected_baseline,
    )

    try:
        return dict(owner_selected_baseline().axis_value_ids), "S0_D80_W1_P1"
    except NamedBaselineUnavailableError:
        return {}, None


def _roots(monkeypatch, tmp_path):
    roots = {
        "research": tmp_path / "search" / "v1",
        "verification": tmp_path / "search_test" / "v1",
        "state": tmp_path / "state",
        "drafts": tmp_path / "drafts",
    }
    for attribute, name in (
        ("STORE_ROOT_RESEARCH", "research"),
        ("STORE_ROOT_VERIFICATION", "verification"),
        ("STATE_ROOT", "state"),
        ("DRAFT_ROOT", "drafts"),
    ):
        monkeypatch.setattr(study_tab, attribute, roots[name])
    return roots


def _draft(*, fixed=None, mode="single_configuration", question="evaluate_one_configuration",
           purpose="development_research"):
    from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

    draft = new_draft(mode, display_name="Exact configuration AppTest")
    draft.step_index = 1
    draft.current_step_key = "baseline"
    draft.purpose_annotation = {
        "schema_version": 1, "purpose": purpose, "derivation": "card_selected",
        "owner_confirmed": True, "updated_at": "2026-09-09T00:00:00Z",
    }
    draft.steps = {
        "objective": {
            "mode_id": mode, "question_id": question,
            "template_id": "strategy_quality_only", "custom_objectives": (),
        },
        "baseline": {
            "baseline_profile_name": BASELINE,
            "baseline_section_config_hash": ifvg_profile_hash(
                canonicalize_section(resolve_profile_config({"profile_name": BASELINE}).section)
            ),
            "baseline_blocked_reason": None, "fixed_axis_value_ids": dict(fixed or {}),
        },
        "search_space": {"mode_id": mode, "axis_selections": {}},
        "benchmarks": {},
        "validation": {
            "run_scope": "verification_5d" if purpose == "implementation_verification"
            else "full_authorized_development",
            "evidence_class": "synthetic_fixture" if purpose == "implementation_verification"
            else "real",
            "real_dates": ["2026-06-04"] if purpose == "implementation_verification"
            else ["2026-01-13", "2026-01-14"],
            "warmup_dates": [], "seed": 7, "worker_limit": 1,
        },
        "review": {"n_children": 1},
    }
    return draft


def _wizard_app():
    import ifvg_study_tab as study_tab
    import ifvg_study_wizard as wizard
    import streamlit as st

    wizard.render_new_study(st, roots=study_tab.workspace_roots(st))


def _focused_app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_workspace(st, roots=ifvg_workspace._IFSM_TEST_ROOTS)


def _focused_roots(monkeypatch, tmp_path):
    import ifvg_workspace

    from alpha_lab.agents.data_infra.ifvg.presentation import workspace_mode

    roots = {
        "store_root": tmp_path / "store",
        "store_roots": {"research": tmp_path / "store", "test": tmp_path / "test"},
        "draft_root": tmp_path / "drafts", "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines", "repo_root": tmp_path,
    }
    monkeypatch.setattr(ifvg_workspace, "_IFSM_TEST_ROOTS", roots, raising=False)
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", False)
    return roots


def _open(draft, roots):
    save_draft(roots["drafts"], draft)
    app = apptest.AppTest.from_function(_wizard_app, default_timeout=120)
    app.session_state[f"{wizard.STATE_PREFIX}draft_id"] = draft.draft_id
    app.run()
    assert not app.exception
    return app


def _fixed_widget(app, axis):
    return next(widget for widget in app.selectbox
                if "fixed_" in str(widget.key)
                and str(widget.key).endswith(f"{BASELINE}_{axis}"))


def _save(app, draft, roots):
    next(button for button in app.button if button.label == "Save Draft").click().run()
    assert not app.exception
    return load_draft(roots["drafts"], draft.draft_id)


def _enumerate_fixture(fields):
    # In-memory synthetic enumeration tests profile resolution only. The marker
    # is never saved, authorized against real data, or passed to a replay runner.
    payload = SearchCharterPayload.model_validate({
        **fields, "owner_authorization": wizard.SyntheticAuthorizationMarker(),
    })
    return enumerate_children(
        SearchCharterEnvelope.from_payload(payload),
        identity_resolver=lambda spec: spec.resolved_section_config_hash,
    )


def test_fixed_numeric_settings_save_exactly_one_replay(monkeypatch, tmp_path):
    roots = _roots(monkeypatch, tmp_path)
    draft = _draft()
    app = _open(draft, roots)
    for axis, value_id in NUMERIC_CHOICES.items():
        _fixed_widget(app, axis).set_value(value_id).run()
        assert not app.exception
    assert "160 ticks" in _fixed_widget(app, "parent_htf_distance_ticks_max").options
    stored = _save(app, draft, roots)
    expected = _with_explicit_gap_policy(NUMERIC_CHOICES)
    assert stored.step_payload("baseline")["fixed_axis_value_ids"] == expected
    fields = wizard._charter_fields(stored, roots)
    assert fields["axes"] == {axis: (value,) for axis, value in expected.items()}
    assert fields["max_child_count"] == 1
    children = _enumerate_fixture(fields)
    assert len(children) == 1
    expected_overrides = {
        "parent_retest_timeout_1m_bars": 60, "opposing_timeout_1m_bars": 90,
        "htf_registry_max_age_days": 15, "parent_reaction_window_parent_bars": 20,
        "parent_htf_distance_ticks_max": 160, "opposing_parent_distance_ticks_max": 160,
    }
    if "htf_gap_invalidation_policy" in SEARCH_AXIS_REGISTRY_V1:
        expected_overrides["htf_gap_invalidation_policy"] = "execution_wick_full_fill_v1"
    assert children[0].section_overrides == expected_overrides
    assert not roots["research"].exists() and not roots["state"].exists()


def test_fixed_unbounded_value_and_keep_baseline_are_distinct(monkeypatch, tmp_path):
    roots = _roots(monkeypatch, tmp_path)
    axis = "parent_retest_timeout_1m_bars"
    draft = _draft(fixed={axis: f"{axis}.240"})
    app = _open(draft, roots)
    _fixed_widget(app, axis).set_value(f"{axis}.none").run()
    stored = _save(app, draft, roots)
    expected = _with_explicit_gap_policy({axis: f"{axis}.none"})
    assert wizard._charter_fields(stored, roots)["axes"] == {
        key: (value,) for key, value in expected.items()
    }
    from ifsm_replication_controls import KEEP_BASELINE

    _fixed_widget(app, axis).set_value(KEEP_BASELINE).run()
    assert _fixed_widget(app, axis).value == KEEP_BASELINE
    stored = _save(app, draft, roots)
    assert stored.step_payload("baseline")["fixed_axis_value_ids"] == _with_explicit_gap_policy()
    assert wizard._charter_fields(stored, roots)["axes"] == {
        key: (value,) for key, value in _with_explicit_gap_policy().items()
    }
    assert len(_enumerate_fixture(wizard._charter_fields(stored, roots))) == 1


def test_htf_cap_two_fixed_widget_save_reload_and_worker_section(monkeypatch, tmp_path):
    roots = _roots(monkeypatch, tmp_path)
    draft = _draft(fixed={"parent_retest_timeout_1m_bars": "parent_retest_timeout_1m_bars.240"})
    app = _open(draft, roots)
    axis = "htf_selection_max_per_timeframe"
    _fixed_widget(app, axis).set_value(f"{axis}.2").run()
    stored = _save(app, draft, roots)
    fields = wizard._charter_fields(stored, roots)
    assert fields["axes"][axis] == (f"{axis}.2",)
    children = _enumerate_fixture(fields)
    assert len(children) == 1
    resolved = resolve_profile_config({"profile_name": BASELINE,
                                       "section_overrides": children[0].section_overrides})
    assert resolved.section.htf_selection_max_per_timeframe == 2
    assert resolved.section.parent_retest_timeout_1m_bars == 240


@pytest.mark.parametrize(("mode", "question", "purpose"), [
    ("single_configuration", "compare_one_with_baseline", "development_research"),
    ("fsm_config_search", "find_robust_fsm", "development_research"),
    ("single_configuration", "evaluate_one_configuration", "implementation_verification"),
])
def test_other_goals_and_verification_ignore_stale_fixed_values(
    monkeypatch, tmp_path, mode, question, purpose,
):
    roots = _roots(monkeypatch, tmp_path)
    draft = _draft(fixed=NUMERIC_CHOICES, mode=mode, question=question, purpose=purpose)
    assert fixed_axis_values(draft) == {}
    assert wizard._charter_fields(draft, roots)["axes"] == {}
    app = _open(draft, roots)
    assert not any("fixed_" in str(widget.key) for widget in app.selectbox)
    assert not roots["research"].exists() and not roots["state"].exists()


def test_fixed_values_refuse_a_list_instead_of_expanding_a_search():
    draft = _draft(fixed={"parent_htf_distance_ticks_max": [
        "parent_htf_distance_ticks_max.80", "parent_htf_distance_ticks_max.160",
    ]})
    with pytest.raises(ValueError, match="exactly one value"):
        fixed_axis_values(draft)


def test_fixed_optional_widgets_follow_actual_core_support(monkeypatch, tmp_path):
    roots = _roots(monkeypatch, tmp_path)
    draft = _draft()
    app = _open(draft, roots)
    keys = {widget.key for widget in app.selectbox}
    expected = _with_explicit_gap_policy()
    for axis, value in OPTIONAL_CHOICES.items():
        supported = axis in SEARCH_AXIS_REGISTRY_V1
        assert any("fixed_" in str(key) and str(key).endswith(f"{BASELINE}_{axis}")
                   for key in keys) == supported
        if supported:
            _fixed_widget(app, axis).set_value(value).run()
            assert not app.exception
            expected[axis] = value
    if "setup_timeout_1m_bars" in SEARCH_AXIS_REGISTRY_V1:
        _fixed_widget(app, "setup_timeout_1m_bars").set_value("setup_timeout_1m_bars.240").run()
        expected["setup_timeout_1m_bars"] = "setup_timeout_1m_bars.240"
    stored = _save(app, draft, roots)
    assert stored.step_payload("baseline")["fixed_axis_value_ids"] == expected
    fields = wizard._charter_fields(stored, roots)
    assert fields["axes"] == {axis: (value,) for axis, value in expected.items()}
    assert len(_enumerate_fixture(fields)) == 1


def _assert_no_replication_picker(app):
    assert not any(widget.key == "ifsm_completed_recipe" for widget in app.selectbox)
    assert not any(widget.key == "ifsm_use_recipe" for widget in app.button)
    assert not any(button.label == "Use this configuration" for button in app.button)
    assert not any("Replicate completed IFSM studies" in str(element.value)
                   for kind in ("markdown", "caption", "subheader")
                   for element in app.get(kind))
    assert not any("Replicate completed IFSM studies" in element.label
                   for element in app.expander)


def _legacy_start_app():
    import ifvg_study_tab as study_tab
    import streamlit as st

    study_tab._render_start(st, study_tab.workspace_roots(st))


def test_focused_home_has_no_replication_picker_or_configuration_fields(monkeypatch, tmp_path):
    roots = _focused_roots(monkeypatch, tmp_path)
    app = apptest.AppTest.from_function(_focused_app, default_timeout=120).run()
    assert not app.exception
    _assert_no_replication_picker(app)
    assert not any("fixed_" in str(widget.key) for widget in app.selectbox)
    assert not roots["draft_root"].exists() and not roots["state_root"].exists()


def test_legacy_start_has_no_replication_picker_or_configuration_fields(monkeypatch, tmp_path):
    roots = _roots(monkeypatch, tmp_path)
    app = apptest.AppTest.from_function(_legacy_start_app, default_timeout=120).run()
    assert not app.exception
    _assert_no_replication_picker(app)
    assert not any("fixed_" in str(widget.key) for widget in app.selectbox)
    assert not roots["drafts"].exists() and not roots["state"].exists()


def test_ordinary_new_evaluate_study_config_edits_survive_save_and_reopen(monkeypatch, tmp_path):
    roots = _focused_roots(monkeypatch, tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Configuring or saving a study must never launch it")

    monkeypatch.setattr(wizard, "_spawn_search_job", forbidden)
    monkeypatch.setattr(wizard, "_freeze_and_launch", forbidden)
    app = apptest.AppTest.from_function(_focused_app, default_timeout=120).run()
    assert not app.exception
    app.button(key="ifvg_workspace_new").click().run()
    assert not app.exception
    _assert_no_replication_picker(app)
    study_type = next(widget for widget in app.radio
                      if widget.label == "What would you like to research?")
    study_type.set_value("Evaluate").run()
    next(button for button in app.button if button.label == "Configure study").click().run()
    assert not app.exception
    assert not any("fixed_" in str(widget.key) for widget in app.selectbox)
    next(button for button in app.button if button.label == "Next").click().run()
    assert not app.exception
    _assert_no_replication_picker(app)
    next(widget for widget in app.selectbox if widget.label == "Configuration").set_value(
        BASELINE,
    ).run()
    assert not app.exception
    choices = _with_explicit_gap_policy(NUMERIC_CHOICES)
    choices.update({axis: value for axis, value in OPTIONAL_CHOICES.items()
                    if axis in SEARCH_AXIS_REGISTRY_V1})
    for axis, value in choices.items():
        _fixed_widget(app, axis).set_value(value).run()
        assert not app.exception
    next(button for button in app.button if button.label == "Save draft").click().run()
    assert not app.exception
    draft_id = app.session_state[f"{wizard.STATE_PREFIX}draft_id"]
    stored = load_draft(roots["draft_root"], draft_id)
    assert stored.step_payload("objective")["question_id"] == "evaluate_one_configuration"
    # repair R7: a new Evaluate study starts on the owner's selected configuration
    # (S0_D80_W1_P1's saved values); the edits made here are applied on top of it
    start, named_id = _named_start()
    expected = {**start, **choices}
    assert stored.step_payload("baseline")["fixed_axis_value_ids"] == expected
    assert stored.step_payload("baseline").get("named_baseline_id") == named_id
    assert stored.step_payload("baseline").get("replication_recipe_id") is None
    assert stored.step_payload("baseline").get("historical_search_id") is None
    fields = wizard._charter_fields(stored, roots)
    assert fields["axes"] == {axis: (value,) for axis, value in expected.items()}
    assert fields["max_child_count"] == 1
    assert len(_enumerate_fixture(fields)) == 1

    # A fresh Streamlit session must reconstruct the widgets from the saved draft.
    reopened = apptest.AppTest.from_function(_focused_app, default_timeout=120)
    reopened.session_state["ifvg_workspace_screen"] = "new"
    reopened.session_state[f"{wizard.STATE_PREFIX}draft_id"] = draft_id
    reopened.run()
    assert not reopened.exception
    _assert_no_replication_picker(reopened)
    for axis, value in choices.items():
        assert _fixed_widget(reopened, axis).value == value
    assert not roots["store_root"].exists() and not roots["state_root"].exists()


def test_focused_goal_visit_preserves_recipe_custom_tie_policy(monkeypatch, tmp_path):
    if not all(axis in SEARCH_AXIS_REGISTRY_V1 for axis in OPTIONAL_CHOICES):
        pytest.skip("Exact historical recipes require the isolated repaired Core")
    roots = _focused_roots(monkeypatch, tmp_path)
    draft = draft_from_recipe("ParentDistance160")
    draft.current_step_key = "objective"
    draft.step_index = 0
    save_draft(roots["draft_root"], draft)
    app = apptest.AppTest.from_function(_focused_app, default_timeout=120)
    app.session_state["ifvg_workspace_screen"] = "new"
    app.session_state[f"{wizard.STATE_PREFIX}draft_id"] = draft.draft_id
    app.run()
    assert not app.exception
    next(button for button in app.button if button.label == "Next").click().run()
    assert not app.exception
    stored = load_draft(roots["draft_root"], draft.draft_id)
    assert stored.current_step_key == "baseline"
    assert stored.step_payload("objective") == draft.step_payload("objective")
    fields = wizard._charter_fields(stored, roots)
    assert fields["objective_policy"].model_dump(mode="json") == catalog()["common"][
        "objective_policy"
    ]


@pytest.mark.parametrize("recipe_id", [recipe["id"] for recipe in catalog()["recipes"]])
def test_completed_recipe_roundtrips_full_section_dates_and_gates(
    monkeypatch, tmp_path, recipe_id,
):
    if not all(axis in SEARCH_AXIS_REGISTRY_V1 for axis in OPTIONAL_CHOICES):
        pytest.skip("Exact historical recipes require the isolated repaired Core")
    roots = _roots(monkeypatch, tmp_path)
    data = catalog()
    assert len(data["recipes"]) == 13
    recipe = next(row for row in data["recipes"] if row["id"] == recipe_id)
    draft = draft_from_recipe(recipe_id)
    app = _open(draft, roots)
    stored = _save(app, draft, roots)
    fixed = stored.step_payload("baseline")["fixed_axis_value_ids"]
    # repair R7: saving without a change keeps the recipe's settings exactly; the
    # missing gap rule resolves to the original rule (checked below)
    assert fixed == draft.step_payload("baseline")["fixed_axis_value_ids"]
    assert stored.step_payload("baseline")["historical_search_id"] == recipe["search_id"]
    fields = wizard._charter_fields(stored, roots)
    assert fields["axes"] == {axis: (value,) for axis, value in fixed.items()}
    assert fields["max_child_count"] == 1
    common = data["common"]
    date_policy = fields["date_policy"].model_dump(mode="json")
    assert date_policy["warmup_dates"] == common["warmup_dates"]
    assert date_policy["replay_dates"] == common["warmup_dates"] + common["evaluation_dates"]
    assert len(common["warmup_dates"]) == 10
    assert len(common["evaluation_dates"]) == 107
    assert "2026-06-11" not in date_policy["replay_dates"]
    assert fields["objective_policy"].model_dump(mode="json") == common["objective_policy"]
    assert fields["cost_policy"].model_dump(mode="json") == common["cost_policy"]
    assert fields["simulation_protocol"].model_dump(mode="json") == common["simulation_protocol"]
    assert fields["seed"] == common["seed"]
    children = _enumerate_fixture(fields)
    assert len(children) == 1
    actual = resolve_profile_config({
        "profile_name": fields["baseline_profile_name"],
        "section_overrides": children[0].section_overrides,
    }).effective_config
    ignored = set(data["comparison_normalization"]["ignored_metadata_fields"])
    actual = json.loads(json.dumps({key: value for key, value in actual.items()
                                    if key not in ignored}))
    expected = dict(recipe["normalized_effective_section"])
    for field, default in (
        ("opposing_min_gap_ticks", None),
        ("htf_gap_invalidation_policy", "execution_wick_full_fill_v1"),
        ("htf_direction_selection_policy", "mixed_direction_rank_v1"),
        ("holding_policy", "legacy_unrestricted_v1"),
        ("daily_close_timezone", "America/Chicago"),
        ("daily_close_time", "15:55"),
        ("daily_close_buffer_minutes", 5),
        ("entry_schedule_policy", "legacy_doc_sessions_v1"),
        ("entry_schedule_timezone", "America/Chicago"),
        ("entry_schedule_windows", []),
    ):
        if field in actual:
            expected.setdefault(field, default)
    assert actual == expected
    assert not roots["research"].exists() and not roots["state"].exists()
