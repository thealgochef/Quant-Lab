"""Lifecycle and fixed-draft explanations follow the actual research section."""

from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.ifsm_replication import catalog, draft_from_recipe
from alpha_lab.agents.data_infra.ifvg.presentation.strategy_rules import describe_strategy
from alpha_lab.agents.data_infra.ifvg.presentation.study_rules import _draft_rules
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1
from alpha_lab.agents.data_infra.ifvg.study_drafts import new_draft

OPTIONAL_DEFAULTS = {
    "setup_timeout_1m_bars": None,
    "parent_replacement_policy": "highest_tf_newest",
    "parent_retest_depth_policy": "any_live_touch",
}


def _text(description):
    return " ".join(description.detail_bullets)


def _draft(**changes):
    draft = new_draft("single_configuration", display_name="Preview settings")
    draft.purpose_annotation = {"purpose": "development_research"}
    draft.steps = {
        "objective": {"question_id": "evaluate_one_configuration"},
        "baseline": {
            "baseline_profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "fixed_axis_value_ids": changes,
        },
        "validation": {"run_scope": "full_authorized_development"},
    }
    return draft


def test_fixed_draft_explains_its_selected_distance_without_a_baseline_comparison():
    draft = _draft(parent_htf_distance_ticks_max="parent_htf_distance_ticks_max.160")
    result = _draft_rules(SimpleNamespace(draft=draft))
    assert result.status == "available" and result.is_preview
    assert "within 160 ticks of the touched zone" in _text(result)
    assert "within 80 ticks of the touched zone" not in _text(result)
    assert result.variations == ()


@pytest.mark.parametrize("change", ["verification", "other_goal"])
def test_rule_preview_ignores_fixed_payload_outside_evaluate_one(change):
    draft = _draft(parent_htf_distance_ticks_max="parent_htf_distance_ticks_max.160")
    if change == "verification":
        draft.purpose_annotation = {"purpose": "implementation_verification"}
        draft.steps["validation"]["run_scope"] = "verification_5d"
    else:
        draft.steps["objective"]["question_id"] = "compare_one_with_baseline"
    result = _draft_rules(SimpleNamespace(draft=draft))
    assert "within 80 ticks of the touched zone" in _text(result)
    assert "within 160 ticks of the touched zone" not in _text(result)


def test_legacy_absent_optional_fields_retain_their_established_rule_defaults():
    section = resolve_profile_config().effective_config
    legacy = {key: value for key, value in section.items() if key not in OPTIONAL_DEFAULTS}
    explicit = {**legacy, **OPTIONAL_DEFAULTS}
    old = describe_strategy(legacy)
    current = describe_strategy(explicit)
    assert old.status == current.status == "available"
    assert old.detail_bullets == current.detail_bullets


@pytest.mark.parametrize(("field", "value", "phrases"), [
    ("setup_timeout_1m_bars", 180,
     ("more than 180 recorded one-minute candles", "original higher-timeframe touch",
      "do not restart this clock", "does not close an open trade")),
    ("parent_replacement_policy", "preserve_selected",
     ("Keep the selected main zone", "do not replace it or restart its wait",
      "newly confirmed eligible gaps only", "original selection clock")),
    ("parent_retest_depth_policy", "strictly_before_ce",
     ("first later overlap", "strictly less than half its width",
      "midpoint or deeper clears", "including new arrivals on that candle",
      "Full-fill and structural invalidation take precedence")),
])
def test_optional_lifecycle_rules_are_explicit_or_refused_on_an_unsupported_engine(
    field, value, phrases,
):
    section = {**resolve_profile_config().effective_config, field: value}
    result = describe_strategy(section)
    if field not in SEARCH_AXIS_REGISTRY_V1:
        assert result.status == "unavailable"
        assert "isolated research implementation" in result.issues[0]
        assert not result.detail_bullets
        return
    assert result.status == "available"
    for phrase in phrases:
        assert phrase in _text(result)


def test_all_completed_recipe_previews_are_complete_and_use_exact_settings():
    if not all(field in SEARCH_AXIS_REGISTRY_V1 for field in OPTIONAL_DEFAULTS):
        pytest.skip("Exact historical recipes require the isolated repaired Core")
    for recipe in catalog()["recipes"]:
        draft = draft_from_recipe(recipe["id"])
        result = _draft_rules(SimpleNamespace(draft=draft))
        expected = describe_strategy(recipe["full_effective_section"])
        assert result.status == "available", (recipe["id"], result.issues)
        assert result.detail_bullets == expected.detail_bullets, recipe["id"]
        assert result.variations == ()
