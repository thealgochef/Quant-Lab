"""Behavioral wording checks for the deterministic strategy description."""

from dataclasses import FrozenInstanceError

import pytest
from strategy_core.strategies.ifvg_smc.section import (
    IfvgSmcSection,
    default_ifvg_smc_section,
    ict_clean_fresh_ifvg_smc_section,
)

from alpha_lab.agents.data_infra.ifvg.presentation.strategy_rules import (
    _CLASSIFIED_FIELDS,
    describe_strategy,
    describe_strategy_variations,
)


def _section(**updates):
    return IfvgSmcSection.model_validate({
        **default_ifvg_smc_section().model_dump(mode="json"), **updates,
    })


def _text(description):
    return "\n".join(description.detail_bullets + description.variations)


def test_baseline_has_concise_conditional_rules_with_real_session_times():
    result = describe_strategy(_section())

    assert result.status == "available"
    assert len(result.preview_bullets) == 3
    assert 6 <= len(result.detail_bullets) <= 10
    assert len(_text(result).split()) <= 410
    assert result.detail_bullets[0].startswith("Enter during Asia 16:00–01:45")
    assert "New York 08:00–14:00" in _text(result)
    assert "Eastern time" in _text(result)
    assert "end times excluded" in _text(result)
    assert "at least 4 ticks" in _text(result)
    assert "40 candles of its own timeframe" in _text(result)
    assert "price to return to it (a retest)" in result.preview_bullets[0]


def test_baseline_uses_actual_stop_anchor_and_exit_order_without_extra_gates():
    text = _text(describe_strategy(_section()))

    assert "1 tick below the lowest price since the main-zone retest" in text
    assert "including the entry candle" in text
    assert "profit target the same distance from entry as the stop" in text
    assert "next one-minute candle" in text
    assert "count the stop first" in text
    assert "Session or trading-day changes do not close the trade" in text
    assert "sweep" not in text.lower()
    assert "confirmed swing" not in text.lower()
    assert "16:40" not in text
    assert "0.70" not in text


def test_staleness_study_has_shared_rules_and_four_distinct_wait_policies():
    base = _section()
    result = describe_strategy_variations(base, [
        ("Baseline", base),
        *[(f"Wait {limit}", _section(parent_retest_timeout_1m_bars=limit))
          for limit in (240, 360, 480)],
    ])

    assert result.status == "available"
    assert len(result.preview_bullets) == 3
    assert result.preview_bullets[-1] == (
        "Compare no main-zone retest timeout with limits of 240, 360 and 480 one-minute candles."
    )
    assert len(result.variations) == 2
    assert "no time limit for the main-zone retest" in result.variations[0]
    assert "other invalidation rules still apply" in result.variations[1]
    for limit in (240, 360, 480):
        assert f"more than {limit} candles" in result.variations[0]
    assert "from the current main zone's selection" in result.variations[1]
    assert "replacement main zone restarts the count" in result.variations[1]
    assert "recorded one-minute candles" in result.variations[1]
    assert "not clock time" in result.variations[1]
    assert "time limit for this retest" not in " ".join(result.detail_bullets)
    assert "returns to the main zone (a retest), continue" in _text(result)
    assert "trading rules match" not in _text(result)


def test_dependency_disabled_does_not_invent_an_entry_distance_filter():
    base = _section()
    inactive = _section(entry_near_parent=False, entry_parent_distance_ticks_max=40)
    active = _section(entry_near_parent=True, entry_parent_distance_ticks_max=40)

    same = describe_strategy_variations(base, [("Inactive distance", inactive)])
    changed = describe_strategy_variations(base, [("Nearby entry", active)])

    assert same.variations == ()
    assert "entry price must also" not in _text(describe_strategy(inactive)).lower()
    assert "entry price must also be within 40 ticks" in _text(changed)


def test_timeout_stages_and_replacement_are_not_conflated():
    text = _text(describe_strategy(_section(
        opposing_timeout_1m_bars=50, inversion_timeout_1m_bars=70,
        post_inversion_expiry_1m_bars_max=25,
    )))

    assert "more than 50 one-minute candles pass after the retest" in text
    assert "more than 70 one-minute candles pass after selecting the opposing gap" in text
    assert "newer qualifying opposing gap replaces the old one and restarts its wait" in text
    assert "within the next 25 one-minute candles" in text
    assert "more than 25 candles have passed" in text


def test_exact_staleness_variant_preserves_the_selection_clock():
    text = _text(describe_strategy(_section(parent_retest_timeout_1m_bars=240)))

    assert "more than 240 one-minute candles pass after selecting the main zone" in text
    assert "without a retest" in text
    assert "replacement main zone restarts this wait" in text
    assert "recorded candles, not clock time" in text


def test_invalidations_are_independent_and_pre_retest_recovery_is_preserved():
    text = _text(describe_strategy(_section()))

    assert "price completely fills the main zone or a candle" in text
    assert "before its retest, seek another main zone" in text
    assert "after the retest, abandon the setup" in text


def test_disabled_invalidation_policies_do_not_claim_to_cancel_setups():
    text = _text(describe_strategy(_section(
        parent_full_fill_invalidation=False, parent_structural_invalidation=False,
    )))

    assert "A main-zone fill or a close through its far edge does not cancel the setup" in text
    assert "completely fills the original higher-timeframe gap, abandon" in text


@pytest.mark.parametrize("section", [
    ict_clean_fresh_ifvg_smc_section(),
    _section(entry_family="ifvg_retest"),
])
def test_blocked_profiles_never_get_executable_entry_instructions(section):
    result = describe_strategy(section)

    assert result.status == "unavailable"
    assert result.preview_bullets == ()
    assert result.detail_bullets == ()
    assert result.issues


def test_inert_break_even_setting_is_not_explained_as_an_active_rule():
    result = describe_strategy(_section(break_even_enabled=True))

    assert result.status == "partial"
    assert "The stop stays fixed" in _text(result)
    assert any("current engine does not apply it" in issue for issue in result.issues)


def test_full_section_is_required_so_historical_missing_fields_do_not_get_defaults():
    raw = _section().model_dump(mode="json")
    del raw["parent_retest_timeout_1m_bars"]

    result = describe_strategy(raw)

    assert result.status == "unavailable"
    assert "incomplete" in result.issues[0]
    assert not result.detail_bullets


@pytest.mark.parametrize("missing", ["resolver_policy", "anchor_policy"])
def test_engine_policy_defaults_are_not_supplied_to_saved_mappings(missing):
    raw = _section().model_dump(mode="json")
    del raw[missing]

    assert describe_strategy(raw).status == "unavailable"


@pytest.mark.parametrize("metadata", [
    {"strategy_id": "touch_reversal"},
    {"strategy_version": "1"},
    {"resolver_policy": "target_first"},
])
def test_unknown_engine_semantics_are_not_guessed(metadata):
    result = describe_strategy(_section(), metadata)

    assert result.status == "unavailable"
    assert not result.detail_bullets


def test_new_fields_are_visible_as_partial_coverage_without_echoing_machine_names():
    raw = _section().model_dump(mode="json")
    raw["future_entry_gate"] = True

    result = describe_strategy(raw)

    assert result.status == "partial"
    assert "not yet covered" in result.issues[0]
    assert "future_entry_gate" not in _text(result)


def test_nondefault_selection_and_retention_are_explained_in_variations():
    result = describe_strategy_variations(_section(), [("More zones", _section(
        htf_selection_max_per_timeframe=2, htf_registry_max_age_days=20,
        ltf_registry_max_live=100,
    ))])

    assert "2 most recent active" in _text(result)
    assert "older than 20 days" in _text(result)
    assert "at most 100 active gaps" in _text(result)


def test_exact_selection_and_retention_variants_show_their_rules():
    result = describe_strategy(_section(
        htf_selection_max_per_timeframe=2, htf_registry_max_age_days=20,
        ltf_registry_max_live=100,
    ))

    assert len(result.detail_bullets) == 10
    assert "2 most recent active" in _text(result)
    assert "older than 20 days" in _text(result)
    assert "at most 100 active gaps" in _text(result)


def test_cartesian_search_groups_repeated_rule_changes():
    configurations = [
        (f"Wait {timeout}, target {target}", _section(
            parent_retest_timeout_1m_bars=timeout, tp_r_multiple=target,
        ))
        for timeout in (240, 360)
        for target in (2.0, 3.0)
    ]

    result = describe_strategy_variations(_section(), configurations)

    # Two baseline rules, two distinct timeouts and two distinct exits,
    # rather than repeating both rules for each of four combinations.
    assert len(result.variations) == 6
    assert _text(result).count("more than 240 one-minute candles") == 1
    assert _text(result).count("Set a fixed profit target 2 times") == 1


def test_disabled_baseline_does_not_invent_shared_short_trading_rules():
    result = describe_strategy_variations(
        _section(enable_longs=False, enable_shorts=False), [("Long version", _section())],
    )

    assert "takes no trades" in result.detail_bullets[0]
    assert "selling opportunities" not in _text(result)
    assert any("Long version: Look for buying opportunities" in text for text in result.variations)


def test_short_direction_target_and_entry_session_changes_are_all_explained():
    result = describe_strategy_variations(_section(), [("Short mornings", _section(
        enable_longs=False, enable_shorts=True, enabled_entry_sessions=("ny",),
        tp_r_multiple=2.0, max_executed_trades_per_day=2,
        outside_session_policy="reset_setup_as_missed",
    ))])

    text = _text(result)
    assert "Look for selling opportunities" in text
    assert "close below its lower edge" in text
    assert "above the highest price since the main-zone retest" in text
    assert "2 times the initial distance to the stop" in text
    assert "Enter during New York 08:00–14:00" in text
    assert "After 2 executed trades" in text
    assert "outside these windows, abandon that setup" in text


def test_source_sections_are_unchanged_and_results_are_immutable():
    raw = _section().model_dump(mode="json")
    before = _section().model_dump(mode="json")
    result = describe_strategy(raw)

    assert raw == before
    with pytest.raises(FrozenInstanceError):
        result.status = "unavailable"


def test_every_current_engine_field_has_an_explicit_description_classification():
    assert set(IfvgSmcSection.model_fields) == _CLASSIFIED_FIELDS
