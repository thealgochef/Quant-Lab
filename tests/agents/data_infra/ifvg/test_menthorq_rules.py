"""A1 fields join the existing rule groups without changing neutral descriptions."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.strategy_rules import (
    _CLASSIFIED_FIELDS,
    describe_strategy,
    describe_strategy_variations,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config

_NEUTRAL = {
    "menthorq_context_version": None,
    "regime_gate_policy": "off",
    "regime_unknown_policy": "allow",
    "nearest_support_gex1_block": False,
}


def _section(**overrides):
    return resolve_profile_config({"section_overrides": overrides}).section


def test_neutral_fields_and_their_historical_absence_keep_exact_concise_rules():
    section = _section()
    present = section.model_dump(mode="json") | _NEUTRAL
    absent = {key: value for key, value in present.items() if key not in _NEUTRAL}
    current = describe_strategy(present)
    historical = describe_strategy(absent)
    assert set(_NEUTRAL) <= _CLASSIFIED_FIELDS
    assert current.status == historical.status == "available"
    assert current == historical == describe_strategy(section)
    assert len(current.preview_bullets) == 3
    assert not current.issues
    assert all("MenthorQ" not in text for text in (*current.preview_bullets,
                                                  *current.detail_bullets))


def test_context_only_is_available_and_does_not_describe_an_active_entry_filter():
    description = describe_strategy(_section(menthorq_context_version="menthorq_eod_v1"))
    assert description.status == "available"
    assert len(description.preview_bullets) == 3
    assert "context is recorded" in description.preview_bullets[1]
    details = " ".join(description.detail_bullets)
    assert "6:00 AM to 5:00 PM America/Chicago" in details
    assert "end time excluded" in details
    assert "Outside those hours, both level entry gates are skipped" in details
    assert "both level entry gates are off" in details


@pytest.mark.parametrize("policy,regime", [("positive_only", "positive"),
                                          ("negative_only", "negative")])
@pytest.mark.parametrize("unknown,handling", [("allow", "passes"), ("block", "blocks")])
def test_regime_rule_names_required_regime_and_inside_hours_missing_handling(
    policy, regime, unknown, handling,
):
    description = describe_strategy(_section(
        menthorq_context_version="menthorq_eod_v1", regime_gate_policy=policy,
        regime_unknown_policy=unknown,
    ))
    assert description.status == "available"
    assert len(description.preview_bullets) == 3
    details = " ".join(description.detail_bullets)
    assert f"only in the {regime} dealer-gamma regime" in details
    assert f"missing level row or unknown regime inside the window {handling} this gate" in details
    assert "Outside those hours, both level entry gates are skipped" in details


def test_nearest_support_rule_preserves_ties_strict_below_and_unknown_pass():
    description = describe_strategy(_section(
        menthorq_context_version="menthorq_eod_v1", nearest_support_gex1_block=True,
        regime_unknown_policy="block",
    ))
    assert description.status == "available"
    details = " ".join(description.detail_bullets)
    assert "GEX 1 is among the levels tied" in details
    assert "strictly below the entry price" in details
    assert "level equal to entry is excluded" in details
    assert "no lower level or unavailable context passes this gate" in details
    assert "unknown regime inside the window blocks" not in details


def test_shorts_enabled_description_skips_both_configured_level_gates():
    description = describe_strategy(_section(
        menthorq_context_version="menthorq_eod_v1", regime_gate_policy="positive_only",
        regime_unknown_policy="block", nearest_support_gex1_block=True, enable_shorts=True,
    ))
    assert description.status == "available"
    details = " ".join(description.detail_bullets)
    assert "both level entry gates are skipped" in details
    assert "also skipped while selling is enabled" in details
    assert "allow a buy only" not in details
    assert "Block a buy when" not in details


def test_enabled_level_controls_are_visible_in_existing_entry_variations():
    base = _section()
    alternative = _section(
        menthorq_context_version="menthorq_eod_v1", regime_gate_policy="positive_only",
        nearest_support_gex1_block=True,
    )
    description = describe_strategy_variations(base, [("Level rules", alternative)])
    assert description.status == "available"
    assert len(description.preview_bullets) == 3
    variations = " ".join(description.variations)
    assert "Level rules:" in variations
    assert "positive dealer-gamma regime" in variations
    assert "GEX 1 is among the levels tied" in variations
