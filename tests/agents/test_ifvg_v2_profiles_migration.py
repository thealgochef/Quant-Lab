"""Resolved configuration hashing and exhaustive v1 migration cases."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.migration import (
    migrate_v1_experiment_config,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config


def test_doc_default_profile_is_literal_and_runnable() -> None:
    resolved = resolve_profile_config()
    section = resolved.section
    assert section.profile_name == "ifvg_v2_doc_default_fresh_static_1r"
    assert section.qualification_mode.value == "doc_default"
    assert section.runnable is True
    assert section.execution_enabled is True
    assert section.entry_family == "fresh_fvg_continuation"
    assert section.enable_longs is True
    assert section.enable_shorts is False
    assert section.parent_reaction_window_parent_bars == 40
    assert section.post_inversion_expiry_1m_bars_max == 80
    assert section.max_executed_trades_per_day is None
    assert section.entry_near_parent is False


@pytest.mark.parametrize(
    ("profile_name", "reason_fragment"),
    [
        (
            "ifvg_v2_ict_clean_fresh_static_1r",
            "same-reaction-leg",
        ),
        (
            "ifvg_v2_ict_clean_pure_retest_static_1r",
            "pure retest trigger",
        ),
        (
            "ifvg_v2_weak_counter_displacement_research",
            "candidate-only",
        ),
        (
            "ifvg_v1_legacy_candidate_stream",
            "read-only",
        ),
    ],
)
def test_unresolved_and_legacy_profiles_are_non_runnable(
    profile_name: str,
    reason_fragment: str,
) -> None:
    resolved = resolve_profile_config({"profile_name": profile_name})
    assert resolved.runnable is False
    assert resolved.section.execution_enabled is False
    assert reason_fragment in str(resolved.section.non_runnable_reason)


def test_active_and_evaluator_hash_boundaries() -> None:
    base = resolve_profile_config()
    execution_cap = resolve_profile_config(
        {
            "section_overrides": {
                "max_executed_trades_per_day": 2,
            }
        }
    )
    candidate_cap = resolve_profile_config({"max_candidates_per_day": 2})
    diagnostic = resolve_profile_config({"retest_fraction": 0.75})
    assert execution_cap.section_config_hash != base.section_config_hash
    assert execution_cap.qualification_mode == "custom_profile"
    assert candidate_cap.section_config_hash == base.section_config_hash
    assert candidate_cap.evaluation_config_hash != base.evaluation_config_hash
    assert diagnostic.section_config_hash == base.section_config_hash
    assert diagnostic.evaluation_config_hash == base.evaluation_config_hash
    assert diagnostic.diagnostics["retest_fraction"]["status"] == "diagnostic_only"


def test_non_runnable_profile_cannot_be_ui_escalated() -> None:
    with pytest.raises(ValueError, match="cannot be enabled"):
        resolve_profile_config(
            {
                "profile_name": "ifvg_v2_ict_clean_pure_retest_static_1r",
                "section_overrides": {
                    "runnable": True,
                    "execution_enabled": True,
                },
            }
        )


def test_fixed_bootstrap_contract_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="10,000"):
        resolve_profile_config({"bootstrap_samples": 999})
    with pytest.raises(ValueError, match="0.95"):
        resolve_profile_config({"confidence_level": 0.9})


def test_apply_false_migrates_to_read_only_broad_capture() -> None:
    migrated = migrate_v1_experiment_config(
        {
            "doc_defaults": {"apply": False, "min_gap_ticks": 99},
            "filters": {"max_trades_per_day": 3},
            "selected": True,
        }
    )
    assert migrated["profile_name"] == "ifvg_v1_legacy_candidate_stream"
    assert migrated["qualification_mode"] == "broad_capture"
    assert migrated["execution_enabled"] is False
    assert migrated["legacy_candidate_row_limit"] == 3
    assert migrated["max_executed_trades_per_day"] is None
    assert migrated["legacy_selected_read_only"] is True
    assert migrated["legacy_outcomes_read_only"] is True


def test_apply_true_canonical_values_migrate_to_doc_default() -> None:
    migrated = migrate_v1_experiment_config(
        {
            "doc_defaults": {
                "apply": True,
                "min_gap_ticks": 4,
                "parent_htf_distance_ticks_max": 80,
                "opposing_parent_distance_ticks_max": 80,
                "post_inversion_expiry_1m_bars_max": 80,
            }
        }
    )
    assert migrated["profile_name"] == "ifvg_v2_doc_default_fresh_static_1r"
    assert migrated["qualification_mode"] == "doc_default"
    assert migrated["execution_enabled"] is True


def test_apply_true_deviation_migrates_to_custom_without_trade_promotion() -> None:
    migrated = migrate_v1_experiment_config(
        {
            "doc_defaults": {
                "apply": True,
                "min_gap_ticks": 5,
            },
            "outcome": "win",
            "bars_to_resolution": 0,
        }
    )
    assert migrated["qualification_mode"] == "custom_profile"
    assert migrated["profile_name"] == "ifvg_v2_custom_migrated"
    assert migrated["base_profile_name"] == (
        "ifvg_v2_doc_default_fresh_static_1r"
    )
    assert migrated["section_overrides"] == {"min_gap_ticks_capture": 5}
    resolved = resolve_profile_config(
        {
            "profile_name": migrated["base_profile_name"],
            "section_overrides": migrated["section_overrides"],
        }
    )
    assert resolved.qualification_mode == "custom_profile"
    assert resolved.section.min_gap_ticks_capture == 5
    assert migrated["max_executed_trades_per_day"] is None
    assert migrated["legacy_outcomes_read_only"] is True
