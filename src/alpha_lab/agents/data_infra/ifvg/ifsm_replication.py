"""Completed IFSM recipe loading and exact, one-configuration draft creation.

Recipes are configuration evidence, never inherited launch authority. No source
bars, studies or approvals are opened or created by this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from .profiles import resolve_profile_config
from .search.axis_registry import (
    SEARCH_AXIS_REGISTRY_V1,
    assert_axes_authorized,
    resolve_axis_overrides,
)
from .search.identities import canonicalize_section
from .study_drafts import new_draft

LEGACY_MORNING = "enabled_entry_sessions.ny_0700_1030"
CORRECTED_MORNING = "enabled_entry_sessions.morning_chicago_0700_1030_v1"


def uses_legacy_morning(draft) -> bool:
    fixed = draft.step_payload("baseline").get("fixed_axis_value_ids") or {}
    axes = draft.step_payload("search_space").get("axis_selections") or {}
    return fixed.get("enabled_entry_sessions") == LEGACY_MORNING or LEGACY_MORNING in (
        axes.get("enabled_entry_sessions") or ()
    )


def corrected_morning_copy(draft):
    """Create a separately identified draft; retain the legacy source unchanged."""
    from .study_drafts import clone_draft

    if not uses_legacy_morning(draft):
        raise ValueError("This draft does not use the historical morning preset.")
    assert_axes_authorized(
        {"enabled_entry_sessions": CORRECTED_MORNING}, require_ratified=False,
    )
    result = clone_draft(draft)
    result.display_name = f"{draft.display_name} (corrected Chicago morning)"
    fixed = result.steps.get("baseline", {}).get("fixed_axis_value_ids") or {}
    if fixed.get("enabled_entry_sessions") == LEGACY_MORNING:
        fixed["enabled_entry_sessions"] = CORRECTED_MORNING
    axes = result.steps.get("search_space", {}).get("axis_selections") or {}
    if LEGACY_MORNING in (axes.get("enabled_entry_sessions") or ()):
        axes["enabled_entry_sessions"] = [
            CORRECTED_MORNING if value == LEGACY_MORNING else value
            for value in axes["enabled_entry_sessions"]
        ]
    return result


def catalog() -> dict:
    return json.loads(Path(__file__).with_name("ifsm_replay_recipes.json").read_text(
        encoding="utf-8"
    ))


def recipe_axis_values(recipe: dict) -> dict[str, str]:
    """Resolve every actual nondefault field to an available registered value."""
    from .search.axis_registry import AXIS_VALUE_REGISTRY_V1

    expected = recipe["normalized_effective_section"]
    baseline = resolve_profile_config().effective_config
    selected = {}
    for field, value in expected.items():
        if field not in baseline:
            raise ValueError(
                f"This runtime lacks {field}. Open the isolated IFSM research UI."
            )
        if baseline[field] == value:
            continue
        spec = SEARCH_AXIS_REGISTRY_V1[field]
        matches = [
            value_id for value_id in spec.registered_values
            if json.loads(json.dumps(
                getattr(AXIS_VALUE_REGISTRY_V1[value_id], "payload", "__composite__")
            )) == value
        ]
        if len(matches) != 1:
            raise ValueError(f"No unique selectable value for {field} = {value!r}")
        selected[field] = matches[0]
    assert_axes_authorized(selected, require_ratified=False)
    resolved = resolve_profile_config({"section_overrides": resolve_axis_overrides(selected)})
    actual = resolved.effective_config
    assert all(actual[field] == value for field, value in expected.items())
    return selected


def fixed_axis_values(draft) -> dict[str, str]:
    """Only Evaluate One consumes the fixed settings from its Configuration step."""
    if (
        draft.mode_id != "single_configuration"
        or draft.step_payload("objective").get("question_id") != "evaluate_one_configuration"
        or (draft.purpose_annotation or {}).get("purpose") == "implementation_verification"
    ):
        return {}
    selected = dict(draft.step_payload("baseline").get("fixed_axis_value_ids") or {})
    if any(not isinstance(value, str) for value in selected.values()):
        raise ValueError("Each fixed configuration setting requires exactly one value.")
    assert_axes_authorized(selected, require_ratified=False)
    return selected


def export_fixed_configuration(baseline_name: str, selected: dict[str, str]) -> str:
    """Portable authoring settings only; never includes launch authorization."""
    resolved = resolve_profile_config({"profile_name": baseline_name})
    if any(not isinstance(value, str) for value in selected.values()):
        raise ValueError("Each fixed configuration setting requires exactly one value.")
    assert_axes_authorized(selected, require_ratified=False)
    effective = resolve_profile_config({
        "profile_name": baseline_name, "section_overrides": resolve_axis_overrides(selected),
    })
    return json.dumps({
        "schema_version": 1,
        "baseline_profile_name": baseline_name,
        "baseline_section_config_hash": resolved.section_config_hash,
        "fixed_axis_value_ids": selected,
        "effective_section": effective.effective_config,
    }, indent=2, sort_keys=True)


def import_fixed_configuration(content: str | bytes, baseline_name: str) -> dict[str, str]:
    """Verify the baseline and full effective section before accepting settings."""
    payload = json.loads(content)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version", "baseline_profile_name", "baseline_section_config_hash",
        "fixed_axis_value_ids", "effective_section",
    }:
        raise ValueError("Use a configuration exported by this study screen.")
    if payload["schema_version"] != 1 or payload["baseline_profile_name"] != baseline_name:
        raise ValueError("Choose the exported baseline configuration before importing.")
    selected = payload["fixed_axis_value_ids"]
    if not isinstance(selected, dict):
        raise ValueError("The imported fixed settings must be a mapping.")
    # Recompute instead of trusting stale source defaults or supplied hashes.
    expected = json.loads(export_fixed_configuration(baseline_name, selected))
    if payload != expected:
        raise ValueError("The exported baseline or effective configuration has changed.")
    return dict(selected)


def draft_from_recipe(recipe_id: str):
    data = catalog()
    recipe = next(row for row in data["recipes"] if row["id"] == recipe_id)
    selected = recipe_axis_values(recipe)
    common = data["common"]
    base = resolve_profile_config({"profile_name": data["baseline_profile_name"]})
    from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

    draft = new_draft("single_configuration", display_name=f"Replicate {recipe['name']}")
    draft.purpose_annotation = {
        "schema_version": 1, "purpose": "development_research",
        "derivation": "card_selected", "owner_confirmed": True,
        "updated_at": draft.created_at_utc,
    }
    objective = common["objective_policy"]
    draft.steps = {
        "objective": {
            "mode_id": "single_configuration", "question_id": "evaluate_one_configuration",
            "template_id": "custom", "custom_objectives": objective["pareto_objectives"],
            "custom_tie_breaks": objective["lexicographic_tie_breaks"],
        },
        "baseline": {
            "baseline_profile_name": data["baseline_profile_name"],
            "baseline_section_config_hash": ifvg_profile_hash(canonicalize_section(base.section)),
            "baseline_blocked_reason": None, "fixed_axis_value_ids": selected,
            "replication_recipe_id": recipe_id,
            "historical_search_id": recipe["search_id"],
        },
        "benchmarks": {
            "strategy_gates": objective["feasibility_gates"],
            "robustness_gates": objective["robustness_gates"], "prop_gates": {},
        },
        "validation": {
            "run_scope": "full_authorized_development", "evidence_class": "real",
            "real_dates": common["evaluation_dates"], "warmup_dates": common["warmup_dates"],
            "seed": common["seed"], "worker_limit": 1,
        },
    }
    draft.current_step_key = "baseline"
    draft.step_index = 1
    return draft
