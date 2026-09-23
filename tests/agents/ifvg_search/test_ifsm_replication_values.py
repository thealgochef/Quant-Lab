"""Replicate saved IFSM choices without advertising unsupported engine fields."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.axis_values import format_axis_value
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    AxisAuthorizationError,
    assert_axes_authorized,
    resolve_axis_overrides,
)

ROOT = Path(__file__).resolve().parents[3]
OPTIONAL_FIELDS = (
    "setup_timeout_1m_bars",
    "parent_replacement_policy",
    "parent_retest_depth_policy",
)


@pytest.mark.parametrize(
    ("axis", "payload"),
    [
        ("parent_htf_distance_ticks_max", 160),
        ("opposing_parent_distance_ticks_max", 160),
        ("parent_reaction_window_parent_bars", 20),
        ("parent_retest_timeout_1m_bars", 60),
        ("parent_retest_timeout_1m_bars", 240),
        ("opposing_timeout_1m_bars", 90),
        ("htf_registry_max_age_days", 15),
    ],
)
def test_researched_numeric_choices_resolve_without_implicit_approval(axis, payload):
    value_id = f"{axis}.{payload}"
    assert value_id in SEARCH_AXIS_REGISTRY_V1[axis].registered_values
    value = AXIS_VALUE_REGISTRY_V1[value_id]
    assert value.payload == payload
    assert value.owner_ratification_status == "pending"
    assert_axes_authorized({axis: value_id}, require_ratified=False)
    with pytest.raises(AxisAuthorizationError, match="ratification"):
        assert_axes_authorized({axis: value_id}, require_ratified=True)
    resolved = resolve_profile_config(
        {"section_overrides": resolve_axis_overrides({axis: value_id})}
    )
    assert getattr(resolved.section, axis) == payload
    restored = type(value).model_validate(value.model_dump(mode="json"))
    assert restored.value_id == value_id and restored.payload == payload


@pytest.mark.parametrize(
    ("axis", "value", "expected"),
    [
        ("setup_timeout_1m_bars", None, "No setup lifetime limit (unbounded)"),
        ("setup_timeout_1m_bars", 180, "180 1m bars"),
        (
            "parent_replacement_policy",
            "preserve_selected",
            "Preserve the selected parent (preserve_selected)",
        ),
        (
            "parent_retest_depth_policy",
            "strictly_before_ce",
            "First touch strictly before the midpoint (strictly_before_ce)",
        ),
    ],
)
def test_replication_value_labels_keep_units_and_exact_policy_values(axis, value, expected):
    assert format_axis_value(axis, value) == expected


def _probe_registry(core_src: Path | None = None) -> dict:
    """A clean subprocess prevents either installed Core from leaking into the other."""
    paths = [str(ROOT / "src"), *([str(core_src)] if core_src else [])]
    program = f"""
import json,sys
sys.path[:0] = {paths!r}
import strategy_core
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    SEARCH_AXIS_REGISTRY_V1 as axes, AXIS_VALUE_REGISTRY_V1 as values,
    AxisAuthorizationError, assert_axes_authorized, resolve_axis_overrides,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    AXIS_GROUP_ORDER, axis_group_for, axis_renders_widget,
)
fields = set(IfvgSmcSection.model_fields) - {{'profile_name'}}
assert set(axes) == fields, (sorted(set(axes)-fields), sorted(fields-set(axes)))
expected = {{
    'setup_timeout_1m_bars': {{'none':None, '180':180, '240':240}},
    'parent_replacement_policy': {{'highest_tf_newest':'highest_tf_newest',
                                  'preserve_selected':'preserve_selected'}},
    'parent_retest_depth_policy': {{'any_live_touch':'any_live_touch',
                                  'strictly_before_ce':'strictly_before_ce'}},
}}
found = {{}}
for key, payloads in expected.items():
    assert (key in axes) == (key in fields)
    if key not in axes:
        continue
    spec = axes[key]
    assert axis_renders_widget(spec)
    assert axis_group_for(key) in AXIS_GROUP_ORDER
    assert spec.requires_full_sequential_replay and not spec.changes_capture_artifacts
    assert set(spec.registered_values) == {{key+'.'+token for token in payloads}}
    for token, payload in payloads.items():
        value_id = key+'.'+token
        assert values[value_id].payload == payload
        assert values[value_id].owner_ratification_status == 'pending'
        assert_axes_authorized({{key:value_id}}, require_ratified=False)
        try:
            assert_axes_authorized({{key:value_id}}, require_ratified=True)
        except AxisAuthorizationError:
            pass
        else:
            raise AssertionError('pending value acquired real authorization')
        overrides = resolve_axis_overrides({{key:value_id}})
        resolved = resolve_profile_config({{'section_overrides':overrides}})
        assert getattr(resolved.section,key) == payload
    found[key] = {{'group':axis_group_for(key), 'values':list(spec.registered_values)}}
print(json.dumps({{'core_import':strategy_core.__file__, 'optional':found,
                  'field_count':len(fields), 'registry_count':len(axes)}}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def test_installed_runtime_registry_is_complete_and_hides_unsupported_choices():
    result = _probe_registry()
    assert result["field_count"] == result["registry_count"]


def test_pinned_repaired_runtime_registry_supports_exact_research_choices():
    configured = os.environ.get("IFSM_REPAIRED_CORE_SOURCE")
    core_src = (
        Path(configured)
        if configured
        else (ROOT / "reports/ifsm_parent_selection_20260909/workspace/strategy_core/src")
    )
    if not (core_src / "strategy_core").is_dir():
        pytest.skip("Optional pinned IFSM research checkout is not present on this host")
    result = _probe_registry(core_src.resolve())
    assert Path(result["core_import"]).is_relative_to(core_src.resolve())
    assert set(result["optional"]) == set(OPTIONAL_FIELDS)
    assert result["optional"]["setup_timeout_1m_bars"]["group"] == "Staleness"
    for key in OPTIONAL_FIELDS[1:]:
        assert result["optional"][key]["group"] == "Parent Handling"
