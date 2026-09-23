"""Exact, fixed-profile HTF cap batch using the supported saved-draft path.

The four members are four Evaluate One jobs. No default combinations are
injected, and each member requires its own exact saved strategy authorization.
"""

from __future__ import annotations

import json

from ..profiles import resolve_profile_config
from ..study_drafts import new_draft
from .axis_registry import AXIS_VALUE_REGISTRY_V1, assert_axes_authorized

STUDY_NAME = "HTF cap 1 vs 2 - fixed 240/90 waits"
INPUT_BUNDLE_ID = "f3443d0de5c153da9b0ca8dffeb243b700887947fab24d8267a86a8f830d0816"
MEMBERS = (
    ("A_cap1_opp80_control", 1, 80),
    ("B_cap2_opp80", 2, 80),
    ("C_cap1_opp160_control", 1, 160),
    ("D_cap2_opp160", 2, 160),
)
AXES = ("htf_selection_max_per_timeframe", "opposing_parent_distance_ticks_max")
EXCLUDED_EFFECTIVE_IDENTIFIERS = ("section.profile_name",)
FIXED = {
    "parent_retest_timeout_1m_bars": 240,
    "opposing_timeout_1m_bars": 90,
    "inversion_timeout_1m_bars": None,
    "post_inversion_expiry_1m_bars_max": 80,
    "parent_reaction_window_parent_bars": 40,
    "parent_reaction_window_1m_bars_max": None,
    "parent_htf_distance_ticks_max": 160,
    "htf_registry_max_age_days": 15,
    "ltf_registry_max_live": 512,
    "min_gap_ticks_capture": 4,
    "swing_strength_bars": 3,
    "swing_pool_max": 64,
    "htf_timeframes": ["1H", "4H"],
    "parent_timeframes": ["3m", "5m", "10m", "15m", "30m"],
    "entry_near_parent": False,
    "entry_parent_distance_ticks_max": None,
    "enabled_entry_sessions": ["asia", "london", "ny"],
    "outside_session_policy": "keep_waiting",
    "enable_longs": True,
    "enable_shorts": False,
    "entry_family": "fresh_fvg_continuation",
    "parent_full_fill_invalidation": True,
    "parent_structural_invalidation": True,
    "sl_buffer_ticks": 1,
    "tp_r_multiple": 1.0,
    "max_executed_trades_per_day": None,
    "anchor_policy": "trading_day_18et_elapsed_v1",
    "resolver_policy": "next_1m_bar_stop_first_v1",
    "doc_sessions": {
        "asia": ["16:00", "01:45"],
        "london": ["02:00", "07:00"],
        "ny": ["08:00", "14:00"],
    },
}


def assert_control(control):
    section = control["section"]
    for field, expected in FIXED.items():
        if section.get(field, "__missing__") != expected:
            raise ValueError(f"canonical control conflicts with requested {field}")
    if section["htf_selection_max_per_timeframe"] != 1:
        raise ValueError("historical control must have HTF cap 1")
    if section["session_scheme"] != {
        "timezone": "America/New_York",
        "trading_day_boundary": "18:00",
        "closed_window": {"start": "17:00", "end": "18:00", "crosses_midnight": False},
        "sessions": {
            name: {"start": times[0], "end": times[1], "crosses_midnight": name == "asia"}
            for name, times in FIXED["doc_sessions"].items()
        },
    }:
        raise ValueError("session scheme conflicts with exact saved control")


def exact_value_ids(values):
    result = {}
    for field, value in values.items():
        matches = [
            key
            for key, item in AXIS_VALUE_REGISTRY_V1.items()
            if getattr(item, "axis_technical_key", None) == field
            and json.loads(json.dumps(item.payload)) == value
        ]
        if len(matches) != 1:
            raise ValueError(f"no unique registered value for {field}={value!r}")
        result[field] = matches[0]
    assert_axes_authorized(result, require_ratified=False)
    return result


def fixed_profile_drafts(control, charter_payload):
    assert_control(control)
    base_name = charter_payload["baseline_profile_name"]
    base = resolve_profile_config({"profile_name": base_name})
    base_section = base.effective_config
    fields = control["section"]
    unknown = set(fields) - set(base_section)
    if unknown:
        raise ValueError(f"runtime lacks baseline fields: {sorted(unknown)}")
    base_overrides = {
        k: v
        for k, v in fields.items()
        if k not in {"profile_name", "qualification_mode"} and base_section[k] != v
    }
    drafts = []
    objective = charter_payload["objective_policy"]
    dates = charter_payload["date_policy"]
    for name, cap, distance in MEMBERS:
        selected = exact_value_ids({**base_overrides, AXES[0]: cap, AXES[1]: distance})
        draft = new_draft("single_configuration", display_name=f"{STUDY_NAME} / {name}")
        draft.purpose_annotation = {
            "schema_version": 1,
            "purpose": "development_research",
            "derivation": "card_selected",
            "owner_confirmed": True,
            "updated_at": draft.created_at_utc,
        }
        draft.steps = {
            "objective": {
                "mode_id": "single_configuration",
                "question_id": "evaluate_one_configuration",
                "template_id": "custom",
                "custom_objectives": objective["pareto_objectives"],
                "custom_tie_breaks": objective["lexicographic_tie_breaks"],
            },
            "baseline": {
                "baseline_profile_name": base_name,
                "baseline_section_config_hash": base.section_config_hash,
                "baseline_blocked_reason": None,
                "fixed_axis_value_ids": selected,
            },
            "benchmarks": {
                "strategy_gates": objective["feasibility_gates"],
                "robustness_gates": objective["robustness_gates"],
                "prop_gates": {},
            },
            "validation": {
                "run_scope": "full_authorized_development",
                "evidence_class": "real",
                "real_dates": [d for d in dates["replay_dates"] if d not in dates["warmup_dates"]],
                "warmup_dates": dates["warmup_dates"],
                "seed": charter_payload["seed"],
                "worker_limit": 1,
            },
        }
        draft.current_step_key = "baseline"
        drafts.append((name, draft))
    return drafts


def assert_effective_matrix(configurations, control):
    """Compare every effective field, excluding only generated profile names."""
    assert_control(control)
    if set(configurations) != {name for name, _, _ in MEMBERS}:
        raise ValueError("exactly the four named economic profiles are required")
    differences = {}
    normalized = {}
    for name, cap, distance in MEMBERS:
        config = json.loads(json.dumps(configurations[name]))
        expected = json.loads(json.dumps(control))
        section = config["section"]
        for field, value in FIXED.items():
            if section.get(field, "__missing__") != value:
                raise ValueError(f"{name}: fixed {field} differs")
        for field, normal in (
            ("parent_replacement_policy", "highest_tf_newest"),
            ("parent_retest_depth_policy", "any_live_touch"),
        ):
            if field in section:
                expected["section"][field] = normal
        for field, value in zip(AXES, (cap, distance), strict=True):
            expected["section"][field] = value
        config["section"].pop("profile_name")
        expected["section"].pop("profile_name")
        if config != expected:
            raise ValueError(f"{name}: full effective configuration differs from baseline")
        normalized[name] = config
    for left, right in ((MEMBERS[0][0], MEMBERS[1][0]), (MEMBERS[2][0], MEMBERS[3][0])):
        changed = [
            k
            for k in normalized[left]["section"]
            if normalized[left]["section"][k] != normalized[right]["section"][k]
        ]
        if changed != [AXES[0]]:
            raise ValueError("primary pair must differ only in HTF cap")
        differences[f"{right}_minus_{left}"] = ["section." + k for k in changed]
    return {
        "passed": True,
        "economic_profiles": 4,
        "pairs": [[cap, distance] for _, cap, distance in MEMBERS],
        "excluded_generated_identifiers": list(EXCLUDED_EFFECTIVE_IDENTIFIERS),
        "pair_differences": differences,
    }
