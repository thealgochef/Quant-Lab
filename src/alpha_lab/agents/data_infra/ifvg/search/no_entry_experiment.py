"""Exact B/D pairs for the one evidence-supported no-entry research policy."""

from __future__ import annotations

import copy
import json

from ..study_drafts import new_draft
from . import htf_cap_experiment as baseline

INPUT_BUNDLE_ID = baseline.INPUT_BUNDLE_ID
STUDY_NAME = "IFVG no-entry drought: enabled-direction HTF admission"
FIELD = "htf_direction_selection_policy"
BASELINE_POLICY = "mixed_direction_rank_v1"
CHALLENGER_POLICY = "enabled_before_rank_v1"
MEMBERS = (
    ("B0", "B_cap2_opp80", 80, BASELINE_POLICY),
    ("D0", "D_cap2_opp160", 160, BASELINE_POLICY),
    ("B1", "B_cap2_opp80", 80, CHALLENGER_POLICY),
    ("D1", "D_cap2_opp160", 160, CHALLENGER_POLICY),
)


def fixed_profile_drafts(control, charter_payload):
    source = dict(baseline.fixed_profile_drafts(control, charter_payload))
    drafts = []
    for name, reference, _, policy in MEMBERS:
        draft = new_draft("single_configuration", display_name=f"{STUDY_NAME} / {name}")
        draft.steps = copy.deepcopy(source[reference].steps)
        draft.purpose_annotation = copy.deepcopy(source[reference].purpose_annotation)
        draft.steps["baseline"]["fixed_axis_value_ids"].update(
            baseline.exact_value_ids({FIELD: policy})
        )
        draft.current_step_key = "baseline"
        drafts.append((name, draft))
    return drafts


def assert_effective_matrix(configurations, control):
    baseline.assert_control(control)
    if set(configurations) != {name for name, _, _, _ in MEMBERS}:
        raise ValueError(
            "exactly B0/D0/B1/D1 are required; no additional mechanisms or combinations"
        )
    normalized = {}
    for name, _, distance, policy in MEMBERS:
        actual = json.loads(json.dumps(configurations[name]))
        intended = copy.deepcopy(control)
        intended["section"].update(
            {
                "htf_selection_max_per_timeframe": 2,
                "opposing_parent_distance_ticks_max": distance,
                FIELD: policy,
            }
        )
        actual["section"].pop("profile_name")
        intended["section"].pop("profile_name")
        if actual != intended:
            raise ValueError(
                f"{name}: unexpected full effective configuration or automatic default"
            )
        normalized[name] = actual
    diffs = {}
    for a, b in (("B0", "B1"), ("D0", "D1")):
        changed = sorted(
            k
            for k in normalized[a]["section"]
            if normalized[a]["section"][k] != normalized[b]["section"][k]
        )
        if changed != [FIELD]:
            raise ValueError("policy pair must differ in exactly the frozen categorical field")
        diffs[f"{b}_minus_{a}"] = ["section." + key for key in changed]
    return {
        "passed": True,
        "economic_profiles": 4,
        "finite_profile_list": [m[0] for m in MEMBERS],
        "pair_differences": diffs,
        "excluded_generated_identifiers": ["section.profile_name"],
        "declared_baseline_schema_extension": {FIELD: BASELINE_POLICY},
        "mechanisms": [
            {"field": FIELD, "baseline": BASELINE_POLICY, "challenger": CHALLENGER_POLICY}
        ],
        "combined_policies": False,
    }
