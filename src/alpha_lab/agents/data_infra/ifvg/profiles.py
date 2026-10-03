"""Named IFVG v2 profiles and canonical raw/effective configuration hashes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from strategy_core.strategies.ifvg_smc.section import (
    MENTHORQ_NEUTRAL_PROFILE_FIELDS,
    IfvgSmcSection,
    QualificationMode,
    default_ifvg_smc_section,
    ict_clean_fresh_ifvg_smc_section,
    ict_clean_retest_ifvg_smc_section,
    ifvg_profile_hash,
    legacy_ifvg_smc_section,
    weak_counter_displacement_ifvg_smc_section,
)

__all__ = [
    "PROFILE_BUILDERS",
    "ResolvedProfileConfig",
    "resolve_profile_config",
    "saved_section_matches_profile_hash",
]

PROFILE_BUILDERS = {
    "ifvg_v2_doc_default_fresh_static_1r": default_ifvg_smc_section,
    "ifvg_v2_ict_clean_fresh_static_1r": ict_clean_fresh_ifvg_smc_section,
    "ifvg_v2_ict_clean_pure_retest_static_1r": (
        ict_clean_retest_ifvg_smc_section
    ),
    "ifvg_v2_weak_counter_displacement_research": (
        weak_counter_displacement_ifvg_smc_section
    ),
    "ifvg_v1_legacy_candidate_stream": legacy_ifvg_smc_section,
}

_EVALUATOR_DEFAULTS = {
    "max_candidates_per_day": None,
    "bootstrap_samples": 10_000,
    "confidence_level": 0.95,
}

_DIAGNOSTIC_ONLY = {
    "retest_fraction",
    "retest_min_penetration",
}

_FORBIDDEN_PROFILE_ESCALATIONS = {
    "runnable",
    "execution_enabled",
    "non_runnable_reason",
}


def _hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def saved_section_matches_profile_hash(section: dict, expected_hash: str) -> bool:
    """Verify a preserved mapping without adding today's section defaults.

    Earlier saved records used the exact mapping hash. Full current sections
    use Core's established neutral behavior projection instead. Both checks
    retain every saved non-neutral field, including unknown historical fields;
    neither changes the mapping, its published identity, or artifact bytes.
    """
    if _hash(section) == expected_hash:
        return True
    behavior = dict(section)
    if behavior.get("opposing_min_gap_ticks") is None:
        behavior.pop("opposing_min_gap_ticks", None)
    if behavior.get("holding_policy") == "legacy_unrestricted_v1":
        for key in (
            "holding_policy", "daily_close_timezone", "daily_close_time",
            "daily_close_buffer_minutes",
        ):
            behavior.pop(key, None)
    if behavior.get("entry_schedule_policy") == "legacy_doc_sessions_v1":
        for key in (
            "entry_schedule_policy", "entry_schedule_timezone", "entry_schedule_windows",
        ):
            behavior.pop(key, None)
    for key, default in MENTHORQ_NEUTRAL_PROFILE_FIELDS.items():
        if behavior.get(key) == default:
            behavior.pop(key, None)
    return _hash(behavior) == expected_hash


@dataclass(frozen=True)
class ResolvedProfileConfig:
    raw_ui_config: dict
    effective_config: dict
    evaluator_config: dict
    diagnostics: dict
    section: IfvgSmcSection
    section_config_hash: str
    evaluation_config_hash: str

    @property
    def qualification_mode(self) -> str:
        return self.section.qualification_mode.value

    @property
    def runnable(self) -> bool:
        return self.section.runnable


def resolve_profile_config(raw_ui_config: dict | None = None) -> ResolvedProfileConfig:
    raw = dict(raw_ui_config or {})
    name = raw.get(
        "profile_name", "ifvg_v2_doc_default_fresh_static_1r"
    )
    if name not in PROFILE_BUILDERS:
        raise ValueError(f"unknown IFVG profile {name!r}")
    section = PROFILE_BUILDERS[name]()
    overrides = dict(raw.get("section_overrides") or {})
    if overrides:
        if not section.runnable and _FORBIDDEN_PROFILE_ESCALATIONS & overrides.keys():
            raise ValueError(
                "an unresolved/non-runnable profile cannot be enabled by UI overrides"
            )
        if section.runnable:
            overrides = {
                **overrides,
                "qualification_mode": QualificationMode.CUSTOM_PROFILE,
            }
        section = IfvgSmcSection.model_validate(
            {**section.model_dump(mode="json"), **overrides}
        )

    evaluator = {
        key: raw.get(key, default)
        for key, default in _EVALUATOR_DEFAULTS.items()
    }
    if evaluator["bootstrap_samples"] != 10_000:
        raise ValueError("IFVG v2 bootstrap_samples is fixed at 10,000")
    if evaluator["confidence_level"] != 0.95:
        raise ValueError("IFVG v2 confidence_level is fixed at 0.95")
    candidate_cap = evaluator["max_candidates_per_day"]
    if candidate_cap is not None and (
        not isinstance(candidate_cap, int) or candidate_cap < 1
    ):
        raise ValueError("max_candidates_per_day must be null or a positive integer")
    diagnostics: dict[str, dict] = {}
    for key in sorted(_DIAGNOSTIC_ONLY & raw.keys()):
        diagnostics[key] = {
            "value": raw[key],
            "status": "diagnostic_only",
            "reason": (
                "inactive for the selected fresh-entry profile"
                if section.entry_family == "fresh_fvg_continuation"
                else "unratified retest semantic"
            ),
        }
    effective = section.model_dump(mode="json")
    # The shared neutral map includes B0.2's all_19 universe default and
    # preserves the historical effective-section shape.
    # The typed section still exposes every field to validation and the axes;
    # an enabled context or gate override remains explicit in this projection.
    for key, default in MENTHORQ_NEUTRAL_PROFILE_FIELDS.items():
        if effective.get(key) == default:
            effective.pop(key, None)
    section_hash = ifvg_profile_hash(section)
    evaluation_hash = _hash(
        {
            "section_config_hash": section_hash,
            "evaluator": evaluator,
        }
    )
    return ResolvedProfileConfig(
        raw_ui_config=raw,
        effective_config=effective,
        evaluator_config=evaluator,
        diagnostics=diagnostics,
        section=section,
        section_config_hash=section_hash,
        evaluation_config_hash=evaluation_hash,
    )
