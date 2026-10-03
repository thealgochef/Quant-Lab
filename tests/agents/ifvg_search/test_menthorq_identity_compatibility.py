"""Neutral level fields preserve historical effective shapes and generated names."""

import hashlib
import json

from strategy_core.strategies.ifvg_smc.section import (
    MENTHORQ_NEUTRAL_PROFILE_FIELDS,
    default_ifvg_smc_section,
)

from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import SEARCH_AXIS_REGISTRY_V1
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_profile_id_for,
    name_free_section_hash,
)

_NEUTRAL_FIELDS = {
    "menthorq_context_version": None,
    "regime_gate_policy": "off",
    "regime_unknown_policy": "allow",
    "nearest_support_gex1_block": False,
    "nearest_support_universe": "all_19",
}


def test_neutral_effective_projection_preserves_old_full_shape_and_typed_axes():
    resolved = resolve_profile_config()
    full = resolved.section.model_dump(mode="json")
    historical = {key: value for key, value in full.items() if key not in _NEUTRAL_FIELDS}
    assert MENTHORQ_NEUTRAL_PROFILE_FIELDS == _NEUTRAL_FIELDS
    assert resolved.effective_config == historical
    for key, default in _NEUTRAL_FIELDS.items():
        assert full[key] == default
        assert getattr(resolved.section, key) == default
        assert key in type(resolved.section).model_fields
        assert key in SEARCH_AXIS_REGISTRY_V1
    restored = type(resolved.section).model_validate(historical)
    assert restored == resolved.section
    assert resolved.section_config_hash == (
        "e0f318732cb59d844ac14b5e3839862146e7da1f612f9884f767247f66dd39dd"
    )


def test_historical_name_free_hash_and_every_valid_nondefault_remain_distinct():
    base = default_ifvg_smc_section()
    historical = {
        key: value for key, value in base.model_dump(mode="json").items()
        if key not in {"profile_name", *_NEUTRAL_FIELDS}
    }
    if historical.get("exit_policy") == "fixed_target_v1":
        historical.pop("exit_policy")
    encoded = json.dumps(historical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    old_hash = hashlib.sha256(encoded).hexdigest()
    assert old_hash == "c48be3d011ce38004790b6bb6c6808798e35287947efae63a511b13b552d48e5"
    assert name_free_section_hash(base) == old_hash
    assert canonical_profile_id_for(base) == f"ifvg_search_profile_{old_hash[:16]}"
    context_overrides = {"menthorq_context_version": "menthorq_eod_v1"}
    context = base.model_validate({**base.model_dump(), **context_overrides})
    assert name_free_section_hash(context) != old_hash
    assert resolve_profile_config({"section_overrides": context_overrides}).effective_config[
        "menthorq_context_version"
    ] == "menthorq_eod_v1"
    context_hash = name_free_section_hash(context)
    for key, value in (
        ("regime_gate_policy", "positive_only"),
        ("regime_gate_policy", "negative_only"),
        ("regime_unknown_policy", "block"),
        ("nearest_support_gex1_block", True),
        ("nearest_support_universe", "studied_8"),
    ):
        overrides = {**context_overrides, key: value}
        changed = base.model_validate({**base.model_dump(), **overrides})
        assert name_free_section_hash(changed) != context_hash
        assert canonical_profile_id_for(changed) != canonical_profile_id_for(context)
        projected = resolve_profile_config({"section_overrides": overrides})
        assert projected.effective_config[key] == value
        assert projected.effective_config["menthorq_context_version"] == "menthorq_eod_v1"
        for neutral_key, default in _NEUTRAL_FIELDS.items():
            if getattr(projected.section, neutral_key) == default:
                assert neutral_key not in projected.effective_config
