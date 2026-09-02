"""Search-charter suites (TEST_MATRIX §3.1)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.charter import (
    CharterValidationError,
    SearchCharterEnvelope,
    SearchMode,
    _example_charter_payload,
    validate_charter,
)

_AS_OF = "2026-08-18T00:00:00Z"


def test_same_charter_yields_same_search_id() -> None:
    a = SearchCharterEnvelope.from_payload(_example_charter_payload())
    b = SearchCharterEnvelope.from_payload(_example_charter_payload())
    assert a.search_id == b.search_id


def test_semantic_change_changes_search_id_but_presentation_never_enters() -> None:
    base = _example_charter_payload()
    changed = base.model_copy(update={"seed": 8})
    assert (
        SearchCharterEnvelope.from_payload(base).search_id
        != SearchCharterEnvelope.from_payload(changed).search_id
    )
    assert "display_name" not in type(base).model_fields


def test_blocked_axis_value_cannot_freeze() -> None:
    payload = _example_charter_payload().model_copy(
        update={
            "axes": {
                "parent_full_fill_invalidation": (
                    "parent_full_fill_invalidation.baseline",
                )
            }
        }
    )
    with pytest.raises(CharterValidationError):
        validate_charter(payload, as_of_utc=_AS_OF)


def test_unknown_value_cannot_freeze() -> None:
    payload = _example_charter_payload().model_copy(
        update={"axes": {"parent_retest_timeout_1m_bars": ("parent_retest_timeout_1m_bars.777",)}}
    )
    with pytest.raises(CharterValidationError, match="not registered"):
        validate_charter(payload, as_of_utc=_AS_OF)


def test_child_count_ceiling_is_enforced() -> None:
    payload = _example_charter_payload().model_copy(update={"max_child_count": 1})
    with pytest.raises(CharterValidationError, match="max_child_count"):
        validate_charter(payload, as_of_utc=_AS_OF)
    over_ceiling = {
        **_example_charter_payload().model_dump(mode="json"),
        "max_child_count": 257,
    }
    with pytest.raises(ValueError):
        type(payload).model_validate(over_ceiling)


def test_non_runnable_baseline_cannot_freeze() -> None:
    payload = _example_charter_payload().model_copy(
        update={"baseline_profile_name": "ifvg_v2_ict_clean_fresh_static_1r"}
    )
    with pytest.raises(CharterValidationError, match="blocked"):
        validate_charter(payload, as_of_utc=_AS_OF)


def test_prop_mode_requires_a_firm_contract() -> None:
    payload = _example_charter_payload().model_copy(
        update={"search_mode": SearchMode.PROP_BENCHMARK}
    )
    with pytest.raises(CharterValidationError, match="firm contract"):
        validate_charter(payload, as_of_utc=_AS_OF)


def test_real_charter_requires_owner_evidence_for_its_path() -> None:
    from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
    from alpha_lab.agents.data_infra.ifvg.search.authorization import (
        OwnerAuthorizationBundle,
    )
    from alpha_lab.agents.data_infra.ifvg.search.charter import DatePolicy
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
        SupersessionHeadWitness,
    )

    real = _example_charter_payload().model_copy(
        update={
            "owner_authorization": OwnerAuthorizationBundle(
                requirement_set_id="1" * 64,
                decision_refs={},
                store_namespace_id="e" * 64,
                supersession_head_witness=SupersessionHeadWitness(
                    store_namespace_id="e" * 64, line_count=0, head_sha256="f" * 64
                ),
            ),
            "date_policy": DatePolicy(
                replay_dates=(*FROZEN_WARMUP_DATES, "2026-01-13"),
                warmup_dates=FROZEN_WARMUP_DATES,
                access_policy_id="development_explicit_dates_before_path_v2",
            ),
        }
    )
    with pytest.raises(CharterValidationError):
        validate_charter(real, as_of_utc=_AS_OF)


def test_pending_value_accepted_only_under_synthetic_marker() -> None:
    # the example charter IS synthetic and carries a pending 480 value: passes
    validate_charter(_example_charter_payload(), as_of_utc=_AS_OF)


def test_verification_date_policy_hard_caps() -> None:
    from alpha_lab.agents.data_infra.ifvg.search.charter import DatePolicy

    with pytest.raises(ValueError, match="at most five"):
        DatePolicy(
            replay_dates=(
                "2026-06-02",
                "2026-06-03",
                "2026-06-04",
                "2026-06-05",
                "2026-06-08",
                "2026-06-09",
            ),
            warmup_dates=(),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        )
    with pytest.raises(ValueError, match="protected or sealed"):
        DatePolicy(
            replay_dates=("2026-06-11",),
            warmup_dates=(),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        )
    with pytest.raises(ValueError, match="zero real warmup"):
        DatePolicy(
            replay_dates=("2026-06-04",),
            warmup_dates=("2026-06-03",),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        )
    with pytest.raises(ValueError, match="before the permitted development window"):
        DatePolicy(
            replay_dates=("2025-12-31",),
            warmup_dates=(),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        )


def test_generated_baseline_requires_its_capability() -> None:
    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        GeneratedProfileCapability,
    )

    generated = _example_charter_payload().model_copy(
        update={"baseline_profile_name": "ifvg_search_profile_0123456789abcdef"}
    )
    with pytest.raises(CharterValidationError, match="GeneratedProfileCapability"):
        validate_charter(generated, as_of_utc=_AS_OF)
    runnable = GeneratedProfileCapability(
        capability_id="cap-1",
        status="generated_runnable",
        baseline_capability_ref="ifvg_v2_doc_default_fresh_static_1r",
        registry_hash="a" * 64,
        authorization_ref="auth-1",
        validation_report_ref=None,
        reason=None,
    )
    validate_charter(
        generated, as_of_utc=_AS_OF, generated_profile_capability=runnable
    )
    blocked = runnable.model_copy(
        update={"status": "blocked_owner_decision", "reason": "unratified value"}
    )
    with pytest.raises(CharterValidationError, match="blocked_owner_decision"):
        validate_charter(
            generated, as_of_utc=_AS_OF, generated_profile_capability=blocked
        )
    rogue = _example_charter_payload().model_copy(
        update={"baseline_profile_name": "rogue_profile"}
    )
    with pytest.raises(CharterValidationError, match="neither a registered"):
        validate_charter(rogue, as_of_utc=_AS_OF)


def test_synthetic_charters_are_namespace_confined(tmp_path) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter

    envelope = SearchCharterEnvelope.from_payload(_example_charter_payload())
    research_root = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    with pytest.raises(PermissionError, match="confined to test namespaces"):
        save_charter(research_root, envelope)
    test_root = tmp_path / "data" / "ifvg_datasets" / "search_test" / "v1"
    saved, reused = save_charter(test_root, envelope)
    assert reused is False and saved.search_id == envelope.search_id
