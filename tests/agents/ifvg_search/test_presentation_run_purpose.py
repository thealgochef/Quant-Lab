"""UI-1 §5.6 / §8 — one presentation-only purpose authority (pure)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    ANNOTATION_SCHEMA_VERSION,
    PURPOSE_LABELS,
    EvidenceClass,
    NamespaceState,
    PurposeResolutionStatus,
    RunPurpose,
    RunPurposeAnnotation,
    derive_purpose_from_legacy,
    namespace_state_for_store,
    resolve_draft_purpose,
    resolve_purpose,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    initialize_test_namespace,
    load_store_namespace,
)


def test_purpose_resolves_scope_namespace_class_and_locks_options() -> None:
    verification = resolve_purpose(RunPurpose.IMPLEMENTATION_VERIFICATION)
    assert verification.run_scope == "verification_5d"
    assert verification.namespace_class == "test"
    assert verification.publication_available is False
    assert verification.evidence_class is EvidenceClass.SYNTHETIC_FIXTURE
    assert verification.authorization_class == "synthetic_marker"
    assert "not research evidence" in verification.result_label.lower()
    real = resolve_purpose(
        RunPurpose.IMPLEMENTATION_VERIFICATION, evidence_class=EvidenceClass.REAL
    )
    assert real.authorization_class == "verification_authorization_ref"
    assert real.run_scope == "verification_5d"

    research = resolve_purpose(RunPurpose.DEVELOPMENT_RESEARCH)
    assert research.run_scope == "full_authorized_development"
    assert research.namespace_class == "research"
    assert research.publication_available is False
    assert research.authorization_class == "owner_authorization_bundle"
    assert research.evidence_class is EvidenceClass.REAL
    assert research.stage_plan_policy == "question_derived_restricted"

    full = resolve_purpose(RunPurpose.FULL_AUTHORIZED_DEVELOPMENT)
    assert full.run_scope == "full_authorized_development"
    assert full.namespace_class == "research"
    assert full.publication_available is True  # a separate eligibility gate
    assert full.stage_plan_policy == "full_owner_authorized"
    # RunScope values are unchanged (owner Q1)
    assert {p.run_scope for p in (verification, research, full)} == {
        "verification_5d",
        "full_authorized_development",
    }
    assert set(PURPOSE_LABELS) == set(RunPurpose)


def test_synthetic_evidence_is_confined_to_implementation_verification() -> None:
    with pytest.raises(ValueError, match="synthetic fixture"):
        resolve_purpose(
            RunPurpose.DEVELOPMENT_RESEARCH, evidence_class=EvidenceClass.SYNTHETIC_FIXTURE
        )


def test_namespace_id_comes_only_from_the_verified_envelope(tmp_path) -> None:
    unmarked = namespace_state_for_store(tmp_path / "unmarked", expected_class="test")
    assert unmarked.status == "unmarked"
    assert unmarked.store_namespace_id is None
    marked_root = tmp_path / "marked"
    initialize_test_namespace(marked_root)
    envelope = load_store_namespace(marked_root)
    verified = namespace_state_for_store(marked_root, expected_class="test")
    assert verified.status == "verified"
    assert verified.store_namespace_id == envelope.store_namespace_id
    assert verified.namespace_class == "test"
    mismatch = namespace_state_for_store(marked_root, expected_class="research")
    assert mismatch.status == "class_mismatch"
    assert mismatch.store_namespace_id == envelope.store_namespace_id
    (marked_root / "STORE_NAMESPACE.json").write_text("{not json", encoding="utf-8")
    corrupt = namespace_state_for_store(marked_root, expected_class="test")
    assert corrupt.status == "corrupt"
    assert corrupt.store_namespace_id is None
    resolved = resolve_purpose(RunPurpose.IMPLEMENTATION_VERIFICATION, namespace=verified)
    assert resolved.store_namespace_id == envelope.store_namespace_id
    assert resolved.namespace_status == "verified"
    # a research-looking LOCAL PATH never defines authority: the class comes
    # from the purpose and the id from the verified envelope only
    research_path = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    state = namespace_state_for_store(research_path, expected_class="research")
    assert state.status == "unmarked" and state.store_namespace_id is None


def test_purpose_annotation_round_trips_and_is_non_semantic() -> None:
    annotation = RunPurposeAnnotation(
        purpose=RunPurpose.DEVELOPMENT_RESEARCH,
        derivation="card_selected",
        owner_confirmed=True,
        updated_at="2026-09-04T00:00:00+00:00",
    )
    payload = annotation.to_dict()
    assert payload["schema_version"] == ANNOTATION_SCHEMA_VERSION
    assert RunPurposeAnnotation.from_dict(payload) == annotation
    assert RunPurposeAnnotation.from_dict({"purpose": "nonsense"}) is None
    assert RunPurposeAnnotation.from_dict(None) is None
    # the annotation carries no identity-bearing field
    assert set(payload) == {
        "schema_version",
        "purpose",
        "derivation",
        "owner_confirmed",
        "updated_at",
    }


def test_ambiguous_legacy_drafts_become_purpose_unresolved() -> None:
    assert (
        derive_purpose_from_legacy(run_scope="verification_5d")
        is RunPurpose.IMPLEMENTATION_VERIFICATION
    )
    assert (
        derive_purpose_from_legacy(run_scope="synthetic_fixture")
        is RunPurpose.IMPLEMENTATION_VERIFICATION
    )
    # full_authorized_development alone is ambiguous (Development Research vs
    # Full Authorized Development share the RunScope)
    assert derive_purpose_from_legacy(run_scope="full_authorized_development") is None
    assert (
        derive_purpose_from_legacy(
            run_scope="full_authorized_development",
            stage_plan_kind="full_owner_authorized",
            authorization_kind="owner_authorization_bundle",
        )
        is RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
    )
    assert (
        derive_purpose_from_legacy(
            run_scope="full_authorized_development",
            stage_plan_kind="question_derived_restricted",
        )
        is RunPurpose.DEVELOPMENT_RESEARCH
    )
    assert derive_purpose_from_legacy(run_scope="nonsense") is None

    stored = resolve_draft_purpose(
        {"purpose": "full_authorized_development", "derivation": "card_selected"},
        run_scope="full_authorized_development",
    )
    assert stored.status is PurposeResolutionStatus.RESOLVED
    assert stored.purpose is RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
    assert stored.annotation is not None and stored.annotation.derivation == "card_selected"

    derived = resolve_draft_purpose(None, run_scope="verification_5d")
    assert derived.status is PurposeResolutionStatus.RESOLVED
    assert derived.purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
    assert derived.annotation is not None
    assert derived.annotation.derivation == "derived_from_legacy_scope"
    assert derived.annotation.owner_confirmed is False

    unresolved = resolve_draft_purpose(None, run_scope="full_authorized_development")
    assert unresolved.status is PurposeResolutionStatus.PURPOSE_UNRESOLVED
    assert unresolved.purpose is None
    assert unresolved.annotation is None
    assert "confirm" in unresolved.reason.lower()


def test_resolved_purpose_carries_typed_readiness_without_asserting_it() -> None:
    namespace = NamespaceState(
        namespace_class="test",
        store_namespace_id="a" * 64,
        status="verified",
        detail="verified",
    )
    resolved = resolve_purpose(
        RunPurpose.IMPLEMENTATION_VERIFICATION,
        evidence_class=EvidenceClass.REAL,
        namespace=namespace,
        authorization_readiness="missing",
        authorization_detail="no persisted verification run carries the ref",
    )
    assert resolved.authorization_readiness == "missing"
    assert resolved.freeze_allowed is False
    ready = resolve_purpose(
        RunPurpose.IMPLEMENTATION_VERIFICATION,
        evidence_class=EvidenceClass.REAL,
        namespace=namespace,
        authorization_readiness="ready",
    )
    assert ready.freeze_allowed is True
    synthetic = resolve_purpose(RunPurpose.IMPLEMENTATION_VERIFICATION, namespace=namespace)
    assert synthetic.authorization_readiness == "not_required"
    assert synthetic.freeze_allowed is True
    unmarked = resolve_purpose(
        RunPurpose.DEVELOPMENT_RESEARCH,
        namespace=NamespaceState(
            namespace_class=None, store_namespace_id=None, status="unmarked", detail="x"
        ),
        authorization_readiness="ready",
    )
    # a real purpose over an unmarked store can never freeze (no semantic authority)
    assert unmarked.freeze_allowed is False
    with pytest.raises(ValueError, match="readiness"):
        resolve_purpose(RunPurpose.DEVELOPMENT_RESEARCH, authorization_readiness="maybe")
