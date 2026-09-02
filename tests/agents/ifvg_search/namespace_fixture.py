"""Test helpers for the HARDENING-BACKEND semantic namespace (§4.1 / §4.2).

Every test store that carries owner authority is EXPLICITLY marked as a
``test`` namespace here (the class is stated, never inferred from the tmp
path); the helpers return the store's verified namespace id and its
CURRENT supersession-head witness so authorization refs and bundles bind
to the exact store they authorize.
"""

from __future__ import annotations

from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    OwnerAuthorizationBundle,
    VerificationAuthorizationRef,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    SupersessionHeadWitness,
    initialize_test_namespace,
    namespace_class_of,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    current_supersession_head_witness,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import VERIFICATION_POLICY_ID

__all__ = [
    "namespace_and_witness",
    "verification_authorization_ref",
    "owner_authorization_bundle",
    "DUMMY_NAMESPACE_ID",
    "dummy_witness",
]

DUMMY_NAMESPACE_ID = "e" * 64


def dummy_witness(store_namespace_id: str = DUMMY_NAMESPACE_ID) -> SupersessionHeadWitness:
    """A witness for contract-only tests that never bind to a store."""

    return SupersessionHeadWitness(
        store_namespace_id=store_namespace_id, line_count=0, head_sha256="f" * 64
    )


def namespace_and_witness(root: Path) -> tuple[str, SupersessionHeadWitness]:
    """Mark ``root`` as a test namespace (idempotent) and return its verified
    namespace id with the current supersession-head witness."""

    root = Path(root)
    if namespace_class_of(root) is None:
        initialize_test_namespace(root)
    witness = current_supersession_head_witness(root)
    return witness.store_namespace_id, witness


def verification_authorization_ref(
    root: Path | None,
    *,
    approved_allowlist_hash: str,
    coverage_matrix_artifact_id: str,
    seed_snapshot_id: str,
    approved_by: str = "owner",
    approved_at: str = "2026-08-18T00:00:00Z",
    content_hash: str = "d" * 64,
    verification_policy_id: str = VERIFICATION_POLICY_ID,
) -> VerificationAuthorizationRef:
    """A verification authorization bound to ``root`` (or, with ``None``, to
    the dummy namespace for contract-only tests)."""

    if root is None:
        namespace_id, witness = DUMMY_NAMESPACE_ID, dummy_witness()
    else:
        namespace_id, witness = namespace_and_witness(root)
    return VerificationAuthorizationRef(
        verification_policy_id=verification_policy_id,
        approved_allowlist_hash=approved_allowlist_hash,
        coverage_matrix_artifact_id=coverage_matrix_artifact_id,
        seed_snapshot_id=seed_snapshot_id,
        approved_by=approved_by,
        approved_at=approved_at,
        content_hash=content_hash,
        store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )


def owner_authorization_bundle(
    root: Path | None, *, requirement_set_id: str, decision_refs
) -> OwnerAuthorizationBundle:
    if root is None:
        namespace_id, witness = DUMMY_NAMESPACE_ID, dummy_witness()
    else:
        namespace_id, witness = namespace_and_witness(root)
    return OwnerAuthorizationBundle(
        requirement_set_id=requirement_set_id,
        decision_refs=decision_refs,
        store_namespace_id=namespace_id,
        supersession_head_witness=witness,
    )
