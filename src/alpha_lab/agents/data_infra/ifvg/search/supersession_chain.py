"""Immutable owner-decision supersession chain + head witnesses (§4.2; F-12).

R6.1 kept supersessions as hash-chained LINES of a mutable JSONL log beside a
head file; deleting the log and its head restored a superseded decision, and
the earlier plan's in-directory marker would have mutated an immutable
decision artifact. This module never writes into an existing
``owner_decisions/<decision-id>/`` directory. Instead:

* every supersession is an IMMUTABLE store entry
  ``owner_decision_supersessions/<supersession-record-id>/`` —
  :class:`OwnerDecisionSupersessionPayload` binds the namespace, the
  superseded and replacement decision ids, the prior head
  (``prior_head_record_id`` / ``prior_head_sha256`` / ``prior_line_count``),
  the reason, the effective instant and the owner-evidence ref; its id is
  the payload hash (manifest-verified like every store entry);
* the mandatory head pointer ``owner_decisions/SUPERSESSIONS.head`` names
  ``{record_id, line_count, head_sha256}`` where
  ``head_sha256 = chain_head_digest(prior_head_sha256, record_id)`` commits
  to the whole chain back to the namespace's genesis anchor;
* publication runs under the liveness-aware lock: (1) verify namespace,
  current head, source and replacement decisions and nondivergence; (2/3)
  write + verify the immutable record in the store's temporary directory
  and publish it atomically; (4) atomically replace the head. A failure
  before (4) leaves an orphan record with NO authority (the chain follows
  the head only); a failure after (4) leaves the head pointing at an
  already verified immutable record;
* a verified identical replay is idempotent reuse; a divergent record for
  the same transition is refused;
* :func:`current_supersession_head_witness` /
  :func:`assert_head_witness_current` implement the witness rule every real
  charter, authorization bundle, seed-production and verification
  authorization carries — a missing, shorter, or different current head is
  refused (local rollback detection; deleting or rewriting the entire store
  plus every external witness is outside the trust boundary, stated
  honestly).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from .identities import SHA256_PATTERN, EnvelopeBase, FrozenContract, register_identity_pair
from .owner_decision_lock import OwnerDecisionLock
from .store import SearchStoreError, load_verified_envelope, save_or_reuse_envelope
from .store_namespace import (
    StoreNamespaceEnvelope,
    StoreNamespaceError,
    SupersessionHeadWitness,
    chain_head_digest,
    read_supersession_head,
    require_store_namespace,
    write_supersession_head_atomic,
)

__all__ = [
    "OWNER_DECISION_SUPERSESSION_STORE",
    "OwnerDecisionSupersessionPayload",
    "OwnerDecisionSupersessionEnvelope",
    "SupersessionRecord",
    "verify_chain_structure",
    "load_supersession_records",
    "current_supersession_head_witness",
    "assert_head_witness_current",
    "publish_supersession",
]

OWNER_DECISION_SUPERSESSION_STORE = "owner_decision_supersessions"


def _parse_instant(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must carry an explicit UTC offset")
    return parsed


class OwnerDecisionSupersessionPayload(FrozenContract):
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    superseded_decision_id: str = Field(pattern=SHA256_PATTERN)
    replacement_decision_id: str = Field(pattern=SHA256_PATTERN)
    prior_head_record_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prior_head_sha256: str = Field(pattern=SHA256_PATTERN)
    prior_line_count: int = Field(ge=0)
    reason: str = Field(min_length=1)
    effective_at: str
    owner_evidence_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def _well_formed(self):
        if self.superseded_decision_id == self.replacement_decision_id:
            raise ValueError("a decision cannot supersede itself")
        if (self.prior_head_record_id is None) != (self.prior_line_count == 0):
            raise ValueError("prior_head_record_id is null exactly at the genesis head")
        _parse_instant(self.effective_at, field="effective_at")
        return self


class OwnerDecisionSupersessionEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "supersession_record_id"

    supersession_record_id: str = Field(pattern=SHA256_PATTERN)
    payload: OwnerDecisionSupersessionPayload


class SupersessionRecord(FrozenContract):
    """The consumer projection of one chain entry (chain order, 1-based)."""

    supersession_record_id: str = Field(pattern=SHA256_PATTERN)
    line_number: int = Field(ge=1)
    superseded_artifact_id: str = Field(pattern=SHA256_PATTERN)
    replacement_artifact_id: str = Field(pattern=SHA256_PATTERN)
    recorded_at: str
    head_sha256: str = Field(pattern=SHA256_PATTERN)


def _load_record(root: Path, record_id: str) -> OwnerDecisionSupersessionEnvelope:
    try:
        return load_verified_envelope(
            Path(root),
            OWNER_DECISION_SUPERSESSION_STORE,
            record_id,
            OwnerDecisionSupersessionEnvelope,
        )
    except SearchStoreError as error:
        raise StoreNamespaceError(
            "supersession_record_unverifiable",
            f"supersession record {record_id[:12]}… is not a verified store entry ({error})",
        ) from error


def verify_chain_structure(
    root: Path, *, namespace: StoreNamespaceEnvelope | None = None
) -> tuple[OwnerDecisionSupersessionEnvelope, ...]:
    """Walk the head back to the genesis anchor, verifying every record's
    manifest, namespace, prior-head linkage and the head digests; return the
    records in chain order (oldest first). No owner-DECISION artifact is
    loaded here (that is :func:`load_supersession_records`' caller's job)."""

    root = Path(root)
    namespace = namespace or require_store_namespace(root)
    head = read_supersession_head(root, store_namespace_id=namespace.store_namespace_id)
    record_id = head["record_id"]
    line_count = int(head["line_count"])
    digest = str(head["head_sha256"])
    records: list[OwnerDecisionSupersessionEnvelope] = []
    seen: set[str] = set()
    while line_count > 0:
        if record_id is None or record_id in seen:
            raise StoreNamespaceError(
                "supersession_chain_broken", "the supersession chain is cyclic or truncated"
            )
        seen.add(record_id)
        record = _load_record(root, record_id)
        payload = record.payload
        if payload.store_namespace_id != namespace.store_namespace_id:
            raise StoreNamespaceError(
                "supersession_chain_broken",
                f"supersession record {record_id[:12]}… belongs to another namespace",
            )
        if payload.prior_line_count != line_count - 1:
            raise StoreNamespaceError(
                "supersession_chain_broken",
                f"supersession record {record_id[:12]}… does not continue line {line_count - 1}",
            )
        if chain_head_digest(payload.prior_head_sha256, record_id) != digest:
            raise StoreNamespaceError(
                "supersession_chain_broken",
                f"the head digest at line {line_count} does not commit to record "
                f"{record_id[:12]}… (a replaced head or a rewritten record)",
            )
        records.append(record)
        record_id = payload.prior_head_record_id
        line_count = payload.prior_line_count
        digest = payload.prior_head_sha256
    if record_id is not None or digest != namespace.payload.authority_genesis_id:
        raise StoreNamespaceError(
            "supersession_chain_broken",
            "the supersession chain does not terminate at the namespace's genesis anchor",
        )
    ordered = tuple(reversed(records))
    transitions = {
        (r.payload.superseded_decision_id, r.payload.replacement_decision_id) for r in ordered
    }
    if len(transitions) != len(ordered):
        raise StoreNamespaceError(
            "supersession_chain_broken", "the supersession chain repeats a transition"
        )
    return ordered


def load_supersession_records(root: Path) -> tuple[SupersessionRecord, ...]:
    """The structure-verified chain as consumer records (oldest first)."""

    namespace = require_store_namespace(root)
    ordered = verify_chain_structure(root, namespace=namespace)
    out: list[SupersessionRecord] = []
    digest = namespace.payload.authority_genesis_id
    for number, record in enumerate(ordered, start=1):
        digest = chain_head_digest(digest, record.supersession_record_id)
        out.append(
            SupersessionRecord(
                supersession_record_id=record.supersession_record_id,
                line_number=number,
                superseded_artifact_id=record.payload.superseded_decision_id,
                replacement_artifact_id=record.payload.replacement_decision_id,
                recorded_at=record.payload.effective_at,
                head_sha256=digest,
            )
        )
    return tuple(out)


def current_supersession_head_witness(root: Path) -> SupersessionHeadWitness:
    """The verified current head as a witness (namespace + structure checked)."""

    namespace = require_store_namespace(root)
    ordered = verify_chain_structure(root, namespace=namespace)
    digest = namespace.payload.authority_genesis_id
    for record in ordered:
        digest = chain_head_digest(digest, record.supersession_record_id)
    return SupersessionHeadWitness(
        store_namespace_id=namespace.store_namespace_id,
        line_count=len(ordered),
        head_sha256=digest,
    )


def assert_head_witness_current(root: Path, witness: SupersessionHeadWitness) -> None:
    """Refuse a missing, shorter, or different current head (§4.2)."""

    if not isinstance(witness, SupersessionHeadWitness):
        raise StoreNamespaceError(
            "supersession_head_witness_mismatch", "a supersession head witness is required"
        )
    current = current_supersession_head_witness(root)
    if current.store_namespace_id != witness.store_namespace_id:
        raise StoreNamespaceError(
            "supersession_head_witness_mismatch",
            "the witness was recorded against another store namespace",
        )
    if current.line_count < witness.line_count:
        raise StoreNamespaceError(
            "supersession_head_shorter_than_witness",
            f"the current supersession head ({current.line_count} records) is SHORTER than "
            f"the witnessed head ({witness.line_count}) — a rolled-back chain",
        )
    if current.line_count != witness.line_count or current.head_sha256 != witness.head_sha256:
        raise StoreNamespaceError(
            "supersession_head_witness_mismatch",
            f"the current supersession head ({current.line_count}, "
            f"{current.head_sha256[:12]}…) differs from the witnessed head "
            f"({witness.line_count}, {witness.head_sha256[:12]}…); the authorization must be "
            "re-signed against the current head",
        )


def publish_supersession(
    root: Path,
    *,
    superseded_decision_id: str,
    replacement_decision_id: str,
    reason: str,
    effective_at: str,
    owner_evidence_ref: str,
    verify_transition: Callable[[StoreNamespaceEnvelope], None] | None = None,
    lock: OwnerDecisionLock | None = None,
) -> OwnerDecisionSupersessionEnvelope:
    """Publish one immutable supersession record under the owner-decision lock
    and advance the head (the four-step protocol of §4.2). ``verify_transition``
    runs inside the lock after the namespace and head verified — the caller
    verifies the source and replacement decisions there. An identical
    existing record for the same transition is returned (idempotent); a
    divergent one is refused."""

    root = Path(root)
    lock = lock or OwnerDecisionLock(root)
    with lock:
        namespace = require_store_namespace(root)
        ordered = verify_chain_structure(root, namespace=namespace)
        for existing in ordered:
            payload = existing.payload
            if (payload.superseded_decision_id, payload.replacement_decision_id) != (
                superseded_decision_id,
                replacement_decision_id,
            ):
                continue
            if (payload.reason, payload.effective_at, payload.owner_evidence_ref) != (
                reason,
                effective_at,
                owner_evidence_ref,
            ):
                raise StoreNamespaceError(
                    "supersession_divergent_replay",
                    "a supersession record for this transition already exists with different "
                    "content; a divergent replay is refused",
                )
            return existing
        if verify_transition is not None:
            verify_transition(namespace)
        digest = namespace.payload.authority_genesis_id
        for record in ordered:
            digest = chain_head_digest(digest, record.supersession_record_id)
        prior_record_id = ordered[-1].supersession_record_id if ordered else None
        payload = OwnerDecisionSupersessionPayload(
            store_namespace_id=namespace.store_namespace_id,
            superseded_decision_id=superseded_decision_id,
            replacement_decision_id=replacement_decision_id,
            prior_head_record_id=prior_record_id,
            prior_head_sha256=digest,
            prior_line_count=len(ordered),
            reason=reason,
            effective_at=effective_at,
            owner_evidence_ref=owner_evidence_ref,
        )
        envelope = OwnerDecisionSupersessionEnvelope.from_payload(payload)
        lock.refresh()  # heartbeat before the immutable write
        stored, _reused = save_or_reuse_envelope(root, OWNER_DECISION_SUPERSESSION_STORE, envelope)
        lock.verify_held()  # the token must still be ours before the head moves
        write_supersession_head_atomic(
            root,
            store_namespace_id=namespace.store_namespace_id,
            record_id=stored.supersession_record_id,
            line_count=len(ordered) + 1,
            head_sha256=chain_head_digest(digest, stored.supersession_record_id),
        )
        return stored


def _example_payload() -> OwnerDecisionSupersessionPayload:
    return OwnerDecisionSupersessionPayload(
        store_namespace_id="a" * 64,
        superseded_decision_id="b" * 64,
        replacement_decision_id="c" * 64,
        prior_head_record_id=None,
        prior_head_sha256="d" * 64,
        prior_line_count=0,
        reason="example",
        effective_at="2026-09-01T00:00:00+00:00",
        owner_evidence_ref="c" * 64,
    )


register_identity_pair(
    name="OwnerDecisionSupersession",
    envelope_cls=OwnerDecisionSupersessionEnvelope,
    payload_cls=OwnerDecisionSupersessionPayload,
    id_field="supersession_record_id",
    example_factory=_example_payload,
)
