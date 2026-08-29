"""Verified owner-decision evidence for the regime lane (R6.1 §6.F; D5).

R6 gated FEATURE_ELIGIBLE by a 64-hex *format* check on
``owner_ratification_ref``. R6.1 replaces that with a VERIFIED-LOADED
artifact: :class:`OwnerDecisionArtifactEnvelope` (store ``owner_decisions``;
its id IS the ``OwnerDecisionEvidenceRef.content_hash``) binds the EXACT
``resolved_regime_protocol_id`` and ``capability_assessment_id`` it ratifies
and covers decisions **25/28/29/30**: the algorithm key plus the exact
pinned KMeans parameter snapshot/hash (25), grain / interval / stage (28),
k (29), and the occupancy / rows / sample floors and the stability minimum
(30). Every value is re-verified against the loaded registry, protocol, and
assessment (:func:`assert_owner_decision_authorizes`); supersession is
store-owned: one lock-guarded, HASH-CHAINED line per replacement in
``owner_decisions/SUPERSESSIONS.jsonl`` (``prev_line_sha256`` →
``line_sha256``) with the chain head in ``SUPERSESSIONS.head``; the line is
written BEFORE the replacement is published (a crash leaves a dangling line
the loader treats as fail-closed until the replacement is re-persisted, which
is idempotent); every line is backed by a verified replacement artifact whose
``supersedes`` names the prior; an edited, reordered, or deleted line — or a
head that disagrees with the log — fails the whole chain closed; a
replacement may never carry weaker provenance than the prior. Synthetic
provenance is refused outside the ``synthetic_fixture`` run scope, and both
synthetic artifacts and the synthetic scope are confined to test namespaces
(the P0-4 mirror of ``search.charter.save_charter``: a root with a ``search``
segment that is not ``search_test`` refuses them at persist AND at load).

:func:`build_owner_decision_proposal` / :func:`render_owner_decision_proposal_markdown`
emit DRAFTS with placeholders that fail validation (they cannot be
persisted by accident) — the owner's ratification workflow replaces the
placeholders; nothing here is an authorization.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..ml.regime_algorithms import (
    REGIME_ALGORITHM_REGISTRY,
    pinned_parameters_hash,
    resolve_regime_algorithm_entry,
)
from ..ml.regime_contracts import (
    REGIME_PROPOSED_DEFAULTS,
    ObservationGranularity,
    RegimeCapabilityAssessmentEnvelope,
    RegimeProtocolEnvelope,
    RegimeStatus,
    sample_adequacy_minimum,
)
from .authorization import OwnerDecisionEvidenceRef, RunScope
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
    register_identity_pair,
)
from .store import (
    SearchStoreError,
    envelope_destination,
    load_verified_envelope,
    save_or_reuse_envelope,
)

__all__ = [
    "OWNER_DECISION_STORE",
    "SUPERSESSIONS_FILE",
    "SUPERSESSIONS_HEAD_FILE",
    "research_namespace_root",
    "assert_run_scope_lawful_for_root",
    "REGIME_DECISION_ID",
    "REGIME_DECISION_KEYS",
    "AUTHORIZED_TRANSITIONS",
    "OwnerDecisionArtifactPayload",
    "OwnerDecisionArtifactEnvelope",
    "OwnerDecisionRefusalError",
    "SupersessionRecord",
    "expected_decision_values",
    "freeze_decision_values",
    "plain_decision_values",
    "persist_owner_decision",
    "load_owner_decision",
    "load_supersession_chain",
    "assert_owner_decision_authorizes",
    "build_owner_decision_proposal",
    "render_owner_decision_proposal_markdown",
    "synthetic_owner_decision_fixture",
]

OWNER_DECISION_STORE = "owner_decisions"
SUPERSESSIONS_FILE = "SUPERSESSIONS.jsonl"
SUPERSESSIONS_HEAD_FILE = "SUPERSESSIONS.head"
#: The chain anchor of the first supersession line.
_GENESIS_SHA256 = "0" * 64
#: A replacement may never carry WEAKER provenance than the prior it revokes.
_PROVENANCE_RANK: dict[str, int] = {"synthetic_test_authorization_v1": 0, "owner_signed": 1}
#: Lock discipline for the supersession log: bounded wait; a lock older than
#: the stale threshold (a killed writer) is reclaimed instead of wedging every
#: later supersession into a timeout.
_LOCK_WAIT_SECONDS = 30.0
_LOCK_STALE_SECONDS = 60.0
REGIME_DECISION_ID = "25+28+29+30:regime_feature_eligibility"
REGIME_DECISION_KEYS: tuple[str, ...] = (
    "25:regime_algorithm_baseline",
    "28:regime_grain",
    "29:cluster_counts",
    "30:occupancy_stability_gates",
)
AUTHORIZED_TRANSITIONS: tuple[str, ...] = (
    "stratification_ready->feature_eligible",
    "feature_eligible->model_feature",
)
_PLACEHOLDER = "<OWNER_TO_FILL>"
_VALUE_KEYS: tuple[str, ...] = (
    "algorithm_key",
    "algorithm_parameters_hash",
    "algorithm_parameters",
    "observation_granularity",
    "panel_interval_seconds",
    "observation_stage",
    "fixed_cluster_count",
    "minimum_cluster_occupancy_fraction",
    "minimum_cluster_rows_per_fold",
    "minimum_bootstrap_aligned_ami_mean",
    "minimum_training_observations",
)


class OwnerDecisionRefusalError(PermissionError):
    """The owner artifact does not authorize the requested transition."""


def _parse_instant(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must carry an explicit UTC offset")
    return parsed


#: A decision value is a scalar or a (deep-immutable) map of scalars — the
#: exact pinned KMeans parameter snapshot rides as the nested map.
DecisionScalar = str | int | float | bool | None
DecisionValue = DecisionScalar | ImmutableMap[str, DecisionScalar]


class OwnerDecisionArtifactPayload(FrozenContract):
    decision_id: Literal["25+28+29+30:regime_feature_eligibility"] = REGIME_DECISION_ID
    decision_keys: tuple[str, ...]
    decision_values: ImmutableMap[str, DecisionValue]
    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    capability_assessment_id: str = Field(pattern=SHA256_PATTERN)
    authorized_transitions: tuple[str, ...] = Field(min_length=1)
    provenance: Literal["owner_signed", "synthetic_test_authorization_v1"]
    author: str = Field(min_length=1)
    approved_at: str
    effective_from: str
    effective_to: str | None
    reviewed_evidence_refs: tuple[str, ...] = Field(min_length=1)
    supersedes: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _well_formed(self):
        missing = sorted(set(REGIME_DECISION_KEYS) - set(self.decision_keys))
        if missing:
            raise ValueError(f"the regime decision artifact must cover keys {missing}")
        unknown = sorted(set(self.authorized_transitions) - set(AUTHORIZED_TRANSITIONS))
        if unknown:
            raise ValueError(f"unregistered authorized transitions {unknown}")
        absent = sorted(set(_VALUE_KEYS) - set(self.decision_values))
        if absent:
            raise ValueError(f"decision_values lacks required entries {absent}")
        if self.capability_assessment_id not in self.reviewed_evidence_refs:
            raise ValueError(
                "reviewed_evidence_refs must contain the capability assessment id"
            )
        for value in self.decision_values.values():
            if isinstance(value, str) and _PLACEHOLDER in value:
                raise ValueError("a proposal placeholder cannot be persisted as a decision")
            if isinstance(value, dict | list | set):
                raise ValueError(
                    "decision_values must not carry plain mutable containers (nested "
                    "maps are ImmutableMap; the identity payload is deep-immutable)"
                )
        if _PLACEHOLDER in self.author or _PLACEHOLDER in self.rationale:
            raise ValueError("a proposal placeholder cannot be persisted as a decision")
        approved = _parse_instant(self.approved_at, field="approved_at")
        effective = _parse_instant(self.effective_from, field="effective_from")
        if effective < approved:
            raise ValueError("effective_from precedes approved_at")
        if self.effective_to is not None:
            until = _parse_instant(self.effective_to, field="effective_to")
            if until <= effective:
                raise ValueError("effective_to must follow effective_from")
        return self


class OwnerDecisionArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "owner_decision_artifact_id"

    owner_decision_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: OwnerDecisionArtifactPayload

    def evidence_ref(self) -> OwnerDecisionEvidenceRef:
        """The bare reference type the authorization layer carries."""

        return OwnerDecisionEvidenceRef(
            decision_id=self.payload.decision_id,
            decision_artifact_id=self.owner_decision_artifact_id,
            content_hash=self.owner_decision_artifact_id,
            author=self.payload.author,
            approved_at=self.payload.approved_at,
            effective_from=self.payload.effective_from,
            reviewed_evidence_refs=self.payload.reviewed_evidence_refs,
        )


class SupersessionRecord(FrozenContract):
    """One hash-chained supersession line: ``line_sha256`` binds the three
    facts AND ``prev_line_sha256`` (the previous line's digest; the genesis
    anchor for the first line)."""

    superseded_artifact_id: str = Field(pattern=SHA256_PATTERN)
    replacement_artifact_id: str = Field(pattern=SHA256_PATTERN)
    recorded_at: str
    prev_line_sha256: str = Field(pattern=SHA256_PATTERN)
    line_sha256: str = Field(pattern=SHA256_PATTERN)


def research_namespace_root(root: Path) -> bool:
    """The P0-4 namespace rule of ``search.charter.save_charter``: a root
    whose resolved path carries a ``search`` store segment that is not
    ``search_test`` is the research namespace."""

    parts = Path(root).resolve().parts
    return "search" in parts and "search_test" not in parts


def assert_run_scope_lawful_for_root(root: Path, run_scope: str) -> None:
    """The ``synthetic_fixture`` run scope is a test-namespace scope: it may
    never unlock synthetic authority in the research namespace (S3)."""

    if str(run_scope) == "synthetic_fixture" and research_namespace_root(root):
        raise OwnerDecisionRefusalError(
            "the synthetic_fixture run scope is confined to test namespaces; the research "
            "store refuses it (P0-4)"
        )


def _assert_provenance_lawful_for_root(root: Path, payload: OwnerDecisionArtifactPayload) -> None:
    if payload.provenance == "synthetic_test_authorization_v1" and research_namespace_root(root):
        raise OwnerDecisionRefusalError(
            "synthetic_test_authorization_v1 owner decisions are confined to test namespaces; "
            "the research store refuses them (P0-4)"
        )


def freeze_decision_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Nested maps become ``ImmutableMap`` (the identity payload is
    deep-immutable); scalars pass through."""

    return {
        key: (ImmutableMap(dict(value)) if isinstance(value, Mapping) else value)
        for key, value in values.items()
    }


def plain_decision_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """The JSON-friendly (plain dict) form of ``decision_values``."""

    return {
        key: (dict(value) if isinstance(value, Mapping) else value)
        for key, value in values.items()
    }


def expected_decision_values(
    protocol: RegimeProtocolEnvelope, assessment: RegimeCapabilityAssessmentEnvelope
) -> dict[str, Any]:
    """The values decisions 25/28/29/30 must state for THIS protocol and
    assessment — derived from the registry, protocol, and assessment (the
    exact pinned parameter snapshot rides as an ``ImmutableMap``)."""

    payload = protocol.payload
    entry = resolve_regime_algorithm_entry(payload.algorithm_key)
    grain = ObservationGranularity(payload.observation_granularity)
    return {
        "algorithm_key": payload.algorithm_key,
        "algorithm_parameters_hash": pinned_parameters_hash(entry),
        "algorithm_parameters": ImmutableMap(dict(entry.pinned_parameters)),
        "observation_granularity": grain.value,
        "panel_interval_seconds": payload.panel_interval_seconds,
        "observation_stage": payload.observation_stage.value,
        "fixed_cluster_count": int(payload.resolved_cluster_count),
        "minimum_cluster_occupancy_fraction": float(
            assessment.payload.minimum_cluster_occupancy_gate
        ),
        "minimum_cluster_rows_per_fold": int(assessment.payload.minimum_cluster_rows_gate),
        "minimum_bootstrap_aligned_ami_mean": float(
            assessment.payload.stability.minimum_bootstrap_aligned_ami_mean_applied
        ),
        "minimum_training_observations": int(
            assessment.payload.coverage.minimum_training_observations_gate
        ),
    }


def _supersessions_path(root: Path) -> Path:
    return Path(root) / OWNER_DECISION_STORE / SUPERSESSIONS_FILE


def _supersessions_head_path(root: Path) -> Path:
    return Path(root) / OWNER_DECISION_STORE / SUPERSESSIONS_HEAD_FILE


def _line_digest(
    *, superseded_artifact_id: str, replacement_artifact_id: str, recorded_at: str,
    prev_line_sha256: str,
) -> str:
    body = json.dumps(
        {
            "superseded_artifact_id": superseded_artifact_id,
            "replacement_artifact_id": replacement_artifact_id,
            "recorded_at": recorded_at,
            "prev_line_sha256": prev_line_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _chain_refusal(message: str) -> OwnerDecisionRefusalError:
    return OwnerDecisionRefusalError(f"{message}; the chain fails closed")


def _lock_is_stale(lock: Path) -> bool:
    try:
        age = time.time() - lock.stat().st_mtime
    except FileNotFoundError:
        return True
    return age > _LOCK_STALE_SECONDS


def _acquire_lock(lock: Path) -> None:
    """O_EXCL lock with a bounded wait; a stale lock (a killed writer) is
    reclaimed so one crash cannot wedge every later supersession (S10)."""

    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    while True:
        try:
            handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _lock_is_stale(lock):
                _release_lock(lock)
                continue
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "owner-decision supersession log is locked by another writer (a stale "
                    f"lock is reclaimed after {_LOCK_STALE_SECONDS:.0f} s)"
                ) from None
            time.sleep(0.05)
        else:
            try:
                os.write(handle, f"{os.getpid()} {time.time():.3f}\n".encode())
            finally:
                os.close(handle)
            return


def _release_lock(lock: Path) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(lock)


def _read_chain_lines(root: Path) -> list[str]:
    path = _supersessions_path(root)
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_head(root: Path) -> dict[str, Any] | None:
    path = _supersessions_head_path(root)
    if not path.exists():
        return None
    try:
        head = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise _chain_refusal("supersession head is unreadable") from error
    if not isinstance(head, dict):
        raise _chain_refusal("supersession head is malformed")
    return head


def _write_head(root: Path, *, line_count: int, head_sha256: str) -> None:
    path = _supersessions_head_path(root)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(
        json.dumps({"line_count": int(line_count), "head_sha256": head_sha256}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _verify_chain_structure(
    root: Path, *, repair_torn_tail: bool = False
) -> tuple[SupersessionRecord, ...]:
    """Parse + hash-verify the log against its head (no store loads). Any
    malformed, edited, reordered, or deleted line — or a head that disagrees
    with the log — fails closed. With ``repair_torn_tail`` (the appender,
    under the lock) a head lagging by EXACTLY the last chained line (a crash
    between the line append and the head write) is rewritten."""

    lines = _read_chain_lines(root)
    head = _read_head(root)
    if not lines and head is None:
        return ()
    if head is None:
        raise _chain_refusal("supersession log exists without its head file")
    records: list[SupersessionRecord] = []
    prev = _GENESIS_SHA256
    seen: set[tuple[str, str]] = set()
    for number, line in enumerate(lines, start=1):
        try:
            record = SupersessionRecord.model_validate(json.loads(line))
        except (ValueError, TypeError) as error:
            raise _chain_refusal(f"supersession log line {number} is malformed") from error
        expected = _line_digest(
            superseded_artifact_id=record.superseded_artifact_id,
            replacement_artifact_id=record.replacement_artifact_id,
            recorded_at=record.recorded_at,
            prev_line_sha256=prev,
        )
        if record.prev_line_sha256 != prev or record.line_sha256 != expected:
            raise _chain_refusal(
                f"supersession log line {number} breaks the hash chain (an edited, "
                "reordered, or deleted line)"
            )
        key = (record.superseded_artifact_id, record.replacement_artifact_id)
        if key in seen:
            raise _chain_refusal(f"supersession log line {number} duplicates an earlier line")
        seen.add(key)
        records.append(record)
        prev = record.line_sha256
    head_count = head.get("line_count")
    head_sha = head.get("head_sha256")
    if head_count == len(records) and head_sha == prev:
        return tuple(records)
    torn = (
        repair_torn_tail
        and len(records) >= 1
        and head_count == len(records) - 1
        and head_sha == (records[-2].line_sha256 if len(records) >= 2 else _GENESIS_SHA256)
    )
    if torn:
        _write_head(root, line_count=len(records), head_sha256=prev)
        return tuple(records)
    raise _chain_refusal(
        "supersession head does not match the log (a deleted or edited line, or a "
        "replaced head)"
    )


def _append_supersession(
    root: Path, *, superseded_artifact_id: str, replacement_artifact_id: str, recorded_at: str
) -> SupersessionRecord:
    """Lock-guarded, idempotent, hash-chained append + head update. Called
    BEFORE the replacement is published."""

    path = _supersessions_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    _acquire_lock(lock)
    try:
        records = _verify_chain_structure(root, repair_torn_tail=True)
        for record in records:
            if (record.superseded_artifact_id, record.replacement_artifact_id) == (
                superseded_artifact_id,
                replacement_artifact_id,
            ):
                return record
        prev = records[-1].line_sha256 if records else _GENESIS_SHA256
        record = SupersessionRecord(
            superseded_artifact_id=superseded_artifact_id,
            replacement_artifact_id=replacement_artifact_id,
            recorded_at=recorded_at,
            prev_line_sha256=prev,
            line_sha256=_line_digest(
                superseded_artifact_id=superseded_artifact_id,
                replacement_artifact_id=replacement_artifact_id,
                recorded_at=recorded_at,
                prev_line_sha256=prev,
            ),
        )
        with path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        _write_head(root, line_count=len(records) + 1, head_sha256=record.line_sha256)
        return record
    finally:
        _release_lock(lock)


def load_owner_decision(root: Path, artifact_id: str) -> OwnerDecisionArtifactEnvelope:
    """Exact-ID verified load; a synthetic-provenance artifact in the
    research namespace is refused at load as well as at persist (P0-4)."""

    envelope = load_verified_envelope(
        Path(root), OWNER_DECISION_STORE, artifact_id, OwnerDecisionArtifactEnvelope
    )
    _assert_provenance_lawful_for_root(Path(root), envelope.payload)
    return envelope


def load_supersession_chain(root: Path) -> tuple[SupersessionRecord, ...]:
    """Every recorded supersession — the hash chain and head verified, each
    line backed by a VERIFIED replacement artifact whose ``supersedes`` names
    the superseded id — fail closed on any inconsistency."""

    root = Path(root)
    records = _verify_chain_structure(root)
    for record in records:
        try:
            replacement = load_owner_decision(root, record.replacement_artifact_id)
        except SearchStoreError as error:
            raise _chain_refusal(
                "supersession log names a replacement that is not a verified store "
                f"entry ({record.replacement_artifact_id[:12]}…)"
            ) from error
        if replacement.payload.supersedes != record.superseded_artifact_id:
            raise _chain_refusal(
                "supersession log line is not backed by its replacement artifact "
                f"(replacement {record.replacement_artifact_id[:12]}… does not name "
                f"{record.superseded_artifact_id[:12]}… as superseded)"
            )
    return records


def persist_owner_decision(
    root: Path, envelope: OwnerDecisionArtifactEnvelope, *, recorded_at: str
) -> OwnerDecisionArtifactEnvelope:
    """Persist (save-or-reuse). A supersession requires the prior to load
    verified, to name the same protocol, and to carry no stronger provenance
    than the replacement; the chained log line is appended (idempotently)
    BEFORE the replacement is published, so a crash in between leaves the
    chain fail-closed until this call is repeated. Synthetic provenance is
    confined to test namespaces (P0-4)."""

    root = Path(root)
    payload = envelope.payload
    _assert_provenance_lawful_for_root(root, payload)
    if payload.supersedes is not None:
        try:
            prior = load_owner_decision(root, payload.supersedes)
        except SearchStoreError as error:
            raise OwnerDecisionRefusalError(
                "supersession refused: the prior decision is not a verified store entry"
            ) from error
        if prior.payload.resolved_regime_protocol_id != payload.resolved_regime_protocol_id:
            raise OwnerDecisionRefusalError(
                "supersession refused: the prior decision names a different regime protocol"
            )
        if _PROVENANCE_RANK[payload.provenance] < _PROVENANCE_RANK[prior.payload.provenance]:
            raise OwnerDecisionRefusalError(
                f"supersession refused: a {payload.provenance} decision cannot supersede a "
                f"{prior.payload.provenance} decision (weaker provenance)"
            )
        _append_supersession(
            root,
            superseded_artifact_id=payload.supersedes,
            replacement_artifact_id=envelope.owner_decision_artifact_id,
            recorded_at=recorded_at,
        )
    stored, _reused = save_or_reuse_envelope(root, OWNER_DECISION_STORE, envelope)
    return stored


def _values_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) < 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not (isinstance(expected, Mapping) and isinstance(actual, Mapping)):
            return False
        return json.dumps(dict(expected), sort_keys=True, default=str) == json.dumps(
            dict(actual), sort_keys=True, default=str
        )
    return expected == actual


def assert_owner_decision_authorizes(
    artifact: OwnerDecisionArtifactEnvelope,
    *,
    protocol_envelope: RegimeProtocolEnvelope,
    assessment_envelope: RegimeCapabilityAssessmentEnvelope,
    transition: str,
    as_of: str,
    run_scope: RunScope,
    supersession_chain: tuple[SupersessionRecord, ...] = (),
) -> None:
    """Exact refusals, in order: provenance scope; superseded; effectivity;
    protocol; assessment reviewed; decision values 25/28/29/30; transition;
    passing assessment."""

    payload = artifact.payload
    if payload.provenance == "synthetic_test_authorization_v1" and run_scope != "synthetic_fixture":
        raise OwnerDecisionRefusalError(
            "owner decision refused: synthetic_test_authorization_v1 provenance is lawful "
            "only in the synthetic_fixture run scope"
        )
    superseded = {record.superseded_artifact_id for record in supersession_chain}
    if artifact.owner_decision_artifact_id in superseded:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact is superseded by a later decision"
        )
    instant = _parse_instant(as_of, field="as_of")
    if _parse_instant(payload.effective_from, field="effective_from") > instant:
        raise OwnerDecisionRefusalError("owner decision refused: the decision is not yet effective")
    if payload.effective_to is not None and (
        _parse_instant(payload.effective_to, field="effective_to") <= instant
    ):
        raise OwnerDecisionRefusalError("owner decision refused: the decision has expired")
    if payload.resolved_regime_protocol_id != protocol_envelope.resolved_regime_protocol_id:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact ratifies a different regime protocol"
        )
    if payload.capability_assessment_id != assessment_envelope.regime_capability_assessment_id:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the artifact reviewed a different capability assessment"
        )
    if assessment_envelope.payload.resolved_regime_protocol_id != (
        protocol_envelope.resolved_regime_protocol_id
    ):
        raise OwnerDecisionRefusalError(
            "owner decision refused: the assessment does not belong to the protocol"
        )
    expected = expected_decision_values(protocol_envelope, assessment_envelope)
    mismatches = sorted(
        key
        for key, value in expected.items()
        if not _values_equal(value, payload.decision_values.get(key))
    )
    if mismatches:
        raise OwnerDecisionRefusalError(
            "owner decision refused: decision values disagree with the protocol / "
            f"registry / assessment on {mismatches}"
        )
    if transition not in payload.authorized_transitions:
        raise OwnerDecisionRefusalError(
            f"owner decision refused: transition {transition!r} is not authorized by the artifact"
        )
    if not assessment_envelope.payload.gates_passed:
        raise OwnerDecisionRefusalError(
            "owner decision refused: the reviewed capability assessment did not pass its gates"
        )


def transition_key(previous: RegimeStatus, proposed: RegimeStatus) -> str:
    return f"{RegimeStatus(previous).value}->{RegimeStatus(proposed).value}"


# ── proposals (drafts that cannot persist) ───────────────────────────────────


def build_owner_decision_proposal(
    protocol: RegimeProtocolEnvelope,
    assessment: RegimeCapabilityAssessmentEnvelope,
    *,
    transition: str = AUTHORIZED_TRANSITIONS[0],
) -> dict[str, Any]:
    """A draft payload with owner placeholders — it FAILS validation."""

    if transition not in AUTHORIZED_TRANSITIONS:
        raise ValueError(f"unregistered transition {transition!r}")
    values = plain_decision_values(expected_decision_values(protocol, assessment))
    return {
        "decision_id": REGIME_DECISION_ID,
        "decision_keys": list(REGIME_DECISION_KEYS),
        "decision_values": values,
        "resolved_regime_protocol_id": protocol.resolved_regime_protocol_id,
        "capability_assessment_id": assessment.regime_capability_assessment_id,
        "authorized_transitions": [transition],
        "provenance": "owner_signed",
        "author": _PLACEHOLDER,
        "approved_at": _PLACEHOLDER,
        "effective_from": _PLACEHOLDER,
        "effective_to": None,
        "reviewed_evidence_refs": [assessment.regime_capability_assessment_id],
        "supersedes": None,
        "rationale": _PLACEHOLDER,
        "_proposal_note": (
            "PROPOSAL — nothing here is an authorization; placeholders fail "
            "validation until the owner completes the ratification workflow"
        ),
    }


def render_owner_decision_proposal_markdown(proposal: dict[str, Any], *, title: str) -> str:
    values = proposal["decision_values"]
    lines = [
        f"# {title}",
        "",
        "**PROPOSAL — nothing here is an authorization.** The owner's ratification "
        "workflow replaces every placeholder; a payload carrying a placeholder cannot "
        "be persisted as an owner decision.",
        "",
        f"- decision_id: `{proposal['decision_id']}`",
        f"- resolved_regime_protocol_id: `{proposal['resolved_regime_protocol_id']}`",
        f"- capability_assessment_id: `{proposal['capability_assessment_id']}`",
        f"- authorized_transitions: {proposal['authorized_transitions']}",
        "",
        "| Decision | Key | Proposed value (proposed_protocol_default) |",
        "|---|---|---|",
        f"| 25 | algorithm_key | `{values['algorithm_key']}` |",
        f"| 25 | algorithm_parameters_hash | `{values['algorithm_parameters_hash']}` |",
        "| 25 | algorithm_parameters | "
        f"`{json.dumps(dict(values['algorithm_parameters']), sort_keys=True)}` |",
        f"| 28 | observation_granularity | `{values['observation_granularity']}` |",
        f"| 28 | panel_interval_seconds | `{values['panel_interval_seconds']}` |",
        f"| 28 | observation_stage | `{values['observation_stage']}` |",
        f"| 29 | fixed_cluster_count | `{values['fixed_cluster_count']}` |",
        "| 30 | minimum_cluster_occupancy_fraction | "
        f"`{values['minimum_cluster_occupancy_fraction']}` |",
        f"| 30 | minimum_cluster_rows_per_fold | `{values['minimum_cluster_rows_per_fold']}` |",
        "| 30 | minimum_bootstrap_aligned_ami_mean | "
        f"`{values['minimum_bootstrap_aligned_ami_mean']}` |",
        f"| 30 | minimum_training_observations | `{values['minimum_training_observations']}` |",
        "",
        "Owner fields to complete: `author`, `approved_at` (ISO-8601 with offset), "
        "`effective_from`, optional `effective_to`, `rationale`; provenance stays "
        "`owner_signed`.",
        "",
        "```json",
        json.dumps(
            {k: v for k, v in proposal.items() if not k.startswith("_")},
            indent=2,
            sort_keys=True,
            default=str,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def synthetic_owner_decision_fixture(
    root: Path,
    *,
    protocol: RegimeProtocolEnvelope,
    assessment: RegimeCapabilityAssessmentEnvelope,
    transitions: tuple[str, ...] = AUTHORIZED_TRANSITIONS,
    approved_at: str = "2026-08-28T00:00:00+00:00",
    effective_from: str = "2026-08-28T00:00:00+00:00",
    effective_to: str | None = None,
    supersedes: str | None = None,
    value_overrides: dict[str, Any] | None = None,
    persist: bool = True,
) -> OwnerDecisionArtifactEnvelope:
    """A SYNTHETIC-provenance decision (lawful in the synthetic run scope
    only) over the exact protocol + assessment."""

    values = expected_decision_values(protocol, assessment)
    values.update(value_overrides or {})
    payload = OwnerDecisionArtifactPayload(
        decision_keys=REGIME_DECISION_KEYS,
        decision_values=freeze_decision_values(values),
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        capability_assessment_id=assessment.regime_capability_assessment_id,
        authorized_transitions=tuple(transitions),
        provenance="synthetic_test_authorization_v1",
        author="synthetic_fixture",
        approved_at=approved_at,
        effective_from=effective_from,
        effective_to=effective_to,
        reviewed_evidence_refs=(assessment.regime_capability_assessment_id,),
        supersedes=supersedes,
        rationale="synthetic test authorization over the exact assessment",
    )
    envelope = OwnerDecisionArtifactEnvelope.from_payload(payload)
    if persist:
        return persist_owner_decision(Path(root), envelope, recorded_at=approved_at)
    return envelope


def _example_payload() -> OwnerDecisionArtifactPayload:
    entry = REGIME_ALGORITHM_REGISTRY["kmeans_v1"]
    return OwnerDecisionArtifactPayload(
        decision_keys=REGIME_DECISION_KEYS,
        decision_values={
            "algorithm_key": "kmeans_v1",
            "algorithm_parameters_hash": pinned_parameters_hash(entry),
            "algorithm_parameters": ImmutableMap(dict(entry.pinned_parameters)),
            "observation_granularity": "candidate_stage_row",
            "panel_interval_seconds": None,
            "observation_stage": "entry_decision",
            "fixed_cluster_count": 3,
            "minimum_cluster_occupancy_fraction": 0.05,
            "minimum_cluster_rows_per_fold": 25,
            "minimum_bootstrap_aligned_ami_mean": float(
                REGIME_PROPOSED_DEFAULTS["minimum_bootstrap_aligned_ami_mean"]["value"]
            ),
            "minimum_training_observations": sample_adequacy_minimum(
                ObservationGranularity.CANDIDATE_STAGE_ROW
            ),
        },
        resolved_regime_protocol_id="a" * 64,
        capability_assessment_id="b" * 64,
        authorized_transitions=AUTHORIZED_TRANSITIONS,
        provenance="synthetic_test_authorization_v1",
        author="example",
        approved_at="2026-08-28T00:00:00+00:00",
        effective_from="2026-08-28T00:00:00+00:00",
        effective_to=None,
        reviewed_evidence_refs=("b" * 64,),
        supersedes=None,
        rationale="example",
    )


register_identity_pair(
    name="OwnerDecisionArtifact",
    envelope_cls=OwnerDecisionArtifactEnvelope,
    payload_cls=OwnerDecisionArtifactPayload,
    id_field="owner_decision_artifact_id",
    example_factory=_example_payload,
)

_ = (canonical_contract_sha256, envelope_destination)  # re-exported helpers for the CLI
