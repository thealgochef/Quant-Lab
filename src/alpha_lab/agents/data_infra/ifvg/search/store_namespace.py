"""Path-independent semantic store namespace (HARDENING-BACKEND §4.1; F-11).

R6.1 inferred research-versus-test authority from filesystem pathname
components (``research_namespace_root``: a resolved path carrying a
``search`` segment that is not ``search_test``). Authority that follows a
directory name relocates with a ``mv`` and can be granted by naming a
folder. This module replaces it with an IMMUTABLE semantic namespace
contract every authority-bearing artifact references by content id:

* :class:`StoreNamespacePayload` / :class:`StoreNamespaceEnvelope` —
  ``namespace_class ∈ {research, test}``, a stable ``store_instance_id``
  (a 32-hex id the operator chooses and records BEFORE the first
  initialization — never an absolute-path hash, so relocation preserves the
  semantic identity) and the ``authority_genesis_id`` that anchors the
  supersession chain (``store_namespace_id = hash(payload)``).
* The envelope lives at ``<root>/STORE_NAMESPACE.json``; its bytes are
  immutable (a second initialization must reproduce the identical payload)
  and re-verified on EVERY authorization load (the envelope id must hash
  the payload; a tampered file is a typed refusal, never a class).
* An unmarked store has NO semantic authority: it cannot carry owner
  decisions, supersession records, owner-evidence promotions, real
  authorization bundles / refs, or real launches. The one-time explicit
  migration (:func:`initialize_store_namespace`; CLI
  ``scripts/ifvg_store_namespace.py``) initializes it after showing the
  operator the intended class — the class is an argument, never inferred
  from a path.
* Filesystem naming checks remain DEFENSE-IN-DEPTH configuration checks
  (:func:`path_looks_like_research_store`): they may refuse a suspicious
  deployment (a ``test`` namespace living under a research-looking path, or
  synthetic authority under a research-looking path) but they never define
  or change semantic authority.
* The namespace is initialized with an explicit GENESIS supersession head
  (``owner_decisions/SUPERSESSIONS.head``); a missing head is corruption,
  never "no supersessions" (§4.2).

**Atomic, recoverable initialization (HARDENING-BACKEND-FIX §4.2).** The
envelope and the genesis head are one coherent pair. Initialization runs
under a dedicated one-time mutex (``owner_decisions/STORE_NAMESPACE.init.mutex``;
independent of the owner-decision lock) and classifies the store as exactly
one of: both absent → publish the precomputed deterministic pair through
temporary files and verified-load both before returning; both present and
coherent → the identical request is idempotent reuse, a divergent class or
instance is ``store_namespace_divergent``; one present → recover ONLY when
the existing object is exactly the deterministic object the same requested
initialization (class + explicit instance id) would produce and no
supersession record exists, else ``incomplete_store_namespace_initialization``
(conflicting bytes are never overwritten); both present but inconsistent →
fail closed (no object is ever selected as authoritative by pathname or
modification time).

**Recoverable first initialization (HARDENING-BACKEND-FIX.1 §2).** The real
initializer NEVER generates a store instance id: the FIRST initialization of
every real persistent store requires an explicit ``store_instance_id`` (typed
``store_instance_id_required`` otherwise, before any file is published) — a
random id minted inside the initializer would be lost with a crash between
the two publications and could never recover the half-initialized store.
Disposable test-only temporary stores use the clearly separate
:func:`initialize_test_namespace`, which generates its id BEFORE calling the
real initializer. Idempotent reuse of a marked store (class-only replay or
the identical explicit request) and explicit-id crash recovery are unchanged.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from ..manifest import canonical_sha256
from .file_mutex import ExclusiveFileMutex, FileMutexError, FileMutexTimeoutError
from .identities import SHA256_PATTERN, EnvelopeBase, FrozenContract, register_identity_pair

__all__ = [
    "STORE_NAMESPACE_FILE",
    "OWNER_DECISION_STORE",
    "OWNER_DECISION_SUPERSESSION_STORE",
    "SUPERSESSIONS_HEAD_FILE",
    "STORE_NAMESPACE_INIT_MUTEX_FILE",
    "NAMESPACE_SCHEMA_VERSION",
    "NAMESPACE_CLASSES",
    "SUPERSESSION_HEAD_SCHEMA_VERSION",
    "STORE_NAMESPACE_FAILURE_REASONS",
    "NamespaceClass",
    "StoreNamespaceError",
    "StoreNamespacePayload",
    "StoreNamespaceEnvelope",
    "SupersessionHeadWitness",
    "genesis_id_for",
    "genesis_head_witness",
    "chain_head_digest",
    "supersession_head_path",
    "read_supersession_head",
    "write_supersession_head_atomic",
    "initialize_store_namespace",
    "initialize_test_namespace",
    "load_store_namespace",
    "require_store_namespace",
    "namespace_class_of",
    "path_looks_like_research_store",
    "assert_namespace_deployment_coherent",
]

STORE_NAMESPACE_FILE = "STORE_NAMESPACE.json"
OWNER_DECISION_STORE = "owner_decisions"
#: The immutable supersession-record store (the chain module publishes into
#: it; initialization only asks whether any record exists).
OWNER_DECISION_SUPERSESSION_STORE = "owner_decision_supersessions"
SUPERSESSIONS_HEAD_FILE = "SUPERSESSIONS.head"
#: The one-time initialization mutex (beside the head; it carries no
#: authority and is never unlinked — see ``file_mutex``).
STORE_NAMESPACE_INIT_MUTEX_FILE = "STORE_NAMESPACE.init.mutex"
NAMESPACE_SCHEMA_VERSION = 1
SUPERSESSION_HEAD_SCHEMA_VERSION = 2
NAMESPACE_CLASSES: tuple[str, ...] = ("research", "test")
NamespaceClass = Literal["research", "test"]
_GENESIS_KIND = "owner_decision_supersessions_v2"
_INSTANCE_PATTERN = r"^[0-9a-f]{32}$"
_INIT_MUTEX_WAIT_SECONDS = 30.0

#: Every typed refusal of this module (never inferred from message text).
STORE_NAMESPACE_FAILURE_REASONS: tuple[str, ...] = (
    "store_namespace_missing",
    "store_namespace_malformed",
    "store_namespace_identity_mismatch",
    "store_namespace_class_mismatch",
    "store_namespace_divergent",
    "store_namespace_deployment_incoherent",
    "supersession_head_missing",
    "supersession_head_malformed",
    "supersession_head_namespace_mismatch",
    "supersession_head_witness_mismatch",
    "supersession_head_shorter_than_witness",
    "supersession_chain_broken",
    "supersession_record_unverifiable",
    "supersession_divergent_replay",
    "atomic_write_failed",
    # HARDENING-BACKEND-FIX §4.2: a half-initialized store that the request
    # cannot provably complete, and a contended initialization mutex
    "incomplete_store_namespace_initialization",
    "store_namespace_initialization_busy",
    # review RA-02: a persistent non-contention failure of the init mutex file
    "store_namespace_initialization_failed",
    # HARDENING-BACKEND-FIX.1 §2: the first initialization of a real store
    # without an explicit instance id (the initializer never generates one)
    "store_instance_id_required",
    # HARDENING-BACKEND-FIX §10: the complete owner-authority chain proof
    "supersession_decision_unverifiable",
    "supersession_transition_unlawful",
    "supersession_chain_divergent",
)
#: Adversarial RA-10: the head / namespace ``os.replace`` retries a transient
#: sharing violation (a concurrent reader) with a short backoff; a persistent
#: failure is a typed refusal.
_REPLACE_RETRY_ATTEMPTS = 40
_REPLACE_RETRY_DELAY_SECONDS = 0.025


class StoreNamespaceError(PermissionError):
    """A namespace / head-witness invariant refused (typed ``reason``)."""

    def __init__(self, reason: str, message: str) -> None:
        if reason not in STORE_NAMESPACE_FAILURE_REASONS:
            raise ValueError(f"unregistered store namespace failure reason {reason!r}")
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def genesis_id_for(store_instance_id: str, namespace_class: str) -> str:
    """The supersession-chain genesis anchor of one namespace instance."""

    return canonical_sha256(
        {
            "genesis": _GENESIS_KIND,
            "namespace_class": str(namespace_class),
            "store_instance_id": str(store_instance_id),
        }
    )


class StoreNamespacePayload(FrozenContract):
    namespace_schema_version: Literal[1] = NAMESPACE_SCHEMA_VERSION
    namespace_class: NamespaceClass
    store_instance_id: str = Field(pattern=_INSTANCE_PATTERN)
    authority_genesis_id: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _genesis_binds_instance(self):
        expected = genesis_id_for(self.store_instance_id, self.namespace_class)
        if self.authority_genesis_id != expected:
            raise ValueError(
                "authority_genesis_id does not derive from the namespace class and instance"
            )
        return self


class StoreNamespaceEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "store_namespace_id"

    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    payload: StoreNamespacePayload


class SupersessionHeadWitness(FrozenContract):
    """The current supersession-chain head an authorization was signed
    against: ``{store_namespace_id, line_count, head_sha256}``. Verification
    refuses a missing, shorter, or different current head (§4.2) — local
    rollback detection, not cryptographic owner authenticity."""

    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    line_count: int = Field(ge=0)
    head_sha256: str = Field(pattern=SHA256_PATTERN)


def genesis_head_witness(namespace: StoreNamespaceEnvelope) -> SupersessionHeadWitness:
    return SupersessionHeadWitness(
        store_namespace_id=namespace.store_namespace_id,
        line_count=0,
        head_sha256=namespace.payload.authority_genesis_id,
    )


def chain_head_digest(prior_head_sha256: str, record_id: str) -> str:
    """The head digest after appending ``record_id`` to a chain whose head
    digest was ``prior_head_sha256`` (commits to the whole chain)."""

    return canonical_sha256(
        {"prior_head_sha256": str(prior_head_sha256), "record_id": str(record_id)}
    )


# ── the head pointer file (owner_decisions/SUPERSESSIONS.head) ──────────────


def supersession_head_path(root: Path) -> Path:
    return Path(root) / OWNER_DECISION_STORE / SUPERSESSIONS_HEAD_FILE


def _namespace_path(root: Path) -> Path:
    return Path(root) / STORE_NAMESPACE_FILE


def _init_mutex_path(root: Path) -> Path:
    return Path(root) / OWNER_DECISION_STORE / STORE_NAMESPACE_INIT_MUTEX_FILE


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    last: OSError | None = None
    for _ in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except OSError as error:  # a transient sharing violation (RA-10)
            last = error
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS)
    with contextlib.suppress(OSError):
        os.unlink(temporary)
    raise StoreNamespaceError(
        "atomic_write_failed",
        f"{path.name} could not be published atomically after retries ({last})",
    )


def _head_document(
    *, store_namespace_id: str, record_id: str | None, line_count: int, head_sha256: str
) -> dict:
    return {
        "head_schema_version": SUPERSESSION_HEAD_SCHEMA_VERSION,
        "store_namespace_id": str(store_namespace_id),
        "record_id": None if record_id is None else str(record_id),
        "line_count": int(line_count),
        "head_sha256": str(head_sha256),
    }


def _head_text(head: dict) -> str:
    return json.dumps(head, sort_keys=True, indent=2) + "\n"


def _namespace_text(envelope: StoreNamespaceEnvelope) -> str:
    return json.dumps(envelope.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


def _genesis_head_document(envelope: StoreNamespaceEnvelope) -> dict:
    """The deterministic genesis head of ``envelope`` (the exact bytes an
    initialization publishes; recovery compares against it)."""

    return _head_document(
        store_namespace_id=envelope.store_namespace_id,
        record_id=None,
        line_count=0,
        head_sha256=envelope.payload.authority_genesis_id,
    )


def write_supersession_head_atomic(
    root: Path,
    *,
    store_namespace_id: str,
    record_id: str | None,
    line_count: int,
    head_sha256: str,
) -> None:
    """Atomically replace the head pointer (``os.replace`` of a temp file)."""

    if record_id is not None and (line_count < 1):
        raise ValueError("a head that names a record must have line_count >= 1")
    if record_id is None and line_count != 0:
        raise ValueError("the genesis head has line_count 0")
    head = _head_document(
        store_namespace_id=store_namespace_id,
        record_id=record_id,
        line_count=line_count,
        head_sha256=head_sha256,
    )
    _atomic_write_text(supersession_head_path(root), _head_text(head))


def read_supersession_head(root: Path, *, store_namespace_id: str) -> dict:
    """The head pointer, structurally verified against the namespace id.
    Missing is corruption (``supersession_head_missing``), never "no
    supersessions"."""

    path = supersession_head_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise StoreNamespaceError(
            "supersession_head_missing",
            "the owner-decision supersession head is missing (a namespace is initialized "
            "with an explicit genesis head; a missing head is corruption)",
        ) from error
    except OSError as error:
        raise StoreNamespaceError("supersession_head_malformed", str(error)) from error
    try:
        head = json.loads(raw)
    except ValueError as error:
        raise StoreNamespaceError(
            "supersession_head_malformed", "the supersession head is not JSON"
        ) from error
    if not isinstance(head, dict):
        raise StoreNamespaceError(
            "supersession_head_malformed", "the supersession head is not an object"
        )
    if head.get("head_schema_version") != SUPERSESSION_HEAD_SCHEMA_VERSION:
        raise StoreNamespaceError(
            "supersession_head_malformed",
            f"the supersession head schema is not {SUPERSESSION_HEAD_SCHEMA_VERSION} "
            "(a pre-hardening JSONL log head is not a v2 head; migrate the store)",
        )
    record_id = head.get("record_id")
    line_count = head.get("line_count")
    digest = head.get("head_sha256")
    if (
        not isinstance(line_count, int)
        or isinstance(line_count, bool)
        or line_count < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or not (record_id is None or (isinstance(record_id, str) and len(record_id) == 64))
        or (record_id is None) != (line_count == 0)
    ):
        raise StoreNamespaceError(
            "supersession_head_malformed", "the supersession head is malformed"
        )
    if head.get("store_namespace_id") != store_namespace_id:
        raise StoreNamespaceError(
            "supersession_head_namespace_mismatch",
            "the supersession head belongs to another store namespace",
        )
    return head


# ── namespace initialization / verified load ────────────────────────────────


def _envelope_for(namespace_class: str, store_instance_id: str) -> StoreNamespaceEnvelope:
    payload = StoreNamespacePayload(
        namespace_class=namespace_class,  # type: ignore[arg-type]
        store_instance_id=store_instance_id,
        authority_genesis_id=genesis_id_for(store_instance_id, namespace_class),
    )
    return StoreNamespaceEnvelope.from_payload(payload)


def _read_namespace_state(root: Path) -> tuple[bool, StoreNamespaceEnvelope | None]:
    """``(present, verified envelope)``; a malformed or tampered file is a
    typed refusal that initialization never overwrites."""

    try:
        return True, load_store_namespace(root)
    except StoreNamespaceError as error:
        if error.reason == "store_namespace_missing":
            return False, None
        raise


def _read_head_state(root: Path) -> tuple[Literal["absent", "malformed", "present"], dict | None]:
    path = supersession_head_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "absent", None
    except OSError as error:
        raise StoreNamespaceError("supersession_head_malformed", str(error)) from error
    try:
        head = json.loads(raw)
    except ValueError:
        return "malformed", None
    if not isinstance(head, dict):
        return "malformed", None
    return "present", head


def _supersession_records_exist(root: Path) -> bool:
    directory = Path(root) / OWNER_DECISION_SUPERSESSION_STORE
    if not directory.is_dir():
        return False
    return any(entry.is_dir() for entry in directory.iterdir())


def _assert_same_request(
    existing: StoreNamespaceEnvelope, namespace_class: str, store_instance_id: str | None
) -> None:
    same_class = existing.payload.namespace_class == namespace_class
    same_instance = store_instance_id is None or (
        existing.payload.store_instance_id == store_instance_id
    )
    if not (same_class and same_instance):
        raise StoreNamespaceError(
            "store_namespace_divergent",
            f"the store is already marked {existing.payload.namespace_class!r} "
            f"(instance {existing.payload.store_instance_id[:8]}…); the namespace "
            "envelope is immutable and cannot be re-initialized differently",
        )


def _verified_genesis_pair(
    root: Path, expected: StoreNamespaceEnvelope
) -> StoreNamespaceEnvelope:
    """Verified-load BOTH objects and prove the pair before success:
    ``namespace.store_namespace_id == head.store_namespace_id``,
    ``namespace.authority_genesis_id == head.head_sha256`` (the genesis
    anchor), the requested class is the persisted class."""

    reloaded = load_store_namespace(root)
    if reloaded.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise StoreNamespaceError(
            "store_namespace_malformed", "the reloaded namespace differs from the written one"
        )
    head = read_supersession_head(root, store_namespace_id=reloaded.store_namespace_id)
    if head != _genesis_head_document(reloaded):
        raise StoreNamespaceError(
            "supersession_head_malformed",
            "the reloaded supersession head is not the genesis head of the namespace",
        )
    return reloaded


def _initialize_under_mutex(
    root: Path, *, namespace_class: str, store_instance_id: str | None
) -> StoreNamespaceEnvelope:
    _namespace_present, existing = _read_namespace_state(root)
    head_state, head = _read_head_state(root)

    if existing is not None and head_state != "absent":
        # ── both present: coherent, or fail closed ────────────────────────
        verified = read_supersession_head(root, store_namespace_id=existing.store_namespace_id)
        if verified["line_count"] == 0 and verified != _genesis_head_document(existing):
            raise StoreNamespaceError(
                "supersession_head_malformed",
                "the genesis head does not anchor this namespace's authority genesis "
                "(an inconsistent namespace / head pair; refusing to select either)",
            )
        _assert_same_request(existing, namespace_class, store_instance_id)
        return existing

    if existing is None and head_state == "absent":
        # ── both absent: publish the precomputed deterministic pair ───────
        if store_instance_id is None:
            # HARDENING-BACKEND-FIX.1 §2: never mint a random id here — a crash
            # between the two publications would lose it and the half-initialized
            # store could never be recovered (recovery needs the explicit id)
            raise StoreNamespaceError(
                "store_instance_id_required",
                "the first initialization of an unmarked store requires an explicit "
                "store_instance_id (32 lowercase hex characters) chosen and recorded by the "
                "operator BEFORE initialization — the initializer never generates one, so an "
                "interrupted initialization stays recoverable by the identical request; "
                "nothing was published (disposable test stores use "
                "initialize_test_namespace)",
            )
        envelope = _envelope_for(namespace_class, store_instance_id)
        genesis_text = _head_text(_genesis_head_document(envelope))
        namespace_text = _namespace_text(envelope)
        _atomic_write_text(supersession_head_path(root), genesis_text)
        _atomic_write_text(_namespace_path(root), namespace_text)
        return _verified_genesis_pair(root, envelope)

    if existing is not None:
        # ── namespace present / genesis absent ────────────────────────────
        if store_instance_id is None or (
            existing.payload.namespace_class,
            existing.payload.store_instance_id,
        ) != (namespace_class, store_instance_id):
            raise StoreNamespaceError(
                "incomplete_store_namespace_initialization",
                "the store carries a namespace envelope but no supersession head; recovery "
                "requires the identical initialization request (the same class AND the "
                "explicit store instance id) — otherwise operator intervention",
            )
        if _supersession_records_exist(root):
            raise StoreNamespaceError(
                "incomplete_store_namespace_initialization",
                "the store carries a namespace envelope, no supersession head, and "
                "supersession records: the missing head cannot be recovered as the genesis "
                "head (a rolled-back or truncated chain) — operator intervention",
            )
        expected = _envelope_for(namespace_class, store_instance_id)
        if expected.model_dump(mode="json") != existing.model_dump(mode="json"):
            raise StoreNamespaceError(
                "incomplete_store_namespace_initialization",
                "the existing namespace envelope is not the deterministic envelope of the "
                "requested initialization; conflicting bytes are never overwritten",
            )
        _atomic_write_text(
            supersession_head_path(root), _head_text(_genesis_head_document(expected))
        )
        return _verified_genesis_pair(root, expected)

    # ── genesis present / namespace absent ────────────────────────────────
    if head_state == "malformed" or head is None:
        raise StoreNamespaceError(
            "supersession_head_malformed",
            "a malformed supersession head exists for an unmarked store (corruption; "
            "refusing to initialize over it)",
        )
    if head.get("record_id") is not None or head.get("line_count") != 0:
        raise StoreNamespaceError(
            "supersession_head_malformed",
            "a supersession head naming records exists for an unmarked store "
            "(corruption; refusing to initialize over it)",
        )
    if store_instance_id is None:
        raise StoreNamespaceError(
            "incomplete_store_namespace_initialization",
            "the store carries a genesis supersession head but no namespace envelope; "
            "recovery requires the identical initialization request (the same class AND "
            "the explicit store instance id) — otherwise operator intervention",
        )
    expected = _envelope_for(namespace_class, store_instance_id)
    if head != _genesis_head_document(expected):
        raise StoreNamespaceError(
            "incomplete_store_namespace_initialization",
            "the orphan genesis head is not the deterministic genesis head of the requested "
            "initialization; conflicting bytes are never overwritten",
        )
    _atomic_write_text(_namespace_path(root), _namespace_text(expected))
    return _verified_genesis_pair(root, expected)


def initialize_store_namespace(
    root: Path,
    *,
    namespace_class: str,
    store_instance_id: str | None = None,
) -> StoreNamespaceEnvelope:
    """The ONE-TIME explicit migration: mark ``root`` with its semantic class.

    The class is an argument the operator states (the CLI shows it before
    acting); it is never inferred from the path. The FIRST initialization of
    an unmarked store requires an explicit ``store_instance_id`` (32 lowercase
    hex characters; HARDENING-BACKEND-FIX.1 §2: the initializer never
    generates one — typed ``store_instance_id_required`` before any file is
    published). Idempotent on an identical payload (the identical explicit
    request, or a class-only replay of an already-marked store); a different
    class or instance for an already-marked store is a typed
    ``store_namespace_divergent`` refusal (namespace bytes are immutable). The
    envelope and the genesis head are published as one coherent pair under a
    one-time initialization mutex (HARDENING-BACKEND-FIX §4.2): a crash
    between the two writes is recovered EXACTLY by the same request (class +
    explicit instance id) and refused otherwise
    (``incomplete_store_namespace_initialization``); an inconsistent pair
    fails closed.
    """

    root = Path(root)
    if namespace_class not in NAMESPACE_CLASSES:
        raise ValueError(
            f"namespace_class must be one of {NAMESPACE_CLASSES}, got {namespace_class!r}"
        )
    if store_instance_id is not None and (
        type(store_instance_id) is not str
        or re.fullmatch(_INSTANCE_PATTERN, store_instance_id) is None
    ):
        raise ValueError(
            "store_instance_id must be a native string of exactly 32 lowercase hexadecimal "
            f"characters, got {store_instance_id!r}"
        )
    mutex = ExclusiveFileMutex(_init_mutex_path(root), wait_seconds=_INIT_MUTEX_WAIT_SECONDS)
    try:
        with mutex:
            return _initialize_under_mutex(
                root, namespace_class=namespace_class, store_instance_id=store_instance_id
            )
    except FileMutexTimeoutError as error:
        raise StoreNamespaceError(
            "store_namespace_initialization_busy",
            "another initializer holds the store-namespace initialization mutex; refusing "
            "to proceed without it",
        ) from error
    except FileMutexError as error:
        raise StoreNamespaceError(
            "store_namespace_initialization_failed",
            f"the store-namespace initialization mutex failed persistently ({error}); "
            "this is not a busy initializer — operator intervention required",
        ) from error


def initialize_test_namespace(root: Path) -> StoreNamespaceEnvelope:
    """DISPOSABLE TEST-ONLY helper (HARDENING-BACKEND-FIX.1 §2): mark a
    temporary root as an explicit ``test`` namespace.

    This is the clearly separate seam that GENERATES an instance id — a
    fresh ``uuid4().hex`` minted BEFORE the real initializer runs (the real
    initializer never generates one). A disposable store needs no recovery
    key: a crash between the two publications leaves a half-initialized
    temporary store that is simply discarded. An already-marked root is
    replayed class-only (idempotent reuse of a ``test`` store; a marked
    ``research`` root or a half-initialized store is refused by the real
    initializer exactly as any class-only request). Never use this helper
    for a real persistent store.
    """

    root = Path(root)
    if namespace_class_of(root) is not None:
        return initialize_store_namespace(root, namespace_class="test")
    return initialize_store_namespace(
        root, namespace_class="test", store_instance_id=uuid.uuid4().hex
    )


def load_store_namespace(root: Path) -> StoreNamespaceEnvelope:
    """Verified load: the envelope id must hash the payload (a tampered or
    rewritten file is ``store_namespace_identity_mismatch``); an unmarked
    store is ``store_namespace_missing``."""

    path = _namespace_path(Path(root))
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        # the message names the root's LAST segment only: an absolute path would
        # be withheld by the pipeline's sanitized stage explanations (Fix-E note)
        raise StoreNamespaceError(
            "store_namespace_missing",
            f"store root {Path(root).name!r} carries no {STORE_NAMESPACE_FILE}: an unmarked "
            "store has no semantic authority (initialize it explicitly with its intended class)",
        ) from error
    except OSError as error:
        raise StoreNamespaceError("store_namespace_malformed", str(error)) from error
    try:
        document = json.loads(raw)
    except ValueError as error:
        raise StoreNamespaceError(
            "store_namespace_malformed", f"{STORE_NAMESPACE_FILE} is not JSON"
        ) from error
    if not isinstance(document, dict):
        raise StoreNamespaceError(
            "store_namespace_malformed", f"{STORE_NAMESPACE_FILE} is not an object"
        )
    payload_document = document.get("payload")
    declared_id = document.get("store_namespace_id")
    try:
        payload = StoreNamespacePayload.model_validate(payload_document)
    except (ValueError, TypeError) as error:
        raise StoreNamespaceError(
            "store_namespace_malformed", f"{STORE_NAMESPACE_FILE} payload is malformed"
        ) from error
    try:
        envelope = StoreNamespaceEnvelope(store_namespace_id=declared_id, payload=payload)
    except ValueError as error:
        raise StoreNamespaceError(
            "store_namespace_identity_mismatch",
            f"{STORE_NAMESPACE_FILE} id does not hash its payload (tampered or rewritten)",
        ) from error
    return envelope


def require_store_namespace(
    root: Path, *, expected_class: str | None = None
) -> StoreNamespaceEnvelope:
    """A verified namespace of the expected class (typed refusals)."""

    envelope = load_store_namespace(root)
    if expected_class is not None and envelope.payload.namespace_class != expected_class:
        raise StoreNamespaceError(
            "store_namespace_class_mismatch",
            f"the store is a {envelope.payload.namespace_class!r} namespace; "
            f"{expected_class!r} is required",
        )
    return envelope


def namespace_class_of(root: Path) -> str | None:
    """``research`` / ``test`` for a marked store, ``None`` for an unmarked
    one; a corrupt namespace file is a typed refusal (never ``None``)."""

    try:
        return load_store_namespace(root).payload.namespace_class
    except StoreNamespaceError as error:
        if error.reason == "store_namespace_missing":
            return None
        raise


def path_looks_like_research_store(root: Path) -> bool:
    """DEFENSE-IN-DEPTH only: the pre-hardening pathname heuristic (a resolved
    path with a ``search`` segment that is not ``search_test``). It may
    refuse a suspicious deployment; it never grants or defines authority."""

    parts = Path(root).resolve().parts
    return "search" in parts and "search_test" not in parts


def assert_namespace_deployment_coherent(root: Path, namespace: StoreNamespaceEnvelope) -> None:
    """Refuse a suspicious deployment: a ``test`` namespace living under a
    research-looking path (the configuration check of §4.1)."""

    if namespace.payload.namespace_class == "test" and path_looks_like_research_store(root):
        raise StoreNamespaceError(
            "store_namespace_deployment_incoherent",
            "a test namespace is deployed under a research-looking store path; refusing "
            "the deployment (defense in depth — the path never defines authority)",
        )


def _example_namespace_payload() -> StoreNamespacePayload:
    instance = "0" * 32
    return StoreNamespacePayload(
        namespace_class="test",
        store_instance_id=instance,
        authority_genesis_id=genesis_id_for(instance, "test"),
    )


register_identity_pair(
    name="StoreNamespace",
    envelope_cls=StoreNamespaceEnvelope,
    payload_cls=StoreNamespacePayload,
    id_field="store_namespace_id",
    example_factory=_example_namespace_payload,
)
