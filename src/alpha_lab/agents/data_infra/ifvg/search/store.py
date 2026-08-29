"""Immutable envelope stores on the verified manifest protocol (§8).

Every search-lane store follows the same discipline as the v2/v3 savers:
identity → refuse-if-exists → tmp-dir write → manifest with
``manifest_payload_sha256`` → ``os.replace`` publication → reload → assert.
Child v2 replay tables use the existing heavyweight saver; these stores hold
envelopes plus references. The research namespace is
``data/ifvg_datasets/search/v1/``; verification outputs live under the
isolated ``data/ifvg_datasets/search_test/v1/`` namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..manifest import canonical_sha256, file_sha256
from .identities import EnvelopeBase, canonical_contract_sha256

__all__ = [
    "SEARCH_STORE_ROOT",
    "SEARCH_TEST_STORE_ROOT",
    "SEARCH_STORE_NAMES",
    "SearchStoreError",
    "ProducedSidecar",
    "SidecarProducer",
    "write_produced_sidecar",
    "envelope_destination",
    "has_envelope",
    "save_envelope_immutable",
    "load_verified_envelope",
    "load_sidecar_bytes",
    "save_or_reuse_envelope",
]

SEARCH_STORE_ROOT = Path("data/ifvg_datasets/search/v1")
SEARCH_TEST_STORE_ROOT = Path("data/ifvg_datasets/search_test/v1")

#: The complete store vocabulary (§8). A store name outside this tuple is a
#: programming error — never a user-controlled value.
SEARCH_STORE_NAMES: tuple[str, ...] = (
    "core_replays",
    "replay_input_bundles",
    "memberships",
    "costed_evaluations",
    "charters",
    "prop_contracts",
    "contract_evidence",
    "risk_policies",
    "withdrawal_policies",
    "account_policy_sets",
    "portfolio_policies",
    "trade_path_artifacts",
    "trade_path_bundles",
    "account_replays",
    "account_simulations",
    "portfolio_simulations",
    "frontiers",
    "insights",
    "search_results",
    "pipeline_stage_results",
    "pipeline_specs",
    "seed_snapshots",
    "coverage_matrices",
    "verification_runs",
    "neutrality_reports",
    "fsm_audit_companions",
    "lineage_reports",
    # R5B — the offline MBP-1 lane (research-only)
    "mbp1_source_artifacts",
    "mbp1_feature_artifacts",
    "mbp1_coverage_reports",
    "controlled_feature_studies",
    # R5B.1 — MBP-1 coverage policy v2 evidence artifacts
    "mbp1_gap_manifests",
    "mbp1_completeness_reports",
    "mbp1_coverage_diagnostics",
    # R6 — the V1 KMeans regime lane
    "regime_protocols",
    "regime_fits",
    "regime_assessments",
    "regime_promotions",
    # R6.1 — verified observation seam, panel materialization, fold schedules
    # and fold sets, the descriptive OOS assignment, fold-local regime
    # features, verified owner evidence, and stratified reports
    "bundle_feature_views",
    "context_bar_panels",
    "fold_schedules",
    "fold_sets",
    "regime_oos_assignments",
    "regime_fold_features",
    "owner_decisions",
    "regime_stratified_reports",
    "regime_controlled_studies",
    "regime_cohort_model_studies",
)

_ENVELOPE_FILE = "envelope.json"
_MANIFEST_FILE = "manifest.json"


class SearchStoreError(ValueError):
    """A store invariant was violated (identity/manifest/overwrite)."""


def _validate_store_name(store_name: str) -> None:
    if store_name not in SEARCH_STORE_NAMES:
        raise SearchStoreError(f"unknown search store {store_name!r}")


_ENVELOPE_ID_PATTERN = re.compile(r"[0-9a-f]{64}")

#: sidecar names are simple filenames only — no separators, no leading dot,
#: no drive-relative (`C:evil`) or ADS (`name:stream`) colons (R5 safety F5)
_SIDECAR_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class ProducedSidecar:
    """One sidecar a producer wrote into the publication directory, with the
    sha256 / byte count the producer computed while streaming it. The store
    re-hashes the file and refuses any disagreement (the producer's
    bookkeeping is never trusted over the bytes on disk)."""

    name: str
    sha256: str
    bytes: int


#: A sidecar PRODUCER streams large sidecars straight into the store's
#: temporary publication directory (R6.1 D15 / safety review S12): it is
#: called with that directory, writes its files there, and returns one
#: :class:`ProducedSidecar` per file it wrote. It runs BEFORE publication —
#: any exception it raises discards the directory and nothing is published;
#: on a verified-reuse publication it runs into a scratch directory whose
#: bytes must reproduce the stored manifest hashes (then it is discarded).
SidecarProducer = Callable[[Path], Iterable[ProducedSidecar]]


def _assert_sidecar_name(name: str) -> None:
    if not _SIDECAR_NAME_PATTERN.fullmatch(name or "") or name in (
        _ENVELOPE_FILE,
        _MANIFEST_FILE,
    ):
        raise SearchStoreError(f"invalid sidecar file name {name!r}")


def write_produced_sidecar(directory: Path, name: str, data: bytes) -> ProducedSidecar:
    """Write one small in-memory sidecar for a producer (name held to the
    whitelist) and return its record."""

    _assert_sidecar_name(name)
    path = Path(directory) / name
    if path.exists():
        raise SearchStoreError(f"sidecar {name!r} was produced twice")
    path.write_bytes(data)
    return ProducedSidecar(name=name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def _run_sidecar_producer(
    producer: SidecarProducer, directory: Path, *, reserved: set[str]
) -> list[ProducedSidecar]:
    """Run ``producer`` into ``directory`` and verify its bookkeeping: every
    produced name is a whitelisted bare file name, unique, not reserved by
    the envelope / an ``extra_files`` sidecar, present as a regular file
    directly under ``directory`` with EXACTLY the claimed byte count and
    sha256 (the file is re-hashed by streaming); every file the producer
    left in the directory must be declared (an undeclared stray file
    refuses)."""

    directory = Path(directory)
    produced = list(producer(directory))
    seen: set[str] = set()
    for record in produced:
        if not isinstance(record, ProducedSidecar):
            raise SearchStoreError("a sidecar producer must return ProducedSidecar records")
        _assert_sidecar_name(record.name)
        if record.name in reserved or record.name in seen:
            raise SearchStoreError(
                f"sidecar producer declared {record.name!r} twice or over a reserved name"
            )
        seen.add(record.name)
        path = directory / record.name
        if not path.is_file() or path.parent.resolve() != directory.resolve():
            raise SearchStoreError(
                f"sidecar producer declared {record.name!r} but wrote no such file"
            )
        if path.stat().st_size != int(record.bytes) or file_sha256(path) != record.sha256:
            raise SearchStoreError(
                f"sidecar producer bookkeeping for {record.name!r} disagrees with the bytes "
                "on disk; refusing to publish"
            )
    stray = set(os.listdir(directory)) - reserved - seen
    if stray:
        raise SearchStoreError(
            "sidecar producer left undeclared files in the publication directory: "
            f"{sorted(stray)}"
        )
    return produced


def envelope_destination(root: Path, store_name: str, envelope_id: str) -> Path:
    _validate_store_name(store_name)
    # Full 64-hex only: no separators, no drive-relative segments, no ADS
    # colons — the id IS the directory name and never a user-shaped path.
    if not _ENVELOPE_ID_PATTERN.fullmatch(envelope_id or ""):
        raise SearchStoreError("envelope id is not a valid store key")
    return Path(root) / store_name / envelope_id


def has_envelope(root: Path, store_name: str, envelope_id: str) -> bool:
    """Exact-ID existence check; the store root is never listed."""

    return (envelope_destination(root, store_name, envelope_id) / _MANIFEST_FILE).exists()


def save_envelope_immutable(
    root: Path,
    store_name: str,
    envelope: EnvelopeBase,
    *,
    extra_files: dict[str, bytes] | None = None,
    sidecar_producer: SidecarProducer | None = None,
) -> Path:
    """Write one envelope exactly once, then atomically publish.

    ``extra_files`` carries opaque sidecar bytes (e.g. a pickled seed) whose
    SHA-256 and sizes enter the manifest; they never enter the payload hash.
    ``sidecar_producer`` streams large sidecars straight into the temporary
    publication directory (see :data:`SidecarProducer`); its files enter the
    manifest exactly like ``extra_files`` (sorted by name), after the store
    re-hashed them. Any failure before ``os.replace`` discards the directory.
    """

    envelope_id = getattr(envelope, type(envelope)._ID_FIELD)
    expected = canonical_contract_sha256(envelope.payload)
    if envelope_id != expected:
        raise SearchStoreError(
            f"{type(envelope).__name__} id does not hash its payload; refusing to save"
        )
    destination = envelope_destination(root, store_name, envelope_id)
    if destination.exists():
        raise FileExistsError(
            f"immutable search-store entry already exists: {store_name}/{envelope_id}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{envelope_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        envelope_path = temporary / _ENVELOPE_FILE
        envelope_path.write_text(
            json.dumps(envelope.model_dump(mode="json"), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        artifacts = [
            {
                "path": _ENVELOPE_FILE,
                "sha256": file_sha256(envelope_path),
                "bytes": envelope_path.stat().st_size,
            }
        ]
        sidecar_entries: dict[str, dict] = {}
        for name, payload in sorted((extra_files or {}).items()):
            _assert_sidecar_name(name)
            sidecar = temporary / name
            sidecar.write_bytes(payload)
            sidecar_entries[name] = {
                "path": name,
                "sha256": file_sha256(sidecar),
                "bytes": sidecar.stat().st_size,
            }
        if sidecar_producer is not None:
            reserved = {_ENVELOPE_FILE, *sidecar_entries}
            for record in _run_sidecar_producer(sidecar_producer, temporary, reserved=reserved):
                sidecar_entries[record.name] = {
                    "path": record.name,
                    "sha256": record.sha256,
                    "bytes": int(record.bytes),
                }
        artifacts.extend(sidecar_entries[name] for name in sorted(sidecar_entries))
        manifest_core = {
            "manifest_schema_version": 1,
            "store_name": store_name,
            "envelope_id": envelope_id,
            "envelope_type": type(envelope).__name__,
            "immutable": True,
            "artifacts": artifacts,
        }
        manifest = {
            **manifest_core,
            "manifest_payload_sha256": canonical_sha256(manifest_core),
        }
        (temporary / _MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise FileExistsError(
                f"search-store entry appeared concurrently: {store_name}/{envelope_id}"
            )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    # reload → assert (the manifest protocol's final step)
    reloaded = load_verified_envelope(root, store_name, envelope_id, type(envelope))
    if reloaded.model_dump(mode="json") != envelope.model_dump(mode="json"):
        raise SearchStoreError(
            f"reloaded envelope differs from the saved envelope: {store_name}/{envelope_id}"
        )
    return destination


def load_verified_envelope[E: EnvelopeBase](
    root: Path, store_name: str, envelope_id: str, envelope_cls: type[E]
) -> E:
    """Exact-ID load with manifest, file-hash, and identity verification."""

    destination = envelope_destination(root, store_name, envelope_id)
    manifest_path = destination / _MANIFEST_FILE
    if not manifest_path.exists():
        raise SearchStoreError(f"missing search-store entry {store_name}/{envelope_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    if manifest.get("manifest_payload_sha256") != canonical_sha256(core):
        raise SearchStoreError(f"manifest hash mismatch for {store_name}/{envelope_id}")
    if manifest.get("envelope_id") != envelope_id or manifest.get("store_name") != store_name:
        raise SearchStoreError(f"manifest identity mismatch for {store_name}/{envelope_id}")
    for entry in manifest.get("artifacts", ()):
        path = destination / entry["path"]
        if not path.exists() or file_sha256(path) != entry["sha256"]:
            raise SearchStoreError(
                f"artifact {entry['path']} failed verification in {store_name}/{envelope_id}"
            )
    envelope = envelope_cls.model_validate(
        json.loads((destination / _ENVELOPE_FILE).read_text(encoding="utf-8"))
    )
    actual_id = getattr(envelope, envelope_cls._ID_FIELD)
    if actual_id != envelope_id:
        raise SearchStoreError(
            f"stored envelope id {actual_id} does not match directory {envelope_id}"
        )
    return envelope


def load_sidecar_bytes(root: Path, store_name: str, envelope_id: str, name: str) -> bytes:
    """Load one manifest-verified sidecar file from a store entry.

    The sidecar NAME is held to the save-side whitelist (a bare file name —
    never a path), the manifest's own hash is verified before its entries
    are trusted, and the returned bytes are the bytes that were hashed
    (R6 safety review S6 — no second read, no traversal, no stale manifest).
    """

    if not _SIDECAR_NAME_PATTERN.fullmatch(name or "") or name in (
        _ENVELOPE_FILE,
        _MANIFEST_FILE,
    ):
        raise SearchStoreError(f"invalid sidecar file name {name!r}")
    destination = envelope_destination(root, store_name, envelope_id)
    manifest_path = destination / _MANIFEST_FILE
    if not manifest_path.exists():
        raise SearchStoreError(f"missing search-store entry {store_name}/{envelope_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    if manifest.get("manifest_payload_sha256") != canonical_sha256(core):
        raise SearchStoreError(f"manifest hash mismatch for {store_name}/{envelope_id}")
    if manifest.get("envelope_id") != envelope_id or manifest.get("store_name") != store_name:
        raise SearchStoreError(f"manifest identity mismatch for {store_name}/{envelope_id}")
    for entry in manifest.get("artifacts", ()):
        if entry["path"] == name:
            data = (destination / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise SearchStoreError(
                    f"sidecar {name} failed hash verification in {store_name}/{envelope_id}"
                )
            return data
    raise SearchStoreError(f"sidecar {name!r} is not in {store_name}/{envelope_id}")


def save_or_reuse_envelope[E: EnvelopeBase](
    root: Path,
    store_name: str,
    envelope: E,
    *,
    extra_files: dict[str, bytes] | None = None,
    sidecar_producer: SidecarProducer | None = None,
) -> tuple[E, bool]:
    """Idempotent publish: verified reuse when the exact identity exists.

    Returns ``(envelope, reused)``. Reuse verifies the stored entry against
    the manifest and asserts byte-equal envelope content — a same-ID entry
    with different content fails closed. A ``sidecar_producer`` is run into a
    scratch directory on reuse and every file it produces must reproduce the
    stored manifest hash (then the scratch directory is discarded).
    """

    envelope_id = getattr(envelope, type(envelope)._ID_FIELD)
    if has_envelope(root, store_name, envelope_id):
        stored = load_verified_envelope(root, store_name, envelope_id, type(envelope))
        if stored.model_dump(mode="json") != envelope.model_dump(mode="json"):
            raise SearchStoreError(
                f"store entry {store_name}/{envelope_id} exists with DIFFERENT content"
            )
        if extra_files:
            # sidecar bytes are outside the payload identity, so reuse must
            # verify them against the stored manifest — a same-identity
            # publication with DIFFERENT sidecar bytes fails closed instead of
            # silently "reusing" the old bytes.
            manifest = json.loads(
                (
                    envelope_destination(root, store_name, envelope_id)
                    / _MANIFEST_FILE
                ).read_text(encoding="utf-8")
            )
            stored_hashes = {
                entry["path"]: entry["sha256"]
                for entry in manifest.get("artifacts", ())
            }
            for name, payload in sorted(extra_files.items()):
                digest = hashlib.sha256(payload).hexdigest()
                if stored_hashes.get(name) != digest:
                    raise SearchStoreError(
                        f"store entry {store_name}/{envelope_id} exists with "
                        f"DIFFERENT sidecar content for {name!r}"
                    )
        if sidecar_producer is not None:
            _verify_reuse_with_producer(root, store_name, envelope_id, sidecar_producer)
        return stored, True
    save_envelope_immutable(
        root, store_name, envelope, extra_files=extra_files, sidecar_producer=sidecar_producer
    )
    return envelope, False


def _verify_reuse_with_producer(
    root: Path, store_name: str, envelope_id: str, producer: SidecarProducer
) -> None:
    """Reuse with a producer: the producer runs into a scratch directory next
    to the entry and every produced file must reproduce the stored manifest
    hash byte for byte; the scratch directory never survives."""

    destination = envelope_destination(root, store_name, envelope_id)
    manifest = json.loads((destination / _MANIFEST_FILE).read_text(encoding="utf-8"))
    stored_hashes = {entry["path"]: entry["sha256"] for entry in manifest.get("artifacts", ())}
    scratch = destination.parent / f".{envelope_id}.verify-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True)
    try:
        produced = _run_sidecar_producer(producer, scratch, reserved={_ENVELOPE_FILE})
        for record in produced:
            if stored_hashes.get(record.name) != record.sha256:
                raise SearchStoreError(
                    f"store entry {store_name}/{envelope_id} exists with "
                    f"DIFFERENT sidecar content for {record.name!r}"
                )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
