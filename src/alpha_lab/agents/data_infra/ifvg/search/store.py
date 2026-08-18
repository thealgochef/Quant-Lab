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

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from ..manifest import canonical_sha256, file_sha256
from .identities import EnvelopeBase, canonical_contract_sha256

__all__ = [
    "SEARCH_STORE_ROOT",
    "SEARCH_TEST_STORE_ROOT",
    "SEARCH_STORE_NAMES",
    "SearchStoreError",
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
    "seed_snapshots",
    "coverage_matrices",
    "verification_runs",
    "neutrality_reports",
    "fsm_audit_companions",
    "lineage_reports",
)

_ENVELOPE_FILE = "envelope.json"
_MANIFEST_FILE = "manifest.json"


class SearchStoreError(ValueError):
    """A store invariant was violated (identity/manifest/overwrite)."""


def _validate_store_name(store_name: str) -> None:
    if store_name not in SEARCH_STORE_NAMES:
        raise SearchStoreError(f"unknown search store {store_name!r}")


_ENVELOPE_ID_PATTERN = re.compile(r"[0-9a-f]{64}")


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
) -> Path:
    """Write one envelope exactly once, then atomically publish.

    ``extra_files`` carries opaque sidecar bytes (e.g. a pickled seed) whose
    SHA-256 and sizes enter the manifest; they never enter the payload hash.
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
        for name, payload in sorted((extra_files or {}).items()):
            if "/" in name or "\\" in name or name in (_ENVELOPE_FILE, _MANIFEST_FILE):
                raise SearchStoreError(f"invalid sidecar file name {name!r}")
            sidecar = temporary / name
            sidecar.write_bytes(payload)
            artifacts.append(
                {
                    "path": name,
                    "sha256": file_sha256(sidecar),
                    "bytes": sidecar.stat().st_size,
                }
            )
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
    """Load one manifest-verified sidecar file from a store entry."""

    destination = envelope_destination(root, store_name, envelope_id)
    manifest = json.loads((destination / _MANIFEST_FILE).read_text(encoding="utf-8"))
    for entry in manifest.get("artifacts", ()):
        if entry["path"] == name:
            path = destination / name
            data = path.read_bytes()
            if file_sha256(path) != entry["sha256"]:
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
) -> tuple[E, bool]:
    """Idempotent publish: verified reuse when the exact identity exists.

    Returns ``(envelope, reused)``. Reuse verifies the stored entry against
    the manifest and asserts byte-equal envelope content — a same-ID entry
    with different content fails closed.
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
            import hashlib  # noqa: PLC0415

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
        return stored, True
    save_envelope_immutable(root, store_name, envelope, extra_files=extra_files)
    return envelope, False
