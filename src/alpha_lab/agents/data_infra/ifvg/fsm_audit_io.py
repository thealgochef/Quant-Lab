"""Verified read access to immutable ``ifvg_fsm_audit_v1`` artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .audit_contracts import AuditTable
from .manifest import canonical_sha256, file_sha256

__all__ = [
    "VerifiedFsmAuditArtifact",
    "load_verified_fsm_audit_artifact",
]

_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class VerifiedFsmAuditArtifact:
    artifact_id: str
    directory: Path
    manifest: dict[str, Any]
    tables: dict[AuditTable, pd.DataFrame]

    @property
    def manifest_payload_sha256(self) -> str:
        return str(self.manifest["manifest_payload_sha256"])

    @property
    def accepted_v2_dataset_id(self) -> str:
        return str(self.manifest["identity"]["accepted_v2_dataset_id"])


def load_verified_fsm_audit_artifact(
    base_dir: Path,
    artifact_id: str,
    *,
    expected_accepted_v2_dataset_id: str | None = None,
) -> VerifiedFsmAuditArtifact:
    """Exact-ID verified load: manifest payload hash, per-file SHA-256/bytes/
    row counts, parity + reconciliation flags, optional accepted-v2 pin."""
    if not _FULL_SHA256.fullmatch(artifact_id):
        raise ValueError("fsm audit artifact ID must be a full SHA-256")
    root = Path(base_dir).resolve()
    directory = (root / artifact_id / "exploration").resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise ValueError("fsm audit artifact escaped its store") from error
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("fsm audit manifest is unreadable") from error
    if manifest.get("dataset_id") != artifact_id:
        raise ValueError("fsm audit manifest identity mismatch")
    if manifest.get("artifact_kind") != "ifvg_fsm_audit_v1":
        raise ValueError("fsm audit manifest kind mismatch")
    core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    if canonical_sha256(core) != manifest.get("manifest_payload_sha256"):
        raise ValueError("fsm audit manifest payload hash mismatch")
    if manifest.get("parity_passed") is not True:
        raise ValueError("fsm audit artifact lacks a passed parity gate")
    if manifest.get("reconciliation_passed") is not True:
        raise ValueError("fsm audit artifact lacks a passed reconciliation")
    if (
        expected_accepted_v2_dataset_id is not None
        and manifest.get("identity", {}).get("accepted_v2_dataset_id")
        != expected_accepted_v2_dataset_id
    ):
        raise ValueError("fsm audit artifact pins a different accepted v2 dataset")

    tables: dict[AuditTable, pd.DataFrame] = {}
    entries = {str(e.get("path")): e for e in manifest.get("artifacts", ())}
    for table in AuditTable:
        relative = f"exploration/{table.value}.parquet"
        entry = entries.get(relative)
        if entry is None:
            raise ValueError(f"fsm audit manifest lacks {relative}")
        path = (directory / f"{table.value}.parquet").resolve()
        if not path.is_file():
            raise ValueError(f"fsm audit table is missing: {relative}")
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"fsm audit table byte mismatch: {relative}")
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"fsm audit table SHA-256 mismatch: {relative}")
        frame = pd.read_parquet(path)
        if "rows" in entry and len(frame) != entry["rows"]:
            raise ValueError(f"fsm audit row-count mismatch: {relative}")
        tables[table] = frame
    return VerifiedFsmAuditArtifact(
        artifact_id=artifact_id,
        directory=directory,
        manifest=manifest,
        tables=tables,
    )
