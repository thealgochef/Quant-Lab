"""Bounded recovery of metric inputs from an already published strategy child.

This verifies the selected execution evidence, not all replay tables again.
It opens no source bars and never starts a replay or changes stored identities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..artifact_io import (
    ArtifactVerificationError,
    _artifact_path,
    _read_manifest,
    resolve_artifact_directory,
)
from ..contracts import RecordTable, validate_primary_keys, validate_table_identity
from ..dataset import table_content_hash
from ..manifest import canonical_sha256, file_sha256
from .identities import CoreReplayArtifactReference, CoreStrategyReplayIdentity
from .store import load_sidecar_bytes, load_verified_envelope


def make_saved_strategy_result_loader(store_root: Path):
    """Adapt saved evidence to the orchestrator's spec-bound recovery hook."""
    root = Path(store_root)

    def result_loader(*, spec, core_replay_id):
        result = load_saved_strategy_result(root, core_replay_id)
        if canonical_sha256(result.effective_section) != spec.resolved_section_config_hash:
            raise PermissionError("saved result differs from the requested child section")
        if result.canonical_profile_id != spec.canonical_profile_id:
            raise PermissionError("saved result differs from the requested canonical profile")
        return result

    return result_loader


@dataclass(frozen=True, slots=True)
class SavedStrategyResult:
    """Only the one execution table required by ``compute_strategy_metrics``."""

    core_replay_id: str
    canonical_profile_id: str
    tables: dict[RecordTable, pd.DataFrame]
    effective_section: dict
    gross_trade_stream_hash: str
    artifact_reference: CoreReplayArtifactReference


def load_saved_strategy_result(store_root: Path, core_replay_id: str) -> SavedStrategyResult:
    """Load one exact preserved child's verified execution table without replay.

    Warmup rows, realized values and all execution columns remain unchanged.
    Cost application and warmup exclusion belong to the existing metric code.
    Missing/corrupt evidence raises; it never becomes an empty successful result.
    """
    root = Path(store_root)
    core = load_verified_envelope(root, "core_replays", core_replay_id, CoreStrategyReplayIdentity)
    ref = CoreReplayArtifactReference.model_validate_json(
        load_sidecar_bytes(root, "core_replays", core_replay_id, "artifact_reference.json")
    )
    if ref.core_replay_id != core.core_replay_id:
        raise ArtifactVerificationError("saved result reference belongs to another Core replay")
    exploration = resolve_artifact_directory(root / "v2_datasets", ref.v2_dataset_artifact_id)
    manifest = _read_manifest(exploration, ref.v2_dataset_artifact_id)
    if manifest.get("manifest_schema_version") != 2:
        raise ArtifactVerificationError("saved result requires a v2 manifest")
    if manifest["manifest_payload_sha256"] != ref.manifest_payload_sha256:
        raise ArtifactVerificationError("saved result manifest differs from its Core reference")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or canonical_sha256(identity) != ref.v2_dataset_artifact_id:
        raise ArtifactVerificationError("saved result dataset identity mismatch")
    if identity.get("resolved_profile_hash") != core.payload.resolved_section_config_hash:
        raise ArtifactVerificationError("saved result profile differs from its Core replay")

    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ArtifactVerificationError("saved result manifest artifacts must be a list")
    declared = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ArtifactVerificationError("saved result has an invalid artifact entry")
        relative = entry["path"]
        if relative in declared:
            raise ArtifactVerificationError("saved result has duplicate artifact paths")
        # Validate confinement without opening unneeded tables or source bars.
        declared[relative] = (entry, _artifact_path(exploration, relative))

    def verified_path(filename):
        relative = "exploration/" + filename
        if relative not in declared:
            raise ArtifactVerificationError(f"saved result is missing {filename}")
        entry, path = declared[relative]
        if not path.is_file():
            raise ArtifactVerificationError(f"saved result artifact is missing: {filename}")
        if path.stat().st_size != entry.get("bytes") or file_sha256(path) != entry.get("sha256"):
            raise ArtifactVerificationError(f"saved result artifact changed: {filename}")
        return entry, path

    def report(filename):
        _, path = verified_path(filename)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ArtifactVerificationError(
                f"saved result report is unreadable: {filename}"
            ) from error
        if not isinstance(value, dict):
            raise ArtifactVerificationError(f"saved result report must be an object: {filename}")
        return value

    if report("raw_config.json").get("core_replay_id") != core_replay_id:
        raise ArtifactVerificationError("saved result dataset belongs to another Core replay")
    section = report("effective_config.json").get("section")
    if (
        not isinstance(section, dict)
        or canonical_sha256(section) != identity["resolved_profile_hash"]
    ):
        raise ArtifactVerificationError("saved result effective section identity mismatch")
    if report("invariant_audit.json").get("passed") is not True:
        raise ArtifactVerificationError("saved result did not pass its published invariant audit")

    entry, path = verified_path("executed_trade.parquet")
    trades = pd.read_parquet(path)
    if len(trades) != entry.get("rows"):
        raise ArtifactVerificationError("saved result execution row count mismatch")
    validate_primary_keys(RecordTable.EXECUTED_TRADE, trades)
    validate_table_identity(RecordTable.EXECUTED_TRADE, trades)
    if (
        not trades.empty
        and not trades["envelope_section_config_hash"]
        .eq(core.payload.resolved_section_config_hash)
        .all()
    ):
        raise ArtifactVerificationError("saved executions belong to another section")
    gross_hash = table_content_hash(RecordTable.EXECUTED_TRADE, trades)
    if gross_hash != ref.gross_trade_stream_hash:
        raise ArtifactVerificationError("saved execution stream differs from its Core reference")
    return SavedStrategyResult(
        core_replay_id=core_replay_id,
        canonical_profile_id=core.payload.canonical_profile_id,
        tables={RecordTable.EXECUTED_TRADE: trades},
        effective_section=section,
        gross_trade_stream_hash=gross_hash,
        artifact_reference=ref,
    )
