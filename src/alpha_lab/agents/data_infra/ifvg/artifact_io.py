"""Verified, exact-ID loaders for immutable IFVG v2/v3 artifacts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from .context_contracts import (
    ContextRecordTable,
    validate_context_foreign_keys,
    validate_context_primary_keys,
    validate_context_table_identity,
)
from .context_experiment_contracts import (
    ArtifactPreparationStatus,
    ArtifactReference,
    PairedIfvgArtifactReference,
)
from .context_schemas import (
    IFVG_CONTEXT_ARROW_REGISTRY_HASH,
    IFVG_CONTEXT_ARROW_REGISTRY_ID,
    context_arrow_schema_hash,
    validate_context_arrow_table,
)
from .contracts import (
    RecordTable,
    validate_foreign_keys,
    validate_primary_keys,
    validate_table_identity,
)
from .manifest import canonical_sha256, file_sha256

__all__ = [
    "ArtifactVerificationError",
    "VerifiedIfvgArtifact",
    "VerifiedIfvgPair",
    "resolve_artifact_directory",
    "load_verified_v2_artifact",
    "load_verified_v2_configuration",
    "load_verified_v3_artifact",
    "load_verified_ifvg_pair",
    "load_verified_label_source_bars",
    "validate_exact_context_links",
]

_FULL_ID = re.compile(r"^[0-9a-f]{64}$")


class ArtifactVerificationError(ValueError):
    """An immutable artifact failed content or relational verification."""


@dataclass(frozen=True, slots=True)
class VerifiedIfvgArtifact:
    reference: ArtifactReference
    exploration_dir: Path
    manifest: dict[str, Any]
    tables: dict[RecordTable | ContextRecordTable, pd.DataFrame]
    reports: dict[str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class VerifiedIfvgPair:
    reference: PairedIfvgArtifactReference
    v2: VerifiedIfvgArtifact
    v3: VerifiedIfvgArtifact


def resolve_artifact_directory(root: Path, artifact_id: str) -> Path:
    """Resolve an exact content ID without listing the catalog root."""

    if not _FULL_ID.fullmatch(artifact_id):
        raise ArtifactVerificationError("artifact ID must be a full lowercase SHA-256")
    resolved_root = Path(root).resolve()
    candidate = (resolved_root / artifact_id / "exploration").resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ArtifactVerificationError(
            "artifact resolution escaped its configured root"
        ) from error
    if not candidate.is_dir():
        raise ArtifactVerificationError("artifact ID is not prepared under its configured root")
    return candidate


def _read_manifest(exploration: Path, artifact_id: str) -> dict[str, Any]:
    try:
        manifest = json.loads((exploration / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactVerificationError("artifact manifest is unreadable") from error
    if not isinstance(manifest, dict):
        raise ArtifactVerificationError("artifact manifest must be an object")
    if manifest.get("dataset_id") != artifact_id:
        raise ArtifactVerificationError("artifact manifest dataset ID mismatch")
    claimed = manifest.get("manifest_payload_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    computed = canonical_sha256(core)
    if claimed != computed:
        raise ArtifactVerificationError("artifact manifest payload hash mismatch")
    return manifest


def _artifact_path(exploration: Path, relative: str) -> Path:
    dataset_root = exploration.parent.resolve()
    candidate = (dataset_root / relative).resolve()
    try:
        candidate.relative_to(dataset_root)
    except ValueError as error:
        raise ArtifactVerificationError("manifest artifact path escapes its dataset") from error
    return candidate


def _verify_artifacts(exploration: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ArtifactVerificationError("manifest artifacts must be a list")
    for entry in entries:
        relative = entry.get("path")
        if not isinstance(relative, str) or relative in result:
            raise ArtifactVerificationError("manifest contains an invalid/duplicate artifact path")
        path = _artifact_path(exploration, relative)
        if not path.is_file():
            raise ArtifactVerificationError(f"manifest artifact is missing: {relative}")
        if path.stat().st_size != entry.get("bytes"):
            raise ArtifactVerificationError(f"artifact byte size mismatch: {relative}")
        if file_sha256(path) != entry.get("sha256"):
            raise ArtifactVerificationError(f"artifact SHA-256 mismatch: {relative}")
        if "rows" in entry:
            try:
                rows = pq.read_metadata(path).num_rows
            except Exception as error:
                raise ArtifactVerificationError(
                    f"artifact Parquet metadata is invalid: {relative}"
                ) from error
            if rows != entry["rows"]:
                raise ArtifactVerificationError(f"artifact row-count mismatch: {relative}")
        result[relative] = path
    return result


def _report_files(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for relative, path in paths.items():
        if not relative.endswith(".json") or relative.endswith("manifest.json"):
            continue
        try:
            reports[Path(relative).name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactVerificationError(
                f"artifact JSON report is unreadable: {relative}"
            ) from error
    return reports


def _table_path(paths: dict[str, Path], table_name: str) -> Path:
    suffix = f"/{table_name}.parquet"
    matches = [path for relative, path in paths.items() if relative.endswith(suffix)]
    if len(matches) != 1:
        raise ArtifactVerificationError(
            f"artifact requires exactly one declared {table_name} table"
        )
    return matches[0]


def load_verified_v2_artifact(root: Path, artifact_id: str) -> VerifiedIfvgArtifact:
    exploration = resolve_artifact_directory(root, artifact_id)
    manifest = _read_manifest(exploration, artifact_id)
    if manifest.get("manifest_schema_version") != 2:
        raise ArtifactVerificationError("v2 artifact manifest schema is not 2")
    paths = _verify_artifacts(exploration, manifest)
    tables: dict[RecordTable, pd.DataFrame] = {}
    for table in RecordTable:
        frame = pd.read_parquet(_table_path(paths, table.value))
        validate_primary_keys(table, frame)
        validate_table_identity(table, frame)
        tables[table] = frame
    validate_foreign_keys(tables)
    identity = manifest.get("identity") or {}
    reference = ArtifactReference(
        artifact_id=artifact_id,
        manifest_payload_sha256=manifest["manifest_payload_sha256"],
        artifact_kind="v2",
        dataset_schema_version=int(identity.get("dataset_schema_version", 2)),
        preparation_status=ArtifactPreparationStatus.CONTEXT_READY,
        profile_hash=identity.get("resolved_profile_hash"),
    )
    return VerifiedIfvgArtifact(
        reference=reference,
        exploration_dir=exploration,
        manifest=manifest,
        tables=tables,
        reports=_report_files(paths),
    )


def load_verified_v2_configuration(
    root: Path,
    artifact_id: str,
    *,
    expected_manifest_hash: str,
    expected_profile_hash: str | None = None,
    supported_strategy_source_trees: Collection[str] | None = None,
) -> dict[str, Any]:
    """Read the exact saved strategy section without opening any trade tables.

    This verifies the manifest and configuration, not the complete replay.
    Its result must never be used as evidence that trade tables are valid.
    """
    exploration = resolve_artifact_directory(root, artifact_id)
    manifest = _read_manifest(exploration, artifact_id)
    if manifest.get("manifest_schema_version") != 2:
        raise ArtifactVerificationError("unsupported configuration manifest schema")
    if manifest["manifest_payload_sha256"] != expected_manifest_hash:
        raise ArtifactVerificationError("configuration manifest differs from its exact reference")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or canonical_sha256(identity) != artifact_id:
        raise ArtifactVerificationError("configuration dataset identity mismatch")
    if supported_strategy_source_trees is not None:
        repositories = identity.get("repositories")
        if not isinstance(repositories, list):
            raise ArtifactVerificationError("configuration source evidence is missing")
        strategy_sources = [
            item
            for item in repositories
            if isinstance(item, dict) and item.get("name") == "strategy-core"
        ]
        if (
            len(strategy_sources) != 1
            or strategy_sources[0].get("source_tree_hash") not in supported_strategy_source_trees
        ):
            raise ArtifactVerificationError("configuration uses unreviewed strategy semantics")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list):
        raise ArtifactVerificationError("configuration manifest artifacts must be a list")
    paths: set[str] = set()
    matches = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ArtifactVerificationError("invalid configuration artifact entry")
        relative = entry["path"]
        if relative in paths:
            raise ArtifactVerificationError("duplicate configuration artifact entry")
        paths.add(relative)
        path = _artifact_path(exploration, relative)
        if relative == "exploration/effective_config.json":
            matches.append((entry, path))
    if len(matches) != 1:
        raise ArtifactVerificationError("saved effective configuration is missing")
    entry, path = matches[0]
    try:
        data = path.read_bytes()
        if len(data) != entry.get("bytes") or hashlib.sha256(data).hexdigest() != entry.get(
            "sha256"
        ):
            raise ArtifactVerificationError("saved effective configuration was modified")
        section = json.loads(data)["section"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ArtifactVerificationError("saved effective configuration is unreadable") from error
    if not isinstance(section, dict):
        raise ArtifactVerificationError("saved effective section is not an object")
    # Hash the SAVED mapping, never fill absent fields from today's model defaults.
    section_hash = canonical_sha256(section)
    if section_hash != identity.get("resolved_profile_hash") or (
        expected_profile_hash is not None and section_hash != expected_profile_hash
    ):
        raise ArtifactVerificationError("saved effective section identity mismatch")
    return section


def load_verified_v3_artifact(root: Path, artifact_id: str) -> VerifiedIfvgArtifact:
    exploration = resolve_artifact_directory(root, artifact_id)
    manifest = _read_manifest(exploration, artifact_id)
    if manifest.get("manifest_schema_version") != 4:
        raise ArtifactVerificationError("formula-v2 v3 manifest schema is not 4")
    registry = manifest.get("context_arrow_registry") or {}
    if (
        registry.get("registry_id") != IFVG_CONTEXT_ARROW_REGISTRY_ID
        or registry.get("aggregate_schema_sha256") != IFVG_CONTEXT_ARROW_REGISTRY_HASH
    ):
        raise ArtifactVerificationError("v3 Arrow registry identity mismatch")
    paths = _verify_artifacts(exploration, manifest)
    tables: dict[ContextRecordTable, pd.DataFrame] = {}
    schema_hashes = registry.get("table_schema_sha256") or {}
    artifact_entries = {entry["path"]: entry for entry in manifest["artifacts"]}
    for table in ContextRecordTable:
        path = _table_path(paths, table.value)
        arrow = pq.read_table(path)
        validate_context_arrow_table(table, arrow)
        expected_hash = context_arrow_schema_hash(table)
        relative = next(relative for relative, value in paths.items() if value == path)
        if schema_hashes.get(table.value) != expected_hash:
            raise ArtifactVerificationError(f"v3 registry hash mismatch: {table.value}")
        if artifact_entries[relative].get("arrow_schema_sha256") != expected_hash:
            raise ArtifactVerificationError(f"v3 artifact schema hash mismatch: {table.value}")
        frame = arrow.to_pandas(date_as_object=False)
        validate_context_primary_keys(table, frame)
        validate_context_table_identity(table, frame)
        tables[table] = frame
    validate_context_foreign_keys(tables)
    validate_exact_context_links(tables)
    identity = manifest.get("identity") or {}
    reference = ArtifactReference(
        artifact_id=artifact_id,
        manifest_payload_sha256=manifest["manifest_payload_sha256"],
        artifact_kind="v3",
        dataset_schema_version=int(identity.get("dataset_schema_version", 0)),
        preparation_status=ArtifactPreparationStatus.CONTEXT_READY,
        profile_hash=identity.get("resolved_profile_hash"),
        feature_formula_version=identity.get("feature_formula_version"),
    )
    return VerifiedIfvgArtifact(
        reference=reference,
        exploration_dir=exploration,
        manifest=manifest,
        tables=tables,
        reports=_report_files(paths),
    )


def validate_exact_context_links(
    tables: dict[ContextRecordTable, pd.DataFrame],
    *,
    core_tables: dict[RecordTable, pd.DataFrame] | None = None,
) -> None:
    captures = tables[ContextRecordTable.CONTEXT_CAPTURE]
    capture_index = captures.set_index("context_capture_id", verify_integrity=True)
    link_specs = (
        (
            ContextRecordTable.CANDIDATE_CONTEXT_LINK,
            "candidate_id",
            "entry_candidate",
        ),
        (
            ContextRecordTable.DECISION_CONTEXT_LINK,
            "decision_id",
            "eligible_decision",
        ),
        (
            ContextRecordTable.TRADE_CONTEXT_LINK,
            "trade_id",
            "executed_trade_link",
        ),
    )
    for table, primary_key, expected_kind in link_specs:
        frame = tables[table]
        if frame.empty:
            continue
        if frame[primary_key].duplicated().any():
            raise ArtifactVerificationError(f"{table.value} has duplicate exact links")
        context_ts = pd.to_datetime(frame["context_as_of_ts"], utc=True, errors="raise")
        feature_ts = pd.to_datetime(frame["feature_as_of_ts"], utc=True, errors="raise")
        if (context_ts > feature_ts).any():
            raise ArtifactVerificationError(f"{table.value} violates context as-of causality")
        for row in frame.to_dict("records"):
            capture_id = row["context_capture_id"]
            if capture_id not in capture_index.index:
                raise ArtifactVerificationError(f"{table.value} has a missing exact capture")
            capture = capture_index.loc[capture_id]
            if row.get("stage") != expected_kind or row.get("capture_kind") != expected_kind:
                raise ArtifactVerificationError(f"{table.value} has a mismatched stage")
            if str(capture["capture_kind"]) != expected_kind:
                raise ArtifactVerificationError(
                    f"{table.value} points to the wrong capture kind"
                )
            if str(capture[primary_key]) != str(row[primary_key]):
                raise ArtifactVerificationError(
                    f"{table.value} does not match exact capture {primary_key}"
                )
            if str(capture["evidence_id"]) != str(row[primary_key]):
                raise ArtifactVerificationError(
                    f"{table.value} capture evidence is not its exact stage record"
                )
            exact_fields = [("geometry_evidence_cursor", "evidence_cursor")]
            if "context_state_id" in row:
                exact_fields.append(("context_state_id", "context_state_id"))
            for link_name, capture_name in exact_fields:
                if str(row[link_name]) != str(capture[capture_name]):
                    raise ArtifactVerificationError(
                        f"{table.value} does not match exact capture {capture_name}"
                    )
            capture_ts = pd.Timestamp(capture["as_of_ts"])
            if capture_ts != pd.Timestamp(row["context_as_of_ts"]):
                raise ArtifactVerificationError(
                    f"{table.value} context timestamp differs from its exact capture"
                )

    if core_tables is None:
        return
    validate_context_foreign_keys(tables, core_tables=core_tables)
    candidates = core_tables[RecordTable.ENTRY_CANDIDATE].set_index(
        "candidate_id", verify_integrity=True
    )
    links = tables[ContextRecordTable.CANDIDATE_CONTEXT_LINK]
    for row in links.to_dict("records"):
        if row["candidate_id"] not in candidates.index:
            raise ArtifactVerificationError("candidate link has no exact v2 candidate")
        candidate = candidates.loc[row["candidate_id"]]
        for core_name, link_name in (
            ("trigger_evidence_id", "geometry_evidence_id"),
            ("trigger_cursor", "geometry_evidence_cursor"),
        ):
            if (
                core_name in candidate.index
                and pd.notna(candidate[core_name])
                and str(candidate[core_name]) != str(row[link_name])
            ):
                raise ArtifactVerificationError(
                    f"candidate link does not match exact v2 {core_name}"
                )


def load_verified_ifvg_pair(
    *,
    v2_root: Path,
    v2_artifact_id: str,
    v3_root: Path,
    v3_artifact_id: str,
) -> VerifiedIfvgPair:
    v2 = load_verified_v2_artifact(v2_root, v2_artifact_id)
    v3 = load_verified_v3_artifact(v3_root, v3_artifact_id)
    accepted = v3.manifest.get("accepted_v2_reference") or {}
    if accepted.get("dataset_id") != v2.reference.artifact_id:
        raise ArtifactVerificationError("v3 references a different v2 artifact ID")
    if accepted.get("manifest_payload_sha256") != v2.reference.manifest_payload_sha256:
        raise ArtifactVerificationError("v3 references a different v2 manifest hash")
    if accepted.get("core_tables_duplicated") is not False:
        raise ArtifactVerificationError("v3 artifact improperly duplicates core tables")
    if v2.reference.profile_hash != v3.reference.profile_hash:
        raise ArtifactVerificationError("v2/v3 resolved profile hashes differ")
    validate_exact_context_links(v3.tables, core_tables=v2.tables)
    return VerifiedIfvgPair(
        reference=PairedIfvgArtifactReference(v2=v2.reference, v3=v3.reference),
        v2=v2,
        v3=v3,
    )


def load_verified_label_source_bars(artifact: VerifiedIfvgArtifact) -> pd.DataFrame:
    if artifact.reference.artifact_kind != "v2":
        raise ArtifactVerificationError("label-source bars must belong to a v2 artifact")
    matching = [
        entry
        for entry in artifact.manifest.get("artifacts", ())
        if entry.get("path", "").endswith("/label_source_1m.parquet")
    ]
    if len(matching) != 1:
        raise ArtifactVerificationError("v2 artifact has no unique verified label-source stream")
    path = _artifact_path(artifact.exploration_dir, matching[0]["path"])
    bars = pd.read_parquet(path)
    required = {"source_date", "bar_id", "close_ts_utc", "high_ticks", "low_ticks"}
    if not required.issubset(bars):
        raise ArtifactVerificationError("verified label-source stream has missing columns")
    if bars["bar_id"].isna().any() or bars["bar_id"].astype(str).duplicated().any():
        raise ArtifactVerificationError("verified label-source bar IDs are invalid")
    closes = pd.to_datetime(bars["close_ts_utc"], utc=True, errors="raise")
    if (closes > pd.Timestamp("2026-06-10T21:00:00Z")).any():
        raise ArtifactVerificationError("verified label-source stream exceeds cutoff")
    return bars.sort_values(["close_ts_utc", "bar_id"], kind="mergesort").reset_index(
        drop=True
    )
