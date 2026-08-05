"""Content-addressed, immutable IFVG v2 dataset manifests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from strategy_core.strategies.ifvg_smc.records import IFVG_RECORD_SCHEMA_VERSION

from .context_contracts import (
    IFVG_CONTEXT_CONTAINER_SCHEMA_VERSION,
    IFVG_CONTEXT_DATASET_SCHEMA_VERSION,
    ContextRecordTable,
    context_count_reconciliation,
    validate_context_foreign_keys,
    validate_context_primary_keys,
    validate_context_table_identity,
)
from .context_schemas import (
    IFVG_CONTEXT_ARROW_REGISTRY_HASH,
    IFVG_CONTEXT_ARROW_REGISTRY_ID,
    context_arrow_schema_hash,
    context_table_from_frame,
)
from .contracts import (
    IFVG_CAPTURE_SCHEMA_VERSION,
    IFVG_DATASET_SCHEMA_VERSION,
    IFVG_REPORT_SCHEMA_VERSION,
    RecordTable,
    count_reconciliation,
    validate_foreign_keys,
    validate_primary_keys,
    validate_table_identity,
)

__all__ = [
    "RepositoryState",
    "DatasetIdentity",
    "canonical_sha256",
    "file_sha256",
    "source_tree_hash",
    "read_repository_state",
    "dataset_id_for",
    "save_v2_dataset_immutable",
    "V3DatasetIdentity",
    "v3_dataset_id_for",
    "save_v3_dataset_immutable",
    "FsmAuditIdentity",
    "fsm_audit_dataset_id_for",
    "save_fsm_audit_immutable",
]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: Path, relative_paths: tuple[str, ...]) -> str:
    """Hash explicitly scoped source/docs/tests without reading data artifacts."""
    root = Path(root).resolve()
    files: set[Path] = set()
    for relative in relative_paths:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source hash path escapes repository: {relative}") from exc
        if candidate.is_file():
            files.add(candidate)
        elif candidate.is_dir():
            files.update(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and ".pytest_cache" not in path.parts
                and ".ruff_cache" not in path.parts
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class RepositoryState:
    name: str
    path: str
    head: str
    dirty_status_sha256: str
    source_tree_hash: str

    def identity_payload(self) -> dict:
        """Repository content identity; local checkout paths are provenance only."""
        return {
            "name": self.name,
            "head": self.head,
            "dirty_status_sha256": self.dirty_status_sha256,
            "source_tree_hash": self.source_tree_hash,
        }


def read_repository_state(
    name: str,
    root: Path,
    *,
    source_paths: tuple[str, ...],
) -> RepositoryState:
    """Capture Git identity plus exact, source-scoped working-tree provenance.

    The dirty-status hash uses the same path scope as ``source_tree_hash``.
    Generated datasets and unrelated artifacts therefore cannot change the
    identity of an otherwise identical replay or bypass immutable-save refusal.
    """
    root = Path(root).resolve()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            *source_paths,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return RepositoryState(
        name=name,
        path=str(root),
        head=head,
        dirty_status_sha256=hashlib.sha256(status.encode("utf-8")).hexdigest(),
        source_tree_hash=source_tree_hash(root, source_paths),
    )


@dataclass(frozen=True)
class DatasetIdentity:
    repositories: tuple[RepositoryState, ...]
    authoritative_source_blob: str
    resolved_profile_hash: str
    evaluation_config_hash: str
    date_allowlist: tuple[str, ...]
    permitted_source_hashes: tuple[tuple[str, str], ...]
    record_schema_version: int = IFVG_RECORD_SCHEMA_VERSION
    capture_schema_version: int = IFVG_CAPTURE_SCHEMA_VERSION
    dataset_schema_version: int = IFVG_DATASET_SCHEMA_VERSION
    report_schema_version: int = IFVG_REPORT_SCHEMA_VERSION

    def payload(self) -> dict:
        return {
            **asdict(self),
            "repositories": [
                state.identity_payload()
                for state in sorted(self.repositories, key=lambda item: item.name)
            ],
            "date_allowlist": sorted(self.date_allowlist),
            "permitted_source_hashes": [
                list(item)
                for item in sorted(self.permitted_source_hashes)
            ],
        }


def dataset_id_for(identity: DatasetIdentity) -> str:
    """Full SHA-256 content identity; directory names are never mutable labels."""
    return canonical_sha256(identity.payload())


def _json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _artifact_entry(path: Path, root: Path, *, rows: int | None = None) -> dict:
    entry = {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = int(rows)
    return entry


def save_v2_dataset_immutable(
    *,
    base_dir: Path,
    identity: DatasetIdentity,
    raw_config: dict,
    effective_config: dict,
    tables: dict[RecordTable, pd.DataFrame],
    candidate_report: dict,
    decision_report: dict,
    executed_trade_report: dict,
    invariant_audit: dict,
    count_reconciliation_report: dict | None = None,
    data_access_audit: dict,
    label_source_bars: pd.DataFrame | None = None,
) -> Path:
    """Write one content-derived dataset exactly once, then atomically publish."""
    if invariant_audit.get("passed") is not True:
        raise ValueError("immutable IFVG v2 save requires a passing invariant audit")
    dataset_id = dataset_id_for(identity)
    base_dir = Path(base_dir).resolve()
    destination = base_dir / dataset_id
    if destination.exists():
        raise FileExistsError(
            f"immutable IFVG v2 dataset already exists: {destination}"
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    temporary = base_dir / f".{dataset_id}.tmp-{uuid.uuid4().hex}"
    exploration = temporary / "exploration"
    exploration.mkdir(parents=True)

    try:
        normalized: dict[RecordTable, pd.DataFrame] = {}
        for table in RecordTable:
            frame = tables.get(table, pd.DataFrame()).copy()
            validate_primary_keys(table, frame)
            validate_table_identity(table, frame)
            normalized[table] = frame
        validate_foreign_keys(normalized)
        computed_reconciliation = count_reconciliation(normalized)
        if (
            count_reconciliation_report is not None
            and count_reconciliation_report != computed_reconciliation
        ):
            raise ValueError(
                "supplied IFVG count reconciliation does not match table contents"
            )
        reconciliation = computed_reconciliation

        artifacts: list[dict] = []
        for filename, payload in (
            ("raw_config.json", raw_config),
            ("effective_config.json", effective_config),
            ("candidate_report.json", candidate_report),
            ("decision_report.json", decision_report),
            ("executed_trade_report.json", executed_trade_report),
            ("invariant_audit.json", invariant_audit),
            ("count_reconciliation.json", reconciliation),
            ("data_access_audit.json", data_access_audit),
        ):
            path = exploration / filename
            _json_write(path, payload)
            artifacts.append(_artifact_entry(path, temporary))

        for table in RecordTable:
            path = exploration / f"{table.value}.parquet"
            normalized[table].to_parquet(path, index=False)
            artifacts.append(
                _artifact_entry(path, temporary, rows=len(normalized[table]))
            )
        if label_source_bars is not None:
            required = {
                "source_date",
                "bar_id",
                "close_ts_utc",
                "high_ticks",
                "low_ticks",
            }
            missing = sorted(required - set(label_source_bars))
            if missing:
                raise ValueError(f"label-source bars are missing columns {missing}")
            if label_source_bars["bar_id"].isna().any() or label_source_bars[
                "bar_id"
            ].duplicated().any():
                raise ValueError("label-source bar IDs must be unique and non-null")
            closes = pd.to_datetime(
                label_source_bars["close_ts_utc"], utc=True, errors="raise"
            )
            cutoff = pd.Timestamp("2026-06-10T21:00:00Z")
            if (closes > cutoff).any():
                raise ValueError("label-source bars exceed the development cutoff")
            path = exploration / "label_source_1m.parquet"
            label_source_bars.to_parquet(path, index=False)
            artifacts.append(
                _artifact_entry(path, temporary, rows=len(label_source_bars))
            )

        manifest_core = {
            "manifest_schema_version": 2,
            "dataset_id": dataset_id,
            "immutable": True,
            "scope": "exploration",
            "identity": identity.payload(),
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            "count_reconciliation": reconciliation,
            "invariant_audit_passed": bool(invariant_audit.get("passed")),
        }
        manifest = {
            **manifest_core,
            "manifest_payload_sha256": canonical_sha256(manifest_core),
        }
        _json_write(exploration / "manifest.json", manifest)

        # Directory rename is the publication point.  A destination check is
        # repeated immediately beforehand to fail closed under a concurrent run.
        if destination.exists():
            raise FileExistsError(
                f"immutable IFVG v2 dataset appeared concurrently: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination / "exploration"


@dataclass(frozen=True)
class V3DatasetIdentity:
    """Content identity for additive context tables referencing accepted v2."""

    repositories: tuple[RepositoryState, ...]
    authoritative_source_blob: str
    accepted_v2_dataset_id: str
    accepted_v2_manifest_payload_sha256: str
    resolved_profile_hash: str
    feature_set_version: str
    feature_formula_version: str
    feature_schema_hash: str
    context_config_hash: str
    normalized_timeframes: tuple[str, ...]
    anchor_status_240m: str
    date_allowlist: tuple[str, ...]
    warmup_dates: tuple[str, ...]
    evidence_dates: tuple[str, ...]
    permitted_source_hashes: tuple[tuple[str, str], ...]
    context_record_schema_version: int = 2
    context_container_schema_version: int = IFVG_CONTEXT_CONTAINER_SCHEMA_VERSION
    dataset_schema_version: int = IFVG_CONTEXT_DATASET_SCHEMA_VERSION
    context_arrow_registry: str = IFVG_CONTEXT_ARROW_REGISTRY_ID
    context_arrow_registry_hash: str = IFVG_CONTEXT_ARROW_REGISTRY_HASH
    context_arrow_schema_hashes: tuple[tuple[str, str], ...] = tuple(
        (table.value, context_arrow_schema_hash(table)) for table in ContextRecordTable
    )

    def payload(self) -> dict:
        return {
            **asdict(self),
            "repositories": [
                state.identity_payload()
                for state in sorted(self.repositories, key=lambda item: item.name)
            ],
            "date_allowlist": sorted(self.date_allowlist),
            "warmup_dates": list(self.warmup_dates),
            "evidence_dates": list(self.evidence_dates),
            "permitted_source_hashes": [
                list(item) for item in sorted(self.permitted_source_hashes)
            ],
        }


def v3_dataset_id_for(identity: V3DatasetIdentity) -> str:
    return canonical_sha256(identity.payload())


def _reject_performance_payload(value: Any, *, path: str = "report") -> None:
    forbidden = (
        "profit",
        "pnl",
        "return",
        "drawdown",
        "mfe",
        "mae",
        "realized",
        "outcome",
        "label",
        "win_rate",
    )
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            allowed_access_counter = (
                path.startswith("data_access_audit.json")
                and lowered == "label_bar"
                and type(item) is int
                and item >= 0
            )
            if any(token in lowered for token in forbidden) and not allowed_access_counter:
                raise ValueError(f"v3 verification report contains forbidden key {path}.{key}")
            _reject_performance_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_performance_payload(item, path=f"{path}[{index}]")


def save_v3_dataset_immutable(
    *,
    base_dir: Path,
    identity: V3DatasetIdentity,
    raw_config: dict,
    effective_config: dict,
    context_tables: dict[ContextRecordTable, pd.DataFrame],
    validity_report: dict,
    coverage_report: dict,
    reconciliation_report: dict,
    capacity_report: dict,
    identity_report: dict,
    performance_report: dict,
    data_access_audit: dict,
    diagnostics_out: dict[str, object] | None = None,
) -> Path:
    """Persist only new normalized v3 evidence, atomically and exactly once."""

    persistence_started_ns = perf_counter_ns()
    expected_schema_hashes = tuple(
        (table.value, context_arrow_schema_hash(table)) for table in ContextRecordTable
    )
    if (
        identity.context_record_schema_version != 2
        or identity.context_container_schema_version != 4
        or identity.dataset_schema_version != 4
        or identity.context_arrow_registry != IFVG_CONTEXT_ARROW_REGISTRY_ID
        or identity.context_arrow_registry_hash != IFVG_CONTEXT_ARROW_REGISTRY_HASH
        or identity.context_arrow_schema_hashes != expected_schema_hashes
    ):
        raise ValueError("v3 identity does not match the locked formula-v2 Arrow contract")

    reports = {
        "validity_report.json": validity_report,
        "coverage_report.json": coverage_report,
        "reconciliation_report.json": reconciliation_report,
        "capacity_report.json": capacity_report,
        "identity_report.json": identity_report,
        "performance_report.json": performance_report,
        "data_access_audit.json": data_access_audit,
    }
    for name, report in reports.items():
        _reject_performance_payload(report, path=name)
    if reconciliation_report.get("passed") is not True:
        raise ValueError("immutable IFVG v3 save requires passing reconciliation")
    if identity_report.get("passed") is not True:
        raise ValueError("immutable IFVG v3 save requires passing identity checks")
    if validity_report.get("passed") is not True:
        raise ValueError("immutable IFVG v3 save requires passing validity checks")
    if capacity_report.get("passed") is not True:
        raise ValueError("immutable IFVG v3 save requires passing capacity checks")
    if performance_report.get("passed") is not True:
        raise ValueError("immutable IFVG v3 save requires passing performance checks")

    normalized: dict[ContextRecordTable, pd.DataFrame] = {}
    arrow_tables = {}
    for table in ContextRecordTable:
        frame = context_tables.get(table, pd.DataFrame()).copy()
        validate_context_primary_keys(table, frame)
        validate_context_table_identity(table, frame)
        normalized[table] = frame
        arrow_tables[table] = context_table_from_frame(table, frame)
    validate_context_foreign_keys(normalized)
    counts = context_count_reconciliation(normalized)

    dataset_id = v3_dataset_id_for(identity)
    base_dir = Path(base_dir).resolve()
    destination = base_dir / dataset_id
    if destination.exists():
        raise FileExistsError(
            f"immutable IFVG v3 dataset already exists: {destination}"
        )
    base_dir.mkdir(parents=True, exist_ok=True)
    temporary = base_dir / f".{dataset_id}.tmp-{uuid.uuid4().hex}"
    exploration = temporary / "exploration"
    exploration.mkdir(parents=True)

    try:
        artifacts: list[dict] = []
        for filename, payload in (
            ("raw_config.json", raw_config),
            ("effective_config.json", effective_config),
            *reports.items(),
            ("context_count_reconciliation.json", counts),
        ):
            path = exploration / filename
            _json_write(path, payload)
            artifacts.append(_artifact_entry(path, temporary))
        for table in ContextRecordTable:
            path = exploration / f"{table.value}.parquet"
            pq.write_table(arrow_tables[table], path)
            entry = _artifact_entry(path, temporary, rows=len(normalized[table]))
            entry["arrow_schema_sha256"] = context_arrow_schema_hash(table)
            artifacts.append(entry)

        manifest_core = {
            "manifest_schema_version": 4,
            "dataset_id": dataset_id,
            "immutable": True,
            "scope": "exploration",
            "identity": identity.payload(),
            "accepted_v2_reference": {
                "dataset_id": identity.accepted_v2_dataset_id,
                "manifest_payload_sha256": (
                    identity.accepted_v2_manifest_payload_sha256
                ),
                "core_tables_duplicated": False,
            },
            "context_arrow_registry": {
                "registry_id": IFVG_CONTEXT_ARROW_REGISTRY_ID,
                "aggregate_schema_sha256": IFVG_CONTEXT_ARROW_REGISTRY_HASH,
                "table_schema_sha256": {
                    table.value: context_arrow_schema_hash(table)
                    for table in ContextRecordTable
                },
            },
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            "context_count_reconciliation": counts,
        }
        _json_write(
            exploration / "manifest.json",
            {
                **manifest_core,
                "manifest_payload_sha256": canonical_sha256(manifest_core),
            },
        )
        if destination.exists():
            raise FileExistsError(
                f"immutable IFVG v3 dataset appeared concurrently: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        if diagnostics_out is not None:
            diagnostics_out["persistence_failed_seconds"] = (
                perf_counter_ns() - persistence_started_ns
            ) / 1_000_000_000
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    if diagnostics_out is not None:
        diagnostics_out["persistence_seconds"] = (
            perf_counter_ns() - persistence_started_ns
        ) / 1_000_000_000
    return destination / "exploration"


# ── FSM audit companion artifact (ifvg_fsm_audit_v1) ─────────────────────────


@dataclass(frozen=True)
class FsmAuditIdentity:
    """Content identity for the additive FSM audit companion artifact.

    Pins the NEW Strategy-Core commit (via ``repositories``), the accepted
    FINAL-REVIEW v2 dataset (``143b510f…`` lineage — deliberately distinct
    from the v3 baseline pin; see the open-decisions register), the resolved
    profile/evaluation hashes, and the audit contract fingerprint. NO
    outcome or performance fields may enter this identity or its reports.
    """

    repositories: tuple
    authoritative_source_blob: str
    accepted_v2_dataset_id: str
    accepted_v2_manifest_payload_sha256: str
    resolved_profile_hash: str
    evaluation_config_hash: str
    date_allowlist: tuple[str, ...]
    permitted_source_hashes: tuple[tuple[str, str], ...]
    audit_contract_fingerprint_sha256: str
    audit_schema_version: int = 1
    record_schema_version: int = IFVG_RECORD_SCHEMA_VERSION
    capture_schema_version: int = IFVG_CAPTURE_SCHEMA_VERSION

    def payload(self) -> dict:
        return {
            **asdict(self),
            "repositories": [
                state.identity_payload()
                for state in sorted(self.repositories, key=lambda item: item.name)
            ],
            "date_allowlist": sorted(self.date_allowlist),
            "permitted_source_hashes": [
                list(item) for item in sorted(self.permitted_source_hashes)
            ],
        }


def fsm_audit_dataset_id_for(identity: FsmAuditIdentity) -> str:
    return canonical_sha256(identity.payload())


def save_fsm_audit_immutable(
    *,
    base_dir: Path,
    identity: FsmAuditIdentity,
    raw_config: dict,
    effective_config: dict,
    audit_tables: dict,
    parity_report: dict,
    reconciliation_report: dict,
    capacity_report: dict,
    coverage_report: dict,
    data_access_audit: dict,
) -> Path:
    """Write one content-derived FSM audit artifact exactly once, then
    atomically publish — the same overwrite-refusing discipline as the v2/v3
    savers. Publication REQUIRES a passed exact-parity report and a passed
    funnel reconciliation."""
    from .audit_contracts import (
        AuditTable,
        audit_contract_fingerprint,
        validate_audit_links,
        validate_audit_table,
    )

    if parity_report.get("passed") is not True:
        raise ValueError("fsm audit save requires a PASSED exact v2 parity report")
    if reconciliation_report.get("passed") is not True:
        raise ValueError("fsm audit save requires a PASSED funnel reconciliation")
    fingerprint = audit_contract_fingerprint()
    expected_fingerprint = canonical_sha256(fingerprint)
    if identity.audit_contract_fingerprint_sha256 != expected_fingerprint:
        raise ValueError("fsm audit identity fingerprint does not match the contract")
    _reject_performance_payload(capacity_report, path="capacity_report")
    _reject_performance_payload(coverage_report, path="coverage_report")

    dataset_id = fsm_audit_dataset_id_for(identity)
    base_dir = Path(base_dir).resolve()
    destination = base_dir / dataset_id
    if destination.exists():
        raise FileExistsError(f"immutable IFVG fsm audit already exists: {destination}")
    base_dir.mkdir(parents=True, exist_ok=True)
    temporary = base_dir / f".{dataset_id}.tmp-{uuid.uuid4().hex}"
    exploration = temporary / "exploration"
    exploration.mkdir(parents=True)
    try:
        for table in AuditTable:
            frame = audit_tables.get(table, pd.DataFrame())
            if not frame.empty:
                validate_audit_table(table, frame)
        validate_audit_links(audit_tables)

        artifacts: list[dict] = []
        for filename, payload in (
            ("raw_config.json", raw_config),
            ("effective_config.json", effective_config),
            ("parity_report.json", parity_report),
            ("count_reconciliation.json", reconciliation_report),
            ("capacity_report.json", capacity_report),
            ("evidence_coverage.json", coverage_report),
            ("audit_contract.json", fingerprint),
            ("data_access_audit.json", data_access_audit),
        ):
            path = exploration / filename
            _json_write(path, payload)
            artifacts.append(_artifact_entry(path, temporary))
        for table in AuditTable:
            frame = audit_tables.get(table, pd.DataFrame())
            path = exploration / f"{table.value}.parquet"
            frame.to_parquet(path, index=False)
            artifacts.append(_artifact_entry(path, temporary, rows=len(frame)))

        manifest_core = {
            "manifest_schema_version": 1,
            "artifact_kind": "ifvg_fsm_audit_v1",
            "dataset_id": dataset_id,
            "immutable": True,
            "scope": "exploration",
            "identity": identity.payload(),
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
            "parity_passed": True,
            "reconciliation_passed": True,
        }
        manifest = {
            **manifest_core,
            "manifest_payload_sha256": canonical_sha256(manifest_core),
        }
        _json_write(exploration / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError(
                f"immutable IFVG fsm audit appeared concurrently: {destination}"
            )
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination / "exploration"
