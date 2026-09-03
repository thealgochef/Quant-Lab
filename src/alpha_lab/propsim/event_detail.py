"""Versioned, compressed, bounded prop-event detail (R6.1 D15; plan §6.G).

R3–R6 account simulations persisted the ordered account-event stream as
``account_events.json`` for HISTORICAL modes only — an unbounded
row-oriented JSON sidecar that bootstrap / stress simulations (thousands of
synthetic-clock paths) could never carry. D15 adds an IMMUTABLE, versioned,
compressed, partitioned, and budgeted representation for every mode:

* ``event_detail_persistence_policy_id ∈ {none_v0,
  account_event_detail_by_path_parquet_v2}`` plus the storage policy, the
  schema version, and the registered budget ENTER the account- and
  portfolio-simulation identities (``simulation.py``) — changing the
  representation, the budget, or the policy mints a new simulation id, and
  an artifact persisted under ``none_v0`` is never widened (the immutable
  store refuses a second publication of the same id with different
  sidecars).
* Under v2 the writer streams ZSTD Parquet partitions by
  ``path_block_id = floor(path_ordinal / path_block_size)`` carrying the
  exact event timestamp (verbatim ISO-8601 plus its UTC nanosecond value),
  the trading day the walk played, the clock policy, the total-order fields
  (``path_ordinal``, ``event_ordinal`` — the walk's ONE strictly ordered
  stream), path / account / event ids, source trade / candidate, and
  type / phase / amount. A preflight row count, a streaming row overrun, or
  a streaming byte overrun fails BEFORE atomic publication, so no partial
  artifact ever exists. The writer is a store *sidecar producer* (R6.1
  safety review S12): each partition is built column-wise for ONE path
  block, written straight into the store's temporary publication directory,
  hashed by streaming and released.
* HARDENING-BACKEND §4.4 (F-17): the writer consumes an ITERABLE of
  ``(path_record, walk_result)`` pairs in draw-ordinal order exactly once —
  a generator is never materialized into an all-path event list — and the
  caller declares the preflight ``total_rows`` / ``path_count`` it will
  stream (verified exactly at the end). No whole-artifact event-id index
  exists any more: event ids are the deterministic SHA-256 projection of a
  key that includes ``path_instance_id`` and the per-path ``event_ordinal``
  (``AccountWalk._emit``; see :data:`EVENT_ID_CANONICAL_KEY`), so unique
  path ids × strictly increasing per-path ordinals — both enforced while
  streaming — imply unique ids for every production emitter. Because the
  writer cannot re-derive a foreign envelope's projection (the emitter's
  ``account_namespace`` is not on the envelope), the ARTIFACT-level proof is
  a disk-backed DuckDB distinct check over the written partitions under an
  explicit memory limit and an attempt-local spill directory
  (:func:`_external_uniqueness_check`, :data:`EVENT_ID_UNIQUENESS_CHECK_V1`)
  — it also proves the path ids are one-to-one with the path ordinals. A
  forged duplicate across path blocks is refused before publication; the
  process never holds more than one block of rows.
* Every partition and the ``event_detail_manifest.json`` index are
  manifest-listed with sha256 / bytes / rows / schema hash; the reader
  verifies the store manifest, the detail manifest, and every partition's
  bytes, row count, and schema before yielding a frame.

The raw ``account_events.json`` detail (historical modes) remains audit
evidence; the Parquet detail is the bounded analytical representation the
regime-stratified prop reports consume through ``source_trade_id`` /
``source_candidate_id`` (bootstrap / stress rows attribute ONLY through
their source trade — the synthetic clock is descriptive, never a join key).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_serializer

from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256, file_sha256
from alpha_lab.agents.data_infra.ifvg.search.identities import FrozenContract
from alpha_lab.agents.data_infra.ifvg.search.store import ProducedSidecar, SidecarProducer

__all__ = [
    "EVENT_DETAIL_POLICY_NONE",
    "EVENT_DETAIL_POLICY_PARQUET_V2",
    "EVENT_DETAIL_PERSISTENCE_POLICIES",
    "EVENT_DETAIL_STORAGE_NONE",
    "EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2",
    "EVENT_DETAIL_SCHEMA_VERSION_NONE",
    "EVENT_DETAIL_SCHEMA_VERSION_V2",
    "EVENT_DETAIL_MANIFEST_SIDECAR",
    "EVENT_DETAIL_PARTITION_PREFIX",
    "EVENT_DETAIL_SCHEMA",
    "EVENT_DETAIL_SCHEMA_HASH",
    "EVENT_DETAIL_ROW_GROUP_SIZE",
    "EVENT_ID_CANONICAL_KEY",
    "EVENT_ID_UNIQUENESS_CHECK_V1",
    "EXTERNAL_CHECK_MEMORY_LIMIT_BYTES",
    "EVENT_TYPE_PRECEDENCE",
    "EVENT_AMOUNT_FIELD",
    "EventDetailBudget",
    "EVENT_DETAIL_BUDGET_V1",
    "EVENT_DETAIL_BUDGET_V2",
    "EVENT_DETAIL_PARTITION_BOUND_POLICY_V2",
    "EventDetailBudgetError",
    "EventDetailIntegrityError",
    "EventDetailUnavailableError",
    "EventDetailBundle",
    "assert_event_detail_policy_coherent",
    "event_detail_identity_fields",
    "build_account_event_detail",
    "account_event_detail_producer",
    "partition_sidecar_name",
    "load_account_event_detail_manifest",
    "load_account_event_detail",
]

EventDetailPersistencePolicy = Literal["none_v0", "account_event_detail_by_path_parquet_v2"]
EventDetailStoragePolicy = Literal["none", "zstd_parquet_path_blocks_v2"]

EVENT_DETAIL_POLICY_NONE = "none_v0"
EVENT_DETAIL_POLICY_PARQUET_V2 = "account_event_detail_by_path_parquet_v2"
EVENT_DETAIL_PERSISTENCE_POLICIES: tuple[str, ...] = (
    EVENT_DETAIL_POLICY_NONE,
    EVENT_DETAIL_POLICY_PARQUET_V2,
)
EVENT_DETAIL_STORAGE_NONE = "none"
EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2 = "zstd_parquet_path_blocks_v2"
EVENT_DETAIL_SCHEMA_VERSION_NONE = 0
EVENT_DETAIL_SCHEMA_VERSION_V2 = 2
EVENT_DETAIL_MANIFEST_SIDECAR = "event_detail_manifest.json"
EVENT_DETAIL_PARTITION_PREFIX = "event_detail_block_"
#: Bounded Parquet row groups (the writer's setting since R6.1 — unchanged,
#: so partition bytes are byte-identical to the R6.1 writer).
EVENT_DETAIL_ROW_GROUP_SIZE = 65_536
#: The key ``AccountWalk._emit`` hashes into ``event_id`` (with the account
#: namespace / ordinal, the type, the instant and the body): unique path ids
#: × strictly increasing per-path ordinals imply unique ids.
EVENT_ID_CANONICAL_KEY: tuple[str, ...] = ("path_instance_id", "event_ordinal")
#: The artifact-level uniqueness proof over the WRITTEN partitions.
EVENT_ID_UNIQUENESS_CHECK_V1 = "external_duckdb_distinct_v1"
#: The explicit DuckDB memory limit of the external check (spills to the
#: attempt-local temp directory beyond it).
EXTERNAL_CHECK_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
_EXTERNAL_CHECK_THREADS = 2
_PARTITION_SUFFIX = ".parquet"
_STORE = "account_simulations"


#: HARDENING-BACKEND-FIX §9: the versioned, identity-bearing resident-row
#: bound of the writer — every partition holds at most
#: ``max_rows_per_partition`` rows and the writer never holds more rows than
#: that in memory, whatever the events-per-path shape (a path block, even a
#: single path, is split across partitions at the bound).
EVENT_DETAIL_PARTITION_BOUND_POLICY_V2 = "event_detail_partition_row_bound_v2"


class EventDetailBudget(FrozenContract):
    """Registered storage budget — part of the simulation identity.

    ``max_rows_per_partition`` (HARDENING-BACKEND-FIX §9) is the hard
    per-partition / resident-row bound of the writer. A pre-bound budget
    (``event_detail_budget_v1``) carries none: it stays LOADABLE (its
    serialization is unchanged — the absent bound is omitted, so every
    identity minted under it is preserved) but the row-bounded writer
    refuses to write under it.
    """

    budget_id: str = Field(min_length=1)
    max_event_detail_rows: int = Field(ge=1)
    max_published_bytes: int = Field(ge=1)
    path_block_size: int = Field(ge=1)
    max_rows_per_partition: int | None = Field(default=None, ge=1)

    @model_serializer(mode="wrap")
    def _omit_absent_bound(self, handler):
        data = handler(self)
        if isinstance(data, dict) and data.get("max_rows_per_partition") is None:
            data.pop("max_rows_per_partition", None)
        return data


#: The registered V1 limits (plan §6.G / D15) — pre-bound; loadable, not writable.
EVENT_DETAIL_BUDGET_V1 = EventDetailBudget(
    budget_id="event_detail_budget_v1",
    max_event_detail_rows=10_000_000,
    max_published_bytes=2_147_483_648,
    path_block_size=250,
)
#: The registered V2 limits (HARDENING-BACKEND-FIX §9): V1 plus the hard
#: per-partition row bound — the benchmark's measured row-group size (250
#: paths × 200 events). No ceiling was lowered.
EVENT_DETAIL_BUDGET_V2 = EventDetailBudget(
    budget_id="event_detail_budget_v2",
    max_event_detail_rows=10_000_000,
    max_published_bytes=2_147_483_648,
    path_block_size=250,
    max_rows_per_partition=50_000,
)


class EventDetailBudgetError(PermissionError):
    """A preflight estimate or streaming overrun exceeded the registered
    budget — raised BEFORE any publication; no partial artifact exists."""


class EventDetailIntegrityError(ValueError):
    """The event stream violates uniqueness / total-order invariants."""


class EventDetailUnavailableError(LookupError):
    """The simulation was persisted under ``none_v0`` (evidence not persisted)."""


#: Descriptive same-instant precedence of the registered event types. The
#: TOTAL order is ``(path_ordinal, event_ordinal)`` — the walk's one strictly
#: ordered stream; the precedence rank is a reporting descriptor only.
EVENT_TYPE_PRECEDENCE: MappingProxyType[str, int] = MappingProxyType(
    {
        "fee": 0,
        "equity_update": 1,
        "threshold_ratchet": 2,
        "daily_halt": 3,
        "phase_transition": 4,
        "payout": 5,
        "replacement": 6,
        "breach": 7,
    }
)

#: The event body field projected into the ``amount`` column (``None`` →
#: the type carries no monetary magnitude; the row's amount is null).
EVENT_AMOUNT_FIELD: MappingProxyType[str, str | None] = MappingProxyType(
    {
        "fee": "amount",
        "payout": "trader_amount",
        "replacement": "reset_fee",
        "equity_update": "realized_delta",
        "breach": "observed_equity",
        "threshold_ratchet": "new_floor",
        "phase_transition": None,
        "daily_halt": None,
    }
)

EVENT_DETAIL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("path_instance_id", pa.string(), nullable=False),
        pa.field("path_ordinal", pa.int64(), nullable=False),
        pa.field("path_block_id", pa.int64(), nullable=False),
        pa.field("account_id", pa.string(), nullable=False),
        pa.field("account_ordinal", pa.int64(), nullable=False),
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_ts_utc", pa.string(), nullable=False),
        pa.field("event_ts_ns", pa.int64(), nullable=False),
        pa.field("trading_day", pa.string(), nullable=True),
        pa.field("clock_policy_id", pa.string(), nullable=False),
        pa.field("event_precedence", pa.int32(), nullable=False),
        pa.field("event_ordinal", pa.int64(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("account_phase", pa.string(), nullable=False),
        pa.field("source_trade_id", pa.string(), nullable=True),
        pa.field("source_candidate_id", pa.string(), nullable=True),
        pa.field("amount", pa.float64(), nullable=True),
    ]
)
EVENT_DETAIL_SCHEMA_HASH = canonical_sha256(
    {
        "schema_version": EVENT_DETAIL_SCHEMA_VERSION_V2,
        "fields": [
            [field.name, str(field.type), bool(field.nullable)] for field in EVENT_DETAIL_SCHEMA
        ],
    }
)


def assert_event_detail_policy_coherent(
    persistence_policy_id: str,
    storage_policy_id: str,
    schema_version: int,
    budget: EventDetailBudget | None,
) -> None:
    """``none_v0`` ⇔ no storage / schema 0 / no budget; v2 ⇔ ZSTD Parquet
    path blocks / schema 2 / a registered budget — anything else refuses."""

    if persistence_policy_id == EVENT_DETAIL_POLICY_NONE:
        if (
            storage_policy_id != EVENT_DETAIL_STORAGE_NONE
            or schema_version != EVENT_DETAIL_SCHEMA_VERSION_NONE
            or budget is not None
        ):
            raise ValueError(
                "event_detail_persistence_policy_id none_v0 carries no storage policy, "
                "schema version 0, and no budget"
            )
        return
    if persistence_policy_id == EVENT_DETAIL_POLICY_PARQUET_V2:
        if storage_policy_id != EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2:
            raise ValueError(
                f"{EVENT_DETAIL_POLICY_PARQUET_V2} requires the storage policy "
                f"{EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2}"
            )
        if schema_version != EVENT_DETAIL_SCHEMA_VERSION_V2:
            raise ValueError(
                f"{EVENT_DETAIL_POLICY_PARQUET_V2} requires event_detail_schema_version "
                f"{EVENT_DETAIL_SCHEMA_VERSION_V2}"
            )
        if budget is None:
            raise ValueError(f"{EVENT_DETAIL_POLICY_PARQUET_V2} requires a registered budget")
        return
    raise ValueError(
        f"unregistered event_detail_persistence_policy_id {persistence_policy_id!r}; "
        f"registered: {EVENT_DETAIL_PERSISTENCE_POLICIES}"
    )


def event_detail_identity_fields(
    persistence_policy_id: str, *, budget: EventDetailBudget = EVENT_DETAIL_BUDGET_V2
) -> dict[str, Any]:
    """The coherent identity tuple for one persistence policy (payload kwargs)."""

    if persistence_policy_id == EVENT_DETAIL_POLICY_NONE:
        return {
            "event_detail_persistence_policy_id": EVENT_DETAIL_POLICY_NONE,
            "event_detail_storage_policy_id": EVENT_DETAIL_STORAGE_NONE,
            "event_detail_schema_version": EVENT_DETAIL_SCHEMA_VERSION_NONE,
            "event_detail_budget": None,
        }
    if persistence_policy_id == EVENT_DETAIL_POLICY_PARQUET_V2:
        return {
            "event_detail_persistence_policy_id": EVENT_DETAIL_POLICY_PARQUET_V2,
            "event_detail_storage_policy_id": EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2,
            "event_detail_schema_version": EVENT_DETAIL_SCHEMA_VERSION_V2,
            "event_detail_budget": budget,
        }
    raise ValueError(
        f"unregistered event_detail_persistence_policy_id {persistence_policy_id!r}; "
        f"registered: {EVENT_DETAIL_PERSISTENCE_POLICIES}"
    )


def partition_sidecar_name(path_block_id: int, partition_ordinal: int | None = 0) -> str:
    """The deterministic partition file name keyed by ``(path_block_id,
    partition_ordinal_within_path_block)`` (HARDENING-BACKEND-FIX §9);
    ``partition_ordinal=None`` is the pre-bound single-partition-per-block
    form kept only to READ artifacts persisted under it."""

    stem = f"{EVENT_DETAIL_PARTITION_PREFIX}{int(path_block_id):06d}"
    if partition_ordinal is None:
        return f"{stem}{_PARTITION_SUFFIX}"
    return f"{stem}_{int(partition_ordinal):03d}{_PARTITION_SUFFIX}"


@dataclass(frozen=True)
class EventDetailBundle:
    """The sidecars one v2 publication carries, already written under
    ``directory`` (the store's temporary publication directory): the
    partitions in ``path_block_id`` order plus the detail manifest — each as
    the (name, sha256, bytes) record the store re-verifies."""

    directory: Path
    partitions: tuple[ProducedSidecar, ...]
    manifest_sidecar: ProducedSidecar
    manifest: dict[str, Any]
    total_rows: int
    total_bytes: int
    partition_count: int
    #: HARDENING-BACKEND-FIX §9: the largest partition written — the proof of the
    #: resident-row bound (never above ``budget.max_rows_per_partition``)
    max_partition_rows: int = 0

    def produced(self) -> tuple[ProducedSidecar, ...]:
        """Every sidecar record the store must list (partitions + manifest)."""

        return (*self.partitions, self.manifest_sidecar)


def _event_ts_ns(value: str) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise EventDetailIntegrityError(f"event timestamp {value!r} carries no UTC offset")
    return int(stamp.tz_convert("UTC").value)


def _amount(event) -> float | None:
    field = EVENT_AMOUNT_FIELD.get(event.event_type)
    if field is None:
        return None
    value = getattr(event.payload, field, None)
    return None if value is None else float(value)


def _empty_columns() -> dict[str, list[Any]]:
    return {field.name: [] for field in EVENT_DETAIL_SCHEMA}


def _append_event(
    columns: dict[str, list[Any]],
    event,
    *,
    record,
    path_id: str,
    path_ordinal: int,
    block_id: int,
    previous_ordinal: int,
    clock_policy_id: str,
) -> int:
    """Append ONE event of one walk to the current partition's COLUMNS (no
    per-row dict is ever built); ordinals must be strictly increasing on the
    path. Returns the event ordinal (the next call's ``previous_ordinal``)."""

    if event.path_instance_id != record.path_instance_id:
        raise EventDetailIntegrityError(
            "an account event names a different path than its walk record"
        )
    ordinal = int(event.event_ordinal)
    if ordinal <= previous_ordinal:
        raise EventDetailIntegrityError(
            f"event ordinals are not strictly increasing on path "
            f"{record.path_instance_id} ({event.event_ordinal} after {previous_ordinal})"
        )
    precedence = EVENT_TYPE_PRECEDENCE.get(event.event_type)
    if precedence is None:
        raise EventDetailIntegrityError(f"unregistered event type {event.event_type!r}")
    stamp = str(event.event_ts_utc)
    columns["path_instance_id"].append(path_id)
    columns["path_ordinal"].append(int(path_ordinal))
    columns["path_block_id"].append(int(block_id))
    columns["account_id"].append(str(event.account_id))
    columns["account_ordinal"].append(int(event.account_ordinal))
    columns["event_id"].append(str(event.event_id))
    columns["event_ts_utc"].append(stamp)
    columns["event_ts_ns"].append(_event_ts_ns(stamp))
    columns["trading_day"].append(getattr(event, "trading_day", None))
    columns["clock_policy_id"].append(str(clock_policy_id))
    columns["event_precedence"].append(int(precedence))
    columns["event_ordinal"].append(ordinal)
    columns["event_type"].append(str(event.event_type))
    columns["account_phase"].append(str(event.account_phase.value))
    columns["source_trade_id"].append(event.source_trade_id)
    columns["source_candidate_id"].append(event.source_candidate_id)
    columns["amount"].append(_amount(event))
    return ordinal


def _first_duplicate(event_ids: Sequence[str]) -> str | None:
    """The bounded WITHIN-BLOCK duplicate check (one path block of ids)."""

    seen: set[str] = set()
    for event_id in event_ids:
        if event_id in seen:
            return event_id
        seen.add(event_id)
    return None


def _write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, compression="zstd", row_group_size=EVENT_DETAIL_ROW_GROUP_SIZE)


def _schema_hash_of(table_schema: pa.Schema) -> str:
    return canonical_sha256(
        {
            "schema_version": EVENT_DETAIL_SCHEMA_VERSION_V2,
            "fields": [
                [field.name, str(field.type), bool(field.nullable)] for field in table_schema
            ],
        }
    )


def _sql_path(path: Path) -> str:
    return str(Path(path).as_posix()).replace("'", "''")


def _external_uniqueness_check(
    directory: Path,
    partition_names: Sequence[str],
    *,
    expected_rows: int,
    expected_paths: int,
) -> dict[str, Any]:
    """The disk-backed artifact-level proof (§4.4): DuckDB reads the WRITTEN
    partitions under an explicit memory limit with an attempt-local spill
    directory and proves ``event_id`` distinct and ``path_instance_id``
    one-to-one with ``path_ordinal``. Never an in-memory whole-artifact set."""

    facts: dict[str, Any] = {
        "canonical_key": list(EVENT_ID_CANONICAL_KEY),
        "artifact_check": EVENT_ID_UNIQUENESS_CHECK_V1,
        "memory_limit_bytes": int(EXTERNAL_CHECK_MEMORY_LIMIT_BYTES),
        "partitions_checked": int(len(partition_names)),
        "rows_checked": int(expected_rows),
        "distinct_event_ids": int(expected_rows),
        "distinct_paths": int(expected_paths),
    }
    if not partition_names:
        return facts
    import duckdb  # noqa: PLC0415 — the engine is loaded only for the check

    temp_root = tempfile.mkdtemp(prefix="ifvg_event_detail_uniqueness_")
    try:
        connection = duckdb.connect(database=":memory:")
        try:
            connection.execute(
                f"SET memory_limit='{EXTERNAL_CHECK_MEMORY_LIMIT_BYTES // (1024 * 1024)}MiB'"
            )
            connection.execute(f"SET temp_directory='{_sql_path(Path(temp_root))}'")
            connection.execute(f"SET threads={_EXTERNAL_CHECK_THREADS}")
            files = ", ".join(f"'{_sql_path(Path(directory) / name)}'" for name in partition_names)
            source = f"read_parquet([{files}])"
            rows, ids, paths, ordinals = connection.execute(
                "SELECT COUNT(*), COUNT(DISTINCT event_id), COUNT(DISTINCT path_instance_id), "
                f"COUNT(DISTINCT path_ordinal) FROM {source}"
            ).fetchone()
            if int(rows) != int(expected_rows):
                raise EventDetailIntegrityError(
                    "the written partitions do not carry the streamed row count "
                    f"({rows} != {expected_rows})"
                )
            if int(ids) != int(rows):
                example = connection.execute(
                    f"SELECT event_id FROM {source} GROUP BY event_id HAVING COUNT(*) > 1 "
                    "ORDER BY event_id LIMIT 1"
                ).fetchone()
                sample = str(example[0])[:12] if example else "?"
                raise EventDetailIntegrityError(
                    f"duplicate event id across path blocks ({sample}…)"
                )
            if int(paths) != int(expected_paths) or int(ordinals) != int(expected_paths):
                raise EventDetailIntegrityError(
                    "path instance ids are not one-to-one with path ordinals across path "
                    f"blocks (paths={paths}, ordinals={ordinals}, declared={expected_paths})"
                )
            facts["distinct_event_ids"] = int(ids)
            facts["distinct_paths"] = int(paths)
        finally:
            connection.close()
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return facts


def _pair_stream(
    walks: Any,
    path_records: Sequence[Any] | None,
    *,
    total_rows: int | None,
    path_count: int | None,
) -> tuple[Iterator[tuple[Any, Any]], int, int]:
    """Normalize the two call shapes to ``(pairs in draw-ordinal order,
    declared_rows, declared_paths)``. The legacy sequence form derives the
    preflight from the sequences it holds; the iterable form REQUIRES the
    caller's declared counts (verified exactly after streaming)."""

    if path_records is not None:
        walk_results = walks if isinstance(walks, list | tuple) else tuple(walks)
        if len(walk_results) != len(path_records):
            raise EventDetailIntegrityError("walk results and path records disagree in length")
        derived_rows = sum(len(result.events) for result in walk_results)
        ordinals = [int(record.draw_ordinal) for record in path_records]
        if len(set(ordinals)) != len(ordinals):
            raise EventDetailIntegrityError("path records repeat a draw ordinal")
        path_ids = [str(record.path_instance_id) for record in path_records]
        if len(set(path_ids)) != len(path_ids):
            raise EventDetailIntegrityError("path records repeat a path instance id")
        if total_rows is not None and int(total_rows) != derived_rows:
            raise EventDetailIntegrityError(
                f"declared total_rows {total_rows} disagrees with the walk results ({derived_rows})"
            )
        if path_count is not None and int(path_count) != len(path_records):
            raise EventDetailIntegrityError(
                f"declared path_count {path_count} disagrees with the path records "
                f"({len(path_records)})"
            )
        ordered = sorted(
            zip(path_records, walk_results, strict=True),
            key=lambda pair: int(pair[0].draw_ordinal),
        )
        return iter(ordered), int(derived_rows), int(len(path_records))
    if total_rows is None or path_count is None:
        raise ValueError(
            "an iterable of (path_record, walk_result) pairs requires the declared "
            "total_rows and path_count preflight counts"
        )
    if isinstance(walks, str | bytes):
        raise ValueError("walks must be an iterable of (path_record, walk_result) pairs")
    if int(total_rows) < 0 or int(path_count) < 0:
        raise ValueError("total_rows and path_count cannot be negative")
    return iter(walks), int(total_rows), int(path_count)


def build_account_event_detail(
    walks: Iterable[tuple[Any, Any]] | Sequence[Any],
    path_records: Sequence[Any] | None = None,
    *,
    clock_policy_id: str,
    budget: EventDetailBudget,
    event_order_policy_id: str,
    directory: Path,
    total_rows: int | None = None,
    path_count: int | None = None,
) -> EventDetailBundle:
    """Stream the run's events into ZSTD Parquet path blocks under ``directory``
    within the budget (``directory`` is the store's temporary publication
    directory — a caller that provides its own must discard it on failure).

    Two call shapes: the legacy ``(walk_results, path_records)`` sequences,
    or ``walks`` = an ITERABLE of ``(path_record, walk_result)`` pairs in
    strictly increasing draw-ordinal order with the declared ``total_rows``
    and ``path_count`` (consumed exactly once; never materialized).

    Preflight: the declared row count must fit ``max_event_detail_rows``.
    Streaming: partitions are produced one path block at a time in
    ``path_block_id`` order — built column-wise, written to disk, hashed by
    streaming, released — the cumulative rows must fit the row budget and the
    cumulative written bytes must fit ``max_published_bytes``; the first
    overrun raises and nothing is published (the store discards the
    directory). The streamed counts must equal the declared preflight.
    Uniqueness of ``event_id`` and of ``(path_instance_id, event_ordinal)``:
    unique path ids × strictly increasing ordinals per path (streamed) plus
    the disk-backed external distinct check over the written partitions
    (never a whole-artifact in-memory index); the total order is
    (path_ordinal, event_ordinal).
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("build_account_event_detail needs an existing publication directory")
    stream, declared_rows, declared_paths = _pair_stream(
        walks, path_records, total_rows=total_rows, path_count=path_count
    )
    if declared_rows > budget.max_event_detail_rows:
        raise EventDetailBudgetError(
            f"event detail preflight: {declared_rows} rows exceed the registered budget of "
            f"{budget.max_event_detail_rows} ({budget.budget_id}); refusing before publication"
        )
    if budget.max_rows_per_partition is None:
        raise EventDetailBudgetError(
            "the row-bounded event-detail writer requires a budget that registers "
            f"max_rows_per_partition; {budget.budget_id} carries none (a pre-bound budget is "
            f"loadable, never writable — register {EVENT_DETAIL_BUDGET_V2.budget_id})"
        )
    block_size = int(budget.path_block_size)
    rows_per_partition = int(budget.max_rows_per_partition)
    partitions: list[dict[str, Any]] = []
    records: list[ProducedSidecar] = []
    columns = _empty_columns()
    written: list[Path] = []
    state: dict[str, Any] = {
        "cumulative": 0,
        "partition_ordinal": 0,
        "max_partition_rows": 0,
        "block": None,
    }

    def _flush() -> None:
        """Write the resident rows as ONE partition keyed by (path block,
        partition ordinal within the block) — at most ``rows_per_partition``
        rows, whatever the events-per-path shape."""

        rows = len(columns["event_id"])
        block_id = state["block"]
        if rows == 0 or block_id is None:
            return
        if rows > rows_per_partition:  # pragma: no cover - the loop flushes AT the bound
            raise EventDetailIntegrityError(
                "resident rows exceeded the registered per-partition bound"
            )
        duplicate = _first_duplicate(columns["event_id"])
        if duplicate is not None:
            raise EventDetailIntegrityError(f"duplicate event id {duplicate[:12]}…")
        table = pa.Table.from_pydict(columns, schema=EVENT_DETAIL_SCHEMA)
        ordinal = int(state["partition_ordinal"])
        name = partition_sidecar_name(block_id, ordinal)
        path = directory / name
        if path.exists():
            raise EventDetailIntegrityError(f"partition {name} already exists in the directory")
        # review RB-03: registered BEFORE the write — a partial file is cleaned too
        written.append(path)
        _write_parquet(table, path)
        size = int(path.stat().st_size)
        state["cumulative"] = int(state["cumulative"]) + size
        if state["cumulative"] > budget.max_published_bytes:
            raise EventDetailBudgetError(
                f"event detail streaming overrun: {state['cumulative']} bytes exceed the "
                f"registered budget of {budget.max_published_bytes} ({budget.budget_id}) at "
                f"path block {block_id}; refusing before publication"
            )
        digest = file_sha256(path)
        partitions.append(
            {
                "name": name,
                "path_block_id": int(block_id),
                "partition_ordinal": ordinal,
                "path_ordinal_min": int(min(columns["path_ordinal"])),
                "path_ordinal_max": int(max(columns["path_ordinal"])),
                "first_event_key": [
                    int(columns["path_ordinal"][0]),
                    int(columns["event_ordinal"][0]),
                ],
                "last_event_key": [
                    int(columns["path_ordinal"][-1]),
                    int(columns["event_ordinal"][-1]),
                ],
                "rows": int(rows),
                "bytes": size,
                "sha256": digest,
                "schema_hash": _schema_hash_of(table.schema),
            }
        )
        records.append(ProducedSidecar(name=name, sha256=digest, bytes=size))
        state["max_partition_rows"] = max(int(state["max_partition_rows"]), rows)
        state["partition_ordinal"] = ordinal + 1
        for values in columns.values():
            values.clear()

    rows_seen = 0
    paths_seen = 0
    last_ordinal: int | None = None
    try:
        for record, result in stream:
            ordinal = int(record.draw_ordinal)
            if last_ordinal is not None and ordinal <= last_ordinal:
                raise EventDetailIntegrityError(
                    "walk pairs are not in strictly increasing draw-ordinal order "
                    f"({ordinal} after {last_ordinal})"
                )
            last_ordinal = ordinal
            paths_seen += 1
            if paths_seen > declared_paths:
                raise EventDetailIntegrityError(
                    f"declared path_count {declared_paths} exceeded while streaming"
                )
            block_id = ordinal // block_size
            if state["block"] is not None and block_id != state["block"]:
                _flush()  # the remainder of the previous path block
                state["partition_ordinal"] = 0
            state["block"] = block_id
            path_id = str(record.path_instance_id)
            previous_event_ordinal = -1
            for event in result.events:
                previous_event_ordinal = _append_event(
                    columns,
                    event,
                    record=record,
                    path_id=path_id,
                    path_ordinal=ordinal,
                    block_id=block_id,
                    previous_ordinal=previous_event_ordinal,
                    clock_policy_id=clock_policy_id,
                )
                rows_seen += 1
                if rows_seen > budget.max_event_detail_rows:
                    raise EventDetailBudgetError(
                        f"event detail streaming row overrun: {rows_seen} rows exceed the "
                        f"registered budget of {budget.max_event_detail_rows} "
                        f"({budget.budget_id}) at path block {block_id}; refusing before "
                        "publication"
                    )
                if len(columns["event_id"]) >= rows_per_partition:
                    _flush()  # §9: flush AT the bound — even inside one path
        _flush()
        if rows_seen != declared_rows:
            raise EventDetailIntegrityError(
                f"declared total_rows {declared_rows} disagrees with the streamed rows "
                f"({rows_seen})"
            )
        if paths_seen != declared_paths:
            raise EventDetailIntegrityError(
                f"declared path_count {declared_paths} disagrees with the streamed paths "
                f"({paths_seen})"
            )
        uniqueness = _external_uniqueness_check(
            directory,
            [entry["name"] for entry in partitions],
            expected_rows=rows_seen,
            expected_paths=paths_seen,
        )
        cumulative = int(state["cumulative"])
        manifest = {
            "event_detail_persistence_policy_id": EVENT_DETAIL_POLICY_PARQUET_V2,
            "event_detail_storage_policy_id": EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2,
            "event_detail_schema_version": EVENT_DETAIL_SCHEMA_VERSION_V2,
            "schema_hash": EVENT_DETAIL_SCHEMA_HASH,
            "compression": "zstd",
            "row_group_size": EVENT_DETAIL_ROW_GROUP_SIZE,
            "budget": budget.model_dump(mode="json"),
            "partition_bound_policy_id": EVENT_DETAIL_PARTITION_BOUND_POLICY_V2,
            "partition_key": ["path_block_id", "partition_ordinal"],
            "max_rows_per_partition": rows_per_partition,
            "max_partition_rows_written": int(state["max_partition_rows"]),
            "clock_policy_id": clock_policy_id,
            "event_order_policy_id": event_order_policy_id,
            "total_order": ["path_ordinal", "event_ordinal"],
            "event_type_precedence": dict(EVENT_TYPE_PRECEDENCE),
            "amount_field_by_event_type": dict(EVENT_AMOUNT_FIELD),
            "event_id_uniqueness": uniqueness,
            "path_count": int(paths_seen),
            "total_rows": int(rows_seen),
            "total_bytes": cumulative,
            "partitions": partitions,
        }
        manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
        manifest_path = directory / EVENT_DETAIL_MANIFEST_SIDECAR
        if manifest_path.exists():
            raise EventDetailIntegrityError("the detail manifest already exists in the directory")
        written.append(manifest_path)  # a partial manifest is cleaned too (review RB-03)
        manifest_path.write_bytes(manifest_bytes)
    except BaseException:
        # §9: a refused build leaves no partition (and no manifest) behind —
        # review RB-03: the manifest stage runs INSIDE this guard and a partially
        # written file is registered before its write — the store discards
        # its temporary directory as well; a caller-owned directory is clean
        for path in written:
            with contextlib.suppress(OSError):
                path.unlink()
        raise
    manifest_sidecar = ProducedSidecar(
        name=EVENT_DETAIL_MANIFEST_SIDECAR,
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        bytes=len(manifest_bytes),
    )
    return EventDetailBundle(
        directory=directory,
        partitions=tuple(records),
        manifest_sidecar=manifest_sidecar,
        manifest=manifest,
        total_rows=int(rows_seen),
        total_bytes=cumulative,
        partition_count=len(partitions),
        max_partition_rows=int(state["max_partition_rows"]),
    )


def account_event_detail_producer(
    walk_results: Sequence[Any],
    path_records: Sequence[Any],
    *,
    clock_policy_id: str,
    budget: EventDetailBudget,
    event_order_policy_id: str,
    on_built: Callable[[EventDetailBundle], None] | None = None,
) -> SidecarProducer:
    """The store sidecar producer for one simulation's v2 detail: it streams
    the partitions + manifest into the directory the store hands it and
    returns their records; ``on_built`` sees the bundle (counts) before the
    store lists it — the search bridge writes ``walk_summary.json`` there."""

    def _produce(directory: Path) -> tuple[ProducedSidecar, ...]:
        bundle = build_account_event_detail(
            walk_results,
            path_records,
            clock_policy_id=clock_policy_id,
            budget=budget,
            event_order_policy_id=event_order_policy_id,
            directory=directory,
        )
        if on_built is not None:
            on_built(bundle)
        return bundle.produced()

    return _produce


def load_account_event_detail_manifest(root: Path, account_simulation_id: str) -> dict[str, Any]:
    """The verified detail manifest of one v2 account simulation."""

    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        load_sidecar_bytes,
        load_verified_envelope,
    )
    from alpha_lab.propsim.simulation import AccountSimulationEnvelope  # noqa: PLC0415

    root = Path(root)
    envelope = load_verified_envelope(
        root, _STORE, account_simulation_id, AccountSimulationEnvelope
    )
    payload = envelope.payload
    if payload.event_detail_persistence_policy_id != EVENT_DETAIL_POLICY_PARQUET_V2:
        raise EventDetailUnavailableError(
            f"account simulation {account_simulation_id[:12]}… was persisted under "
            f"{payload.event_detail_persistence_policy_id}: event detail evidence_not_persisted"
        )
    manifest = json.loads(
        load_sidecar_bytes(
            root, _STORE, account_simulation_id, EVENT_DETAIL_MANIFEST_SIDECAR
        ).decode("utf-8")
    )
    if (
        manifest.get("event_detail_persistence_policy_id") != EVENT_DETAIL_POLICY_PARQUET_V2
        or manifest.get("event_detail_storage_policy_id")
        != payload.event_detail_storage_policy_id
        or int(manifest.get("event_detail_schema_version", -1))
        != int(payload.event_detail_schema_version)
        or manifest.get("schema_hash") != EVENT_DETAIL_SCHEMA_HASH
    ):
        raise EventDetailIntegrityError(
            "the event detail manifest disagrees with the simulation identity"
        )
    if payload.event_detail_budget is None or manifest.get("budget") != (
        payload.event_detail_budget.model_dump(mode="json")
    ):
        raise EventDetailIntegrityError(
            "the event detail manifest budget disagrees with the simulation identity"
        )
    return manifest


def load_account_event_detail(
    root: Path, account_simulation_id: str
) -> Iterator[pd.DataFrame]:
    """Yield the VERIFIED partitions of one v2 account simulation in
    ``path_block_id`` order (each frame in ``(path_ordinal, event_ordinal)``
    order). Every partition's bytes, row count, and schema are checked
    against the detail manifest (itself behind the store manifest) before
    it is yielded; a ``none_v0`` simulation raises
    :class:`EventDetailUnavailableError`.
    """

    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        load_sidecar_bytes,
    )

    root = Path(root)
    manifest = load_account_event_detail_manifest(root, account_simulation_id)
    def _key(entry: dict) -> tuple[int, int]:
        ordinal = entry.get("partition_ordinal")
        return int(entry["path_block_id"]), (0 if ordinal is None else int(ordinal))

    partitions = sorted(manifest.get("partitions", ()), key=_key)
    if len({_key(entry) for entry in partitions}) != len(partitions):
        raise EventDetailIntegrityError("the event detail manifest repeats a partition key")
    # review RB-04: the bound the reader proves is the IDENTITY-bound budget's (the
    # manifest budget was proven equal to the simulation payload's); the manifest's
    # loose top-level field must agree with it
    budget_bound = (manifest.get("budget") or {}).get("max_rows_per_partition")
    bound = manifest.get("max_rows_per_partition")
    if bound != budget_bound:
        raise EventDetailIntegrityError(
            "the detail manifest's per-partition row bound disagrees with the identity-bound "
            "event-detail budget"
        )
    previous_last: tuple[int, int] | None = None
    rows_seen = 0
    for entry in partitions:
        name = str(entry["name"])
        ordinal = entry.get("partition_ordinal")
        expected_name = partition_sidecar_name(
            int(entry["path_block_id"]), None if ordinal is None else int(ordinal)
        )
        if name != expected_name:
            raise EventDetailIntegrityError(f"partition name {name!r} is not the registered form")
        data = load_sidecar_bytes(root, _STORE, account_simulation_id, name)
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != int(entry["bytes"]):
            raise EventDetailIntegrityError(f"partition {name} does not match the detail manifest")
        table = pq.read_table(io.BytesIO(data))
        if _schema_hash_of(table.schema) != entry["schema_hash"] or (
            entry["schema_hash"] != EVENT_DETAIL_SCHEMA_HASH
        ):
            raise EventDetailIntegrityError(f"partition {name} schema disagrees with the contract")
        if table.num_rows != int(entry["rows"]):
            raise EventDetailIntegrityError(
                f"partition {name} row count disagrees with the manifest"
            )
        if bound is not None and table.num_rows > int(bound):
            raise EventDetailIntegrityError(
                f"partition {name} exceeds the registered per-partition row bound"
            )
        frame = table.to_pandas()
        if (frame["path_block_id"] != int(entry["path_block_id"])).any():
            raise EventDetailIntegrityError(f"partition {name} carries rows of another path block")
        ordered = frame.sort_values(["path_ordinal", "event_ordinal"], kind="mergesort")
        if not ordered.index.equals(frame.index):
            raise EventDetailIntegrityError(f"partition {name} is not in total order")
        if len(frame):
            first = (int(frame["path_ordinal"].iloc[0]), int(frame["event_ordinal"].iloc[0]))
            last = (int(frame["path_ordinal"].iloc[-1]), int(frame["event_ordinal"].iloc[-1]))
            if "first_event_key" in entry and (
                list(first) != list(entry["first_event_key"])
                or list(last) != list(entry.get("last_event_key", ()))
            ):
                raise EventDetailIntegrityError(
                    f"partition {name} boundary keys disagree with the detail manifest"
                )
            if previous_last is not None and first <= previous_last:
                raise EventDetailIntegrityError(
                    f"partition {name} breaks the total order across partitions"
                )
            previous_last = last
        rows_seen += table.num_rows
        yield frame.reset_index(drop=True)
    if rows_seen != int(manifest.get("total_rows", -1)):
        raise EventDetailIntegrityError("the partitions do not add up to the manifest row total")
