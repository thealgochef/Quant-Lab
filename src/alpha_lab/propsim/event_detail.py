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
  type / phase / amount. A preflight row count or a streaming byte overrun
  fails BEFORE atomic publication, so no partial artifact ever exists.
  The writer is a store *sidecar producer* (R6.1 safety review S12): each
  partition is built column-wise for ONE path block, written straight into
  the store's temporary publication directory, hashed by streaming and
  released — the process never holds more than one block of rows plus a
  32-byte-per-row event-id uniqueness index; the temporary directory is
  discarded on any refusal.
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

import hashlib
import io
import json
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

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
    "EVENT_TYPE_PRECEDENCE",
    "EVENT_AMOUNT_FIELD",
    "EventDetailBudget",
    "EVENT_DETAIL_BUDGET_V1",
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
_PARTITION_SUFFIX = ".parquet"
_STORE = "account_simulations"


class EventDetailBudget(FrozenContract):
    """Registered storage budget — part of the simulation identity."""

    budget_id: str = Field(min_length=1)
    max_event_detail_rows: int = Field(ge=1)
    max_published_bytes: int = Field(ge=1)
    path_block_size: int = Field(ge=1)


#: The registered V1 limits (plan §6.G / D15).
EVENT_DETAIL_BUDGET_V1 = EventDetailBudget(
    budget_id="event_detail_budget_v1",
    max_event_detail_rows=10_000_000,
    max_published_bytes=2_147_483_648,
    path_block_size=250,
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
    persistence_policy_id: str, *, budget: EventDetailBudget = EVENT_DETAIL_BUDGET_V1
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


def partition_sidecar_name(path_block_id: int) -> str:
    return f"{EVENT_DETAIL_PARTITION_PREFIX}{int(path_block_id):06d}{_PARTITION_SUFFIX}"


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


def _append_path(
    columns: dict[str, list[Any]], record, result, *, clock_policy_id: str, block_size: int
) -> None:
    """Append one walk's events to the current block's COLUMNS (no per-row
    dict is ever built); ordinals must be strictly increasing on the path."""

    path_id = str(record.path_instance_id)
    path_ordinal = int(record.draw_ordinal)
    block_id = path_ordinal // int(block_size)
    previous_ordinal = -1
    for event in result.events:
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
        previous_ordinal = ordinal
        precedence = EVENT_TYPE_PRECEDENCE.get(event.event_type)
        if precedence is None:
            raise EventDetailIntegrityError(f"unregistered event type {event.event_type!r}")
        stamp = str(event.event_ts_utc)
        columns["path_instance_id"].append(path_id)
        columns["path_ordinal"].append(path_ordinal)
        columns["path_block_id"].append(block_id)
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


def _event_id_index(event_ids: Sequence[str]) -> np.ndarray:
    """The exact 32-byte uniqueness index of a block's event ids (the raw
    value of a 64-hex id; the sha256 digest of any other form)."""

    values = []
    for event_id in event_ids:
        try:
            raw = bytes.fromhex(event_id) if len(event_id) == 64 else b""
        except ValueError:
            raw = b""
        values.append(raw if len(raw) == 32 else hashlib.sha256(event_id.encode()).digest())
    return np.array(values, dtype="S32")


def _assert_unique(index: np.ndarray, *, scope: str) -> None:
    if index.size and np.unique(index).size != index.size:
        raise EventDetailIntegrityError(f"duplicate event id {scope}")


def _first_duplicate(event_ids: Sequence[str]) -> str | None:
    seen: set[str] = set()
    for event_id in event_ids:
        if event_id in seen:
            return event_id
        seen.add(event_id)
    return None


def _write_parquet(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path, compression="zstd", row_group_size=65_536)


def _schema_hash_of(table_schema: pa.Schema) -> str:
    return canonical_sha256(
        {
            "schema_version": EVENT_DETAIL_SCHEMA_VERSION_V2,
            "fields": [
                [field.name, str(field.type), bool(field.nullable)] for field in table_schema
            ],
        }
    )


def build_account_event_detail(
    walk_results: Sequence[Any],
    path_records: Sequence[Any],
    *,
    clock_policy_id: str,
    budget: EventDetailBudget,
    event_order_policy_id: str,
    directory: Path,
) -> EventDetailBundle:
    """Stream the run's events into ZSTD Parquet path blocks under ``directory``
    within the budget (``directory`` is the store's temporary publication
    directory — a caller that provides its own must discard it on failure).

    Preflight: the exact row count must fit ``max_event_detail_rows``.
    Streaming: partitions are produced one path block at a time in
    ``path_block_id`` order — built column-wise, written to disk, hashed by
    streaming, released — and the cumulative written bytes must fit
    ``max_published_bytes``; the first overrun raises and nothing is
    published (the store discards the directory). Uniqueness of ``event_id``
    (an exact 32-byte-per-row index, the only whole-artifact state) and of
    ``(path_instance_id, event_ordinal)`` (unique path ids × strictly
    increasing ordinals per path) is enforced; the total order is
    (path_ordinal, event_ordinal).
    """

    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("build_account_event_detail needs an existing publication directory")
    if len(walk_results) != len(path_records):
        raise EventDetailIntegrityError("walk results and path records disagree in length")
    total_rows = sum(len(result.events) for result in walk_results)
    if total_rows > budget.max_event_detail_rows:
        raise EventDetailBudgetError(
            f"event detail preflight: {total_rows} rows exceed the registered budget of "
            f"{budget.max_event_detail_rows} ({budget.budget_id}); refusing before publication"
        )
    ordinals = [int(record.draw_ordinal) for record in path_records]
    if len(set(ordinals)) != len(ordinals):
        raise EventDetailIntegrityError("path records repeat a draw ordinal")
    path_ids = [str(record.path_instance_id) for record in path_records]
    if len(set(path_ids)) != len(path_ids):
        raise EventDetailIntegrityError("path records repeat a path instance id")
    ordered = sorted(
        zip(path_records, walk_results, strict=True), key=lambda pair: int(pair[0].draw_ordinal)
    )
    block_size = int(budget.path_block_size)
    partitions: list[dict[str, Any]] = []
    records: list[ProducedSidecar] = []
    id_indexes: list[np.ndarray] = []
    columns = _empty_columns()
    cumulative = 0
    current_block: int | None = None

    def _flush(block_id: int) -> None:
        nonlocal cumulative
        duplicate = _first_duplicate(columns["event_id"])
        if duplicate is not None:
            raise EventDetailIntegrityError(f"duplicate event id {duplicate[:12]}…")
        id_indexes.append(_event_id_index(columns["event_id"]))
        table = pa.Table.from_pydict(columns, schema=EVENT_DETAIL_SCHEMA)
        name = partition_sidecar_name(block_id)
        path = directory / name
        if path.exists():
            raise EventDetailIntegrityError(f"partition {name} already exists in the directory")
        _write_parquet(table, path)
        size = int(path.stat().st_size)
        cumulative += size
        if cumulative > budget.max_published_bytes:
            raise EventDetailBudgetError(
                f"event detail streaming overrun: {cumulative} bytes exceed the registered "
                f"budget of {budget.max_published_bytes} ({budget.budget_id}) at path block "
                f"{block_id}; refusing before publication"
            )
        digest = file_sha256(path)
        rows = len(columns["event_id"])
        partitions.append(
            {
                "name": name,
                "path_block_id": int(block_id),
                "path_ordinal_min": int(min(columns["path_ordinal"])),
                "path_ordinal_max": int(max(columns["path_ordinal"])),
                "rows": int(rows),
                "bytes": size,
                "sha256": digest,
                "schema_hash": _schema_hash_of(table.schema),
            }
        )
        records.append(ProducedSidecar(name=name, sha256=digest, bytes=size))
        for values in columns.values():
            values.clear()

    for record, result in ordered:
        block_id = int(record.draw_ordinal) // block_size
        if current_block is not None and block_id != current_block and columns["event_id"]:
            _flush(current_block)
        current_block = block_id
        _append_path(
            columns, record, result, clock_policy_id=clock_policy_id, block_size=block_size
        )
    if current_block is not None and columns["event_id"]:
        _flush(current_block)
    if len(id_indexes) > 1:
        _assert_unique(np.concatenate(id_indexes), scope="across path blocks")
    manifest = {
        "event_detail_persistence_policy_id": EVENT_DETAIL_POLICY_PARQUET_V2,
        "event_detail_storage_policy_id": EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2,
        "event_detail_schema_version": EVENT_DETAIL_SCHEMA_VERSION_V2,
        "schema_hash": EVENT_DETAIL_SCHEMA_HASH,
        "compression": "zstd",
        "budget": budget.model_dump(mode="json"),
        "clock_policy_id": clock_policy_id,
        "event_order_policy_id": event_order_policy_id,
        "total_order": ["path_ordinal", "event_ordinal"],
        "event_type_precedence": dict(EVENT_TYPE_PRECEDENCE),
        "amount_field_by_event_type": dict(EVENT_AMOUNT_FIELD),
        "path_count": int(len(ordered)),
        "total_rows": int(total_rows),
        "total_bytes": int(cumulative),
        "partitions": partitions,
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
    manifest_path = directory / EVENT_DETAIL_MANIFEST_SIDECAR
    if manifest_path.exists():
        raise EventDetailIntegrityError("the detail manifest already exists in the directory")
    manifest_path.write_bytes(manifest_bytes)
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
        total_rows=int(total_rows),
        total_bytes=int(cumulative),
        partition_count=len(partitions),
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
    partitions = sorted(
        manifest.get("partitions", ()), key=lambda entry: int(entry["path_block_id"])
    )
    if len({int(entry["path_block_id"]) for entry in partitions}) != len(partitions):
        raise EventDetailIntegrityError("the event detail manifest repeats a path block")
    rows_seen = 0
    for entry in partitions:
        name = str(entry["name"])
        if name != partition_sidecar_name(int(entry["path_block_id"])):
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
        frame = table.to_pandas()
        if (frame["path_block_id"] != int(entry["path_block_id"])).any():
            raise EventDetailIntegrityError(f"partition {name} carries rows of another path block")
        ordered = frame.sort_values(["path_ordinal", "event_ordinal"], kind="mergesort")
        if not ordered.index.equals(frame.index):
            raise EventDetailIntegrityError(f"partition {name} is not in total order")
        rows_seen += table.num_rows
        yield frame.reset_index(drop=True)
    if rows_seen != int(manifest.get("total_rows", -1)):
        raise EventDetailIntegrityError("the partitions do not add up to the manifest row total")
