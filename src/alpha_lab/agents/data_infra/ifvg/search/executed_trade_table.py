"""The immutable executed-trade table artifact (R6.1-FIX §3.7, F-06).

A core replay persists its IDENTITY only (``CoreStrategyReplayPayload``
carries no tables). Costed evaluation and the stratified regime reports
consume the child's executed-trade table, so the table itself is now a
content-addressed store artifact:

* ``EXECUTED_TRADE_TABLE_SCHEMA_V1`` is the EXACT ordered Arrow projection
  those consumers read — the columns ``_validate_and_normalize_executed_trades``
  requires, the envelope identity columns, and the session/setup columns the
  strategy metrics consult — typed the way the real v2 capture types them
  (double ticks, tz-aware timestamps, ``date32`` trading day). It is declared
  here and never inferred from a frame; a frame lacking a projection column
  is refused, extra columns of the wide v2 union are dropped.
* ``ExecutedTradeTablePayload`` derives from the core replay identity, so
  every consumer EXACT-LOADS the table by ``executed_trade_table_id_for(...)``
  without listing the store; the envelope binds the projection bytes
  (``executed_trade_table_sha256``), the raw core table's content hash
  (``source_core_table_hash`` — the value the child neutrality report
  hashes), the row count and the byte size.
* The store's immutability rules apply: identical bytes reuse, different
  bytes under one id fail closed, a tampered sidecar fails closed on load
  and on probe (corrupt is never "absent").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
import pyarrow as pa
from pydantic import Field

from ..contracts import RecordTable
from ..dataset import table_content_hash
from ..features.arrow_tables import arrow_schema_hash, frame_from_arrow_bytes, frame_to_arrow_bytes
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from .store import (
    SidecarLoadError,
    load_sidecar_bytes,
    load_verified_envelope,
    probe_sidecar,
    save_or_reuse_envelope,
)

__all__ = [
    "EXECUTED_TRADE_TABLE_STORE",
    "EXECUTED_TRADE_TABLE_SIDECAR",
    "EXECUTED_TRADE_TABLE_PROJECTION_ID",
    "EXECUTED_TRADE_TABLE_SCHEMA_V1",
    "EXECUTED_TRADE_TABLE_SCHEMA_HASH",
    "ExecutedTradeTablePayload",
    "ExecutedTradeTableEnvelope",
    "VerifiedExecutedTradeTable",
    "project_executed_trades",
    "executed_trade_table_bytes",
    "executed_trade_table_id_for",
    "build_executed_trade_table",
    "save_executed_trade_table",
    "probe_executed_trade_table",
    "load_executed_trade_table",
]

EXECUTED_TRADE_TABLE_STORE = "executed_trade_tables"
EXECUTED_TRADE_TABLE_SIDECAR = "executed_trade.arrow"
EXECUTED_TRADE_TABLE_PROJECTION_ID = "core_executed_trade_exact_v1"

_STRING = pa.large_string()
_TS = pa.timestamp("ns", "UTC")

#: The exact ordered projection (42 columns).
EXECUTED_TRADE_TABLE_SCHEMA_V1: pa.Schema = pa.schema(
    [
        pa.field("record_table", _STRING),
        pa.field("record_schema_version", pa.int64()),
        pa.field("capture_schema_version", pa.int64()),
        pa.field("dataset_schema_version", pa.int64()),
        pa.field("trade_schema_version", pa.int64()),
        pa.field("envelope_strategy_id", _STRING),
        pa.field("envelope_strategy_version", _STRING),
        pa.field("envelope_profile_hash", _STRING),
        pa.field("envelope_profile_name", _STRING),
        pa.field("envelope_qualification_mode", _STRING),
        pa.field("envelope_section_config_hash", _STRING),
        pa.field("envelope_entry_family", _STRING),
        pa.field("envelope_label_family", _STRING),
        pa.field("envelope_entry_session", _STRING),
        pa.field("envelope_anchor_policy", _STRING),
        pa.field("envelope_resolver_policy", _STRING),
        pa.field("envelope_causality_parent", _STRING),
        pa.field("envelope_causality_opposing", _STRING),
        pa.field("envelope_causality_entry", _STRING),
        pa.field("envelope_timeout_policy", _STRING),
        pa.field("envelope_setup_id", _STRING),
        pa.field("setup_id", _STRING),
        pa.field("candidate_id", _STRING),
        pa.field("decision_id", _STRING),
        pa.field("trade_id", _STRING),
        pa.field("status", _STRING),
        pa.field("resolution", _STRING),
        pa.field("trading_day", pa.date32()),
        pa.field("direction", _STRING),
        pa.field("entry_session", _STRING),
        pa.field("entry_ts_utc", _TS),
        pa.field("resolution_ts_utc", _TS),
        pa.field("entry_cursor", _STRING),
        pa.field("resolution_cursor", _STRING),
        pa.field("entry_ticks", pa.float64()),
        pa.field("stop_ticks", pa.float64()),
        pa.field("target_ticks", pa.float64()),
        pa.field("risk_ticks", pa.float64()),
        pa.field("bars_after_entry_to_resolution", pa.float64()),
        pa.field("realized_ticks", pa.float64()),
        pa.field("mfe_ticks", pa.float64()),
        pa.field("mae_ticks", pa.float64()),
    ]
)
EXECUTED_TRADE_TABLE_SCHEMA_HASH = arrow_schema_hash(EXECUTED_TRADE_TABLE_SCHEMA_V1)

_INT_COLUMNS = (
    "record_schema_version",
    "capture_schema_version",
    "dataset_schema_version",
    "trade_schema_version",
)
_FLOAT_COLUMNS = (
    "entry_ticks",
    "stop_ticks",
    "target_ticks",
    "risk_ticks",
    "bars_after_entry_to_resolution",
    "realized_ticks",
    "mfe_ticks",
    "mae_ticks",
)
_TS_COLUMNS = ("entry_ts_utc", "resolution_ts_utc")


class ExecutedTradeTablePayload(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    source_record_schema_version: int = Field(ge=1)
    table_projection_id: Literal["core_executed_trade_exact_v1"] = (
        EXECUTED_TRADE_TABLE_PROJECTION_ID
    )
    executed_trade_arrow_schema_hash: str = Field(pattern=SHA256_PATTERN)
    source_table_name: Literal["executed_trade"] = "executed_trade"


class ResearchExecutedTradeTablePayload(FrozenContract):
    """A projection scoped to one immutable research population.

    Kept separate from the legacy payload so old table identities and readers
    retain their exact serialization.
    """

    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    source_record_schema_version: int = Field(ge=1)
    table_projection_id: Literal["research_scoped_executed_trade_v2"] = (
        "research_scoped_executed_trade_v2"
    )
    executed_trade_arrow_schema_hash: str = Field(pattern=SHA256_PATTERN)
    source_table_name: Literal["executed_trade"] = "executed_trade"
    research_subject_id: str = Field(pattern=SHA256_PATTERN)


class ExecutedTradeTableEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "executed_trade_table_id"

    executed_trade_table_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchExecutedTradeTablePayload | ExecutedTradeTablePayload
    #: post-materialization facts binding the exact projection bytes
    executed_trade_table_sha256: str = Field(pattern=SHA256_PATTERN)
    #: the RAW core table's content hash (``dataset.table_content_hash``) —
    #: the value the child audit-neutrality report hashes
    source_core_table_hash: str = Field(pattern=SHA256_PATTERN)
    row_count: int = Field(ge=0)
    byte_size: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class VerifiedExecutedTradeTable:
    envelope: ExecutedTradeTableEnvelope
    frame: pd.DataFrame
    table_bytes: bytes

    @property
    def executed_trade_table_id(self) -> str:
        return self.envelope.executed_trade_table_id


def project_executed_trades(frame: pd.DataFrame) -> pd.DataFrame:
    """The exact projection of a (raw or already projected) executed-trade
    frame: the 42 declared columns, deterministically typed, rows sorted by
    ``trade_id``. A missing column refuses; no column is inferred or filled."""

    names = list(EXECUTED_TRADE_TABLE_SCHEMA_V1.names)
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"executed-trade frame lacks projection columns: {missing}")
    out = frame.loc[:, names].copy().reset_index(drop=True)
    for column in names:
        if column in _INT_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="raise").astype("int64")
        elif column in _FLOAT_COLUMNS:
            out[column] = pd.to_numeric(out[column], errors="raise").astype("float64")
        elif column in _TS_COLUMNS:
            out[column] = pd.to_datetime(out[column], utc=True, errors="raise")
        elif column == "trading_day":
            out[column] = pd.to_datetime(out[column], errors="raise").dt.date
        else:
            out[column] = out[column].astype(object).where(out[column].notna(), None)
            out[column] = out[column].map(lambda v: None if v is None else str(v))
    out["trade_id"] = out["trade_id"].astype(str)
    if out["trade_id"].duplicated().any():
        raise ValueError("executed-trade frame repeats a trade_id")
    return out.sort_values("trade_id", kind="mergesort").reset_index(drop=True)


def executed_trade_table_bytes(projected: pd.DataFrame) -> bytes:
    return frame_to_arrow_bytes(projected, EXECUTED_TRADE_TABLE_SCHEMA_V1)


def executed_trade_table_id_for(core_replay_id: str, *, record_schema_version: int) -> str:
    """The table's identity DERIVED from the core replay — no listing needed."""

    return canonical_contract_sha256(
        ExecutedTradeTablePayload(
            core_replay_id=str(core_replay_id),
            source_record_schema_version=int(record_schema_version),
            executed_trade_arrow_schema_hash=EXECUTED_TRADE_TABLE_SCHEMA_HASH,
        )
    )


def build_executed_trade_table(
    core_replay_id: str,
    trades: pd.DataFrame,
    *,
    record_schema_version: int,
) -> tuple[ExecutedTradeTableEnvelope, bytes]:
    """Project the child's executed-trade table and mint the envelope; the
    table's own ``record_schema_version`` must equal the declared one."""

    projected = project_executed_trades(trades)
    versions = set(int(v) for v in projected["record_schema_version"].unique())
    if versions and versions != {int(record_schema_version)}:
        raise ValueError(
            f"executed-trade table carries record_schema_version {sorted(versions)}, not the "
            f"declared {record_schema_version}"
        )
    table_bytes = executed_trade_table_bytes(projected)
    payload = ExecutedTradeTablePayload(
        core_replay_id=str(core_replay_id),
        source_record_schema_version=int(record_schema_version),
        executed_trade_arrow_schema_hash=EXECUTED_TRADE_TABLE_SCHEMA_HASH,
    )
    envelope = ExecutedTradeTableEnvelope.from_payload(
        payload,
        executed_trade_table_sha256=hashlib.sha256(table_bytes).hexdigest(),
        source_core_table_hash=table_content_hash(RecordTable.EXECUTED_TRADE, trades),
        row_count=int(len(projected)),
        byte_size=int(len(table_bytes)),
    )
    return envelope, table_bytes


def build_research_executed_trade_table(
    core_replay_id: str,
    raw_trades: pd.DataFrame,
    *,
    record_schema_version: int,
    research_subject_id: str,
    evaluation_dates: tuple[str, ...],
    candidate_ids: tuple[str, ...],
    cutoff_ts_utc: str,
) -> tuple[ExecutedTradeTableEnvelope, bytes]:
    """Exclude initialization/history before the lossy trade projection.

    The raw source hash remains the hash used by Core neutrality; the produced
    table has its own scope-bearing identity and can coexist with the old one.
    """

    masks = research_trade_cohort_masks(
        raw_trades, candidate_ids=candidate_ids, cutoff_ts_utc=cutoff_ts_utc,
    )
    scoped = raw_trades.loc[masks["included"]].copy()
    legacy, table_bytes = build_executed_trade_table(
        core_replay_id, scoped, record_schema_version=record_schema_version
    )
    payload = ResearchExecutedTradeTablePayload(
        core_replay_id=core_replay_id,
        source_record_schema_version=record_schema_version,
        executed_trade_arrow_schema_hash=EXECUTED_TRADE_TABLE_SCHEMA_HASH,
        research_subject_id=research_subject_id,
    )
    return ExecutedTradeTableEnvelope.from_payload(
        payload,
        executed_trade_table_sha256=legacy.executed_trade_table_sha256,
        source_core_table_hash=table_content_hash(RecordTable.EXECUTED_TRADE, raw_trades),
        row_count=legacy.row_count,
        byte_size=legacy.byte_size,
    ), table_bytes


def research_trade_cohort_masks(raw_trades, *, candidate_ids, cutoff_ts_utc):
    """Exact entry-candidate cohort; Core's resolution-day stamp stays untouched."""
    required = {"candidate_id", "is_warmup", "status", "resolution_ts_utc"}
    if required - set(raw_trades):
        raise ValueError("research trades lack exact candidate, warmup or resolution evidence")
    flags = raw_trades["is_warmup"]
    if flags.isna().any() or not flags.isin([True, False]).all():
        raise ValueError("research trades require explicit non-null boolean warmup provenance")
    cutoff = pd.Timestamp(cutoff_ts_utc)
    if cutoff.tzinfo is None:
        raise ValueError("research trade cutoff must be timezone-aware")
    warmup = flags.astype(bool)
    candidate = raw_trades["candidate_id"].astype(str).isin(set(candidate_ids))
    resolution = pd.to_datetime(raw_trades["resolution_ts_utc"], utc=True, errors="raise")
    resolved = raw_trades["status"].eq("resolved") & resolution.notna()
    in_cohort = candidate & ~warmup
    return {
        "included": in_cohort & resolved & resolution.lt(cutoff),
        "warmup": warmup,
        "out_of_cohort": ~warmup & ~candidate,
        "cutoff_censored": in_cohort & resolved & resolution.ge(cutoff),
        "unresolved": in_cohort & ~resolved,
    }


def _assert_bound(envelope: ExecutedTradeTableEnvelope, table_bytes: bytes) -> None:
    if hashlib.sha256(table_bytes).hexdigest() != envelope.executed_trade_table_sha256:
        raise ValueError("executed-trade table bytes do not hash to the envelope")
    if len(table_bytes) != int(envelope.byte_size):
        raise ValueError("executed-trade table byte size disagrees with the envelope")
    if envelope.payload.executed_trade_arrow_schema_hash != EXECUTED_TRADE_TABLE_SCHEMA_HASH:
        raise ValueError("executed-trade table envelope names another projection schema")


def save_executed_trade_table(
    root: Path, envelope: ExecutedTradeTableEnvelope, table_bytes: bytes
) -> tuple[ExecutedTradeTableEnvelope, bool]:
    """Publish once; identical bytes reuse; different bytes under the same
    identity fail closed (``SearchStoreError``)."""

    _assert_bound(envelope, table_bytes)
    return save_or_reuse_envelope(
        Path(root),
        EXECUTED_TRADE_TABLE_STORE,
        envelope,
        extra_files={EXECUTED_TRADE_TABLE_SIDECAR: table_bytes},
    )


def probe_executed_trade_table(root: Path, executed_trade_table_id: str) -> str:
    """``"present"`` (manifest, sidecar hash and envelope identity verified)
    or ``"absent"`` (the store entry directory does not exist); every corrupt
    state — an entry directory without its manifest included — raises the
    store's typed :class:`SidecarLoadError` (R6.1-FIX §3.7 / §3.8; reviews
    B-02 / B-08: corrupt is never absent, and the reason is raised at the
    point of detection, never inferred from an exception's text)."""

    store = EXECUTED_TRADE_TABLE_STORE
    try:
        state = probe_sidecar(
            Path(root), store, executed_trade_table_id, EXECUTED_TRADE_TABLE_SIDECAR
        )
    except SidecarLoadError as error:
        if error.reason == "store_entry_missing":
            return "absent"
        raise
    if state != "present":
        raise SidecarLoadError(
            "sidecar_missing_but_manifest_declares_it",
            f"{store}/{executed_trade_table_id} carries no table sidecar",
        )
    # the envelope's own verification: every failure is typed by the store
    load_verified_envelope(Path(root), store, executed_trade_table_id, ExecutedTradeTableEnvelope)
    return "present"


def load_executed_trade_table(
    root: Path, executed_trade_table_id: str
) -> VerifiedExecutedTradeTable:
    """Exact-id verified load: manifest, envelope identity, sidecar hash, the
    envelope's own binding, the declared schema, and the row count."""

    store = EXECUTED_TRADE_TABLE_STORE
    envelope = load_verified_envelope(
        Path(root), store, executed_trade_table_id, ExecutedTradeTableEnvelope
    )
    table_bytes = load_sidecar_bytes(
        Path(root), store, executed_trade_table_id, EXECUTED_TRADE_TABLE_SIDECAR
    )
    _assert_bound(envelope, table_bytes)
    import pyarrow.ipc  # noqa: PLC0415

    with pyarrow.ipc.open_file(pa.BufferReader(table_bytes)) as reader:
        schema = reader.schema
    if arrow_schema_hash(schema) != EXECUTED_TRADE_TABLE_SCHEMA_HASH:
        raise ValueError(
            "stored executed-trade table does not carry the declared projection schema"
        )
    frame = frame_from_arrow_bytes(table_bytes)
    if len(frame) != int(envelope.row_count):
        raise ValueError("stored executed-trade table row count disagrees with the envelope")
    return VerifiedExecutedTradeTable(envelope=envelope, frame=frame, table_bytes=table_bytes)


register_identity_pair(
    name="ExecutedTradeTable",
    envelope_cls=ExecutedTradeTableEnvelope,
    payload_cls=ExecutedTradeTablePayload,
    id_field="executed_trade_table_id",
    example_factory=lambda: ExecutedTradeTablePayload(
        core_replay_id="a" * 64,
        source_record_schema_version=2,
        executed_trade_arrow_schema_hash=EXECUTED_TRADE_TABLE_SCHEMA_HASH,
    ),
    extra_envelope_fields=(
        "executed_trade_table_sha256",
        "source_core_table_hash",
        "row_count",
        "byte_size",
    ),
)
