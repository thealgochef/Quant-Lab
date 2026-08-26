"""Immutable MBP-1 source/coverage artifact (R5B deliverable 1).

One artifact freezes the exact MBP-1 evidence a materialization consumed:
per-partition content hashes, row counts, the first/last order keys, the
sequence-gap intervals (``sequence_gap_marks_interval_invalid_v1``), and the
per-day coverage fraction — all under the pinned
:mod:`mbp1_arrow_schemas` contract. Synthetic fixtures persist their
canonicalized event bytes as manifest-hashed sidecars so the artifact is
self-contained; a real artifact references the source partitions by content
hash without copying event data.

Real sources are reachable ONLY through an access policy whose
``authorize_date`` gate runs before any path is constructed
(authorize-before-path); ``legacy_verified_replay_source`` provenance is
refused here outright — opaque legacy replay provenance can never enter
feature materialization (V3 P0-3).

Vendor sequence RESETS (a decrease) are not gaps; only a positive jump
greater than one inside a partition marks the interval between the adjacent
events invalid. Research-only offline (owner decision R-6).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
from pydantic import Field

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)
from ..search.store import (
    load_sidecar_bytes,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .mbp1_arrow_schemas import (
    DATABENTO_PRICE_SCALE,
    INSTRUMENT_TICK_SIZES,
    MBP1_NORMALIZED_EVENT_SCHEMA,
    MBP1_NORMALIZED_EVENT_SCHEMA_HASH,
    MBP1_SOURCE_EVENT_SCHEMA,
    MBP1_SOURCE_EVENT_SCHEMA_HASH,
    assert_schema_names_match,
)
from .mbp1_source_contract import (
    DEEP_BOOK_EXEMPT_LITERALS,
    DEEP_BOOK_IDENTIFIER_REGEX,
    Mbp1SourceContract,
)

__all__ = [
    "MBP1_SOURCE_ARTIFACT_STORE",
    "ORDER_KEY_COLUMNS",
    "Mbp1PartitionCoverage",
    "Mbp1SourceArtifactPayload",
    "Mbp1SourceArtifactEnvelope",
    "normalize_mbp1_events",
    "build_mbp1_source_artifact",
    "read_mbp1_partition_frame",
    "save_mbp1_source_artifact",
    "load_mbp1_source_artifact",
    "load_partition_events",
]

MBP1_SOURCE_ARTIFACT_STORE = "mbp1_source_artifacts"

#: The complete deterministic total order (revision P0-19).
ORDER_KEY_COLUMNS: tuple[str, ...] = ("ts_event", "ts_recv", "sequence", "source_ordinal")


class Mbp1PartitionCoverage(FrozenContract):
    """Coverage facts for one physical MBP-1 partition of one trading day."""

    trading_day: str
    source_partition_utc_date: str
    relative_logical_partition_key: str
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    row_count: int = Field(ge=0)
    first_ts_event: int | None
    last_ts_event: int | None
    first_sequence: int | None
    last_sequence: int | None
    sequence_gap_intervals: tuple[tuple[int, int], ...]
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    instrument_ids: tuple[int, ...]


class Mbp1SourceArtifactPayload(FrozenContract):
    source_contract: Mbp1SourceContract
    authorized_date_set_id: str
    ordered_partitions: tuple[Mbp1PartitionCoverage, ...]
    source_schema_hash: str = Field(pattern=SHA256_PATTERN)
    normalized_schema_hash: str = Field(pattern=SHA256_PATTERN)


class Mbp1SourceArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "mbp1_source_artifact_id"

    mbp1_source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: Mbp1SourceArtifactPayload
    #: STORAGE MODE is operational, never identity (review F7): byte-identical
    #: evidence has ONE artifact id whether its canonical event bytes are
    #: persisted as sidecars (synthetic fixtures) or referenced by hash only
    #: (real sources).
    events_stored: bool


#: A column name carrying a book level beyond 00 (bid_px_01 … ask_ct_09 …).
_BOOK_LEVEL_SUFFIX = re.compile(r"_(0[1-9]|[1-9][0-9])$")


def _refuse_deeper_book_columns(names: list[str]) -> None:
    """A source whose ACTUAL columns carry any book level beyond 00 is not
    an mbp-1 partition (e.g. a legacy-era mbp10 file) — refused before any
    row decodes (safety review S1). The exposure-side schemas already pin
    level-00 fields only; this closes the ingestion side."""

    offenders = sorted(
        name
        for name in names
        if _BOOK_LEVEL_SUFFIX.search(name)
        or (
            name not in DEEP_BOOK_EXEMPT_LITERALS
            and DEEP_BOOK_IDENTIFIER_REGEX.search(name)
        )
    )
    if offenders:
        raise PermissionError(
            "the source carries book levels beyond MBP-1 and is not readable "
            f"by this lane: {offenders}"
        )


def _require_no_legacy_provenance(source_kind: str) -> None:
    """The documented feature-layer boundary for the one opaque provenance
    literal (V3 P0-3). The layer is additionally sealed STRUCTURALLY — no
    public callable in this package accepts a ``source_kind`` at all
    (API-surface-scan tested) — so nothing can ever route the literal here;
    this function is the executable statement of the refusal."""

    if source_kind == "legacy_verified_replay_source":
        raise PermissionError(
            "opaque legacy replay provenance cannot enter MBP-1 feature "
            "materialization (V3 P0-3); it is unqueryable from this layer"
        )


def normalize_mbp1_events(
    raw: pd.DataFrame,
    *,
    instrument: str,
    trading_day: str,
) -> pd.DataFrame:
    """Raw source rows → the pinned normalized working frame.

    ``source_ordinal`` is assigned from the SOURCE row order before any sort
    (it is the deterministic final tie-break of the four-part key); prices
    convert to ticks under the pinned scale policy; unknown instruments are
    refused rather than guessed. The result is stably sorted by the complete
    order key.
    """

    tick_size = INSTRUMENT_TICK_SIZES.get(instrument)
    if tick_size is None:
        raise ValueError(
            f"instrument {instrument!r} has no pinned tick size; refusing to scale prices"
        )
    assert_schema_names_match(MBP1_SOURCE_EVENT_SCHEMA, list(raw.columns))
    frame = pd.DataFrame(
        {
            "ts_event": pd.to_numeric(raw["ts_event"], errors="raise").astype("int64"),
            "ts_recv": pd.to_numeric(raw["ts_recv"], errors="raise").astype("int64"),
            "sequence": pd.to_numeric(raw["sequence"], errors="raise").astype("int64"),
            "source_ordinal": pd.RangeIndex(len(raw)).astype("int64"),
            "action": raw["action"].astype(str),
            "side": raw["side"].astype(str),
            "size": pd.to_numeric(raw["size"], errors="raise").astype("int64"),
            "bid_px_ticks": (
                pd.to_numeric(raw["bid_px_00"], errors="raise").astype("float64")
                * DATABENTO_PRICE_SCALE
                / tick_size
            ),
            "ask_px_ticks": (
                pd.to_numeric(raw["ask_px_00"], errors="raise").astype("float64")
                * DATABENTO_PRICE_SCALE
                / tick_size
            ),
            "bid_sz": pd.to_numeric(raw["bid_sz_00"], errors="raise").astype("int64"),
            "ask_sz": pd.to_numeric(raw["ask_sz_00"], errors="raise").astype("int64"),
            "bid_ct": pd.to_numeric(raw["bid_ct_00"], errors="raise").astype("int64"),
            "ask_ct": pd.to_numeric(raw["ask_ct_00"], errors="raise").astype("int64"),
            "instrument_id": pd.to_numeric(raw["instrument_id"], errors="raise").astype(
                "int64"
            ),
            "symbol": raw["symbol"].astype(str),
            "trading_day": trading_day,
        }
    )
    return frame.sort_values(list(ORDER_KEY_COLUMNS), kind="stable").reset_index(
        drop=True
    )


def _canonical_event_bytes(normalized: pd.DataFrame) -> bytes:
    """Deterministic Arrow IPC bytes of one normalized partition frame."""

    table = pa.Table.from_pandas(
        normalized, schema=MBP1_NORMALIZED_EVENT_SCHEMA, preserve_index=False
    )
    sink = BytesIO()
    with pyarrow.ipc.new_file(sink, MBP1_NORMALIZED_EVENT_SCHEMA) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _decode_event_bytes(data: bytes) -> pd.DataFrame:
    with pyarrow.ipc.open_file(BytesIO(data)) as reader:
        return reader.read_all().to_pandas()


def _sequence_gap_intervals(normalized: pd.DataFrame) -> tuple[tuple[int, int], ...]:
    """Positive sequence jumps > 1 invalidate the interval between the
    adjacent events; vendor sequence RESETS (a decrease) are not gaps."""

    if len(normalized) < 2:
        return ()
    sequences = normalized["sequence"].to_numpy()
    ts_events = normalized["ts_event"].to_numpy()
    intervals: list[tuple[int, int]] = []
    for index in range(1, len(sequences)):
        step = int(sequences[index]) - int(sequences[index - 1])
        if step > 1:
            intervals.append((int(ts_events[index - 1]), int(ts_events[index])))
    return tuple(intervals)


def _partition_coverage(
    normalized: pd.DataFrame,
    *,
    trading_day: str,
    source_partition_utc_date: str,
    relative_logical_partition_key: str,
    content_sha256: str,
) -> Mbp1PartitionCoverage:
    gaps = _sequence_gap_intervals(normalized)
    if normalized.empty:
        first_ts = last_ts = first_seq = last_seq = None
        coverage = 0.0
    else:
        first_ts = int(normalized["ts_event"].iloc[0])
        last_ts = int(normalized["ts_event"].iloc[-1])
        first_seq = int(normalized["sequence"].iloc[0])
        last_seq = int(normalized["sequence"].iloc[-1])
        span = last_ts - first_ts
        if span <= 0:
            coverage = 1.0
        else:
            gap_ns = sum(end - start for start, end in gaps)
            coverage = max(0.0, min(1.0, 1.0 - gap_ns / span))
    return Mbp1PartitionCoverage(
        trading_day=trading_day,
        source_partition_utc_date=source_partition_utc_date,
        relative_logical_partition_key=relative_logical_partition_key,
        content_sha256=content_sha256,
        row_count=int(len(normalized)),
        first_ts_event=first_ts,
        last_ts_event=last_ts,
        first_sequence=first_seq,
        last_sequence=last_seq,
        sequence_gap_intervals=gaps,
        coverage_fraction=coverage,
        instrument_ids=tuple(
            sorted(int(v) for v in normalized["instrument_id"].unique())
        )
        if not normalized.empty
        else (),
    )


def build_mbp1_source_artifact(
    normalized_by_day: Mapping[str, pd.DataFrame],
    *,
    contract: Mbp1SourceContract,
    authorized_date_set_id: str,
    events_stored: bool = True,
) -> tuple[Mbp1SourceArtifactEnvelope, dict[str, bytes]]:
    """Freeze coverage over already-normalized per-day frames.

    Returns the envelope plus the canonical per-day event bytes (persisted
    as sidecars when ``events_stored``). Content addressing hashes the
    canonical bytes, so byte-identical evidence reuses one artifact.
    """

    partitions: list[Mbp1PartitionCoverage] = []
    event_bytes: dict[str, bytes] = {}
    for day in sorted(normalized_by_day):
        normalized = normalized_by_day[day]
        missing = [
            field.name
            for field in MBP1_NORMALIZED_EVENT_SCHEMA
            if field.name not in normalized.columns
        ]
        if missing:
            raise ValueError(
                f"normalized frame for {day} lacks pinned columns: {missing}"
            )
        data = _canonical_event_bytes(normalized)
        event_bytes[day] = data
        partitions.append(
            _partition_coverage(
                normalized,
                trading_day=day,
                source_partition_utc_date=day,
                relative_logical_partition_key="day_utc_date/mbp1",
                content_sha256=_bytes_sha256(data),
            )
        )
    payload = Mbp1SourceArtifactPayload(
        source_contract=contract,
        authorized_date_set_id=authorized_date_set_id,
        ordered_partitions=tuple(partitions),
        source_schema_hash=MBP1_SOURCE_EVENT_SCHEMA_HASH,
        normalized_schema_hash=MBP1_NORMALIZED_EVENT_SCHEMA_HASH,
    )
    return (
        Mbp1SourceArtifactEnvelope.from_payload(payload, events_stored=events_stored),
        event_bytes,
    )


def _bytes_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def read_mbp1_partition_frame(
    day: str,
    *,
    access_policy,
    path_factory: Callable[[str], Path],
    instrument: str,
) -> pd.DataFrame:
    """Authorize-before-path read of ONE real partition, then normalize.

    ``access_policy`` must expose ``authorize_date``/``resolve_source_path``
    (the verification policy family): authorization runs BEFORE the path
    factory is invoked, and every open is recorded in the policy's audit.
    The pinned source schema (including ``ts_recv``) is validated on the
    parquet before any rows decode.
    """

    if access_policy is None:
        raise PermissionError(
            "MBP-1 source reads require an access policy (authorize-before-path); "
            "refusing without one"
        )
    path = access_policy.resolve_source_path(day, path_factory)
    schema = pq.read_schema(path)
    assert_schema_names_match(MBP1_SOURCE_EVENT_SCHEMA, list(schema.names))
    _refuse_deeper_book_columns(list(schema.names))
    # safety review S1: even a lawful file is read through the pinned column
    # projection — nothing beyond the mbp-1 contract is ever materialized
    raw = pq.read_table(
        path, columns=[field.name for field in MBP1_SOURCE_EVENT_SCHEMA]
    ).to_pandas()
    if hasattr(access_policy, "record_file_open"):
        access_policy.record_file_open(day, rows=int(len(raw)))
    return normalize_mbp1_events(raw, instrument=instrument, trading_day=day)


def build_mbp1_source_artifact_from_paths(
    days: Iterable[str],
    *,
    access_policy,
    path_factory: Callable[[str], Path],
    contract: Mbp1SourceContract,
    authorized_date_set_id: str,
) -> tuple[Mbp1SourceArtifactEnvelope, dict[str, bytes]]:
    """The real-source builder: every day authorized before its path exists.

    Real artifacts do not copy event data (``events_stored=False``); the
    returned bytes mapping is empty and the coverage rows carry the content
    hashes of the canonicalized evidence.
    """

    ordered_days = tuple(days)
    if access_policy is None:
        raise PermissionError(
            "the real MBP-1 source builder requires an access policy "
            "(authorize-before-path); refusing without one"
        )
    if not ordered_days:
        raise ValueError("the real MBP-1 source builder requires at least one day")
    normalized_by_day: dict[str, pd.DataFrame] = {}
    for day in ordered_days:
        normalized_by_day[day] = read_mbp1_partition_frame(
            day,
            access_policy=access_policy,
            path_factory=path_factory,
            instrument=contract.instrument,
        )
    envelope, _event_bytes = build_mbp1_source_artifact(
        normalized_by_day,
        contract=contract,
        authorized_date_set_id=authorized_date_set_id,
        events_stored=False,
    )
    return envelope, {}


def save_mbp1_source_artifact(
    root: Path,
    envelope: Mbp1SourceArtifactEnvelope,
    event_bytes: Mapping[str, bytes],
) -> tuple:
    """Immutable save (or verified reuse) with per-day event sidecars."""

    if envelope.events_stored:
        expected_days = {p.trading_day for p in envelope.payload.ordered_partitions}
        if set(event_bytes) != expected_days:
            raise ValueError(
                "events_stored artifacts must persist exactly the covered days"
            )
        # review F1: a sidecar that does not hash to its coverage row's
        # content_sha256 is refused BEFORE any store write — the persisted
        # evidence can never contradict the envelope's own coverage claim
        by_day = {p.trading_day: p for p in envelope.payload.ordered_partitions}
        for day, data in event_bytes.items():
            if _bytes_sha256(data) != by_day[day].content_sha256:
                raise ValueError(
                    f"event bytes for {day} do not hash to the artifact's "
                    "coverage content_sha256; refusing to save"
                )
    extra = {
        f"events_{day.replace('-', '')}.arrow": data
        for day, data in sorted(event_bytes.items())
    }
    return save_or_reuse_envelope(
        Path(root), MBP1_SOURCE_ARTIFACT_STORE, envelope, extra_files=extra
    )


def load_mbp1_source_artifact(root: Path, artifact_id: str) -> Mbp1SourceArtifactEnvelope:
    return load_verified_envelope(
        Path(root), MBP1_SOURCE_ARTIFACT_STORE, artifact_id, Mbp1SourceArtifactEnvelope
    )


def load_partition_events(
    root: Path, envelope: Mbp1SourceArtifactEnvelope, day: str
) -> pd.DataFrame:
    """Manifest-verified reload of one stored day's canonical events.

    The decoded bytes are additionally rehashed against the coverage row's
    ``content_sha256`` — a tampered sidecar fails closed even if a manifest
    were regenerated around it.
    """

    coverage = next(
        (p for p in envelope.payload.ordered_partitions if p.trading_day == day), None
    )
    if coverage is None:
        raise KeyError(f"day {day} is not covered by this MBP-1 source artifact")
    if not envelope.events_stored:
        raise PermissionError(
            "this artifact references real source partitions by hash only; "
            "event reloads require the authorized source access path"
        )
    data = load_sidecar_bytes(
        Path(root),
        MBP1_SOURCE_ARTIFACT_STORE,
        envelope.mbp1_source_artifact_id,
        f"events_{day.replace('-', '')}.arrow",
    )
    if _bytes_sha256(data) != coverage.content_sha256:
        raise ValueError(
            f"stored event bytes for {day} do not hash to the artifact's "
            "content_sha256; refusing to load"
        )
    return _decode_event_bytes(data)


def _example_source_artifact_payload() -> Mbp1SourceArtifactPayload:
    from .mbp1_source_contract import R5B_WINDOW_SPECS  # noqa: PLC0415

    contract = Mbp1SourceContract(
        instrument="NQ",
        contract_roll_policy_id="front_month_open_interest_roll_v1",
        feature_window_specs=R5B_WINDOW_SPECS,
        coverage_policy={"min_day_coverage_fraction": 0.95},
    )
    return Mbp1SourceArtifactPayload(
        source_contract=contract,
        authorized_date_set_id="synthetic_fixture_days_v1",
        ordered_partitions=(
            Mbp1PartitionCoverage(
                trading_day="2026-01-13",
                source_partition_utc_date="2026-01-13",
                relative_logical_partition_key="day_utc_date/mbp1",
                content_sha256="a" * 64,
                row_count=1,
                first_ts_event=1,
                last_ts_event=2,
                first_sequence=10,
                last_sequence=11,
                sequence_gap_intervals=(),
                coverage_fraction=1.0,
                instrument_ids=(1,),
            ),
        ),
        source_schema_hash=MBP1_SOURCE_EVENT_SCHEMA_HASH,
        normalized_schema_hash=MBP1_NORMALIZED_EVENT_SCHEMA_HASH,
    )


_require_no_legacy_provenance("mbp1")  # self-check: the literal itself is lawful

register_identity_pair(
    name="Mbp1SourceArtifact",
    envelope_cls=Mbp1SourceArtifactEnvelope,
    payload_cls=Mbp1SourceArtifactPayload,
    id_field="mbp1_source_artifact_id",
    example_factory=_example_source_artifact_payload,
    extra_envelope_fields=("events_stored",),
)
