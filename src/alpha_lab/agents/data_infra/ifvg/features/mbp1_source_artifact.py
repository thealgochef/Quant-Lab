"""Immutable MBP-1 source/coverage artifact (R5B deliverable 1; R5B.1 policy v2).

One artifact freezes the exact MBP-1 evidence a materialization consumed:
per-partition content hashes, row counts, the first/last order keys, the
EVIDENCE-BASED coverage facts of policy v2 (:mod:`mbp1_coverage_evidence`),
and the raw sequence/``ts_recv`` DIAGNOSTICS — all under the pinned
:mod:`mbp1_arrow_schemas` contract. Synthetic fixtures persist their
canonicalized event bytes as manifest-hashed sidecars so the artifact is
self-contained; a real artifact references the source partitions by content
hash without copying event data.

R5B.1 (owner planning decision Q1 + plan-review correction 4): the rule
"positive raw venue sequence jump > 1 = source gap" is WITHDRAWN. A vendor
sequence RESET or JUMP is recorded as a diagnostic only. Coverage comes from
verified partition-scope evidence (declared gap manifests, the
``F_MAYBE_BAD_BOOK`` channel-gap flag under the documented-recovery rule,
dataset-condition records) at an explicit scope; a partition without
partition-scope evidence is ``completeness_unknown`` and every window of
that day is typed ``coverage_evidence_unavailable`` downstream (fail closed).

Real sources are reachable ONLY through an access policy whose
``authorize_date`` gate runs before any path is constructed
(authorize-before-path); ``legacy_verified_replay_source`` provenance is
refused here outright — opaque legacy replay provenance can never enter
feature materialization (V3 P0-3). Research-only offline (owner decision R-6).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from ..development_access import DEVELOPMENT_CUTOFF_UTC
from ..manifest import file_sha256
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
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
from .mbp1_coverage_evidence import (
    Mbp1CompletenessStatus,
    Mbp1DatasetConditionStatus,
    Mbp1DeclaredGapInterval,
    Mbp1EvidenceProvenance,
    Mbp1EvidenceScope,
    Mbp1EvidenceScopeLevel,
    Mbp1PartitionEvidence,
    Mbp1SequenceJumpDiagnostics,
    Mbp1TsRecvGapDiagnostics,
    assert_evidence_date_representable,
    authorized_session_span_ns,
    bad_book_uncertainty_intervals,
    compute_partition_coverage,
    flag_counts,
    merge_intervals,
    sequence_jump_diagnostics,
    ts_recv_gap_diagnostics,
    weighted_day_coverage,
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
    "Mbp1DayCoverageView",
    "normalize_mbp1_events",
    "build_mbp1_source_artifact",
    "day_coverage_views",
    "read_mbp1_partition_frame",
    "save_mbp1_source_artifact",
    "load_mbp1_source_artifact",
    "load_partition_events",
    "assert_evidence_provenance_permitted",
]

MBP1_SOURCE_ARTIFACT_STORE = "mbp1_source_artifacts"

#: The complete deterministic total order (revision P0-19).
ORDER_KEY_COLUMNS: tuple[str, ...] = ("ts_event", "ts_recv", "sequence", "source_ordinal")


class Mbp1PartitionCoverage(FrozenContract):
    """Coverage facts for one physical MBP-1 partition of one trading day
    (policy v2). ``content_sha256`` is the hash of the trading day's
    canonical event bytes (shared by every physical partition of that day)."""

    trading_day: str
    source_partition_utc_date: str
    relative_logical_partition_key: str
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    row_count: int = Field(ge=0)
    first_ts_event: int | None
    last_ts_event: int | None
    first_sequence: int | None
    last_sequence: int | None
    #: ── evidence-based coverage (D12) ──
    evidence_scope: Mbp1EvidenceScope | None
    evidence_provenance: Mbp1EvidenceProvenance
    declared_gap_intervals: tuple[Mbp1DeclaredGapInterval, ...]
    open_uncertainty_to_partition_end: bool
    completeness_status: Mbp1CompletenessStatus
    dataset_condition_status: Mbp1DatasetConditionStatus
    evidence_sources: tuple[str, ...]
    physical_expected_span_ns: int = Field(ge=0)
    union_gap_ns: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    #: ── diagnostics (never coverage evidence) ──
    sequence_jump_diagnostics: Mbp1SequenceJumpDiagnostics
    ts_recv_gap_diagnostics: Mbp1TsRecvGapDiagnostics
    flag_counts: ImmutableMap[str, int]
    instrument_ids: tuple[int, ...]
    #: the content hashes a positive completeness claim may certify: the
    #: canonical event bytes (``content_sha256``) and, on the real path, the
    #: raw source file's sha256 (review F1)
    source_content_refs: tuple[str, ...] = ()
    #: rows of the day's file that lay outside the authorized session span /
    #: at-or-after DEVELOPMENT_CUTOFF_UTC and were CLIPPED before
    #: normalization (real path; review S1) — never counted, never hashed
    rows_outside_session_span: int = Field(default=0, ge=0)
    rows_after_development_cutoff: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _unknown_never_positive(self):
        assert_evidence_date_representable(self.trading_day)
        assert_evidence_date_representable(self.source_partition_utc_date)
        if (
            self.completeness_status is Mbp1CompletenessStatus.EVIDENCED_COMPLETE
            and self.declared_gap_intervals
        ):
            raise ValueError("evidenced_complete cannot carry declared gap intervals")
        if self.evidence_provenance is not Mbp1EvidenceProvenance.NONE and (
            self.evidence_scope is None
            or not any(source.startswith("gap_manifest:") for source in self.evidence_sources)
        ):
            raise ValueError(
                "a provenance label requires partition-scope evidence (a gap manifest "
                "in evidence_sources); a labeled row without evidence is refused"
            )
        if self.evidence_scope is None and (
            self.completeness_status is not Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
        ):
            raise ValueError("a partition without evidence scope is completeness_unknown")
        if (
            self.completeness_status is Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
            and self.coverage_fraction != 0.0
        ):
            raise ValueError("a completeness_unknown partition never reports positive coverage")
        return self


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


class Mbp1DayCoverageView(FrozenContract):
    """The trading-day aggregate over its physical partitions (duration-
    weighted; the weakest partition status wins; intervals unioned)."""

    trading_day: str
    partition_count: int = Field(ge=1)
    row_count: int = Field(ge=0)
    first_ts_event: int | None
    last_ts_event: int | None
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    completeness_status: Mbp1CompletenessStatus
    dataset_condition_status: Mbp1DatasetConditionStatus
    gap_intervals: tuple[tuple[int, int], ...]
    open_uncertainty_to_partition_end: bool
    sequence_positive_jump_count: int = Field(ge=0)
    #: the authorized session span and the sub-spans NO declared physical
    #: partition evidences (review F2): windows intersecting them are typed
    #: ``coverage_evidence_unavailable`` — never valid on unevidenced time
    session_span_ns: int = Field(default=0, ge=0)
    uncovered_session_intervals: tuple[tuple[int, int], ...] = ()
    uncovered_session_ns: int = Field(default=0, ge=0)
    rows_outside_session_span: int = Field(default=0, ge=0)
    rows_after_development_cutoff: int = Field(default=0, ge=0)


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
    refused rather than guessed; the vendor's ``publisher_id`` and DBN
    ``flags`` are retained for the policy-v2 coverage evidence. The result
    is stably sorted by the complete order key.
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
            "publisher_id": pd.to_numeric(raw["publisher_id"], errors="raise").astype(
                "int64"
            ),
            "flags": pd.to_numeric(raw["flags"], errors="raise").astype("int64"),
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


def _bytes_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _partition_slice(normalized: pd.DataFrame, scope: Mbp1EvidenceScope) -> pd.DataFrame:
    """The day's events inside one physical partition's HALF-OPEN span
    ``[start, end)`` — contiguous partitions never share a boundary event."""

    if normalized.empty:
        return normalized
    mask = (normalized["ts_event"] >= scope.partition_expected_start_ts) & (
        normalized["ts_event"] < scope.partition_expected_end_ts
    )
    return normalized.loc[mask]


def _unknown_partition_coverage(
    normalized: pd.DataFrame,
    *,
    trading_day: str,
    content_sha256: str,
    source_content_refs: tuple[str, ...] = (),
    rows_outside_session_span: int = 0,
    rows_after_development_cutoff: int = 0,
) -> Mbp1PartitionCoverage:
    """No partition-scope evidence at all: completeness_unknown, fail closed.
    The row describes the ONE physical file that was read (``day_utc_date``)."""

    return Mbp1PartitionCoverage(
        trading_day=trading_day,
        source_partition_utc_date=trading_day,
        relative_logical_partition_key="day_utc_date/mbp1",
        content_sha256=content_sha256,
        row_count=int(len(normalized)),
        first_ts_event=int(normalized["ts_event"].iloc[0]) if len(normalized) else None,
        last_ts_event=int(normalized["ts_event"].iloc[-1]) if len(normalized) else None,
        first_sequence=int(normalized["sequence"].iloc[0]) if len(normalized) else None,
        last_sequence=int(normalized["sequence"].iloc[-1]) if len(normalized) else None,
        evidence_scope=None,
        evidence_provenance=Mbp1EvidenceProvenance.NONE,
        declared_gap_intervals=(),
        open_uncertainty_to_partition_end=False,
        completeness_status=Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN,
        dataset_condition_status=Mbp1DatasetConditionStatus.VENDOR_CONDITION_UNAVAILABLE,
        evidence_sources=(),
        physical_expected_span_ns=0,
        union_gap_ns=0,
        coverage_fraction=0.0,
        sequence_jump_diagnostics=sequence_jump_diagnostics(normalized),
        ts_recv_gap_diagnostics=ts_recv_gap_diagnostics(normalized),
        flag_counts=flag_counts(normalized),
        instrument_ids=tuple(
            sorted(int(v) for v in normalized["instrument_id"].unique())
        )
        if len(normalized)
        else (),
        source_content_refs=source_content_refs or (content_sha256,),
        rows_outside_session_span=rows_outside_session_span,
        rows_after_development_cutoff=rows_after_development_cutoff,
    )


def _partition_coverage(
    normalized: pd.DataFrame,
    evidence: Mbp1PartitionEvidence,
    *,
    trading_day: str,
    content_sha256: str,
    source_content_refs: tuple[str, ...] = (),
    rows_outside_session_span: int = 0,
    rows_after_development_cutoff: int = 0,
) -> Mbp1PartitionCoverage:
    scope = evidence.scope
    scoped = _partition_slice(normalized, scope)
    refs = source_content_refs or (content_sha256,)
    computation = compute_partition_coverage(
        evidence,
        trading_day=trading_day,
        flag_intervals=bad_book_uncertainty_intervals(normalized, evidence),
        partition_content_refs=refs,
    )
    return Mbp1PartitionCoverage(
        trading_day=trading_day,
        source_partition_utc_date=scope.utc_date,
        relative_logical_partition_key=scope.physical_partition_key,
        content_sha256=content_sha256,
        row_count=int(len(scoped)),
        first_ts_event=int(scoped["ts_event"].iloc[0]) if len(scoped) else None,
        last_ts_event=int(scoped["ts_event"].iloc[-1]) if len(scoped) else None,
        first_sequence=int(scoped["sequence"].iloc[0]) if len(scoped) else None,
        last_sequence=int(scoped["sequence"].iloc[-1]) if len(scoped) else None,
        evidence_scope=scope,
        evidence_provenance=evidence.provenance,
        declared_gap_intervals=computation.intervals,
        open_uncertainty_to_partition_end=computation.open_uncertainty_to_partition_end,
        completeness_status=computation.completeness_status,
        dataset_condition_status=computation.dataset_condition_status,
        evidence_sources=computation.evidence_sources,
        physical_expected_span_ns=computation.physical_expected_span_ns,
        union_gap_ns=computation.union_gap_ns,
        coverage_fraction=computation.coverage_fraction,
        sequence_jump_diagnostics=sequence_jump_diagnostics(scoped),
        ts_recv_gap_diagnostics=ts_recv_gap_diagnostics(scoped),
        flag_counts=flag_counts(scoped),
        instrument_ids=tuple(sorted(int(v) for v in scoped["instrument_id"].unique()))
        if len(scoped)
        else (),
        source_content_refs=refs,
        rows_outside_session_span=rows_outside_session_span,
        rows_after_development_cutoff=rows_after_development_cutoff,
    )


def _clipped_spans(
    day_evidence: tuple[Mbp1PartitionEvidence, ...], *, trading_day: str
) -> list[tuple[int, int]]:
    """Each declared partition's span clipped to the authorized session span
    (half-open); overlapping spans are refused (review F3)."""

    session_start, session_end = authorized_session_span_ns(trading_day)
    spans: list[tuple[int, int]] = []
    for evidence in day_evidence:
        start = max(int(evidence.scope.partition_expected_start_ts), session_start)
        end = min(int(evidence.scope.partition_expected_end_ts), session_end)
        if end > start:
            spans.append((start, end))
    spans.sort()
    for (_a_start, a_end), (b_start, _b_end) in zip(spans, spans[1:], strict=False):
        if b_start < a_end:
            raise ValueError(
                f"physical partition spans of {trading_day} overlap after clipping to "
                "the authorized session span; overlapping or duplicated partitions "
                "would dilute declared gaps and are refused"
            )
    return spans


def assert_evidence_provenance_permitted(
    partitions: Iterable[Mbp1PartitionCoverage], *, synthetic_scope: bool
) -> None:
    """Synthetic coverage evidence is lawful ONLY inside a synthetic scope;
    a real scope refuses it before any artifact is trusted."""

    if synthetic_scope:
        return
    offenders = sorted(
        {
            f"{row.trading_day}/{row.relative_logical_partition_key}"
            for row in partitions
            if row.evidence_provenance is Mbp1EvidenceProvenance.SYNTHETIC_FIXTURE
        }
    )
    if offenders:
        raise PermissionError(
            "synthetic coverage evidence cannot enter a real scope: "
            f"{offenders} — real coverage requires owner-reviewed evidence"
        )


def build_mbp1_source_artifact(
    normalized_by_day: Mapping[str, pd.DataFrame],
    *,
    contract: Mbp1SourceContract,
    authorized_date_set_id: str,
    events_stored: bool = True,
    coverage_evidence: Mapping[str, tuple[Mbp1PartitionEvidence, ...]] | None = None,
) -> tuple[Mbp1SourceArtifactEnvelope, dict[str, bytes]]:
    """Freeze coverage over already-normalized per-day frames.

    ``coverage_evidence`` maps a trading day to its physical partitions'
    evidence (policy v2). A day without evidence is recorded as ONE
    ``completeness_unknown`` partition (fail closed); a day with evidence is
    recorded as one row per physical partition, each measured against
    ``intersection(partition span, authorized session span)``. Returns the
    envelope plus the canonical per-day event bytes (persisted as sidecars
    when ``events_stored``). Content addressing hashes the canonical bytes,
    so byte-identical evidence reuses one artifact.
    """

    partitions: list[Mbp1PartitionCoverage] = []
    event_bytes: dict[str, bytes] = {}
    evidence_map = dict(coverage_evidence or {})
    unknown_days = sorted(set(evidence_map) - set(normalized_by_day))
    if unknown_days:
        raise ValueError(
            f"coverage evidence supplied for days without events: {unknown_days}"
        )
    for day in sorted(normalized_by_day):
        normalized = normalized_by_day[day]
        attrs = dict(getattr(normalized, "attrs", {}) or {})
        content_refs: tuple[str, ...] = ()
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
        content = _bytes_sha256(data)
        content_refs = (content,) + tuple(
            ref for ref in (attrs.get("source_file_sha256"),) if ref
        )
        session_start, session_end = authorized_session_span_ns(day)
        ts = (
            normalized["ts_event"].to_numpy(dtype=np.int64)
            if len(normalized)
            else np.array([], dtype=np.int64)
        )
        in_session = (ts >= session_start) & (ts < session_end)
        counters = {
            "rows_outside_session_span": int((~in_session).sum())
            + int(attrs.get("rows_outside_session_span", 0)),
            "rows_after_development_cutoff": int(attrs.get("rows_after_development_cutoff", 0)),
        }
        day_evidence = tuple(evidence_map.get(day) or ())
        if not day_evidence:
            partitions.append(
                _unknown_partition_coverage(
                    normalized,
                    trading_day=day,
                    content_sha256=content,
                    source_content_refs=content_refs,
                    **counters,
                )
            )
            continue
        keys = [evidence.scope.physical_partition_key for evidence in day_evidence]
        if len(set(keys)) != len(keys):
            raise ValueError(
                f"coverage evidence for {day} repeats a physical partition key"
            )
        spans = _clipped_spans(day_evidence, trading_day=day)
        # review F2: every in-session event must lie inside a DECLARED span —
        # a manifest that contradicts the data is refused, never silently clipped
        covered = np.zeros(len(ts), dtype=bool)
        for start, end in spans:
            covered |= (ts >= start) & (ts < end)
        stray = int((in_session & ~covered).sum())
        if stray:
            raise ValueError(
                f"{stray} in-session events of {day} fall outside every declared "
                "physical partition span; the coverage evidence contradicts the data"
            )
        for evidence in sorted(day_evidence, key=lambda e: e.scope.physical_partition_key):
            partitions.append(
                _partition_coverage(
                    normalized,
                    evidence,
                    trading_day=day,
                    content_sha256=content,
                    source_content_refs=content_refs,
                    **counters,
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


def day_coverage_views(
    envelope: Mbp1SourceArtifactEnvelope,
) -> dict[str, Mbp1DayCoverageView]:
    """Trading day → the duration-weighted aggregate of its partitions."""

    by_day: dict[str, list[Mbp1PartitionCoverage]] = {}
    for row in envelope.payload.ordered_partitions:
        by_day.setdefault(row.trading_day, []).append(row)
    views: dict[str, Mbp1DayCoverageView] = {}
    for day, rows in sorted(by_day.items()):
        coverage, status = weighted_day_coverage(
            [
                (row.physical_expected_span_ns, row.coverage_fraction, row.completeness_status)
                for row in rows
            ]
        )
        intervals: list[tuple[int, int]] = []
        for row in rows:
            intervals.extend(
                (interval.start_ts, interval.end_ts) for interval in row.declared_gap_intervals
            )
        firsts = [row.first_ts_event for row in rows if row.first_ts_event is not None]
        lasts = [row.last_ts_event for row in rows if row.last_ts_event is not None]
        condition = max(
            (row.dataset_condition_status for row in rows),
            key=lambda value: _CONDITION_RANK[value],
        )
        session_start, session_end = authorized_session_span_ns(day)
        declared = merge_intervals(
            [
                (
                    row.evidence_scope.partition_expected_start_ts,
                    row.evidence_scope.partition_expected_end_ts,
                )
                for row in rows
                if row.evidence_scope is not None
            ],
            clip=(session_start, session_end),
        )
        uncovered: list[tuple[int, int]] = []
        cursor = session_start
        for start, end in declared:
            if start > cursor:
                uncovered.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < session_end:
            uncovered.append((cursor, session_end))
        views[day] = Mbp1DayCoverageView(
            trading_day=day,
            partition_count=len(rows),
            row_count=sum(row.row_count for row in rows),
            first_ts_event=min(firsts) if firsts else None,
            last_ts_event=max(lasts) if lasts else None,
            coverage_fraction=coverage,
            completeness_status=status,
            dataset_condition_status=condition,
            gap_intervals=merge_intervals(intervals),
            open_uncertainty_to_partition_end=any(
                row.open_uncertainty_to_partition_end for row in rows
            ),
            sequence_positive_jump_count=sum(
                row.sequence_jump_diagnostics.positive_jump_count for row in rows
            ),
            session_span_ns=int(session_end - session_start),
            uncovered_session_intervals=tuple(uncovered),
            uncovered_session_ns=int(sum(end - start for start, end in uncovered)),
            rows_outside_session_span=max(row.rows_outside_session_span for row in rows),
            rows_after_development_cutoff=max(
                row.rows_after_development_cutoff for row in rows
            ),
        )
    return views


_CONDITION_RANK = {
    Mbp1DatasetConditionStatus.VENDOR_NO_KNOWN_DATASET_ISSUE: 0,
    Mbp1DatasetConditionStatus.VENDOR_CONDITION_UNAVAILABLE: 1,
    Mbp1DatasetConditionStatus.VENDOR_DATASET_PENDING: 2,
    Mbp1DatasetConditionStatus.VENDOR_DATASET_DEGRADED: 3,
    Mbp1DatasetConditionStatus.VENDOR_DATASET_MISSING: 4,
}


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
    # review S1: the UTC-date file is NOT the trading day — rows outside the
    # authorized session span ``[18:00 ET D-1, 17:00 ET D)`` (which includes
    # the next trading day's 18:00 ET tail, protected on the last exposed
    # day) and rows at/after DEVELOPMENT_CUTOFF_UTC are clipped BEFORE
    # normalization and hashing; the counts ride the artifact
    session_start, session_end = authorized_session_span_ns(day)
    cutoff_ns = int(pd.Timestamp(DEVELOPMENT_CUTOFF_UTC).value)
    ts = pd.to_numeric(raw["ts_event"], errors="raise").astype("int64").to_numpy()
    in_session = (ts >= session_start) & (ts < session_end)
    before_cutoff = ts < cutoff_ns
    clipped = raw.loc[in_session & before_cutoff].reset_index(drop=True)
    normalized = normalize_mbp1_events(clipped, instrument=instrument, trading_day=day)
    normalized.attrs["source_file_sha256"] = file_sha256(Path(path))
    normalized.attrs["rows_outside_session_span"] = int((~in_session).sum())
    normalized.attrs["rows_after_development_cutoff"] = int((in_session & ~before_cutoff).sum())
    return normalized


def build_mbp1_source_artifact_from_paths(
    days: Iterable[str],
    *,
    access_policy,
    path_factory: Callable[[str], Path],
    contract: Mbp1SourceContract,
    authorized_date_set_id: str,
    coverage_evidence: Mapping[str, tuple[Mbp1PartitionEvidence, ...]] | None = None,
) -> tuple[Mbp1SourceArtifactEnvelope, dict[str, bytes]]:
    """The real-source builder: every day authorized before its path exists.

    Real artifacts do not copy event data (``events_stored=False``); the
    returned bytes mapping is empty and the coverage rows carry the content
    hashes of the canonicalized evidence. Synthetic coverage evidence is
    refused on this path — real coverage requires owner-reviewed evidence.
    """

    ordered_days = tuple(days)
    if access_policy is None:
        raise PermissionError(
            "the real MBP-1 source builder requires an access policy "
            "(authorize-before-path); refusing without one"
        )
    if not ordered_days:
        raise ValueError("the real MBP-1 source builder requires at least one day")
    for day_evidence in (coverage_evidence or {}).values():
        for evidence in day_evidence:
            if evidence.provenance is Mbp1EvidenceProvenance.SYNTHETIC_FIXTURE:
                raise PermissionError(
                    "synthetic coverage evidence cannot enter the real source "
                    "builder; real coverage requires owner-reviewed evidence"
                )
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
        coverage_evidence=coverage_evidence,
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
        by_day: dict[str, set[str]] = {}
        for p in envelope.payload.ordered_partitions:
            by_day.setdefault(p.trading_day, set()).add(p.content_sha256)
        for day, data in event_bytes.items():
            if by_day[day] != {_bytes_sha256(data)}:
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
    scope = Mbp1EvidenceScope(
        scope_level=Mbp1EvidenceScopeLevel.PUBLISHER_PARTITION,
        dataset="GLBX.MDP3",
        publisher_id=1,
        channel_id=None,
        instrument_id=None,
        symbol=None,
        physical_partition_key="day_utc_date/mbp1",
        source_partition_id="databento/NQ/2026-01-13/mbp1",
        utc_date="2026-01-13",
        partition_expected_start_ts=1,
        partition_expected_end_ts=2,
        partition_span_source_id="example_span_v1",
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
                evidence_scope=scope,
                evidence_provenance=Mbp1EvidenceProvenance.NONE,
                declared_gap_intervals=(),
                open_uncertainty_to_partition_end=False,
                completeness_status=Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN,
                dataset_condition_status=(
                    Mbp1DatasetConditionStatus.VENDOR_CONDITION_UNAVAILABLE
                ),
                evidence_sources=(),
                physical_expected_span_ns=1,
                union_gap_ns=0,
                coverage_fraction=0.0,
                sequence_jump_diagnostics=Mbp1SequenceJumpDiagnostics(
                    event_count=1,
                    positive_jump_count=0,
                    max_positive_jump=0,
                    total_skipped_numbers=0,
                    reset_count=0,
                ),
                ts_recv_gap_diagnostics=Mbp1TsRecvGapDiagnostics(
                    max_gap_ns=0, gaps_over_60s=0, gaps_over_1s=0
                ),
                flag_counts={},
                instrument_ids=(1,),
                source_content_refs=("e" * 64,),
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
