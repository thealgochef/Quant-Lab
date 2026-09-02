"""MBP-1 source-coverage policy v2 — evidence-based, scope-explicit (R5B.1).

The R5B rule "positive raw venue sequence jump > 1 = source gap" is
WITHDRAWN (owner planning decision Q1, 2026-08-28). Databento's ``sequence``
is the venue's ORIGINAL channel sequence number and ``mbp-1`` emits only the
events that change the top price level — not every venue message — so a
symbol-level MBP-1 stream skips sequence numbers legitimately. Raw sequence
jumps and ``ts_recv`` spacing are therefore DIAGNOSTICS ONLY here; they never
open an interval and never type a missing reason.

Coverage is computed only from VERIFIED evidence at an explicit scope level
(``Mbp1EvidenceScope``): a partition-scope declared gap manifest (which may
carry positive completeness only when a verified compiler authorized it), the
vendor's ``F_MAYBE_BAD_BOOK`` flag (a CHANNEL-gap warning — see below), and
the dataset-condition record (coarse dataset/UTC-date evidence that can
downgrade a scope but never establish completeness).

``F_MAYBE_BAD_BOOK`` semantics (Databento DBN ``flags`` bit ``4``): the
vendor documents it as "an unrecoverable gap was detected in the channel".
Two consequences are enforced structurally:

* the uncertainty it opens closes ONLY at a documented recovery boundary
  (a manifest-declared end, a documented vendor recovery/clear event, a
  documented snapshot recovery, or an owner-approved boundary); the next
  unflagged row never closes it — without a documented recovery the
  interval runs to ``partition_expected_end_ts`` (``fail closed``);
* the flag is channel-scoped; because standard mbp-1 rows carry no
  ``channel_id``, the interval may be scoped to a channel partition only
  behind a VERIFIED publisher/channel map, otherwise it expands
  conservatively to the publisher/physical partition and is never narrowed
  to the flagged instrument.

The uncertainty START is (a) a manifest-declared start, else (b) the
immediately prior in-scope event's timestamp when that row is itself
trustworthy (no bad-book and no bad-``ts_recv`` flag), else (c)
``partition_expected_start_ts`` — never automatically the detection row.

Coverage calculation (D12): each physical partition's denominator is
``intersection([partition_expected_start_ts, partition_expected_end_ts],
authorized_session_span)``; in-scope intervals are unioned, overlap-merged,
and clipped to that physical span; ``coverage = 1 − union_gap_ns /
physical_expected_span_ns``. Multiple physical partitions of one trading day
are measured separately and duration-weighted — the trading-day span is
never repeated per partition. A partition without partition-scope evidence
is ``completeness_unknown`` (fail closed); a positive completeness claim can
come only from :func:`compile_mbp1_partition_gap_manifest` over a verified
:class:`Mbp1CompletenessCompilationReportEnvelope` — never from an
unstructured boolean.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from ..development_access import SourceDateClass, _classify
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from ..search.store import load_verified_envelope, save_or_reuse_envelope

_SHA256_RE = re.compile(SHA256_PATTERN)

__all__ = [
    "F_MAYBE_BAD_BOOK_FLAG",
    "F_BAD_TS_RECV_FLAG",
    "DBN_FLAG_BITS",
    "MBP1_COVERAGE_POLICY_V2",
    "SEQUENCE_JUMP_SEMANTICS_V2",
    "TS_RECV_GAP_SEMANTICS_V2",
    "EVIDENCE_KINDS_ACCEPTED",
    "RECOVERY_BOUNDARY_KINDS_ACCEPTED",
    "EXPECTED_SPAN_SOURCE_V2",
    "BAD_BOOK_SCOPE_FALLBACK_V1",
    "COMPLETENESS_COMPILER_POLICY_V1",
    "Mbp1CoverageEvidenceKind",
    "Mbp1EvidenceScopeLevel",
    "Mbp1RecoveryBoundaryKind",
    "Mbp1UncertaintyStartKind",
    "Mbp1CompletenessStatus",
    "Mbp1DatasetConditionStatus",
    "Mbp1EvidenceProvenance",
    "Mbp1EvidenceScope",
    "Mbp1DeclaredGapInterval",
    "Mbp1RecoveryBoundary",
    "Mbp1DatasetConditionRecord",
    "Mbp1CompletenessCompilationReportPayload",
    "Mbp1CompletenessCompilationReportEnvelope",
    "Mbp1PartitionGapManifestPayload",
    "Mbp1PartitionGapManifestEnvelope",
    "Mbp1PartitionEvidence",
    "Mbp1SequenceJumpDiagnostics",
    "Mbp1TsRecvGapDiagnostics",
    "Mbp1PartitionCoverageComputation",
    "dataset_condition_status",
    "compile_mbp1_partition_gap_manifest",
    "authorized_session_span_ns",
    "merge_intervals",
    "sequence_jump_diagnostics",
    "ts_recv_gap_diagnostics",
    "flag_counts",
    "bad_book_uncertainty_intervals",
    "compute_partition_coverage",
    "weighted_day_coverage",
    "MBP1_GAP_MANIFEST_STORE",
    "MBP1_COMPLETENESS_REPORT_STORE",
    "assert_evidence_date_representable",
    "save_mbp1_completeness_report",
    "load_mbp1_completeness_report",
    "save_mbp1_partition_gap_manifest",
    "load_mbp1_partition_gap_manifest",
    "load_verified_partition_evidence",
]

MBP1_GAP_MANIFEST_STORE = "mbp1_gap_manifests"
MBP1_COMPLETENESS_REPORT_STORE = "mbp1_completeness_reports"

#: Protected/sealed dates are UNREPRESENTABLE inside coverage evidence (defense
#: in depth: no path is ever built from these fields, but a sealed date can
#: never be baked into an immutable evidence artifact either).
_UNREPRESENTABLE_DATE_CLASSES = frozenset(
    {
        SourceDateClass.PROTECTED_BUFFER,
        SourceDateClass.SEALED,
        SourceDateClass.OUTSIDE_POLICY,
    }
)


def assert_evidence_date_representable(day: str) -> str:
    try:
        klass = _classify(day)
    except ValueError as error:
        raise ValueError(f"evidence date {day!r} is not an ISO date") from error
    if klass in _UNREPRESENTABLE_DATE_CLASSES:
        raise ValueError(
            f"evidence date {day} is {klass.value}: protected, sealed, or "
            "outside-policy dates are unrepresentable in coverage evidence"
        )
    return day

# ── Databento DBN flag bits (diagnostic vocabulary; only F_MAYBE_BAD_BOOK
#    carries coverage semantics, and only under the documented recovery rule)
F_LAST_FLAG = 128
F_TOB_FLAG = 64
F_SNAPSHOT_FLAG = 32
F_MBP_FLAG = 16
F_BAD_TS_RECV_FLAG = 8
#: "Indicates an unrecoverable gap was detected in the channel" (vendor doc).
F_MAYBE_BAD_BOOK_FLAG = 4

DBN_FLAG_BITS: ImmutableMap[str, int] = ImmutableMap(
    {
        "F_LAST": F_LAST_FLAG,
        "F_TOB": F_TOB_FLAG,
        "F_SNAPSHOT": F_SNAPSHOT_FLAG,
        "F_MBP": F_MBP_FLAG,
        "F_BAD_TS_RECV": F_BAD_TS_RECV_FLAG,
        "F_MAYBE_BAD_BOOK": F_MAYBE_BAD_BOOK_FLAG,
    }
)

MBP1_COVERAGE_POLICY_V2 = "mbp1_source_coverage_declared_evidence_v2"
SEQUENCE_JUMP_SEMANTICS_V2 = "sequence_jump_diagnostic_only_v2"
TS_RECV_GAP_SEMANTICS_V2 = "ts_recv_gap_diagnostic_only_v2"
EXPECTED_SPAN_SOURCE_V2 = "verified_physical_partition_intersect_authorized_session_v2"
BAD_BOOK_SCOPE_FALLBACK_V1 = "publisher_physical_partition_v1"
COMPLETENESS_COMPILER_POLICY_V1 = "mbp1_completeness_compiler_v1"


class Mbp1CoverageEvidenceKind(StrEnum):
    VENDOR_FLAG_MAYBE_BAD_BOOK = "vendor_flag_maybe_bad_book"
    DECLARED_PARTITION_GAP_MANIFEST = "declared_partition_gap_manifest"
    DATASET_CONDITION_RECORD = "dataset_condition_record"


class Mbp1EvidenceScopeLevel(StrEnum):
    DATASET_DATE = "dataset_date"
    PUBLISHER_PARTITION = "publisher_partition"
    CHANNEL_PARTITION = "channel_partition"
    INSTRUMENT_PARTITION = "instrument_partition"


class Mbp1RecoveryBoundaryKind(StrEnum):
    MANIFEST_DECLARED_END = "manifest_declared_end"
    VENDOR_RECOVERY_EVENT = "vendor_recovery_event"
    SNAPSHOT_RECOVERY = "snapshot_recovery"
    OWNER_APPROVED_BOUNDARY = "owner_approved_boundary"
    #: no documented recovery: the interval runs to the partition end
    PARTITION_END_FAIL_CLOSED = "partition_end_fail_closed"


class Mbp1UncertaintyStartKind(StrEnum):
    MANIFEST_DECLARED_START = "manifest_declared_start"
    LAST_TRUSTED_IN_SCOPE_EVENT = "last_trusted_in_scope_event"
    PARTITION_EXPECTED_START = "partition_expected_start"


class Mbp1CompletenessStatus(StrEnum):
    EVIDENCED_COMPLETE = "evidenced_complete"
    DECLARED_GAPS = "declared_gaps"
    COMPLETENESS_UNKNOWN = "completeness_unknown"


class Mbp1DatasetConditionStatus(StrEnum):
    VENDOR_NO_KNOWN_DATASET_ISSUE = "vendor_no_known_dataset_issue"
    VENDOR_DATASET_DEGRADED = "vendor_dataset_degraded"
    VENDOR_DATASET_PENDING = "vendor_dataset_pending"
    VENDOR_DATASET_MISSING = "vendor_dataset_missing"
    VENDOR_CONDITION_UNAVAILABLE = "vendor_condition_unavailable"


class Mbp1EvidenceProvenance(StrEnum):
    OWNER_REVIEWED = "owner_reviewed"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    NONE = "none"


EVIDENCE_KINDS_ACCEPTED: tuple[str, ...] = tuple(kind.value for kind in Mbp1CoverageEvidenceKind)
RECOVERY_BOUNDARY_KINDS_ACCEPTED: tuple[str, ...] = (
    Mbp1RecoveryBoundaryKind.MANIFEST_DECLARED_END.value,
    Mbp1RecoveryBoundaryKind.VENDOR_RECOVERY_EVENT.value,
    Mbp1RecoveryBoundaryKind.SNAPSHOT_RECOVERY.value,
    Mbp1RecoveryBoundaryKind.OWNER_APPROVED_BOUNDARY.value,
)

_ET = ZoneInfo("America/New_York")
_TRADING_DAY_BOUNDARY = time(18, 0)
_CLOSED_WINDOW_START = time(17, 0)


class Mbp1EvidenceScope(FrozenContract):
    """The exact physical scope one piece of coverage evidence describes."""

    scope_level: Mbp1EvidenceScopeLevel
    dataset: str
    publisher_id: int = Field(ge=0)
    channel_id: int | None
    schema_name: Literal["mbp-1"] = Field(default="mbp-1", alias="schema")
    instrument_id: int | None
    symbol: str | None
    physical_partition_key: str
    source_partition_id: str
    utc_date: str
    partition_expected_start_ts: int
    partition_expected_end_ts: int
    partition_span_source_id: str

    model_config = FrozenContract.model_config | {"populate_by_name": True}

    @model_validator(mode="after")
    def _coherent(self):
        assert_evidence_date_representable(self.utc_date)
        if self.partition_expected_end_ts < self.partition_expected_start_ts:
            raise ValueError("partition_expected_end_ts precedes partition_expected_start_ts")
        if (
            self.scope_level is Mbp1EvidenceScopeLevel.CHANNEL_PARTITION
            and self.channel_id is None
        ):
            raise ValueError("channel_partition scope requires a verified channel_id")
        if (
            self.scope_level is Mbp1EvidenceScopeLevel.INSTRUMENT_PARTITION
            and self.instrument_id is None
        ):
            raise ValueError("instrument_partition scope requires an instrument_id")
        for value in (self.source_partition_id, self.physical_partition_key):
            if value.startswith(("/", "\\")) or (len(value) > 1 and value[1] == ":"):
                raise ValueError("absolute paths may not enter an evidence scope")
        return self


class Mbp1DeclaredGapInterval(FrozenContract):
    """One typed uncertainty/gap interval ``[start_ts, end_ts)`` in epoch ns."""

    start_ts: int
    end_ts: int
    evidence_kind: Mbp1CoverageEvidenceKind
    start_kind: Mbp1UncertaintyStartKind
    recovery_boundary_kind: Mbp1RecoveryBoundaryKind
    scope_level: Mbp1EvidenceScopeLevel
    #: typed fact (never free text): this interval is the union of ≥2
    #: overlapping declared/uncertainty intervals
    merged: bool = False
    #: every evidence kind that contributed to this (possibly merged)
    #: interval — provenance is never lost on merge (review F12)
    contributing_evidence_kinds: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _non_empty(self):
        if self.end_ts <= self.start_ts:
            raise ValueError("a declared gap interval must have end_ts > start_ts")
        if not self.merged and self.contributing_evidence_kinds not in (
            (),
            (self.evidence_kind.value,),
        ):
            raise ValueError(
                "an unmerged interval's contributing kinds are its own evidence kind"
            )
        if self.scope_level is Mbp1EvidenceScopeLevel.DATASET_DATE:
            raise ValueError(
                "dataset-date evidence is never an interval: it can downgrade a "
                "scope but never declares partition gaps"
            )
        return self


class Mbp1RecoveryBoundary(FrozenContract):
    """A DOCUMENTED recovery boundary that may close a bad-book interval —
    bound to the physical partition it was documented for (review F6/F7)."""

    ts: int
    kind: Mbp1RecoveryBoundaryKind
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    utc_date: str
    physical_partition_key: str

    @model_validator(mode="after")
    def _documented(self):
        assert_evidence_date_representable(self.utc_date)
        if self.kind.value not in RECOVERY_BOUNDARY_KINDS_ACCEPTED:
            raise ValueError(
                f"recovery boundary kind {self.kind.value!r} is not a documented "
                f"recovery kind; accepted: {RECOVERY_BOUNDARY_KINDS_ACCEPTED}"
            )
        return self


class Mbp1DatasetConditionRecord(FrozenContract):
    """The vendor's dataset-condition record — dataset/UTC-date granularity."""

    dataset: str
    utc_date: str
    condition: Literal["available", "degraded", "pending", "missing"]
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    recorded_at: str

    @model_validator(mode="after")
    def _date_representable(self):
        assert_evidence_date_representable(self.utc_date)
        return self


def dataset_condition_status(
    record: Mbp1DatasetConditionRecord | None,
) -> Mbp1DatasetConditionStatus:
    """Map the coarse vendor condition onto the v2 vocabulary. ``available``
    means "no known dataset issue" — it never proves a partition complete."""

    if record is None:
        return Mbp1DatasetConditionStatus.VENDOR_CONDITION_UNAVAILABLE
    return {
        "available": Mbp1DatasetConditionStatus.VENDOR_NO_KNOWN_DATASET_ISSUE,
        "degraded": Mbp1DatasetConditionStatus.VENDOR_DATASET_DEGRADED,
        "pending": Mbp1DatasetConditionStatus.VENDOR_DATASET_PENDING,
        "missing": Mbp1DatasetConditionStatus.VENDOR_DATASET_MISSING,
    }[record.condition]


#: Dataset conditions that DOWNGRADE a scope to completeness_unknown.
_DOWNGRADING_CONDITIONS = frozenset(
    {
        Mbp1DatasetConditionStatus.VENDOR_DATASET_DEGRADED,
        Mbp1DatasetConditionStatus.VENDOR_DATASET_PENDING,
        Mbp1DatasetConditionStatus.VENDOR_DATASET_MISSING,
    }
)


class Mbp1CompletenessCompilationReportPayload(FrozenContract):
    """The ONLY producer of positive completeness: a verified compilation
    over source inventory/manifest evidence plus the owner's review."""

    scope: Mbp1EvidenceScope
    source_inventory_id: str
    verified_partition_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    owner_review_decision_id: str
    provenance: Literal["owner_reviewed", "synthetic_fixture"]
    compiled_intervals: tuple[Mbp1DeclaredGapInterval, ...]
    positive_completeness_authorized: bool
    compiler_policy_id: Literal["mbp1_completeness_compiler_v1"] = (
        COMPLETENESS_COMPILER_POLICY_V1
    )
    compiled_at: str

    @model_validator(mode="after")
    def _authorized_requires_review(self):
        if self.positive_completeness_authorized and not _SHA256_RE.fullmatch(
            self.owner_review_decision_id or ""
        ):
            raise ValueError(
                "positive completeness requires an owner review decision id — the "
                "64-hex content hash of the owner's decision evidence, never free text"
            )
        for ref in self.verified_partition_refs:
            if not _SHA256_RE.fullmatch(ref):
                raise ValueError("verified_partition_refs must be content sha256 values")
        for ref in self.evidence_refs:
            if not _SHA256_RE.fullmatch(ref):
                raise ValueError("evidence_refs must be source-document sha256 values")
        if self.positive_completeness_authorized and not self.verified_partition_refs:
            raise ValueError(
                "positive completeness must name the partition content it certifies"
            )
        return self


class Mbp1CompletenessCompilationReportEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "mbp1_completeness_compilation_report_id"

    mbp1_completeness_compilation_report_id: str = Field(pattern=SHA256_PATTERN)
    payload: Mbp1CompletenessCompilationReportPayload


class Mbp1PartitionGapManifestPayload(FrozenContract):
    """Partition-scope declared gaps (+ an optional COMPILED completeness claim)."""

    scope: Mbp1EvidenceScope
    #: the hash of the SOURCE document the declaration was compiled from —
    #: never the manifest artifact's own hash (non-self-referential)
    source_document_sha256: str = Field(pattern=SHA256_PATTERN)
    intervals: tuple[Mbp1DeclaredGapInterval, ...]
    completeness_assertion: Literal["none", "verified_complete_outside_intervals"]
    completeness_compilation_report_id: str | None = Field(
        default=None, pattern=SHA256_PATTERN
    )
    #: the compiled report's provenance, carried forward so the evidence
    #: wrapper can never relabel synthetic evidence as owner-reviewed
    provenance: Literal["owner_reviewed", "synthetic_fixture"]
    declared_by: str
    declared_at: str

    @model_validator(mode="after")
    def _positive_claim_is_compiled(self):
        if (
            self.completeness_assertion == "verified_complete_outside_intervals"
            and self.completeness_compilation_report_id is None
        ):
            raise ValueError(
                "a positive completeness assertion requires the verified "
                "completeness compilation report id (never a bare boolean)"
            )
        if self.scope.scope_level is Mbp1EvidenceScopeLevel.DATASET_DATE:
            raise ValueError("a gap manifest must be partition-scoped, not dataset-scoped")
        return self


class Mbp1PartitionGapManifestEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "mbp1_partition_gap_manifest_id"

    mbp1_partition_gap_manifest_id: str = Field(pattern=SHA256_PATTERN)
    payload: Mbp1PartitionGapManifestPayload


def compile_mbp1_partition_gap_manifest(
    report: Mbp1CompletenessCompilationReportEnvelope,
    *,
    source_document_sha256: str,
    declared_by: str,
    declared_at: str,
) -> Mbp1PartitionGapManifestEnvelope:
    """The ONLY route to a positive completeness assertion.

    The manifest inherits the report's compiled intervals and asserts
    ``verified_complete_outside_intervals`` iff the verified report
    authorized positive completeness; otherwise the assertion is ``none``
    (the intervals remain negative evidence).
    """

    payload = report.payload
    if source_document_sha256 == report.mbp1_completeness_compilation_report_id:
        raise ValueError(
            "source_document_sha256 must be the SOURCE document's hash, never the "
            "compilation report's own artifact id"
        )
    if source_document_sha256 not in payload.evidence_refs:
        raise ValueError(
            "source_document_sha256 must be one of the compilation report's "
            "evidence_refs — the manifest declares from a document the compiler consulted"
        )
    assertion: Literal["none", "verified_complete_outside_intervals"] = (
        "verified_complete_outside_intervals"
        if payload.positive_completeness_authorized
        else "none"
    )
    return Mbp1PartitionGapManifestEnvelope.from_payload(
        Mbp1PartitionGapManifestPayload(
            scope=payload.scope,
            source_document_sha256=source_document_sha256,
            intervals=payload.compiled_intervals,
            completeness_assertion=assertion,
            completeness_compilation_report_id=(
                report.mbp1_completeness_compilation_report_id
                if payload.positive_completeness_authorized
                else None
            ),
            provenance=payload.provenance,
            declared_by=declared_by,
            declared_at=declared_at,
        )
    )


class Mbp1PartitionEvidence(FrozenContract):
    """Everything the coverage computation may consult for ONE physical
    partition: its scope, the partition-scope manifest (if any), documented
    recovery boundaries, whether a verified channel map exists, and the
    dataset-condition record."""

    scope: Mbp1EvidenceScope
    gap_manifest: Mbp1PartitionGapManifestEnvelope | None
    #: the compilation report a positive claim was compiled from — REQUIRED
    #: whenever the manifest asserts completeness (review F1/S3); it must
    #: name the manifest's report id, scope, provenance, and intervals
    completeness_report: Mbp1CompletenessCompilationReportEnvelope | None = None
    recovery_boundaries: tuple[Mbp1RecoveryBoundary, ...] = ()
    channel_map_verified: bool = False
    dataset_condition: Mbp1DatasetConditionRecord | None = None
    provenance: Mbp1EvidenceProvenance = Mbp1EvidenceProvenance.NONE

    @model_validator(mode="after")
    def _scope_agrees(self):
        manifest = self.gap_manifest
        if manifest is None:
            if self.provenance is not Mbp1EvidenceProvenance.NONE:
                raise ValueError(
                    "a provenance label requires partition-scope evidence (a gap "
                    "manifest); an owner_reviewed label with no manifest is refused"
                )
            if self.completeness_report is not None:
                raise ValueError("a completeness report requires its gap manifest")
        else:
            # R6.1-FIX §3.9 (F-10C): FULL scope equality — every scope field,
            # not only the partition key and the UTC date
            if manifest.payload.scope != self.scope:
                raise ValueError(
                    "the gap manifest describes a different physical partition scope "
                    "(full scope equality is required); scope-mismatched evidence is refused"
                )
            if self.provenance is Mbp1EvidenceProvenance.NONE:
                raise ValueError("partition-scope evidence must declare its provenance")
            if self.provenance.value != manifest.payload.provenance:
                raise ValueError(
                    "the evidence provenance must equal the compiled manifest's "
                    f"provenance ({manifest.payload.provenance}); synthetic "
                    "evidence can never be relabeled owner-reviewed"
                )
            report_id = manifest.payload.completeness_compilation_report_id
            report = self.completeness_report
            if report_id is not None:
                if report is None:
                    raise ValueError(
                        "a manifest asserting completeness requires its compilation "
                        "report (the positive claim is compiled, never asserted)"
                    )
                if report.mbp1_completeness_compilation_report_id != report_id:
                    raise ValueError("the compilation report is not the one the manifest names")
                if not report.payload.positive_completeness_authorized:
                    raise ValueError(
                        "the manifest claims completeness but its compilation report "
                        "does not authorize positive completeness; refused"
                    )
                if report.payload.scope != manifest.payload.scope:
                    raise ValueError("the compilation report describes a different scope; refused")
                if report.payload.provenance != manifest.payload.provenance:
                    raise ValueError("the compilation report's provenance disagrees; refused")
                if report.payload.compiled_intervals != manifest.payload.intervals:
                    raise ValueError("the manifest intervals differ from the compiled intervals")
            elif report is not None:
                raise ValueError("the manifest names no compilation report; refused")
        if self.channel_map_verified and self.scope.channel_id is None:
            raise ValueError("a verified channel map requires the scope's channel_id")
        condition = self.dataset_condition
        if condition is not None and (
            condition.dataset != self.scope.dataset or condition.utc_date != self.scope.utc_date
        ):
            raise ValueError(
                "the dataset-condition record describes another dataset or UTC date; "
                "scope-mismatched evidence is refused"
            )
        for boundary in self.recovery_boundaries:
            if (
                boundary.utc_date != self.scope.utc_date
                or boundary.physical_partition_key != self.scope.physical_partition_key
            ):
                raise ValueError(
                    "a recovery boundary documented for another physical partition "
                    "cannot close this partition's uncertainty; refused"
                )
        return self


class Mbp1SequenceJumpDiagnostics(FrozenContract):
    """Raw venue sequence-number facts — DIAGNOSTIC ONLY (never an interval)."""

    semantics: Literal["sequence_jump_diagnostic_only_v2"] = SEQUENCE_JUMP_SEMANTICS_V2
    event_count: int = Field(ge=0)
    positive_jump_count: int = Field(ge=0)
    max_positive_jump: int = Field(ge=0)
    total_skipped_numbers: int = Field(ge=0)
    reset_count: int = Field(ge=0)


class Mbp1TsRecvGapDiagnostics(FrozenContract):
    """Receive-timestamp spacing facts — DIAGNOSTIC ONLY."""

    semantics: Literal["ts_recv_gap_diagnostic_only_v2"] = TS_RECV_GAP_SEMANTICS_V2
    max_gap_ns: int = Field(ge=0)
    gaps_over_60s: int = Field(ge=0)
    gaps_over_1s: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class Mbp1PartitionCoverageComputation:
    """The v2 coverage facts of one physical partition."""

    completeness_status: Mbp1CompletenessStatus
    intervals: tuple[Mbp1DeclaredGapInterval, ...]
    physical_expected_span_ns: int
    union_gap_ns: int
    coverage_fraction: float
    open_uncertainty_to_partition_end: bool
    dataset_condition_status: Mbp1DatasetConditionStatus
    evidence_sources: tuple[str, ...]
    evidence_scope_level: Mbp1EvidenceScopeLevel


def authorized_session_span_ns(trading_day: str) -> tuple[int, int]:
    """The authorized session span of one trading day in epoch ns:
    ``[18:00 ET the previous calendar day, 17:00 ET the trading day)`` —
    DST-aware via the ET zone (the closed window 17:00–18:00 is excluded)."""

    day = date.fromisoformat(trading_day)
    opened = datetime.combine(day - timedelta(days=1), _TRADING_DAY_BOUNDARY, tzinfo=_ET)
    closed = datetime.combine(day, _CLOSED_WINDOW_START, tzinfo=_ET)
    return int(pd.Timestamp(opened).value), int(pd.Timestamp(closed).value)


def merge_intervals(
    intervals: Iterable[tuple[int, int]], *, clip: tuple[int, int] | None = None
) -> tuple[tuple[int, int], ...]:
    """Union of half-open intervals: sorted, overlap-merged, optionally
    clipped to ``clip``. Empty or fully-clipped intervals vanish."""

    cleaned: list[tuple[int, int]] = []
    for start, end in intervals:
        start, end = int(start), int(end)
        if clip is not None:
            start, end = max(start, clip[0]), min(end, clip[1])
        if end > start:
            cleaned.append((start, end))
    cleaned.sort()
    merged: list[tuple[int, int]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def sequence_jump_diagnostics(normalized: pd.DataFrame) -> Mbp1SequenceJumpDiagnostics:
    """Positive jumps > 1 and resets (decreases) — counted, never typed."""

    count = int(len(normalized))
    if count < 2:
        return Mbp1SequenceJumpDiagnostics(
            event_count=count,
            positive_jump_count=0,
            max_positive_jump=0,
            total_skipped_numbers=0,
            reset_count=0,
        )
    steps = np.diff(normalized["sequence"].to_numpy(dtype=np.int64))
    jumps = steps[steps > 1]
    return Mbp1SequenceJumpDiagnostics(
        event_count=count,
        positive_jump_count=int(len(jumps)),
        max_positive_jump=int(jumps.max()) if len(jumps) else 0,
        total_skipped_numbers=int((jumps - 1).sum()) if len(jumps) else 0,
        reset_count=int((steps < 0).sum()),
    )


def ts_recv_gap_diagnostics(normalized: pd.DataFrame) -> Mbp1TsRecvGapDiagnostics:
    if len(normalized) < 2:
        return Mbp1TsRecvGapDiagnostics(max_gap_ns=0, gaps_over_60s=0, gaps_over_1s=0)
    ordered = np.sort(normalized["ts_recv"].to_numpy(dtype=np.int64))
    gaps = np.diff(ordered)
    return Mbp1TsRecvGapDiagnostics(
        max_gap_ns=int(gaps.max()) if len(gaps) else 0,
        gaps_over_60s=int((gaps > 60 * 1_000_000_000).sum()),
        gaps_over_1s=int((gaps > 1_000_000_000).sum()),
    )


def flag_counts(normalized: pd.DataFrame) -> dict[str, int]:
    """Per-bit DBN flag counts over a normalized frame (all diagnostic)."""

    if "flags" not in normalized.columns or normalized.empty:
        return {name: 0 for name in DBN_FLAG_BITS}
    flags = normalized["flags"].to_numpy(dtype=np.int64)
    return {name: int(((flags & bit) != 0).sum()) for name, bit in DBN_FLAG_BITS.items()}


def _flag_scope_level(evidence: Mbp1PartitionEvidence) -> Mbp1EvidenceScopeLevel:
    """Channel scope only behind a verified channel map on a scope that IS
    channel-scoped; else the conservative publisher/physical-partition
    scope. Never instrument."""

    if (
        evidence.channel_map_verified
        and evidence.scope.channel_id is not None
        and evidence.scope.scope_level is Mbp1EvidenceScopeLevel.CHANNEL_PARTITION
    ):
        return Mbp1EvidenceScopeLevel.CHANNEL_PARTITION
    return Mbp1EvidenceScopeLevel.PUBLISHER_PARTITION


def bad_book_uncertainty_intervals(
    normalized: pd.DataFrame, evidence: Mbp1PartitionEvidence
) -> tuple[Mbp1DeclaredGapInterval, ...]:
    """The uncertainty intervals opened by ``F_MAYBE_BAD_BOOK`` rows.

    Start: a manifest-declared start containing the flagged timestamp, else
    the immediately prior in-scope event when it is trustworthy (no
    bad-book / bad-ts_recv flag), else the partition's expected start.
    End: the first DOCUMENTED recovery boundary at or after the flagged
    timestamp, else the partition's expected end (fail closed). The next
    unflagged row never closes the interval.
    """

    if normalized.empty or "flags" not in normalized.columns:
        return ()
    scope = evidence.scope
    in_scope = normalized[
        (normalized["ts_event"] >= scope.partition_expected_start_ts)
        & (normalized["ts_event"] <= scope.partition_expected_end_ts)
    ]
    if in_scope.empty:
        return ()
    flags = in_scope["flags"].to_numpy(dtype=np.int64)
    ts = in_scope["ts_event"].to_numpy(dtype=np.int64)
    flagged_positions = np.flatnonzero((flags & F_MAYBE_BAD_BOOK_FLAG) != 0)
    if len(flagged_positions) == 0:
        return ()
    declared = (
        [
            (interval.start_ts, interval.end_ts)
            for interval in evidence.gap_manifest.payload.intervals
        ]
        if evidence.gap_manifest is not None
        else []
    )
    boundaries = sorted(
        (boundary.ts, boundary.kind) for boundary in evidence.recovery_boundaries
    )
    level = _flag_scope_level(evidence)
    out: list[Mbp1DeclaredGapInterval] = []
    for position in flagged_positions:
        flagged_ts = int(ts[position])
        declared_hit = next(
            (
                (start, end)
                for start, end in declared
                if start <= flagged_ts < end
            ),
            None,
        )
        if declared_hit is not None:
            start_ts, start_kind = declared_hit[0], Mbp1UncertaintyStartKind.MANIFEST_DECLARED_START
        elif position > 0 and (
            int(flags[position - 1]) & (F_MAYBE_BAD_BOOK_FLAG | F_BAD_TS_RECV_FLAG)
        ) == 0:
            start_ts = int(ts[position - 1])
            start_kind = Mbp1UncertaintyStartKind.LAST_TRUSTED_IN_SCOPE_EVENT
        else:
            start_ts = int(scope.partition_expected_start_ts)
            start_kind = Mbp1UncertaintyStartKind.PARTITION_EXPECTED_START
        recovery = next(
            (
                (b_ts, b_kind)
                for b_ts, b_kind in boundaries
                # a documented boundary closes the interval only when it lies
                # INSIDE this partition at or after the flagged instant (F6)
                if flagged_ts <= b_ts <= int(scope.partition_expected_end_ts)
            ),
            None,
        )
        if recovery is not None:
            end_ts, end_kind = int(recovery[0]), Mbp1RecoveryBoundaryKind(recovery[1])
        else:
            end_ts = int(scope.partition_expected_end_ts)
            end_kind = Mbp1RecoveryBoundaryKind.PARTITION_END_FAIL_CLOSED
        # the flagged record itself is always inside the uncertainty (F5): a
        # boundary AT the flagged instant closes AFTER the flagged event
        end_ts = max(end_ts, flagged_ts + 1, start_ts + 1)
        out.append(
            Mbp1DeclaredGapInterval(
                start_ts=start_ts,
                end_ts=end_ts,
                evidence_kind=Mbp1CoverageEvidenceKind.VENDOR_FLAG_MAYBE_BAD_BOOK,
                start_kind=start_kind,
                recovery_boundary_kind=end_kind,
                scope_level=level,
                contributing_evidence_kinds=(
                    Mbp1CoverageEvidenceKind.VENDOR_FLAG_MAYBE_BAD_BOOK.value,
                ),
            )
        )
    return tuple(out)


def _merged_typed_intervals(
    intervals: Sequence[Mbp1DeclaredGapInterval], *, clip: tuple[int, int]
) -> tuple[Mbp1DeclaredGapInterval, ...]:
    """Merge typed intervals once (overlaps collapse; the merged interval keeps
    the strongest fail-closed recovery kind and the earliest start kind)."""

    cleaned = [
        interval
        for interval in intervals
        if min(interval.end_ts, clip[1]) > max(interval.start_ts, clip[0])
    ]
    cleaned.sort(key=lambda interval: (interval.start_ts, interval.end_ts))
    merged: list[Mbp1DeclaredGapInterval] = []
    for interval in cleaned:
        start = max(interval.start_ts, clip[0])
        end = min(interval.end_ts, clip[1])
        own_kinds = interval.contributing_evidence_kinds or (interval.evidence_kind.value,)
        if merged and start <= merged[-1].end_ts:
            previous = merged[-1]
            fail_closed = Mbp1RecoveryBoundaryKind.PARTITION_END_FAIL_CLOSED
            kinds = (previous.recovery_boundary_kind, interval.recovery_boundary_kind)
            merged[-1] = previous.model_copy(
                update={
                    "end_ts": max(previous.end_ts, end),
                    "recovery_boundary_kind": (
                        fail_closed if fail_closed in kinds else previous.recovery_boundary_kind
                    ),
                    "merged": True,
                    "contributing_evidence_kinds": tuple(
                        sorted(set(previous.contributing_evidence_kinds) | set(own_kinds))
                    ),
                }
            )
        else:
            merged.append(
                interval.model_copy(
                    update={
                        "start_ts": start,
                        "end_ts": end,
                        "contributing_evidence_kinds": tuple(own_kinds),
                    }
                )
            )
    return tuple(merged)


def compute_partition_coverage(
    evidence: Mbp1PartitionEvidence,
    *,
    trading_day: str,
    flag_intervals: Sequence[Mbp1DeclaredGapInterval] = (),
    partition_content_refs: Sequence[str] | None = None,
) -> Mbp1PartitionCoverageComputation:
    """The D12 coverage rule for ONE physical partition.

    Denominator = ``intersection(verified partition span, authorized session
    span)``; intervals (declared + vendor-flag) are unioned, merged once, and
    clipped to that span. Positive completeness requires the manifest's
    compiled assertion AND (review F1) that the compilation report's
    ``verified_partition_refs`` name the partition content being certified
    (``partition_content_refs`` — the artifact builder always passes them;
    ``None`` is lawful only for pure interval diagnostics). A downgrading
    dataset condition, a missing partition-scope manifest, or an empty
    physical span is ``completeness_unknown``.
    """

    scope = evidence.scope
    manifest = evidence.gap_manifest
    positive_claim = (
        manifest is not None
        and manifest.payload.completeness_assertion == "verified_complete_outside_intervals"
    )
    if positive_claim:
        # R6.1-FIX §3.9 (F-10C): a positive claim requires the COMPLETE ordered
        # partition-content refs, and the compilation report's certified refs
        # must EQUAL them — never merely intersect, never an unbound claim
        if partition_content_refs is None:
            raise ValueError(
                "positive completeness claim requires the partition content refs it "
                "certifies (partition_content_refs=None is lawful for interval diagnostics "
                "only); refused"
            )
        if manifest.payload.scope != scope:
            raise ValueError(
                "positive completeness claim: the gap manifest scope is not the evidence "
                "scope (full scope equality is required); refused"
            )
        report = evidence.completeness_report
        certified = tuple(report.payload.verified_partition_refs) if report is not None else ()
        supplied = tuple(str(ref) for ref in partition_content_refs)
        if not supplied or set(certified) != set(supplied) or len(set(supplied)) != len(supplied):
            raise ValueError(
                "positive completeness claim does not reference the partition content it "
                "certifies exactly (verified_partition_refs must equal the complete "
                "partition content refs); refused"
            )
    session_start, session_end = authorized_session_span_ns(trading_day)
    span_start = max(int(scope.partition_expected_start_ts), session_start)
    span_end = min(int(scope.partition_expected_end_ts), session_end)
    expected_ns = max(0, span_end - span_start)
    condition = dataset_condition_status(evidence.dataset_condition)
    sources: list[str] = []
    typed: list[Mbp1DeclaredGapInterval] = list(flag_intervals)
    if manifest is not None:
        typed.extend(manifest.payload.intervals)
        sources.append(f"gap_manifest:{manifest.mbp1_partition_gap_manifest_id}")
        if manifest.payload.completeness_compilation_report_id:
            sources.append(
                "completeness_report:"
                f"{manifest.payload.completeness_compilation_report_id}"
            )
    if evidence.dataset_condition is not None:
        sources.append(
            f"dataset_condition:{evidence.dataset_condition.source_document_sha256}"
        )
    for boundary in evidence.recovery_boundaries:
        sources.append(f"recovery_boundary:{boundary.source_document_sha256}")
    if expected_ns == 0:
        return Mbp1PartitionCoverageComputation(
            completeness_status=Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN,
            intervals=(),
            physical_expected_span_ns=0,
            union_gap_ns=0,
            coverage_fraction=0.0,
            open_uncertainty_to_partition_end=False,
            dataset_condition_status=condition,
            evidence_sources=tuple(sources),
            evidence_scope_level=scope.scope_level,
        )
    merged = _merged_typed_intervals(typed, clip=(span_start, span_end))
    union_gap_ns = sum(interval.end_ts - interval.start_ts for interval in merged)
    coverage = max(0.0, min(1.0, 1.0 - union_gap_ns / expected_ns))
    open_to_end = any(
        interval.recovery_boundary_kind is Mbp1RecoveryBoundaryKind.PARTITION_END_FAIL_CLOSED
        for interval in merged
    )
    positive = positive_claim
    if condition in _DOWNGRADING_CONDITIONS or not positive:
        status = Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
        # fail closed: an unknown partition never reports a positive
        # coverage number (a consumer reading the fraction alone must not
        # fail open); the intervals stay as negative evidence
        coverage = 0.0
    elif merged:
        status = Mbp1CompletenessStatus.DECLARED_GAPS
    else:
        status = Mbp1CompletenessStatus.EVIDENCED_COMPLETE
    return Mbp1PartitionCoverageComputation(
        completeness_status=status,
        intervals=merged,
        physical_expected_span_ns=int(expected_ns),
        union_gap_ns=int(union_gap_ns),
        coverage_fraction=float(coverage),
        open_uncertainty_to_partition_end=open_to_end,
        dataset_condition_status=condition,
        evidence_sources=tuple(sources),
        evidence_scope_level=scope.scope_level,
    )


_STATUS_RANK = {
    Mbp1CompletenessStatus.EVIDENCED_COMPLETE: 0,
    Mbp1CompletenessStatus.DECLARED_GAPS: 1,
    Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN: 2,
}


def weighted_day_coverage(
    computations: Sequence[tuple[int, float, Mbp1CompletenessStatus]],
) -> tuple[float, Mbp1CompletenessStatus]:
    """Duration-weighted coverage over the physical partitions of ONE trading
    day (spans are never repeated); the day's status is the WEAKEST partition
    status. An empty or zero-span set is unknown."""

    total = sum(max(0, span) for span, _cov, _status in computations)
    if not computations or total == 0:
        return 0.0, Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
    coverage = sum(max(0, span) * cov for span, cov, _status in computations) / total
    status = max((status for _span, _cov, status in computations), key=_STATUS_RANK.get)
    return float(max(0.0, min(1.0, coverage))), status


def save_mbp1_completeness_report(
    root: Path, envelope: Mbp1CompletenessCompilationReportEnvelope
) -> tuple:
    return save_or_reuse_envelope(Path(root), MBP1_COMPLETENESS_REPORT_STORE, envelope)


def load_mbp1_completeness_report(
    root: Path, report_id: str
) -> Mbp1CompletenessCompilationReportEnvelope:
    return load_verified_envelope(
        Path(root),
        MBP1_COMPLETENESS_REPORT_STORE,
        report_id,
        Mbp1CompletenessCompilationReportEnvelope,
    )


def save_mbp1_partition_gap_manifest(
    root: Path, envelope: Mbp1PartitionGapManifestEnvelope
) -> tuple:
    return save_or_reuse_envelope(Path(root), MBP1_GAP_MANIFEST_STORE, envelope)


def load_mbp1_partition_gap_manifest(
    root: Path, manifest_id: str
) -> Mbp1PartitionGapManifestEnvelope:
    return load_verified_envelope(
        Path(root), MBP1_GAP_MANIFEST_STORE, manifest_id, Mbp1PartitionGapManifestEnvelope
    )


def load_verified_partition_evidence(
    root: Path,
    *,
    manifest_id: str,
    recovery_boundaries: tuple[Mbp1RecoveryBoundary, ...] = (),
    channel_map_verified: bool = False,
    dataset_condition: Mbp1DatasetConditionRecord | None = None,
) -> Mbp1PartitionEvidence:
    """Build partition evidence from STORE-VERIFIED artifacts only.

    The manifest is verified-loaded; when it claims positive completeness the
    named compilation report is verified-loaded too and must authorize that
    claim for the same scope and provenance — a manifest whose report is
    absent, tampered, unauthorized, or foreign is refused.
    """

    manifest = load_mbp1_partition_gap_manifest(Path(root), manifest_id)
    report_id = manifest.payload.completeness_compilation_report_id
    report = None
    if report_id is not None:
        report = load_mbp1_completeness_report(Path(root), report_id)
    # every agreement check (report id, scope, provenance, intervals,
    # authorization) is enforced by the evidence contract itself
    return Mbp1PartitionEvidence(
        scope=manifest.payload.scope,
        gap_manifest=manifest,
        completeness_report=report,
        recovery_boundaries=recovery_boundaries,
        channel_map_verified=channel_map_verified,
        dataset_condition=dataset_condition,
        provenance=Mbp1EvidenceProvenance(manifest.payload.provenance),
    )


def _example_scope() -> Mbp1EvidenceScope:
    return Mbp1EvidenceScope(
        scope_level=Mbp1EvidenceScopeLevel.PUBLISHER_PARTITION,
        dataset="GLBX.MDP3",
        publisher_id=1,
        channel_id=None,
        instrument_id=None,
        symbol=None,
        physical_partition_key="day_utc_date/mbp1",
        source_partition_id="databento/NQ/2026-01-13/mbp1",
        utc_date="2026-01-13",
        partition_expected_start_ts=1_768_312_800_000_000_000,
        partition_expected_end_ts=1_768_395_600_000_000_000,
        partition_span_source_id="example_span_v1",
    )


def _example_report() -> Mbp1CompletenessCompilationReportPayload:
    return Mbp1CompletenessCompilationReportPayload(
        scope=_example_scope(),
        source_inventory_id="example_inventory_v1",
        verified_partition_refs=("a" * 64,),
        evidence_refs=("b" * 64,),
        owner_review_decision_id="d" * 64,
        provenance="synthetic_fixture",
        compiled_intervals=(),
        positive_completeness_authorized=False,
        compiled_at="2026-08-28T00:00:00Z",
    )


def _example_manifest() -> Mbp1PartitionGapManifestPayload:
    return Mbp1PartitionGapManifestPayload(
        scope=_example_scope(),
        source_document_sha256="c" * 64,
        intervals=(),
        completeness_assertion="none",
        completeness_compilation_report_id=None,
        provenance="synthetic_fixture",
        declared_by="example",
        declared_at="2026-08-28T00:00:00Z",
    )


register_identity_pair(
    name="Mbp1CompletenessCompilationReport",
    envelope_cls=Mbp1CompletenessCompilationReportEnvelope,
    payload_cls=Mbp1CompletenessCompilationReportPayload,
    id_field="mbp1_completeness_compilation_report_id",
    example_factory=_example_report,
)
register_identity_pair(
    name="Mbp1PartitionGapManifest",
    envelope_cls=Mbp1PartitionGapManifestEnvelope,
    payload_cls=Mbp1PartitionGapManifestPayload,
    id_field="mbp1_partition_gap_manifest_id",
    example_factory=_example_manifest,
)
