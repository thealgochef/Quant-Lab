"""Bounded real-data MBP-1 coverage diagnostic (R5B.1; owner Q1 item 6).

The diagnostic CHARACTERIZES the actual partition-quality evidence of the
one canonical authorized ≤5-day fixture — per physical partition: row
counts, DBN flag counts, sequence-jump and ``ts_recv``-gap distributions,
which verified partition-scope manifests / compilation reports / condition
records were loaded, the resulting policy-v2 completeness status, and — ONLY
when partition-scope evidence exists — the open-interval facts. It writes an
immutable :class:`Mbp1CoverageDiagnosticReportEnvelope` into the
verification namespace and never infers completeness from raw sequence
continuity (a structural ``Literal[False]``).

The REAL run is an owner action. Its gate (:func:`assert_diagnostic_authorized`)
is the R1 real-slice gate, not a weaker mirror (adversarial review S2/F9):
a persisted, verified ``VerificationRunEnvelope`` whose
``VerificationAuthorizationRef`` binds the requested allowlist
(``allowlist_hash == sha256(allowlist) == approved_allowlist_hash``), the
verification policy identity, the ``search_test/v1`` namespace, a
verified-loaded coverage-matrix artifact over exactly those days, and the
ONE canonical program allowlist (``register_program_allowlist``); the access
policy must BE a ``VerificationReplayPolicy`` over the same days. Everything
is checked before any source path is constructed. The synthetic path proves
the report shape on fixture artifacts only.

Partition-scope evidence enters the real run ONLY through store-verified
manifests (``load_partition_evidence_manifest`` → ``load_verified_partition_evidence``);
without evidence every partition is ``completeness_unknown`` and its
open-uncertainty fact is ``None`` (not computed), never a misleading ``False``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..data_access import allowlist_sha256
from ..development_access import VerificationReplayPolicy
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from ..search.store import load_verified_envelope, save_or_reuse_envelope
from .mbp1_coverage_evidence import (
    MBP1_COVERAGE_POLICY_V2,
    Mbp1CompletenessStatus,
    Mbp1DatasetConditionRecord,
    Mbp1DatasetConditionStatus,
    Mbp1EvidenceScopeLevel,
    Mbp1PartitionEvidence,
    Mbp1RecoveryBoundary,
    assert_evidence_date_representable,
    load_verified_partition_evidence,
)
from .mbp1_source_artifact import Mbp1PartitionCoverage, Mbp1SourceArtifactEnvelope

__all__ = [
    "MBP1_COVERAGE_DIAGNOSTIC_STORE",
    "Mbp1PartitionDiagnosticRow",
    "Mbp1CoverageDiagnosticReportPayload",
    "Mbp1CoverageDiagnosticReportEnvelope",
    "build_mbp1_coverage_diagnostic",
    "assert_diagnostic_authorized",
    "assert_verification_namespace",
    "load_partition_evidence_manifest",
    "save_mbp1_coverage_diagnostic",
    "load_mbp1_coverage_diagnostic",
]

MBP1_COVERAGE_DIAGNOSTIC_STORE = "mbp1_coverage_diagnostics"


class Mbp1PartitionDiagnosticRow(FrozenContract):
    trading_day: str
    source_partition_utc_date: str
    relative_logical_partition_key: str
    row_count: int = Field(ge=0)
    evidence_scope_level: Mbp1EvidenceScopeLevel | None
    #: whether any partition-scope evidence was LOADED for this partition;
    #: every interval fact below is ``None``/zero when it was not
    partition_scope_evidence_present: bool
    completeness_status: Mbp1CompletenessStatus
    dataset_condition_status: Mbp1DatasetConditionStatus
    declared_gap_count: int = Field(ge=0)
    declared_gap_ns: int = Field(ge=0)
    physical_expected_span_ns: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    #: tri-state: ``None`` = not computed (no partition-scope evidence)
    open_uncertainty_to_partition_end: bool | None
    gap_manifest_present: bool
    completeness_report_present: bool
    dataset_condition_present: bool
    rows_outside_session_span: int = Field(ge=0)
    rows_after_development_cutoff: int = Field(ge=0)
    flag_counts: ImmutableMap[str, int]
    sequence_positive_jump_count: int = Field(ge=0)
    sequence_max_positive_jump: int = Field(ge=0)
    sequence_reset_count: int = Field(ge=0)
    ts_recv_max_gap_ns: int = Field(ge=0)
    ts_recv_gaps_over_1s: int = Field(ge=0)
    ts_recv_gaps_over_60s: int = Field(ge=0)

    @model_validator(mode="after")
    def _representable(self):
        assert_evidence_date_representable(self.trading_day)
        assert_evidence_date_representable(self.source_partition_utc_date)
        if (self.open_uncertainty_to_partition_end is None) != (
            not self.partition_scope_evidence_present
        ):
            raise ValueError(
                "open_uncertainty_to_partition_end is computed exactly when "
                "partition-scope evidence is present"
            )
        return self


class Mbp1CoverageDiagnosticReportPayload(FrozenContract):
    mbp1_source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    run_scope: Literal["synthetic_fixture", "verification_5d"]
    verification_policy_id: str
    allowlist: tuple[str, ...]
    allowlist_hash: str = Field(pattern=SHA256_PATTERN)
    authorization_content_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    #: the verified manifest ids the run loaded (empty = no evidence seam used)
    partition_evidence_manifest_ids: tuple[str, ...] = ()
    coverage_policy_id: Literal["mbp1_source_coverage_declared_evidence_v2"] = (
        MBP1_COVERAGE_POLICY_V2
    )
    rows: tuple[Mbp1PartitionDiagnosticRow, ...]
    #: STRUCTURAL: this report characterizes evidence; it never infers
    #: completeness from sequence continuity and never authorizes anything
    completeness_inferred_from_sequence_continuity: Literal[False] = False
    research_boundary: Literal["research_only_offline"] = "research_only_offline"

    @model_validator(mode="after")
    def _lawful(self):
        for day in self.allowlist:
            assert_evidence_date_representable(day)
        if self.allowlist_hash != allowlist_sha256(self.allowlist):
            raise ValueError("allowlist_hash does not hash the allowlist")
        if self.run_scope == "verification_5d" and not self.authorization_content_hash:
            raise ValueError("a verification-scope diagnostic requires the authorization hash")
        for manifest_id in self.partition_evidence_manifest_ids:
            if len(manifest_id) != 64:
                raise ValueError("partition_evidence_manifest_ids must be 64-hex store ids")
        return self


class Mbp1CoverageDiagnosticReportEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "mbp1_coverage_diagnostic_id"

    mbp1_coverage_diagnostic_id: str = Field(pattern=SHA256_PATTERN)
    payload: Mbp1CoverageDiagnosticReportPayload


def _row(partition: Mbp1PartitionCoverage) -> Mbp1PartitionDiagnosticRow:
    sources = partition.evidence_sources
    scoped = partition.evidence_scope is not None
    return Mbp1PartitionDiagnosticRow(
        trading_day=partition.trading_day,
        source_partition_utc_date=partition.source_partition_utc_date,
        relative_logical_partition_key=partition.relative_logical_partition_key,
        row_count=partition.row_count,
        evidence_scope_level=(
            partition.evidence_scope.scope_level if partition.evidence_scope else None
        ),
        partition_scope_evidence_present=scoped,
        completeness_status=partition.completeness_status,
        dataset_condition_status=partition.dataset_condition_status,
        declared_gap_count=len(partition.declared_gap_intervals),
        declared_gap_ns=partition.union_gap_ns,
        physical_expected_span_ns=partition.physical_expected_span_ns,
        coverage_fraction=partition.coverage_fraction,
        open_uncertainty_to_partition_end=(
            partition.open_uncertainty_to_partition_end if scoped else None
        ),
        gap_manifest_present=any(s.startswith("gap_manifest:") for s in sources),
        completeness_report_present=any(
            s.startswith("completeness_report:") for s in sources
        ),
        dataset_condition_present=any(
            s.startswith("dataset_condition:") for s in sources
        ),
        rows_outside_session_span=partition.rows_outside_session_span,
        rows_after_development_cutoff=partition.rows_after_development_cutoff,
        flag_counts=partition.flag_counts,
        sequence_positive_jump_count=partition.sequence_jump_diagnostics.positive_jump_count,
        sequence_max_positive_jump=partition.sequence_jump_diagnostics.max_positive_jump,
        sequence_reset_count=partition.sequence_jump_diagnostics.reset_count,
        ts_recv_max_gap_ns=partition.ts_recv_gap_diagnostics.max_gap_ns,
        ts_recv_gaps_over_1s=partition.ts_recv_gap_diagnostics.gaps_over_1s,
        ts_recv_gaps_over_60s=partition.ts_recv_gap_diagnostics.gaps_over_60s,
    )


def assert_verification_namespace(store_root: Path) -> Path:
    """The store root must BE a ``search_test/v1`` namespace (its last two
    path components) — a substring can never satisfy it (review S4)."""

    root = Path(store_root)
    if tuple(root.parts[-2:]) != ("search_test", "v1"):
        raise PermissionError(
            "the MBP-1 coverage diagnostic writes only into a verification "
            "namespace whose path ends in search_test/v1; refused"
        )
    return root


def assert_diagnostic_authorized(
    *,
    store_root: Path,
    run_envelope,
    access_policy,
    allowlist: Iterable[str],
    canonical_root: Path | None = None,
) -> str:
    """The R1 real-slice gate for the diagnostic (fail-before-path).

    Refuses — before any source path — an absent run envelope, a policy
    that is not a ``VerificationReplayPolicy`` over exactly ``allowlist``,
    a run envelope whose allowlist / hash / policy / namespace / authorization
    disagree with the request, an authorization whose coverage-matrix
    artifact is not a verified store entry over exactly these days, and any
    allowlist other than the ONE canonical program allowlist.
    """

    from ..search.verification import (  # noqa: PLC0415
        VERIFICATION_POLICY_ID,
        CoverageMatrixEnvelope,
        VerificationDataPolicy,
        VerificationRunEnvelope,
        register_program_allowlist,
    )

    days = tuple(str(day) for day in allowlist)
    root = assert_verification_namespace(Path(store_root))
    if not isinstance(access_policy, VerificationReplayPolicy):
        raise PermissionError(
            "the MBP-1 coverage diagnostic runs only under VerificationReplayPolicy "
            "(fail-before-path); no other access policy is accepted"
        )
    if run_envelope is None:
        raise PermissionError(
            "the real MBP-1 coverage diagnostic requires the owner's persisted "
            "VerificationRunEnvelope + VerificationAuthorizationRef; it does not "
            "exist — refused before any source path is constructed"
        )
    if not isinstance(run_envelope, VerificationRunEnvelope):
        raise PermissionError("the run envelope is not a VerificationRunEnvelope; refused")
    payload = run_envelope.payload
    authorization = payload.verification_authorization
    problems: list[str] = []
    if tuple(payload.allowlist) != days:
        problems.append("run envelope allowlist differs from the requested days")
    if payload.allowlist_hash != allowlist_sha256(days):
        problems.append("run envelope allowlist_hash does not hash its allowlist")
    if authorization.approved_allowlist_hash != allowlist_sha256(days):
        problems.append("the VerificationAuthorizationRef approves a different allowlist")
    if payload.verification_policy_id != VERIFICATION_POLICY_ID or (
        authorization.verification_policy_id != VERIFICATION_POLICY_ID
    ):
        problems.append("verification policy identity mismatch")
    if payload.output_namespace != "search_test/v1":
        problems.append("run envelope output namespace is not search_test/v1")
    if payload.coverage_matrix_artifact_id != authorization.coverage_matrix_artifact_id:
        problems.append("coverage matrix artifact mismatch between run and authorization")
    if tuple(getattr(access_policy, "allowed_dates", ())) != days:
        problems.append("the access policy's allowlist differs from the requested days")
    if problems:
        raise PermissionError(
            "diagnostic authorization refused before any source path: "
            + "; ".join(problems)
        )
    # HARDENING-BACKEND §4.1 / §4.2 (adversarial RA-01): the pathname rule
    # above is defense in depth only — authority is the store's VERIFIED
    # ``test`` namespace the authorization names, coherently deployed, with
    # the CURRENT supersession head equal to the signed witness; refused
    # before the coverage matrix loads, before the program allowlist is
    # registered, before any source path exists.
    from ..search.authorization import (  # noqa: PLC0415
        AuthorizationError,
        assert_authorization_bound_to_store,
    )

    try:
        assert_authorization_bound_to_store(
            root,
            store_namespace_id=authorization.store_namespace_id,
            supersession_head_witness=authorization.supersession_head_witness,
            expected_namespace_class="test",
        )
    except AuthorizationError as error:
        raise PermissionError(
            f"diagnostic authorization refused before any source path: {error}"
        ) from error
    try:
        matrix = load_verified_envelope(
            root,
            "coverage_matrices",
            authorization.coverage_matrix_artifact_id,
            CoverageMatrixEnvelope,
        )
    except Exception as error:
        raise PermissionError(
            "the authorization's coverage-matrix artifact is not a verified store "
            "entry; refused before any source path"
        ) from error
    if tuple(matrix.payload.candidate_allowlist) != days:
        raise PermissionError(
            "the coverage-matrix artifact covers a different allowlist; refused "
            "before any source path"
        )
    register_program_allowlist(
        root, VerificationDataPolicy.from_allowlist(days), canonical_root=canonical_root
    )
    return authorization.content_hash


def load_partition_evidence_manifest(
    store_root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, tuple[Mbp1PartitionEvidence, ...]], tuple[str, ...]]:
    """``{trading_day: [{manifest_id, channel_map_verified, recovery_boundaries,
    dataset_condition}, ...]}`` → store-VERIFIED partition evidence per day.

    Every manifest is verified-loaded (with its compilation report when it
    claims completeness); recovery boundaries and condition records are typed
    records whose scope the evidence contract checks. Nothing here is a path.
    """

    evidence: dict[str, tuple[Mbp1PartitionEvidence, ...]] = {}
    manifest_ids: list[str] = []
    for day in sorted(manifest):
        entries = manifest[day]
        loaded: list[Mbp1PartitionEvidence] = []
        for entry in entries:
            manifest_id = str(entry["manifest_id"])
            boundaries = tuple(
                Mbp1RecoveryBoundary.model_validate(item)
                for item in (entry.get("recovery_boundaries") or ())
            )
            condition = entry.get("dataset_condition")
            loaded.append(
                load_verified_partition_evidence(
                    Path(store_root),
                    manifest_id=manifest_id,
                    recovery_boundaries=boundaries,
                    channel_map_verified=bool(entry.get("channel_map_verified", False)),
                    dataset_condition=(
                        Mbp1DatasetConditionRecord.model_validate(condition)
                        if condition
                        else None
                    ),
                )
            )
            manifest_ids.append(manifest_id)
        evidence[str(day)] = tuple(loaded)
    return evidence, tuple(sorted(set(manifest_ids)))


def build_mbp1_coverage_diagnostic(
    source: Mbp1SourceArtifactEnvelope,
    *,
    run_scope: Literal["synthetic_fixture", "verification_5d"],
    allowlist: Iterable[str],
    authorization_content_hash: str | None,
    partition_evidence_manifest_ids: tuple[str, ...] = (),
    verification_policy_id: str = "verification_fixed_allowlist_max5_v1",
) -> Mbp1CoverageDiagnosticReportEnvelope:
    days = tuple(allowlist)
    if run_scope == "verification_5d" and not authorization_content_hash:
        raise PermissionError(
            "a verification-scope diagnostic requires the verified authorization hash"
        )
    payload = Mbp1CoverageDiagnosticReportPayload(
        mbp1_source_artifact_id=source.mbp1_source_artifact_id,
        run_scope=run_scope,
        verification_policy_id=verification_policy_id,
        allowlist=days,
        allowlist_hash=allowlist_sha256(days),
        authorization_content_hash=authorization_content_hash,
        partition_evidence_manifest_ids=tuple(partition_evidence_manifest_ids),
        rows=tuple(_row(partition) for partition in source.payload.ordered_partitions),
    )
    return Mbp1CoverageDiagnosticReportEnvelope.from_payload(payload)


def save_mbp1_coverage_diagnostic(
    root: Path, envelope: Mbp1CoverageDiagnosticReportEnvelope
) -> tuple:
    return save_or_reuse_envelope(Path(root), MBP1_COVERAGE_DIAGNOSTIC_STORE, envelope)


def load_mbp1_coverage_diagnostic(
    root: Path, diagnostic_id: str
) -> Mbp1CoverageDiagnosticReportEnvelope:
    return load_verified_envelope(
        Path(root),
        MBP1_COVERAGE_DIAGNOSTIC_STORE,
        diagnostic_id,
        Mbp1CoverageDiagnosticReportEnvelope,
    )


def _example_payload() -> Mbp1CoverageDiagnosticReportPayload:
    return Mbp1CoverageDiagnosticReportPayload(
        mbp1_source_artifact_id="a" * 64,
        run_scope="synthetic_fixture",
        verification_policy_id="verification_fixed_allowlist_max5_v1",
        allowlist=("2026-01-13",),
        allowlist_hash=allowlist_sha256(("2026-01-13",)),
        authorization_content_hash=None,
        rows=(),
    )


register_identity_pair(
    name="Mbp1CoverageDiagnosticReport",
    envelope_cls=Mbp1CoverageDiagnosticReportEnvelope,
    payload_cls=Mbp1CoverageDiagnosticReportPayload,
    id_field="mbp1_coverage_diagnostic_id",
    example_factory=_example_payload,
)

_ = json  # the CLI parses the evidence manifest; kept importable here
