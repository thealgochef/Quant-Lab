"""Two-path verification design (CONTRACTS_AND_SCHEMAS.md §6; TEST_MATRIX §1–§2).

Path A — the real five-day baseline vertical slice — is the ONLY real-data
verification: one canonical ≤5-trading-day allowlist for the complete
implementation-verification program, one exact baseline profile with its
profile-matching seed, ``verification_control_flow_gates_v1`` only. Path B —
everything multi-child — is synthetic. Research gates are never applied to
verification fixtures, and every verification report stamps
``verification_only`` / ``not_for_research_interpretation`` /
``full_pipeline_not_run``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from pydantic import Field, model_validator

from ..data_access import allowlist_sha256
from .authorization import SyntheticAuthorizationMarker, VerificationAuthorizationRef
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from .store import SEARCH_TEST_STORE_ROOT

__all__ = [
    "VERIFICATION_POLICY_ID",
    "PROPOSED_VERIFICATION_ALLOWLIST",
    "VERIFICATION_CONTROL_FLOW_GATE_IDS",
    "VerificationDataPolicy",
    "VerificationRunPayload",
    "VerificationRunEnvelope",
    "VerificationRunValidationError",
    "validate_verification_run",
    "ControlFlowGateReport",
    "evaluate_control_flow_gates",
    "verification_report_stamps",
    "register_program_allowlist",
    "DayCoverageRow",
    "CoverageMatrixPayload",
    "CoverageMatrixEnvelope",
    "build_verification_coverage_matrix",
]

VERIFICATION_POLICY_ID = "verification_fixed_allowlist_max5_v1"

#: Candidate allowlist (the five trading days inside 2026-06-04…06-10) —
#: PENDING coverage evidence and the owner's item-21/R-5 sign-off. It is a
#: proposal, never an authorization.
PROPOSED_VERIFICATION_ALLOWLIST: tuple[str, ...] = (
    "2026-06-04",
    "2026-06-05",
    "2026-06-08",
    "2026-06-09",
    "2026-06-10",
)

#: The NONRESEARCH gate policy applied to verification fixtures. Research
#: thresholds (≥30 trades, ≥20 days, …) are mathematically unreachable in five
#: days and are never applied here.
VERIFICATION_CONTROL_FLOW_GATE_IDS: tuple[str, ...] = (
    "replay_completed",
    "invariants_passed",
    "artifacts_published_and_reloaded",
    "neutrality_passed",
    "verifier_link_resolves",
    "zero_forbidden_counters",
)

_DEV_WINDOW_FIRST = "2026-01-01"
_DEV_WINDOW_LAST = "2026-06-10"

_MARKER_FILENAME = "VERIFICATION_ALLOWLIST_MARKER.json"


class VerificationDataPolicy(FrozenContract):
    """The frozen ≤5-day verification data policy (identity-bearing)."""

    policy_id: Literal["verification_fixed_allowlist_max5_v1"] = VERIFICATION_POLICY_ID
    allowlist: tuple[str, ...]
    allowlist_hash: str = Field(pattern=SHA256_PATTERN)
    max_real_days: Literal[5] = 5

    @model_validator(mode="after")
    def _lawful(self):
        days = self.allowlist
        if not days:
            raise ValueError("the verification allowlist cannot be empty")
        if len(days) > 5:
            raise ValueError("the verification allowlist admits at most five days")
        if days != tuple(sorted(days)) or len(days) != len(set(days)):
            raise ValueError("verification allowlist dates must be unique and chronological")
        for day in days:
            if not (_DEV_WINDOW_FIRST <= day <= _DEV_WINDOW_LAST):
                raise ValueError(
                    f"verification date {day} is outside the development window"
                )
        if self.allowlist_hash != allowlist_sha256(days):
            raise ValueError("allowlist_hash does not hash the allowlist")
        return self

    @classmethod
    def from_allowlist(cls, days: tuple[str, ...]) -> VerificationDataPolicy:
        return cls(allowlist=tuple(days), allowlist_hash=allowlist_sha256(days))


class VerificationRunPayload(FrozenContract):
    pipeline_semantic_id: str = Field(pattern=SHA256_PATTERN)
    verification_policy_id: Literal["verification_fixed_allowlist_max5_v1"] = (
        VERIFICATION_POLICY_ID
    )
    verification_authorization: VerificationAuthorizationRef
    allowlist: tuple[str, ...]
    allowlist_hash: str = Field(pattern=SHA256_PATTERN)
    seed_snapshot_id: str = Field(pattern=SHA256_PATTERN)
    baseline_profile_id: str
    baseline_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    coverage_matrix_artifact_id: str = Field(pattern=SHA256_PATTERN)
    output_namespace: Literal["search_test/v1"] = "search_test/v1"
    gate_policy_id: Literal["verification_control_flow_gates_v1"] = (
        "verification_control_flow_gates_v1"
    )


class VerificationRunEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "verification_run_id"

    verification_run_id: str = Field(pattern=SHA256_PATTERN)
    payload: VerificationRunPayload


class VerificationRunValidationError(PermissionError):
    """The real slice may not construct any source path (fail-before-path)."""


def validate_verification_run(
    envelope: VerificationRunEnvelope,
    *,
    expected_pipeline_semantic_id: str,
    expected_baseline_profile_id: str,
    expected_baseline_section_config_hash: str,
    expected_seed_snapshot_id: str,
    authorization: object,
) -> None:
    """Assert authorization, allowlist, seed, coverage, profile, and pipeline
    identity agree EXACTLY — before any source path is constructed (§6)."""

    payload = envelope.payload
    if isinstance(authorization, SyntheticAuthorizationMarker):
        raise VerificationRunValidationError(
            "the synthetic authorization marker is invalid for the real "
            "verification slice; a real VerificationAuthorizationRef is required"
        )
    if not isinstance(authorization, VerificationAuthorizationRef):
        raise VerificationRunValidationError(
            "the real verification slice requires a VerificationAuthorizationRef"
        )
    problems: list[str] = []
    if authorization != payload.verification_authorization:
        problems.append("authorization reference does not match the run payload")
    if authorization.verification_policy_id != payload.verification_policy_id:
        problems.append("authorization policy id mismatch")
    if authorization.approved_allowlist_hash != payload.allowlist_hash:
        problems.append("approved allowlist hash mismatch")
    if payload.allowlist_hash != allowlist_sha256(payload.allowlist):
        problems.append("allowlist hash does not hash the allowlist")
    if len(payload.allowlist) > 5:
        problems.append("allowlist exceeds five days")
    if authorization.seed_snapshot_id != payload.seed_snapshot_id:
        problems.append("seed snapshot mismatch")
    if authorization.coverage_matrix_artifact_id != payload.coverage_matrix_artifact_id:
        problems.append("coverage matrix mismatch")
    if payload.pipeline_semantic_id != expected_pipeline_semantic_id:
        problems.append("pipeline semantic identity mismatch")
    if payload.baseline_profile_id != expected_baseline_profile_id:
        problems.append("baseline profile mismatch")
    if payload.baseline_section_config_hash != expected_baseline_section_config_hash:
        problems.append("baseline section hash mismatch")
    if payload.seed_snapshot_id != expected_seed_snapshot_id:
        problems.append("seed snapshot does not match the runner's snapshot")
    if problems:
        raise VerificationRunValidationError(
            "verification run refused before source-path construction: "
            + "; ".join(problems)
        )


class ControlFlowGateReport(FrozenContract):
    gate_policy_id: Literal["verification_control_flow_gates_v1"] = (
        "verification_control_flow_gates_v1"
    )
    results: ImmutableMap[str, bool]
    passed: bool

    @model_validator(mode="after")
    def _complete(self):
        missing = sorted(set(VERIFICATION_CONTROL_FLOW_GATE_IDS) - set(self.results))
        extra = sorted(set(self.results) - set(VERIFICATION_CONTROL_FLOW_GATE_IDS))
        if missing or extra:
            raise ValueError(
                "control-flow gate report must cover exactly the registered gates "
                f"(missing={missing}, extra={extra})"
            )
        if self.passed != all(self.results.values()):
            raise ValueError("passed flag disagrees with the gate results")
        return self


def evaluate_control_flow_gates(results: dict[str, bool]) -> ControlFlowGateReport:
    unknown = sorted(set(results) - set(VERIFICATION_CONTROL_FLOW_GATE_IDS))
    if unknown:
        raise ValueError(f"unknown control-flow gates {unknown}")
    complete = {
        gate: bool(results.get(gate, False))
        for gate in VERIFICATION_CONTROL_FLOW_GATE_IDS
    }
    return ControlFlowGateReport(results=complete, passed=all(complete.values()))


def verification_report_stamps(
    *,
    allowlist: tuple[str, ...],
    synthetic_fixture_ids: tuple[str, ...] = (),
) -> dict:
    return {
        "verification_only": True,
        "not_for_research_interpretation": True,
        "full_pipeline_not_run": True,
        "real_date_count": len(allowlist),
        "real_date_allowlist_hash": allowlist_sha256(allowlist),
        "synthetic_fixture_ids": list(synthetic_fixture_ids),
    }


def register_program_allowlist(
    root: Path,
    policy: VerificationDataPolicy,
    *,
    canonical_root: Path | None = None,
) -> Path:
    """One canonical allowlist across the ENTIRE implementation-verification
    program (V3 P0-7). The first registration persists a marker; any later,
    different allowlist is refused regardless of release. Idempotent on match.

    A different ``root`` can never evade canonicality: when the CANONICAL
    program marker (under the repo's ``search_test/v1`` store) exists, every
    registration — whatever root it targets — must match it. ``canonical_root``
    is overridable only for tests; production callers leave the default.
    """

    root = Path(root)
    canonical = Path(canonical_root) if canonical_root is not None else SEARCH_TEST_STORE_ROOT
    canonical_marker = Path(canonical) / _MARKER_FILENAME
    if canonical_marker.exists():
        canonical_record = json.loads(canonical_marker.read_text(encoding="utf-8"))
        if canonical_record.get("allowlist_hash") != policy.allowlist_hash:
            raise PermissionError(
                "the canonical program allowlist marker already records a "
                "different allowlist; rotating or substituting dates is refused "
                "(one canonical allowlist, V3 P0-7)"
            )
    root.mkdir(parents=True, exist_ok=True)
    marker_path = root / _MARKER_FILENAME
    record = {
        "verification_policy_id": policy.policy_id,
        "allowlist": list(policy.allowlist),
        "allowlist_hash": policy.allowlist_hash,
    }
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing.get("allowlist_hash") != policy.allowlist_hash:
            raise PermissionError(
                "a different verification allowlist is already registered for this "
                "implementation-verification program; rotating or substituting "
                "dates is refused (one canonical allowlist, V3 P0-7)"
            )
        return marker_path
    temporary = marker_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker_path)
    return marker_path


# ─────────────────────────────────────────────────────────────────────────────
# Coverage matrix — built ONLY from already-authorized existing artifacts
# ─────────────────────────────────────────────────────────────────────────────


class DayCoverageRow(FrozenContract):
    trading_day: str
    source_partition_recorded: bool
    setup_lifecycle_rows: int
    entry_candidate_rows: int
    eligible_decision_rows: int
    executed_trade_rows: int
    candidate_label_rows: int
    audit_event_rows: int | None
    replay_chart_available: bool | None
    mbp1_coverage: Literal["not_evaluated"] = "not_evaluated"


class CoverageMatrixPayload(FrozenContract):
    verification_policy_id: Literal["verification_fixed_allowlist_max5_v1"] = (
        VERIFICATION_POLICY_ID
    )
    evidence_source_dataset_id: str = Field(pattern=SHA256_PATTERN)
    evidence_source_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_allowlist: tuple[str, ...]
    candidate_allowlist_hash: str = Field(pattern=SHA256_PATTERN)
    rows: tuple[DayCoverageRow, ...]
    lifecycle_paths_covered: ImmutableMap[str, bool]
    uncovered_paths_note: str

    @model_validator(mode="after")
    def _consistent(self):
        if self.candidate_allowlist_hash != allowlist_sha256(self.candidate_allowlist):
            raise ValueError("candidate allowlist hash mismatch")
        row_days = tuple(row.trading_day for row in self.rows)
        if row_days != self.candidate_allowlist:
            raise ValueError("coverage rows must cover exactly the candidate allowlist")
        return self


class CoverageMatrixEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "coverage_matrix_id"

    coverage_matrix_id: str = Field(pattern=SHA256_PATTERN)
    payload: CoverageMatrixPayload


def _day_counts(frame: pd.DataFrame, day: str) -> int:
    if frame.empty or "envelope_trading_day" not in frame:
        return 0
    return int((frame["envelope_trading_day"].astype(str) == day).sum())


def build_verification_coverage_matrix(
    *,
    candidate_allowlist: tuple[str, ...],
    v2_tables: dict,
    evidence_source_dataset_id: str,
    evidence_source_manifest_sha256: str,
    permitted_source_days: tuple[str, ...],
    audit_day_counts: dict[str, int] | None = None,
    replay_chart_days: tuple[str, ...] | None = None,
) -> CoverageMatrixEnvelope:
    """Score the candidate window from already-authorized artifact evidence.

    No raw-source discovery occurs: source availability comes from the
    accepted dataset's ``permitted_source_hashes`` day keys, funnel evidence
    from its typed tables, audit/chart coverage from existing companions.
    """

    from ..contracts import RecordTable  # noqa: PLC0415

    rows = []
    for day in candidate_allowlist:
        rows.append(
            DayCoverageRow(
                trading_day=day,
                source_partition_recorded=day in permitted_source_days,
                setup_lifecycle_rows=_day_counts(
                    v2_tables.get(RecordTable.SETUP_LIFECYCLE, pd.DataFrame()), day
                ),
                entry_candidate_rows=_day_counts(
                    v2_tables.get(RecordTable.ENTRY_CANDIDATE, pd.DataFrame()), day
                ),
                eligible_decision_rows=_day_counts(
                    v2_tables.get(RecordTable.ELIGIBLE_DECISION, pd.DataFrame()), day
                ),
                executed_trade_rows=_day_counts(
                    v2_tables.get(RecordTable.EXECUTED_TRADE, pd.DataFrame()), day
                ),
                candidate_label_rows=_day_counts(
                    v2_tables.get(RecordTable.CANDIDATE_LABEL, pd.DataFrame()), day
                ),
                audit_event_rows=(
                    None if audit_day_counts is None else int(audit_day_counts.get(day, 0))
                ),
                replay_chart_available=(
                    None if replay_chart_days is None else day in replay_chart_days
                ),
            )
        )
    lifecycle_paths = {
        "setup_activation": any(row.setup_lifecycle_rows for row in rows),
        "entry_candidate": any(row.entry_candidate_rows for row in rows),
        "eligible_decision": any(row.eligible_decision_rows for row in rows),
        "execution_resolution": any(row.executed_trade_rows for row in rows),
        "candidate_labeling": any(row.candidate_label_rows for row in rows),
        "audit_events": any((row.audit_event_rows or 0) for row in rows),
        "replay_chart": any(bool(row.replay_chart_available) for row in rows),
    }
    uncovered = sorted(path for path, covered in lifecycle_paths.items() if not covered)
    payload = CoverageMatrixPayload(
        evidence_source_dataset_id=evidence_source_dataset_id,
        evidence_source_manifest_sha256=evidence_source_manifest_sha256,
        candidate_allowlist=tuple(candidate_allowlist),
        candidate_allowlist_hash=allowlist_sha256(candidate_allowlist),
        rows=tuple(rows),
        lifecycle_paths_covered=lifecycle_paths,
        uncovered_paths_note=(
            "all scored lifecycle paths are covered by the candidate window"
            if not uncovered
            else "synthetic fixtures must cover: " + ", ".join(uncovered)
        ),
    )
    return CoverageMatrixEnvelope.from_payload(payload)


def _example_verification_run() -> VerificationRunPayload:
    policy = VerificationDataPolicy.from_allowlist(PROPOSED_VERIFICATION_ALLOWLIST)
    return VerificationRunPayload(
        pipeline_semantic_id="a" * 64,
        verification_authorization=VerificationAuthorizationRef(
            verification_policy_id=VERIFICATION_POLICY_ID,
            approved_allowlist_hash=policy.allowlist_hash,
            coverage_matrix_artifact_id="b" * 64,
            seed_snapshot_id="c" * 64,
            approved_by="owner@example",
            approved_at="2026-08-18T00:00:00Z",
            content_hash="d" * 64,
        ),
        allowlist=policy.allowlist,
        allowlist_hash=policy.allowlist_hash,
        seed_snapshot_id="c" * 64,
        baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
        baseline_section_config_hash="e" * 64,
        coverage_matrix_artifact_id="b" * 64,
    )


def _example_coverage_matrix() -> CoverageMatrixPayload:
    return CoverageMatrixPayload(
        evidence_source_dataset_id="1" * 64,
        evidence_source_manifest_sha256="2" * 64,
        candidate_allowlist=("2026-06-04",),
        candidate_allowlist_hash=allowlist_sha256(("2026-06-04",)),
        rows=(
            DayCoverageRow(
                trading_day="2026-06-04",
                source_partition_recorded=True,
                setup_lifecycle_rows=1,
                entry_candidate_rows=1,
                eligible_decision_rows=0,
                executed_trade_rows=0,
                candidate_label_rows=1,
                audit_event_rows=None,
                replay_chart_available=None,
            ),
        ),
        lifecycle_paths_covered={
            "setup_activation": True,
            "entry_candidate": True,
            "eligible_decision": False,
            "execution_resolution": False,
            "candidate_labeling": True,
            "audit_events": False,
            "replay_chart": False,
        },
        uncovered_paths_note="synthetic fixtures must cover: audit_events, …",
    )


register_identity_pair(
    name="VerificationRun",
    envelope_cls=VerificationRunEnvelope,
    payload_cls=VerificationRunPayload,
    id_field="verification_run_id",
    example_factory=_example_verification_run,
)
register_identity_pair(
    name="CoverageMatrix",
    envelope_cls=CoverageMatrixEnvelope,
    payload_cls=CoverageMatrixPayload,
    id_field="coverage_matrix_id",
    example_factory=_example_coverage_matrix,
)
