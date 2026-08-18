"""Verification policy / run / coverage suites (TEST_MATRIX §1–§2, §3.8–§3.9)."""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.development_access import VerificationReplayPolicy
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    SyntheticAuthorizationMarker,
    VerificationAuthorizationRef,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    PROPOSED_VERIFICATION_ALLOWLIST,
    VERIFICATION_POLICY_ID,
    VerificationDataPolicy,
    VerificationRunEnvelope,
    VerificationRunValidationError,
    _example_verification_run,
    build_verification_coverage_matrix,
    evaluate_control_flow_gates,
    register_program_allowlist,
    validate_verification_run,
    verification_report_stamps,
)


def test_proposed_allowlist_is_five_trading_days_in_the_candidate_window() -> None:
    assert len(PROPOSED_VERIFICATION_ALLOWLIST) == 5
    assert PROPOSED_VERIFICATION_ALLOWLIST[0] >= "2026-06-04"
    assert PROPOSED_VERIFICATION_ALLOWLIST[-1] <= "2026-06-10"
    VerificationDataPolicy.from_allowlist(PROPOSED_VERIFICATION_ALLOWLIST)


def test_policy_rejects_sixth_and_forbidden_dates_before_any_path() -> None:
    with pytest.raises(PermissionError, match="at most five"):
        VerificationReplayPolicy(
            ("2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09")
        )
    with pytest.raises(PermissionError, match="before path construction"):
        VerificationReplayPolicy(("2026-06-11",))
    with pytest.raises(PermissionError, match="before path construction"):
        VerificationReplayPolicy(("2026-06-12",))
    with pytest.raises(PermissionError, match="before path construction"):
        VerificationReplayPolicy(("2026-07-01",))
    policy = VerificationReplayPolicy(PROPOSED_VERIFICATION_ALLOWLIST)
    with pytest.raises(PermissionError):
        policy.authorize_date("2026-06-03")  # off-allowlist inside the window
    assert policy.audit.denied_dates.get("2026-06-03") == 1


def test_program_allowlist_marker_refuses_rotation(tmp_path) -> None:
    policy = VerificationDataPolicy.from_allowlist(PROPOSED_VERIFICATION_ALLOWLIST)
    register_program_allowlist(tmp_path, policy)
    # idempotent on the SAME allowlist
    register_program_allowlist(tmp_path, policy)
    rotated = VerificationDataPolicy.from_allowlist(
        ("2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-06-01")
    )
    with pytest.raises(PermissionError, match="one canonical allowlist"):
        register_program_allowlist(tmp_path, rotated)


def test_verification_run_refuses_synthetic_marker() -> None:
    envelope = VerificationRunEnvelope.from_payload(_example_verification_run())
    with pytest.raises(VerificationRunValidationError, match="synthetic"):
        validate_verification_run(
            envelope,
            expected_pipeline_semantic_id=envelope.payload.pipeline_semantic_id,
            expected_baseline_profile_id=envelope.payload.baseline_profile_id,
            expected_baseline_section_config_hash=envelope.payload.baseline_section_config_hash,
            expected_seed_snapshot_id=envelope.payload.seed_snapshot_id,
            authorization=SyntheticAuthorizationMarker(),
        )


def test_verification_run_requires_exact_binding() -> None:
    payload = _example_verification_run()
    envelope = VerificationRunEnvelope.from_payload(payload)
    good = payload.verification_authorization
    validate_verification_run(
        envelope,
        expected_pipeline_semantic_id=payload.pipeline_semantic_id,
        expected_baseline_profile_id=payload.baseline_profile_id,
        expected_baseline_section_config_hash=payload.baseline_section_config_hash,
        expected_seed_snapshot_id=payload.seed_snapshot_id,
        authorization=good,
    )
    mismatched_allowlist = VerificationAuthorizationRef(
        verification_policy_id=VERIFICATION_POLICY_ID,
        approved_allowlist_hash="9" * 64,
        coverage_matrix_artifact_id=good.coverage_matrix_artifact_id,
        seed_snapshot_id=good.seed_snapshot_id,
        approved_by=good.approved_by,
        approved_at=good.approved_at,
        content_hash=good.content_hash,
    )
    with pytest.raises(VerificationRunValidationError, match="before source-path"):
        validate_verification_run(
            envelope,
            expected_pipeline_semantic_id=payload.pipeline_semantic_id,
            expected_baseline_profile_id=payload.baseline_profile_id,
            expected_baseline_section_config_hash=payload.baseline_section_config_hash,
            expected_seed_snapshot_id=payload.seed_snapshot_id,
            authorization=mismatched_allowlist,
        )
    with pytest.raises(VerificationRunValidationError, match="pipeline semantic"):
        validate_verification_run(
            envelope,
            expected_pipeline_semantic_id="9" * 64,
            expected_baseline_profile_id=payload.baseline_profile_id,
            expected_baseline_section_config_hash=payload.baseline_section_config_hash,
            expected_seed_snapshot_id=payload.seed_snapshot_id,
            authorization=good,
        )


def test_report_stamps_are_exact() -> None:
    stamps = verification_report_stamps(allowlist=PROPOSED_VERIFICATION_ALLOWLIST)
    assert stamps["verification_only"] is True
    assert stamps["not_for_research_interpretation"] is True
    assert stamps["full_pipeline_not_run"] is True
    assert stamps["real_date_count"] == 5
    assert stamps["real_date_allowlist_hash"] == allowlist_sha256(
        PROPOSED_VERIFICATION_ALLOWLIST
    )


def test_control_flow_gates_are_nonresearch_and_complete() -> None:
    passing = evaluate_control_flow_gates(
        {
            "replay_completed": True,
            "invariants_passed": True,
            "artifacts_published_and_reloaded": True,
            "neutrality_passed": True,
            "verifier_link_resolves": True,
            "zero_forbidden_counters": True,
        }
    )
    assert passing.passed is True
    failing = evaluate_control_flow_gates({"replay_completed": True})
    assert failing.passed is False
    # research thresholds are structurally absent from the gate vocabulary
    assert not any("trade" in gate or "expectancy" in gate for gate in passing.results)


def test_coverage_matrix_builds_from_existing_evidence_only() -> None:
    days = ("2026-06-04", "2026-06-05")
    tables = {
        RecordTable.SETUP_LIFECYCLE: pd.DataFrame(
            {"envelope_trading_day": ["2026-06-04", "2026-06-04", "2026-06-05"]}
        ),
        RecordTable.ENTRY_CANDIDATE: pd.DataFrame(
            {"envelope_trading_day": ["2026-06-04"]}
        ),
        RecordTable.ELIGIBLE_DECISION: pd.DataFrame({"envelope_trading_day": []}),
        RecordTable.EXECUTED_TRADE: pd.DataFrame({"envelope_trading_day": []}),
        RecordTable.CANDIDATE_LABEL: pd.DataFrame(
            {"envelope_trading_day": ["2026-06-04"]}
        ),
    }
    matrix = build_verification_coverage_matrix(
        candidate_allowlist=days,
        v2_tables=tables,
        evidence_source_dataset_id="1" * 64,
        evidence_source_manifest_sha256="2" * 64,
        permitted_source_days=("2026-06-04", "2026-06-05"),
        audit_day_counts={"2026-06-04": 12},
        replay_chart_days=("2026-06-04",),
    )
    rows = {row.trading_day: row for row in matrix.payload.rows}
    assert rows["2026-06-04"].setup_lifecycle_rows == 2
    assert rows["2026-06-04"].entry_candidate_rows == 1
    assert rows["2026-06-05"].entry_candidate_rows == 0
    assert matrix.payload.lifecycle_paths_covered["execution_resolution"] is False
    assert "synthetic fixtures must cover" in matrix.payload.uncovered_paths_note
    assert rows["2026-06-04"].mbp1_coverage == "not_evaluated"
