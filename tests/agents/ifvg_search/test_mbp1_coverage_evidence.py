"""R5B.1 — MBP-1 source-coverage policy v2 (evidence-based) suites.

The owner's six proofs (planning decision Q1 #5), the three plan-review
adversarial rows (correction 4 / §9.1), the final contract-closure rows
(physical multi-UTC partition denominators, channel/publisher scope), the
positive-completeness compiler, scope mismatch, head/tail/empty-span
handling, and the bounded diagnostic's fail-before-path refusal.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_diagnostic import (
    assert_diagnostic_authorized,
    build_mbp1_coverage_diagnostic,
    load_mbp1_coverage_diagnostic,
    save_mbp1_coverage_diagnostic,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (
    F_BAD_TS_RECV_FLAG,
    F_MAYBE_BAD_BOOK_FLAG,
    Mbp1CompletenessCompilationReportPayload,
    Mbp1CompletenessStatus,
    Mbp1CoverageEvidenceKind,
    Mbp1DatasetConditionRecord,
    Mbp1EvidenceScopeLevel,
    Mbp1PartitionEvidence,
    Mbp1PartitionGapManifestPayload,
    Mbp1RecoveryBoundaryKind,
    Mbp1UncertaintyStartKind,
    authorized_session_span_ns,
    bad_book_uncertainty_intervals,
    compile_mbp1_partition_gap_manifest,
    compute_partition_coverage,
    dataset_condition_status,
    merge_intervals,
    weighted_day_coverage,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_feature_materializer import (
    materialize_mbp1_features,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
    Mbp1PartitionCoverage,
    assert_evidence_provenance_permitted,
    build_mbp1_source_artifact,
    build_mbp1_source_artifact_from_paths,
    day_coverage_views,
    load_mbp1_source_artifact,
    save_mbp1_source_artifact,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    MBP1_MISSING_REASONS,
    Mbp1SourceContract,
)
from tests.agents.ifvg_search.mbp1_fixture import (
    FIXTURE_DAY,
    MBP1_RESOLVED_BLOCK,
    SYNTHETIC_SOURCE_DOC_SHA,
    anchor_row,
    build_fixture_source,
    canonical_content_sha256,
    declared_interval,
    default_day_events,
    normalized_day,
    ns_at,
    raw_event,
    recovery_boundary,
    synthetic_completeness_report,
    synthetic_contract,
    synthetic_partition_evidence,
    synthetic_scope,
)


def _coverage(evidence, **kwargs):
    """``compute_partition_coverage`` with the certified partition-content
    refs of the evidence's compilation report (R6.1-FIX §3.9, F-10C: a
    positive claim requires EQUALITY with the complete partition content
    refs — the artifact builder always passes them; these unit tests exercise
    the interval arithmetic under the same rule)."""

    report = getattr(evidence, "completeness_report", None)
    refs = tuple(report.payload.verified_partition_refs) if report is not None else None
    kwargs.setdefault("partition_content_refs", refs)
    return compute_partition_coverage(evidence, **kwargs)


def _partition(envelope, index: int = 0) -> Mbp1PartitionCoverage:
    return envelope.payload.ordered_partitions[index]


def _scope(start_s: float = 0.0, end_s: float = 600.0, **kwargs):
    return synthetic_scope(FIXTURE_DAY, start_ns=ns_at(start_s), end_ns=ns_at(end_s), **kwargs)


# ── owner proof 1: normal sequence skips never become source gaps ────────────


def test_normal_sequence_skips_never_become_source_gaps() -> None:
    rows = [
        raw_event(ts_event=ns_at(0), sequence=1000),
        raw_event(ts_event=ns_at(100), sequence=1007),  # legitimate skip
        raw_event(ts_event=ns_at(200), sequence=1050),  # legitimate skip
        raw_event(ts_event=ns_at(600), sequence=1051),
    ]
    envelope, _bytes, _events = build_fixture_source({FIXTURE_DAY: normalized_day(rows)})
    partition = _partition(envelope)
    assert partition.completeness_status is Mbp1CompletenessStatus.EVIDENCED_COMPLETE
    assert partition.coverage_fraction == 1.0
    assert partition.declared_gap_intervals == ()
    assert partition.sequence_jump_diagnostics.positive_jump_count == 2
    assert partition.sequence_jump_diagnostics.total_skipped_numbers == 6 + 42
    assert "sequence_gap" not in MBP1_MISSING_REASONS
    assert envelope.payload.source_contract.gap_semantics == (
        "mbp1_source_coverage_declared_evidence_v2"
    )
    with pytest.raises(ValueError):
        Mbp1SourceContract.model_validate(
            {
                **synthetic_contract().model_dump(mode="json", by_alias=True),
                "gap_semantics": "sequence_gap_marks_interval_invalid_v1",
            }
        )


# ── owner proof 2/3/4: a declared gap types exactly the intersecting windows ─


def test_declared_gap_types_exactly_the_intersecting_windows_never_widened() -> None:
    anchors = pd.DataFrame([anchor_row("cand_a"), anchor_row("cand_b", entry_s=550.0)])
    interval = declared_interval(ns_at(280), ns_at(300))
    source, _bytes, events = build_fixture_source(
        intervals_by_day={FIXTURE_DAY: (interval,)}
    )
    partition = _partition(source)
    assert partition.completeness_status is Mbp1CompletenessStatus.DECLARED_GAPS
    assert partition.declared_gap_intervals[0].start_ts == ns_at(280)
    assert partition.union_gap_ns == ns_at(300) - ns_at(280)
    assert partition.coverage_fraction == pytest.approx(1 - 20 / 600)
    _envelope, features, evidence = materialize_mbp1_features(
        source, anchors, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events
    )
    assert list(features["candidate_id"]) == ["cand_a", "cand_b"]  # preserved
    typed = evidence[evidence["missing_reason"] == "declared_source_gap"]
    # (inversion 240, entry 360] for cand_a and (inversion 240, entry 550]
    # for cand_b intersect (280, 300), as does the entry SNAPSHOT window
    # (day start … entry]; the snapshot windows ending at 60/120/180/240 do
    # not — the interval is never widened onto them
    assert set(typed["feature_window_key"]) == {"ofl_win_inversion_entry", "ofl_snap_entry"}
    assert set(typed["candidate_id"]) == {"cand_a", "cand_b"}
    untouched = evidence[evidence["feature_window_key"] == "ofl_snap_inversion"]
    assert untouched["valid"].all()


# ── owner proof 5: identical corrected evidence reproduces identical artifacts


def test_identical_corrected_evidence_reproduces_identical_artifacts(tmp_path) -> None:
    interval = declared_interval(ns_at(280), ns_at(300))
    one, bytes_one, _e1 = build_fixture_source(intervals_by_day={FIXTURE_DAY: (interval,)})
    two, bytes_two, _e2 = build_fixture_source(intervals_by_day={FIXTURE_DAY: (interval,)})
    assert one.mbp1_source_artifact_id == two.mbp1_source_artifact_id
    plain, _b, _e = build_fixture_source()
    assert plain.mbp1_source_artifact_id != one.mbp1_source_artifact_id  # evidence is identity
    root = tmp_path / "search_test" / "v1"
    save_mbp1_source_artifact(root, one, bytes_one)
    save_mbp1_source_artifact(root, two, bytes_two)  # verified reuse
    reloaded = load_mbp1_source_artifact(root, one.mbp1_source_artifact_id)
    assert reloaded.model_dump(mode="json") == one.model_dump(mode="json")


# ── plan-review row: F_MAYBE_BAD_BOOK without documented recovery fails closed


def _flagged_stream():
    return [
        raw_event(ts_event=ns_at(0), sequence=1),
        raw_event(ts_event=ns_at(100), sequence=2),
        {**raw_event(ts_event=ns_at(200), sequence=3), "flags": F_MAYBE_BAD_BOOK_FLAG},
        raw_event(ts_event=ns_at(250), sequence=4),  # the "next unflagged row"
        raw_event(ts_event=ns_at(600), sequence=5),
    ]


def test_maybe_bad_book_without_documented_recovery_fails_closed() -> None:
    frame = normalized_day(_flagged_stream())
    scope = _scope()
    evidence = synthetic_partition_evidence(scope)
    intervals = bad_book_uncertainty_intervals(frame, evidence)
    assert len(intervals) == 1
    (interval,) = intervals
    # start = the last TRUSTED in-scope event (100 s), not the detection row
    assert interval.start_ts == ns_at(100)
    assert interval.start_kind is Mbp1UncertaintyStartKind.LAST_TRUSTED_IN_SCOPE_EVENT
    # the next unflagged row (250 s) NEVER closes it: it runs to the partition end
    assert interval.end_ts == scope.partition_expected_end_ts
    assert interval.recovery_boundary_kind is Mbp1RecoveryBoundaryKind.PARTITION_END_FAIL_CLOSED
    assert interval.evidence_kind is Mbp1CoverageEvidenceKind.VENDOR_FLAG_MAYBE_BAD_BOOK
    computation = _coverage(
        evidence, trading_day=FIXTURE_DAY, flag_intervals=intervals
    )
    assert computation.open_uncertainty_to_partition_end is True
    assert computation.coverage_fraction == pytest.approx(100 / 600)
    assert computation.completeness_status is Mbp1CompletenessStatus.DECLARED_GAPS
    # a DOCUMENTED recovery boundary closes it exactly there
    recovered = synthetic_partition_evidence(
        scope,
        recovery_boundaries=(
            recovery_boundary(
                scope, ns_at(320), Mbp1RecoveryBoundaryKind.VENDOR_RECOVERY_EVENT
            ),
        ),
    )
    (closed,) = bad_book_uncertainty_intervals(frame, recovered)
    assert (closed.start_ts, closed.end_ts) == (ns_at(100), ns_at(320))
    assert closed.recovery_boundary_kind is Mbp1RecoveryBoundaryKind.VENDOR_RECOVERY_EVENT
    # an undocumented recovery kind is unrepresentable
    with pytest.raises(ValueError, match="not a documented recovery kind"):
        recovery_boundary(scope, ns_at(320), Mbp1RecoveryBoundaryKind.PARTITION_END_FAIL_CLOSED)
    # review F5: a boundary AT the flagged instant closes AFTER the flagged
    # record — the flagged event is always inside the uncertainty
    at_flag = synthetic_partition_evidence(
        scope,
        recovery_boundaries=(
            recovery_boundary(scope, ns_at(200), Mbp1RecoveryBoundaryKind.SNAPSHOT_RECOVERY),
        ),
    )
    (minimal,) = bad_book_uncertainty_intervals(frame, at_flag)
    assert minimal.start_ts == ns_at(100) and minimal.end_ts == ns_at(200) + 1
    # review F6: a boundary documented BEYOND the partition end never closes
    # the interval — it runs to the partition end, fail closed
    beyond = synthetic_partition_evidence(
        scope,
        recovery_boundaries=(
            recovery_boundary(
                scope, ns_at(900), Mbp1RecoveryBoundaryKind.VENDOR_RECOVERY_EVENT
            ),
        ),
    )
    (open_interval,) = bad_book_uncertainty_intervals(frame, beyond)
    assert open_interval.end_ts == scope.partition_expected_end_ts
    assert open_interval.recovery_boundary_kind is (
        Mbp1RecoveryBoundaryKind.PARTITION_END_FAIL_CLOSED
    )
    beyond_computation = _coverage(
        beyond, trading_day=FIXTURE_DAY, flag_intervals=(open_interval,)
    )
    assert beyond_computation.open_uncertainty_to_partition_end is True
    # review F7: a boundary or condition record for ANOTHER partition is refused
    other_scope = synthetic_scope(
        FIXTURE_DAY, start_ns=ns_at(0), end_ns=ns_at(600), partition_key="z"
    )
    with pytest.raises(ValueError, match="another physical partition"):
        synthetic_partition_evidence(
            scope,
            recovery_boundaries=(
                recovery_boundary(
                    other_scope, ns_at(320), Mbp1RecoveryBoundaryKind.SNAPSHOT_RECOVERY
                ),
            ),
        )
    with pytest.raises(ValueError, match="another dataset or UTC date"):
        Mbp1PartitionEvidence(
            scope=scope,
            gap_manifest=None,
            dataset_condition=Mbp1DatasetConditionRecord(
                dataset="OTHER.DATASET",
                utc_date=FIXTURE_DAY,
                condition="degraded",
                source_document_sha256="8" * 64,
                recorded_at="2026-08-28T00:00:00Z",
            ),
        )


def test_uncertainty_start_falls_back_to_partition_start_without_a_trusted_prior() -> None:
    rows = [
        {**raw_event(ts_event=ns_at(0), sequence=1), "flags": F_BAD_TS_RECV_FLAG},
        {**raw_event(ts_event=ns_at(50), sequence=2), "flags": F_MAYBE_BAD_BOOK_FLAG},
        raw_event(ts_event=ns_at(600), sequence=3),
    ]
    scope = _scope(start_s=-30.0)
    (interval,) = bad_book_uncertainty_intervals(
        normalized_day(rows), synthetic_partition_evidence(scope)
    )
    assert interval.start_ts == scope.partition_expected_start_ts
    assert interval.start_kind is Mbp1UncertaintyStartKind.PARTITION_EXPECTED_START
    # a manifest-declared interval containing the flagged instant wins
    declared = declared_interval(ns_at(40), ns_at(70))
    (manifest_start,) = bad_book_uncertainty_intervals(
        normalized_day(rows), synthetic_partition_evidence(scope, intervals=(declared,))
    )
    assert manifest_start.start_ts == ns_at(40)
    assert manifest_start.start_kind is Mbp1UncertaintyStartKind.MANIFEST_DECLARED_START


# ── plan-review row: overlapping declared intervals are merged once ──────────


def test_overlapping_declared_gap_intervals_are_merged_once() -> None:
    scope = _scope()
    flagged = [
        raw_event(ts_event=ns_at(0), sequence=1),
        {**raw_event(ts_event=ns_at(120), sequence=2), "flags": F_MAYBE_BAD_BOOK_FLAG},
        raw_event(ts_event=ns_at(600), sequence=3),
    ]
    evidence = synthetic_partition_evidence(
        scope,
        intervals=(
            declared_interval(ns_at(100), ns_at(200)),
            declared_interval(ns_at(150), ns_at(260)),  # overlaps the first
            declared_interval(ns_at(-50), ns_at(20)),  # head: clipped to the span
        ),
        recovery_boundaries=(
            recovery_boundary(scope, ns_at(180), Mbp1RecoveryBoundaryKind.SNAPSHOT_RECOVERY),
        ),
    )
    flag_intervals = bad_book_uncertainty_intervals(normalized_day(flagged), evidence)
    # the flagged instant (120 s) lies inside the manifest-declared
    # [100 s, 200 s): the uncertainty starts at the DECLARED start (rule a)
    # and closes at the documented 180 s recovery — a SECOND evidence kind
    # overlapping both manifest intervals
    (flag_interval,) = flag_intervals
    assert (flag_interval.start_ts, flag_interval.end_ts) == (ns_at(100), ns_at(180))
    assert flag_interval.start_kind is Mbp1UncertaintyStartKind.MANIFEST_DECLARED_START
    computation = _coverage(
        evidence, trading_day=FIXTURE_DAY, flag_intervals=flag_intervals
    )
    # three overlapping intervals from two evidence kinds → ONE merged
    # interval [100, 260); the clipped head interval stays separate
    assert [(i.start_ts, i.end_ts) for i in computation.intervals] == [
        (ns_at(0), ns_at(20)),
        (ns_at(100), ns_at(260)),
    ]
    merged = computation.intervals[1]
    assert merged.merged is True
    assert computation.intervals[0].merged is False
    # review F12: provenance survives the merge — both evidence kinds are named
    assert merged.contributing_evidence_kinds == (
        "declared_partition_gap_manifest",
        "vendor_flag_maybe_bad_book",
    )
    assert computation.intervals[0].contributing_evidence_kinds == (
        "declared_partition_gap_manifest",
    )
    # subtracted ONCE: 20 s + 160 s of a 600 s physical span
    assert computation.union_gap_ns == ns_at(20) - ns_at(0) + ns_at(260) - ns_at(100)
    assert computation.coverage_fraction == pytest.approx(1 - 180 / 600)
    assert merge_intervals([(5, 10), (8, 12), (20, 25)], clip=(6, 24)) == ((6, 12), (20, 24))
    assert merge_intervals([(5, 5), (30, 20)]) == ()


# ── plan-review row: dataset condition available ≠ complete ─────────────────


def test_dataset_condition_available_without_partition_evidence_is_unknown() -> None:
    scope = _scope(end_s=601.0)  # half-open span past the last fixture event
    available_only = synthetic_partition_evidence(
        scope, with_manifest=False, dataset_condition="available"
    )
    computation = _coverage(available_only, trading_day=FIXTURE_DAY)
    assert computation.completeness_status is Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
    assert computation.dataset_condition_status.value == "vendor_no_known_dataset_issue"
    # the day's windows are typed coverage_evidence_unavailable
    source, _bytes, events = build_fixture_source(
        coverage_evidence={FIXTURE_DAY: (available_only,)}
    )
    _env, _features, evidence = materialize_mbp1_features(
        source,
        pd.DataFrame([anchor_row("cand_x")]),
        resolved_block=MBP1_RESOLVED_BLOCK,
        events_by_day=events,
    )
    assert set(evidence["missing_reason"]) == {"coverage_evidence_unavailable"}
    # `degraded` downgrades the WHOLE day even with a compiled positive claim
    degraded = synthetic_partition_evidence(scope, dataset_condition="degraded")
    downgraded = _coverage(degraded, trading_day=FIXTURE_DAY)
    assert downgraded.completeness_status is Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
    assert downgraded.dataset_condition_status.value == "vendor_dataset_degraded"
    # the five vocabulary states stay distinct
    statuses = {
        dataset_condition_status(
            Mbp1DatasetConditionRecord(
                dataset="D",
                utc_date=FIXTURE_DAY,
                condition=condition,
                source_document_sha256="8" * 64,
                recorded_at="2026-08-28T00:00:00Z",
            )
        ).value
        for condition in ("available", "degraded", "pending", "missing")
    } | {dataset_condition_status(None).value}
    assert statuses == {
        "vendor_no_known_dataset_issue",
        "vendor_dataset_degraded",
        "vendor_dataset_pending",
        "vendor_dataset_missing",
        "vendor_condition_unavailable",
    }


# ── positive completeness comes ONLY from the verified compiler ──────────────


def test_positive_completeness_only_from_the_verified_compiler() -> None:
    scope = _scope()
    unauthorized = synthetic_completeness_report(scope, positive_completeness_authorized=False)
    manifest = compile_mbp1_partition_gap_manifest(
        unauthorized,
        source_document_sha256=SYNTHETIC_SOURCE_DOC_SHA,
        declared_by="t",
        declared_at="2026-08-28T00:00:00Z",
    )
    assert manifest.payload.completeness_assertion == "none"
    assert manifest.payload.completeness_compilation_report_id is None
    evidence = Mbp1PartitionEvidence(
        scope=scope, gap_manifest=manifest, provenance="synthetic_fixture"
    )
    assert (
        _coverage(evidence, trading_day=FIXTURE_DAY).completeness_status
        is Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
    )
    # a bare positive assertion without the compiler's report id is unrepresentable
    with pytest.raises(ValueError, match="never a bare boolean"):
        Mbp1PartitionGapManifestPayload(
            scope=scope,
            source_document_sha256=SYNTHETIC_SOURCE_DOC_SHA,
            intervals=(),
            completeness_assertion="verified_complete_outside_intervals",
            completeness_compilation_report_id=None,
            provenance="synthetic_fixture",
            declared_by="t",
            declared_at="2026-08-28T00:00:00Z",
        )
    # the manifest's source-document hash is the SOURCE document's — the
    # artifact's own hash never enters its payload (non-self-referential)
    assert manifest.payload.source_document_sha256 != manifest.mbp1_partition_gap_manifest_id
    # the compiled provenance rides the manifest and can never be relabeled
    assert manifest.payload.provenance == "synthetic_fixture"
    with pytest.raises(ValueError, match="never be relabeled owner-reviewed"):
        Mbp1PartitionEvidence(scope=scope, gap_manifest=manifest, provenance="owner_reviewed")
    # completeness_unknown NEVER reports positive coverage — the fraction is
    # fail-closed even when the intervals are empty
    unknown_evidence = Mbp1PartitionEvidence(
        scope=scope, gap_manifest=manifest, provenance="synthetic_fixture"
    )
    unknown_computation = _coverage(unknown_evidence, trading_day=FIXTURE_DAY)
    assert unknown_computation.coverage_fraction == 0.0
    # review S3: a provenance label without a manifest is unrepresentable,
    # and a positive manifest without its compilation report is refused
    with pytest.raises(ValueError, match="requires partition-scope evidence"):
        Mbp1PartitionEvidence(scope=scope, gap_manifest=None, provenance="owner_reviewed")
    positive_report = synthetic_completeness_report(scope, positive_completeness_authorized=True)
    positive_manifest = compile_mbp1_partition_gap_manifest(
        positive_report,
        source_document_sha256=SYNTHETIC_SOURCE_DOC_SHA,
        declared_by="t",
        declared_at="2026-08-28T00:00:00Z",
    )
    with pytest.raises(ValueError, match="requires its compilation report"):
        Mbp1PartitionEvidence(
            scope=scope, gap_manifest=positive_manifest, provenance="synthetic_fixture"
        )
    # review S7: the source-document hash must be one of the report's evidence
    # refs and can never be the report's own artifact id
    with pytest.raises(ValueError, match="evidence_refs"):
        compile_mbp1_partition_gap_manifest(
            positive_report,
            source_document_sha256="4" * 64,
            declared_by="t",
            declared_at="2026-08-28T00:00:00Z",
        )
    with pytest.raises(ValueError, match="never the"):
        compile_mbp1_partition_gap_manifest(
            positive_report,
            source_document_sha256=positive_report.mbp1_completeness_compilation_report_id,
            declared_by="t",
            declared_at="2026-08-28T00:00:00Z",
        )
    # review F1/S3: the report's refs are content hashes and the owner review
    # id is a 64-hex decision hash, never free text
    with pytest.raises(ValueError, match="content sha256"):
        synthetic_completeness_report(scope, content_refs=("zz" * 32,))
    with pytest.raises(ValueError, match="64-hex"):
        Mbp1CompletenessCompilationReportPayload(
            **{
                **positive_report.payload.model_dump(mode="json"),
                "owner_review_decision_id": "fabricated",
            }
        )
    with pytest.raises(ValueError, match="never reports positive coverage"):
        Mbp1PartitionCoverage.model_validate(
            {
                **_partition(build_fixture_source()[0]).model_dump(mode="json"),
                "completeness_status": "completeness_unknown",
                "coverage_fraction": 1.0,
            }
        )


def test_evidence_stores_round_trip_and_verified_load(tmp_path) -> None:
    """The compilation report and gap manifest are immutable store artifacts;
    `load_verified_partition_evidence` builds evidence from STORE-VERIFIED
    envelopes only and refuses a manifest whose report is absent, tampered,
    unauthorized, or foreign."""

    import json

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (
        load_mbp1_completeness_report,
        load_mbp1_partition_gap_manifest,
        load_verified_partition_evidence,
        save_mbp1_completeness_report,
        save_mbp1_partition_gap_manifest,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError

    root = tmp_path / "search_test" / "v1"
    scope = _scope()
    report = synthetic_completeness_report(
        scope, intervals=(declared_interval(ns_at(100), ns_at(200)),)
    )
    manifest = compile_mbp1_partition_gap_manifest(
        report,
        source_document_sha256=SYNTHETIC_SOURCE_DOC_SHA,
        declared_by="t",
        declared_at="2026-08-28T00:00:00Z",
    )
    # a manifest claiming completeness whose report is not in the store refuses
    save_mbp1_partition_gap_manifest(root, manifest)
    with pytest.raises(SearchStoreError, match="missing search-store entry"):
        load_verified_partition_evidence(root, manifest_id=manifest.mbp1_partition_gap_manifest_id)
    save_mbp1_completeness_report(root, report)
    save_mbp1_completeness_report(root, report)  # verified reuse
    evidence = load_verified_partition_evidence(
        root, manifest_id=manifest.mbp1_partition_gap_manifest_id
    )
    assert evidence.provenance.value == "synthetic_fixture"
    assert evidence.gap_manifest.mbp1_partition_gap_manifest_id == (
        manifest.mbp1_partition_gap_manifest_id
    )
    computation = _coverage(evidence, trading_day=FIXTURE_DAY)
    assert computation.completeness_status is Mbp1CompletenessStatus.DECLARED_GAPS
    assert load_mbp1_completeness_report(
        root, report.mbp1_completeness_compilation_report_id
    ).model_dump(mode="json") == report.model_dump(mode="json")
    assert load_mbp1_partition_gap_manifest(
        root, manifest.mbp1_partition_gap_manifest_id
    ).model_dump(mode="json") == manifest.model_dump(mode="json")
    # a tampered report fails closed on load
    report_dir = (
        root / "mbp1_completeness_reports" / report.mbp1_completeness_compilation_report_id
    )
    envelope_path = report_dir / "envelope.json"
    payload = json.loads(envelope_path.read_text(encoding="utf-8"))
    payload["payload"]["positive_completeness_authorized"] = False
    envelope_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SearchStoreError, match="failed verification"):
        load_verified_partition_evidence(root, manifest_id=manifest.mbp1_partition_gap_manifest_id)


def test_protected_and_sealed_dates_are_unrepresentable_in_evidence() -> None:
    for day in ("2026-06-11", "2026-06-12", "2026-07-01", "2025-12-31"):
        with pytest.raises(ValueError, match="unrepresentable"):
            synthetic_scope(day, start_ns=1, end_ns=2)
        with pytest.raises(ValueError, match="unrepresentable"):
            Mbp1DatasetConditionRecord(
                dataset="D",
                utc_date=day,
                condition="available",
                source_document_sha256="8" * 64,
                recorded_at="2026-08-28T00:00:00Z",
            )
    synthetic_scope("2026-06-10", start_ns=1, end_ns=2)  # the last exposed development day


# ── final-closure row: multi-UTC partitions + channel-wide bad-book scope ────


def test_multi_utc_partition_denominators_and_channel_wide_bad_book_scope(tmp_path) -> None:
    session_start, session_end = authorized_session_span_ns(FIXTURE_DAY)
    # two physical UTC partitions of ONE trading day: the first covers the
    # session's first 40%, the second the remaining 60% — never the whole
    # trading-day span repeated per partition
    split = session_start + int(0.4 * (session_end - session_start))
    scope_a = synthetic_scope(
        FIXTURE_DAY,
        start_ns=session_start - 3_600_000_000_000,  # starts before the session: clipped
        end_ns=split,
        partition_key="utc_a/mbp1",
        utc_date="2026-01-12",
    )
    scope_b = synthetic_scope(
        FIXTURE_DAY, start_ns=split, end_ns=session_end, partition_key="utc_b/mbp1"
    )
    gap_b = declared_interval(split, split + (session_end - split) // 2)  # half of B
    content = canonical_content_sha256(default_day_events())
    evidence_a = synthetic_partition_evidence(scope_a, content_refs=(content,))
    evidence_b = synthetic_partition_evidence(scope_b, intervals=(gap_b,), content_refs=(content,))
    comp_a = _coverage(evidence_a, trading_day=FIXTURE_DAY)
    comp_b = _coverage(evidence_b, trading_day=FIXTURE_DAY)
    assert comp_a.physical_expected_span_ns == split - session_start  # clipped head
    assert comp_b.physical_expected_span_ns == session_end - split
    assert comp_a.coverage_fraction == 1.0
    assert comp_b.coverage_fraction == pytest.approx(0.5)
    day_coverage, day_status = weighted_day_coverage(
        [
            (comp.physical_expected_span_ns, comp.coverage_fraction, comp.completeness_status)
            for comp in (comp_a, comp_b)
        ]
    )
    assert day_coverage == pytest.approx(0.4 * 1.0 + 0.6 * 0.5)
    assert day_status is Mbp1CompletenessStatus.DECLARED_GAPS
    assert weighted_day_coverage([]) == (0.0, Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN)
    # through the artifact: two rows for one trading day, one day view
    events = {FIXTURE_DAY: default_day_events()}
    envelope, _bytes = build_mbp1_source_artifact(
        events,
        contract=synthetic_contract(),
        authorized_date_set_id="synthetic_fixture_days_v1",
        coverage_evidence={FIXTURE_DAY: (evidence_a, evidence_b)},
    )
    assert [p.relative_logical_partition_key for p in envelope.payload.ordered_partitions] == [
        "utc_a/mbp1",
        "utc_b/mbp1",
    ]
    view = day_coverage_views(envelope)[FIXTURE_DAY]
    assert view.partition_count == 2
    assert view.coverage_fraction == pytest.approx(0.7)
    assert view.uncovered_session_ns == 0 and view.uncovered_session_intervals == ()
    # every event counted exactly once across the two half-open partitions
    assert sum(p.row_count for p in envelope.payload.ordered_partitions) == len(
        default_day_events()
    )
    # review F3: overlapping / duplicated physical partitions are refused
    duplicate = synthetic_partition_evidence(
        synthetic_scope(
            FIXTURE_DAY, start_ns=split, end_ns=session_end, partition_key="utc_c/mbp1"
        ),
        content_refs=(content,),
    )
    with pytest.raises(ValueError, match="overlap"):
        build_mbp1_source_artifact(
            events,
            contract=synthetic_contract(),
            authorized_date_set_id="synthetic_fixture_days_v1",
            coverage_evidence={FIXTURE_DAY: (evidence_a, evidence_b, duplicate)},
        )
    # review F1: a positive claim that does not name this partition's content is refused
    unbound = synthetic_partition_evidence(scope_b, intervals=(gap_b,))  # placeholder ref
    with pytest.raises(ValueError, match="does not reference the partition content"):
        build_mbp1_source_artifact(
            events,
            contract=synthetic_contract(),
            authorized_date_set_id="synthetic_fixture_days_v1",
            coverage_evidence={FIXTURE_DAY: (evidence_a, unbound)},
        )
    # the multi-row day persists: the sidecar bytes verify against EVERY
    # row's content hash, and a forged row refuses before any write
    root = tmp_path / "search_test" / "v1"
    save_mbp1_source_artifact(root, envelope, _bytes)
    reloaded = load_mbp1_source_artifact(root, envelope.mbp1_source_artifact_id)
    assert len(reloaded.payload.ordered_partitions) == 2
    forged_rows = list(envelope.payload.ordered_partitions)
    forged_rows[1] = forged_rows[1].model_copy(update={"content_sha256": "0" * 64})
    forged = type(envelope).from_payload(
        envelope.payload.model_copy(update={"ordered_partitions": tuple(forged_rows)}),
        events_stored=True,
    )
    with pytest.raises(ValueError, match="coverage content_sha256"):
        save_mbp1_source_artifact(tmp_path / "search_test" / "forged", forged, _bytes)
    # channel scope: a bad-book flag is publisher/physical-partition scoped
    # unless a VERIFIED channel map exists — never instrument-scoped
    frame = normalized_day(_flagged_stream())
    scope = _scope()
    (publisher_scoped,) = bad_book_uncertainty_intervals(
        frame, synthetic_partition_evidence(scope)
    )
    assert publisher_scoped.scope_level is Mbp1EvidenceScopeLevel.PUBLISHER_PARTITION
    channel_scope = _scope(scope_level=Mbp1EvidenceScopeLevel.CHANNEL_PARTITION, channel_id=7)
    (channel_scoped,) = bad_book_uncertainty_intervals(
        frame, synthetic_partition_evidence(channel_scope, channel_map_verified=True)
    )
    assert channel_scoped.scope_level is Mbp1EvidenceScopeLevel.CHANNEL_PARTITION
    (unverified,) = bad_book_uncertainty_intervals(
        frame, synthetic_partition_evidence(channel_scope, channel_map_verified=False)
    )
    assert unverified.scope_level is Mbp1EvidenceScopeLevel.PUBLISHER_PARTITION
    with pytest.raises(ValueError, match="instrument_partition scope requires"):
        _scope(scope_level=Mbp1EvidenceScopeLevel.INSTRUMENT_PARTITION)
    with pytest.raises(ValueError, match="verified channel map requires"):
        Mbp1PartitionEvidence(
            scope=scope, gap_manifest=None, channel_map_verified=True
        )


# ── review F2: unevidenced session time never yields valid features ──────────


def test_undeclared_hole_between_partitions_types_windows_and_stray_events_refuse() -> None:
    """Two declared partitions that leave a HOLE in the session span: the hole
    is an uncovered session interval, every window touching it is typed
    ``coverage_evidence_unavailable``, and an in-session event outside every
    declared span refuses the build (the manifest contradicts the data)."""

    events = default_day_events()
    content = canonical_content_sha256(events)
    scope_a = synthetic_scope(
        FIXTURE_DAY, start_ns=ns_at(0), end_ns=ns_at(200), partition_key="a"
    )
    scope_b = synthetic_scope(
        FIXTURE_DAY, start_ns=ns_at(400), end_ns=ns_at(601), partition_key="b"
    )
    evidence = {
        FIXTURE_DAY: (
            synthetic_partition_evidence(scope_a, content_refs=(content,)),
            synthetic_partition_evidence(scope_b, content_refs=(content,)),
        )
    }
    with pytest.raises(ValueError, match="outside every declared physical partition span"):
        build_mbp1_source_artifact(
            {FIXTURE_DAY: events},
            contract=synthetic_contract(),
            authorized_date_set_id="x",
            coverage_evidence=evidence,
        )
    # drop the events inside the hole: the build succeeds, the hole is typed
    holed = events[(events["ts_event"] < ns_at(200)) | (events["ts_event"] >= ns_at(400))]
    holed_content = canonical_content_sha256(holed)
    holed_evidence = {
        FIXTURE_DAY: (
            synthetic_partition_evidence(scope_a, content_refs=(holed_content,)),
            synthetic_partition_evidence(scope_b, content_refs=(holed_content,)),
        )
    }
    source, _bytes, _ = build_fixture_source(
        {FIXTURE_DAY: holed}, coverage_evidence=holed_evidence
    )
    view = day_coverage_views(source)[FIXTURE_DAY]
    assert view.completeness_status is Mbp1CompletenessStatus.EVIDENCED_COMPLETE
    assert view.uncovered_session_ns > 0
    assert any(start <= ns_at(200) < end for start, end in view.uncovered_session_intervals)
    anchors = pd.DataFrame([anchor_row("in_hole")])  # inversion 240 s, entry 360 s
    _env, features, evidence_frame = materialize_mbp1_features(
        source, anchors, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day={FIXTURE_DAY: holed}
    )
    row = features.set_index("candidate_id").loc["in_hole"]
    assert row["ofl_win_inversion_entry_missing_reason"] == "coverage_evidence_unavailable"
    assert row["ofl_snap_entry_missing_reason"] == "coverage_evidence_unavailable"
    assert row["ofl_snap_htf_tap_valid"]  # 60 s: inside partition a, untouched
    assert set(
        evidence_frame[evidence_frame["candidate_id"] == "in_hole"]["missing_reason"]
    ) >= {"coverage_evidence_unavailable"}


def test_real_read_clips_to_trading_day_session_and_development_cutoff(tmp_path) -> None:
    """Review S1: the UTC-date file is not the trading day — rows outside
    ``[18:00 ET D-1, 17:00 ET D)`` and rows at/after DEVELOPMENT_CUTOFF_UTC
    are clipped BEFORE normalization/hashing, counted, and never labeled with
    the file's day; the raw file hash rides the artifact as a content ref."""

    from alpha_lab.agents.data_infra.ifvg.development_access import (
        VerificationReplayPolicy,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
        read_mbp1_partition_frame,
    )
    from alpha_lab.agents.data_infra.ifvg.manifest import file_sha256

    day = FIXTURE_DAY
    session_start, session_end = authorized_session_span_ns(day)
    rows = [
        raw_event(ts_event=session_start - 1, sequence=1),  # previous trading day
        raw_event(ts_event=session_start, sequence=2),  # first in-session instant
        raw_event(ts_event=ns_at(0), sequence=3),  # 14:00Z, in session
        raw_event(ts_event=session_end - 1, sequence=4),  # last in-session instant
        raw_event(ts_event=session_end, sequence=5),  # 17:00 ET closed window
        raw_event(ts_event=session_end + 3_600_000_000_000, sequence=6),  # 18:00 ET = next day
    ]
    partition_dir = tmp_path / "NQ" / day
    partition_dir.mkdir(parents=True)
    path = partition_dir / "mbp1.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    policy = VerificationReplayPolicy((day,))
    frame = read_mbp1_partition_frame(
        day, access_policy=policy, path_factory=lambda d: tmp_path / "NQ" / d / "mbp1.parquet",
        instrument="NQ",
    )
    assert list(frame["sequence"]) == [2, 3, 4]
    assert frame.attrs["rows_outside_session_span"] == 3
    assert frame.attrs["rows_after_development_cutoff"] == 0
    assert frame.attrs["source_file_sha256"] == file_sha256(path)
    assert set(frame["trading_day"]) == {day}
    artifact, _ = build_mbp1_source_artifact(
        {day: frame}, contract=synthetic_contract(), authorized_date_set_id="x"
    )
    partition = artifact.payload.ordered_partitions[0]
    assert partition.rows_outside_session_span == 3
    assert file_sha256(path) in partition.source_content_refs
    assert partition.row_count == 3
    # the development cutoff clips at-or-after 2026-06-10T21:00Z on the last
    # exposed day (pure timestamp arithmetic; no real path is constructed)
    from alpha_lab.agents.data_infra.ifvg.development_access import DEVELOPMENT_CUTOFF_UTC

    last_day = "2026-06-10"
    cutoff = int(pd.Timestamp(DEVELOPMENT_CUTOFF_UTC).value)
    last_start, last_end = authorized_session_span_ns(last_day)
    assert last_end == cutoff  # the session end IS the cutoff on the last exposed day
    late_rows = [
        raw_event(ts_event=last_start + 60_000_000_000, sequence=1),
        raw_event(ts_event=cutoff - 1, sequence=2),
        raw_event(ts_event=cutoff, sequence=3),  # at the cutoff: clipped
        raw_event(ts_event=cutoff + 3_600_000_000_000 * 2, sequence=4),  # 18:00 ET → 06-11
    ]
    late_dir = tmp_path / "NQ" / last_day
    late_dir.mkdir(parents=True)
    pd.DataFrame(late_rows).to_parquet(late_dir / "mbp1.parquet", index=False)
    late = read_mbp1_partition_frame(
        last_day,
        access_policy=VerificationReplayPolicy((last_day,)),
        path_factory=lambda d: tmp_path / "NQ" / d / "mbp1.parquet",
        instrument="NQ",
    )
    assert list(late["sequence"]) == [1, 2]
    assert late.attrs["rows_outside_session_span"] == 2


# ── scope mismatch, empty span, head/tail ────────────────────────────────────


def test_scope_mismatch_empty_span_and_head_tail_handling() -> None:
    scope = _scope()
    other = synthetic_scope(FIXTURE_DAY, start_ns=ns_at(0), end_ns=ns_at(600), partition_key="x")
    good = synthetic_partition_evidence(other)
    with pytest.raises(ValueError, match="different physical partition"):
        Mbp1PartitionEvidence(
            scope=scope, gap_manifest=good.gap_manifest, provenance="synthetic_fixture"
        )
    # an empty physical span is unknown (never "complete over nothing")
    empty = synthetic_partition_evidence(
        synthetic_scope(FIXTURE_DAY, start_ns=ns_at(10), end_ns=ns_at(10))
    )
    computation = _coverage(empty, trading_day=FIXTURE_DAY)
    assert computation.physical_expected_span_ns == 0
    assert computation.completeness_status is Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN
    # head/tail gaps count only when DECLARED; undeclared silence is not a gap
    tail = synthetic_partition_evidence(
        scope, intervals=(declared_interval(ns_at(500), ns_at(700)),)
    )
    tail_computation = _coverage(tail, trading_day=FIXTURE_DAY)
    assert tail_computation.union_gap_ns == ns_at(600) - ns_at(500)  # clipped tail
    silent = synthetic_partition_evidence(scope)
    assert _coverage(silent, trading_day=FIXTURE_DAY).coverage_fraction == 1.0
    # a partition outside the authorized session span contributes nothing
    session_start, _session_end = authorized_session_span_ns(FIXTURE_DAY)
    outside = synthetic_partition_evidence(
        synthetic_scope(
            FIXTURE_DAY,
            start_ns=session_start - 7_200_000_000_000,
            end_ns=session_start - 3_600_000_000_000,
        )
    )
    outside_computation = _coverage(outside, trading_day=FIXTURE_DAY)
    assert outside_computation.physical_expected_span_ns == 0


# ── provenance: synthetic evidence never enters a real scope ─────────────────


def test_synthetic_evidence_is_refused_outside_the_synthetic_scope(tmp_path) -> None:
    envelope, _bytes, _events = build_fixture_source()
    assert_evidence_provenance_permitted(
        envelope.payload.ordered_partitions, synthetic_scope=True
    )
    with pytest.raises(PermissionError, match="synthetic coverage evidence"):
        assert_evidence_provenance_permitted(
            envelope.payload.ordered_partitions, synthetic_scope=False
        )

    class _NeverPolicy:
        def resolve_source_path(self, day, path_factory):  # pragma: no cover
            raise AssertionError("must not be reached")

    calls: list[str] = []
    with pytest.raises(PermissionError, match="synthetic coverage evidence"):
        build_mbp1_source_artifact_from_paths(
            (FIXTURE_DAY,),
            access_policy=_NeverPolicy(),
            path_factory=lambda day: calls.append(day) or tmp_path / day,
            contract=synthetic_contract(),
            authorized_date_set_id="x",
            coverage_evidence={FIXTURE_DAY: (synthetic_partition_evidence(_scope()),)},
        )
    assert calls == []  # refused before any path


# ── the bounded diagnostic: shape on synthetic; the R1 gate on real ──────────


def _verification_store(tmp_path, days):
    """A tmp ``search_test/v1`` store carrying a coverage matrix + a
    verification-run envelope over ``days`` (synthetic, never signed)."""

    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope
    from alpha_lab.agents.data_infra.ifvg.search.verification import (
        CoverageMatrixEnvelope,
        CoverageMatrixPayload,
        DayCoverageRow,
        VerificationRunEnvelope,
        VerificationRunPayload,
    )

    root = tmp_path / "search_test" / "v1"
    matrix = CoverageMatrixEnvelope.from_payload(
        CoverageMatrixPayload(
            evidence_source_dataset_id="1" * 64,
            evidence_source_manifest_sha256="2" * 64,
            candidate_allowlist=tuple(days),
            candidate_allowlist_hash=allowlist_sha256(days),
            rows=tuple(
                DayCoverageRow(
                    trading_day=day,
                    source_partition_recorded=True,
                    setup_lifecycle_rows=1,
                    entry_candidate_rows=1,
                    eligible_decision_rows=0,
                    executed_trade_rows=0,
                    candidate_label_rows=1,
                    audit_event_rows=None,
                    replay_chart_available=None,
                )
                for day in days
            ),
            lifecycle_paths_covered={"entry_candidate": True},
            uncovered_paths_note="synthetic",
        )
    )
    save_or_reuse_envelope(root, "coverage_matrices", matrix)
    from tests.agents.ifvg_search.namespace_fixture import verification_authorization_ref

    authorization = verification_authorization_ref(
        root,
        approved_allowlist_hash=allowlist_sha256(days),
        coverage_matrix_artifact_id=matrix.coverage_matrix_id,
        seed_snapshot_id="b" * 64,
        approved_by="synthetic",
        approved_at="2026-08-28T00:00:00Z",
        content_hash="c" * 64,
    )
    run = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=authorization,
            allowlist=tuple(days),
            allowlist_hash=allowlist_sha256(days),
            seed_snapshot_id="b" * 64,
            baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            baseline_section_config_hash="e" * 64,
            coverage_matrix_artifact_id=matrix.coverage_matrix_id,
        )
    )
    save_or_reuse_envelope(root, "verification_runs", run)
    return root, run, matrix


def test_coverage_diagnostic_shape_and_fail_before_path(tmp_path) -> None:
    from alpha_lab.agents.data_infra.ifvg.development_access import (
        VerificationReplayPolicy,
    )
    from alpha_lab.agents.data_infra.ifvg.search.verification import (
        VerificationRunEnvelope,
    )

    envelope, event_bytes, _events = build_fixture_source(
        intervals_by_day={FIXTURE_DAY: (declared_interval(ns_at(280), ns_at(300)),)}
    )
    root = tmp_path / "search_test" / "v1"
    save_mbp1_source_artifact(root, envelope, event_bytes)
    report = build_mbp1_coverage_diagnostic(
        envelope,
        run_scope="synthetic_fixture",
        allowlist=(FIXTURE_DAY,),
        authorization_content_hash=None,
    )
    save_mbp1_coverage_diagnostic(root, report)
    reloaded = load_mbp1_coverage_diagnostic(root, report.mbp1_coverage_diagnostic_id)
    row = reloaded.payload.rows[0]
    assert row.completeness_status is Mbp1CompletenessStatus.DECLARED_GAPS
    assert row.declared_gap_count == 1
    assert row.gap_manifest_present and row.completeness_report_present
    assert row.partition_scope_evidence_present and row.open_uncertainty_to_partition_end is False
    assert row.sequence_positive_jump_count == 0
    assert reloaded.payload.completeness_inferred_from_sequence_continuity is False
    # an evidence-less partition reports NO open-uncertainty fact (tri-state)
    plain, plain_bytes, _ = build_fixture_source(coverage_evidence={})
    plain_report = build_mbp1_coverage_diagnostic(
        plain,
        run_scope="synthetic_fixture",
        allowlist=(FIXTURE_DAY,),
        authorization_content_hash=None,
    )
    assert plain_report.payload.rows[0].partition_scope_evidence_present is False
    assert plain_report.payload.rows[0].open_uncertainty_to_partition_end is None
    # review S5: a sealed day is unrepresentable in the report
    with pytest.raises(ValueError, match="unrepresentable"):
        build_mbp1_coverage_diagnostic(
            plain,
            run_scope="synthetic_fixture",
            allowlist=("2026-06-12",),
            authorization_content_hash=None,
        )
    # the real gate (review S2/F9): the R1 real-slice checks, before any path
    policy = VerificationReplayPolicy((FIXTURE_DAY,))
    with pytest.raises(PermissionError, match="does not exist"):
        assert_diagnostic_authorized(
            store_root=root, run_envelope=None, access_policy=policy, allowlist=(FIXTURE_DAY,)
        )
    with pytest.raises(PermissionError, match="VerificationReplayPolicy"):
        assert_diagnostic_authorized(
            store_root=root, run_envelope=None, access_policy=object(), allowlist=(FIXTURE_DAY,)
        )
    with pytest.raises(PermissionError, match="search_test/v1"):
        assert_diagnostic_authorized(
            store_root=tmp_path / "search_test_evil" / "v1",
            run_envelope=None,
            access_policy=policy,
            allowlist=(FIXTURE_DAY,),
        )
    store_root, run, matrix = _verification_store(tmp_path / "gate", (FIXTURE_DAY,))
    # HARDENING-BACKEND (RA-01): the namespace + head binding now fires FIRST —
    # an unmarked root refuses `store_namespace_missing` before the matrix is
    # consulted; mark `root` as the SAME namespace instance so the matrix
    # refusal below is reached (the binding itself is proven in
    # test_hardening_fix_round.py)
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
        initialize_store_namespace,
        load_store_namespace,
    )

    with pytest.raises(PermissionError, match="store_namespace_missing"):
        assert_diagnostic_authorized(
            store_root=root, run_envelope=run, access_policy=policy, allowlist=(FIXTURE_DAY,)
        )
    initialize_store_namespace(
        root,
        namespace_class="test",
        store_instance_id=load_store_namespace(store_root).payload.store_instance_id,
    )
    # a self-signed run envelope whose coverage matrix is NOT in this store refuses
    with pytest.raises(PermissionError, match="coverage-matrix artifact is not a verified"):
        assert_diagnostic_authorized(
            store_root=root, run_envelope=run, access_policy=policy, allowlist=(FIXTURE_DAY,)
        )
    # a run envelope over a different allowlist refuses
    with pytest.raises(PermissionError, match="differs from the requested days"):
        assert_diagnostic_authorized(
            store_root=store_root,
            run_envelope=run,
            access_policy=VerificationReplayPolicy(("2026-01-14",)),
            allowlist=("2026-01-14",),
        )
    # the lawful gate returns the authorization hash and registers the ONE
    # canonical program allowlist; a different allowlist is then refused
    content_hash = assert_diagnostic_authorized(
        store_root=store_root,
        run_envelope=run,
        access_policy=policy,
        allowlist=(FIXTURE_DAY,),
        canonical_root=store_root,
    )
    assert content_hash == "c" * 64
    other_root, other_run, _ = _verification_store(tmp_path / "other", ("2026-01-14",))
    with pytest.raises(PermissionError, match="one canonical allowlist"):
        assert_diagnostic_authorized(
            store_root=other_root,
            run_envelope=other_run,
            access_policy=VerificationReplayPolicy(("2026-01-14",)),
            allowlist=("2026-01-14",),
            canonical_root=store_root,
        )
    # a forged envelope carrying a mismatched allowlist hash refuses
    forged = VerificationRunEnvelope.from_payload(
        run.payload.model_copy(update={"allowlist_hash": "9" * 64})
    )
    with pytest.raises(PermissionError, match="does not hash its allowlist"):
        assert_diagnostic_authorized(
            store_root=store_root,
            run_envelope=forged,
            access_policy=policy,
            allowlist=(FIXTURE_DAY,),
        )
    with pytest.raises(PermissionError, match="verified authorization hash"):
        build_mbp1_coverage_diagnostic(
            envelope,
            run_scope="verification_5d",
            allowlist=(FIXTURE_DAY,),
            authorization_content_hash=None,
        )


def test_partition_evidence_manifest_seam_loads_store_verified_evidence(tmp_path) -> None:
    """Review F4/S6: the real diagnostic's evidence seam consumes STORE ids
    and typed records only — never in-memory evidence, never a path."""

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_diagnostic import (
        load_partition_evidence_manifest,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (
        save_mbp1_completeness_report,
        save_mbp1_partition_gap_manifest,
    )

    root = tmp_path / "search_test" / "v1"
    scope = _scope(end_s=601.0)  # half-open span past the last fixture event
    content = canonical_content_sha256(default_day_events())
    report = synthetic_completeness_report(
        scope, intervals=(declared_interval(ns_at(100), ns_at(200)),), content_refs=(content,)
    )
    manifest = compile_mbp1_partition_gap_manifest(
        report,
        source_document_sha256=SYNTHETIC_SOURCE_DOC_SHA,
        declared_by="t",
        declared_at="2026-08-28T00:00:00Z",
    )
    save_mbp1_completeness_report(root, report)
    save_mbp1_partition_gap_manifest(root, manifest)
    evidence, ids = load_partition_evidence_manifest(
        root,
        {
            FIXTURE_DAY: [
                {
                    "manifest_id": manifest.mbp1_partition_gap_manifest_id,
                    "channel_map_verified": False,
                    "recovery_boundaries": [
                        {
                            "ts": ns_at(500),
                            "kind": "snapshot_recovery",
                            "source_document_sha256": "9" * 64,
                            "utc_date": scope.utc_date,
                            "physical_partition_key": scope.physical_partition_key,
                        }
                    ],
                    "dataset_condition": {
                        "dataset": scope.dataset,
                        "utc_date": scope.utc_date,
                        "condition": "available",
                        "source_document_sha256": "8" * 64,
                        "recorded_at": "2026-08-28T00:00:00Z",
                    },
                }
            ]
        },
    )
    assert ids == (manifest.mbp1_partition_gap_manifest_id,)
    (loaded,) = evidence[FIXTURE_DAY]
    assert loaded.completeness_report is not None
    assert loaded.recovery_boundaries[0].kind is Mbp1RecoveryBoundaryKind.SNAPSHOT_RECOVERY
    # the loaded evidence binds the artifact and reports the interval facts
    artifact, _bytes = build_mbp1_source_artifact(
        {FIXTURE_DAY: default_day_events()},
        contract=synthetic_contract(),
        authorized_date_set_id="x",
        coverage_evidence=evidence,
    )
    report_env = build_mbp1_coverage_diagnostic(
        artifact,
        run_scope="synthetic_fixture",
        allowlist=(FIXTURE_DAY,),
        authorization_content_hash=None,
        partition_evidence_manifest_ids=ids,
    )
    row = report_env.payload.rows[0]
    assert row.partition_scope_evidence_present and row.declared_gap_count == 1
    assert row.completeness_status is Mbp1CompletenessStatus.DECLARED_GAPS
    assert report_env.payload.partition_evidence_manifest_ids == ids
    # an unknown manifest id fails closed at the store
    from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError

    with pytest.raises(SearchStoreError):
        load_partition_evidence_manifest(root, {FIXTURE_DAY: [{"manifest_id": "0" * 64}]})


def test_diagnostic_cli_refuses_real_path_and_runs_synthetic_shape(tmp_path, capsys) -> None:
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import ifvg_mbp1_coverage_diagnostic as cli

    root = tmp_path / "search_test" / "v1"
    with pytest.raises(SystemExit, match="refused before any source path"):
        cli.main(["--store-root", str(root)])
    envelope, event_bytes, _events = build_fixture_source()
    save_mbp1_source_artifact(root, envelope, event_bytes)
    code = cli.main(
        [
            "--store-root",
            str(root),
            "--synthetic-source-artifact-id",
            envelope.mbp1_source_artifact_id,
        ]
    )
    assert code == 0
    import json

    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["status"] == "synthetic_shape_persisted"
    assert len(payload["mbp1_coverage_diagnostic_id"]) == 64
    # BOTH paths refuse a non-verification namespace before any load — and a
    # SUBSTRING is never a namespace (review S4)
    for namespace in (("search", "v1"), ("search_test_evil", "v1"), ("xsearch_testx", "v1")):
        for argv in (
            ["--store-root", str(tmp_path.joinpath(*namespace)),
             "--synthetic-source-artifact-id", envelope.mbp1_source_artifact_id],
            ["--store-root", str(tmp_path.joinpath(*namespace)),
             "--verification-run-id", "a" * 64],
        ):
            with pytest.raises(SystemExit, match="search_test/v1"):
                cli.main(argv)
    # a REAL-shaped artifact (events referenced by hash only, no stored
    # bytes) is refused by the synthetic path even with NONE provenance
    real_shaped, _ = build_mbp1_source_artifact(
        {FIXTURE_DAY: default_day_events()},
        contract=synthetic_contract(),
        authorized_date_set_id="x",
        events_stored=False,
    )
    save_mbp1_source_artifact(root, real_shaped, {})
    with pytest.raises(SystemExit, match="synthetic-fixture source artifacts only"):
        cli.main(
            [
                "--store-root",
                str(root),
                "--synthetic-source-artifact-id",
                real_shaped.mbp1_source_artifact_id,
            ]
        )
    # the real path refuses a persisted-but-unbacked run envelope (no
    # coverage matrix in the store) BEFORE any source path
    store_root, run, _matrix = _verification_store(tmp_path / "gate", (FIXTURE_DAY,))
    import shutil

    shutil.rmtree(store_root / "coverage_matrices")
    with pytest.raises(SystemExit, match="coverage-matrix artifact is not a verified"):
        cli.main(
            ["--store-root", str(store_root), "--verification-run-id", run.verification_run_id]
        )
