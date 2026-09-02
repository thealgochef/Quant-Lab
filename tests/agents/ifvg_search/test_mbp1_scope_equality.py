"""R6.1-FIX §3.9 (F-10C / F-10D) — a positive MBP-1 completeness claim
requires FULL equality between the partition evidence scope, the gap
manifest scope, and the complete ordered partition-content refs; enum-typed
identity payloads serialize without warnings."""

from __future__ import annotations

import warnings

import pytest

from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (
    Mbp1PartitionEvidence,
    compute_partition_coverage,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    R5B_WINDOW_SPECS,
    IntervalBound,
    Mbp1FeatureWindowSpec,
    WindowTriggerSemantics,
)
from tests.agents.ifvg_search.mbp1_fixture import (
    FIXTURE_DAY,
    canonical_content_sha256,
    default_day_events,
    synthetic_partition_evidence,
    synthetic_scope,
)


def _evidence(**overrides):
    frame = default_day_events()
    start = int(frame["ts_event"].min())
    end = int(frame["ts_event"].max()) + 1
    scope = synthetic_scope(FIXTURE_DAY, start_ns=start, end_ns=end)
    content = canonical_content_sha256(frame)
    evidence = synthetic_partition_evidence(scope, content_refs=(content,), **overrides)
    return evidence, content


def test_positive_claim_requires_full_scope_equality() -> None:
    evidence, content = _evidence()
    manifest = evidence.gap_manifest
    assert manifest is not None
    # the manifest describes the same partition key + date but a DIFFERENT
    # expected span (a scope field outside the two the old check compared)
    other_scope = manifest.payload.scope.model_copy(
        update={"partition_expected_end_ts": manifest.payload.scope.partition_expected_end_ts + 1}
    )
    with pytest.raises(ValueError, match="scope"):
        Mbp1PartitionEvidence(
            scope=other_scope,
            gap_manifest=manifest,
            completeness_report=evidence.completeness_report,
            recovery_boundaries=evidence.recovery_boundaries,
            channel_map_verified=evidence.channel_map_verified,
            dataset_condition=evidence.dataset_condition.model_copy(
                update={"utc_date": other_scope.utc_date}
            )
            if evidence.dataset_condition is not None
            else None,
            provenance=evidence.provenance,
        )
    # the exact evidence computes with the complete content refs …
    computation = compute_partition_coverage(
        evidence, trading_day=FIXTURE_DAY, partition_content_refs=(content,)
    )
    assert computation.completeness_status.value == "evidenced_complete"
    # … a positive claim without the refs is refused (never a pure diagnostic)
    with pytest.raises(ValueError, match="partition content"):
        compute_partition_coverage(evidence, trading_day=FIXTURE_DAY, partition_content_refs=None)
    # … and a ref set that merely INTERSECTS the certified refs is refused:
    # equality with the complete ordered partition content refs is required
    with pytest.raises(ValueError, match="equal"):
        compute_partition_coverage(
            evidence, trading_day=FIXTURE_DAY, partition_content_refs=(content, "b" * 64)
        )
    with pytest.raises(ValueError, match="equal"):
        compute_partition_coverage(
            evidence, trading_day=FIXTURE_DAY, partition_content_refs=("b" * 64,)
        )


def test_mbp1_window_specs_serialize_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        spec = R5B_WINDOW_SPECS[0]
        changed = spec.model_copy(
            update={
                "trigger_semantics": WindowTriggerSemantics.PRE_TRIGGER_EXCLUSIVE,
                "lower_bound": IntervalBound.CLOSED,
            }
        )
        dumped = changed.model_dump(mode="json")
        assert dumped["trigger_semantics"] == "pre_trigger_exclusive"
        assert dumped["lower_bound"] == "closed"
        rebuilt = Mbp1FeatureWindowSpec.model_validate(dumped)
        assert rebuilt.trigger_semantics is WindowTriggerSemantics.PRE_TRIGGER_EXCLUSIVE
        # a raw string in an enum-typed field is refused by the copy seam
        with pytest.raises(TypeError, match="enum"):
            spec.model_copy(update={"trigger_semantics": "pre_trigger_exclusive"})
