"""MBP-1 feature coverage/validity reports (R5B deliverables 6 and 9).

One report freezes, per materialization: per-day EVIDENCE-BASED source
coverage (policy v2 — rows, completeness status, verified gap intervals,
open-uncertainty facts, the dataset-condition status, and the raw
sequence-jump DIAGNOSTIC count), per-window validity with typed
missing-reason counts, and per-feature non-null fractions. The report pins the exact
source and feature artifact ids it describes, carries the permanent
``research_only_offline`` boundary stamp (owner decision R-6), and is
content-addressed — identical coverage evidence reuses one immutable
artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from pydantic import Field

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from ..search.store import load_verified_envelope, save_or_reuse_envelope
from .mbp1_arrow_schemas import mbp1_window_keys
from .mbp1_coverage_evidence import (
    MBP1_COVERAGE_POLICY_V2,
    Mbp1CompletenessStatus,
    Mbp1DatasetConditionStatus,
)
from .mbp1_feature_materializer import Mbp1FeatureArtifactEnvelope
from .mbp1_source_artifact import Mbp1SourceArtifactEnvelope, day_coverage_views
from .mbp1_source_contract import MBP1_MISSING_REASONS, mbp1_feature_names
from .mbp1_stage_windows import ns_to_ts_utc

__all__ = [
    "MBP1_COVERAGE_REPORT_STORE",
    "Mbp1DayCoverageRow",
    "Mbp1WindowCoverageRow",
    "Mbp1CoverageReportPayload",
    "Mbp1CoverageReportEnvelope",
    "build_mbp1_coverage_report",
    "save_mbp1_coverage_report",
    "load_mbp1_coverage_report",
]

MBP1_COVERAGE_REPORT_STORE = "mbp1_coverage_reports"


class Mbp1DayCoverageRow(FrozenContract):
    trading_day: str
    partition_count: int = Field(ge=1)
    row_count: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    completeness_status: Mbp1CompletenessStatus
    dataset_condition_status: Mbp1DatasetConditionStatus
    declared_gap_count: int = Field(ge=0)
    open_uncertainty_to_partition_end: bool
    #: DIAGNOSTIC only — never coverage evidence (policy v2)
    sequence_positive_jump_count: int = Field(ge=0)
    first_ts_utc: str | None
    last_ts_utc: str | None


class Mbp1WindowCoverageRow(FrozenContract):
    feature_window_key: str
    valid_count: int = Field(ge=0)
    invalid_count: int = Field(ge=0)
    reason_counts: ImmutableMap[str, int]


class Mbp1CoverageReportPayload(FrozenContract):
    mbp1_source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    mbp1_feature_artifact_id: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=0)
    day_rows: tuple[Mbp1DayCoverageRow, ...]
    window_rows: tuple[Mbp1WindowCoverageRow, ...]
    per_feature_nonnull_fraction: ImmutableMap[str, float]
    coverage_policy_id: Literal["mbp1_source_coverage_declared_evidence_v2"] = (
        MBP1_COVERAGE_POLICY_V2
    )
    research_boundary: Literal["research_only_offline"] = "research_only_offline"


class Mbp1CoverageReportEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "mbp1_coverage_report_id"

    mbp1_coverage_report_id: str = Field(pattern=SHA256_PATTERN)
    payload: Mbp1CoverageReportPayload


def build_mbp1_coverage_report(
    source: Mbp1SourceArtifactEnvelope,
    feature_artifact: Mbp1FeatureArtifactEnvelope,
    feature_frame: pd.DataFrame,
    evidence_frame: pd.DataFrame,
) -> Mbp1CoverageReportEnvelope:
    day_rows = tuple(
        Mbp1DayCoverageRow(
            trading_day=view.trading_day,
            partition_count=view.partition_count,
            row_count=view.row_count,
            coverage_fraction=view.coverage_fraction,
            completeness_status=view.completeness_status,
            dataset_condition_status=view.dataset_condition_status,
            declared_gap_count=len(view.gap_intervals),
            open_uncertainty_to_partition_end=view.open_uncertainty_to_partition_end,
            sequence_positive_jump_count=view.sequence_positive_jump_count,
            first_ts_utc=ns_to_ts_utc(view.first_ts_event),
            last_ts_utc=ns_to_ts_utc(view.last_ts_event),
        )
        for _day, view in sorted(day_coverage_views(source).items())
    )
    window_rows: list[Mbp1WindowCoverageRow] = []
    for key in mbp1_window_keys():
        scoped = evidence_frame[evidence_frame["feature_window_key"] == key]
        valid_count = int(scoped["valid"].sum()) if len(scoped) else 0
        reasons = (
            scoped.loc[~scoped["valid"], "missing_reason"].value_counts().to_dict()
            if len(scoped)
            else {}
        )
        unknown = set(reasons) - set(MBP1_MISSING_REASONS)
        if unknown:
            raise ValueError(f"unregistered missing reasons in evidence: {sorted(unknown)}")
        window_rows.append(
            Mbp1WindowCoverageRow(
                feature_window_key=key,
                valid_count=valid_count,
                invalid_count=int(len(scoped)) - valid_count,
                reason_counts={str(k): int(v) for k, v in sorted(reasons.items())},
            )
        )
    fractions = {
        name: (
            float(feature_frame[name].notna().mean()) if len(feature_frame) else 0.0
        )
        for name in mbp1_feature_names()
        if name in feature_frame.columns
    }
    payload = Mbp1CoverageReportPayload(
        mbp1_source_artifact_id=source.mbp1_source_artifact_id,
        mbp1_feature_artifact_id=feature_artifact.mbp1_feature_artifact_id,
        candidate_count=int(len(feature_frame)),
        day_rows=day_rows,
        window_rows=tuple(window_rows),
        per_feature_nonnull_fraction=fractions,
    )
    return Mbp1CoverageReportEnvelope.from_payload(payload)


def save_mbp1_coverage_report(root: Path, envelope: Mbp1CoverageReportEnvelope) -> tuple:
    return save_or_reuse_envelope(Path(root), MBP1_COVERAGE_REPORT_STORE, envelope)


def load_mbp1_coverage_report(root: Path, report_id: str) -> Mbp1CoverageReportEnvelope:
    return load_verified_envelope(
        Path(root), MBP1_COVERAGE_REPORT_STORE, report_id, Mbp1CoverageReportEnvelope
    )


def _example_coverage_payload() -> Mbp1CoverageReportPayload:
    return Mbp1CoverageReportPayload(
        mbp1_source_artifact_id="a" * 64,
        mbp1_feature_artifact_id="b" * 64,
        candidate_count=0,
        day_rows=(),
        window_rows=(),
        per_feature_nonnull_fraction={},
    )


register_identity_pair(
    name="Mbp1CoverageReport",
    envelope_cls=Mbp1CoverageReportEnvelope,
    payload_cls=Mbp1CoverageReportPayload,
    id_field="mbp1_coverage_report_id",
    example_factory=_example_coverage_payload,
)
