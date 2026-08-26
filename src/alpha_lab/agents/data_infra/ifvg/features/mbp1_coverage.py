"""MBP-1 feature coverage/validity reports (R5B deliverables 6 and 9).

One report freezes, per materialization: per-day source coverage (rows,
gaps, coverage fraction), per-window validity with typed missing-reason
counts, and per-feature non-null fractions. The report pins the exact
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
from .mbp1_feature_materializer import Mbp1FeatureArtifactEnvelope
from .mbp1_source_artifact import Mbp1SourceArtifactEnvelope
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
    row_count: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    sequence_gap_count: int = Field(ge=0)
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
            trading_day=partition.trading_day,
            row_count=partition.row_count,
            coverage_fraction=partition.coverage_fraction,
            sequence_gap_count=len(partition.sequence_gap_intervals),
            first_ts_utc=ns_to_ts_utc(partition.first_ts_event),
            last_ts_utc=ns_to_ts_utc(partition.last_ts_event),
        )
        for partition in source.payload.ordered_partitions
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
