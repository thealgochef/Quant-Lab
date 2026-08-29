"""Offline MBP-1 feature materializer (R5B deliverable 5) — research-only.

Materializes the activated ``IFVG_ORDER_FLOW_MBP1_V1`` block over one
immutable MBP-1 source artifact and one candidate stage-anchor frame. Every
candidate row is PRESERVED: missing or ambiguous evidence becomes typed
nulls with a registered reason — the cohort never changes (acceptance
§7A.19.14). Windows follow the registered ``Mbp1FeatureWindowSpec``s
exactly; admission runs on the complete four-part order key through
:mod:`mbp1_stage_windows`; no ``+inf`` bound exists anywhere.

Typed-missing precedence (deterministic, day → anchor → boundary → content;
coverage policy v2 — R5B.1):

1. ``no_mbp1_partition``   — the candidate's day is not covered
2. ``coverage_evidence_unavailable`` — the day has no partition-scope
   completeness evidence (``completeness_unknown``; fail closed)
3. ``coverage_below_threshold`` — evidence-based day coverage under the policy
4. ``stage_outside_coverage``   — anchor missing, or outside the evidence span
5. ``same_timestamp_order_unavailable`` — unorderable boundary ties
6. ``declared_source_gap`` — a VERIFIED declared/uncertainty interval
   intersects the window (raw sequence jumps never type this)
7. ``instrument_roll_boundary`` — admitted events span >1 instrument id
8. ``minimum_event_count_not_met``

Formulas (``ifvg_order_flow_mbp1_formula_v2`` — numerically identical to
v1; the version bump records the coverage-semantics change): snapshots read the LAST
admitted event's top-of-book state; transition aggregates run over the
admitted events only — OFI uses the Cont–Kukanov–Stoikov event formula on
consecutive admitted pairs, aggressor fractions read Databento trade
``side`` ('B' = buy aggressor), intensities divide by the anchor-to-anchor
window span in seconds. Formula edge cases (zero denominators, zero trades)
are NaN with the window still VALID — NaN-by-formula is not missing
evidence.

The activated block is offline/research-only (owner decision R-6): nothing
here can become a live model feature, execution gate, or Trade-Lab serving
feature without a later Strategy-Core formula/parity contract and a
separately approved sequential model-gated replay.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc
from pydantic import Field

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from ..search.store import (
    load_sidecar_bytes,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .mbp1_arrow_schemas import (
    MBP1_FEATURE_TABLE_SCHEMA,
    MBP1_FEATURE_TABLE_SCHEMA_HASH,
    MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA,
    MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA_HASH,
    mbp1_window_missing_reason_fields,
    mbp1_window_validity_fields,
)
from .mbp1_coverage_evidence import Mbp1CompletenessStatus
from .mbp1_source_artifact import (
    Mbp1SourceArtifactEnvelope,
    day_coverage_views,
    load_partition_events,
)
from .mbp1_source_contract import (
    MBP1_FORMULA_VERSION,
    MBP1_MATERIALIZER_VERSION,
    MBP1_SNAPSHOT_METRICS,
    MBP1_TRANSITION_METRICS,
    Mbp1FeatureWindowSpec,
    StageEvidenceCutoff,
    mbp1_feature_names,
)
from .mbp1_stage_windows import (
    COMPLETED_BAR_CUTOFF_POLICY_ID,
    STAGE_ANCHOR_COLUMNS,
    stage_cutoffs_from_candidate_row,
    ts_utc_to_ns,
    window_admission,
)

__all__ = [
    "MBP1_FEATURE_ARTIFACT_STORE",
    "Mbp1FeatureArtifactPayload",
    "Mbp1FeatureArtifactEnvelope",
    "candidate_anchor_hash",
    "materialize_mbp1_features",
    "verify_mbp1_feature_frame",
    "save_mbp1_feature_artifact",
    "load_mbp1_feature_artifact",
    "load_feature_frame",
    "load_stage_evidence_frame",
]

MBP1_FEATURE_ARTIFACT_STORE = "mbp1_feature_artifacts"

_ANCHOR_COLUMNS = ("candidate_id", "setup_id", "trading_day", *STAGE_ANCHOR_COLUMNS.values())


class Mbp1FeatureArtifactPayload(FrozenContract):
    """The materialization recipe identity — never the produced table's hash."""

    mbp1_source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    resolved_feature_block_id: str = Field(pattern=SHA256_PATTERN)
    candidate_anchor_hash: str = Field(pattern=SHA256_PATTERN)
    cutoff_policy_id: str
    formula_version: str
    materializer_version: str
    feature_table_schema_hash: str = Field(pattern=SHA256_PATTERN)
    stage_window_evidence_schema_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=0)


class Mbp1FeatureArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "mbp1_feature_artifact_id"

    mbp1_feature_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: Mbp1FeatureArtifactPayload
    #: post-materialization facts (R3 DECISIONS_TAKEN #21): they bind the
    #: produced tables to the envelope without entering the pre-run identity
    feature_table_sha256: str = Field(pattern=SHA256_PATTERN)
    stage_evidence_sha256: str = Field(pattern=SHA256_PATTERN)


def candidate_anchor_hash(anchors: pd.DataFrame) -> str:
    """Order-invariant canonical hash of the exact anchor evidence.

    Covers every column the materialization consumes AND emits — including
    ``setup_id`` (review F8): two anchor frames differing only in setup
    labeling are two materializations.
    """

    records = sorted(
        (
            str(row["candidate_id"]),
            str(row["setup_id"]),
            str(row["trading_day"]),
            *(
                str(row[column]) if not pd.isna(row[column]) else None
                for column in STAGE_ANCHOR_COLUMNS.values()
            ),
        )
        for _, row in anchors.iterrows()
    )
    return canonical_contract_sha256({"anchors": records})


def _verify_events_against_source(
    source: Mbp1SourceArtifactEnvelope,
    events_by_day: Mapping[str, pd.DataFrame],
    anchor_days: set[str],
) -> None:
    """Review F1: supplied evidence must BE the source artifact's evidence.

    Every supplied day's canonical bytes must hash to that day's coverage
    ``content_sha256``; a supplied day the artifact does not cover is
    unverifiable and refused; an ANCHOR day the artifact covers must be
    supplied — materializing it as ``no_mbp1_partition`` while the artifact
    claims coverage would be untruthful.
    """

    from .mbp1_source_artifact import _canonical_event_bytes  # noqa: PLC0415

    coverage_by_day = {row.trading_day: row for row in source.payload.ordered_partitions}
    unknown = sorted(set(events_by_day) - set(coverage_by_day))
    # every physical partition of one day shares the day's canonical bytes
    if unknown:
        raise ValueError(
            f"supplied events for days the source artifact does not cover: {unknown}"
        )
    for day, frame in events_by_day.items():
        digest = hashlib.sha256(_canonical_event_bytes(frame)).hexdigest()
        if digest != coverage_by_day[day].content_sha256:
            raise ValueError(
                f"supplied events for {day} do not hash to the source "
                "artifact's coverage content_sha256 — the evidence is not "
                "the artifact's evidence (refused; provenance is verified, "
                "never asserted)"
            )
    withheld = sorted((anchor_days & set(coverage_by_day)) - set(events_by_day))
    if withheld:
        raise ValueError(
            "the source artifact covers anchor days whose events were not "
            f"supplied: {withheld} — typing them no_mbp1_partition would be "
            "untruthful"
        )


def _snapshot_features(admitted: pd.DataFrame, spec: Mbp1FeatureWindowSpec) -> dict[str, float]:
    last = admitted.iloc[-1]
    bid_px = float(last["bid_px_ticks"])
    ask_px = float(last["ask_px_ticks"])
    bid_sz = float(last["bid_sz"])
    ask_sz = float(last["ask_sz"])
    bid_ct = float(last["bid_ct"])
    ask_ct = float(last["ask_ct"])
    size_total = bid_sz + ask_sz
    count_total = bid_ct + ask_ct
    mid = (bid_px + ask_px) / 2.0
    microprice = (
        (ask_px * bid_sz + bid_px * ask_sz) / size_total if size_total > 0 else np.nan
    )
    values = {
        "spread_ticks": ask_px - bid_px,
        "queue_imbalance": ((bid_sz - ask_sz) / size_total) if size_total > 0 else np.nan,
        "order_count_imbalance": (
            (bid_ct - ask_ct) / count_total if count_total > 0 else np.nan
        ),
        "microprice_offset_ticks": (microprice - mid) if size_total > 0 else np.nan,
        "bid_sz": bid_sz,
        "ask_sz": ask_sz,
        "bid_ct": bid_ct,
        "ask_ct": ask_ct,
    }
    if set(values) != set(MBP1_SNAPSHOT_METRICS):
        raise AssertionError("snapshot formula set drifted from the registered metrics")
    return {
        f"{spec.feature_window_key}_{metric}": values[metric]
        for metric in MBP1_SNAPSHOT_METRICS
    }


def _ofi_sum(admitted: pd.DataFrame) -> float:
    """Cont–Kukanov–Stoikov OFI over consecutive ADMITTED pairs."""

    if len(admitted) < 2:
        return 0.0
    b_px = admitted["bid_px_ticks"].to_numpy(dtype=float)
    a_px = admitted["ask_px_ticks"].to_numpy(dtype=float)
    b_sz = admitted["bid_sz"].to_numpy(dtype=float)
    a_sz = admitted["ask_sz"].to_numpy(dtype=float)
    total = 0.0
    for i in range(1, len(admitted)):
        e = 0.0
        if b_px[i] >= b_px[i - 1]:
            e += b_sz[i]
        if b_px[i] <= b_px[i - 1]:
            e -= b_sz[i - 1]
        if a_px[i] <= a_px[i - 1]:
            e -= a_sz[i]
        if a_px[i] >= a_px[i - 1]:
            e += a_sz[i - 1]
        total += e
    return float(total)


def _transition_features(
    admitted: pd.DataFrame,
    spec: Mbp1FeatureWindowSpec,
    *,
    window_seconds: float | None,
) -> dict[str, float]:
    trades = admitted[admitted["action"].astype(str) == "T"]
    trade_count = int(len(trades))
    quote_count = int(len(admitted)) - trade_count
    depletion = 0
    replenishment = 0
    if len(admitted) >= 2:
        b_px = admitted["bid_px_ticks"].to_numpy(dtype=float)
        a_px = admitted["ask_px_ticks"].to_numpy(dtype=float)
        b_sz = admitted["bid_sz"].to_numpy(dtype=float)
        a_sz = admitted["ask_sz"].to_numpy(dtype=float)
        for i in range(1, len(admitted)):
            if b_px[i] < b_px[i - 1] or (b_px[i] == b_px[i - 1] and b_sz[i] < b_sz[i - 1]):
                depletion += 1
            if a_px[i] > a_px[i - 1] or (a_px[i] == a_px[i - 1] and a_sz[i] < a_sz[i - 1]):
                depletion += 1
            if b_px[i] > b_px[i - 1] or (b_px[i] == b_px[i - 1] and b_sz[i] > b_sz[i - 1]):
                replenishment += 1
            if a_px[i] < a_px[i - 1] or (a_px[i] == a_px[i - 1] and a_sz[i] > a_sz[i - 1]):
                replenishment += 1
    mid_first = float(
        (admitted["bid_px_ticks"].iloc[0] + admitted["ask_px_ticks"].iloc[0]) / 2.0
    )
    mid_last = float(
        (admitted["bid_px_ticks"].iloc[-1] + admitted["ask_px_ticks"].iloc[-1]) / 2.0
    )
    traded_size = float(trades["size"].sum()) if trade_count else 0.0
    span_ok = window_seconds is not None and window_seconds > 0
    values = {
        "ofi_sum": _ofi_sum(admitted),
        "aggressive_buy_frac": (
            float((trades["side"].astype(str) == "B").sum()) / trade_count
            if trade_count
            else np.nan
        ),
        "aggressive_sell_frac": (
            float((trades["side"].astype(str) == "A").sum()) / trade_count
            if trade_count
            else np.nan
        ),
        "depletion_events": float(depletion),
        "replenishment_events": float(replenishment),
        "absorption_score": traded_size / (1.0 + abs(mid_last - mid_first)),
        "event_count": float(len(admitted)),
        "quote_intensity": (quote_count / window_seconds) if span_ok else np.nan,
        "trade_intensity": (trade_count / window_seconds) if span_ok else np.nan,
    }
    if set(values) != set(MBP1_TRANSITION_METRICS):
        raise AssertionError("transition formula set drifted from the registered metrics")
    return {
        f"{spec.feature_window_key}_{metric}": values[metric]
        for metric in MBP1_TRANSITION_METRICS
    }


def _null_features(spec: Mbp1FeatureWindowSpec) -> dict[str, float]:
    return dict.fromkeys(spec.feature_names, np.nan)


def _boundary_ns(cutoff: StageEvidenceCutoff | None) -> int | None:
    if cutoff is None:
        return None
    if cutoff.exact_source_order_key is not None:
        return int(cutoff.exact_source_order_key[0])
    return ts_utc_to_ns(cutoff.completed_bar_close_ts_utc or cutoff.stage_as_of_ts_utc)


def _validate_window_specs(resolved_block) -> tuple[Mbp1FeatureWindowSpec, ...]:
    # review F8: the resolution's formula/materializer versions ARE the
    # identity this materializer implements — a superseded (v1) resolution
    # can never be stamped with the v2 semantics
    if resolved_block.payload.formula_version != MBP1_FORMULA_VERSION or (
        resolved_block.payload.materializer_version != MBP1_MATERIALIZER_VERSION
    ):
        raise ValueError(
            "resolved block versions "
            f"({resolved_block.payload.formula_version} / "
            f"{resolved_block.payload.materializer_version}) are not the versions this "
            f"materializer implements ({MBP1_FORMULA_VERSION} / "
            f"{MBP1_MATERIALIZER_VERSION}); a superseded resolution is refused"
        )
    specs = tuple(resolved_block.payload.mbp1_feature_windows)
    union: list[str] = []
    for spec in specs:
        union.extend(spec.feature_names)
    if len(union) != len(set(union)):
        raise ValueError("window specs assign at least one feature twice")
    if set(union) != set(mbp1_feature_names()) or set(union) != set(
        resolved_block.payload.feature_names
    ):
        raise ValueError(
            "every order-flow feature must map to exactly one registered "
            "Mbp1FeatureWindowSpec of the resolved block"
        )
    return specs


def materialize_mbp1_features(
    source: Mbp1SourceArtifactEnvelope,
    anchors: pd.DataFrame,
    *,
    resolved_block,
    events_by_day: Mapping[str, pd.DataFrame],
    cutoff_builder: Callable[[pd.Series], dict[str, StageEvidenceCutoff | None]] | None = None,
    cutoff_policy_id: str = COMPLETED_BAR_CUTOFF_POLICY_ID,
) -> tuple[Mbp1FeatureArtifactEnvelope, pd.DataFrame, pd.DataFrame]:
    """One deterministic materialization pass.

    ``events_by_day`` is VERIFIED to be the source artifact's own evidence
    (review F1): every supplied day's canonical bytes must hash to the
    coverage row's ``content_sha256`` before any window computes.
    ``cutoff_builder`` defaults to the completed-bar builder over the v2
    anchors and may be replaced by an exact-key builder where triggering
    events are known (the policy id then identifies that choice).
    """

    for column in _ANCHOR_COLUMNS:
        if column not in anchors.columns:
            raise ValueError(f"anchor frame lacks required column {column!r}")
    if anchors["candidate_id"].duplicated().any():
        raise ValueError("anchor frame has duplicate candidate ids")
    _verify_events_against_source(
        source,
        events_by_day,
        set(anchors["trading_day"].astype(str)) if len(anchors) else set(),
    )
    specs = _validate_window_specs(resolved_block)
    build_cutoffs = cutoff_builder or stage_cutoffs_from_candidate_row
    # policy v2: the DAY view aggregates every physical partition of the
    # trading day (duration-weighted coverage, weakest status, unioned
    # verified intervals) — the source of every day-level typed reason
    coverage_by_day = day_coverage_views(source)
    min_coverage = float(
        dict(source.payload.source_contract.coverage_policy).get(
            "min_day_coverage_fraction", 0.0
        )
    )
    feature_rows: list[dict] = []
    evidence_rows: list[dict] = []
    for _, anchor_row in anchors.sort_values("candidate_id").iterrows():
        candidate_id = str(anchor_row["candidate_id"])
        day = str(anchor_row["trading_day"])
        row: dict = {
            "candidate_id": candidate_id,
            "setup_id": str(anchor_row["setup_id"]),
            "trading_day": day,
        }
        coverage = coverage_by_day.get(day)
        day_events = events_by_day.get(day)
        day_reason: str | None = None
        if coverage is None or day_events is None or coverage.row_count == 0:
            day_reason = "no_mbp1_partition"
        elif coverage.completeness_status is Mbp1CompletenessStatus.COMPLETENESS_UNKNOWN:
            # no partition-scope completeness evidence (or a downgrading
            # dataset condition): the day's windows are typed, never
            # inferred complete from raw sequence continuity (fail closed)
            day_reason = "coverage_evidence_unavailable"
        elif coverage.coverage_fraction < min_coverage:
            day_reason = "coverage_below_threshold"
        cutoffs = build_cutoffs(anchor_row)
        for spec in specs:
            reason: str | None = day_reason
            admitted_count = 0
            ambiguous = False
            to_cutoff = cutoffs.get(spec.to_stage)
            from_cutoff = cutoffs.get(spec.from_stage) if spec.from_stage else None
            from_ns = _boundary_ns(from_cutoff)
            to_ns = _boundary_ns(to_cutoff)
            anchors_missing = to_cutoff is None or (
                spec.from_stage is not None and from_cutoff is None
            )
            outside_span = (
                coverage is not None
                and (
                    coverage.first_ts_event is None
                    or to_ns is None
                    or to_ns < coverage.first_ts_event
                    or to_ns > coverage.last_ts_event
                    or (
                        from_ns is not None
                        and (
                            from_ns < coverage.first_ts_event
                            or from_ns > coverage.last_ts_event
                        )
                    )
                )
            )
            if reason is None and (anchors_missing or outside_span):
                reason = "stage_outside_coverage"
            if reason is None:
                admission = window_admission(
                    day_events, spec, from_cutoff=from_cutoff, to_cutoff=to_cutoff
                )
                admitted_count = admission.admitted_count
                ambiguous = admission.same_timestamp_ambiguous
                if ambiguous:
                    reason = "same_timestamp_order_unavailable"
            window_start = from_ns if from_ns is not None else (
                coverage.first_ts_event if coverage is not None else None
            )
            # review F2: a window that touches session time NO declared
            # physical partition evidences has no completeness evidence
            if reason is None and any(
                hole_start < to_ns and hole_end > window_start
                for hole_start, hole_end in coverage.uncovered_session_intervals
            ):
                reason = "coverage_evidence_unavailable"
            # a window whose span intersects a VERIFIED declared or
            # uncertainty interval is typed — never widened, never imputed
            if reason is None and any(
                gap_start < to_ns and gap_end > window_start
                for gap_start, gap_end in coverage.gap_intervals
            ):
                reason = "declared_source_gap"
            if reason is None:
                admitted = day_events.loc[admission.mask]
                if admitted["instrument_id"].nunique() > 1:
                    reason = "instrument_roll_boundary"
                elif admitted_count < spec.minimum_event_count:
                    reason = "minimum_event_count_not_met"
            if reason is None:
                if spec.from_stage is None:
                    row.update(_snapshot_features(admitted, spec))
                else:
                    window_seconds = (
                        (to_ns - from_ns) / 1e9
                        if to_ns is not None and from_ns is not None
                        else None
                    )
                    row.update(
                        _transition_features(admitted, spec, window_seconds=window_seconds)
                    )
                row[f"{spec.feature_window_key}_valid"] = True
                row[f"{spec.feature_window_key}_missing_reason"] = None
            else:
                row.update(_null_features(spec))
                row[f"{spec.feature_window_key}_valid"] = False
                row[f"{spec.feature_window_key}_missing_reason"] = reason
            evidence_rows.append(
                {
                    "candidate_id": candidate_id,
                    "trading_day": day,
                    "feature_window_key": spec.feature_window_key,
                    "from_stage": spec.from_stage or "",
                    "to_stage": spec.to_stage,
                    "trigger_semantics": spec.trigger_semantics.value,
                    "cutoff_kind": (
                        to_cutoff.cutoff_kind.value if to_cutoff is not None else ""
                    ),
                    "from_ts_utc": (
                        from_cutoff.stage_as_of_ts_utc if from_cutoff is not None else ""
                    ),
                    "to_ts_utc": (
                        to_cutoff.stage_as_of_ts_utc if to_cutoff is not None else ""
                    ),
                    "admitted_event_count": admitted_count,
                    "same_timestamp_ambiguous": ambiguous,
                    "valid": reason is None,
                    "missing_reason": reason or "",
                }
            )
        feature_rows.append(row)

    ordered_columns = [field.name for field in MBP1_FEATURE_TABLE_SCHEMA]
    feature_frame = pd.DataFrame(feature_rows)
    if feature_frame.empty:
        feature_frame = pd.DataFrame(columns=ordered_columns)
    feature_frame = feature_frame.loc[:, ordered_columns]
    evidence_columns = [field.name for field in MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA]
    evidence_frame = pd.DataFrame(evidence_rows)
    if evidence_frame.empty:
        evidence_frame = pd.DataFrame(columns=evidence_columns)
    evidence_frame = evidence_frame.loc[:, evidence_columns]

    payload = Mbp1FeatureArtifactPayload(
        mbp1_source_artifact_id=source.mbp1_source_artifact_id,
        resolved_feature_block_id=resolved_block.resolved_feature_block_id,
        candidate_anchor_hash=candidate_anchor_hash(anchors),
        cutoff_policy_id=cutoff_policy_id,
        # stamped FROM the validated resolution (review F8)
        formula_version=resolved_block.payload.formula_version,
        materializer_version=resolved_block.payload.materializer_version,
        feature_table_schema_hash=MBP1_FEATURE_TABLE_SCHEMA_HASH,
        stage_window_evidence_schema_hash=MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA_HASH,
        candidate_count=int(len(feature_frame)),
    )
    feature_bytes = _table_bytes(feature_frame, MBP1_FEATURE_TABLE_SCHEMA)
    evidence_bytes = _table_bytes(evidence_frame, MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA)
    envelope = Mbp1FeatureArtifactEnvelope.from_payload(
        payload,
        feature_table_sha256=hashlib.sha256(feature_bytes).hexdigest(),
        stage_evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
    )
    return envelope, feature_frame, evidence_frame


def _table_bytes(frame: pd.DataFrame, schema: pa.Schema) -> bytes:
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    sink = BytesIO()
    with pyarrow.ipc.new_file(sink, schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _frame_from_bytes(data: bytes) -> pd.DataFrame:
    with pyarrow.ipc.open_file(BytesIO(data)) as reader:
        return reader.read_all().to_pandas()


_FEATURES_SIDECAR = "mbp1_features.arrow"
_EVIDENCE_SIDECAR = "mbp1_stage_evidence.arrow"


def save_mbp1_feature_artifact(
    root: Path,
    envelope: Mbp1FeatureArtifactEnvelope,
    feature_frame: pd.DataFrame,
    evidence_frame: pd.DataFrame,
) -> tuple:
    feature_bytes = _table_bytes(feature_frame, MBP1_FEATURE_TABLE_SCHEMA)
    evidence_bytes = _table_bytes(evidence_frame, MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA)
    if hashlib.sha256(feature_bytes).hexdigest() != envelope.feature_table_sha256:
        raise ValueError("feature frame does not hash to the envelope's table hash")
    if hashlib.sha256(evidence_bytes).hexdigest() != envelope.stage_evidence_sha256:
        raise ValueError("evidence frame does not hash to the envelope's evidence hash")
    return save_or_reuse_envelope(
        Path(root),
        MBP1_FEATURE_ARTIFACT_STORE,
        envelope,
        extra_files={
            _FEATURES_SIDECAR: feature_bytes,
            _EVIDENCE_SIDECAR: evidence_bytes,
        },
    )


def load_mbp1_feature_artifact(root: Path, artifact_id: str) -> Mbp1FeatureArtifactEnvelope:
    return load_verified_envelope(
        Path(root), MBP1_FEATURE_ARTIFACT_STORE, artifact_id, Mbp1FeatureArtifactEnvelope
    )


def verify_mbp1_feature_frame(
    envelope: Mbp1FeatureArtifactEnvelope, feature_frame: pd.DataFrame
) -> None:
    """Review F1 (seam half): the frame IS the artifact's table, or refuse.

    Every consumer that pins ``mbp1_feature_artifact_id`` next to a frame
    (bundle views, the controlled study) calls this first — the binding is
    verified by rehash, never asserted by the caller.
    """

    data = _table_bytes(feature_frame, MBP1_FEATURE_TABLE_SCHEMA)
    if hashlib.sha256(data).hexdigest() != envelope.feature_table_sha256:
        raise ValueError(
            "the supplied MBP-1 feature frame does not hash to the pinned "
            "artifact's feature_table_sha256 — the evidence binding is "
            "verified, never asserted"
        )


def load_feature_frame(root: Path, envelope: Mbp1FeatureArtifactEnvelope) -> pd.DataFrame:
    data = load_sidecar_bytes(
        Path(root),
        MBP1_FEATURE_ARTIFACT_STORE,
        envelope.mbp1_feature_artifact_id,
        _FEATURES_SIDECAR,
    )
    if hashlib.sha256(data).hexdigest() != envelope.feature_table_sha256:
        raise ValueError("stored feature table fails the envelope hash check")
    return _frame_from_bytes(data)


def load_stage_evidence_frame(
    root: Path, envelope: Mbp1FeatureArtifactEnvelope
) -> pd.DataFrame:
    data = load_sidecar_bytes(
        Path(root),
        MBP1_FEATURE_ARTIFACT_STORE,
        envelope.mbp1_feature_artifact_id,
        _EVIDENCE_SIDECAR,
    )
    if hashlib.sha256(data).hexdigest() != envelope.stage_evidence_sha256:
        raise ValueError("stored stage evidence fails the envelope hash check")
    return _frame_from_bytes(data)


def materialize_from_stored_source(
    root: Path,
    source: Mbp1SourceArtifactEnvelope,
    anchors: pd.DataFrame,
    *,
    resolved_block,
    **kwargs,
) -> tuple[Mbp1FeatureArtifactEnvelope, pd.DataFrame, pd.DataFrame]:
    """Materialize over a STORED source artifact's own verified event bytes."""

    events_by_day = {
        partition.trading_day: load_partition_events(root, source, partition.trading_day)
        for partition in source.payload.ordered_partitions
    }
    return materialize_mbp1_features(
        source,
        anchors,
        resolved_block=resolved_block,
        events_by_day=events_by_day,
        **kwargs,
    )


def _example_feature_artifact_payload() -> Mbp1FeatureArtifactPayload:
    return Mbp1FeatureArtifactPayload(
        mbp1_source_artifact_id="a" * 64,
        resolved_feature_block_id="b" * 64,
        candidate_anchor_hash="c" * 64,
        cutoff_policy_id=COMPLETED_BAR_CUTOFF_POLICY_ID,
        formula_version=MBP1_FORMULA_VERSION,
        materializer_version=MBP1_MATERIALIZER_VERSION,
        feature_table_schema_hash=MBP1_FEATURE_TABLE_SCHEMA_HASH,
        stage_window_evidence_schema_hash=MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA_HASH,
        candidate_count=0,
    )


register_identity_pair(
    name="Mbp1FeatureArtifact",
    envelope_cls=Mbp1FeatureArtifactEnvelope,
    payload_cls=Mbp1FeatureArtifactPayload,
    id_field="mbp1_feature_artifact_id",
    example_factory=_example_feature_artifact_payload,
    extra_envelope_fields=("feature_table_sha256", "stage_evidence_sha256"),
)

# static guard: the schema-declared evidence columns exist for every window
_declared = {*mbp1_window_validity_fields(), *mbp1_window_missing_reason_fields()}
_schema_names = {field.name for field in MBP1_FEATURE_TABLE_SCHEMA}
if not _declared.issubset(_schema_names):
    raise AssertionError("feature-table schema lost the per-window evidence columns")
