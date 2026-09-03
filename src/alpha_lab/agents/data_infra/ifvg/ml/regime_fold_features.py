"""Fold-local regime features — the ONLY supervised regime feature source
(R6.1 D7 / D8; plan §6.G "S09b").

Two regime-assignment artifacts exist with different uses (D7). The
descriptive ``RegimeOosAssignmentArtifact`` (``regime_oos_assignment.py``)
serves stratification only. THIS artifact — keyed by (fold, candidate,
partition, fit) — is the only feature source a supervised model may read
(the ladder never opens the descriptive artifact; test-enforced).

For every VALID supervised fold ``k`` whose regime fit ``k`` exists, the
fold's TRAIN rows receive fit ``k``'s in-sample assignment (``partition =
"train"``) and its TEST rows receive fit ``k``'s out-of-sample assignment
(``partition = "test"``). The candidate grain reads fit ``k``'s own
assignment rows by ``candidate_id``; the panel grain applies the normative
PIT rule (``assign_panel_regimes_to_candidates`` with
``partition_for_candidate`` — the candidate's own partition in fold ``k``
from fit ``k`` only). No row or alignment from another fold is consulted:
perturbing every observation after fold ``k``'s test window, or every other
fit's rows, cannot change fold ``k``'s rows (test-pinned).

Model-facing columns are FIT-LOCAL (D8): ``ctx_regime_<p12>_local_distance_<i>``
(one per centroid of the fixed ``k``), ``…_local_assigned_distance``,
``…_local_margin`` (d2 − d1) and the block-declared categorical
``…_local_id`` — ``<p12>`` is the first twelve hex characters of the
resolved regime protocol id, so features of different protocols never
collide. ``canonical_reporting_cluster_id`` is a SEPARATE reporting-only
column that no model runner selects (canonical alignment is
reporting-only until a causal earlier-reference-only mapping is separately
contracted — D7/final closure #6). ``hard_id_encoding`` decides whether the
local id is a MODEL input (``fit_local_categorical_v1``) or a stratification /
reporting field only (``"none"`` — the stamped default); the column itself
is always materialized so ``cohort_model`` can stratify on it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import Field, model_validator

from ..context_folds import ContextFoldSet
from ..features.arrow_tables import (
    arrow_schema_hash,
    bytes_sha256,
    frame_from_arrow_bytes,
    frame_to_arrow_bytes,
)
from ..features.context_bar_panel_contract import (
    PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS,
    PANEL_ASSIGNMENT_MISSING_REASONS,
)
from ..features.feature_blocks import AvailabilityStage
from ..fold_schedules import FoldScheduleEnvelope
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .fold_set_artifact import (
    FoldSetArtifactEnvelope,
    assert_same_fold_schedule,
)
from .fold_set_artifact import (
    fold_set_id as _legacy_fold_set_id,
)
from .regime_contracts import (
    REGIME_ASSIGNMENT_MISSING_REASONS,
    ObservationGranularity,
    RegimeAssignmentEvidenceError,
    RegimeProtocolEnvelope,
    validate_native_values,
)
from .regime_oos_assignment import (
    PanelAssignmentContext,
    assign_panel_regimes_to_candidates,
    candidate_as_of_frame,
)

__all__ = [
    "REGIME_FOLD_FEATURE_STORE",
    "FOLD_FEATURE_SIDECAR",
    "REGIME_FOLD_FEATURE_FORMULA_VERSION",
    "REGIME_FOLD_FEATURE_MATERIALIZER_VERSION",
    "HARD_ID_ENCODINGS",
    "HARD_ID_ENCODING_NONE",
    "HARD_ID_ENCODING_CATEGORICAL",
    "REGIME_FOLD_FEATURE_MISSING_REASONS",
    "HardIdEncoding",
    "FoldFitRef",
    "RegimeFoldFeatureArtifactPayload",
    "RegimeFoldFeatureArtifactEnvelope",
    "PanelFoldFeatureInputs",
    "RegimeFoldFeatureColumns",
    "RegimeFoldFeatureFrameSource",
    "regime_feature_prefix",
    "fold_feature_columns",
    "model_feature_names",
    "categorical_model_features",
    "fold_feature_schema",
    "build_regime_fold_features",
    "fold_feature_table_bytes",
    "fold_feature_native_spec",
    "validate_fold_feature_rows",
    "assert_fold_feature_spine_bound",
    "verify_regime_fold_feature_frame",
    "verify_fold_fit_refs_against_store",
    "save_regime_fold_features",
    "load_regime_fold_features",
    "load_regime_fold_feature_frame",
    "load_regime_fold_feature_source",
]

REGIME_FOLD_FEATURE_STORE = "regime_fold_features"
FOLD_FEATURE_SIDECAR = "regime_fold_features.arrow"
REGIME_FOLD_FEATURE_FORMULA_VERSION = "ifvg_regime_context_formula_v1"
REGIME_FOLD_FEATURE_MATERIALIZER_VERSION = "regime_fold_feature_materializer_v1"
#: The ONE ``hard_id_encoding`` vocabulary (adversarial R6.1 F11): shared by
#: ``RegimeStudyRequest``, the fold-feature artifact and S09b; the stamped
#: default is ``none`` (``regime_feature_hard_id_encoding_default``).
HARD_ID_ENCODING_NONE = "none"
HARD_ID_ENCODING_CATEGORICAL = "fit_local_categorical_v1"
HARD_ID_ENCODINGS: tuple[str, ...] = (HARD_ID_ENCODING_NONE, HARD_ID_ENCODING_CATEGORICAL)
HardIdEncoding = Literal["none", "fit_local_categorical_v1"]

#: Typed reasons a (fold, candidate) row may carry — the fit's own typed
#: reasons, the panel PIT reasons, and the two fold-feature-specific ones.
REGIME_FOLD_FEATURE_MISSING_REASONS: tuple[str, ...] = (
    "no_valid_regime_fit",
    "no_fit_assignment_row",
    *REGIME_ASSIGNMENT_MISSING_REASONS,
    *PANEL_ASSIGNMENT_MISSING_REASONS,
)

_KEY_COLUMNS: tuple[str, ...] = (
    "fold_index",
    "candidate_id",
    "partition",
    "regime_fit_id",
    "panel_row_id",
)


# ── column naming (D8) ───────────────────────────────────────────────────────


def regime_feature_prefix(resolved_regime_protocol_id: str) -> str:
    """``ctx_regime_<p12>`` — protocol-specific so features never collide."""

    return f"ctx_regime_{str(resolved_regime_protocol_id)[:12]}"


@dataclass(frozen=True, slots=True)
class RegimeFoldFeatureColumns:
    prefix: str
    cluster_count: int

    @property
    def distances(self) -> tuple[str, ...]:
        return tuple(f"{self.prefix}_local_distance_{index}" for index in range(self.cluster_count))

    @property
    def assigned_distance(self) -> str:
        return f"{self.prefix}_local_assigned_distance"

    @property
    def margin(self) -> str:
        return f"{self.prefix}_local_margin"

    @property
    def local_id(self) -> str:
        return f"{self.prefix}_local_id"

    @property
    def valid(self) -> str:
        return f"{self.prefix}_valid"

    @property
    def missing_reason(self) -> str:
        return f"{self.prefix}_missing_reason"

    @property
    def numeric(self) -> tuple[str, ...]:
        return (*self.distances, self.assigned_distance, self.margin)

    @property
    def ordered(self) -> tuple[str, ...]:
        """Every column of the persisted table, in the frozen order."""

        return (
            *_KEY_COLUMNS,
            *self.numeric,
            self.local_id,
            "canonical_reporting_cluster_id",
            self.valid,
            self.missing_reason,
        )


def fold_feature_columns(
    resolved_regime_protocol_id: str, cluster_count: int
) -> RegimeFoldFeatureColumns:
    return RegimeFoldFeatureColumns(
        prefix=regime_feature_prefix(resolved_regime_protocol_id),
        cluster_count=int(cluster_count),
    )


def model_feature_names(
    columns: RegimeFoldFeatureColumns, hard_id_encoding: HardIdEncoding
) -> tuple[str, ...]:
    """The model-facing names: the local distance vector, the assigned
    distance, the margin, and — under ``fit_local_categorical_v1`` — the local id."""

    if hard_id_encoding not in HARD_ID_ENCODINGS:
        raise ValueError(
            f"hard_id_encoding {hard_id_encoding!r} is not registered; "
            f"registered: {HARD_ID_ENCODINGS}"
        )
    names = list(columns.numeric)
    if hard_id_encoding == HARD_ID_ENCODING_CATEGORICAL:
        names.append(columns.local_id)
    return tuple(names)


def categorical_model_features(
    columns: RegimeFoldFeatureColumns, hard_id_encoding: HardIdEncoding
) -> tuple[str, ...]:
    return (columns.local_id,) if hard_id_encoding == HARD_ID_ENCODING_CATEGORICAL else ()


def fold_feature_schema(columns: RegimeFoldFeatureColumns) -> pa.Schema:
    fields = [
        pa.field("fold_index", pa.int64()),
        pa.field("candidate_id", pa.string()),
        pa.field("partition", pa.string()),
        pa.field("regime_fit_id", pa.string()),
        pa.field("panel_row_id", pa.string()),
    ]
    fields.extend(pa.field(name, pa.float64()) for name in columns.numeric)
    fields.extend(
        (
            pa.field(columns.local_id, pa.string()),
            pa.field("canonical_reporting_cluster_id", pa.int64()),
            pa.field(columns.valid, pa.bool_()),
            pa.field(columns.missing_reason, pa.string()),
        )
    )
    return pa.schema(fields)


def fold_feature_native_spec(columns: RegimeFoldFeatureColumns) -> dict[str, tuple[str, bool]]:
    """HARDENING-BACKEND-FIX §6.2: the native spec proven BEFORE any conversion
    of a fold-feature table (the spine columns are never null)."""

    spec: dict[str, tuple[str, bool]] = {
        "fold_index": ("int", False),
        "candidate_id": ("id", False),
        "partition": ("str", False),
        "regime_fit_id": ("id", True),
        "panel_row_id": ("str", True),
    }
    spec.update({name: ("float", True) for name in columns.numeric})
    spec[columns.local_id] = ("str", True)
    spec["canonical_reporting_cluster_id"] = ("int", True)
    spec[columns.valid] = ("bool", False)
    spec[columns.missing_reason] = ("str", True)
    return spec


# ── contracts ────────────────────────────────────────────────────────────────


class FoldFitRef(FrozenContract):
    """fold index → the regime fit consulted for it. R6.1-FIX (§3.2, F-03):
    whenever a fit is present the ref also binds the fit's manifest-verified
    assignment sidecar SHA-256 and the enforced schema hash — the three
    fields are null together only for a legitimately absent fit."""

    fold_index: int = Field(ge=0)
    regime_fit_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    assignments_sidecar_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    assignment_schema_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _bound_together(self):
        bound = (self.regime_fit_id, self.assignments_sidecar_sha256, self.assignment_schema_hash)
        present = [value is not None for value in bound]
        if any(present) and not all(present):
            raise ValueError(
                "a FoldFitRef binds regime_fit_id, assignments_sidecar_sha256 and "
                "assignment_schema_hash together (all present for a fit, all null for an "
                "absent fit)"
            )
        return self


class RegimeFoldFeatureArtifactPayload(FrozenContract):
    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    observation_granularity: ObservationGranularity
    #: fold index → the regime fit consulted (None: no valid fit for that fold)
    regime_fit_ids_by_fold: tuple[FoldFitRef, ...]
    #: the fold-set ARTIFACT ids (never the row-population hashes) …
    candidate_fold_set_id: str = Field(pattern=SHA256_PATTERN)
    regime_fold_set_id: str = Field(pattern=SHA256_PATTERN)
    #: … and the candidate fold set's legacy row-population hash — the D13
    #: ``candidate_fold_set_id`` key every comparison_row_id of this study uses
    candidate_fold_set_hash: str = Field(pattern=SHA256_PATTERN)
    fold_schedule_id: str = Field(pattern=SHA256_PATTERN)
    candidate_view_id: str = Field(pattern=SHA256_PATTERN)
    #: the panel grain's resolved input block (the panel block); None for the
    #: candidate grain (its inputs are the protocol's bundle view)
    resolved_feature_block_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    panel_context: PanelAssignmentContext | None
    hard_id_encoding: HardIdEncoding
    resolved_cluster_count: int = Field(ge=2)
    feature_prefix: str
    model_feature_names: tuple[str, ...]
    categorical_model_features: tuple[str, ...]
    formula_version: Literal["ifvg_regime_context_formula_v1"] = (
        REGIME_FOLD_FEATURE_FORMULA_VERSION
    )
    materializer_version: Literal["regime_fold_feature_materializer_v1"] = (
        REGIME_FOLD_FEATURE_MATERIALIZER_VERSION
    )
    feature_schema_hash: str = Field(pattern=SHA256_PATTERN)
    row_count: int = Field(ge=0)
    valid_row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _coherent(self):
        folds = [ref.fold_index for ref in self.regime_fit_ids_by_fold]
        if folds != sorted(set(folds)):
            raise ValueError("regime_fit_ids_by_fold must be sorted by fold index, no repeats")
        columns = RegimeFoldFeatureColumns(
            prefix=self.feature_prefix, cluster_count=self.resolved_cluster_count
        )
        if self.feature_prefix != regime_feature_prefix(self.resolved_regime_protocol_id):
            raise ValueError("feature_prefix must be derived from the resolved protocol id")
        if tuple(self.model_feature_names) != model_feature_names(columns, self.hard_id_encoding):
            raise ValueError("model_feature_names disagree with the encoding contract")
        if tuple(self.categorical_model_features) != categorical_model_features(
            columns, self.hard_id_encoding
        ):
            raise ValueError("categorical_model_features disagree with the encoding contract")
        panel = self.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL
        if panel != (self.panel_context is not None):
            raise ValueError("panel context is required exactly for the panel grain")
        if panel != (self.resolved_feature_block_id is not None):
            raise ValueError("resolved_feature_block_id is the panel grain's input block")
        if self.valid_row_count > self.row_count:
            raise ValueError("valid_row_count exceeds row_count")
        return self

    @property
    def columns(self) -> RegimeFoldFeatureColumns:
        return RegimeFoldFeatureColumns(
            prefix=self.feature_prefix, cluster_count=self.resolved_cluster_count
        )


class RegimeFoldFeatureArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_fold_feature_artifact_id"

    regime_fold_feature_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeFoldFeatureArtifactPayload
    #: post-materialization fact binding the Arrow sidecar
    feature_table_sha256: str = Field(pattern=SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class PanelFoldFeatureInputs:
    """The panel grain's inputs: the VERIFIED panel frame + its artifact
    envelope, and the candidate as-of stage whose anchor instants the PIT
    rule consults."""

    panel_frame: pd.DataFrame
    panel_artifact: Any
    candidate_as_of_stage: AvailabilityStage = AvailabilityStage.ENTRY_DECISION


# ── the builder (S09b) ───────────────────────────────────────────────────────


def _typed_row(
    columns: RegimeFoldFeatureColumns,
    *,
    fold_index: int,
    candidate_id: str,
    partition: str,
    regime_fit_id: str | None,
    reason: str,
    panel_row_id: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold_index": int(fold_index),
        "candidate_id": str(candidate_id),
        "partition": partition,
        "regime_fit_id": regime_fit_id,
        "panel_row_id": panel_row_id,
    }
    for name in columns.numeric:
        row[name] = np.nan
    row[columns.local_id] = None
    row["canonical_reporting_cluster_id"] = None
    row[columns.valid] = False
    row[columns.missing_reason] = reason
    return row


def _feature_row(
    columns: RegimeFoldFeatureColumns,
    *,
    fold_index: int,
    candidate_id: str,
    partition: str,
    regime_fit_id: str,
    distances: Any,
    assigned_distance: Any,
    margin: Any,
    local_id: Any,
    canonical_id: Any,
    panel_row_id: str | None = None,
) -> dict[str, Any]:
    values = [float(value) for value in (distances if distances is not None else ())]
    if len(values) != columns.cluster_count:
        raise ValueError(
            f"fit {regime_fit_id[:12]}… assigned {len(values)} centroid distances to "
            f"{candidate_id}; the protocol's fixed k is {columns.cluster_count}"
        )
    row: dict[str, Any] = {
        "fold_index": int(fold_index),
        "candidate_id": str(candidate_id),
        "partition": partition,
        "regime_fit_id": str(regime_fit_id),
        "panel_row_id": panel_row_id,
    }
    for name, value in zip(columns.distances, values, strict=True):
        row[name] = value
    row[columns.assigned_distance] = float(assigned_distance)
    row[columns.margin] = float(margin)
    row[columns.local_id] = str(int(local_id))
    row["canonical_reporting_cluster_id"] = (
        int(canonical_id) if canonical_id is not None and pd.notna(canonical_id) else None
    )
    row[columns.valid] = True
    row[columns.missing_reason] = None
    return row


def _fit_rows_by_key(assignments: pd.DataFrame, fold_index: int) -> dict[tuple[str, str], Any]:
    """fit ``k``'s assignment rows keyed by (row_id, partition) — ONLY fold k."""

    if assignments.empty:
        return {}
    rows = assignments[assignments["fold_index"].astype(int) == int(fold_index)]
    keyed: dict[tuple[str, str], Any] = {}
    for row in rows.itertuples():
        key = (str(row.row_id), str(row.partition))
        if key in keyed:
            raise ValueError(
                f"fit rows repeat (row {key[0]}, partition {key[1]}) inside fold {fold_index}"
            )
        keyed[key] = row
    return keyed


def _candidate_grain_rows(
    columns: RegimeFoldFeatureColumns,
    *,
    fold_index: int,
    fit_id: str,
    fit_rows: dict[tuple[str, str], Any],
    members: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id, partition in members:
        row = fit_rows.get((candidate_id, partition))
        if row is None:
            rows.append(
                _typed_row(
                    columns,
                    fold_index=fold_index,
                    candidate_id=candidate_id,
                    partition=partition,
                    regime_fit_id=fit_id,
                    reason="no_fit_assignment_row",
                )
            )
            continue
        if not bool(row.valid):
            missing_reason = row.missing_reason
            if missing_reason is None or (
                not isinstance(missing_reason, str) and pd.isna(missing_reason)
            ) or str(missing_reason).strip() == "":
                raise RegimeAssignmentEvidenceError(
                    "assignment_provenance_incomplete",
                    f"fit {fit_id[:12]}… row {candidate_id} is invalid without a typed reason",
                )
            reason = str(missing_reason)
            rows.append(
                _typed_row(
                    columns,
                    fold_index=fold_index,
                    candidate_id=candidate_id,
                    partition=partition,
                    regime_fit_id=fit_id,
                    reason=reason,
                )
            )
            continue
        rows.append(
            _feature_row(
                columns,
                fold_index=fold_index,
                candidate_id=candidate_id,
                partition=partition,
                regime_fit_id=fit_id,
                distances=row.distances,
                assigned_distance=row.assigned_distance,
                margin=row.assignment_margin,
                local_id=row.fold_local_cluster_id,
                canonical_id=row.canonical_reporting_cluster_id,
            )
        )
    return rows


def _panel_grain_rows(
    columns: RegimeFoldFeatureColumns,
    *,
    protocol: RegimeProtocolEnvelope,
    fold_index: int,
    fit_id: str,
    fold_assignments: pd.DataFrame,
    members: tuple[tuple[str, str], ...],
    candidate_as_of: pd.DataFrame,
    panel_frame: pd.DataFrame,
    max_staleness_seconds: int,
) -> list[dict[str, Any]]:
    partition_for = {
        candidate_id: (int(fold_index), partition) for candidate_id, partition in members
    }
    as_of = candidate_as_of[candidate_as_of["candidate_id"].astype(str).isin(partition_for)]
    assigned = assign_panel_regimes_to_candidates(
        panel_frame,
        fold_assignments,
        as_of,
        protocol=protocol,
        partition_for_candidate=partition_for,
        max_staleness_seconds=max_staleness_seconds,
    ).set_index("candidate_id")
    rows: list[dict[str, Any]] = []
    for candidate_id, partition in members:
        if candidate_id not in assigned.index:
            rows.append(
                _typed_row(
                    columns,
                    fold_index=fold_index,
                    candidate_id=candidate_id,
                    partition=partition,
                    regime_fit_id=fit_id,
                    reason="no_fit_assignment_row",
                )
            )
            continue
        row = assigned.loc[candidate_id]
        bar_id = row["panel_row_id"]
        bar_id = None if bar_id is None or pd.isna(bar_id) else str(bar_id)
        if not bool(row["valid"]):
            rows.append(
                _typed_row(
                    columns,
                    fold_index=fold_index,
                    candidate_id=candidate_id,
                    partition=partition,
                    regime_fit_id=fit_id,
                    reason=str(row["missing_reason"]),
                    panel_row_id=bar_id,
                )
            )
            continue
        if str(row["regime_fit_id"]) != fit_id or int(row["fold_index"]) != int(fold_index):
            raise AssertionError("the PIT rule consulted a fit outside fold k")  # pragma: no cover
        rows.append(
            _feature_row(
                columns,
                fold_index=fold_index,
                candidate_id=candidate_id,
                partition=partition,
                regime_fit_id=fit_id,
                distances=row["distances"],
                assigned_distance=row["assigned_distance"],
                margin=row["assignment_margin"],
                local_id=row["fold_local_cluster_id"],
                canonical_id=row["canonical_reporting_cluster_id"],
                panel_row_id=bar_id,
            )
        )
    return rows


def _frame_for_schema(frame: pd.DataFrame, columns: RegimeFoldFeatureColumns) -> pd.DataFrame:
    # §6.2: native values are proven BEFORE the canonical conversions below
    validate_native_values(frame, fold_feature_native_spec(columns), context="fold-feature table")
    out = frame.loc[:, list(columns.ordered)].copy()
    out["fold_index"] = out["fold_index"].astype("int64")
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["partition"] = out["partition"].astype(str)
    for column in ("regime_fit_id", "panel_row_id", columns.local_id, columns.missing_reason):
        out[column] = out[column].astype(object).where(out[column].notna(), None)
    for column in columns.numeric:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    out["canonical_reporting_cluster_id"] = out["canonical_reporting_cluster_id"].astype("Int64")
    out[columns.valid] = out[columns.valid].astype(bool)
    return out.sort_values(["fold_index", "partition", "candidate_id"], kind="stable").reset_index(
        drop=True
    )


def fold_feature_table_bytes(frame: pd.DataFrame, columns: RegimeFoldFeatureColumns) -> bytes:
    return frame_to_arrow_bytes(_frame_for_schema(frame, columns), fold_feature_schema(columns))


def validate_fold_feature_rows(frame: pd.DataFrame, columns: RegimeFoldFeatureColumns) -> None:
    """R6.1-FIX §3.4 — the model-facing row invariants of the fold-feature
    table: a VALID row carries a 64-hex fit id, a lawful partition, every
    numeric feature finite and the fit-local id present (the canonical
    reporting id is reporting-only and may be null); an INVALID row carries
    no feature value (every numeric NaN, local id null) and exactly one
    registered missing reason. Applied on build, on every verified load and
    by the ladder seam."""

    required = set(columns.ordered)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"fold-feature table lacks columns: {missing}")
    # HARDENING-BACKEND-FIX §6.2 / §6.4: native values first — a malformed value
    # is never coerced into lawful missingness (no ``errors="coerce"``)
    validate_native_values(frame, fold_feature_native_spec(columns), context="fold-feature table")
    valid = frame[columns.valid].astype(bool).to_numpy()
    numeric = frame.loc[:, list(columns.numeric)].apply(pd.to_numeric, errors="raise")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    local = frame[columns.local_id]
    local_present = local.notna().to_numpy() & (local.astype(str) != "")
    reasons = frame[columns.missing_reason]
    partitions = frame["partition"].astype(str)
    fit_ids = frame["regime_fit_id"]
    if (valid & ~finite).any():
        raise ValueError("a valid fold-feature row carries a non-finite model feature")
    if (valid & ~local_present).any():
        raise ValueError("a valid fold-feature row lacks the fit-local cluster id")
    if (valid & ~partitions.isin(("train", "test")).to_numpy()).any():
        raise ValueError("a valid fold-feature row carries an unlawful partition")
    if (valid & ~fit_ids.astype(str).str.fullmatch(r"[0-9a-f]{64}").fillna(False).to_numpy()).any():
        raise ValueError("a valid fold-feature row lacks a 64-hex regime fit id")
    if (valid & reasons.notna().to_numpy()).any():
        raise ValueError("a valid fold-feature row carries a missing reason")
    invalid = ~valid
    if (invalid & numeric.notna().to_numpy().any(axis=1)).any():
        raise ValueError("an invalid fold-feature row carries a model feature value")
    if (invalid & local_present).any():
        raise ValueError("an invalid fold-feature row carries a fit-local cluster id")
    unregistered = set(reasons[invalid].dropna().astype(str)) - set(
        REGIME_FOLD_FEATURE_MISSING_REASONS
    )
    if (invalid & reasons.isna().to_numpy()).any() or unregistered:
        raise ValueError(
            "an invalid fold-feature row requires exactly one registered missing reason; "
            f"unregistered: {sorted(unregistered)}"
        )
    # HARDENING-BACKEND-FIX §6.4: EVERY row keeps its reconciliation spine —
    # a non-negative fold index, a lawful partition, and (for an invalid row
    # that came from an applicable fit) that fit's 64-hex id; the fit-less
    # reason ``no_valid_regime_fit`` never names a fit
    folds = pd.to_numeric(frame["fold_index"], errors="raise")
    if (folds < 0).any():
        raise RegimeAssignmentEvidenceError(
            "assignment_row_invariant_violated", "a fold-feature row carries a negative fold index"
        )
    if (~partitions.isin(("train", "test")).to_numpy()).any():
        raise RegimeAssignmentEvidenceError(
            "assignment_row_invariant_violated",
            "every fold-feature row carries a lawful partition (train / test)",
        )
    fit_nonnull = fit_ids.notna().to_numpy()
    fit_wellformed = fit_ids.astype(str).str.fullmatch(r"[0-9a-f]{64}").fillna(False).to_numpy()
    if (fit_nonnull & ~fit_wellformed).any():
        raise RegimeAssignmentEvidenceError(
            "assignment_provenance_incomplete",
            "a fold-feature row carries a malformed regime fit id",
        )
    fit_less = (reasons.astype(object) == "no_valid_regime_fit").to_numpy()
    if (invalid & fit_less & fit_nonnull).any():
        raise RegimeAssignmentEvidenceError(
            "assignment_provenance_incomplete",
            "no_valid_regime_fit never names a fit (the fold had none)",
        )
    if (invalid & ~fit_less & ~fit_nonnull).any():
        raise RegimeAssignmentEvidenceError(
            "assignment_provenance_incomplete",
            "an invalid fold-feature row from an applicable fit retains that fit's id (the "
            "reconciliation spine); a known invalid assignment never collapses into "
            "generic absence",
        )


def assert_fold_feature_spine_bound(
    frame: pd.DataFrame, payload: RegimeFoldFeatureArtifactPayload
) -> None:
    """HARDENING-BACKEND-FIX §6.4: every row's ``(fold_index, regime_fit_id)`` is
    bound by the artifact's ``FoldFitRef`` for that fold — the ref carries the
    source assignment-table SHA-256 and schema hash, so each row's spine
    resolves to exact evidence; a fold whose ref names no fit carries only
    ``no_valid_regime_fit`` rows."""

    refs = {int(ref.fold_index): ref for ref in payload.regime_fit_ids_by_fold}
    columns = payload.columns
    for fold_value, group in frame.groupby(frame["fold_index"].astype(int), sort=True):
        ref = refs.get(int(fold_value))
        if ref is None:
            raise RegimeAssignmentEvidenceError(
                "assignment_provenance_incomplete",
                f"fold {fold_value}: rows exist but the artifact binds no FoldFitRef for it",
            )
        fits = set(group["regime_fit_id"].dropna().astype(str))
        if ref.regime_fit_id is None:
            if fits or group[columns.valid].astype(bool).any():
                raise RegimeAssignmentEvidenceError(
                    "assignment_provenance_incomplete",
                    f"fold {fold_value}: rows name a fit but the artifact binds none for it",
                )
            continue
        fitless = group["regime_fit_id"].isna()
        if fitless.any():
            # review RB-02: the artifact binds a fit for this fold, so EVERY row of
            # the fold names it — a fit-less row here (a null id, the fit-less
            # reason) is a known invalid assignment of the applicable fit
            # collapsed into generic absence
            raise RegimeAssignmentEvidenceError(
                "assignment_provenance_incomplete",
                f"fold {fold_value}: {int(fitless.sum())} row(s) name no fit although the "
                f"artifact binds FoldFitRef {ref.regime_fit_id[:12]}… for it",
            )
        outside = sorted(fits - {ref.regime_fit_id})
        if outside:
            raise RegimeAssignmentEvidenceError(
                "assignment_provenance_incomplete",
                f"fold {fold_value}: rows name fit(s) {[f[:12] for f in outside[:2]]}… outside "
                f"the bound FoldFitRef {ref.regime_fit_id[:12]}…",
            )


def _verified_fit_frames(regime_run, fit_assignments: Mapping[int, Any]) -> dict[int, Any]:
    """R6.1-FIX (§3.2, F-03): the ONLY assignment source of the fold-local
    features is the VERIFIED per-fit evidence (``VerifiedFitAssignments``
    keyed by fold index). Every fit of the run must be present, each entry
    must be the run's fit for that fold, and an in-memory frame is refused."""

    from .regime_store import VerifiedFitAssignments  # noqa: PLC0415

    if not isinstance(fit_assignments, Mapping):
        raise TypeError(
            "fit_assignments must map fold index → VerifiedFitAssignments (exact store "
            f"loads); got {type(fit_assignments).__name__}"
        )
    verified: dict[int, Any] = {}
    for fold_index, value in fit_assignments.items():
        if not isinstance(value, VerifiedFitAssignments):
            raise TypeError(
                "build_regime_fold_features consumes VerifiedFitAssignments only — fold "
                f"{fold_index} supplied {type(value).__name__} (an in-memory assignment "
                "frame is not evidence)"
            )
        verified[int(fold_index)] = value
    for fit in regime_run.fold_fits:
        loaded = verified.get(int(fit.fold_index))
        if loaded is None:
            raise ValueError(
                f"fold {fit.fold_index}: the run's fit {fit.fit_envelope.regime_fit_id[:12]}… "
                "has no verified assignment evidence (every fit must be exact-loaded)"
            )
        if loaded.regime_fit_id != fit.fit_envelope.regime_fit_id or (
            loaded.fold_index != int(fit.fold_index)
        ):
            raise ValueError(
                f"fold {fit.fold_index}: the verified assignment evidence names fit "
                f"{loaded.regime_fit_id[:12]}… (fold {loaded.fold_index}), not the run's fit "
                f"{fit.fit_envelope.regime_fit_id[:12]}…"
            )
        if loaded.envelope.payload.resolved_regime_protocol_id != (
            regime_run.protocol.resolved_regime_protocol_id
        ):
            raise ValueError(f"fold {fit.fold_index}: the verified fit is of another protocol")
    return verified


def build_regime_fold_features(
    *,
    protocol: RegimeProtocolEnvelope,
    regime_run,
    fit_assignments: Mapping[int, Any],
    candidate_fold_set: FoldSetArtifactEnvelope,
    candidate_folds: ContextFoldSet,
    regime_fold_set: FoldSetArtifactEnvelope,
    schedule: FoldScheduleEnvelope,
    candidate_view_frame: pd.DataFrame,
    candidate_view_id: str,
    hard_id_encoding: HardIdEncoding = "none",
    panel: PanelFoldFeatureInputs | None = None,
) -> tuple[RegimeFoldFeatureArtifactEnvelope, pd.DataFrame]:
    """The S09b materializer (module docstring).

    Preconditions (each refuses): ``assert_same_fold_schedule`` between the
    candidate and regime fold sets; the schedule IS the fold sets' schedule;
    ``regime_run`` ran on the regime fold set and under ``protocol``; the
    candidate ``ContextFoldSet`` is the candidate fold-set artifact's
    population; the panel grain supplies its verified panel inputs;
    ``fit_assignments`` is the VERIFIED per-fit evidence (R6.1-FIX §3.2 —
    the run's in-memory frame is never consulted).
    """

    payload = protocol.payload
    grain = ObservationGranularity(payload.observation_granularity)
    if regime_run.protocol.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id:
        raise ValueError("the regime run belongs to a different regime protocol")
    verified = _verified_fit_frames(regime_run, fit_assignments)
    assert_same_fold_schedule(candidate_fold_set.payload, regime_fold_set.payload)
    if schedule.fold_schedule_id != candidate_fold_set.payload.fold_schedule_id:
        raise ValueError("the fold schedule is not the fold sets' schedule")
    candidate_grain = ObservationGranularity.CANDIDATE_STAGE_ROW
    if candidate_fold_set.payload.observation_grain is not candidate_grain:
        raise ValueError("the supervised fold set must be a candidate-grain fold set")
    if regime_fold_set.payload.observation_grain is not grain:
        raise ValueError("the regime fold set's grain does not match the protocol grain")
    if regime_run.fold_set_id != regime_fold_set.payload.fold_set_id:
        raise ValueError("the regime run did not run on the regime fold-set artifact's population")
    if _legacy_fold_set_id(candidate_folds) != candidate_fold_set.payload.fold_set_id:
        raise ValueError("the candidate folds are not the candidate fold-set artifact's population")
    if hard_id_encoding not in HARD_ID_ENCODINGS:
        raise ValueError(f"hard_id_encoding {hard_id_encoding!r} is not registered")
    if "candidate_id" not in candidate_view_frame.columns:
        raise ValueError("candidate view frame lacks candidate_id")
    view_ids = set(candidate_view_frame["candidate_id"].astype(str))
    if len(view_ids) != len(candidate_view_frame):
        raise ValueError("candidate view frame repeats a candidate_id")

    panel_context: PanelAssignmentContext | None = None
    resolved_block_id: str | None = None
    candidate_as_of: pd.DataFrame | None = None
    max_staleness = 0
    if grain is ObservationGranularity.CONTEXT_BAR_PANEL:
        if panel is None:
            raise ValueError("the panel grain requires PanelFoldFeatureInputs")
        artifact = panel.panel_artifact
        if artifact.context_bar_panel_artifact_id != payload.panel_source_artifact_id:
            raise ValueError("the panel artifact is not the protocol's pinned panel source")
        interval = int(payload.panel_interval_seconds or 0)
        max_staleness = interval * PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS
        panel_context = PanelAssignmentContext(
            context_bar_panel_artifact_id=str(payload.panel_source_artifact_id),
            panel_as_of_policy_id=str(payload.panel_as_of_policy_id),
            candidate_as_of_stage=panel.candidate_as_of_stage,
            panel_interval_seconds=interval,
            max_staleness_seconds=max_staleness,
        )
        resolved_block_id = str(artifact.payload.resolved_feature_block_id)
        candidate_as_of = candidate_as_of_frame(
            candidate_view_frame, stage=panel.candidate_as_of_stage
        )
    elif panel is not None:
        raise ValueError("panel inputs are lawful only for the panel grain")

    columns = fold_feature_columns(
        protocol.resolved_regime_protocol_id, payload.resolved_cluster_count
    )
    fits_by_fold = {int(fit.fold_index): fit for fit in regime_run.fold_fits}
    rows: list[dict[str, Any]] = []
    fit_refs: list[FoldFitRef] = []
    for fold in candidate_folds.folds:
        if not fold.valid:
            continue
        members = tuple(
            [(str(candidate_id), "train") for candidate_id in sorted(fold.train_candidate_ids)]
            + [(str(candidate_id), "test") for candidate_id in sorted(fold.test_candidate_ids)]
        )
        outside = sorted({candidate_id for candidate_id, _ in members} - view_ids)
        if outside:
            raise ValueError(
                f"fold {fold.fold_index} references candidates absent from the view frame: "
                f"{outside[:3]}"
            )
        fit = fits_by_fold.get(int(fold.fold_index))
        if fit is None:
            fit_refs.append(FoldFitRef(fold_index=int(fold.fold_index), regime_fit_id=None))
            rows.extend(
                _typed_row(
                    columns,
                    fold_index=fold.fold_index,
                    candidate_id=candidate_id,
                    partition=partition,
                    regime_fit_id=None,
                    reason="no_valid_regime_fit",
                )
                for candidate_id, partition in members
            )
            continue
        fit_id = fit.fit_envelope.regime_fit_id
        source = verified[int(fold.fold_index)]
        fit_refs.append(
            FoldFitRef(
                fold_index=int(fold.fold_index),
                regime_fit_id=fit_id,
                assignments_sidecar_sha256=source.assignments_sidecar_sha256,
                assignment_schema_hash=source.assignment_schema_hash,
            )
        )
        # fit k's VERIFIED sidecar rows only (fold k); never another fold's
        fold_assignments = source.frame[
            source.frame["fold_index"].astype(int) == int(fold.fold_index)
        ]
        if grain is ObservationGranularity.CONTEXT_BAR_PANEL:
            if candidate_as_of is None or panel is None:  # pragma: no cover - guarded above
                raise RuntimeError(
                    "panel-grain fold features require the candidate as-of frame and the "
                    "verified panel inputs (wiring error)"
                )
            rows.extend(
                _panel_grain_rows(
                    columns,
                    protocol=protocol,
                    fold_index=fold.fold_index,
                    fit_id=fit_id,
                    fold_assignments=fold_assignments,
                    members=members,
                    candidate_as_of=candidate_as_of,
                    panel_frame=panel.panel_frame,
                    max_staleness_seconds=max_staleness,
                )
            )
        else:
            rows.extend(
                _candidate_grain_rows(
                    columns,
                    fold_index=fold.fold_index,
                    fit_id=fit_id,
                    fit_rows=_fit_rows_by_key(fold_assignments, fold.fold_index),
                    members=members,
                )
            )
    frame = _frame_for_schema(pd.DataFrame(rows, columns=list(columns.ordered)), columns)
    validate_fold_feature_rows(frame, columns)
    table_bytes = fold_feature_table_bytes(frame, columns)
    spine_refs = tuple(fit_refs)
    artifact_payload = RegimeFoldFeatureArtifactPayload(
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        observation_granularity=grain,
        regime_fit_ids_by_fold=tuple(fit_refs),
        candidate_fold_set_id=candidate_fold_set.fold_set_artifact_id,
        regime_fold_set_id=regime_fold_set.fold_set_artifact_id,
        candidate_fold_set_hash=candidate_fold_set.payload.fold_set_id,
        fold_schedule_id=schedule.fold_schedule_id,
        candidate_view_id=candidate_view_id,
        resolved_feature_block_id=resolved_block_id,
        panel_context=panel_context,
        hard_id_encoding=hard_id_encoding,
        resolved_cluster_count=int(payload.resolved_cluster_count),
        feature_prefix=columns.prefix,
        model_feature_names=model_feature_names(columns, hard_id_encoding),
        categorical_model_features=categorical_model_features(columns, hard_id_encoding),
        feature_schema_hash=arrow_schema_hash(fold_feature_schema(columns)),
        row_count=int(len(frame)),
        valid_row_count=int(frame[columns.valid].sum()),
    )
    if tuple(artifact_payload.regime_fit_ids_by_fold) != spine_refs:  # pragma: no cover
        raise RuntimeError("fold fit refs diverged from the built rows (wiring error)")
    assert_fold_feature_spine_bound(frame, artifact_payload)
    envelope = RegimeFoldFeatureArtifactEnvelope.from_payload(
        artifact_payload, feature_table_sha256=bytes_sha256(table_bytes)
    )
    return envelope, frame


# ── persistence ──────────────────────────────────────────────────────────────


def verify_regime_fold_feature_frame(
    envelope: RegimeFoldFeatureArtifactEnvelope, frame: pd.DataFrame
) -> None:
    """The frame IS the persisted table, or refuse (rehash)."""

    columns = envelope.payload.columns
    if bytes_sha256(fold_feature_table_bytes(frame, columns)) != envelope.feature_table_sha256:
        raise ValueError(
            "the fold-feature frame does not hash to the artifact's feature_table_sha256"
        )


def save_regime_fold_features(
    root: Path, envelope: RegimeFoldFeatureArtifactEnvelope, frame: pd.DataFrame
):
    verify_regime_fold_feature_frame(envelope, frame)
    return save_or_reuse_envelope(
        Path(root),
        REGIME_FOLD_FEATURE_STORE,
        envelope,
        extra_files={
            FOLD_FEATURE_SIDECAR: fold_feature_table_bytes(frame, envelope.payload.columns)
        },
    )


def verify_fold_fit_refs_against_store(
    root: Path, envelope: RegimeFoldFeatureArtifactEnvelope
) -> None:
    """R6.1-FIX (§3.2): every ``FoldFitRef`` of the artifact is re-checked
    against the store by EXACT id — the fit's sidecar must still hash to the
    bound SHA-256 under the bound schema hash. No listing; a missing,
    tampered or differently-typed sidecar fails the fold-feature load closed."""

    from .regime_store import load_regime_fit_assignments  # noqa: PLC0415

    for ref in envelope.payload.regime_fit_ids_by_fold:
        if ref.regime_fit_id is None:
            continue
        loaded = load_regime_fit_assignments(Path(root), ref.regime_fit_id)
        if loaded.fold_index != int(ref.fold_index):
            raise ValueError(
                f"fit {ref.regime_fit_id[:12]}… is fold {loaded.fold_index} in the store, not "
                f"fold {ref.fold_index} as the fold-feature artifact binds"
            )
        if loaded.assignments_sidecar_sha256 != ref.assignments_sidecar_sha256 or (
            loaded.assignment_schema_hash != ref.assignment_schema_hash
        ):
            raise ValueError(
                f"fit {ref.regime_fit_id[:12]}… assignment sidecar in the store does not "
                "match the sidecar ref the fold-feature artifact binds (sha256 / schema); "
                "refusing"
            )


def load_regime_fold_features(root: Path, artifact_id: str) -> RegimeFoldFeatureArtifactEnvelope:
    envelope = load_verified_envelope(
        Path(root), REGIME_FOLD_FEATURE_STORE, artifact_id, RegimeFoldFeatureArtifactEnvelope
    )
    verify_fold_fit_refs_against_store(Path(root), envelope)
    return envelope


def load_regime_fold_feature_frame(
    root: Path, envelope: RegimeFoldFeatureArtifactEnvelope
) -> pd.DataFrame:
    data = load_sidecar_bytes(
        Path(root),
        REGIME_FOLD_FEATURE_STORE,
        envelope.regime_fold_feature_artifact_id,
        FOLD_FEATURE_SIDECAR,
    )
    if bytes_sha256(data) != envelope.feature_table_sha256:
        raise ValueError("stored fold-feature table fails the envelope hash check")
    frame = frame_from_arrow_bytes(data)
    if len(frame) != envelope.payload.row_count:
        raise ValueError("stored fold-feature table row count disagrees with the payload")
    validate_fold_feature_rows(frame, envelope.payload.columns)
    # HARDENING-BACKEND-FIX §6.4: the loader re-proves the spine binding
    assert_fold_feature_spine_bound(frame, envelope.payload)
    return frame


# ── the supervised feature source (fork J's Protocol) ───────────────────────


class RegimeFoldFeatureFrameSource:
    """``RegimeFoldFeatureSource`` over a VERIFIED artifact: per fold ``k``
    the fold-k rows (train + test partitions) keyed by ``candidate_id`` with
    the model feature names — invalid rows carry NaN numerics and a null
    categorical id (the model runner's fold-fitted imputer / native NaN
    handles them). Nothing outside fold ``k`` is ever returned."""

    def __init__(self, envelope: RegimeFoldFeatureArtifactEnvelope, frame: pd.DataFrame):
        verify_regime_fold_feature_frame(envelope, frame)
        self._envelope = envelope
        self._columns = envelope.payload.columns
        self._frame = _frame_for_schema(frame, self._columns)
        validate_fold_feature_rows(self._frame, self._columns)

    @property
    def envelope(self) -> RegimeFoldFeatureArtifactEnvelope:
        return self._envelope

    @property
    def artifact_id(self) -> str:
        return self._envelope.regime_fold_feature_artifact_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self._envelope.payload.model_feature_names)

    @property
    def categorical_features(self) -> tuple[str, ...]:
        return tuple(self._envelope.payload.categorical_model_features)

    @property
    def local_id_column(self) -> str:
        return self._columns.local_id

    @property
    def fold_indices(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in self._frame["fold_index"].unique()))

    def rows_for_fold(self, fold_index: int) -> pd.DataFrame:
        """Every persisted column of fold ``k`` (keys, features, reporting id,
        validity, typed reason)."""

        rows = self._frame[self._frame["fold_index"].astype(int) == int(fold_index)]
        return rows.reset_index(drop=True)

    def frame_for_fold(self, fold_index: int) -> pd.DataFrame:
        rows = self.rows_for_fold(fold_index)
        if rows["candidate_id"].duplicated().any():
            raise ValueError(f"fold {fold_index} carries a candidate twice")
        out = rows.loc[:, ["candidate_id", *self.feature_names]].copy()
        if self._columns.local_id in out.columns:
            out[self._columns.local_id] = out[self._columns.local_id].astype(object).where(
                out[self._columns.local_id].notna(), None
            )
        return out.reset_index(drop=True)

    def reasons_for_fold(self, fold_index: int) -> pd.DataFrame:
        rows = self.rows_for_fold(fold_index)
        out = rows.loc[
            :, ["candidate_id", "partition", self._columns.valid, self._columns.missing_reason]
        ].copy()
        return out.rename(
            columns={self._columns.valid: "valid", self._columns.missing_reason: "missing_reason"}
        )

    def local_ids_for_fold(self, fold_index: int) -> pd.DataFrame:
        """``(candidate_id, partition, local_id)`` of the VALID rows of fold
        ``k`` — the stratification key for ``cohort_model``."""

        rows = self.rows_for_fold(fold_index)
        valid = rows[rows[self._columns.valid].astype(bool)]
        return pd.DataFrame(
            {
                "candidate_id": valid["candidate_id"].astype(str).to_numpy(),
                "partition": valid["partition"].astype(str).to_numpy(),
                "local_id": valid[self._columns.local_id].astype(str).to_numpy(),
            }
        )


def load_regime_fold_feature_source(root: Path, artifact_id: str) -> RegimeFoldFeatureFrameSource:
    envelope = load_regime_fold_features(root, artifact_id)
    return RegimeFoldFeatureFrameSource(envelope, load_regime_fold_feature_frame(root, envelope))


def _example_payload() -> RegimeFoldFeatureArtifactPayload:
    columns = fold_feature_columns("a" * 64, 3)
    return RegimeFoldFeatureArtifactPayload(
        resolved_regime_protocol_id="a" * 64,
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        regime_fit_ids_by_fold=(
            FoldFitRef(
                fold_index=0,
                regime_fit_id="b" * 64,
                assignments_sidecar_sha256="2" * 64,
                assignment_schema_hash="3" * 64,
            ),
        ),
        candidate_fold_set_id="c" * 64,
        regime_fold_set_id="d" * 64,
        candidate_fold_set_hash="1" * 64,
        fold_schedule_id="e" * 64,
        candidate_view_id="f" * 64,
        resolved_feature_block_id=None,
        panel_context=None,
        hard_id_encoding="none",
        resolved_cluster_count=3,
        feature_prefix=columns.prefix,
        model_feature_names=model_feature_names(columns, "none"),
        categorical_model_features=(),
        feature_schema_hash=arrow_schema_hash(fold_feature_schema(columns)),
        row_count=0,
        valid_row_count=0,
    )


register_identity_pair(
    name="RegimeFoldFeatureArtifact",
    envelope_cls=RegimeFoldFeatureArtifactEnvelope,
    payload_cls=RegimeFoldFeatureArtifactPayload,
    id_field="regime_fold_feature_artifact_id",
    example_factory=_example_payload,
    extra_envelope_fields=("feature_table_sha256",),
)
