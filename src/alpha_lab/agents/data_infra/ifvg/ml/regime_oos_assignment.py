"""The descriptive OOS regime-assignment artifact + the PIT panel rule
(R6.1 D7 / §6.C; owner correction 2).

Two regime-assignment artifacts exist with different uses. THIS one — the
:class:`RegimeOosAssignmentEnvelope` — is descriptive and stratification
use ONLY: one row per candidate carrying the frozen OUT-OF-SAMPLE regime of
the single fold in which the candidate was scored OOS (candidate grain) or
the point-in-time panel regime at the candidate's as-of instant (panel
grain). The supervised feature source is the SEPARATE fold-local
``RegimeFoldFeatureArtifact`` (``regime_fold_features.py``); the ladder
never reads this artifact (test-enforced).

**Panel → candidate PIT rule — normative order of checks**
(:func:`assign_panel_regimes_to_candidates`; vectorized ``searchsorted``
over the same-day sorted closes; exact one-to-one merges; no ``merge_asof``,
no nearest-time fallback):

1. ``candidate.trading_day = classify_session(as_of, IFVG_DOC_SESSION_SCHEME)
   .trading_day``; only panel rows of that trading day are eligible — a
   regime is NEVER carried across the 18:00 ET trading-day reset.
2. Among eligible rows take the latest with ``bar_close_ts_utc <= as_of``
   (ties → lexicographically last ``row_id``); none → ``no_completed_panel_bar``.
3. That row must be ``cbp_valid``: ``insufficient_trading_day_lookback`` →
   ``panel_warmup``; ``lookback_window_gap`` → ``panel_gap``;
   ``source_bar_incomplete`` → ``panel_source_bar_incomplete``.
4. ``elapsed = as_of − bar_close <= max_staleness_seconds`` (V1 stamp: one
   panel interval); otherwise ``panel_stale`` — a missing current bar never
   lets an older bar answer for it.
5. The assignment row must exist for the compatible partition: the
   descriptive artifact consults ``partition == "test"`` valid rows (lowest
   ``fold_index`` wins); fold features consult the candidate's OWN partition
   in fold k from fit k only. None → ``coverage_gap``.

Every candidate is preserved with its typed reason (R6.1-FIX §3.3: a
candidate whose stage anchor is null is ``candidate_as_of_missing``);
missing columns, duplicate candidate ids, and unparseable NON-null as-of
instants are refused.

R6.1-FIX (§3.1, F-01/F-02): the artifact is built ONLY from
:class:`~.regime_store.VerifiedFitAssignments` — the per-fit assignment
sidecars as verified-loaded from the store — and its identity binds every
fit's sidecar SHA-256 + schema hash (``regime_fit_assignment_refs``) and a
consulted-source hash over EVERY value that can change the output.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc
from pydantic import Field, model_validator
from strategy_core.constants import IFVG_DOC_SESSION_SCHEME
from strategy_core.decisions.sessions import classify_session

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
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .regime_contracts import (
    FitAssignmentRef,
    ObservationGranularity,
    RegimeProtocolEnvelope,
    validate_assignment_rows,
)

__all__ = [
    "REGIME_OOS_ASSIGNMENT_STORE",
    "OOS_ASSIGNMENT_SIDECAR",
    "OOS_ASSIGNMENT_FORMULA_VERSION",
    "OOS_ASSIGNMENT_COLUMNS",
    "OOS_ASSIGNMENT_SCHEMA",
    "OOS_ASSIGNMENT_SCHEMA_HASH",
    "STAGE_AS_OF_COLUMNS",
    "CANDIDATE_AS_OF_SOURCE_REF_PATTERN",
    "CANDIDATE_AS_OF_MISSING_REASON",
    "PanelAssignmentContext",
    "RegimeOosAssignmentPayload",
    "RegimeOosAssignmentEnvelope",
    "candidate_as_of_frame",
    "assign_panel_regimes_to_candidates",
    "candidate_fold_oos_assignment",
    "consulted_assignment_frame",
    "consulted_assignments_hash",
    "candidate_as_of_source_hash",
    "build_regime_oos_assignment_artifact",
    "save_regime_oos_assignment",
    "load_regime_oos_assignment",
    "load_regime_oos_assignment_frame",
    "verify_regime_oos_assignment_frame",
]

REGIME_OOS_ASSIGNMENT_STORE = "regime_oos_assignments"
OOS_ASSIGNMENT_SIDECAR = "regime_oos_assignments.arrow"
OOS_ASSIGNMENT_FORMULA_VERSION = "regime_oos_assignment_v2"
#: R6.1-FIX §3.3: the typed reason of a candidate whose stage anchor is null.
CANDIDATE_AS_OF_MISSING_REASON = "candidate_as_of_missing"

#: Availability stage → the candidate row's as-of anchor column (the same
#: anchor semantics the MBP-1 stage windows use: ``STAGE_ANCHOR_COLUMNS``).
STAGE_AS_OF_COLUMNS: Mapping[AvailabilityStage, str] = {
    AvailabilityStage.HTF_TAP: "tap_ts_utc",
    AvailabilityStage.PARENT_LOCK: "lock_ts_utc",
    AvailabilityStage.OPPOSING_CONFIRMATION: "armed_ts_utc",
    AvailabilityStage.INVERSION: "inversion_ts_utc",
    AvailabilityStage.ENTRY_DECISION: "entry_ts_utc",
}

#: The candidate as-of provenance line: ``<source_kind>:<64-hex artifact id>``
#: of the VERIFIED-LOADED observation source whose anchor instants were
#: assigned (adversarial R6.1 F1: a free caller string is not evidence).
CANDIDATE_AS_OF_SOURCE_REF_PATTERN = r"^(bundle_feature_view|context_bar_panel):[0-9a-f]{64}$"

OOS_ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "regime_fit_id",
    "fold_index",
    "partition",
    "panel_row_id",
    "fold_local_cluster_id",
    "canonical_reporting_cluster_id",
    "distances",
    "assigned_distance",
    "assignment_margin",
    "valid",
    "missing_reason",
    "elapsed_seconds_since_bar_close",
)

OOS_ASSIGNMENT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("candidate_id", pa.large_string()),
        pa.field("regime_fit_id", pa.large_string()),
        pa.field("fold_index", pa.int64()),
        pa.field("partition", pa.large_string()),
        pa.field("panel_row_id", pa.large_string()),
        pa.field("fold_local_cluster_id", pa.int64()),
        pa.field("canonical_reporting_cluster_id", pa.int64()),
        pa.field("distances", pa.list_(pa.float64())),
        pa.field("assigned_distance", pa.float64()),
        pa.field("assignment_margin", pa.float64()),
        pa.field("valid", pa.bool_()),
        pa.field("missing_reason", pa.large_string()),
        pa.field("elapsed_seconds_since_bar_close", pa.float64()),
    ]
)
OOS_ASSIGNMENT_SCHEMA_HASH = arrow_schema_hash(OOS_ASSIGNMENT_SCHEMA)

_CANDIDATE_MISSING_REASONS: tuple[str, ...] = ("no_oos_assignment",)
_PANEL_REASON_FOR_VALIDITY: Mapping[str, str] = {
    "insufficient_trading_day_lookback": "panel_warmup",
    "lookback_window_gap": "panel_gap",
    "source_bar_incomplete": "panel_source_bar_incomplete",
}


class PanelAssignmentContext(FrozenContract):
    context_bar_panel_artifact_id: str = Field(pattern=SHA256_PATTERN)
    panel_as_of_policy_id: str
    candidate_as_of_stage: AvailabilityStage
    panel_interval_seconds: int = Field(ge=60)
    max_staleness_seconds: int = Field(ge=1)


class RegimeOosAssignmentPayload(FrozenContract):
    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    observation_granularity: ObservationGranularity
    #: the DERIVED projection of ``regime_fit_assignment_refs`` (kept for
    #: consumers; validated to equal the refs' ordered ids)
    regime_fit_ids: tuple[str, ...]
    #: R6.1-FIX §3.1: the exact per-fit assignment evidence consumed —
    #: canonically sorted by fit id; each ref binds the fit's manifest-verified
    #: sidecar SHA-256 and the enforced schema hash
    regime_fit_assignment_refs: tuple[FitAssignmentRef, ...]
    regime_fold_set_id: str = Field(pattern=SHA256_PATTERN)
    fold_schedule_id: str = Field(pattern=SHA256_PATTERN)
    assignment_source: Literal["candidate_fold_oos", "panel_pit"]
    panel_context: PanelAssignmentContext | None
    #: RA-07: the availability stage whose anchor column supplied the hashed
    #: candidate as-of instants (the panel grain's context names the same
    #: stage; the candidate grain records it here, so the same output bytes
    #: under another stage never share an identity)
    candidate_as_of_stage: AvailabilityStage
    candidate_as_of_source_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_as_of_source_ref: str = Field(pattern=CANDIDATE_AS_OF_SOURCE_REF_PATTERN)
    consulted_assignments_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=0)
    #: the protocol's fixed ``k`` — the row invariants check every valid row's
    #: distance vector against it on build and on every verified load
    resolved_cluster_count: int = Field(ge=2)
    assignment_schema_hash: str = Field(pattern=SHA256_PATTERN)
    formula_version: Literal["regime_oos_assignment_v2"] = OOS_ASSIGNMENT_FORMULA_VERSION

    @model_validator(mode="after")
    def _coherent(self):
        refs = self.regime_fit_assignment_refs
        ordered = tuple(sorted(refs, key=lambda ref: ref.regime_fit_id))
        if tuple(refs) != ordered:
            raise ValueError("regime_fit_assignment_refs must be sorted by fit id")
        if len({ref.regime_fit_id for ref in refs}) != len(refs):
            raise ValueError("regime_fit_assignment_refs must not repeat a fit")
        if tuple(self.regime_fit_ids) != tuple(ref.regime_fit_id for ref in refs):
            raise ValueError(
                "regime_fit_ids must be exactly the ordered projection of "
                "regime_fit_assignment_refs"
            )
        panel = self.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL
        if panel != (self.assignment_source == "panel_pit"):
            raise ValueError("panel grain ⇔ panel_pit assignment source")
        if panel != (self.panel_context is not None):
            raise ValueError("panel context is required exactly for the panel grain")
        if self.panel_context is not None and (
            self.panel_context.candidate_as_of_stage is not self.candidate_as_of_stage
        ):
            raise ValueError(
                "candidate_as_of_stage must equal the panel context's candidate_as_of_stage"
            )
        return self


class RegimeOosAssignmentEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_oos_assignment_id"

    regime_oos_assignment_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeOosAssignmentPayload
    #: post-materialization fact binding the Arrow sidecar
    assignment_table_sha256: str = Field(pattern=SHA256_PATTERN)


# ── candidate as-of instants ─────────────────────────────────────────────────


def candidate_as_of_frame(frame: pd.DataFrame, *, stage: AvailabilityStage) -> pd.DataFrame:
    """``(candidate_id, as_of_ts_utc)`` for one availability stage's anchor
    column; a candidate without that anchor keeps a null as-of (typed
    downstream), duplicates refuse."""

    column = STAGE_AS_OF_COLUMNS[AvailabilityStage(stage)]
    if "candidate_id" not in frame.columns:
        raise ValueError("candidate frame lacks candidate_id")
    if column not in frame.columns:
        raise ValueError(f"candidate frame lacks the {stage.value} anchor column {column!r}")
    ids = frame["candidate_id"].astype(str)
    if ids.duplicated().any():
        raise ValueError("candidate frame repeats a candidate_id")
    return pd.DataFrame(
        {"candidate_id": ids.to_numpy(), "as_of_ts_utc": frame[column].to_numpy()}
    )


def _as_of_ns(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """``(instants_ns, missing_mask)``: a NULL anchor is a typed fact (R6.1-FIX
    §3.3 — the candidate is preserved as ``candidate_as_of_missing``); a
    NON-null value that does not parse as a UTC timestamp is a hard error."""

    raw = pd.Series(values).reset_index(drop=True)
    missing = raw.isna().to_numpy() | (raw.astype(object).map(lambda v: v == "")).to_numpy()
    stamps = pd.to_datetime(raw, utc=True, errors="coerce")
    unparseable = stamps.isna().to_numpy() & ~missing
    if unparseable.any():
        raise ValueError("candidate as-of instants must all parse as UTC timestamps")
    filled = stamps.fillna(pd.Timestamp(0, tz="UTC"))
    instants = filled.astype("int64").to_numpy(copy=True)
    instants[missing] = 0
    return instants, missing


def _trading_day_of(ns: int) -> str | None:
    info = classify_session(
        pd.Timestamp(ns, unit="ns", tz="UTC").to_pydatetime().astimezone(UTC),
        IFVG_DOC_SESSION_SCHEME,
    )
    return info.trading_day.isoformat() if info.trading_day else None


def _typed(candidate_id: str, reason: str, *, bar_id: str | None = None, elapsed=None) -> dict:
    return {
        "candidate_id": candidate_id,
        "regime_fit_id": None,
        "fold_index": None,
        "partition": None,
        "panel_row_id": bar_id,
        "fold_local_cluster_id": None,
        "canonical_reporting_cluster_id": None,
        "distances": None,
        "assigned_distance": np.nan,
        "assignment_margin": np.nan,
        "valid": False,
        "missing_reason": reason,
        "elapsed_seconds_since_bar_close": (
            np.nan if elapsed is None else float(elapsed)
        ),
    }


_REQUIRED_PANEL_COLUMNS = (
    "row_id",
    "trading_day",
    "bar_close_ts_utc",
    "cbp_valid",
    "cbp_missing_reason",
)
#: R6.1-FIX §3.4: every VALUE column is required — there is no optional-column
#: fallback for the distance vector, the assigned distance or the margin.
_REQUIRED_ASSIGNMENT_COLUMNS = (
    "row_id",
    "fold_index",
    "partition",
    "regime_fit_id",
    "fold_local_cluster_id",
    "canonical_reporting_cluster_id",
    "distances",
    "assigned_distance",
    "assignment_margin",
    "valid",
    "missing_reason",
)


def assign_panel_regimes_to_candidates(
    panel_frame: pd.DataFrame,
    panel_assignments: pd.DataFrame,
    candidate_as_of: pd.DataFrame,
    *,
    protocol: RegimeProtocolEnvelope,
    partition_for_candidate: Mapping[str, tuple[int, str]] | None = None,
    max_staleness_seconds: int | None = None,
) -> pd.DataFrame:
    """The normative PIT rule (module docstring). ``partition_for_candidate``
    (candidate → ``(fold_index, partition)``) selects the fold-feature mode
    (the candidate's own partition of fit k); ``None`` selects the descriptive
    mode (OOS test rows only, lowest fold wins)."""

    payload = protocol.payload
    if payload.observation_granularity is not ObservationGranularity.CONTEXT_BAR_PANEL:
        raise ValueError("panel→candidate assignment requires the CONTEXT_BAR_PANEL grain")
    missing = sorted(set(_REQUIRED_PANEL_COLUMNS) - set(panel_frame.columns))
    if missing:
        raise ValueError(f"panel frame lacks required columns: {missing}")
    missing = sorted(set(_REQUIRED_ASSIGNMENT_COLUMNS) - set(panel_assignments.columns))
    if missing:
        raise ValueError(f"panel assignments lack required columns: {missing}")
    if not {"candidate_id", "as_of_ts_utc"} <= set(candidate_as_of.columns):
        raise ValueError("candidate as-of frame requires candidate_id and as_of_ts_utc")
    candidate_ids = candidate_as_of["candidate_id"].astype(str)
    if candidate_ids.duplicated().any():
        raise ValueError("candidate as-of frame repeats a candidate_id")
    if max_staleness_seconds is None:
        max_staleness_seconds = int(payload.panel_interval_seconds or 0) * (
            PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS
        )
    if max_staleness_seconds <= 0:
        raise ValueError("max_staleness_seconds must be positive")

    bars = panel_frame.loc[:, list(_REQUIRED_PANEL_COLUMNS)].copy()
    bars["row_id"] = bars["row_id"].astype(str)
    if bars["row_id"].duplicated().any():
        raise ValueError("panel frame repeats a row_id")
    bars["trading_day"] = bars["trading_day"].astype(str)
    bars["_close_ns"] = pd.to_datetime(bars["bar_close_ts_utc"], utc=True, errors="raise").astype(
        "int64"
    )
    bars = bars.sort_values(["trading_day", "_close_ns", "row_id"], kind="stable")
    by_day: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for day, group in bars.groupby("trading_day", sort=True):
        by_day[str(day)] = (
            group["_close_ns"].to_numpy(dtype=np.int64),
            group["row_id"].to_numpy(dtype=object),
            group["cbp_valid"].astype(bool).to_numpy(),
            group["cbp_missing_reason"].to_numpy(dtype=object),
        )

    lookup: dict[Any, pd.Series] = {}
    if len(panel_assignments):
        rows = panel_assignments.copy()
        rows["row_id"] = rows["row_id"].astype(str)
        rows = rows[rows["valid"].astype(bool)]
        if partition_for_candidate is None:
            rows = rows[rows["partition"].astype(str) == "test"]
            rows = rows.sort_values(["row_id", "fold_index"], kind="stable")
            lookup = {row_id: group.iloc[0] for row_id, group in rows.groupby("row_id", sort=True)}
        else:
            rows = rows.sort_values(["row_id", "fold_index", "partition"], kind="stable")
            for key, group in rows.groupby(["row_id", "fold_index", "partition"], sort=True):
                if len(group) != 1:
                    raise ValueError("panel assignments repeat a (row, fold, partition) row")
                lookup[(str(key[0]), int(key[1]), str(key[2]))] = group.iloc[0]

    as_of_ns, as_of_missing = _as_of_ns(candidate_as_of["as_of_ts_utc"])
    out: list[dict] = []
    for candidate_id, instant, anchor_missing in zip(
        candidate_ids, as_of_ns, as_of_missing, strict=True
    ):
        if anchor_missing:
            # R6.1-FIX §3.3: the candidate never reached (or lacks) the stage
            # anchor — preserved with the typed reason, never refused
            out.append(_typed(candidate_id, CANDIDATE_AS_OF_MISSING_REASON))
            continue
        day = _trading_day_of(int(instant))
        eligible = by_day.get(day) if day is not None else None
        if eligible is None:
            out.append(_typed(candidate_id, "no_completed_panel_bar"))
            continue
        closes, row_ids, valid_flags, reasons = eligible
        position = int(np.searchsorted(closes, int(instant), side="right")) - 1
        if position < 0:
            out.append(_typed(candidate_id, "no_completed_panel_bar"))
            continue
        bar_id = str(row_ids[position])
        elapsed = (int(instant) - int(closes[position])) / 1_000_000_000
        if not bool(valid_flags[position]):
            reason = _PANEL_REASON_FOR_VALIDITY.get(str(reasons[position]), "panel_gap")
            out.append(_typed(candidate_id, reason, bar_id=bar_id, elapsed=elapsed))
            continue
        if elapsed > max_staleness_seconds:
            out.append(_typed(candidate_id, "panel_stale", bar_id=bar_id, elapsed=elapsed))
            continue
        if partition_for_candidate is None:
            chosen = lookup.get(bar_id)
        else:
            own = partition_for_candidate.get(candidate_id)
            chosen = None if own is None else lookup.get((bar_id, int(own[0]), str(own[1])))
        if chosen is None:
            out.append(_typed(candidate_id, "coverage_gap", bar_id=bar_id, elapsed=elapsed))
            continue
        canonical = chosen["canonical_reporting_cluster_id"]
        distances = chosen["distances"]
        if distances is None or (isinstance(distances, float) and np.isnan(distances)):
            raise ValueError(
                f"panel assignment row {bar_id} is valid but carries no distance vector"
            )
        out.append(
            {
                "candidate_id": candidate_id,
                "regime_fit_id": str(chosen["regime_fit_id"]),
                "fold_index": int(chosen["fold_index"]),
                "partition": str(chosen["partition"]),
                "panel_row_id": bar_id,
                "fold_local_cluster_id": int(chosen["fold_local_cluster_id"]),
                "canonical_reporting_cluster_id": (
                    int(canonical) if canonical is not None and pd.notna(canonical) else None
                ),
                "distances": [float(v) for v in distances],
                "assigned_distance": float(chosen["assigned_distance"]),
                "assignment_margin": float(chosen["assignment_margin"]),
                "valid": True,
                "missing_reason": None,
                "elapsed_seconds_since_bar_close": float(elapsed),
            }
        )
    frame = pd.DataFrame(out, columns=list(OOS_ASSIGNMENT_COLUMNS))
    unknown = set(frame.loc[~frame["valid"].astype(bool), "missing_reason"].dropna()) - set(
        PANEL_ASSIGNMENT_MISSING_REASONS
    )
    if unknown:  # pragma: no cover - the vocabulary is closed above
        raise AssertionError(f"unregistered panel assignment reasons {sorted(unknown)}")
    return frame


def candidate_fold_oos_assignment(
    assignments: pd.DataFrame, candidate_ids: tuple[str, ...]
) -> pd.DataFrame:
    """Candidate grain: one row per candidate — the single fold in which it
    was scored OOS (``build_context_folds`` guarantees at most one); a
    candidate never scored OOS is typed ``no_oos_assignment``."""

    ids = tuple(str(value) for value in candidate_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate ids must be unique")
    rows: dict[str, dict] = {}
    missing = sorted(set(_REQUIRED_ASSIGNMENT_COLUMNS) - set(assignments.columns))
    if missing:
        raise ValueError(f"fit assignments lack required columns: {missing}")
    if len(assignments):
        oos = assignments[
            (assignments["partition"].astype(str) == "test")
            & assignments["valid"].astype(bool)
        ]
        duplicated = oos["row_id"].astype(str).duplicated()
        if duplicated.any():
            raise ValueError("a candidate carries more than one OOS assignment row")
        for row in oos.itertuples():
            distances = row.distances
            if distances is None or (isinstance(distances, float) and np.isnan(distances)):
                raise ValueError(
                    f"fit assignment row {row.row_id} is valid but carries no distance vector"
                )
            rows[str(row.row_id)] = {
                "candidate_id": str(row.row_id),
                "regime_fit_id": str(row.regime_fit_id),
                "fold_index": int(row.fold_index),
                "partition": "test",
                "panel_row_id": None,
                "fold_local_cluster_id": int(row.fold_local_cluster_id),
                "canonical_reporting_cluster_id": (
                    int(row.canonical_reporting_cluster_id)
                    if row.canonical_reporting_cluster_id is not None
                    and pd.notna(row.canonical_reporting_cluster_id)
                    else None
                ),
                "distances": [float(v) for v in distances],
                "assigned_distance": float(row.assigned_distance),
                "assignment_margin": float(row.assignment_margin),
                "valid": True,
                "missing_reason": None,
                "elapsed_seconds_since_bar_close": np.nan,
            }
    out = [
        rows.get(candidate_id) or _typed(candidate_id, "no_oos_assignment")
        for candidate_id in ids
    ]
    return pd.DataFrame(out, columns=list(OOS_ASSIGNMENT_COLUMNS))


# ── identity binding ─────────────────────────────────────────────────────────


def _null_or(value: Any, cast):
    if value is None:
        return None
    if isinstance(value, float | np.floating) and np.isnan(value):
        return None
    try:
        if pd.isna(value) is True:
            return None
    except (TypeError, ValueError):
        pass
    return cast(value)


def consulted_assignments_hash(assignments: pd.DataFrame) -> str:
    """Order-invariant hash of the assignment rows that were consulted —
    recomputable from the persisted fits' assignment frames. R6.1-FIX §3.1:
    it covers EVERY consulted value that can change the output: fit id, row
    id, fold, partition, local cluster id, canonical reporting id, the full
    distance vector, the assigned distance, the margin, validity and the
    missing reason."""

    if assignments.empty:
        return canonical_contract_sha256({"rows": [], "formula": OOS_ASSIGNMENT_FORMULA_VERSION})
    required = (
        "regime_fit_id",
        "row_id",
        "fold_index",
        "partition",
        "fold_local_cluster_id",
        "canonical_reporting_cluster_id",
        "distances",
        "assigned_distance",
        "assignment_margin",
        "valid",
        "missing_reason",
    )
    missing = sorted(set(required) - set(assignments.columns))
    if missing:
        raise ValueError(f"consulted assignments lack value columns: {missing}")
    keys = sorted(
        (
            str(row.regime_fit_id),
            str(row.row_id),
            int(row.fold_index),
            str(row.partition),
            _null_or(row.fold_local_cluster_id, int),
            _null_or(row.canonical_reporting_cluster_id, int),
            (
                None
                if row.distances is None
                or (isinstance(row.distances, float) and np.isnan(row.distances))
                else [float(v) for v in row.distances]
            ),
            _null_or(row.assigned_distance, float),
            _null_or(row.assignment_margin, float),
            bool(row.valid),
            _null_or(row.missing_reason, str),
        )
        for row in assignments.itertuples()
    )
    return canonical_contract_sha256({"rows": keys, "formula": OOS_ASSIGNMENT_FORMULA_VERSION})


def consulted_assignment_frame(verified_fit_assignments: Mapping[int, Any]) -> pd.DataFrame:
    """The concatenation (fold order) of VERIFIED per-fit assignment frames —
    the only lawful ``assignments`` input of the descriptive builders."""

    from .regime_store import VerifiedFitAssignments  # noqa: PLC0415

    frames = []
    for fold_index in sorted(verified_fit_assignments):
        verified = verified_fit_assignments[fold_index]
        if not isinstance(verified, VerifiedFitAssignments):
            raise TypeError(
                "descriptive assignments are built from VerifiedFitAssignments only "
                f"(fold {fold_index} supplied {type(verified).__name__}; an in-memory run "
                "frame is not evidence)"
            )
        frames.append(verified.frame)
    if not frames:
        return pd.DataFrame(columns=list(_REQUIRED_ASSIGNMENT_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def candidate_as_of_source_hash(candidate_as_of: pd.DataFrame) -> str:
    """Order-invariant hash of the ``(candidate_id, as_of)`` pairs. RA-07: the
    instants are parsed by the SAME rule as the assignment (`_as_of_ns`) — an
    unparseable NON-null anchor is a hard error, never silently a null; a
    true null is represented deterministically as ``None``."""

    if not {"candidate_id", "as_of_ts_utc"} <= set(candidate_as_of.columns):
        raise ValueError("candidate as-of frame requires candidate_id and as_of_ts_utc")
    instants, missing = _as_of_ns(candidate_as_of["as_of_ts_utc"])
    pairs = sorted(
        (
            str(candidate_id),
            None if anchor_missing else pd.Timestamp(int(instant), tz="UTC").isoformat(),
        )
        for candidate_id, instant, anchor_missing in zip(
            candidate_as_of["candidate_id"], instants, missing, strict=True
        )
    )
    return canonical_contract_sha256({"candidate_as_of": pairs})


def _frame_for_schema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(OOS_ASSIGNMENT_COLUMNS)].copy()
    for column in ("fold_index", "fold_local_cluster_id", "canonical_reporting_cluster_id"):
        out[column] = out[column].astype("Int64")
    for column in ("regime_fit_id", "partition", "panel_row_id", "missing_reason"):
        out[column] = out[column].astype(object).where(out[column].notna(), None)
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["valid"] = out["valid"].astype(bool)
    return out


def assignment_table_bytes(frame: pd.DataFrame) -> bytes:
    return frame_to_arrow_bytes(_frame_for_schema(frame), OOS_ASSIGNMENT_SCHEMA)


def build_regime_oos_assignment_artifact(
    frame: pd.DataFrame,
    *,
    protocol: RegimeProtocolEnvelope,
    verified_fit_assignments: Mapping[int, Any],
    regime_fold_set_id: str,
    fold_schedule_id: str,
    candidate_as_of: pd.DataFrame,
    candidate_as_of_source_ref: str,
    candidate_as_of_stage: AvailabilityStage,
    panel_context: PanelAssignmentContext | None = None,
) -> tuple[RegimeOosAssignmentEnvelope, bytes]:
    """The descriptive artifact over ``frame`` (built by the candidate-grain
    or panel-grain rule from the SAME verified frames). ``verified_fit_assignments``
    (fold index → ``VerifiedFitAssignments``) is the ONLY lawful assignment
    source (R6.1-FIX §3.1): the refs, the fit-id projection and the consulted
    hash all derive from it; every row is validated against the descriptive
    invariants (§3.4) before the identity is minted. ``candidate_as_of_stage``
    (RA-07) names the anchor column the hashed as-of instants came from and
    must equal the panel context's stage on the panel grain."""

    from .regime_store import VerifiedFitAssignments  # noqa: PLC0415

    payload = protocol.payload
    panel = payload.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL
    stage = AvailabilityStage(candidate_as_of_stage)
    if panel and panel_context is None:
        raise ValueError("the panel grain requires the panel assignment context")
    if panel and panel_context is not None and panel_context.candidate_as_of_stage is not stage:
        raise ValueError(
            "candidate_as_of_stage must equal the panel context's candidate_as_of_stage"
        )
    if panel and panel_context is not None:
        if panel_context.context_bar_panel_artifact_id != payload.panel_source_artifact_id:
            raise ValueError("panel context names a different context_bar_panel_artifact_id")
        if panel_context.panel_as_of_policy_id != payload.panel_as_of_policy_id:
            raise ValueError("panel context names a different as-of policy")
        if panel_context.panel_interval_seconds != payload.panel_interval_seconds:
            raise ValueError("panel context names a different panel interval")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise ValueError("assignment frame repeats a candidate")
    for fold_index, verified in verified_fit_assignments.items():
        if not isinstance(verified, VerifiedFitAssignments):
            raise TypeError(
                "build_regime_oos_assignment_artifact consumes VerifiedFitAssignments only "
                f"(fold {fold_index} supplied {type(verified).__name__})"
            )
        if verified.envelope.payload.resolved_regime_protocol_id != (
            protocol.resolved_regime_protocol_id
        ):
            raise ValueError(
                f"fit {verified.regime_fit_id[:12]}… belongs to another regime protocol"
            )
    refs = tuple(
        sorted(
            (verified.ref for verified in verified_fit_assignments.values()),
            key=lambda ref: ref.regime_fit_id,
        )
    )
    consulted = consulted_assignment_frame(verified_fit_assignments)
    validate_assignment_rows(
        _frame_for_schema(frame),
        cluster_count=int(payload.resolved_cluster_count),
        kind="descriptive",
    )
    table_bytes = assignment_table_bytes(frame)
    envelope_payload = RegimeOosAssignmentPayload(
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        observation_granularity=payload.observation_granularity,
        regime_fit_ids=tuple(ref.regime_fit_id for ref in refs),
        regime_fit_assignment_refs=refs,
        regime_fold_set_id=regime_fold_set_id,
        fold_schedule_id=fold_schedule_id,
        assignment_source="panel_pit" if panel else "candidate_fold_oos",
        panel_context=panel_context if panel else None,
        candidate_as_of_stage=stage,
        candidate_as_of_source_hash=candidate_as_of_source_hash(candidate_as_of),
        candidate_as_of_source_ref=candidate_as_of_source_ref,
        consulted_assignments_hash=consulted_assignments_hash(consulted),
        candidate_count=int(len(frame)),
        resolved_cluster_count=int(payload.resolved_cluster_count),
        assignment_schema_hash=OOS_ASSIGNMENT_SCHEMA_HASH,
    )
    envelope = RegimeOosAssignmentEnvelope.from_payload(
        envelope_payload, assignment_table_sha256=bytes_sha256(table_bytes)
    )
    return envelope, table_bytes


def save_regime_oos_assignment(
    root: Path, envelope: RegimeOosAssignmentEnvelope, table_bytes: bytes
):
    if bytes_sha256(table_bytes) != envelope.assignment_table_sha256:
        raise ValueError("assignment table bytes do not hash to the envelope")
    return save_or_reuse_envelope(
        Path(root),
        REGIME_OOS_ASSIGNMENT_STORE,
        envelope,
        extra_files={OOS_ASSIGNMENT_SIDECAR: table_bytes},
    )


def load_regime_oos_assignment(root: Path, artifact_id: str) -> RegimeOosAssignmentEnvelope:
    return load_verified_envelope(
        Path(root), REGIME_OOS_ASSIGNMENT_STORE, artifact_id, RegimeOosAssignmentEnvelope
    )


def load_regime_oos_assignment_frame(
    root: Path, envelope: RegimeOosAssignmentEnvelope
) -> pd.DataFrame:
    data = load_sidecar_bytes(
        Path(root),
        REGIME_OOS_ASSIGNMENT_STORE,
        envelope.regime_oos_assignment_id,
        OOS_ASSIGNMENT_SIDECAR,
    )
    if bytes_sha256(data) != envelope.assignment_table_sha256:
        raise ValueError("stored assignment table fails the envelope hash check")
    frame = frame_from_arrow_bytes(data)
    if len(frame) != envelope.payload.candidate_count:
        raise ValueError("stored assignment table row count disagrees with the payload")
    if arrow_schema_hash(pa.ipc.open_file(pa.BufferReader(data)).schema) != (
        envelope.payload.assignment_schema_hash
    ):
        raise ValueError("stored assignment table schema disagrees with the payload")
    frame["distances"] = frame["distances"].map(
        lambda v: None if v is None else [float(x) for x in v]
    )
    # R6.1-FIX §3.4: the descriptive invariants hold on every verified load
    validate_assignment_rows(
        frame, cluster_count=int(envelope.payload.resolved_cluster_count), kind="descriptive"
    )
    return frame


def verify_regime_oos_assignment_frame(
    envelope: RegimeOosAssignmentEnvelope, frame: pd.DataFrame
) -> None:
    if bytes_sha256(assignment_table_bytes(frame)) != envelope.assignment_table_sha256:
        raise ValueError(
            "the supplied assignment frame does not hash to the artifact's table hash"
        )


def _example_payload() -> RegimeOosAssignmentPayload:
    return RegimeOosAssignmentPayload(
        resolved_regime_protocol_id="a" * 64,
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        regime_fit_ids=("b" * 64,),
        regime_fit_assignment_refs=(
            FitAssignmentRef(
                regime_fit_id="b" * 64,
                assignments_sidecar_sha256="2" * 64,
                assignment_schema_hash="3" * 64,
            ),
        ),
        regime_fold_set_id="c" * 64,
        fold_schedule_id="d" * 64,
        assignment_source="candidate_fold_oos",
        panel_context=None,
        candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
        candidate_as_of_source_hash="e" * 64,
        candidate_as_of_source_ref="bundle_feature_view:" + "f" * 64,
        consulted_assignments_hash="1" * 64,
        candidate_count=0,
        resolved_cluster_count=3,
        assignment_schema_hash=OOS_ASSIGNMENT_SCHEMA_HASH,
    )


register_identity_pair(
    name="RegimeOosAssignment",
    envelope_cls=RegimeOosAssignmentEnvelope,
    payload_cls=RegimeOosAssignmentPayload,
    id_field="regime_oos_assignment_id",
    example_factory=_example_payload,
    extra_envelope_fields=("assignment_table_sha256",),
)

# static guard: the module's typed vocabulary is the registered one
if not set(_PANEL_REASON_FOR_VALIDITY.values()) <= set(PANEL_ASSIGNMENT_MISSING_REASONS):
    raise AssertionError("panel validity reasons must map onto registered assignment reasons")
