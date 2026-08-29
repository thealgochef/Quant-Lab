"""Regime assignment sources for stratification (R6.1 §6.G; D7).

* ``load_regime_oos_assignment_table`` — the DESCRIPTIVE
  ``RegimeOosAssignmentArtifact`` (verified envelope + rehashed Arrow frame;
  never unpickles anything);
* ``regime_for_trades`` — the EXACT one-to-one join of executed trades onto
  that artifact on ``candidate_id`` (no nearest-time fallback; a trade whose
  candidate has no row is a TYPED miss);
* ``regime_filter_mask`` / ``regime_filter_ref`` / ``stratum_cohorts`` — the
  first consumer of ``study.cohort.RegimeFilterRef``: one
  ``CohortEnvelope`` per stratum (``interpretation_mode=DESCRIPTIVE_SLICE``,
  ``row_kind="executed_trade"``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..study.cohort import BASELINE_COHORT, CohortEnvelope, InterpretationMode, RegimeFilterRef
from .regime_oos_assignment import (
    RegimeOosAssignmentEnvelope,
    load_regime_oos_assignment,
    load_regime_oos_assignment_frame,
    verify_regime_oos_assignment_frame,
)

__all__ = [
    "TRADE_REGIME_COLUMNS",
    "UNASSIGNED_EVIDENCE_QUALITY_TOKEN",
    "load_regime_oos_assignment_table",
    "regime_for_trades",
    "regime_filter_mask",
    "regime_filter_ref",
    "stratum_cohorts",
]

TRADE_REGIME_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "candidate_id",
    "regime_fit_id",
    "fold_index",
    "canonical_reporting_cluster_id",
    "assignment_margin",
    "valid",
    "missing_reason",
)
#: The residual "unassigned" stratum is not a regime filter; its cohort is
#: typed through the evidence-quality filter instead.
UNASSIGNED_EVIDENCE_QUALITY_TOKEN = "regime_unassigned_v1"


def load_regime_oos_assignment_table(
    root: Path, artifact_id: str
) -> tuple[RegimeOosAssignmentEnvelope, pd.DataFrame]:
    """Verified envelope + rehashed frame of the descriptive OOS artifact."""

    envelope = load_regime_oos_assignment(Path(root), artifact_id)
    frame = load_regime_oos_assignment_frame(Path(root), envelope)
    verify_regime_oos_assignment_frame(envelope, frame)
    return envelope, frame


def regime_for_trades(trades: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    """Exact one-to-one join on ``candidate_id`` (descriptive OOS regimes).

    Refuses duplicated candidate ids on either side and missing columns; a
    trade whose candidate is absent from the artifact keeps its row with
    ``valid=False, missing_reason="candidate_not_in_assignment"``; a present
    but typed-null assignment keeps the artifact's own reason. Never a
    nearest-time or row-order fallback.
    """

    for column in ("trade_id", "candidate_id"):
        if column not in trades.columns:
            raise ValueError(f"executed trades lack the {column!r} column")
    for column in (
        "candidate_id",
        "regime_fit_id",
        "fold_index",
        "canonical_reporting_cluster_id",
        "assignment_margin",
        "valid",
        "missing_reason",
    ):
        if column not in assignment.columns:
            raise ValueError(f"the assignment artifact lacks the {column!r} column")
    trade_keys = trades["candidate_id"].astype(str)
    if trade_keys.duplicated().any():
        raise ValueError("executed trades repeat a candidate_id; the join is one-to-one")
    if trades["trade_id"].astype(str).duplicated().any():
        raise ValueError("executed trades repeat a trade_id")
    keyed = assignment.copy()
    keyed["candidate_id"] = keyed["candidate_id"].astype(str)
    if keyed["candidate_id"].duplicated().any():
        raise ValueError("the assignment artifact repeats a candidate_id")
    keyed = keyed.set_index("candidate_id")
    rows = []
    for trade_id, candidate_id in zip(trades["trade_id"].astype(str), trade_keys, strict=True):
        if candidate_id not in keyed.index:
            rows.append(
                {
                    "trade_id": trade_id,
                    "candidate_id": candidate_id,
                    "regime_fit_id": None,
                    "fold_index": None,
                    "canonical_reporting_cluster_id": None,
                    "assignment_margin": float("nan"),
                    "valid": False,
                    "missing_reason": "candidate_not_in_assignment",
                }
            )
            continue
        row = keyed.loc[candidate_id]
        valid = bool(row["valid"])
        cluster = row["canonical_reporting_cluster_id"]
        rows.append(
            {
                "trade_id": trade_id,
                "candidate_id": candidate_id,
                "regime_fit_id": str(row["regime_fit_id"]) if valid else None,
                "fold_index": int(row["fold_index"]) if valid else None,
                "canonical_reporting_cluster_id": (
                    int(cluster) if valid and pd.notna(cluster) else None
                ),
                "assignment_margin": (
                    float(row["assignment_margin"]) if valid else float("nan")
                ),
                "valid": valid,
                "missing_reason": None if valid else str(row["missing_reason"]),
            }
        )
    frame = pd.DataFrame(rows, columns=list(TRADE_REGIME_COLUMNS))
    frame["valid"] = frame["valid"].astype(bool)
    return frame


def regime_filter_mask(assigned: pd.DataFrame, cluster_ids: tuple[int, ...]) -> pd.Series:
    """Rows validly assigned to one of ``cluster_ids`` (a Boolean mask)."""

    ids = {int(value) for value in cluster_ids}
    clusters = pd.to_numeric(assigned["canonical_reporting_cluster_id"], errors="coerce")
    return assigned["valid"].astype(bool) & clusters.isin(ids)


def regime_filter_ref(
    *, protocol_id: str, fit_ids: tuple[str, ...], cluster_ids: tuple[int, ...]
) -> RegimeFilterRef:
    return RegimeFilterRef(
        resolved_regime_protocol_id=protocol_id,
        regime_fit_ids=tuple(sorted(set(fit_ids))),
        canonical_reporting_cluster_ids=tuple(sorted({int(value) for value in cluster_ids})),
    )


def stratum_cohorts(
    *,
    protocol_id: str,
    fit_ids: tuple[str, ...],
    cluster_ids: tuple[int, ...],
    base: CohortEnvelope = BASELINE_COHORT,
) -> dict[str, CohortEnvelope]:
    """One descriptive-slice cohort per stratum over EXECUTED trades:
    ``pooled_all`` (no regime filter), ``pooled_regime_covered`` (every
    reporting id), one per regime id, and ``unassigned`` (typed through the
    evidence-quality filter — the residual is not a regime filter)."""

    payload = base.payload.model_copy(
        update={
            "interpretation_mode": InterpretationMode.DESCRIPTIVE_SLICE,
            "row_kind": "executed_trade",
            "executed_counterfactual_scope": "executed_only",
        }
    )
    ordered = tuple(sorted({int(value) for value in cluster_ids}))
    cohorts: dict[str, CohortEnvelope] = {
        "pooled_all": CohortEnvelope.from_payload(
            payload.model_copy(update={"regime_filter": None})
        )
    }
    if ordered:
        cohorts["pooled_regime_covered"] = CohortEnvelope.from_payload(
            payload.model_copy(
                update={
                    "regime_filter": regime_filter_ref(
                        protocol_id=protocol_id, fit_ids=fit_ids, cluster_ids=ordered
                    )
                }
            )
        )
        for cluster in ordered:
            cohorts[f"regime:{cluster}"] = CohortEnvelope.from_payload(
                payload.model_copy(
                    update={
                        "regime_filter": regime_filter_ref(
                            protocol_id=protocol_id, fit_ids=fit_ids, cluster_ids=(cluster,)
                        )
                    }
                )
            )
    cohorts["unassigned"] = CohortEnvelope.from_payload(
        payload.model_copy(
            update={
                "regime_filter": None,
                "evidence_quality_filter": (UNASSIGNED_EVIDENCE_QUALITY_TOKEN,),
            }
        )
    )
    return cohorts
