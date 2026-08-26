"""Exact MBP-1 candidate/stage joins (R5B deliverable 7).

Joins are one-to-one on ``candidate_id`` — and nothing else. There is no
nearest-time fallback, no row-order fallback, no as-of merge: the cohort
never changes (rows are preserved exactly), and duplicate keys on either
side refuse outright. A candidate absent from the materialized feature
table receives NaN values; the registered ``stage_outside_coverage``
reason for such rows attaches only when ``include_evidence_columns`` is
set (review F9) — which is why the bundle-view seam separately REFUSES
absent rows outright: the materializer preserves every anchored candidate,
so a missing row there is a cohort misalignment, not typed-null evidence.
"""

from __future__ import annotations

import pandas as pd

from .mbp1_arrow_schemas import (
    mbp1_window_missing_reason_fields,
    mbp1_window_validity_fields,
)
from .mbp1_source_contract import assert_no_deep_book_identifiers, mbp1_feature_names

__all__ = ["join_mbp1_features"]

_JOIN_KEY = "candidate_id"
_MISSING_ROW_REASON = "stage_outside_coverage"


def join_mbp1_features(
    candidate_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | None = None,
    include_evidence_columns: bool = False,
) -> pd.DataFrame:
    """Extend ``candidate_frame`` by the MBP-1 columns, exactly one-to-one.

    ``feature_columns`` defaults to the complete registered metric set; the
    per-window validity/missing-reason evidence columns join only when
    ``include_evidence_columns`` is set (they are evidence, not model
    features). Rows of ``candidate_frame`` are preserved exactly — order,
    count, and identity columns untouched.
    """

    if _JOIN_KEY not in candidate_frame.columns:
        raise ValueError("candidate frame lacks the exact join key 'candidate_id'")
    if _JOIN_KEY not in feature_frame.columns:
        raise ValueError("MBP-1 feature frame lacks the exact join key 'candidate_id'")
    if candidate_frame[_JOIN_KEY].duplicated().any():
        raise ValueError("candidate frame has duplicate candidate ids; exact join refused")
    if feature_frame[_JOIN_KEY].duplicated().any():
        raise ValueError("MBP-1 feature frame has duplicate candidate ids; exact join refused")

    metrics = tuple(feature_columns or mbp1_feature_names())
    assert_no_deep_book_identifiers(metrics)
    validity = mbp1_window_validity_fields()
    reasons = mbp1_window_missing_reason_fields()
    wanted = [*metrics, *(validity if include_evidence_columns else ())]
    reason_columns = tuple(reasons) if include_evidence_columns else ()
    missing_source = [
        column
        for column in (*wanted, *reason_columns)
        if column not in feature_frame.columns
    ]
    if missing_source:
        raise ValueError(
            f"MBP-1 feature frame lacks requested columns: {missing_source}"
        )
    overlap = [c for c in (*wanted, *reason_columns) if c in candidate_frame.columns]
    if overlap:
        raise ValueError(f"candidate frame already carries MBP-1 columns: {overlap}")

    lookup = feature_frame.set_index(_JOIN_KEY, verify_integrity=True)
    joined = candidate_frame.merge(
        feature_frame[[_JOIN_KEY, *wanted, *reason_columns]],
        on=_JOIN_KEY,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    matched = candidate_frame[_JOIN_KEY].astype(str).isin(
        lookup.index.astype(str)
    )
    if include_evidence_columns:
        for column in validity:
            joined.loc[~matched.to_numpy(), column] = False
            joined[column] = joined[column].astype(bool)
        for column in reason_columns:
            joined.loc[~matched.to_numpy(), column] = _MISSING_ROW_REASON
    if len(joined) != len(candidate_frame):
        raise AssertionError("exact join changed the candidate cohort")
    return joined
