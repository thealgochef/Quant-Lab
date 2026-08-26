"""Exact point-in-time MBP-1 stage cutoffs and window construction (R5B 3/4).

Implements ``StageEvidenceCutoff`` admission on the complete four-part order
key ``(ts_event, ts_recv, sequence, source_ordinal)``:

* ``EXACT_SOURCE_ORDER_KEY`` — the registered ``WindowTriggerSemantics``
  selects the comparator: ``PRE_TRIGGER_EXCLUSIVE`` uses ``<`` on the exact
  trigger key, ``POST_TRIGGER_INCLUSIVE`` uses ``<=`` — a same-``ts_event``
  event AFTER the trigger key is excluded exactly.
* ``TIMESTAMP_EXCLUSIVE`` / ``COMPLETED_BAR_BOUNDARY`` — strict
  ``ts_event < boundary``; EVERY same-timestamp event is excluded, and if any
  exists the window is *ambiguous*: the row is preserved with the typed
  missing reason ``same_timestamp_order_unavailable``. Windows are never
  widened.

No ``+inf`` bound is ever constructed (Amendment P0-G withdrew that rule);
exact keys serialize their timestamps as decimal nanosecond strings.

Stage anchors come from the v2 candidate evidence itself: the registered
IFVG lifecycle anchors map onto the candidate row's exact stage timestamps
(``tap_ts_utc`` → htf_tap, ``lock_ts_utc`` → parent_lock, ``armed_ts_utc`` →
opposing confirmation, ``inversion_ts_utc`` → inversion, ``entry_ts_utc`` →
entry decision). Those decisions are completed-bar decisions, so v2-derived
cutoffs are ``COMPLETED_BAR_BOUNDARY`` cutoffs under the versioned policy
``completed_bar_boundary_exclusive_v1`` — a numerically equal feed timestamp
is never assumed known before the bar-close decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .mbp1_source_contract import (
    MBP1_STAGES,
    IntervalBound,
    Mbp1FeatureWindowSpec,
    StageCutoffKind,
    StageEvidenceCutoff,
    WindowTriggerSemantics,
)

__all__ = [
    "STAGE_ANCHOR_COLUMNS",
    "COMPLETED_BAR_SAME_TS_POLICY_ID",
    "COMPLETED_BAR_CUTOFF_POLICY_ID",
    "ts_utc_to_ns",
    "ns_to_ts_utc",
    "exact_key_from_event",
    "cutoff_from_exact_event",
    "stage_cutoffs_from_candidate_row",
    "stage_anchor_frame_from_candidates",
    "WindowAdmission",
    "admit_events",
    "window_admission",
]

#: Registered lifecycle anchor → the exact v2 candidate-row timestamp column.
STAGE_ANCHOR_COLUMNS: dict[str, str] = {
    "htf_tap": "tap_ts_utc",
    "parent_lock": "lock_ts_utc",
    "opposing": "armed_ts_utc",
    "inversion": "inversion_ts_utc",
    "entry": "entry_ts_utc",
}

COMPLETED_BAR_SAME_TS_POLICY_ID = "exclude_all_same_timestamp_v1"
COMPLETED_BAR_CUTOFF_POLICY_ID = "completed_bar_boundary_exclusive_v1"

if tuple(STAGE_ANCHOR_COLUMNS) != MBP1_STAGES:
    raise AssertionError("stage anchor columns must cover the registered stages exactly")


def ts_utc_to_ns(value) -> int | None:
    """ISO timestamp (tz-aware or naive-UTC) → epoch nanoseconds, or None."""

    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return int(stamp.value)


def ns_to_ts_utc(value: int | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value, unit="ns", tz="UTC").isoformat()


def exact_key_from_event(row) -> tuple[str, str, int, int]:
    """One event's complete order key with ns timestamps as decimal strings."""

    return (
        str(int(row["ts_event"])),
        str(int(row["ts_recv"])),
        int(row["sequence"]),
        int(row["source_ordinal"]),
    )


def _numeric_key(key: tuple[str, str, int, int]) -> tuple[int, int, int, int]:
    return (int(key[0]), int(key[1]), int(key[2]), int(key[3]))


def cutoff_from_exact_event(
    stage_id: str,
    event_row,
    *,
    source_evidence_ref: str | None = None,
) -> StageEvidenceCutoff:
    """An exact-key cutoff for a stage whose triggering EVENT is known."""

    key = exact_key_from_event(event_row)
    return StageEvidenceCutoff(
        stage_id=stage_id,
        stage_as_of_ts_utc=ns_to_ts_utc(int(event_row["ts_event"])) or "",
        cutoff_kind=StageCutoffKind.EXACT_SOURCE_ORDER_KEY,
        exact_source_order_key=key,
        completed_bar_close_ts_utc=None,
        same_timestamp_policy_id="exact_key_total_order_v1",
        source_evidence_ref=source_evidence_ref,
    )


def stage_cutoffs_from_candidate_row(row) -> dict[str, StageEvidenceCutoff | None]:
    """The five registered anchors of ONE candidate as completed-bar cutoffs.

    A missing anchor (the candidate never reached that stage, or the
    evidence lacks the timestamp) yields ``None`` — downstream windows are
    preserved as typed nulls, never fabricated.
    """

    cutoffs: dict[str, StageEvidenceCutoff | None] = {}
    for stage, column in STAGE_ANCHOR_COLUMNS.items():
        ns = ts_utc_to_ns(row.get(column, None))
        if ns is None:
            cutoffs[stage] = None
            continue
        iso = ns_to_ts_utc(ns) or ""
        cutoffs[stage] = StageEvidenceCutoff(
            stage_id=stage,
            stage_as_of_ts_utc=iso,
            cutoff_kind=StageCutoffKind.COMPLETED_BAR_BOUNDARY,
            exact_source_order_key=None,
            completed_bar_close_ts_utc=iso,
            same_timestamp_policy_id=COMPLETED_BAR_SAME_TS_POLICY_ID,
            source_evidence_ref=None,
        )
    return cutoffs


def stage_anchor_frame_from_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Extract the exact anchor frame from a v2 ENTRY_CANDIDATE table."""

    required = ["candidate_id", "setup_id", "trading_day"]
    missing = [c for c in required if c not in candidates.columns]
    if missing:
        raise ValueError(f"candidate table lacks identity columns: {missing}")
    columns = [*required, *STAGE_ANCHOR_COLUMNS.values()]
    frame = pd.DataFrame(index=candidates.index)
    for column in columns:
        frame[column] = candidates[column] if column in candidates.columns else None
    if frame["candidate_id"].duplicated().any():
        raise ValueError("candidate anchors must be unique per candidate_id")
    return frame.reset_index(drop=True)


@dataclass(frozen=True, slots=True)
class WindowAdmission:
    """The admitted-event mask of one window plus its ambiguity evidence."""

    mask: np.ndarray
    same_timestamp_ambiguous: bool

    @property
    def admitted_count(self) -> int:
        return int(self.mask.sum())


def _cutoff_boundary_ns(cutoff: StageEvidenceCutoff) -> int:
    if cutoff.cutoff_kind is StageCutoffKind.COMPLETED_BAR_BOUNDARY:
        boundary = ts_utc_to_ns(cutoff.completed_bar_close_ts_utc)
    else:
        boundary = ts_utc_to_ns(cutoff.stage_as_of_ts_utc)
    if boundary is None:
        raise ValueError(f"cutoff for stage {cutoff.stage_id} has no boundary timestamp")
    return boundary


def admit_events(
    events: pd.DataFrame,
    cutoff: StageEvidenceCutoff,
    *,
    semantics: WindowTriggerSemantics,
    side: str = "upper",
    bound: IntervalBound = IntervalBound.CLOSED,
) -> WindowAdmission:
    """Admission mask against ONE cutoff on the complete order key.

    ``side='upper'`` keeps events at/before the cutoff per the semantics;
    ``side='lower'`` keeps events strictly after it (an open lower bound —
    the from-trigger itself is excluded under exact keys, and under
    timestamp-only evidence every same-timestamp event is excluded with the
    window marked ambiguous when any exists).
    """

    if side not in ("upper", "lower"):
        raise ValueError("side must be 'upper' or 'lower'")
    n = len(events)
    if n == 0:
        return WindowAdmission(mask=np.zeros(0, dtype=bool), same_timestamp_ambiguous=False)
    if cutoff.cutoff_kind is StageCutoffKind.EXACT_SOURCE_ORDER_KEY:
        trigger = _numeric_key(cutoff.exact_source_order_key)  # type: ignore[arg-type]
        keys = list(
            zip(
                events["ts_event"].astype("int64"),
                events["ts_recv"].astype("int64"),
                events["sequence"].astype("int64"),
                events["source_ordinal"].astype("int64"),
                strict=True,
            )
        )
        if side == "upper":
            if semantics is WindowTriggerSemantics.PRE_TRIGGER_EXCLUSIVE:
                mask = np.fromiter((k < trigger for k in keys), dtype=bool, count=n)
            else:  # POST_TRIGGER_INCLUSIVE / COMPLETED_BAR_AS_OF on an exact key
                mask = np.fromiter((k <= trigger for k in keys), dtype=bool, count=n)
        else:
            if bound is IntervalBound.CLOSED:
                mask = np.fromiter((k >= trigger for k in keys), dtype=bool, count=n)
            else:
                mask = np.fromiter((k > trigger for k in keys), dtype=bool, count=n)
        return WindowAdmission(mask=mask, same_timestamp_ambiguous=False)
    # timestamp-only evidence: strict comparison; ALL same-timestamp events
    # excluded; presence of any same-timestamp event marks ambiguity
    boundary = _cutoff_boundary_ns(cutoff)
    ts = events["ts_event"].astype("int64").to_numpy()
    same_ts = bool((ts == boundary).any())
    mask = ts < boundary if side == "upper" else ts > boundary
    return WindowAdmission(mask=mask, same_timestamp_ambiguous=same_ts)


def window_admission(
    events: pd.DataFrame,
    spec: Mbp1FeatureWindowSpec,
    *,
    from_cutoff: StageEvidenceCutoff | None,
    to_cutoff: StageEvidenceCutoff,
) -> WindowAdmission:
    """The combined admission of one registered window.

    Snapshot windows (``from_stage is None``) admit everything at/before the
    ``to`` trigger; transition windows intersect the open lower bound with
    the upper bound. Ambiguity at EITHER boundary marks the whole window
    ambiguous — its features become typed nulls, never approximations.
    """

    if spec.from_stage is not None and from_cutoff is None:
        raise ValueError(
            f"window {spec.feature_window_key} requires the {spec.from_stage} "
            "anchor; a missing lower anchor is typed-null evidence, never a "
            "widened window"
        )
    upper = admit_events(
        events,
        to_cutoff,
        semantics=spec.trigger_semantics,
        side="upper",
        bound=spec.upper_bound,
    )
    if from_cutoff is None:
        return upper
    lower = admit_events(
        events,
        from_cutoff,
        semantics=spec.trigger_semantics,
        side="lower",
        bound=spec.lower_bound,
    )
    return WindowAdmission(
        mask=upper.mask & lower.mask,
        same_timestamp_ambiguous=(
            upper.same_timestamp_ambiguous or lower.same_timestamp_ambiguous
        ),
    )
