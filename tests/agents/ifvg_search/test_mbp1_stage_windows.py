"""R5B stage-cutoff and window-semantics suites.

TEST_MATRIX §3.8 'MBP-1 same-timestamp future-event exclusion' +
'Timestamp-only conservative cutoff' and §3.9 'Window-trigger semantics':
``PRE_TRIGGER_EXCLUSIVE`` uses ``<`` on the exact key,
``POST_TRIGGER_INCLUSIVE`` uses ``<=``; timestamp-only cutoffs are strictly
``ts_event < stage_ts`` with EVERY same-timestamp event excluded (or the
window typed ambiguous); transition windows honor their declared bounds;
no window is ever widened.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    IntervalBound,
    Mbp1FeatureWindowSpec,
    StageCutoffKind,
    StageEvidenceCutoff,
    WindowTriggerSemantics,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_stage_windows import (
    COMPLETED_BAR_CUTOFF_POLICY_ID,
    STAGE_ANCHOR_COLUMNS,
    admit_events,
    cutoff_from_exact_event,
    exact_key_from_event,
    ns_to_ts_utc,
    stage_anchor_frame_from_candidates,
    stage_cutoffs_from_candidate_row,
    ts_utc_to_ns,
    window_admission,
)
from tests.agents.ifvg_search.mbp1_fixture import (
    anchor_row,
    normalized_day,
    ns_at,
    raw_event,
)


def _same_ts_burst() -> pd.DataFrame:
    """Three events sharing one ts_event, totally ordered by (ts_recv,
    sequence, source_ordinal); one earlier and one later event around them."""

    t = ns_at(100)
    return normalized_day(
        [
            raw_event(ts_event=ns_at(50), ts_recv=ns_at(50) + 1, sequence=10),
            raw_event(ts_event=t, ts_recv=t + 1, sequence=11, bid_sz=11),
            raw_event(ts_event=t, ts_recv=t + 2, sequence=12, bid_sz=12),
            raw_event(ts_event=t, ts_recv=t + 3, sequence=13, bid_sz=13),
            raw_event(ts_event=ns_at(200), ts_recv=ns_at(200) + 1, sequence=14),
        ]
    )


def _spec(
    *,
    key: str = "ofl_snap_entry",
    from_stage: str | None = None,
    to_stage: str = "entry",
    semantics: WindowTriggerSemantics = WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE,
) -> Mbp1FeatureWindowSpec:
    return Mbp1FeatureWindowSpec(
        feature_window_key=key,
        feature_names=(f"{key}_spread_ticks",),
        from_stage=from_stage,
        to_stage=to_stage,
        lower_bound=IntervalBound.OPEN,
        upper_bound=IntervalBound.CLOSED,
        trigger_semantics=semantics,
        cutoff_policy_id="stage_evidence_cutoff_v2",
        minimum_event_count=1,
        missingness_policy_id="typed_null_preserve_row_v1",
    )


# ── exact-key semantics ──────────────────────────────────────────────────────


def test_exact_key_post_trigger_inclusive_admits_exactly_through_the_trigger():
    events = _same_ts_burst()
    trigger = events.iloc[2]  # the MIDDLE same-ts event
    cutoff = cutoff_from_exact_event("entry", trigger)
    admission = admit_events(
        events, cutoff, semantics=WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE
    )
    assert not admission.same_timestamp_ambiguous
    assert admission.admitted_count == 3  # earlier + first-burst + trigger itself
    admitted = events.loc[admission.mask]
    assert exact_key_from_event(admitted.iloc[-1]) == exact_key_from_event(trigger)


def test_exact_key_pre_trigger_exclusive_excludes_the_trigger():
    events = _same_ts_burst()
    trigger = events.iloc[2]
    cutoff = cutoff_from_exact_event("entry", trigger)
    admission = admit_events(
        events, cutoff, semantics=WindowTriggerSemantics.PRE_TRIGGER_EXCLUSIVE
    )
    assert admission.admitted_count == 2  # the trigger itself is OUT
    admitted = events.loc[admission.mask]
    assert int(admitted.iloc[-1]["sequence"]) == 11


def test_same_ts_event_after_the_stage_is_excluded_under_the_exact_key():
    """TEST_MATRIX §3.8: a same-ts event AFTER the stage cannot be admitted —
    only earlier ``(ts_recv, sequence, source_ordinal)`` pass."""

    events = _same_ts_burst()
    trigger = events.iloc[2]
    cutoff = cutoff_from_exact_event("entry", trigger)
    admission = admit_events(
        events, cutoff, semantics=WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE
    )
    excluded = events.loc[~admission.mask]
    assert int(events.iloc[3]["sequence"]) == 13  # the after-trigger same-ts event
    assert 13 in set(excluded["sequence"].astype(int))


# ── timestamp-only semantics ─────────────────────────────────────────────────


def _ts_cutoff(boundary_ns: int) -> StageEvidenceCutoff:
    iso = ns_to_ts_utc(boundary_ns) or ""
    return StageEvidenceCutoff(
        stage_id="entry",
        stage_as_of_ts_utc=iso,
        cutoff_kind=StageCutoffKind.TIMESTAMP_EXCLUSIVE,
        exact_source_order_key=None,
        completed_bar_close_ts_utc=None,
        same_timestamp_policy_id="exclude_all_same_timestamp_v1",
        source_evidence_ref=None,
    )


def test_timestamp_only_cutoff_is_strict_and_marks_same_ts_ambiguous():
    events = _same_ts_burst()
    admission = admit_events(
        events,
        _ts_cutoff(ns_at(100)),
        semantics=WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE,
    )
    # strictly ts_event < stage_ts: ALL three same-ts events excluded
    assert admission.admitted_count == 1
    assert admission.same_timestamp_ambiguous is True
    clean = admit_events(
        events,
        _ts_cutoff(ns_at(150)),
        semantics=WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE,
    )
    assert clean.admitted_count == 4
    assert clean.same_timestamp_ambiguous is False


def test_transition_window_bounds_are_honored_and_never_widened():
    events = _same_ts_burst()
    spec = _spec(key="ofl_win_inversion_entry", from_stage="inversion", to_stage="entry")
    from_cutoff = cutoff_from_exact_event("inversion", events.iloc[1])
    to_cutoff = cutoff_from_exact_event("entry", events.iloc[3])
    admission = window_admission(
        events, spec, from_cutoff=from_cutoff, to_cutoff=to_cutoff
    )
    # (from, to] on exact keys: strictly after event seq 11, through seq 13
    assert set(events.loc[admission.mask, "sequence"].astype(int)) == {12, 13}
    # a missing lower anchor can never silently widen the window
    with pytest.raises(ValueError, match="never a widened window"):
        window_admission(events, spec, from_cutoff=None, to_cutoff=to_cutoff)


def test_timestamp_only_transition_excludes_boundary_ties_on_both_sides():
    events = _same_ts_burst()
    spec = _spec(key="ofl_win_inversion_entry", from_stage="inversion", to_stage="entry")
    admission = window_admission(
        events,
        spec,
        from_cutoff=_ts_cutoff(ns_at(100)),
        to_cutoff=_ts_cutoff(ns_at(200)),
    )
    # lower boundary ties (three events at ns_at(100)) and the upper-boundary
    # tie (the event at ns_at(200)) are ALL excluded; ambiguity is flagged
    assert admission.admitted_count == 0
    assert admission.same_timestamp_ambiguous is True


# ── candidate-anchor cutoffs ─────────────────────────────────────────────────


def test_candidate_row_anchors_become_completed_bar_cutoffs():
    row = pd.Series(anchor_row("cand_x"))
    cutoffs = stage_cutoffs_from_candidate_row(row)
    assert set(cutoffs) == set(STAGE_ANCHOR_COLUMNS)
    entry = cutoffs["entry"]
    assert entry is not None
    assert entry.cutoff_kind is StageCutoffKind.COMPLETED_BAR_BOUNDARY
    assert entry.same_timestamp_policy_id == "exclude_all_same_timestamp_v1"
    assert ts_utc_to_ns(entry.completed_bar_close_ts_utc) == ns_at(360.0)
    assert COMPLETED_BAR_CUTOFF_POLICY_ID == "completed_bar_boundary_exclusive_v1"


def test_missing_anchor_yields_none_never_a_fabricated_cutoff():
    row = pd.Series(anchor_row("cand_y", inversion_s=None, entry_s=None))
    cutoffs = stage_cutoffs_from_candidate_row(row)
    assert cutoffs["inversion"] is None
    assert cutoffs["entry"] is None
    assert cutoffs["htf_tap"] is not None


def test_anchor_frame_extraction_requires_unique_candidates():
    table = pd.DataFrame([anchor_row("cand_a"), anchor_row("cand_a")])
    with pytest.raises(ValueError, match="unique per candidate_id"):
        stage_anchor_frame_from_candidates(table)
    good = stage_anchor_frame_from_candidates(pd.DataFrame([anchor_row("cand_a")]))
    assert list(good.columns[:3]) == ["candidate_id", "setup_id", "trading_day"]


def test_exact_cutoff_refuses_the_withdrawn_inf_bound():
    with pytest.raises(ValueError, match="unrepresentable"):
        StageEvidenceCutoff(
            stage_id="entry",
            stage_as_of_ts_utc="2026-01-13T14:06:00Z",
            cutoff_kind=StageCutoffKind.EXACT_SOURCE_ORDER_KEY,
            exact_source_order_key=("+inf", "0", 0, 0),
            completed_bar_close_ts_utc=None,
            same_timestamp_policy_id="exact_key_total_order_v1",
            source_evidence_ref=None,
        )
