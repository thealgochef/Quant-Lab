"""R5B offline materializer suites: formulas, typed missingness, determinism,
immutable persistence, exact joins, and coverage reports.

The synthetic feature-formula fixtures are hand-computed (R5B deliverable
12): every asserted number below is derived on paper from the crafted event
stream in ``mbp1_fixture.default_day_events``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage import (
    build_mbp1_coverage_report,
    load_mbp1_coverage_report,
    save_mbp1_coverage_report,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_feature_join import (
    join_mbp1_features,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_feature_materializer import (
    Mbp1FeatureArtifactEnvelope,
    load_feature_frame,
    load_mbp1_feature_artifact,
    load_stage_evidence_frame,
    materialize_from_stored_source,
    materialize_mbp1_features,
    save_mbp1_feature_artifact,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
    save_mbp1_source_artifact,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    MBP1_MISSING_REASONS,
    mbp1_feature_names,
)
from tests.agents.ifvg_search.mbp1_fixture import (
    FIXTURE_DAY,
    MBP1_RESOLVED_BLOCK,
    anchor_row,
    build_fixture_source,
    declared_interval,
    default_anchors,
    default_day_events,
    normalized_day,
    ns_at,
    raw_event,
)


def _materialize(anchors: pd.DataFrame | None = None, events=None, *, intervals=None):
    source, _bytes, events_by_day = build_fixture_source(
        events, intervals_by_day={FIXTURE_DAY: tuple(intervals or ())}
    )
    return materialize_mbp1_features(
        source,
        anchors if anchors is not None else default_anchors(),
        resolved_block=MBP1_RESOLVED_BLOCK,
        events_by_day=events_by_day,
    )


def _row(frame: pd.DataFrame, candidate_id: str) -> pd.Series:
    return frame.set_index("candidate_id").loc[candidate_id]


# ── hand-computed formula fixtures ───────────────────────────────────────────


def test_snapshot_formulas_match_hand_computed_values() -> None:
    _envelope, features, _evidence = _materialize()
    row = _row(features, "cand_a")
    # htf_tap cutoff 60s (completed-bar, strict <): last admitted event is
    # the 30s cancel — bid 20000×4(2ct), ask 20001×6(3ct)
    assert row["ofl_snap_htf_tap_valid"]
    assert row["ofl_snap_htf_tap_spread_ticks"] == pytest.approx(1.0)
    assert row["ofl_snap_htf_tap_queue_imbalance"] == pytest.approx((4 - 6) / 10)
    assert row["ofl_snap_htf_tap_order_count_imbalance"] == pytest.approx((2 - 3) / 5)
    micro = (20001 * 4 + 20000 * 6) / 10
    assert row["ofl_snap_htf_tap_microprice_offset_ticks"] == pytest.approx(
        micro - 20000.5
    )
    assert row["ofl_snap_htf_tap_bid_sz"] == pytest.approx(4.0)
    assert row["ofl_snap_htf_tap_ask_ct"] == pytest.approx(3.0)
    # entry cutoff 360s: last admitted is the 310s sell trade — bid
    # 20000×6(3), ask 20001×5(2)
    assert row["ofl_snap_entry_queue_imbalance"] == pytest.approx((6 - 5) / 11)
    assert row["ofl_snap_entry_microprice_offset_ticks"] == pytest.approx(
        (20001 * 6 + 20000 * 5) / 11 - 20000.5
    )


def test_transition_formulas_match_hand_computed_values() -> None:
    _envelope, features, _evidence = _materialize()
    row = _row(features, "cand_a")
    # (inversion 240s, entry 360s]: exactly the two trades at 250s and 310s
    assert row["ofl_win_inversion_entry_valid"]
    assert row["ofl_win_inversion_entry_event_count"] == pytest.approx(2.0)
    # CKS OFI over the single admitted pair: bid equal px (+6−8), ask px
    # improved down (−5) → −7
    assert row["ofl_win_inversion_entry_ofi_sum"] == pytest.approx(-7.0)
    assert row["ofl_win_inversion_entry_aggressive_buy_frac"] == pytest.approx(0.5)
    assert row["ofl_win_inversion_entry_aggressive_sell_frac"] == pytest.approx(0.5)
    assert row["ofl_win_inversion_entry_depletion_events"] == pytest.approx(1.0)
    assert row["ofl_win_inversion_entry_replenishment_events"] == pytest.approx(1.0)
    # absorption: traded 5 / (1 + |20000.5 − 20001.0|) = 5 / 1.5
    assert row["ofl_win_inversion_entry_absorption_score"] == pytest.approx(5 / 1.5)
    # 120-second anchor-to-anchor span; both events are trades
    assert row["ofl_win_inversion_entry_quote_intensity"] == pytest.approx(0.0)
    assert row["ofl_win_inversion_entry_trade_intensity"] == pytest.approx(2 / 120)


def test_formula_edge_cases_are_nan_with_the_window_still_valid() -> None:
    """Zero trades in a window → NaN aggressor fractions, window VALID —
    NaN-by-formula is not missing evidence."""

    anchors = pd.DataFrame([anchor_row("cand_q", armed_s=35.0, inversion_s=75.0)])
    _envelope, features, _evidence = _materialize(anchors)
    row = _row(features, "cand_q")
    # (opposing 35s, inversion 75s]: exactly the 70s quote — no trades
    assert row["ofl_win_opposing_inversion_valid"]
    assert row["ofl_win_opposing_inversion_event_count"] == pytest.approx(1.0)
    assert np.isnan(row["ofl_win_opposing_inversion_aggressive_buy_frac"])
    assert row["ofl_win_opposing_inversion_trade_intensity"] == pytest.approx(0.0)


# ── typed missingness (every registered reason) ──────────────────────────────


def test_missing_day_marks_every_window_no_mbp1_partition() -> None:
    anchors = pd.DataFrame([anchor_row("cand_gone", day="2026-01-14")])
    source, _bytes, events = build_fixture_source()
    envelope, features, evidence = materialize_mbp1_features(
        source, anchors, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events
    )
    assert len(features) == 1  # the row is PRESERVED
    row = _row(features, "cand_gone")
    assert not row["ofl_snap_entry_valid"]
    assert row["ofl_snap_entry_missing_reason"] == "no_mbp1_partition"
    assert set(evidence["missing_reason"]) == {"no_mbp1_partition"}


def test_low_coverage_day_marks_windows_coverage_below_threshold() -> None:
    """A VERIFIED declared interval invalidating ~97% of the physical span
    drops the day below the contract's coverage policy (policy v2)."""

    anchors = pd.DataFrame([anchor_row("cand_cov")])
    _envelope, features, _evidence = _materialize(
        anchors, intervals=(declared_interval(ns_at(10), ns_at(590)),)
    )
    row = _row(features, "cand_cov")
    assert row["ofl_snap_entry_missing_reason"] == "coverage_below_threshold"


def test_giant_sequence_jump_never_lowers_coverage() -> None:
    """R5B.1: the R5B fixture's 'giant sequence gap' is now a diagnostic —
    the day stays evidenced_complete and every window that has events stays
    valid (no window is silently typed from raw sequence continuity)."""

    rows = [
        raw_event(ts_event=ns_at(0), sequence=100),
        raw_event(ts_event=ns_at(10), sequence=101),
        raw_event(ts_event=ns_at(590), sequence=900),
        raw_event(ts_event=ns_at(600), sequence=901),
    ]
    anchors = pd.DataFrame([anchor_row("cand_cov")])
    events = {FIXTURE_DAY: normalized_day(rows)}
    source, _bytes, events_by_day = build_fixture_source(events)
    _envelope, features, _evidence = materialize_mbp1_features(
        source, anchors, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events_by_day
    )
    row = _row(features, "cand_cov")
    assert row["ofl_snap_entry_missing_reason"] != "coverage_below_threshold"
    assert row["ofl_snap_entry_valid"]  # the 10 s quote is admitted (< 360 s)
    partition = source.payload.ordered_partitions[0]
    assert partition.coverage_fraction == 1.0
    assert partition.sequence_jump_diagnostics.positive_jump_count == 1


def test_superseded_resolution_is_refused_by_the_materializer() -> None:
    """Review F8: the HISTORICAL v1 resolution (R5B activation) can never be
    materialized under the v2 semantics — the block's formula/materializer
    versions are the identity this materializer implements."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
        MBP1_ACTIVATION_ENVELOPE,
    )

    source, _bytes, events_by_day = build_fixture_source()
    with pytest.raises(ValueError, match="superseded resolution is refused"):
        materialize_mbp1_features(
            source,
            default_anchors(),
            resolved_block=MBP1_ACTIVATION_ENVELOPE,
            events_by_day=events_by_day,
        )
    envelope, _features, _evidence = _materialize()
    assert envelope.payload.formula_version == MBP1_RESOLVED_BLOCK.payload.formula_version
    assert envelope.payload.materializer_version == (
        MBP1_RESOLVED_BLOCK.payload.materializer_version
    )


def test_missing_anchor_and_out_of_span_anchor_mark_stage_outside_coverage() -> None:
    anchors = pd.DataFrame(
        [
            anchor_row("cand_noinv", inversion_s=None),
            anchor_row("cand_late", entry_s=4000.0),
        ]
    )
    _envelope, features, _evidence = _materialize(anchors)
    no_inversion = _row(features, "cand_noinv")
    assert no_inversion["ofl_snap_inversion_missing_reason"] == "stage_outside_coverage"
    assert no_inversion["ofl_win_inversion_entry_missing_reason"] == (
        "stage_outside_coverage"
    )
    assert no_inversion["ofl_snap_entry_valid"]  # unaffected windows stay valid
    late = _row(features, "cand_late")
    assert late["ofl_snap_entry_missing_reason"] == "stage_outside_coverage"


def test_boundary_tie_marks_same_timestamp_order_unavailable() -> None:
    anchors = pd.DataFrame([anchor_row("cand_tie", entry_s=310.0)])
    _envelope, features, evidence = _materialize(anchors)
    row = _row(features, "cand_tie")
    assert not row["ofl_snap_entry_valid"]
    assert row["ofl_snap_entry_missing_reason"] == "same_timestamp_order_unavailable"
    scoped = evidence[
        (evidence["candidate_id"] == "cand_tie")
        & (evidence["feature_window_key"] == "ofl_snap_entry")
    ]
    assert bool(scoped.iloc[0]["same_timestamp_ambiguous"])


def test_declared_gap_overlapping_window_marks_declared_source_gap() -> None:
    """An explicitly DECLARED real source gap (partition-scope manifest)
    becomes typed missing evidence on exactly the intersecting windows; no
    window is widened or imputed; the candidate row is preserved."""

    rows = [
        raw_event(ts_event=ns_at(0), sequence=100),
        raw_event(ts_event=ns_at(100), sequence=101),
        raw_event(ts_event=ns_at(200), sequence=102),
        raw_event(ts_event=ns_at(280), sequence=103),
        raw_event(ts_event=ns_at(300), sequence=110),  # a raw jump — diagnostic only
        raw_event(ts_event=ns_at(340), sequence=111),
        raw_event(ts_event=ns_at(600), sequence=112),
    ]
    anchors = pd.DataFrame([anchor_row("cand_gap")])
    # WITHOUT a declared interval the jump types nothing
    _envelope, features, _evidence = _materialize(
        anchors, events={FIXTURE_DAY: normalized_day(rows)}
    )
    row = _row(features, "cand_gap")
    assert row["ofl_win_inversion_entry_valid"]
    # WITH the declared (280 s, 300 s) interval: the 20 s gap keeps the DAY
    # above the 0.95 coverage threshold, and (inversion 240s, entry 360s]
    # intersects it → declared_source_gap on exactly that window
    _envelope, features, evidence = _materialize(
        anchors,
        events={FIXTURE_DAY: normalized_day(rows)},
        intervals=(declared_interval(ns_at(280), ns_at(300)),),
    )
    row = _row(features, "cand_gap")
    assert row["ofl_win_inversion_entry_missing_reason"] == "declared_source_gap"
    assert not row["ofl_win_inversion_entry_valid"]
    # the pre-gap htf_tap snapshot (cutoff 60s) is untouched; row preserved
    assert row["ofl_snap_htf_tap_valid"]
    assert list(features["candidate_id"]) == ["cand_gap"]
    scoped = evidence[
        (evidence["candidate_id"] == "cand_gap")
        & (evidence["feature_window_key"] == "ofl_win_inversion_entry")
    ]
    assert scoped.iloc[0]["missing_reason"] == "declared_source_gap"


def test_day_without_partition_evidence_is_coverage_evidence_unavailable() -> None:
    """A day with events but NO partition-scope completeness evidence is
    ``completeness_unknown`` → every window typed
    ``coverage_evidence_unavailable`` (fail closed); the row is preserved."""

    events = {FIXTURE_DAY: default_day_events()}
    source, _bytes, events_by_day = build_fixture_source(events, coverage_evidence={})
    anchors = pd.DataFrame([anchor_row("cand_unknown")])
    _envelope, features, evidence = materialize_mbp1_features(
        source, anchors, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events_by_day
    )
    row = _row(features, "cand_unknown")
    assert set(evidence["missing_reason"]) == {"coverage_evidence_unavailable"}
    assert not row["ofl_snap_entry_valid"]
    assert source.payload.ordered_partitions[0].completeness_status.value == (
        "completeness_unknown"
    )


def test_roll_boundary_marks_instrument_roll_boundary() -> None:
    rows = [
        raw_event(ts_event=ns_at(0), sequence=100, instrument_id=42),
        raw_event(ts_event=ns_at(250), sequence=101, instrument_id=42),
        raw_event(ts_event=ns_at(300), sequence=102, instrument_id=43),
        raw_event(ts_event=ns_at(600), sequence=103, instrument_id=43),
    ]
    anchors = pd.DataFrame([anchor_row("cand_roll")])
    _envelope, features, _evidence = _materialize(
        anchors, events={FIXTURE_DAY: normalized_day(rows)}
    )
    row = _row(features, "cand_roll")
    assert row["ofl_win_inversion_entry_missing_reason"] == "instrument_roll_boundary"


def test_empty_window_marks_minimum_event_count_not_met() -> None:
    anchors = pd.DataFrame([anchor_row("cand_min", armed_s=185.0, inversion_s=186.0)])
    _envelope, features, _evidence = _materialize(anchors)
    row = _row(features, "cand_min")
    assert row["ofl_win_opposing_inversion_missing_reason"] == (
        "minimum_event_count_not_met"
    )


def test_every_emitted_reason_is_registered() -> None:
    anchors = pd.DataFrame(
        [
            anchor_row("cand_a"),
            anchor_row("cand_tie", entry_s=310.0),
            anchor_row("cand_noinv", inversion_s=None),
        ]
    )
    _envelope, _features, evidence = _materialize(anchors)
    emitted = {r for r in evidence["missing_reason"].unique() if r}
    assert emitted <= set(MBP1_MISSING_REASONS)


# ── determinism and cohort preservation ──────────────────────────────────────


def test_batch_and_repeat_materialization_are_identical() -> None:
    envelope_one, features_one, evidence_one = _materialize()
    envelope_two, features_two, evidence_two = _materialize()
    assert (
        envelope_one.mbp1_feature_artifact_id == envelope_two.mbp1_feature_artifact_id
    )
    assert envelope_one.feature_table_sha256 == envelope_two.feature_table_sha256
    pd.testing.assert_frame_equal(features_one, features_two)
    pd.testing.assert_frame_equal(evidence_one, evidence_two)
    # a single-candidate batch reproduces that candidate's row exactly
    solo_anchors = default_anchors().iloc[[0]]
    _solo_env, solo_features, _solo_evidence = _materialize(solo_anchors)
    pd.testing.assert_series_equal(
        _row(solo_features, "cand_a"), _row(features_one, "cand_a")
    )


def test_rows_are_preserved_and_identity_is_input_sensitive() -> None:
    envelope, features, _evidence = _materialize()
    assert list(features["candidate_id"]) == ["cand_a", "cand_b"]
    assert envelope.payload.candidate_count == 2
    # changed anchors → new artifact identity
    moved = default_anchors().copy()
    moved.loc[0, "entry_ts_utc"] = pd.Timestamp(ns_at(361), unit="ns", tz="UTC").isoformat()
    moved_envelope, _f, _e = _materialize(moved)
    assert (
        moved_envelope.mbp1_feature_artifact_id != envelope.mbp1_feature_artifact_id
    )
    # changed event bytes → new source id → new feature identity
    changed = default_day_events()
    changed.loc[0, "bid_sz"] = 77
    changed_envelope, _f2, _e2 = _materialize(events={FIXTURE_DAY: changed})
    assert (
        changed_envelope.payload.mbp1_source_artifact_id
        != envelope.payload.mbp1_source_artifact_id
    )


def test_divergent_window_registry_is_refused() -> None:
    source, _bytes, events = build_fixture_source()
    truncated = MBP1_RESOLVED_BLOCK.model_copy(
        update={
            "payload": MBP1_RESOLVED_BLOCK.payload.model_copy(
                update={
                    "mbp1_feature_windows": MBP1_RESOLVED_BLOCK.payload.mbp1_feature_windows[:3]
                }
            )
        }
    )
    with pytest.raises(Exception, match="exactly one registered|id does not hash"):
        materialize_mbp1_features(
            source,
            default_anchors(),
            resolved_block=truncated,
            events_by_day=events,
        )


# ── immutable persistence ────────────────────────────────────────────────────


def test_feature_artifact_save_reload_reuse_and_tamper_refusal(tmp_path) -> None:
    source, event_bytes, events = build_fixture_source()
    envelope, features, evidence = materialize_mbp1_features(
        source, default_anchors(), resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events
    )
    root = tmp_path / "store"
    save_mbp1_source_artifact(root, source, event_bytes)
    save_mbp1_feature_artifact(root, envelope, features, evidence)
    reloaded = load_mbp1_feature_artifact(root, envelope.mbp1_feature_artifact_id)
    assert isinstance(reloaded, Mbp1FeatureArtifactEnvelope)
    assert reloaded.feature_table_sha256 == envelope.feature_table_sha256
    features_back = load_feature_frame(root, reloaded)
    evidence_back = load_stage_evidence_frame(root, reloaded)
    assert list(features_back.columns) == list(features.columns)
    assert len(features_back) == len(features)
    assert len(evidence_back) == len(evidence)
    # verified reuse on the second save
    save_mbp1_feature_artifact(root, envelope, features, evidence)
    # a divergent frame refuses BEFORE any store write
    with pytest.raises(ValueError, match="does not hash"):
        save_mbp1_feature_artifact(
            root, envelope, features.assign(ofl_snap_entry_bid_sz=0.0), evidence
        )
    # re-materializing from the STORED source reproduces the identity
    re_envelope, _f, _e = materialize_from_stored_source(
        root, source, default_anchors(), resolved_block=MBP1_RESOLVED_BLOCK
    )
    assert re_envelope.mbp1_feature_artifact_id == envelope.mbp1_feature_artifact_id


# ── exact join ───────────────────────────────────────────────────────────────


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["cand_a", "cand_b", "cand_absent"],
            "direction": ["long", "short", "long"],
        }
    )


def test_join_is_exact_one_to_one_with_typed_nulls() -> None:
    _envelope, features, _evidence = _materialize()
    joined = join_mbp1_features(
        _candidate_frame(), features, include_evidence_columns=True
    )
    assert len(joined) == 3  # cohort preserved exactly
    absent = joined.set_index("candidate_id").loc["cand_absent"]
    assert np.isnan(absent["ofl_snap_entry_spread_ticks"])
    assert absent["ofl_snap_entry_valid"] == False  # noqa: E712 — numpy bool
    assert absent["ofl_snap_entry_missing_reason"] == "stage_outside_coverage"
    present = joined.set_index("candidate_id").loc["cand_a"]
    assert present["ofl_snap_entry_valid"] == True  # noqa: E712


def test_join_refuses_duplicates_and_overlaps() -> None:
    _envelope, features, _evidence = _materialize()
    duplicated = pd.concat([_candidate_frame(), _candidate_frame().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate candidate ids"):
        join_mbp1_features(duplicated, features)
    with pytest.raises(ValueError, match="duplicate candidate ids"):
        join_mbp1_features(
            _candidate_frame(), pd.concat([features, features.iloc[[0]]])
        )
    overlapping = _candidate_frame().assign(ofl_snap_entry_bid_sz=1.0)
    with pytest.raises(ValueError, match="already carries MBP-1 columns"):
        join_mbp1_features(overlapping, features)
    with pytest.raises(ValueError, match="lacks requested columns"):
        join_mbp1_features(
            _candidate_frame(), features.drop(columns=["ofl_snap_entry_bid_sz"])
        )


# ── coverage report ──────────────────────────────────────────────────────────


def test_coverage_report_aggregates_and_persists(tmp_path) -> None:
    source, _event_bytes, events = build_fixture_source()
    anchors = pd.DataFrame(
        [anchor_row("cand_a"), anchor_row("cand_tie", entry_s=310.0)]
    )
    envelope, features, evidence = materialize_mbp1_features(
        source, anchors, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events
    )
    report = build_mbp1_coverage_report(source, envelope, features, evidence)
    payload = report.payload
    assert payload.research_boundary == "research_only_offline"
    assert payload.coverage_policy_id == "mbp1_source_coverage_declared_evidence_v2"
    assert payload.candidate_count == 2
    assert payload.day_rows[0].trading_day == FIXTURE_DAY
    assert payload.day_rows[0].completeness_status.value == "evidenced_complete"
    assert payload.day_rows[0].sequence_positive_jump_count == 0
    entry_row = next(
        row for row in payload.window_rows if row.feature_window_key == "ofl_snap_entry"
    )
    assert entry_row.valid_count == 1
    assert entry_row.reason_counts == {"same_timestamp_order_unavailable": 1}
    assert set(payload.per_feature_nonnull_fraction) == set(mbp1_feature_names())
    root = tmp_path / "store"
    save_mbp1_coverage_report(root, report)
    reloaded = load_mbp1_coverage_report(root, report.mbp1_coverage_report_id)
    assert reloaded.model_dump(mode="json") == report.model_dump(mode="json")


# ── review-round additions (adversarial findings F1/F3/F4/F8) ────────────────


def test_supplied_events_must_be_the_source_artifacts_evidence() -> None:
    """Review F1: evidence↔identity binding is VERIFIED — events that do not
    hash to the source artifact's coverage refuse before any window, as does
    withholding a covered anchor day or supplying an uncovered day."""

    source, _bytes, events = build_fixture_source()
    tampered = {FIXTURE_DAY: events[FIXTURE_DAY].copy()}
    tampered[FIXTURE_DAY].loc[0, "bid_sz"] = 999
    with pytest.raises(ValueError, match="not the artifact's evidence"):
        materialize_mbp1_features(
            source,
            default_anchors(),
            resolved_block=MBP1_RESOLVED_BLOCK,
            events_by_day=tampered,
        )
    with pytest.raises(ValueError, match="whose events were not supplied"):
        materialize_mbp1_features(
            source,
            default_anchors(),
            resolved_block=MBP1_RESOLVED_BLOCK,
            events_by_day={},
        )
    with pytest.raises(ValueError, match="does not cover"):
        materialize_mbp1_features(
            source,
            default_anchors(),
            resolved_block=MBP1_RESOLVED_BLOCK,
            events_by_day={**events, "2026-01-14": events[FIXTURE_DAY]},
        )


def test_same_ts_after_stage_event_cannot_change_any_feature() -> None:
    """TEST_MATRIX §3.8 / DT §6.2 mandatory row (review F3): under an
    EXACT-KEY cutoff, mutating a same-timestamp event ordered AFTER the
    stage trigger leaves every materialized feature and validity flag
    byte-identical."""

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_stage_windows import (
        cutoff_from_exact_event,
        stage_cutoffs_from_candidate_row,
    )

    t = ns_at(360)
    burst = [
        raw_event(ts_event=ns_at(100), ts_recv=ns_at(100) + 1, sequence=10),
        raw_event(ts_event=ns_at(250), ts_recv=ns_at(250) + 1, sequence=11,
                  action="T", side="B", size=2),
        raw_event(ts_event=t, ts_recv=t + 1, sequence=12, bid_sz=7),
        raw_event(ts_event=t, ts_recv=t + 2, sequence=13, bid_sz=8),  # trigger
        raw_event(ts_event=t, ts_recv=t + 3, sequence=14, bid_sz=9),  # AFTER
        raw_event(ts_event=ns_at(500), ts_recv=ns_at(500) + 1, sequence=15),
    ]

    def _materialize_with(after_bid_sz: int, after_ask_sz: int):
        rows = [dict(row) for row in burst]
        rows[4] = raw_event(
            ts_event=t, ts_recv=t + 3, sequence=14,
            bid_sz=after_bid_sz, ask_sz=after_ask_sz,
        )
        events = {FIXTURE_DAY: normalized_day(rows)}
        source, _b, _e = build_fixture_source(events)
        day_events = events[FIXTURE_DAY]

        def _exact_cutoffs(row):
            cutoffs = stage_cutoffs_from_candidate_row(row)
            trigger_row = day_events[day_events["sequence"] == 13].iloc[0]
            cutoffs["entry"] = cutoff_from_exact_event("entry", trigger_row)
            return cutoffs

        return materialize_mbp1_features(
            source,
            pd.DataFrame([anchor_row("cand_exact")]),
            resolved_block=MBP1_RESOLVED_BLOCK,
            events_by_day=events,
            cutoff_builder=_exact_cutoffs,
            cutoff_policy_id="exact_key_entry_fixture_v1",
        )

    _env_one, features_one, evidence_one = _materialize_with(9, 5)
    _env_two, features_two, evidence_two = _materialize_with(999, 999)
    pd.testing.assert_frame_equal(features_one, features_two)
    pd.testing.assert_frame_equal(evidence_one, evidence_two)
    # the exact-key entry snapshot USED the trigger event (bid_sz 8), not
    # the mutated after-stage neighbor — the admission was exact, not vacuous
    row = _row(features_one, "cand_exact")
    assert row["ofl_snap_entry_valid"]
    assert row["ofl_snap_entry_bid_sz"] == pytest.approx(8.0)


def test_window_definition_changes_mint_a_new_resolved_block_id() -> None:
    """TEST_MATRIX §3.10 (review F4): comparator/bounds/cutoff/min-count/
    missingness enter the resolved block identity — changing any window
    field changes `resolved_feature_block_id`."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
        FeatureBlockResolutionEnvelope,
        mbp1_activation_resolution_payload,
    )

    base_payload = mbp1_activation_resolution_payload()
    base_id = FeatureBlockResolutionEnvelope.from_payload(
        base_payload
    ).resolved_feature_block_id
    windows = list(base_payload.mbp1_feature_windows)
    for update in (
        {"minimum_event_count": 2},
        {"trigger_semantics": "pre_trigger_exclusive"},
        {"lower_bound": "closed"},
        {"missingness_policy_id": "some_other_policy_v1"},
    ):
        changed = [windows[0].model_copy(update=update), *windows[1:]]
        changed_id = FeatureBlockResolutionEnvelope.from_payload(
            base_payload.model_copy(update={"mbp1_feature_windows": tuple(changed)})
        ).resolved_feature_block_id
        assert changed_id != base_id, update


def test_anchor_setup_relabel_is_a_new_materialization_identity() -> None:
    """Review F8: `setup_id` is emitted into the feature table, so it rides
    the anchor hash — a relabeled setup is a different artifact."""

    source, _bytes, events = build_fixture_source()
    anchors = default_anchors()
    relabeled = anchors.copy()
    relabeled.loc[0, "setup_id"] = "setup_relabeled"
    envelope_one, _f1, _e1 = materialize_mbp1_features(
        source, anchors, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events
    )
    envelope_two, _f2, _e2 = materialize_mbp1_features(
        source, relabeled, resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events
    )
    assert (
        envelope_one.mbp1_feature_artifact_id != envelope_two.mbp1_feature_artifact_id
    )


def test_feature_frame_binding_is_verified_not_asserted() -> None:
    """Review F1 (seam half): `verify_mbp1_feature_frame` refuses a frame
    that does not hash to the envelope's table hash."""

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_feature_materializer import (
        verify_mbp1_feature_frame,
    )

    source, _bytes, events = build_fixture_source()
    envelope, features, _evidence = materialize_mbp1_features(
        source, default_anchors(), resolved_block=MBP1_RESOLVED_BLOCK, events_by_day=events
    )
    verify_mbp1_feature_frame(envelope, features)  # the true frame passes
    tampered = features.copy()
    tampered.loc[0, "ofl_snap_entry_bid_sz"] = 12345.0
    with pytest.raises(ValueError, match="verified, never asserted"):
        verify_mbp1_feature_frame(envelope, tampered)
