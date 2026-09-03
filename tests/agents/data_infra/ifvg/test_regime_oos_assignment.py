"""R6.1 workstream C — the descriptive OOS-assignment artifact and the
normative panel→candidate PIT rule (plan §6.C; §9.1
``test_next_day_warmup_never_inherits_previous_day_regime``; §9.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import AvailabilityStage
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.fold_schedules import derive_fold_schedule
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import ObservationGranularity
from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
    OOS_ASSIGNMENT_COLUMNS,
    OOS_ASSIGNMENT_SCHEMA_HASH,
    PanelAssignmentContext,
    RegimeOosAssignmentPayload,
    assign_panel_regimes_to_candidates,
    build_regime_oos_assignment_artifact,
    candidate_as_of_frame,
    candidate_fold_oos_assignment,
    consulted_assignments_hash,
    load_regime_oos_assignment,
    load_regime_oos_assignment_frame,
    save_regime_oos_assignment,
    verify_regime_oos_assignment_frame,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import persist_regime_fit
from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id
# R6.1 (§6.A grain/bundle-key coherence): a panel-grain protocol must reference
# the panel bundle (row_id join) and panel features — never a candidate bundle
_BP0 = resolve_bundle("BP0_CONTEXT_BAR_PANEL").resolved_feature_bundle_id
_PANEL_INPUTS = ("cbp_realized_range_12", "cbp_realized_volatility_12")
_PANEL_ID = "b" * 64


def _panel_protocol(interval: int = 300):
    return resolve_kmeans_protocol(
        input_feature_bundle_ref=_BP0,
        resolved_input_features=_PANEL_INPUTS,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=interval,
        panel_source_artifact_id=_PANEL_ID,
        panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
    )


def _panel(day: str, *, interval: int = 300, bars: int = 16, start="14:00:00Z", **overrides):
    base = pd.Timestamp(f"{day}T{start}")
    rows = []
    for index in range(bars):
        close = base + pd.Timedelta(seconds=interval * (index + 1))
        rows.append(
            {
                "row_id": f"{day}:{index}",
                "trading_day": day,
                "bar_close_ts_utc": close.isoformat(),
                "cbp_valid": index >= 12,
                "cbp_missing_reason": (
                    None if index >= 12 else "insufficient_trading_day_lookback"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    for row_id, values in overrides.items():
        for column, value in values.items():
            frame.loc[frame["row_id"] == row_id, column] = value
    return frame


def _assignments(frame: pd.DataFrame, *, fold_index: int = 0, partition: str = "test"):
    valid = frame[frame["cbp_valid"].astype(bool)]
    return pd.DataFrame(
        {
            "row_id": valid["row_id"].astype(str),
            "fold_index": fold_index,
            "partition": partition,
            "regime_fit_id": str(fold_index) * 64,
            "fold_local_cluster_id": [index % 3 for index in range(len(valid))],
            "canonical_reporting_cluster_id": [index % 3 for index in range(len(valid))],
            "distances": [[0.1, 0.5, 0.9]] * len(valid),
            "assigned_distance": 0.1,
            "assignment_margin": 0.4,
            "valid": True,
            "missing_reason": None,
        }
    )


def _assign(panel, assignments, candidates, **kwargs):
    return assign_panel_regimes_to_candidates(
        panel,
        assignments,
        pd.DataFrame(candidates, columns=["candidate_id", "as_of_ts_utc"]),
        protocol=_panel_protocol(),
        **kwargs,
    ).set_index("candidate_id")


def test_next_day_warmup_never_inherits_previous_day_regime():
    """§9.1: a candidate inside the next trading day's first 12 bars is
    ``panel_warmup``; one before any completed bar of its day is
    ``no_completed_panel_bar``; a same-day gap is ``panel_gap``; elapsed >
    one interval is ``panel_stale``; no row ever carries a previous-day
    regime across 18:00 ET."""

    day_a, day_b = "2026-01-13", "2026-01-14"
    # day A's bars close 14:05 … 15:20 UTC; day B (next trading day begins
    # 18:00 ET = 23:00 UTC on 01-13) has its own bars starting 23:05 UTC
    panel = pd.concat(
        [
            _panel(day_a),
            _panel(day_b, start="23:00:00Z").assign(
                bar_close_ts_utc=lambda f: [
                    (pd.Timestamp(f"{day_a}T23:00:00Z") + pd.Timedelta(seconds=300 * (i + 1)))
                    .isoformat()
                    for i in range(len(f))
                ]
            ),
        ],
        ignore_index=True,
    )
    assignments = pd.concat(
        [_assignments(_panel(day_a)), _assignments(panel[panel["trading_day"] == day_b])],
        ignore_index=True,
    )
    out = _assign(
        panel,
        assignments,
        [
            ("next_day_warmup", f"{day_a}T23:40:00Z"),  # day B bar 7 → warmup
            ("next_day_before_first", f"{day_a}T23:02:00Z"),  # day B, no completed bar yet
            ("same_day_valid", f"{day_a}T15:12:00Z"),  # day A bar 13 closed 15:10
            ("same_day_stale", f"{day_a}T15:31:00Z"),  # last bar closed 15:20 (> 300 s)
            ("closed_window", f"{day_a}T22:30:00Z"),  # 17:30 ET: no trading day
        ],
    )
    assert out.loc["next_day_warmup", "missing_reason"] == "panel_warmup"
    assert out.loc["next_day_warmup", "panel_row_id"] == f"{day_b}:7"
    assert out.loc["next_day_before_first", "missing_reason"] == "no_completed_panel_bar"
    assert out.loc["same_day_valid", "valid"]
    assert out.loc["same_day_valid", "panel_row_id"] == f"{day_a}:13"
    assert out.loc["same_day_stale", "missing_reason"] == "panel_stale"
    assert out.loc["closed_window", "missing_reason"] == "no_completed_panel_bar"
    # a regime is never carried across 18:00 ET: nothing of day B resolves to a day-A bar
    day_b_rows = out.loc[["next_day_warmup", "next_day_before_first"]]
    assert not day_b_rows["panel_row_id"].astype(str).str.startswith(day_a).any()
    # a same-day gap types panel_gap; an incomplete source bar its own reason
    gapped = _panel(
        day_a,
        **{f"{day_a}:13": {"cbp_valid": False, "cbp_missing_reason": "lookback_window_gap"}},
        **{f"{day_a}:14": {"cbp_valid": False, "cbp_missing_reason": "source_bar_incomplete"}},
    )
    out = _assign(
        gapped,
        _assignments(gapped),
        [("gap", f"{day_a}T15:11:00Z"), ("incomplete", f"{day_a}T15:16:00Z")],
    )
    assert out.loc["gap", "missing_reason"] == "panel_gap"
    assert out.loc["incomplete", "missing_reason"] == "panel_source_bar_incomplete"


def test_pit_rule_matrix_and_refusals():
    day = "2026-01-13"
    panel = _panel(day)
    assignments = _assignments(panel)
    out = _assign(
        panel,
        assignments,
        [
            ("at_close", f"{day}T15:05:00Z"),  # exactly bar 12's close
            ("mid_bar", f"{day}T15:07:30Z"),  # bar 13 in progress
            ("at_staleness", f"{day}T15:25:00Z"),  # bar 15 closed 15:20; +300 s
            ("past_staleness", f"{day}T15:25:01Z"),
            ("before_first", f"{day}T14:04:59Z"),
        ],
    )
    assert out.loc["at_close", "valid"] and out.loc["at_close", "panel_row_id"] == f"{day}:12"
    assert out.loc["mid_bar", "panel_row_id"] == f"{day}:12"
    assert out.loc["at_staleness", "valid"]
    assert out.loc["at_staleness", "elapsed_seconds_since_bar_close"] == 300.0
    assert out.loc["past_staleness", "missing_reason"] == "panel_stale"
    assert out.loc["before_first", "missing_reason"] == "no_completed_panel_bar"
    # ties on bar close → lexicographically last row_id
    tied = panel.copy()
    tied.loc[tied["row_id"] == f"{day}:13", "bar_close_ts_utc"] = tied.loc[
        tied["row_id"] == f"{day}:12", "bar_close_ts_utc"
    ].iloc[0]
    out = _assign(tied, _assignments(tied), [("tie", f"{day}T15:05:00Z")])
    assert out.loc["tie", "panel_row_id"] == f"{day}:13"
    # lowest OOS fold wins in descriptive mode; fold-feature mode consults
    # the candidate's OWN partition of the named fold only
    both = pd.concat(
        [
            _assignments(panel, fold_index=1),
            _assignments(panel, fold_index=0),
            _assignments(panel, fold_index=2, partition="train"),
        ],
        ignore_index=True,
    )
    out = _assign(panel, both, [("c", f"{day}T15:05:00Z")])
    assert int(out.loc["c", "fold_index"]) == 0 and out.loc["c", "partition"] == "test"
    own = _assign(
        panel,
        both,
        [("c", f"{day}T15:05:00Z"), ("d", f"{day}T15:05:00Z")],
        partition_for_candidate={"c": (2, "train"), "d": (5, "test")},
    )
    assert int(own.loc["c", "fold_index"]) == 2 and own.loc["c", "partition"] == "train"
    assert own.loc["d", "missing_reason"] == "coverage_gap"
    # refusals: duplicate candidates, unparseable as-of, missing columns
    with pytest.raises(ValueError, match="repeats a candidate_id"):
        _assign(panel, assignments, [("x", f"{day}T15:05:00Z"), ("x", f"{day}T15:05:00Z")])
    with pytest.raises(ValueError, match="parse"):
        _assign(panel, assignments, [("x", "not-a-time")])
    with pytest.raises(ValueError, match="required columns"):
        _assign(panel.drop(columns=["trading_day"]), assignments, [("x", f"{day}T15:05:00Z")])
    with pytest.raises(ValueError, match="CONTEXT_BAR_PANEL grain"):
        assign_panel_regimes_to_candidates(
            panel,
            assignments,
            pd.DataFrame({"candidate_id": ["x"], "as_of_ts_utc": [f"{day}T15:05:00Z"]}),
            protocol=resolve_kmeans_protocol(
                input_feature_bundle_ref=_B0, resolved_input_features=REGIME_INPUT_FEATURES
            ),
        )
    assert list(out.reset_index().columns) == list(OOS_ASSIGNMENT_COLUMNS)


def test_candidate_as_of_frame_maps_stages_to_anchors():
    frame = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "entry_ts_utc": ["2026-01-13T15:00:00Z", "2026-01-13T15:05:00Z"],
            "tap_ts_utc": ["2026-01-13T14:00:00Z", None],
        }
    )
    entry = candidate_as_of_frame(frame, stage=AvailabilityStage.ENTRY_DECISION)
    assert list(entry["as_of_ts_utc"]) == list(frame["entry_ts_utc"])
    tap = candidate_as_of_frame(frame, stage=AvailabilityStage.HTF_TAP)
    assert tap["as_of_ts_utc"].iloc[1] is None
    with pytest.raises(ValueError, match="anchor column"):
        candidate_as_of_frame(frame, stage=AvailabilityStage.INVERSION)
    with pytest.raises(ValueError, match="repeats"):
        candidate_as_of_frame(pd.concat([frame, frame]), stage=AvailabilityStage.ENTRY_DECISION)


@pytest.fixture(scope="module")
def candidate_run():
    fixture = known_cluster_fixture(k=3, n=600)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0, resolved_input_features=REGIME_INPUT_FEATURES
    )
    run = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    return fixture, folds, protocol, run


def test_candidate_grain_artifact_one_row_per_candidate_and_store_discipline(
    tmp_path, candidate_run
):
    fixture, folds, protocol, run = candidate_run
    ids = tuple(fixture.view.frame["candidate_id"].astype(str))
    # R6.1-FIX §3.1: the descriptive rows derive from the fits' VERIFIED
    # stored sidecars — persist first, then exact-load
    root = tmp_path / "store"
    for fold_fit in run.fold_fits:
        persist_regime_fit(
            root,
            fold_fit,
            run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
            observation_frame=fixture.view.frame,
        )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
        consulted_assignment_frame,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import load_regime_fit_assignments

    verified = {
        fit.fold_index: load_regime_fit_assignments(root, fit.fit_envelope.regime_fit_id)
        for fit in run.fold_fits
    }
    consulted = consulted_assignment_frame(verified)
    frame = candidate_fold_oos_assignment(consulted, ids)
    assert list(frame["candidate_id"]) == list(ids)
    scored = set(
        run.assignments[(run.assignments["partition"] == "test") & run.assignments["valid"]][
            "row_id"
        ].astype(str)
    )
    assert set(frame[frame["valid"]]["candidate_id"]) == scored
    assert set(frame[~frame["valid"]]["missing_reason"]) == {"no_oos_assignment"}
    assert frame["candidate_id"].is_unique
    schedule = derive_fold_schedule(fixture.trading_days)
    as_of = candidate_as_of_frame(fixture.view.frame, stage=AvailabilityStage.ENTRY_DECISION)
    envelope, table = build_regime_oos_assignment_artifact(
        frame,
        protocol=protocol,
        verified_fit_assignments=verified,
        regime_fold_set_id=run.fold_set_id,
        fold_schedule_id=schedule.fold_schedule_id,
        candidate_as_of=as_of,
        candidate_as_of_source_ref=f"bundle_feature_view:{fixture.view.view_id}",
        candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
    )
    payload = envelope.payload
    assert payload.assignment_source == "candidate_fold_oos" and payload.panel_context is None
    assert payload.regime_fit_ids == tuple(sorted(payload.regime_fit_ids))
    assert payload.candidate_count == len(ids)
    # the consulted-assignments hash is recomputable from the persisted fits
    reloaded = pd.concat(
        [
            load_regime_fit_assignments(root, fit.fit_envelope.regime_fit_id).frame
            for fit in run.fold_fits
        ],
        ignore_index=True,
    )
    assert consulted_assignments_hash(reloaded) == payload.consulted_assignments_hash
    save_regime_oos_assignment(root, envelope, table)
    save_regime_oos_assignment(root, envelope, table)  # verified reuse
    stored = load_regime_oos_assignment(root, envelope.regime_oos_assignment_id)
    stored_frame = load_regime_oos_assignment_frame(root, stored)
    assert len(stored_frame) == len(frame)
    verify_regime_oos_assignment_frame(stored, stored_frame)
    with pytest.raises(ValueError, match="does not hash"):
        verify_regime_oos_assignment_frame(stored, frame.iloc[:-1])
    with pytest.raises(ValueError, match="do not hash"):
        save_regime_oos_assignment(tmp_path / "x", envelope, table + b"x")
    # identity binding: a different as-of source → a different id (the
    # candidate sets must stay exactly equal — HARDENING-BACKEND-FIX §6.5)
    other, _ = build_regime_oos_assignment_artifact(
        frame,
        protocol=protocol,
        verified_fit_assignments=verified,
        regime_fold_set_id=run.fold_set_id,
        fold_schedule_id=schedule.fold_schedule_id,
        candidate_as_of=as_of,
        candidate_as_of_source_ref="bundle_feature_view:" + "0" * 64,
        candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
    )
    assert other.regime_oos_assignment_id != envelope.regime_oos_assignment_id
    # tampering the stored table refuses on load
    from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
        OOS_ASSIGNMENT_SIDECAR,
        REGIME_OOS_ASSIGNMENT_STORE,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import envelope_destination

    sidecar = envelope_destination(
        root, REGIME_OOS_ASSIGNMENT_STORE, envelope.regime_oos_assignment_id
    ) / OOS_ASSIGNMENT_SIDECAR
    sidecar.write_bytes(sidecar.read_bytes() + b"\x00")
    with pytest.raises(SearchStoreError):
        load_regime_oos_assignment_frame(root, stored)


def _payload_fields(**overrides) -> dict:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import FitAssignmentRef

    fields = dict(
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
        candidate_as_of_source_hash="e" * 64,
        candidate_as_of_source_ref="bundle_feature_view:" + "0" * 64,
        consulted_assignments_hash="f" * 64,
        candidate_count=0,
        resolved_cluster_count=3,
        assignment_schema_hash=OOS_ASSIGNMENT_SCHEMA_HASH,
        candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
    )
    fields.update(overrides)
    return fields


def test_panel_context_must_agree_with_the_protocol(candidate_run):
    fixture, folds, protocol, run = candidate_run
    ids = tuple(fixture.view.frame["candidate_id"].astype(str))[:5]
    frame = candidate_fold_oos_assignment(run.assignments, ids)
    as_of = candidate_as_of_frame(
        fixture.view.frame.head(5), stage=AvailabilityStage.ENTRY_DECISION
    )
    panel_protocol = _panel_protocol(300)
    context = PanelAssignmentContext(
        context_bar_panel_artifact_id="c" * 64,  # not the protocol's panel
        panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
        candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
        panel_interval_seconds=300,
        max_staleness_seconds=300,
    )
    with pytest.raises(ValueError, match="different context_bar_panel_artifact_id"):
        build_regime_oos_assignment_artifact(
            frame,
            protocol=panel_protocol,
            verified_fit_assignments={},
            regime_fold_set_id="d" * 64,
            fold_schedule_id="e" * 64,
            candidate_as_of=as_of,
            candidate_as_of_source_ref="bundle_feature_view:" + "0" * 64,
            candidate_as_of_stage=context.candidate_as_of_stage,
            panel_context=context,
        )
    with pytest.raises(ValueError, match="requires the panel assignment context"):
        build_regime_oos_assignment_artifact(
            frame,
            protocol=panel_protocol,
            verified_fit_assignments={},
            regime_fold_set_id="d" * 64,
            fold_schedule_id="e" * 64,
            candidate_as_of=as_of,
            candidate_as_of_source_ref="bundle_feature_view:" + "0" * 64,
            candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
        )
    with pytest.raises(ValueError, match="panel grain"):
        RegimeOosAssignmentPayload(**_payload_fields(assignment_source="panel_pit"))


def test_candidate_as_of_source_ref_is_a_loaded_artifact_line_never_a_caller_string():
    """Adversarial R6.1 F1: the provenance line of the as-of instants is
    ``<source_kind>:<64-hex>`` of a verified-loaded artifact; a free caller
    string is refused by the payload itself."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
        CANDIDATE_AS_OF_SOURCE_REF_PATTERN,
        RegimeOosAssignmentPayload,
    )

    def _payload(ref: str) -> RegimeOosAssignmentPayload:
        return RegimeOosAssignmentPayload(**_payload_fields(candidate_as_of_source_ref=ref))

    assert _payload("bundle_feature_view:" + "0" * 64).candidate_as_of_source_ref.startswith(
        "bundle_feature_view:"
    )
    for bad in ("NOT_AN_ARTIFACT:this-is-a-caller-string", "synthetic_candidates", "x", "0" * 64):
        with pytest.raises(ValueError, match="pattern|string_pattern_mismatch"):
            _payload(bad)
    assert "64" not in CANDIDATE_AS_OF_SOURCE_REF_PATTERN.replace("{64}", "")


# ── HARDENING-BACKEND-FIX §6 — exact OOS provenance, set equality, projection ─


def _fit_row(row_id, *, fold_index, partition, valid, reason=None, local=0):
    fit_id = str(fold_index) * 64
    if valid:
        return {
            "row_id": row_id,
            "fold_index": fold_index,
            "partition": partition,
            "regime_fit_id": fit_id,
            "fold_local_cluster_id": local,
            "canonical_reporting_cluster_id": local,
            "distances": [0.1, 0.5, 0.9] if local == 0 else [0.5, 0.1, 0.9],
            "assigned_distance": 0.1,
            "assignment_margin": 0.4,
            "valid": True,
            "missing_reason": None,
        }
    return {
        "row_id": row_id,
        "fold_index": fold_index,
        "partition": partition,
        "regime_fit_id": fit_id,
        "fold_local_cluster_id": None,
        "canonical_reporting_cluster_id": None,
        "distances": None,
        "assigned_distance": float("nan"),
        "assignment_margin": float("nan"),
        "valid": False,
        "missing_reason": reason,
    }


def test_invalid_oos_rows_retain_exact_provenance_and_no_coverage_is_bare():
    """HB-FIX-04 (candidate grain): A = valid OOS row (fit / fold / partition
    and outputs), B = INVALID OOS row of an applicable fit (fit / fold /
    partition and the fit's own typed reason; outputs null), C / D = no OOS
    test row at all (``no_oos_assignment``; no fit, no fold)."""

    from alpha_lab.agents.data_infra.ifvg.ml import regime_oos_assignment as oos_module
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        RegimeAssignmentEvidenceError,
        validate_assignment_rows,
    )

    assignments = pd.DataFrame(
        [
            _fit_row("A", fold_index=1, partition="test", valid=True),
            _fit_row("A", fold_index=0, partition="train", valid=True),
            _fit_row(
                "B", fold_index=0, partition="test", valid=False, reason="source_feature_missing"
            ),
            _fit_row(
                "B", fold_index=1, partition="train", valid=False, reason="source_feature_missing"
            ),
            _fit_row("C", fold_index=0, partition="train", valid=True),
        ]
    )
    frame = candidate_fold_oos_assignment(assignments, ("A", "B", "C", "D"))
    assert list(frame["candidate_id"]) == ["A", "B", "C", "D"]
    indexed = frame.set_index("candidate_id")
    a = indexed.loc["A"]
    assert bool(a["valid"]) and a["regime_fit_id"] == "1" * 64 and int(a["fold_index"]) == 1
    assert a["partition"] == "test" and a["missing_reason"] is None
    b = indexed.loc["B"]
    assert not bool(b["valid"])
    assert b["regime_fit_id"] == "0" * 64 and int(b["fold_index"]) == 0 and b["partition"] == "test"
    assert b["missing_reason"] == "source_feature_missing"
    assert b["distances"] is None and pd.isna(b["fold_local_cluster_id"])
    assert pd.isna(b["assigned_distance"]) and pd.isna(b["assignment_margin"])
    for candidate_id in ("C", "D"):
        row = indexed.loc[candidate_id]
        assert not bool(row["valid"]) and row["missing_reason"] == "no_oos_assignment"
        assert row["regime_fit_id"] is None and pd.isna(row["fold_index"])
        assert row["partition"] is None
    # the descriptive invariants accept the three shapes …
    canonical = oos_module._frame_for_schema(frame)
    validate_assignment_rows(canonical, cluster_count=3, kind="descriptive")
    # … refuse PARTIAL provenance …
    partial = canonical.copy()
    partial.loc[partial["candidate_id"] == "B", "fold_index"] = pd.NA
    with pytest.raises(RegimeAssignmentEvidenceError) as incomplete:
        validate_assignment_rows(partial, cluster_count=3, kind="descriptive")
    assert incomplete.value.reason == "assignment_provenance_incomplete"
    # … and a fabricated fit / fold under no_oos_assignment
    fabricated = canonical.copy()
    mask = fabricated["candidate_id"] == "D"
    fabricated.loc[mask, "regime_fit_id"] = "2" * 64
    fabricated.loc[mask, "fold_index"] = 2
    fabricated.loc[mask, "partition"] = "test"
    with pytest.raises(RegimeAssignmentEvidenceError) as fake:
        validate_assignment_rows(fabricated, cluster_count=3, kind="descriptive")
    assert fake.value.reason == "assignment_provenance_incomplete"
    # two test rows for one candidate are refused whatever their validity
    twice = pd.concat(
        [assignments, pd.DataFrame([_fit_row("B", fold_index=2, partition="test", valid=True)])],
        ignore_index=True,
    )
    with pytest.raises(RegimeAssignmentEvidenceError) as duplicated:
        candidate_fold_oos_assignment(twice, ("A", "B"))
    assert duplicated.value.reason == "duplicate_candidate_id"
    # a fold that had NO fit (the kernel writes an empty fit id) keeps fold /
    # partition and the typed reason without inventing a fit
    fit_less = pd.DataFrame(
        [{**_fit_row("E", fold_index=3, partition="test", valid=False, reason="fold_invalid"),
          "regime_fit_id": ""}]
    )
    row = candidate_fold_oos_assignment(fit_less, ("E",)).set_index("candidate_id").loc["E"]
    assert row["missing_reason"] == "fold_invalid" and row["regime_fit_id"] is None
    assert int(row["fold_index"]) == 3 and row["partition"] == "test"
    validate_assignment_rows(
        oos_module._frame_for_schema(candidate_fold_oos_assignment(fit_less, ("E",))),
        cluster_count=3,
        kind="descriptive",
    )


def test_panel_descriptive_mode_retains_invalid_fit_provenance():
    """HB-FIX-04 (panel grain): a bar whose ONLY test rows are invalid keeps
    the lowest-fold fit's id / fold / partition and typed reason; a valid
    row still wins; the fold-feature mode keeps the exact fit reason for the
    candidate's OWN row; a bar with no test row stays a bare coverage gap."""

    day = "2026-01-13"
    panel = _panel(day)
    valid_rows = _assignments(panel, fold_index=1)
    invalid = pd.DataFrame(
        [
            _fit_row(f"{day}:12", fold_index=0, partition="test", valid=False,
                     reason="below_confidence_floor"),
            _fit_row(f"{day}:13", fold_index=0, partition="test", valid=False,
                     reason="below_confidence_floor"),
        ]
    )
    assignments = pd.concat(
        [valid_rows[valid_rows["row_id"] != f"{day}:12"], invalid], ignore_index=True
    )
    base = pd.Timestamp(f"{day}T14:00:00Z")
    at_12 = (base + pd.Timedelta(seconds=300 * 13)).isoformat()
    at_13 = (base + pd.Timedelta(seconds=300 * 14)).isoformat()
    assigned = _assign(panel, assignments, [("c12", at_12), ("c13", at_13)])
    c12 = assigned.loc["c12"]
    assert not bool(c12["valid"]) and c12["missing_reason"] == "below_confidence_floor"
    assert c12["regime_fit_id"] == "0" * 64 and int(c12["fold_index"]) == 0
    assert c12["partition"] == "test" and c12["panel_row_id"] == f"{day}:12"
    assert c12["distances"] is None and pd.isna(c12["assigned_distance"])
    c13 = assigned.loc["c13"]
    assert bool(c13["valid"]) and int(c13["fold_index"]) == 1
    own = _assign(
        panel, assignments, [("c13", at_13)], partition_for_candidate={"c13": (0, "test")}
    )
    assert own.loc["c13"]["missing_reason"] == "below_confidence_floor"
    assert own.loc["c13"]["regime_fit_id"] == "0" * 64 and int(own.loc["c13"]["fold_index"]) == 0
    gap = _assign(panel, assignments[assignments["row_id"] != f"{day}:12"], [("c12", at_12)])
    assert gap.loc["c12"]["missing_reason"] == "coverage_gap"
    assert gap.loc["c12"]["regime_fit_id"] is None and pd.isna(gap.loc["c12"]["fold_index"])


def test_candidate_as_of_and_assignment_sets_must_be_exactly_equal(candidate_run, tmp_path):
    """HB-FIX-07: same ids, same count, no duplicates, no extras, no omissions."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import RegimeAssignmentEvidenceError
    from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
        consulted_assignment_frame,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import load_regime_fit_assignments

    fixture, folds, protocol, run = candidate_run
    root = tmp_path / "store"
    for fold_fit in run.fold_fits:
        persist_regime_fit(
            root,
            fold_fit,
            run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
            observation_frame=fixture.view.frame,
        )
    verified = {
        fit.fold_index: load_regime_fit_assignments(root, fit.fit_envelope.regime_fit_id)
        for fit in run.fold_fits
    }
    ids = tuple(fixture.view.frame["candidate_id"].astype(str))
    frame = candidate_fold_oos_assignment(consulted_assignment_frame(verified), ids)
    schedule = derive_fold_schedule(fixture.trading_days)
    as_of = candidate_as_of_frame(fixture.view.frame, stage=AvailabilityStage.ENTRY_DECISION)

    def _build(frame_, as_of_):
        return build_regime_oos_assignment_artifact(
            frame_,
            protocol=protocol,
            verified_fit_assignments=verified,
            regime_fold_set_id=run.fold_set_id,
            fold_schedule_id=schedule.fold_schedule_id,
            candidate_as_of=as_of_,
            candidate_as_of_source_ref=f"bundle_feature_view:{fixture.view.view_id}",
            candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
        )

    _build(frame, as_of)
    extra = pd.concat([as_of, as_of.iloc[:1].assign(candidate_id="extra")], ignore_index=True)
    for bad_as_of in (as_of.iloc[:-1], extra):
        with pytest.raises(RegimeAssignmentEvidenceError) as mismatch:
            _build(frame, bad_as_of)
        assert mismatch.value.reason == "candidate_set_mismatch"
    with pytest.raises(RegimeAssignmentEvidenceError) as omitted:
        _build(frame.iloc[:-1], as_of)
    assert omitted.value.reason == "candidate_set_mismatch"
    with pytest.raises(RegimeAssignmentEvidenceError) as duplicate_output:
        _build(pd.concat([frame, frame.iloc[:1]], ignore_index=True), as_of)
    assert duplicate_output.value.reason == "duplicate_candidate_id"
    with pytest.raises(RegimeAssignmentEvidenceError) as duplicate_input:
        _build(frame, pd.concat([as_of, as_of.iloc[:1]], ignore_index=True))
    assert duplicate_input.value.reason == "duplicate_candidate_id"


def test_trade_projection_retains_invalid_assignment_provenance():
    """HB-FIX-04: the executed-trade projection keeps the known fit / fold of
    an invalid assignment; only the trade-facing outputs are null."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_assignment_sources import regime_for_trades

    assignments = pd.DataFrame(
        [
            _fit_row("A", fold_index=1, partition="test", valid=True),
            _fit_row("B", fold_index=0, partition="test", valid=False,
                     reason="source_feature_missing"),
        ]
    )
    descriptive = candidate_fold_oos_assignment(assignments, ("A", "B", "C"))
    trades = pd.DataFrame(
        {"trade_id": ["t1", "t2", "t3", "t4"], "candidate_id": ["A", "B", "C", "Z"]}
    )
    projected = regime_for_trades(trades, descriptive).set_index("trade_id")
    t1 = projected.loc["t1"]
    assert bool(t1["valid"]) and t1["regime_fit_id"] == "1" * 64 and int(t1["fold_index"]) == 1
    t2 = projected.loc["t2"]
    assert not bool(t2["valid"])
    assert t2["regime_fit_id"] == "0" * 64 and int(t2["fold_index"]) == 0
    assert t2["missing_reason"] == "source_feature_missing"
    assert pd.isna(t2["canonical_reporting_cluster_id"]) and pd.isna(t2["assignment_margin"])
    t3 = projected.loc["t3"]
    assert t3["missing_reason"] == "no_oos_assignment"
    assert t3["regime_fit_id"] is None and pd.isna(t3["fold_index"])
    t4 = projected.loc["t4"]
    assert t4["missing_reason"] == "candidate_not_in_assignment" and t4["regime_fit_id"] is None
    # review RB-05: the projection validates its INPUT natively before any bool()
    # coercion — a "False" string never projects as a valid assignment
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        RegimeAssignmentEvidenceError,
    )

    laundered = descriptive.copy()
    laundered["valid"] = laundered["valid"].astype(object)
    laundered.loc[laundered["candidate_id"] == "B", "valid"] = "False"
    with pytest.raises(RegimeAssignmentEvidenceError) as refused:
        regime_for_trades(trades, laundered)
    assert refused.value.reason == "native_value_refused"
