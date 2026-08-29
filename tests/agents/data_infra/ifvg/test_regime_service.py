"""R6 regime-service suites on fixtures 2 and 4 (ML plan §5/§9; acceptance
7B.22-2/4/8/9/10): fold-local fitting, deterministic OOS KMeans, typed-null
preservation, the sample-adequacy gate, reporting-only alignment, the
point-in-time panel→candidate assignment (Amendment P1-B), and the
adversarial-round closures — content-bound fit identities (F1), the
observation-timestamp temporal facts (F3), OOS-only panel assignment (F4),
the exact Hungarian tie-break (F8), the input-feature alignment space
(F9), all-missing rows outside the fit (F10), unique observation keys
(F11), the executed panel-grain fit path (F14), and the bootstrap /
winsorization / k=2 proofs (F16)."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    IfvgContextFoldDefinition,
)
from alpha_lab.agents.data_infra.ifvg.context_folds import (
    ContextFoldSet,
    build_context_folds,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_alignment import (
    align_reporting_labels,
    feature_space_centroids,
    hungarian_lexicographic,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    ObservationGranularity,
    RegimeProtocolEnvelope,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_preprocessing import (
    fit_regime_preprocessing,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    assign_panel_regimes_to_candidates,
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    CLUSTER_CENTERS,
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_oos_assignment import (
    oos_assignment_fixture,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id


def _folds(fixture):
    return build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )


def _protocol(fixture, **overrides):
    defaults = dict(
        input_feature_bundle_ref=_B0,
        resolved_input_features=fixture.regime_input_features,
    )
    defaults.update(overrides)
    return resolve_kmeans_protocol(**defaults)


def _run(fixture, folds=None, protocol=None, *, frame=None, bootstrap_refits=5):
    folds = folds if folds is not None else _folds(fixture)
    protocol = protocol if protocol is not None else _protocol(fixture)
    return run_regime_protocol(
        fixture.view.frame if frame is None else frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=bootstrap_refits,
    )


@pytest.fixture(scope="module")
def cluster_run():
    fixture = known_cluster_fixture(k=3, n=600)
    folds = _folds(fixture)
    protocol = _protocol(fixture)
    run = _run(fixture, folds, protocol, bootstrap_refits=10)
    return fixture, folds, protocol, run


def _assignments_hash(assignments: pd.DataFrame) -> str:
    return canonical_contract_sha256(
        {
            "rows": sorted(
                (
                    str(row.row_id),
                    int(row.fold_index),
                    str(row.partition),
                    None if row.fold_local_cluster_id is None else int(row.fold_local_cluster_id),
                    bool(row.valid),
                )
                for row in assignments.itertuples()
            )
        }
    )


def test_oos_kmeans_is_deterministic_and_recovers_the_known_clusters(cluster_run):
    fixture, folds, protocol, run = cluster_run
    rerun = _run(fixture, folds, protocol, bootstrap_refits=10)
    assert _assignments_hash(run.assignments) == _assignments_hash(rerun.assignments)
    assert rerun.assessment.regime_capability_assessment_id == (
        run.assessment.regime_capability_assessment_id
    )
    # OOS test rows recover the KNOWN memberships exactly (up to relabeling)
    from sklearn.metrics import adjusted_mutual_info_score

    test_rows = run.assignments[
        (run.assignments["partition"] == "test") & run.assignments["valid"]
    ]
    truth = pd.Series(fixture.true_memberships)
    ami = adjusted_mutual_info_score(
        truth.loc[test_rows["row_id"]].to_numpy(),
        test_rows["canonical_reporting_cluster_id"].astype(int).to_numpy(),
    )
    assert ami > 0.99
    assert run.assessment.payload.gates_passed
    assert run.assessment.payload.oos_assignment_available
    # distances to EVERY centroid, the assigned distance, and margin d2−d1
    valid = run.assignments[run.assignments["valid"]]
    assert (valid["assigned_distance"] >= 0).all()
    assert valid["assignment_margin"].notna().all()
    for row in valid.head(25).itertuples():
        distances = sorted(float(v) for v in row.distances)
        assert len(distances) == 3
        assert row.assigned_distance == pytest.approx(distances[0])
        assert row.assignment_margin == pytest.approx(distances[1] - distances[0])
    assert valid["observation_ts_utc"].notna().all()
    # every fit pins the verified source artifact + the training matrix hash
    for fit in run.fold_fits:
        assert fit.fit_envelope.payload.source_artifact_ids == (fixture.view.view_id,)
        assert fit.fit_envelope.payload.training_feature_matrix_hash == (
            fit.preprocessing.training_feature_matrix_hash
        )


def test_fit_identity_binds_the_training_feature_values(cluster_run):
    """Review F1: identical row ids over different feature values are
    DIFFERENT fits; source artifact ids are required and verified."""

    fixture, folds, protocol, run = cluster_run
    perturbed = fixture.view.frame.copy()
    for feature in fixture.regime_input_features:
        perturbed[feature] = perturbed[feature] * 3.0 + 11.0
    other = _run(fixture, folds, protocol, frame=perturbed, bootstrap_refits=2)
    for one, two in zip(run.fold_fits, other.fold_fits, strict=True):
        assert one.fit_envelope.regime_fit_id != two.fit_envelope.regime_fit_id
        assert one.fit_envelope.payload.training_row_ids_hash == (
            two.fit_envelope.payload.training_row_ids_hash
        )
    with pytest.raises(ValueError, match="source_artifact_id"):
        run_regime_protocol(
            fixture.view.frame, folds, protocol, source_artifact_ids=(), bootstrap_refits=2
        )
    with pytest.raises(ValueError, match="source_artifact_id"):
        run_regime_protocol(
            fixture.view.frame,
            folds,
            protocol,
            source_artifact_ids=("not-verified",),
            bootstrap_refits=2,
        )


def test_fold_locality_changing_a_test_row_never_changes_the_fit(cluster_run):
    fixture, folds, protocol, _run_ = cluster_run
    fold = next(f for f in folds.folds if f.valid)
    fitted = fit_regime_preprocessing(
        fixture.view.frame, fixture.regime_input_features, fold
    )
    poisoned = fixture.view.frame.copy()
    test_id = sorted(fold.test_candidate_ids)[0]
    poisoned.loc[
        poisoned["candidate_id"].astype(str) == str(test_id),
        list(fixture.regime_input_features),
    ] = 1e9
    refitted = fit_regime_preprocessing(
        poisoned, fixture.regime_input_features, fold
    )
    assert (
        refitted.fitted_parameter_payload_hash == fitted.fitted_parameter_payload_hash
    )
    assert refitted.training_feature_matrix_hash == fitted.training_feature_matrix_hash


def test_fit_api_cannot_receive_a_pooled_frame(cluster_run):
    """§5.3: the fit API accepts a FOLD and slices internally — there is no
    row-subset parameter, so pooled/future fitting is unrepresentable at
    this seam (the fold ids themselves are bound into fold_set_id)."""

    import inspect

    parameters = inspect.signature(fit_regime_preprocessing).parameters
    assert set(parameters) == {"frame", "features", "fold", "winsorization_policy"}
    fixture, folds, _protocol_, _run_ = cluster_run
    invalid = next((f for f in folds.folds if not f.valid), None)
    if invalid is not None:
        with pytest.raises(ValueError, match="invalid"):
            fit_regime_preprocessing(
                fixture.view.frame, fixture.regime_input_features, invalid
            )


def test_duplicate_observation_keys_are_refused(cluster_run):
    """Review F11: a repeated candidate id would silently reweight the fit."""

    fixture, folds, protocol, _run_ = cluster_run
    doubled = pd.concat(
        [fixture.view.frame, fixture.view.frame.head(1)], ignore_index=True
    )
    with pytest.raises(ValueError, match="duplicated candidate_id"):
        run_regime_protocol(
            doubled, folds, protocol, source_artifact_ids=(fixture.view.view_id,)
        )
    with pytest.raises(ValueError, match="duplicated candidate_id"):
        fit_regime_preprocessing(
            doubled, fixture.regime_input_features, next(f for f in folds.folds if f.valid)
        )


def test_all_missing_training_rows_never_enter_the_fit():
    """Review F10: an all-missing training row is a typed null, NOT a
    median point inside the fit — identity and coverage agree."""

    fixture4 = oos_assignment_fixture(k=3, n=600)
    base = fixture4.base
    folds = _folds(base)
    run = _run(base, folds, bootstrap_refits=3)
    missing = set(fixture4.missing_candidate_ids)
    for fit in run.fold_fits:
        fold = next(f for f in folds.folds if f.fold_index == fit.fold_index)
        in_train = missing & {str(c) for c in fold.train_candidate_ids}
        assert not (missing & set(fit.preprocessing.training_row_ids))
        assert in_train == set(fit.preprocessing.training_rows_all_missing)
        assert fit.training_row_count == len(fit.preprocessing.training_row_ids)
        assert fit.training_row_count == len(fold.train_candidate_ids) - len(in_train)
    rows = run.assignments[run.assignments["row_id"].isin(missing)]
    assert len(rows) and (~rows["valid"].astype(bool)).all()
    assert set(rows["missing_reason"]) == {"source_feature_missing"}


def test_alignment_is_reporting_only_and_stable_under_fold_permutation(cluster_run):
    _fixture, _folds_, _protocol_, run = cluster_run
    forward = align_reporting_labels(run.fold_fits)
    permuted = align_reporting_labels(tuple(reversed(run.fold_fits)))
    # the reference is fold-index-ordered, so permuting the INPUT order
    # changes nothing (stable canonical ids)
    assert forward.mapping == permuted.mapping
    assert forward.reference_fold_index == permuted.reference_fold_index
    assert forward.alignment_feature_names == REGIME_INPUT_FEATURES
    # prediction-hash invariance (7B.22-9): the fold-local ids are untouched
    # by canonical relabeling
    local_hash = canonical_contract_sha256(
        {
            "rows": sorted(
                (str(r.row_id), int(r.fold_index), int(r.fold_local_cluster_id))
                for r in run.assignments[run.assignments["valid"]].itertuples()
            )
        }
    )
    relabeled = run.assignments.copy()
    relabeled["canonical_reporting_cluster_id"] = (
        relabeled["canonical_reporting_cluster_id"].astype("Int64") + 100
    )
    relabeled_hash = canonical_contract_sha256(
        {
            "rows": sorted(
                (str(r.row_id), int(r.fold_index), int(r.fold_local_cluster_id))
                for r in relabeled[relabeled["valid"]].itertuples()
            )
        }
    )
    assert relabeled_hash == local_hash
    # semantic descriptors exist for every canonical cluster
    assert set(forward.semantic_descriptors) == {0, 1, 2}


def test_hungarian_tie_break_is_ascending_fold_local_id():
    """Review F8: among optimal assignments the lexicographically smallest
    (local 0 first, then local 1, …) wins — exactly, not by float luck."""

    tied = np.array([[0.0, 0.0, 9.0], [0.0, 0.0, 9.0], [9.0, 9.0, 0.0]])
    assert hungarian_lexicographic(tied) == (0, 1, 2)
    assert hungarian_lexicographic(np.ones((2, 2))) == (0, 1)
    # a non-tied matrix is the unique optimum
    unique = np.array([[5.0, 1.0, 9.0], [1.0, 9.0, 5.0], [9.0, 5.0, 1.0]])
    assert hungarian_lexicographic(unique) == (1, 0, 2)
    # ascending-local priority: local 0 takes the smaller canonical when the
    # two optimal assignments differ only by a swap
    swap = np.array([[1.0, 1.0, 9.0], [1.0, 1.0, 9.0], [9.0, 9.0, 0.5]])
    assert hungarian_lexicographic(swap) == (0, 1, 2)
    with pytest.raises(ValueError, match="square"):
        hungarian_lexicographic(np.ones((2, 3)))


def test_alignment_uses_the_scaled_input_feature_space():
    """Review F9: fold-dependent indicator columns never enter an
    alignment — centroids are compared on the input-feature prefix."""

    features = ("a", "b")
    fit0 = SimpleNamespace(
        fold_index=0,
        centroids_scaled=np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0]]),
        preprocessing=SimpleNamespace(features=features),
    )
    fit1 = SimpleNamespace(  # indicator column present only in this fold
        fold_index=1,
        centroids_scaled=np.array([[5.1, 5.0, 3.0], [0.1, 5.0, -3.0], [0.0, 0.1, 9.0]]),
        preprocessing=SimpleNamespace(features=features),
    )
    assert feature_space_centroids(fit1).shape == (3, 2)
    aligned = align_reporting_labels((fit1, fit0))
    assert aligned.reference_fold_index == 0
    # canonical ids are the reference centroids' coordinate ranks — (0,0) → 0,
    # (0,5) → 1, (5,5) → 2 — never the training row order (review F3)
    assert aligned.mapping[0] == {0: 0, 1: 2, 2: 1}
    assert aligned.mapping[1] == {0: 2, 1: 1, 2: 0}
    assert all(len(d) == 2 for d in aligned.semantic_descriptors.values())
    assert aligned.semantic_descriptors[2][0] == ("a", 5.0)
    with pytest.raises(ValueError, match="same input features"):
        align_reporting_labels(
            (
                fit0,
                SimpleNamespace(
                    **{**vars(fit1), "preprocessing": SimpleNamespace(features=("a",))}
                ),
            )
        )


def test_typed_nulls_preserve_rows_on_missing_sources_and_invalid_folds():
    fixture4 = oos_assignment_fixture(k=3, n=600)
    base = fixture4.base
    run = _run(base, bootstrap_refits=5)
    assignments = run.assignments
    in_fold_missing = [
        cid
        for cid in fixture4.missing_candidate_ids
        if cid in set(assignments["row_id"])
    ]
    assert in_fold_missing, "the fixture must place missing rows inside folds"
    for candidate_id in in_fold_missing:
        rows = assignments[assignments["row_id"] == candidate_id]
        assert (~rows["valid"].astype(bool)).all()
        assert set(rows["missing_reason"]) == {"source_feature_missing"}
    typed = dict(run.assessment.payload.coverage.rows_typed_null)
    assert typed.get("source_feature_missing", 0) >= len(in_fold_missing)


def test_invalid_folds_type_every_row_fold_invalid():
    fixture = known_cluster_fixture(k=3, n=80)  # sparse → some invalid folds
    folds = _folds(fixture)
    run = _run(fixture, folds, bootstrap_refits=3)
    invalid_folds = [f.fold_index for f in folds.folds if not f.valid]
    if invalid_folds:
        rows = run.assignments[run.assignments["fold_index"].isin(invalid_folds)]
        assert len(rows)
        assert set(rows["missing_reason"]) == {"fold_invalid"}


def test_sample_adequacy_gate_blocks_under_sampled_fits_without_shrinking_k():
    fixture = known_cluster_fixture(k=3, n=170)  # folds valid (>=30) but < 150
    folds = _folds(fixture)
    assert any(f.valid for f in folds.folds)
    run = _run(fixture, folds, bootstrap_refits=3)
    assessment = run.assessment.payload
    assert "sample_adequacy" in assessment.gate_failures
    assert not assessment.gates_passed
    # k was NEVER shrunk: every fit still carries exactly 3 centroids
    for fit in run.fold_fits:
        assert len(fit.centroids_scaled) == 3
    assert assessment.coverage.minimum_training_observations_gate == 150


def test_temporal_facts_are_id_spelling_invariant_and_oos():
    """Review F3 (+ R6.1 D11): the CANDIDATE-EVENT transition facts come from
    the OOS timeline ordered by the observation timestamp within each fold
    (day/session resets, maximum gap) — renaming every id changes nothing;
    a candidate grain carries NO panel temporal facts."""

    fixture = known_cluster_fixture(k=3, n=600)
    run = _run(fixture, bootstrap_refits=2)
    stability = run.assessment.payload.stability
    assert stability.temporal_order_policy == (
        "oos_test_rows_by_observation_ts_within_fold_day_session_reset_max_gap_v2"
    )
    assert stability.session_scheme_id == "ifvg_doc_session_scheme_et_v1"
    assert stability.candidate_event_pairs_counted > 0
    assert stability.candidate_event_persistence is not None
    assert stability.candidate_event_maximum_gap_seconds == 7200
    # a candidate grain is NOT a regular time series: no panel facts
    assert stability.transition_matrix == ()
    assert stability.temporal_persistence is None
    assert stability.temporal_transition_count == 0

    def _rename(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    frame = fixture.view.frame.copy()
    frame["candidate_id"] = frame["candidate_id"].map(_rename)
    labels = fixture.labeled_candidates.copy()
    labels["candidate_id"] = labels["candidate_id"].map(_rename)
    folds = build_context_folds(labels, authorized_trading_days=fixture.trading_days)
    renamed = run_regime_protocol(
        frame,
        folds,
        run.protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    other = renamed.assessment.payload.stability
    assert other.candidate_event_persistence == stability.candidate_event_persistence
    assert other.candidate_event_transition_matrix == stability.candidate_event_transition_matrix
    assert other.candidate_event_pairs_counted == stability.candidate_event_pairs_counted
    # the OOS test rows are the timeline: the count never exceeds them
    oos = run.assignments[(run.assignments["partition"] == "test") & run.assignments["valid"]]
    assert stability.candidate_event_pairs_counted < len(oos)


def test_bootstrap_and_internal_scores_see_training_rows_only():
    """Review F16: poisoning every row outside fold 0's TRAINING set leaves
    fold 0's fit identity and every bootstrap / separation / silhouette
    fact unchanged."""

    fixture = known_cluster_fixture(k=3, n=600)
    folds = _folds(fixture)
    protocol = _protocol(fixture)
    baseline = _run(fixture, folds, protocol, bootstrap_refits=4)
    fold0 = next(f for f in folds.folds if f.valid)
    train_ids = {str(c) for c in fold0.train_candidate_ids}
    poisoned = fixture.view.frame.copy()
    outside = ~poisoned["candidate_id"].astype(str).isin(train_ids)
    poisoned.loc[outside, list(fixture.regime_input_features)] = 1e9
    attacked = _run(fixture, folds, protocol, frame=poisoned, bootstrap_refits=4)
    assert attacked.fold_fits[0].fit_envelope.regime_fit_id == (
        baseline.fold_fits[0].fit_envelope.regime_fit_id
    )
    base_stability = baseline.assessment.payload.stability
    poisoned_stability = attacked.assessment.payload.stability
    assert poisoned_stability.bootstrap_aligned_ami_mean == (
        base_stability.bootstrap_aligned_ami_mean
    )
    assert poisoned_stability.per_cluster_agreement == base_stability.per_cluster_agreement
    assert poisoned_stability.silhouette_descriptive == base_stability.silhouette_descriptive
    assert poisoned_stability.separation_min_centroid_distance == (
        base_stability.separation_min_centroid_distance
    )


def test_k2_margins_and_winsorization_execute():
    fixture = known_cluster_fixture(k=2, n=400)
    run = _run(fixture, protocol=_protocol(fixture, resolved_cluster_count=2), bootstrap_refits=2)
    valid = run.assignments[run.assignments["valid"]]
    assert valid["assignment_margin"].notna().all()
    assert all(len(d) == 2 for d in valid["distances"].head(10))
    clipped = _run(
        fixture,
        protocol=_protocol(
            fixture, resolved_cluster_count=2, winsorization_policy="clip_p01_p99_train_fitted_v1"
        ),
        bootstrap_refits=2,
    )
    for fit in clipped.fold_fits:
        assert fit.preprocessing.clip_lower is not None
        assert fit.preprocessing.parameter_payload["clip_lower"] is not None
    assert clipped.protocol.resolved_regime_protocol_id != run.protocol.resolved_regime_protocol_id


# ── the CONTEXT_BAR_PANEL grain (Amendment P1-B) ─────────────────────────────


_BP0 = resolve_bundle("BP0_CONTEXT_BAR_PANEL").resolved_feature_bundle_id
#: R6.1 (§6.A): the panel grain references the panel bundle and its features
_PANEL_INPUTS = ("cbp_realized_range_12", "cbp_realized_volatility_12")


def _panel_protocol(interval: int):
    return resolve_kmeans_protocol(
        input_feature_bundle_ref=_BP0,
        resolved_input_features=_PANEL_INPUTS,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=interval,
        panel_source_artifact_id="b" * 64,
        panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
    )


def test_panel_pit_assignment_is_last_completed_bar_never_future_and_oos_only():
    """Amendment P1-B (+ review F4; R6.1 owner correction 2): 5m/15m panels
    assign the frozen OUT-OF-SAMPLE regime of the LAST completed bar at or
    before each candidate's as-of instant; in-sample rows are never
    consulted; the choice is independent of input row order; a candidate
    before the first completed bar is ``no_completed_panel_bar``; a bar
    without an OOS assignment is ``coverage_gap``; a bar older than one
    interval is ``panel_stale``."""

    for interval in (300, 900):
        protocol = _panel_protocol(interval)
        base = pd.Timestamp("2026-01-13T14:00:00Z")
        bar_close = [base + pd.Timedelta(seconds=interval * (i + 1)) for i in range(4)]
        panel_frame = pd.DataFrame(
            {
                "row_id": [f"bar_{interval}_{i}" for i in range(4)],
                "trading_day": "2026-01-13",
                "bar_close_ts_utc": [ts.isoformat() for ts in bar_close],
                "cbp_valid": True,
                "cbp_missing_reason": None,
            }
        )
        rows = [
            # bar 0: OOS fold 0
            dict(row_id=f"bar_{interval}_0", fold_index=0, partition="test", local=0, canonical=0),
            # bar 1: OOS in folds 3 and 0 (listed HIGH fold first on purpose)
            dict(row_id=f"bar_{interval}_1", fold_index=3, partition="test", local=2, canonical=2),
            dict(row_id=f"bar_{interval}_1", fold_index=0, partition="test", local=1, canonical=1),
            # bar 2: OOS fold 1 + an in-sample row of fold 2 that must be ignored
            dict(row_id=f"bar_{interval}_2", fold_index=2, partition="train", local=0, canonical=0),
            dict(row_id=f"bar_{interval}_2", fold_index=1, partition="test", local=2, canonical=2),
            # bar 3: in-sample ONLY → no frozen OOS regime exists
            dict(row_id=f"bar_{interval}_3", fold_index=2, partition="train", local=1, canonical=1),
        ]
        panel_assignments = pd.DataFrame(
            {
                "row_id": [r["row_id"] for r in rows],
                "fold_index": [r["fold_index"] for r in rows],
                "partition": [r["partition"] for r in rows],
                "regime_fit_id": [f"{r['fold_index']}" * 64 for r in rows],
                "fold_local_cluster_id": [r["local"] for r in rows],
                "canonical_reporting_cluster_id": [r["canonical"] for r in rows],
                "valid": [True] * len(rows),
            }
        )
        candidates = pd.DataFrame(
            {
                "candidate_id": ["before_first", "at_bar2_close", "mid_bar3", "after_bar4"],
                "as_of_ts_utc": [
                    (base + pd.Timedelta(seconds=10)).isoformat(),
                    bar_close[1].isoformat(),  # exactly at a completed close
                    (bar_close[2] - pd.Timedelta(seconds=5)).isoformat(),
                    (bar_close[3] + pd.Timedelta(seconds=5)).isoformat(),
                ],
            }
        )
        for order in (panel_assignments, panel_assignments.iloc[::-1]):
            out = assign_panel_regimes_to_candidates(
                panel_frame, order, candidates, protocol=protocol
            ).set_index("candidate_id")
            gap = out.loc["before_first"]
            assert not gap["valid"] and gap["missing_reason"] == "no_completed_panel_bar"
            at_close = out.loc["at_bar2_close"]
            assert at_close["valid"] and at_close["panel_row_id"] == f"bar_{interval}_1"
            assert int(at_close["fold_index"]) == 0  # lowest OOS fold wins
            assert int(at_close["fold_local_cluster_id"]) == 1
            assert at_close["partition"] == "test"
            assert at_close["regime_fit_id"] == "0" * 64
            assert at_close["elapsed_seconds_since_bar_close"] == 0.0
            mid = out.loc["mid_bar3"]  # bar 3 is NOT complete at the as-of instant
            assert mid["valid"] and mid["panel_row_id"] == f"bar_{interval}_1"
            after = out.loc["after_bar4"]  # bar 4 (index 3) has no OOS row
            assert not after["valid"] and after["missing_reason"] == "coverage_gap"
            assert after["panel_row_id"] == f"bar_{interval}_3"
    with pytest.raises(ValueError, match="required columns"):
        assign_panel_regimes_to_candidates(
            panel_frame,
            panel_assignments.drop(columns=["partition"]),
            candidates,
            protocol=protocol,
        )
    with pytest.raises(ValueError, match="required columns"):
        assign_panel_regimes_to_candidates(
            panel_frame.drop(columns=["cbp_valid"]),
            panel_assignments,
            candidates,
            protocol=protocol,
        )


def test_panel_assignment_refuses_non_panel_protocols():
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0,
        resolved_input_features=("distance_to_htf_ticks",),
    )
    with pytest.raises(ValueError, match="CONTEXT_BAR_PANEL grain"):
        assign_panel_regimes_to_candidates(
            pd.DataFrame({"row_id": [], "bar_close_ts_utc": []}),
            pd.DataFrame(),
            pd.DataFrame({"candidate_id": [], "as_of_ts_utc": []}),
            protocol=protocol,
        )


def _synthetic_panel(interval: int = 300, days: int = 40, bars_per_day: int = 12):
    """A completed 5m context-bar panel (fixture-2 geometry per bar) with
    hand-built walk-forward folds over trading days (P1-B; review F14)."""

    rng = np.random.default_rng(7)
    trading_days = tuple(
        day.strftime("%Y-%m-%d") for day in pd.bdate_range("2026-01-05", periods=days)
    )
    centers = np.asarray(CLUSTER_CENTERS[:3], dtype=float)
    rows = []
    for day_index, day in enumerate(trading_days):
        for bar in range(bars_per_day):
            membership = int(rng.integers(0, 3))
            point = centers[membership] + rng.normal(0.0, 0.9, size=2)
            close = pd.Timestamp(f"{day}T14:00:00Z") + pd.Timedelta(seconds=interval * (bar + 1))
            rows.append(
                {
                    "row_id": f"bar_{day_index:02d}_{bar:02d}",
                    "trading_day": day,
                    "bar_close_ts_utc": close.isoformat(),
                    "cbp_valid": True,
                    "cbp_missing_reason": None,
                    _PANEL_INPUTS[0]: float(point[0]),
                    _PANEL_INPUTS[1]: float(point[1]),
                }
            )
    panel = pd.DataFrame(rows)

    def _ids(day_slice) -> tuple[str, ...]:
        return tuple(panel[panel["trading_day"].isin(day_slice)]["row_id"])

    folds = ContextFoldSet(
        folds=(
            IfvgContextFoldDefinition(
                fold_index=0,
                train_days=trading_days[:30],
                test_days=trading_days[30:35],
                train_candidate_ids=_ids(trading_days[:30]),
                test_candidate_ids=_ids(trading_days[30:35]),
                valid=True,
            ),
            IfvgContextFoldDefinition(
                fold_index=1,
                train_days=trading_days[:35],
                test_days=trading_days[35:40],
                train_candidate_ids=_ids(trading_days[:35]),
                test_candidate_ids=_ids(trading_days[35:40]),
                valid=True,
            ),
        ),
        assignment=pd.DataFrame(),
        status="synthetic_panel_folds",
    )
    return panel, folds, trading_days


def test_panel_grain_fit_path_runs_end_to_end_and_assigns_candidates_pit():
    """Review F14: the CONTEXT_BAR_PANEL fit path executes (≥300 training
    bars per fold), and its OOS assignments reach candidate stages through
    the point-in-time rule."""

    panel, folds, trading_days = _synthetic_panel()
    protocol = _panel_protocol(300)
    # R6.1 (D2): the panel protocol's pinned source must be among the
    # verified observation sources
    with pytest.raises(ValueError, match="observation source ids do not include"):
        run_regime_protocol(
            panel, folds, protocol, source_artifact_ids=("c" * 64,), bootstrap_refits=2
        )
    run = run_regime_protocol(
        panel, folds, protocol, source_artifact_ids=("b" * 64,), bootstrap_refits=2
    )
    assessment = run.assessment.payload
    assert assessment.coverage.minimum_training_observations_gate == 300
    assert assessment.coverage.minimum_training_observations_observed >= 300
    assert "sample_adequacy" not in assessment.gate_failures
    assert assessment.gates_passed and assessment.oos_assignment_available
    assert len(run.fold_fits) == 2
    oos = run.assignments[(run.assignments["partition"] == "test") & run.assignments["valid"]]
    assert set(oos["fold_index"]) == {0, 1}
    # candidates: one inside fold 0's test window (OOS), one in the
    # training-only span (no frozen OOS regime → coverage_gap); the as-of
    # instants sit within one interval of the last completed bar
    candidates = pd.DataFrame(
        {
            "candidate_id": ["oos_candidate", "in_sample_span"],
            "as_of_ts_utc": [
                f"{trading_days[32]}T15:03:00Z",
                f"{trading_days[10]}T15:03:00Z",
            ],
        }
    )
    out = assign_panel_regimes_to_candidates(
        panel, run.assignments, candidates, protocol=protocol
    ).set_index("candidate_id")
    assigned = out.loc["oos_candidate"]
    assert assigned["valid"] and assigned["partition"] == "test"
    assert int(assigned["fold_index"]) == 0
    # bars 0..11 close 14:05 … 15:00; at 15:03 the last COMPLETED bar is 11
    assert assigned["panel_row_id"] == "bar_32_11"
    assert assigned["regime_fit_id"] == run.fold_fits[0].fit_envelope.regime_fit_id
    span = out.loc["in_sample_span"]
    assert not span["valid"] and span["missing_reason"] == "coverage_gap"


# ── planned refusals reach the service before any fit (P1-C; S3/S9) ──────────


def test_planned_algorithm_or_policy_cannot_reach_a_fit(monkeypatch):
    import alpha_lab.agents.data_infra.ifvg.ml.regime_service as service

    fixture = known_cluster_fixture(k=3, n=120)
    folds = _folds(fixture)
    protocol = _protocol(fixture)
    calls = {"preprocessing": 0}
    original = service.fit_regime_preprocessing

    def _counting(*args, **kwargs):
        calls["preprocessing"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "fit_regime_preprocessing", _counting)
    hacked_algorithm = protocol.payload.model_copy(update={"algorithm_key": "nystrom_kmeans_v1"})
    hacked_policy = protocol.payload.model_copy(
        update={"out_of_sample_assignment_policy": "none_training_only"}
    )
    for hacked in (hacked_algorithm, hacked_policy):
        with pytest.raises(PermissionError, match="planned"):
            run_regime_protocol(
                fixture.view.frame,
                folds,
                RegimeProtocolEnvelope.from_payload(hacked),
                source_artifact_ids=(fixture.view.view_id,),
                bootstrap_refits=2,
            )
    assert calls["preprocessing"] == 0
    # the private per-fold seam refuses before preprocessing as well (S9)
    with pytest.raises(PermissionError, match="planned"):
        service._fit_one_fold(
            fixture.view.frame,
            service.keyed_observations(fixture.view.frame),
            {},
            RegimeProtocolEnvelope.from_payload(hacked_policy),
            next(f for f in folds.folds if f.valid),
            source_artifact_ids=(fixture.view.view_id,),
        )
    assert calls["preprocessing"] == 0


def test_stability_report_carries_transitions_and_recurrence(cluster_run):
    _fixture, _folds_, _protocol_, run = cluster_run
    stability = run.assessment.payload.stability
    assert stability.bootstrap_refit_count == 10
    assert stability.bootstrap_aligned_ami_mean is not None
    assert stability.bootstrap_aligned_ami_mean > 0.9  # well-separated blobs
    assert stability.bootstrap_aligned_ami_low is not None
    assert set(stability.per_cluster_agreement) == {0, 1, 2}
    assert stability.candidate_event_persistence is not None
    assert len(stability.candidate_event_transition_matrix) == 3
    assert stability.fold_to_fold_recurrence is not None
    assert stability.separation_min_centroid_distance is not None
    assert stability.alignment_space == "scaled_input_features_v1"
    assert stability.stability_gates_passed
    # canonical semantic descriptors label the INPUT-feature space, top-|z| first
    for descriptors in dict(stability.semantic_descriptors).values():
        assert 1 <= len(descriptors) <= len(REGIME_INPUT_FEATURES)
        assert {name for name, _ in descriptors} <= set(REGIME_INPUT_FEATURES)
        magnitudes = [abs(value) for _name, value in descriptors]
        assert magnitudes == sorted(magnitudes, reverse=True)


# ── R6.1 H — per-fold bootstrap, renamed gate, grain-aware transitions ───────

#: The R6 release's fold-0 fit identity for fixture 2 (k=3, n=600) under B0
#: with no winsorization (`../R6/browser-smoke/MANIFEST.json`
#: `healthy.first_fit_id`). Fit identity ignores every stability semantic.
R6_GOLDEN_FOLD0_FIT_ID = "1e183cd722612c28c210396e0350ddc50edf3576c16dddbbb005b2a39360f7d0"


def test_fit_identity_is_the_r6_golden_constant_and_ignores_stability_semantics(cluster_run):
    _fixture, _folds_, _protocol_, run = cluster_run
    assert run.fold_fits[0].fit_envelope.regime_fit_id == R6_GOLDEN_FOLD0_FIT_ID
    assert run.fold_fits[0].fit_envelope.payload.resolved_regime_protocol_id == (
        run.protocol.resolved_regime_protocol_id
    )


def test_bootstrap_runs_on_every_valid_fold_with_fold_seeds(cluster_run):
    """D10: every valid fold is bootstrapped deterministically
    (`rng(7 + fold)`, refit seeds `1000 + 1000·fold + i`); fold 0 reproduces
    the R6 reference-fold numbers; the protocol-wide MINIMUM fold mean is
    the gated quantity under the renamed `minimum_bootstrap_aligned_ami_mean`."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_diagnostics import (
        bootstrap_fold_stability,
        bootstrap_plan,
    )

    fixture, _folds_, _protocol_, run = cluster_run
    stability = run.assessment.payload.stability
    per_fold = stability.per_fold_bootstrap_stability
    assert [entry.fold_index for entry in per_fold] == sorted(
        fit.fold_index for fit in run.fold_fits
    )
    assert all(entry.refit_count == 10 for entry in per_fold)
    assert stability.bootstrap_fold_coverage == 1.0
    reference = stability.reference_fold_bootstrap_stability
    assert reference is per_fold[0] or reference == per_fold[0]
    # the R6 scalar aliases ARE the reference fold's numbers
    assert stability.bootstrap_aligned_ami_mean == reference.aligned_ami_mean
    assert stability.bootstrap_aligned_ami_low == reference.aligned_ami_p05
    assert stability.per_cluster_agreement == reference.per_cluster_agreement
    # the protocol-wide minimum is what the gate reads
    assert stability.protocol_min_bootstrap_aligned_ami_mean == min(
        entry.aligned_ami_mean for entry in per_fold
    )
    assert stability.bootstrap_gate_scope == "protocol_wide_minimum_fold_mean_v1"
    assert stability.minimum_bootstrap_aligned_ami_mean_applied == 0.5
    assert stability.bootstrap_seed_policy.startswith("rng7_plus_fold")
    # deterministic per fold: recomputing fold 1 alone reproduces the report
    fit = run.fold_fits[1]
    indexed = fixture.view.frame.set_index(fixture.view.frame["candidate_id"].astype(str))
    matrix = fit.preprocessing.transform(indexed.loc[list(fit.preprocessing.training_row_ids)])
    again = bootstrap_fold_stability(fit, matrix, k=3, refits=10)
    assert again == per_fold[1]
    # the budget cap reduces the per-fold count only beyond 8 valid folds
    assert bootstrap_plan(3, refits_per_fold=50, total_cap=400) == 50
    assert bootstrap_plan(8, refits_per_fold=50, total_cap=400) == 50
    assert bootstrap_plan(9, refits_per_fold=50, total_cap=400) == 44
    assert bootstrap_plan(0, refits_per_fold=50, total_cap=400) == 0
    assert stability.bootstrap_total_refit_cap == 400
    assert stability.bootstrap_refits_per_fold_requested == 10


def test_bootstrap_uses_the_registry_pinned_parameters(monkeypatch):
    """The refits read the registry pins (never a literal dict): a patched
    pin reaches every KMeans refit."""

    import alpha_lab.agents.data_infra.ifvg.ml.regime_diagnostics as diagnostics

    seen: list[dict] = []
    original = diagnostics.KMeans

    class _Spy(original):
        def __init__(self, **kwargs):
            seen.append(dict(kwargs))
            super().__init__(**kwargs)

    monkeypatch.setattr(diagnostics, "KMeans", _Spy)
    fixture = known_cluster_fixture(k=3, n=200)
    _run(fixture, bootstrap_refits=2)
    pinned = dict(diagnostics.REGIME_ALGORITHM_REGISTRY["kmeans_v1"].pinned_parameters)
    pinned.pop("random_state")
    assert seen, "the bootstrap must construct refits"
    for kwargs in seen:
        assert {k: kwargs[k] for k in pinned} == pinned
        assert kwargs["random_state"] >= 1000


def test_renamed_ami_gate_blocks_on_the_protocol_wide_minimum(monkeypatch):
    """The gate fires as `bootstrap_aligned_ami_below_minimum` when ANY
    fold's mean AMI is below the stamped minimum — the word "advisory" is
    gone from code."""

    import alpha_lab.agents.data_infra.ifvg.ml.regime_diagnostics as diagnostics
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        FoldBootstrapStability,
    )

    original = diagnostics.bootstrap_fold_stability

    def _degraded_last_fold(fit, matrix, *, k, refits):
        entry = original(fit, matrix, k=k, refits=refits)
        if fit.fold_index == 2:
            return FoldBootstrapStability(
                fold_index=entry.fold_index,
                refit_count=entry.refit_count,
                aligned_ami_mean=0.1,
                aligned_ami_p05=0.05,
                per_cluster_agreement=entry.per_cluster_agreement,
            )
        return entry

    monkeypatch.setattr(diagnostics, "bootstrap_fold_stability", _degraded_last_fold)
    fixture = known_cluster_fixture(k=3, n=600)
    run = _run(fixture, bootstrap_refits=2)
    stability = run.assessment.payload.stability
    assert stability.bootstrap_aligned_ami_mean > 0.9  # the REFERENCE fold is fine
    assert stability.protocol_min_bootstrap_aligned_ami_mean == 0.1
    assert "bootstrap_aligned_ami_below_minimum" in stability.gate_failures
    assert not run.assessment.payload.gates_passed
    import inspect

    assert "advisory" not in inspect.getsource(diagnostics).lower()


def test_candidate_event_transitions_reset_on_day_session_and_gap():
    """D11: adjacent OOS candidates pair only within one trading day AND one
    named session and within the stamped maximum gap; every dropped pair is
    counted by cause; elapsed seconds are recorded."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_diagnostics import (
        candidate_event_temporal_facts,
        oos_regime_timeline,
    )

    # NY session (08:00–14:00 ET = 13:00–19:00 UTC in January)
    stamps = [
        ("a", "2026-01-13T14:00:00Z", 0),  # ny
        ("b", "2026-01-13T14:30:00Z", 0),  # ny, +1800 s → counted (same)
        ("c", "2026-01-13T17:00:00Z", 1),  # ny, +9000 s → dropped (gap > 7200)
        ("d", "2026-01-13T17:30:00Z", 1),  # ny, +1800 s → counted (same)
        ("e", "2026-01-13T21:00:00Z", 2),  # 16:00 ET = asia window → session reset
        ("f", "2026-01-13T23:30:00Z", 2),  # 18:30 ET → next trading day → day reset
        ("g", "2026-01-13T23:45:00Z", 0),  # asia, +900 s → counted (2→0)
    ]
    assignments = pd.DataFrame(
        {
            "row_id": [row_id for row_id, _ts, _c in stamps],
            "fold_index": 0,
            "partition": "test",
            "valid": True,
            "observation_ts_utc": [ts for _r, ts, _c in stamps],
            "canonical_reporting_cluster_id": [c for _r, _t, c in stamps],
        }
    )
    timeline = oos_regime_timeline(assignments)
    assert list(timeline["session"]) == ["ny", "ny", "ny", "ny", "asia", "asia", "asia"]
    assert list(timeline["trading_day"])[:5] == ["2026-01-13"] * 5
    assert list(timeline["trading_day"])[5:] == ["2026-01-14"] * 2
    assert timeline["elapsed_seconds"].iloc[1] == 1800.0
    facts = candidate_event_temporal_facts(assignments, 3, maximum_gap_seconds=7200)
    assert facts["pairs_counted"] == 3
    assert facts["pairs_dropped_gap"] == 1
    assert facts["pairs_dropped_boundary"] == {"trading_day": 1, "session": 1}
    assert facts["persistence"] == pytest.approx(2 / 3)
    assert facts["transition_matrix"][0] == (1.0, 0.0, 0.0)
    assert facts["transition_matrix"][2] == (1.0, 0.0, 0.0)
    assert facts["elapsed_seconds_summary"]["max"] == 1800.0
    # pairs never cross folds
    two_folds = assignments.copy()
    two_folds.loc[two_folds["row_id"] == "b", "fold_index"] = 1
    assert (
        candidate_event_temporal_facts(two_folds, 3, maximum_gap_seconds=7200)["pairs_counted"]
        == 2
    )


def test_panel_transitions_are_consecutive_completed_bars_within_a_trading_day():
    """D11 panel half: a pair counts iff same fold, same trading day, elapsed
    == interval, and the earlier bar is not partial."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_diagnostics import (
        panel_temporal_facts,
    )

    base = pd.Timestamp("2026-01-13T14:00:00Z")
    rows = [
        ("b0", base, "2026-01-13", False, 0),
        ("b1", base + pd.Timedelta(seconds=300), "2026-01-13", False, 0),  # counted
        ("b2", base + pd.Timedelta(seconds=600), "2026-01-13", True, 1),  # counted (0→1)
        ("b3", base + pd.Timedelta(seconds=900), "2026-01-13", False, 1),  # dropped: b2 partial
        ("b4", base + pd.Timedelta(seconds=1500), "2026-01-13", False, 1),  # dropped: gap
        ("b5", base + pd.Timedelta(seconds=1800), "2026-01-14", False, 2),  # dropped: day
        ("b6", base + pd.Timedelta(seconds=2100), "2026-01-14", False, 2),  # counted
    ]
    frame = pd.DataFrame(
        {
            "row_id": [r[0] for r in rows],
            "bar_close_ts_utc": [r[1].isoformat() for r in rows],
            "trading_day": [r[2] for r in rows],
            "is_final_partial": [r[3] for r in rows],
        }
    )
    assignments = pd.DataFrame(
        {
            "row_id": [r[0] for r in rows],
            "fold_index": 0,
            "partition": "test",
            "valid": True,
            "observation_ts_utc": [r[1].isoformat() for r in rows],
            "canonical_reporting_cluster_id": [r[4] for r in rows],
        }
    )
    facts = panel_temporal_facts(assignments, frame, 3, interval_seconds=300)
    assert facts["transition_count"] == 3
    assert facts["pairs_dropped_gap"] == 2  # the partial-bar chain break + the 600 s gap
    assert facts["pairs_dropped_boundary"] == 1
    assert facts["persistence"] == pytest.approx(2 / 3)


def test_per_fold_canonical_occupancy_shares_the_rows_gate_key_space(cluster_run):
    _fixture, _folds_, _protocol_, run = cluster_run
    coverage = run.assessment.payload.coverage
    keys = set(coverage.per_fold_canonical_occupancy)
    expected = {f"{fit.fold_index}:{cluster}" for fit in run.fold_fits for cluster in range(3)}
    assert keys == expected
    for fit in run.fold_fits:
        total = sum(
            share
            for key, share in coverage.per_fold_canonical_occupancy.items()
            if key.startswith(f"{fit.fold_index}:")
        )
        assert total == pytest.approx(1.0)


# ── R6.1 safety review S9: the thread-control dependency is imported lazily ──


def test_regime_kernel_names_the_missing_thread_control_dependency(monkeypatch):
    """``threadpoolctl`` is a scikit-learn dependency the repo does not declare:
    the regime lane imports without it and the kernel fails with the exact
    reason when it is missing (never a bare ImportError at module import)."""

    import importlib
    import re
    import sys

    from alpha_lab.agents.data_infra.ifvg.ml import regime_service

    assert "threadpoolctl" not in regime_service.__dict__  # no module-level import
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)  # `import threadpoolctl` → ImportError
    with pytest.raises(RuntimeError, match=re.escape(regime_service.THREADPOOLCTL_MISSING_MESSAGE)):
        regime_service.run_regime_protocol(None, None, None, source_artifact_ids=())
    monkeypatch.undo()
    assert importlib.import_module("threadpoolctl").threadpool_limits is not None
