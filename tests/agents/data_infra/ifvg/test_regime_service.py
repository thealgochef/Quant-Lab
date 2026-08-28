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
    """Review F3: persistence/transitions come from the OOS timeline ordered
    by the observation timestamp within each fold — renaming every id
    changes nothing."""

    fixture = known_cluster_fixture(k=3, n=600)
    run = _run(fixture, bootstrap_refits=2)
    stability = run.assessment.payload.stability
    assert stability.temporal_order_policy == (
        "oos_test_rows_by_observation_ts_within_fold_v1"
    )
    assert stability.temporal_transition_count > 0
    assert stability.temporal_persistence is not None

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
    assert other.temporal_persistence == stability.temporal_persistence
    assert other.transition_matrix == stability.transition_matrix
    assert other.temporal_transition_count == stability.temporal_transition_count
    # the OOS test rows are the timeline: the count never exceeds them
    oos = run.assignments[(run.assignments["partition"] == "test") & run.assignments["valid"]]
    assert stability.temporal_transition_count < len(oos)


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


def _panel_protocol(interval: int):
    return resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0,
        resolved_input_features=REGIME_INPUT_FEATURES,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=interval,
        panel_source_artifact_id="b" * 64,
        panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
    )


def test_panel_pit_assignment_is_last_completed_bar_never_future_and_oos_only():
    """Amendment P1-B (+ review F4): 5m/15m panels assign the frozen
    OUT-OF-SAMPLE regime of the LAST completed bar at or before each
    candidate's as-of instant; in-sample rows are never consulted; the
    choice is independent of input row order; a candidate before the first
    completed bar (or on a bar without an OOS assignment) is a typed
    coverage_gap."""

    for interval in (300, 900):
        protocol = _panel_protocol(interval)
        base = pd.Timestamp("2026-01-13T14:00:00Z")
        bar_close = [base + pd.Timedelta(seconds=interval * (i + 1)) for i in range(4)]
        panel_frame = pd.DataFrame(
            {
                "row_id": [f"bar_{interval}_{i}" for i in range(4)],
                "bar_close_ts_utc": [ts.isoformat() for ts in bar_close],
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
            assert not gap["valid"] and gap["missing_reason"] == "coverage_gap"
            at_close = out.loc["at_bar2_close"]
            assert at_close["valid"] and at_close["panel_row_id"] == f"bar_{interval}_1"
            assert int(at_close["fold_index"]) == 0  # lowest OOS fold wins
            assert int(at_close["fold_local_cluster_id"]) == 1
            assert at_close["partition"] == "test"
            assert at_close["regime_fit_id"] == "0" * 64
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
                    REGIME_INPUT_FEATURES[0]: float(point[0]),
                    REGIME_INPUT_FEATURES[1]: float(point[1]),
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
    # training-only span (no frozen OOS regime → coverage_gap)
    candidates = pd.DataFrame(
        {
            "candidate_id": ["oos_candidate", "in_sample_span"],
            "as_of_ts_utc": [
                f"{trading_days[32]}T15:07:00Z",
                f"{trading_days[10]}T15:07:00Z",
            ],
        }
    )
    out = assign_panel_regimes_to_candidates(
        panel, run.assignments, candidates, protocol=protocol
    ).set_index("candidate_id")
    assigned = out.loc["oos_candidate"]
    assert assigned["valid"] and assigned["partition"] == "test"
    assert int(assigned["fold_index"]) == 0
    # bars 0..11 close 14:05 … 15:00; at 15:07 the last COMPLETED bar is 11
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
    assert stability.temporal_persistence is not None
    assert len(stability.transition_matrix) == 3
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
