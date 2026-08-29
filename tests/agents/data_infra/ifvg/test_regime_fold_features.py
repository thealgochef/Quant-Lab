"""R6.1 workstream G (S09b) — the fold-local regime feature artifact (plan
§6.G, D7 / D8; §9.1 ``test_regime_fold_features_cannot_see_future_outer_folds``).

Per-fold partition semantics (train rows ← fit k in-sample, test rows ← fit k
OOS); every train/test row of a valid fold has a feature row; model-facing
columns are fit-local while canonical alignment is reporting-only; typed
reasons; artifact identity / rehash / relocation; the panel grain through
the PIT rule with the candidate's OWN partition; and the leakage proof.
"""

from __future__ import annotations

import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.fold_schedules import derive_fold_schedule
from alpha_lab.agents.data_infra.ifvg.ml import regime_oos_assignment as oos_module
from alpha_lab.agents.data_infra.ifvg.ml.comparison_rows import RegimeFoldFeatureSource
from alpha_lab.agents.data_infra.ifvg.ml.fold_set_artifact import (
    build_fold_set_artifact,
    persist_fold_schedule,
    persist_fold_set_artifact,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import ObservationGranularity
from alpha_lab.agents.data_infra.ifvg.ml.regime_fold_features import (
    FOLD_FEATURE_SIDECAR,
    REGIME_FOLD_FEATURE_MISSING_REASONS,
    REGIME_FOLD_FEATURE_STORE,
    PanelFoldFeatureInputs,
    RegimeFoldFeatureFrameSource,
    build_regime_fold_features,
    fold_feature_columns,
    fold_feature_table_bytes,
    load_regime_fold_feature_frame,
    load_regime_fold_feature_source,
    load_regime_fold_features,
    save_regime_fold_features,
    verify_regime_fold_feature_frame,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError, envelope_destination
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
    persisted_panel_source,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id


def _candidate_lane(root, n: int = 600):
    fixture = known_cluster_fixture(k=3, n=n)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    schedule = derive_fold_schedule(fixture.trading_days)
    persist_fold_schedule(root, schedule)
    fold_set, definitions = build_fold_set_artifact(
        folds,
        schedule=schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=fixture.view.view_id,
        minimum_train_observations=30,
        labeled=True,
    )
    persist_fold_set_artifact(root, fold_set, definitions)
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0, resolved_input_features=fixture.regime_input_features
    )
    run = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    return {
        "fixture": fixture,
        "folds": folds,
        "schedule": schedule,
        "fold_set": fold_set,
        "protocol": protocol,
        "run": run,
    }


def _build(lane, **overrides):
    kwargs = dict(
        protocol=lane["protocol"],
        regime_run=lane["run"],
        candidate_fold_set=lane["fold_set"],
        candidate_folds=lane["folds"],
        regime_fold_set=lane["fold_set"],
        schedule=lane["schedule"],
        candidate_view_frame=lane["fixture"].view.frame,
        candidate_view_id=lane["fixture"].view.view_id,
    )
    kwargs.update(overrides)
    return build_regime_fold_features(**kwargs)


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    root = tmp_path_factory.mktemp("fold_features")
    state = _candidate_lane(root)
    envelope, frame = _build(state)
    state["root"] = root
    state["envelope"] = envelope
    state["frame"] = frame
    return state


def test_partition_semantics_every_train_and_test_row_has_a_fit_k_row(lane):
    envelope, frame = lane["envelope"], lane["frame"]
    run, folds = lane["run"], lane["folds"]
    columns = envelope.payload.columns
    fits = {fit.fold_index: fit for fit in run.fold_fits}
    valid_folds = [fold for fold in folds.folds if fold.valid]
    assert valid_folds and set(frame["fold_index"]) == {fold.fold_index for fold in valid_folds}
    for fold in valid_folds:
        rows = frame[frame["fold_index"] == fold.fold_index].set_index("candidate_id")
        train = rows[rows["partition"] == "train"]
        test = rows[rows["partition"] == "test"]
        assert set(train.index) == set(fold.train_candidate_ids)
        assert set(test.index) == set(fold.test_candidate_ids)
        fit = fits[fold.fold_index]
        assert (rows["regime_fit_id"] == fit.fit_envelope.regime_fit_id).all()
        own = run.assignments[run.assignments["fold_index"] == fold.fold_index]
        for partition, subset in (("train", train), ("test", test)):
            source = own[own["partition"] == partition].set_index("row_id")
            valid = subset[subset[columns.valid]]
            assert len(valid) > 0
            expected_ids = source.loc[valid.index, "fold_local_cluster_id"].astype(int).astype(str)
            assert (valid[columns.local_id] == expected_ids).all()
            expected_margin = source.loc[valid.index, "assignment_margin"].astype(float)
            assert np.allclose(valid[columns.margin].to_numpy(), expected_margin.to_numpy())
            expected_distances = np.vstack(source.loc[valid.index, "distances"].to_numpy())
            assert np.allclose(valid[list(columns.distances)].to_numpy(), expected_distances)
    # the model-facing names never include the canonical reporting id
    payload = envelope.payload
    assert payload.hard_id_encoding == "none"
    assert "canonical_reporting_cluster_id" not in payload.model_feature_names
    assert set(payload.model_feature_names) == set(columns.numeric)
    assert payload.categorical_model_features == ()
    assert set(frame.loc[frame[columns.valid], "canonical_reporting_cluster_id"].dropna()) <= {
        0,
        1,
        2,
    }
    assert payload.candidate_fold_set_hash == lane["fold_set"].payload.fold_set_id
    assert payload.valid_row_count == int(frame[columns.valid].sum())
    # the categorical encoding exposes the local id as a model feature
    categorical_envelope, _ = _build(lane, hard_id_encoding="fit_local_categorical_v1")
    assert categorical_envelope.payload.model_feature_names[-1] == columns.local_id
    assert categorical_envelope.payload.categorical_model_features == (columns.local_id,)
    assert (
        categorical_envelope.regime_fold_feature_artifact_id
        != envelope.regime_fold_feature_artifact_id
    )


def test_regime_fold_features_cannot_see_future_outer_folds(lane, monkeypatch):
    """§9.1: perturbing every observation after fold k's test window (and every
    OTHER fit's rows) leaves fold k's rows byte-identical; the S09b path never
    opens the descriptive OOS artifact."""

    fixture, folds, protocol = lane["fixture"], lane["folds"], lane["protocol"]
    reference = lane["frame"]
    columns = lane["envelope"].payload.columns
    valid_folds = [fold for fold in folds.folds if fold.valid]
    fold = valid_folds[0]
    last_test_day = max(fold.test_days)
    later = fixture.view.frame["trading_day"].astype(str) > last_test_day
    assert later.sum() > 0
    perturbed = fixture.view.frame.copy()
    for feature in REGIME_INPUT_FEATURES:
        perturbed.loc[later, feature] = perturbed.loc[later, feature] + 50.0

    def _never(*args, **kwargs):
        raise AssertionError("the descriptive OOS artifact loader must never be consulted")

    monkeypatch.setattr(oos_module, "load_regime_oos_assignment", _never)
    monkeypatch.setattr(oos_module, "load_regime_oos_assignment_frame", _never)
    perturbed_run = run_regime_protocol(
        perturbed,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    _envelope, perturbed_frame = _build(
        lane, regime_run=perturbed_run, candidate_view_frame=perturbed
    )
    before = reference[reference["fold_index"] == fold.fold_index]
    after = perturbed_frame[perturbed_frame["fold_index"] == fold.fold_index]
    assert fold_feature_table_bytes(before, columns) == fold_feature_table_bytes(after, columns)
    # later folds DID change (their training rows were perturbed) — the
    # artifact is not trivially constant
    later_fold = valid_folds[-1]
    assert fold_feature_table_bytes(
        reference[reference["fold_index"] == later_fold.fold_index], columns
    ) != fold_feature_table_bytes(
        perturbed_frame[perturbed_frame["fold_index"] == later_fold.fold_index], columns
    )
    # tampering every OTHER fit's assignment rows changes nothing in fold k
    run = lane["run"]
    tampered = run.assignments.copy()
    others = tampered["fold_index"] != fold.fold_index
    tampered.loc[others, "fold_local_cluster_id"] = 0
    tampered.loc[others, "canonical_reporting_cluster_id"] = 0
    tampered.loc[others, "assignment_margin"] = 9.9
    tampered_run = replace(run, assignments=tampered)
    _envelope, tampered_frame = _build(lane, regime_run=tampered_run)
    assert fold_feature_table_bytes(
        tampered_frame[tampered_frame["fold_index"] == fold.fold_index], columns
    ) == fold_feature_table_bytes(before, columns)


def test_typed_reasons_for_missing_fits_rows_and_inputs(lane):
    run, folds, fixture = lane["run"], lane["folds"], lane["fixture"]
    columns = lane["envelope"].payload.columns
    valid_folds = [fold for fold in folds.folds if fold.valid]
    fold = valid_folds[0]
    # (a) no valid regime fit for fold k → every fold-k row typed
    without_fit = replace(
        run,
        fold_fits=tuple(fit for fit in run.fold_fits if fit.fold_index != fold.fold_index),
        assignments=run.assignments[run.assignments["fold_index"] != fold.fold_index],
    )
    envelope, frame = _build(lane, regime_run=without_fit)
    rows = frame[frame["fold_index"] == fold.fold_index]
    assert len(rows) == len(fold.train_candidate_ids) + len(fold.test_candidate_ids)
    assert (~rows[columns.valid]).all()
    assert set(rows[columns.missing_reason]) == {"no_valid_regime_fit"}
    assert rows["regime_fit_id"].isna().all()
    refs = envelope.payload.regime_fit_ids_by_fold
    ref = next(r for r in refs if r.fold_index == fold.fold_index)
    assert ref.regime_fit_id is None
    # (b) a fit row missing for one candidate → no_fit_assignment_row
    victim = str(fold.test_candidate_ids[0])
    dropped = run.assignments[
        ~(
            (run.assignments["fold_index"] == fold.fold_index)
            & (run.assignments["row_id"].astype(str) == victim)
        )
    ]
    _envelope, frame = _build(lane, regime_run=replace(run, assignments=dropped))
    row = frame[(frame["fold_index"] == fold.fold_index) & (frame["candidate_id"] == victim)]
    assert len(row) == 1 and row.iloc[0][columns.missing_reason] == "no_fit_assignment_row"
    assert np.isnan(row.iloc[0][columns.margin])
    # (c) an all-missing input row carries the FIT's own typed reason
    hollow = fixture.view.frame.copy()
    hollow.loc[hollow["candidate_id"] == victim, list(REGIME_INPUT_FEATURES)] = np.nan
    hollow_run = run_regime_protocol(
        hollow,
        folds,
        lane["protocol"],
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    _envelope, frame = _build(lane, regime_run=hollow_run, candidate_view_frame=hollow)
    row = frame[(frame["fold_index"] == fold.fold_index) & (frame["candidate_id"] == victim)]
    assert row.iloc[0][columns.missing_reason] == "source_feature_missing"
    assert set(frame.loc[~frame[columns.valid], columns.missing_reason].dropna()) <= set(
        REGIME_FOLD_FEATURE_MISSING_REASONS
    )


def test_artifact_persists_rehashes_relocates_and_serves_the_ladder_seam(lane, tmp_path):
    root = lane["root"]
    envelope, frame = lane["envelope"], lane["frame"]
    save_regime_fold_features(root, envelope, frame)
    save_regime_fold_features(root, envelope, frame)  # verified reuse
    reloaded = load_regime_fold_features(root, envelope.regime_fold_feature_artifact_id)
    assert reloaded.model_dump(mode="json") == envelope.model_dump(mode="json")
    stored = load_regime_fold_feature_frame(root, reloaded)
    verify_regime_fold_feature_frame(reloaded, stored)
    columns = envelope.payload.columns
    assert fold_feature_table_bytes(stored, columns) == fold_feature_table_bytes(frame, columns)
    # relocation
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    again = load_regime_fold_feature_source(moved, envelope.regime_fold_feature_artifact_id)
    assert again.artifact_id == envelope.regime_fold_feature_artifact_id
    # the seam (fork J's Protocol): fold-k rows only, candidate_id + feature names
    source = RegimeFoldFeatureFrameSource(envelope, frame)
    assert isinstance(source, RegimeFoldFeatureSource)
    assert source.feature_names == envelope.payload.model_feature_names
    assert source.categorical_features == ()
    fold = next(fold for fold in lane["folds"].folds if fold.valid)
    joined = source.frame_for_fold(fold.fold_index)
    assert list(joined.columns) == ["candidate_id", *source.feature_names]
    assert set(joined["candidate_id"]) == set(fold.train_candidate_ids) | set(
        fold.test_candidate_ids
    )
    assert not joined["candidate_id"].duplicated().any()
    assert source.frame_for_fold(10_000).empty
    reasons = source.reasons_for_fold(fold.fold_index)
    assert set(reasons.columns) == {"candidate_id", "partition", "valid", "missing_reason"}
    ids = source.local_ids_for_fold(fold.fold_index)
    assert set(ids["local_id"]) <= {"0", "1", "2"}
    # a perturbed frame no longer hashes to the envelope; a tampered sidecar fails closed
    perturbed = frame.copy()
    perturbed.loc[perturbed.index[0], columns.margin] = 123.0
    with pytest.raises(ValueError, match="does not hash"):
        verify_regime_fold_feature_frame(envelope, perturbed)
    with pytest.raises(ValueError, match="does not hash"):
        RegimeFoldFeatureFrameSource(envelope, perturbed)
    artifact_id = envelope.regime_fold_feature_artifact_id
    sidecar = (
        envelope_destination(root, REGIME_FOLD_FEATURE_STORE, artifact_id) / FOLD_FEATURE_SIDECAR
    )
    original = sidecar.read_bytes()
    try:
        sidecar.write_bytes(original[:-8] + b"\x00" * 8)
        with pytest.raises((SearchStoreError, ValueError)):
            load_regime_fold_feature_source(root, envelope.regime_fold_feature_artifact_id)
    finally:
        sidecar.write_bytes(original)


def test_preconditions_refuse_foreign_runs_schedules_and_populations(lane, tmp_path):
    fixture = lane["fixture"]
    other_protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0,
        resolved_input_features=fixture.regime_input_features,
        winsorization_policy="clip_p01_p99_train_fitted_v1",
    )
    with pytest.raises(ValueError, match="different regime protocol"):
        _build(lane, protocol=other_protocol)
    with pytest.raises(ValueError, match="not the fold sets' schedule"):
        _build(lane, schedule=derive_fold_schedule(fixture.trading_days[:50]))
    smaller = known_cluster_fixture(k=3, n=500)
    other_folds = build_context_folds(
        smaller.labeled_candidates, authorized_trading_days=smaller.trading_days
    )
    with pytest.raises(ValueError, match="not the candidate fold-set artifact's population"):
        _build(lane, candidate_folds=other_folds)
    other_fold_set, _ = build_fold_set_artifact(
        other_folds,
        schedule=lane["schedule"],
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=smaller.view.view_id,
        minimum_train_observations=30,
        labeled=True,
    )
    with pytest.raises(ValueError, match="did not run on the regime fold-set artifact"):
        _build(lane, regime_fold_set=other_fold_set)
    with pytest.raises(ValueError, match="hard_id_encoding"):
        _build(lane, hard_id_encoding="one_hot")
    with pytest.raises(ValueError, match="lawful only for the panel grain"):
        _build(
            lane,
            panel=PanelFoldFeatureInputs(panel_frame=pd.DataFrame(), panel_artifact=object()),
        )


def test_panel_grain_uses_fit_k_and_the_candidates_own_partition(tmp_path):
    root = tmp_path / "panel"
    panel = persisted_panel_source(root)
    fixture = known_cluster_fixture(k=3, n=600)
    assert fixture.trading_days == panel.days
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    candidate_fold_set, definitions = build_fold_set_artifact(
        folds,
        schedule=panel.schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=fixture.view.view_id,
        minimum_train_observations=30,
        labeled=True,
    )
    persist_fold_set_artifact(root, candidate_fold_set, definitions)
    run = run_regime_protocol(
        panel.panel_frame,
        panel.folds,
        panel.protocol,
        source_artifact_ids=(panel.panel_envelope.context_bar_panel_artifact_id,),
        bootstrap_refits=2,
    )
    assert run.fold_fits
    with pytest.raises(ValueError, match="requires PanelFoldFeatureInputs"):
        build_regime_fold_features(
            protocol=panel.protocol,
            regime_run=run,
            candidate_fold_set=candidate_fold_set,
            candidate_folds=folds,
            regime_fold_set=panel.fold_set_envelope,
            schedule=panel.schedule,
            candidate_view_frame=fixture.view.frame,
            candidate_view_id=fixture.view.view_id,
        )
    envelope, frame = build_regime_fold_features(
        protocol=panel.protocol,
        regime_run=run,
        candidate_fold_set=candidate_fold_set,
        candidate_folds=folds,
        regime_fold_set=panel.fold_set_envelope,
        schedule=panel.schedule,
        candidate_view_frame=fixture.view.frame,
        candidate_view_id=fixture.view.view_id,
        panel=PanelFoldFeatureInputs(
            panel_frame=panel.panel_frame, panel_artifact=panel.panel_envelope
        ),
    )
    payload = envelope.payload
    assert payload.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL
    assert payload.panel_context is not None and payload.panel_context.max_staleness_seconds == 300
    panel_block_id = panel.panel_envelope.payload.resolved_feature_block_id
    assert payload.resolved_feature_block_id == panel_block_id
    columns = fold_feature_columns(panel.protocol.resolved_regime_protocol_id, 3)
    fits = {fit.fold_index: fit.fit_envelope.regime_fit_id for fit in run.fold_fits}
    valid = frame[frame[columns.valid]]
    assert len(valid) > 0
    assert valid["panel_row_id"].notna().all()
    for fold in folds.folds:
        if not fold.valid or fold.fold_index not in fits:
            continue
        rows = frame[frame["fold_index"] == fold.fold_index]
        assert (rows["regime_fit_id"] == fits[fold.fold_index]).all()
        train = rows[rows["partition"] == "train"]
        assert set(train["candidate_id"]) == set(fold.train_candidate_ids)
        # every VALID train row of fold k came from fit k's TRAIN partition rows
        own = run.assignments[
            (run.assignments["fold_index"] == fold.fold_index)
            & (run.assignments["partition"] == "train")
        ]
        valid_train = train[train[columns.valid]]
        assert set(valid_train["panel_row_id"]) <= set(own["row_id"].astype(str))
    reasons = set(frame.loc[~frame[columns.valid], columns.missing_reason].dropna())
    assert reasons <= set(REGIME_FOLD_FEATURE_MISSING_REASONS)
    # the panel artifact must BE the protocol's pinned source
    with pytest.raises(ValueError, match="pinned panel source"):
        build_regime_fold_features(
            protocol=panel.protocol,
            regime_run=run,
            candidate_fold_set=candidate_fold_set,
            candidate_folds=folds,
            regime_fold_set=panel.fold_set_envelope,
            schedule=panel.schedule,
            candidate_view_frame=fixture.view.frame,
            candidate_view_id=fixture.view.view_id,
            panel=PanelFoldFeatureInputs(
                panel_frame=panel.panel_frame,
                panel_artifact=panel.panel_envelope.model_copy(
                    update={"context_bar_panel_artifact_id": "9" * 64}
                ),
            ),
        )
