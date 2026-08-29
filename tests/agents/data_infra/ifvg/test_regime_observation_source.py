"""R6.1 workstream D — the verified observation seam and the regime executor
(plan §6.D / D2; §9.2 ``test_regime_observation_source.py``).

Every regime input is a VERIFIED-LOADED artifact: the source ids come from
the loaded envelope only (a tampered store fails at load), the seam run
equals the kernel run fit for fit, the observation-matrix hash binds the
fitted values, grain / pin mismatches refuse, and the executor runs end to
end on persisted synthetic stores for both grains with verified reuse.
"""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    BUNDLE_FEATURE_VIEW_STORE,
    BUNDLE_VIEW_FRAME_SIDECAR,
)
from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
    PANEL_AS_OF_POLICY_ID_V1,
    PANEL_ASSIGNMENT_MISSING_REASONS,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import ObservationGranularity
from alpha_lab.agents.data_infra.ifvg.ml.regime_executor import execute_regime_protocol
from alpha_lab.agents.data_infra.ifvg.ml.regime_observation_source import (
    RegimeObservationSourceRef,
    assert_source_matches_protocol,
    load_regime_observations,
    observation_matrix_hash,
    run_regime_protocol_from_source,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
    load_regime_oos_assignment,
    load_regime_oos_assignment_frame,
    verify_regime_oos_assignment_frame,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    load_regime_assessment,
    load_regime_fit,
)
from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError, envelope_destination
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import REGIME_INPUT_FEATURES
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
    PANEL_NUMERIC_FEATURES,
    persisted_candidate_source,
    persisted_panel_source,
)


@pytest.fixture(scope="module")
def candidate_source(tmp_path_factory):
    return persisted_candidate_source(tmp_path_factory.mktemp("candidate_store"))


@pytest.fixture(scope="module")
def panel_source(tmp_path_factory):
    return persisted_panel_source(tmp_path_factory.mktemp("panel_store"))


def test_source_ids_come_from_the_loaded_envelope_and_tamper_fails_at_load(
    candidate_source, tmp_path
):
    observations = load_regime_observations(candidate_source.root, candidate_source.source_ref)
    assert observations.source_artifact_ids == (
        candidate_source.view_envelope.bundle_feature_view_id,
    )
    assert observations.observation_granularity is ObservationGranularity.CANDIDATE_STAGE_ROW
    assert observations.provenance["frame_table_sha256"] == (
        candidate_source.view_envelope.frame_table_sha256
    )
    assert len(observations.frame) == len(candidate_source.view_frame)
    # a tampered frame sidecar fails at LOAD — nothing downstream ever sees it
    directory = envelope_destination(
        candidate_source.root,
        BUNDLE_FEATURE_VIEW_STORE,
        candidate_source.view_envelope.bundle_feature_view_id,
    )
    sidecar = directory / BUNDLE_VIEW_FRAME_SIDECAR
    original = sidecar.read_bytes()
    try:
        sidecar.write_bytes(original[:-16] + b"\x00" * 16)
        with pytest.raises((SearchStoreError, ValueError)):
            load_regime_observations(candidate_source.root, candidate_source.source_ref)
    finally:
        sidecar.write_bytes(original)
    # an unknown id is refused at the store, never fabricated
    with pytest.raises(SearchStoreError):
        load_regime_observations(
            candidate_source.root,
            RegimeObservationSourceRef(source_kind="bundle_feature_view", artifact_id="9" * 64),
        )


def test_seam_run_equals_kernel_run_fit_for_fit(candidate_source):
    observations = load_regime_observations(candidate_source.root, candidate_source.source_ref)
    seam = run_regime_protocol_from_source(
        observations, candidate_source.folds, candidate_source.protocol, bootstrap_refits=3
    )
    kernel = run_regime_protocol(
        candidate_source.view_frame,
        candidate_source.folds,
        candidate_source.protocol,
        source_artifact_ids=(candidate_source.view_envelope.bundle_feature_view_id,),
        bootstrap_refits=3,
    )
    assert [fit.fit_envelope.regime_fit_id for fit in seam.fold_fits] == [
        fit.fit_envelope.regime_fit_id for fit in kernel.fold_fits
    ]
    assert seam.assessment.regime_capability_assessment_id == (
        kernel.assessment.regime_capability_assessment_id
    )
    assert seam.fold_set_id == kernel.fold_set_id


def test_observation_matrix_hash_binds_values_not_order(candidate_source):
    frame = candidate_source.view_frame
    reference = observation_matrix_hash(frame, REGIME_INPUT_FEATURES)
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert observation_matrix_hash(shuffled, REGIME_INPUT_FEATURES) == reference
    reordered = frame.loc[:, list(reversed(list(frame.columns)))]
    assert observation_matrix_hash(reordered, REGIME_INPUT_FEATURES) == reference
    perturbed = frame.copy()
    column = REGIME_INPUT_FEATURES[0]
    perturbed.loc[perturbed.index[0], column] = float(perturbed[column].iloc[0]) + 1e-6
    assert observation_matrix_hash(perturbed, REGIME_INPUT_FEATURES) != reference
    # the full-frame hash (every column) is what the seam records
    observations = load_regime_observations(candidate_source.root, candidate_source.source_ref)
    assert observations.observation_matrix_hash == observation_matrix_hash(observations.frame)


def test_grain_and_pin_mismatches_are_refused(candidate_source, panel_source):
    candidate_observations = load_regime_observations(
        candidate_source.root, candidate_source.source_ref
    )
    with pytest.raises(ValueError, match="does not match the loaded observation source"):
        assert_source_matches_protocol(candidate_observations, panel_source.protocol)
    panel_observations = load_regime_observations(panel_source.root, panel_source.source_ref)
    assert panel_observations.source_artifact_ids == (
        panel_source.panel_envelope.context_bar_panel_artifact_id,
    )
    with pytest.raises(ValueError, match="does not match the loaded observation source"):
        assert_source_matches_protocol(panel_observations, candidate_source.protocol)
    foreign_panel = resolve_kmeans_protocol(
        input_feature_bundle_ref=panel_source.protocol.payload.input_feature_bundle_ref,
        resolved_input_features=PANEL_NUMERIC_FEATURES,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=300,
        panel_source_artifact_id="c" * 64,
        panel_as_of_policy_id=PANEL_AS_OF_POLICY_ID_V1,
    )
    with pytest.raises(ValueError, match="different context_bar_panel_artifact_id"):
        assert_source_matches_protocol(panel_observations, foreign_panel)
    other_interval = resolve_kmeans_protocol(
        input_feature_bundle_ref=panel_source.protocol.payload.input_feature_bundle_ref,
        resolved_input_features=PANEL_NUMERIC_FEATURES,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=900,
        panel_source_artifact_id=panel_source.panel_envelope.context_bar_panel_artifact_id,
        panel_as_of_policy_id=PANEL_AS_OF_POLICY_ID_V1,
    )
    with pytest.raises(ValueError, match="different panel interval"):
        assert_source_matches_protocol(panel_observations, other_interval)


def test_executor_end_to_end_candidate_grain_with_verified_reuse(candidate_source):
    root = candidate_source.root
    first = execute_regime_protocol(
        root,
        protocol=candidate_source.protocol,
        observation_source=candidate_source.source_ref,
        fold_set_artifact_id=candidate_source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=3,
    )
    assert first.source_artifact_ids == (candidate_source.view_envelope.bundle_feature_view_id,)
    assert first.fold_schedule_id == candidate_source.schedule.fold_schedule_id
    assert first.regime_fit_ids and not any(first.fits_reused)
    for fit_id in first.regime_fit_ids:
        assert load_regime_fit(root, fit_id).envelope.regime_fit_id == fit_id
    assessment = load_regime_assessment(root, first.regime_capability_assessment_id)
    assert assessment.payload.resolved_regime_protocol_id == (
        candidate_source.protocol.resolved_regime_protocol_id
    )
    envelope = load_regime_oos_assignment(root, first.oos_assignment.regime_oos_assignment_id)
    frame = load_regime_oos_assignment_frame(root, envelope)
    verify_regime_oos_assignment_frame(envelope, frame)
    assert envelope.payload.assignment_source == "candidate_fold_oos"
    assert set(frame["candidate_id"]) == set(candidate_source.view_frame["candidate_id"])
    assert frame["valid"].any()
    # a second execution REUSES every fit by reproduction and mints identical ids
    second = execute_regime_protocol(
        root,
        protocol=candidate_source.protocol,
        observation_source=candidate_source.source_ref,
        fold_set_artifact_id=candidate_source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=3,
    )
    assert second.regime_fit_ids == first.regime_fit_ids
    assert all(second.fits_reused)
    assert second.regime_capability_assessment_id == first.regime_capability_assessment_id
    assert second.oos_assignment.regime_oos_assignment_id == (
        first.oos_assignment.regime_oos_assignment_id
    )


def test_executor_end_to_end_panel_grain_assigns_candidates_pit(panel_source, candidate_source):
    root = panel_source.root
    with pytest.raises(ValueError, match="requires the candidate as-of source"):
        execute_regime_protocol(
            root,
            protocol=panel_source.protocol,
            observation_source=panel_source.source_ref,
            fold_set_artifact_id=panel_source.fold_set_envelope.fold_set_artifact_id,
            bootstrap_refits=2,
        )
    # adversarial R6.1 F1: a caller string + in-memory frame is NOT provenance
    with pytest.raises(TypeError, match="not evidence"):
        execute_regime_protocol(
            root,
            protocol=panel_source.protocol,
            observation_source=panel_source.source_ref,
            fold_set_artifact_id=panel_source.fold_set_envelope.fold_set_artifact_id,
            candidate_as_of_source=("synthetic_candidates", panel_source.candidate_frame),
            bootstrap_refits=2,
        )
    with pytest.raises(TypeError, match="not evidence"):
        execute_regime_protocol(
            root,
            protocol=panel_source.protocol,
            observation_source=panel_source.source_ref,
            fold_set_artifact_id=panel_source.fold_set_envelope.fold_set_artifact_id,
            candidate_as_of_source="bundle_feature_view:" + "f" * 64,
            bootstrap_refits=2,
        )
    # a panel artifact is not a candidate as-of source
    with pytest.raises(ValueError, match="must be a persisted bundle_feature_view"):
        execute_regime_protocol(
            root,
            protocol=panel_source.protocol,
            observation_source=panel_source.source_ref,
            fold_set_artifact_id=panel_source.fold_set_envelope.fold_set_artifact_id,
            candidate_as_of_source=panel_source.source_ref,
            bootstrap_refits=2,
        )
    # a well-formed ref that names no persisted bundle view fails closed
    with pytest.raises(Exception, match="bundle_feature_views|not found|missing|exist"):
        execute_regime_protocol(
            root,
            protocol=panel_source.protocol,
            observation_source=panel_source.source_ref,
            fold_set_artifact_id=panel_source.fold_set_envelope.fold_set_artifact_id,
            candidate_as_of_source=RegimeObservationSourceRef(
                source_kind="bundle_feature_view", artifact_id="f" * 64
            ),
            bootstrap_refits=2,
        )
    result = execute_regime_protocol(
        root,
        protocol=panel_source.protocol,
        observation_source=panel_source.source_ref,
        fold_set_artifact_id=panel_source.fold_set_envelope.fold_set_artifact_id,
        candidate_as_of_source=panel_source.candidate_source_ref,
        bootstrap_refits=2,
    )
    assert result.source_artifact_ids == (
        panel_source.panel_envelope.context_bar_panel_artifact_id,
    )
    assert result.regime_fit_ids
    envelope = result.oos_assignment
    assert envelope.payload.assignment_source == "panel_pit"
    assert envelope.payload.panel_context is not None
    assert envelope.payload.panel_context.max_staleness_seconds == 300
    # the provenance line is the LOADED candidate bundle view's id
    view_id = panel_source.candidate_view_envelope.bundle_feature_view_id
    assert envelope.payload.candidate_as_of_source_ref == f"bundle_feature_view:{view_id}"
    frame = load_regime_oos_assignment_frame(root, envelope)
    verify_regime_oos_assignment_frame(envelope, frame)
    loaded_view = load_regime_observations(root, panel_source.candidate_source_ref)
    assert set(frame["candidate_id"]) == set(loaded_view.frame["candidate_id"].astype(str))
    valid = frame[frame["valid"].astype(bool)]
    assert len(valid) > 0 and (valid["partition"] == "test").all()
    assert (valid["elapsed_seconds_since_bar_close"] <= 300).all()
    reasons = set(frame.loc[~frame["valid"].astype(bool), "missing_reason"].dropna())
    assert reasons <= set(PANEL_ASSIGNMENT_MISSING_REASONS)
    # the candidate grain refuses the parameter (its as-of comes from the loaded source)
    with pytest.raises(ValueError, match="legal for the panel grain only"):
        execute_regime_protocol(
            candidate_source.root,
            protocol=candidate_source.protocol,
            observation_source=candidate_source.source_ref,
            fold_set_artifact_id=candidate_source.fold_set_envelope.fold_set_artifact_id,
            candidate_as_of_source=panel_source.candidate_source_ref,
            bootstrap_refits=2,
        )


def test_executor_refuses_foreign_fold_sets_and_sources(candidate_source, panel_source, tmp_path):
    with pytest.raises(ValueError, match="fold set grain"):
        execute_regime_protocol(
            panel_source.root,
            protocol=candidate_source.protocol,
            observation_source=candidate_source.source_ref,
            fold_set_artifact_id=panel_source.fold_set_envelope.fold_set_artifact_id,
            bootstrap_refits=2,
        )
    other = persisted_candidate_source(candidate_source.root, n=500)
    assert other.view_envelope.bundle_feature_view_id != (
        candidate_source.view_envelope.bundle_feature_view_id
    )
    with pytest.raises(ValueError, match="different observation source"):
        execute_regime_protocol(
            candidate_source.root,
            protocol=candidate_source.protocol,
            observation_source=other.source_ref,
            fold_set_artifact_id=candidate_source.fold_set_envelope.fold_set_artifact_id,
            bootstrap_refits=2,
        )
