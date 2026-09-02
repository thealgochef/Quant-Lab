"""R6.1-FIX workstream A — the fold-local feature artifact binds the exact
per-fit assignment sidecars it was built from (plan §3.2; finding F-03).

``FoldFitRef`` carries the sidecar hash + schema hash whenever a fit exists;
``build_regime_fold_features`` consumes VERIFIED fit assignments only and
refuses any in-memory frame; the loader re-checks every ref against the
store without listing; the identity moves when a fit's stored bytes move.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.fold_schedules import derive_fold_schedule
from alpha_lab.agents.data_infra.ifvg.ml.fold_set_artifact import (
    build_fold_set_artifact,
    persist_fold_schedule,
    persist_fold_set_artifact,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    FIT_ASSIGNMENT_SCHEMA_HASH,
    ObservationGranularity,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_fold_features import (
    FoldFitRef,
    build_regime_fold_features,
    load_regime_fold_feature_source,
    load_regime_fold_features,
    save_regime_fold_features,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    ASSIGNMENTS_SIDECAR,
    REGIME_FIT_STORE,
    VerifiedFitAssignments,
    load_regime_fit_assignments,
    persist_regime_fit,
)
from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError, envelope_destination
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import known_cluster_fixture

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    root = tmp_path_factory.mktemp("fold_feature_evidence")
    fixture = known_cluster_fixture(k=3, n=600)
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
    verified: dict[int, VerifiedFitAssignments] = {}
    for fold_fit in run.fold_fits:
        persist_regime_fit(
            root,
            fold_fit,
            run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
            observation_frame=fixture.view.frame,
        )
        verified[fold_fit.fold_index] = load_regime_fit_assignments(
            root, fold_fit.fit_envelope.regime_fit_id
        )
    return {
        "root": root,
        "fixture": fixture,
        "folds": folds,
        "schedule": schedule,
        "fold_set": fold_set,
        "protocol": protocol,
        "run": run,
        "verified": verified,
    }


def _build(lane, **overrides):
    kwargs = dict(
        protocol=lane["protocol"],
        regime_run=lane["run"],
        fit_assignments=lane["verified"],
        candidate_fold_set=lane["fold_set"],
        candidate_folds=lane["folds"],
        regime_fold_set=lane["fold_set"],
        schedule=lane["schedule"],
        candidate_view_frame=lane["fixture"].view.frame,
        candidate_view_id=lane["fixture"].view.view_id,
    )
    kwargs.update(overrides)
    return build_regime_fold_features(**kwargs)


def test_fold_feature_identity_binds_each_fits_assignment_table_hash(lane):
    envelope, frame = _build(lane)
    refs = envelope.payload.regime_fit_ids_by_fold
    assert refs
    verified = lane["verified"]
    for ref in refs:
        if ref.regime_fit_id is None:
            assert ref.assignments_sidecar_sha256 is None
            assert ref.assignment_schema_hash is None
            continue
        source = verified[ref.fold_index]
        assert ref.regime_fit_id == source.regime_fit_id
        assert ref.assignments_sidecar_sha256 == source.assignments_sidecar_sha256
        assert ref.assignment_schema_hash == FIT_ASSIGNMENT_SCHEMA_HASH
    # a ref with a fit id but no sidecar binding is unconstructible, and vice versa
    with pytest.raises(ValueError, match="sidecar|together"):
        FoldFitRef(fold_index=0, regime_fit_id="b" * 64)
    with pytest.raises(ValueError, match="sidecar|together"):
        FoldFitRef(fold_index=0, regime_fit_id=None, assignments_sidecar_sha256="c" * 64)
    # the identity moves when a fit's stored assignment bytes move
    fold_index = refs[0].fold_index if refs[0].regime_fit_id else refs[1].fold_index
    source = verified[fold_index]
    tampered = {**verified}
    tampered[fold_index] = VerifiedFitAssignments(
        regime_fit_id=source.regime_fit_id,
        envelope=source.envelope,
        artifact=source.artifact,
        frame=source.frame,
        assignments_sidecar_sha256="0" * 64,
        assignment_schema_hash=source.assignment_schema_hash,
    )
    other, _ = _build(lane, fit_assignments=tampered)
    assert other.regime_fold_feature_artifact_id != envelope.regime_fold_feature_artifact_id
    # the frame's values come from the verified frames (fit k → fold k only)
    columns = envelope.payload.columns
    valid = frame[frame[columns.valid]]
    assert len(valid) > 0
    for ref in refs:
        if ref.regime_fit_id is None:
            continue
        rows = valid[valid["fold_index"] == ref.fold_index]
        assert (rows["regime_fit_id"] == ref.regime_fit_id).all()


def test_model_facing_rows_are_validated_on_build_and_load(lane):
    """§3.4: a valid fold-feature row carries every finite numeric feature and
    the local id; an invalid row carries none and one registered reason —
    refused on the seam and on the verified load."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_fold_features import (
        RegimeFoldFeatureFrameSource,
        validate_fold_feature_rows,
    )

    envelope, frame = _build(lane)
    columns = envelope.payload.columns
    validate_fold_feature_rows(frame, columns)
    valid_index = frame.index[frame[columns.valid]][0]
    broken = frame.copy()
    broken.loc[valid_index, columns.margin] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_fold_feature_rows(broken, columns)
    with pytest.raises(ValueError):
        RegimeFoldFeatureFrameSource(envelope, broken)
    broken = frame.copy()
    broken.loc[valid_index, columns.local_id] = None
    with pytest.raises(ValueError, match="fit-local cluster id"):
        validate_fold_feature_rows(broken, columns)
    invalid_index = frame.index[~frame[columns.valid]]
    if len(invalid_index):
        broken = frame.copy()
        broken.loc[invalid_index[0], columns.assigned_distance] = 1.0
        with pytest.raises(ValueError, match="invalid fold-feature row"):
            validate_fold_feature_rows(broken, columns)
        broken = frame.copy()
        broken.loc[invalid_index[0], columns.missing_reason] = "made_up"
        with pytest.raises(ValueError, match="registered"):
            validate_fold_feature_rows(broken, columns)


def test_fold_feature_builder_rejects_unverified_assignment_frames(lane):
    run = lane["run"]
    unverified = {
        fit.fold_index: run.assignments[run.assignments["fold_index"] == fit.fold_index]
        for fit in run.fold_fits
    }
    with pytest.raises(TypeError, match="VerifiedFitAssignments"):
        _build(lane, fit_assignments=unverified)
    # a verified mapping whose fit ids disagree with the run's fits is refused
    verified = lane["verified"]
    first, second = sorted(verified)[:2]
    swapped = {**verified, first: verified[second], second: verified[first]}
    with pytest.raises(ValueError, match="fit"):
        _build(lane, fit_assignments=swapped)
    # an incomplete mapping (a fit of the run without its verified frame) is refused
    partial = {k: v for k, v in verified.items() if k != first}
    with pytest.raises(ValueError, match="verified"):
        _build(lane, fit_assignments=partial)


def test_fold_feature_loader_rechecks_every_fit_ref_against_the_store(lane, tmp_path):
    root = lane["root"]
    envelope, frame = _build(lane)
    save_regime_fold_features(root, envelope, frame)
    artifact_id = envelope.regime_fold_feature_artifact_id
    source = load_regime_fold_feature_source(root, artifact_id)
    assert source.artifact_id == artifact_id
    # tampering a bound fit sidecar fails the fold-feature load closed even
    # though the fold-feature table itself is intact
    ref = next(r for r in envelope.payload.regime_fit_ids_by_fold if r.regime_fit_id)
    sidecar = envelope_destination(root, REGIME_FIT_STORE, ref.regime_fit_id) / ASSIGNMENTS_SIDECAR
    original = sidecar.read_bytes()
    try:
        sidecar.write_bytes(original[:-4] + b"\x00" * 4)
        with pytest.raises((SearchStoreError, ValueError)):
            load_regime_fold_feature_source(root, artifact_id)
        with pytest.raises((SearchStoreError, ValueError)):
            load_regime_fold_features(root, artifact_id)
    finally:
        sidecar.write_bytes(original)
    # a fold-feature artifact whose ref names a sidecar hash the store does
    # not hold is refused at load (no listing; exact ids only)
    forged = envelope.model_copy(
        update={
            "payload": envelope.payload.model_copy(
                update={
                    "regime_fit_ids_by_fold": tuple(
                        (
                            r.model_copy(update={"assignments_sidecar_sha256": "1" * 64})
                            if r.regime_fit_id == ref.regime_fit_id
                            else r
                        )
                        for r in envelope.payload.regime_fit_ids_by_fold
                    )
                }
            )
        }
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_fold_features import (
        RegimeFoldFeatureArtifactEnvelope,
    )

    forged = RegimeFoldFeatureArtifactEnvelope.from_payload(
        forged.payload, feature_table_sha256=envelope.feature_table_sha256
    )
    other_root = tmp_path / "forged"
    import shutil

    shutil.copytree(root, other_root)
    save_regime_fold_features(other_root, forged, frame)
    with pytest.raises(ValueError, match="sidecar|ref"):
        load_regime_fold_feature_source(other_root, forged.regime_fold_feature_artifact_id)
    assert isinstance(frame, pd.DataFrame)
