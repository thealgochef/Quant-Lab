"""R6 regime persistence suites (ML plan §5.3 dual-format + P1-4 portability):
bind → verify-before-publish → reload → re-transform allclose; manifest-
relative references only; relocation; tamper refusal; verified reuse; the
structural promotion persistence (reviews F5/S1); the assignment↔fit
binding (F6); publish-only-after-verification (S8); the hardened sidecar
loader (S6); and the UI loader that never unpickles (S7)."""

from __future__ import annotations

import json
import shutil

import joblib
import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeRole,
    RegimeStatus,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    PARAMETERS_SIDECAR,
    REGIME_FIT_STORE,
    load_regime_fit,
    load_regime_fit_assignments,
    load_regime_promotion,
    load_regime_protocol,
    persist_regime_assessment,
    persist_regime_fit,
    persist_regime_promotion,
    persist_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SearchStoreError,
    envelope_destination,
    load_sidecar_bytes,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    known_cluster_fixture,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id
_RATIFICATION = "f" * 64


def _run_fixture(n: int, **protocol_overrides):
    fixture = known_cluster_fixture(k=3, n=n)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0,
        resolved_input_features=fixture.regime_input_features,
        **protocol_overrides,
    )
    run = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=5,
    )
    return fixture, folds, protocol, run


def _fold_assignments(run, fold_fit):
    return run.assignments[run.assignments["fold_index"] == fold_fit.fold_index]


@pytest.fixture(scope="module")
def persisted(tmp_path_factory):
    root = tmp_path_factory.mktemp("regime_store")
    fixture, _folds, protocol, run = _run_fixture(600)
    persist_regime_protocol(root, protocol)
    fold_fit = run.fold_fits[0]
    artifact = persist_regime_fit(
        root,
        fold_fit,
        _fold_assignments(run, fold_fit),
        observation_frame=fixture.view.frame,
    )
    persist_regime_assessment(root, run.assessment)
    # an UNDER-SAMPLED run under a DIFFERENT protocol (winsorized) in the
    # same store: its assessment FAILS the gates
    small_fixture, _small_folds, small_protocol, small_run = _run_fixture(
        170, winsorization_policy="clip_p01_p99_train_fitted_v1"
    )
    assert small_protocol.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id
    persist_regime_protocol(root, small_protocol)
    persist_regime_assessment(root, small_run.assessment)
    assert not small_run.assessment.payload.gates_passed
    return {
        "root": root,
        "fixture": fixture,
        "run": run,
        "fold_fit": fold_fit,
        "artifact": artifact,
        "small_run": small_run,
    }


def test_persist_verify_reload_and_reuse(persisted):
    root = persisted["root"]
    fold_fit = persisted["fold_fit"]
    fixture = persisted["fixture"]
    reloaded = load_regime_fit(root, fold_fit.fit_envelope.regime_fit_id)
    indexed = fixture.view.frame.set_index(fixture.view.frame["candidate_id"].astype(str))
    valid = reloaded.assignments[reloaded.assignments["valid"].astype(bool)]
    rows = indexed.loc[[str(r) for r in valid["row_id"]]]
    expected = fold_fit.preprocessing.transform(rows)
    assert np.allclose(reloaded.transform(rows), expected)
    # the reloaded ESTIMATOR reproduces EVERY persisted fold-local id (F6)
    assert (reloaded.predict(rows) == valid["fold_local_cluster_id"].astype(int).to_numpy()).all()
    assert reloaded.artifact.preprocessing_pipeline_ref == "pipeline.joblib"
    assert reloaded.artifact.training_feature_matrix_hash == (
        fold_fit.fit_envelope.payload.training_feature_matrix_hash
    )
    assert len(reloaded.assignments) == len(_fold_assignments(persisted["run"], fold_fit))
    # verified reuse: persisting the identical fit again is idempotent
    persist_regime_fit(
        root,
        fold_fit,
        _fold_assignments(persisted["run"], fold_fit),
        observation_frame=fixture.view.frame,
    )
    # the protocol reloads verified as well
    protocol = load_regime_protocol(root, persisted["run"].protocol.resolved_regime_protocol_id)
    assert protocol.payload.algorithm_key == "kmeans_v1"


def test_relocation_portability(persisted, tmp_path):
    """P1-4: copy the artifact directory, reload by manifest-relative refs
    + checksums, re-transform, allclose — nothing depends on the fitting
    machine's paths."""

    destination = tmp_path / "relocated_store"
    shutil.copytree(persisted["root"], destination)
    fold_fit = persisted["fold_fit"]
    fixture = persisted["fixture"]
    reloaded = load_regime_fit(destination, fold_fit.fit_envelope.regime_fit_id)
    rows = fixture.view.frame.set_index(fixture.view.frame["candidate_id"].astype(str)).loc[
        list(fold_fit.preprocessing.training_row_ids[:25])
    ]
    assert np.allclose(reloaded.transform(rows), fold_fit.preprocessing.transform(rows))


def test_tampered_parameters_fail_closed(persisted, tmp_path):
    destination = tmp_path / "tampered_store"
    shutil.copytree(persisted["root"], destination)
    fit_id = persisted["fold_fit"].fit_envelope.regime_fit_id
    parameters_path = destination / REGIME_FIT_STORE / fit_id / PARAMETERS_SIDECAR
    parameters_path.write_text(
        parameters_path.read_text(encoding="utf-8").replace("standard", "tampered"),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="verification|hash"):
        load_regime_fit(destination, fit_id)


def test_foreign_or_mismatched_assignment_frames_are_refused(persisted, tmp_path):
    """Review F6: a fit persists only with ITS OWN assignment frame."""

    run = persisted["run"]
    fixture = persisted["fixture"]
    fold0 = run.fold_fits[0]
    with pytest.raises(ValueError, match="different fold|different regime_fit_id"):
        persist_regime_fit(
            tmp_path, fold0, _fold_assignments(run, run.fold_fits[1]),
            observation_frame=fixture.view.frame,
        )
    forged = _fold_assignments(run, fold0).copy()
    valid_index = forged.index[forged["valid"].astype(bool)][0]
    forged.loc[valid_index, "fold_local_cluster_id"] = (
        int(forged.loc[valid_index, "fold_local_cluster_id"]) + 1
    ) % 3
    with pytest.raises(ValueError, match="do not match the fit's own estimator"):
        persist_regime_fit(tmp_path, fold0, forged, observation_frame=fixture.view.frame)
    assert not (tmp_path / REGIME_FIT_STORE).exists()


def test_failed_verification_never_publishes(persisted, tmp_path, monkeypatch):
    """Review S8: the exact bytes are verified BEFORE publication, and a
    fresh entry that fails the post-publish reload is withdrawn."""

    import alpha_lab.agents.data_infra.ifvg.ml.regime_store as store

    run = persisted["run"]
    fixture = persisted["fixture"]
    fold_fit = run.fold_fits[1]

    def _broken(*_args, **_kwargs):
        raise RuntimeError("simulated reload failure")

    monkeypatch.setattr(store, "load_regime_fit", _broken)
    with pytest.raises(RuntimeError, match="simulated"):
        persist_regime_fit(
            tmp_path, fold_fit, _fold_assignments(run, fold_fit),
            observation_frame=fixture.view.frame,
        )
    assert not envelope_destination(
        tmp_path, REGIME_FIT_STORE, fold_fit.fit_envelope.regime_fit_id
    ).exists()


def test_sidecar_loader_is_hardened(persisted, tmp_path):
    """Review S6: no traversal names, no stale manifest, hash-of-returned-bytes."""

    root = persisted["root"]
    fit_id = persisted["fold_fit"].fit_envelope.regime_fit_id
    bad_names = (
        "../../outside.bin", "", ".", "./pipeline.joblib", "sub/pipeline.joblib", "manifest.json"
    )
    for bad in bad_names:
        with pytest.raises(SearchStoreError, match="invalid sidecar"):
            load_sidecar_bytes(root, REGIME_FIT_STORE, fit_id, bad)
    destination = tmp_path / "stale_manifest"
    shutil.copytree(root, destination)
    manifest_path = destination / REGIME_FIT_STORE / fit_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append({"path": "extra.bin", "sha256": "0" * 64, "bytes": 0})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SearchStoreError, match="manifest hash mismatch"):
        load_sidecar_bytes(destination, REGIME_FIT_STORE, fit_id, PARAMETERS_SIDECAR)


def test_ui_loader_never_unpickles(persisted, monkeypatch):
    """Review S7: the assignment view loads JSON + Arrow only."""

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("joblib.load must not run on the UI path")

    monkeypatch.setattr(joblib, "load", _forbidden)
    fit_id = persisted["fold_fit"].fit_envelope.regime_fit_id
    envelope, artifact, assignments = load_regime_fit_assignments(persisted["root"], fit_id)
    assert envelope.regime_fit_id == fit_id
    assert artifact.regime_fit_id == fit_id
    assert (assignments["regime_fit_id"] == fit_id).all()


def _decision(run, **overrides) -> RegimePromotionDecisionEnvelope:
    defaults = dict(
        resolved_regime_protocol_id=run.protocol.resolved_regime_protocol_id,
        role=RegimeRole.DESCRIPTIVE_ONLY,
        status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_status=RegimeStatus.PLANNED,
        previous_decision_ref=None,
        capability_assessment_ref=run.assessment.regime_capability_assessment_id,
        owner_ratification_ref=None,
        decided_at="2026-08-26T00:00:00Z",
    )
    defaults.update(overrides)
    return RegimePromotionDecisionEnvelope.from_payload(RegimePromotionDecision(**defaults))


def test_promotion_persists_only_against_verified_evidence(persisted):
    """Reviews F5/S1: the referenced assessment must exist, verify, name the
    protocol, and PASS its gates before FEATURE_ELIGIBLE; the chain of
    decisions is verified; and no fit identity ever changes."""

    root = persisted["root"]
    run = persisted["run"]
    small_run = persisted["small_run"]
    # (a) an assessment that is not in the store
    with pytest.raises(ValueError, match="not a verified entry"):
        persist_regime_promotion(root, _decision(run, capability_assessment_ref="f" * 64))
    # (b) an assessment of ANOTHER protocol
    with pytest.raises(ValueError, match="different regime protocol"):
        persist_regime_promotion(
            root,
            _decision(
                run,
                capability_assessment_ref=small_run.assessment.regime_capability_assessment_id,
            ),
        )
    # (c) the lawful chain: planned → descriptive → stratification-ready
    first = _decision(run)
    persist_regime_promotion(root, first)
    second = _decision(
        run,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, second)
    assert load_regime_promotion(root, second.regime_promotion_decision_id).payload.status is (
        RegimeStatus.STRATIFICATION_READY
    )
    # (d) a chain link that lies about its predecessor's status
    with pytest.raises(ValueError, match="previous_status does not match"):
        persist_regime_promotion(
            root,
            _decision(
                run,
                role=RegimeRole.STRATIFICATION_ONLY,
                status=RegimeStatus.STRATIFICATION_READY,
                previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
                previous_decision_ref=second.regime_promotion_decision_id,
            ),
        )
    with pytest.raises(ValueError, match="previous decision is not a verified entry"):
        persist_regime_promotion(
            root,
            _decision(
                run,
                role=RegimeRole.STRATIFICATION_ONLY,
                status=RegimeStatus.STRATIFICATION_READY,
                previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
                previous_decision_ref="c" * 64,
            ),
        )
    # (e) feature-eligible over a FAILING assessment is unpersistable even
    # with a ratification reference
    small_first = _decision(small_run)
    persist_regime_promotion(root, small_first)
    small_second = _decision(
        small_run,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=small_first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, small_second)
    with pytest.raises(ValueError, match="PASSING capability"):
        persist_regime_promotion(
            root,
            _decision(
                small_run,
                role=RegimeRole.FEATURE_GENERATOR,
                status=RegimeStatus.FEATURE_ELIGIBLE,
                previous_status=RegimeStatus.STRATIFICATION_READY,
                previous_decision_ref=small_second.regime_promotion_decision_id,
                owner_ratification_ref=_RATIFICATION,
            ),
        )
    # (f) …and over the PASSING assessment with the owner's reference it persists
    eligible = _decision(
        run,
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_status=RegimeStatus.STRATIFICATION_READY,
        previous_decision_ref=second.regime_promotion_decision_id,
        owner_ratification_ref=_RATIFICATION,
    )
    persist_regime_promotion(root, eligible)
    # §3.10: promotion changed NO fit identity — the persisted fit reloads
    # under its original id, byte-verified
    reloaded = load_regime_fit(root, persisted["fold_fit"].fit_envelope.regime_fit_id)
    assert reloaded.envelope.regime_fit_id == persisted["fold_fit"].fit_envelope.regime_fit_id


def test_winsorized_fit_persists_and_reloads(tmp_path):
    """Review F16: the train-fitted clip bounds persist in the parameter
    payload and the reloaded pipeline reproduces the clipped transform."""

    fixture, _folds, protocol, run = _run_fixture(
        600, winsorization_policy="clip_p01_p99_train_fitted_v1"
    )
    persist_regime_protocol(tmp_path, protocol)
    fold_fit = run.fold_fits[0]
    persist_regime_fit(
        tmp_path, fold_fit, _fold_assignments(run, fold_fit),
        observation_frame=fixture.view.frame,
    )
    reloaded = load_regime_fit(tmp_path, fold_fit.fit_envelope.regime_fit_id)
    assert reloaded.parameters["clip_lower"] is not None
    assert reloaded.parameters["winsorization_policy"] == "clip_p01_p99_train_fitted_v1"
    indexed = fixture.view.frame.set_index(fixture.view.frame["candidate_id"].astype(str))
    extreme = indexed.loc[list(fold_fit.preprocessing.training_row_ids[:5])].copy()
    extreme[list(fixture.regime_input_features)] = 1e6  # beyond p99 → clipped
    assert np.allclose(reloaded.transform(extreme), fold_fit.preprocessing.transform(extreme))
