"""Logistic-protocol identity + portable-artifact tests (ML plan §2, P1-4)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.context_feature_view import features_for_tier
from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.ml.logistic_model import (
    LOGISTIC_FIT_PARAMETERS,
    _logistic_frame,
    persist_logistic_fit,
    reload_logistic_fit,
    resolve_logistic_protocol,
    run_logistic_fold_models,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_supervised import (
    class_balanced_supervised_fixture,
)


@pytest.fixture(scope="module")
def fixture():
    return class_balanced_supervised_fixture(n=400)


@pytest.fixture(scope="module")
def folds(fixture):
    return build_context_folds(
        fixture.labeled_candidates,
        authorized_trading_days=fixture.trading_days,
    )


def _features(fixture):
    return features_for_tier(fixture.tier)


def test_protocol_hash_is_deterministic_and_sensitive(fixture):
    features = _features(fixture)
    first = resolve_logistic_protocol(
        ordered_features=features,
        feature_registry_hash=fixture.view.feature_registry_hash,
    )
    second = resolve_logistic_protocol(
        ordered_features=features,
        feature_registry_hash=fixture.view.feature_registry_hash,
    )
    assert first.resolved_hash == second.resolved_hash
    changed = resolve_logistic_protocol(
        ordered_features=features,
        feature_registry_hash="0" * 64,
    )
    assert changed.resolved_hash != first.resolved_hash
    annotated = resolve_logistic_protocol(
        ordered_features=features,
        feature_registry_hash=fixture.view.feature_registry_hash,
        manual_feature_overrides={"note": "registered annotation"},
    )
    assert annotated.resolved_hash != first.resolved_hash


def test_locked_protocol_overrides_are_refused(fixture):
    for forbidden in ("features", "categorical_features", "model_parameters", "threshold"):
        with pytest.raises(ValueError, match="cannot alter locked protocol"):
            resolve_logistic_protocol(
                ordered_features=_features(fixture),
                feature_registry_hash=fixture.view.feature_registry_hash,
                manual_feature_overrides={forbidden: "x"},
            )


def test_logistic_frame_types_and_missing_category(fixture):
    features = _features(fixture)
    protocol = resolve_logistic_protocol(
        ordered_features=features,
        feature_registry_hash=fixture.view.feature_registry_hash,
    )
    frame = _logistic_frame(
        fixture.view.frame,
        features=features,
        categoricals=protocol.categorical_features,
    )
    assert frame["direction"].dtype == object
    assert set(protocol.categorical_features) & {"direction", "entry_family"}
    # numeric NaN survives for the fold-local imputer
    assert frame["distance_to_htf_ticks"].isna().any()
    missing_feature_rows = frame["distance_to_htf_ticks"].isna().sum()
    assert missing_feature_rows > 0


def test_missing_registered_feature_is_refused(fixture):
    features = _features(fixture)
    protocol = resolve_logistic_protocol(
        ordered_features=features,
        feature_registry_hash=fixture.view.feature_registry_hash,
    )
    broken = fixture.view.frame.drop(columns=["risk_ticks"])
    with pytest.raises(ValueError, match="missing registered features"):
        _logistic_frame(
            broken, features=features, categoricals=protocol.categorical_features
        )


def test_permutation_importance_emitted_with_pinned_repeats(fixture, folds):
    run = run_logistic_fold_models(
        fixture.view,
        fixture.labeled_candidates,
        folds,
        features=_features(fixture),
    )
    assert not run.feature_importance.empty
    assert set(run.feature_importance["permutation_repeats"]) == {20}
    valid_folds = {fold.fold_index for fold in folds.folds if fold.valid}
    assert set(run.feature_importance["fold_index"]) == valid_folds


def test_fold_local_fit_statistics_are_blind_to_test_rows(fixture, folds):
    """Direct leak probe (adversarial m-14): perturbing a TEST row's numeric
    feature leaves every fitted statistic (imputer medians, scaler
    mean/scale, coefficients) byte-identical — preprocessing and fit see
    training-fold rows only."""

    import dataclasses

    features = _features(fixture)
    first_valid = next(fold for fold in folds.folds if fold.valid)

    class _OneFold:
        folds = (first_valid,)

    def _fit_stats(view):
        sink: dict[int, object] = {}
        run_logistic_fold_models(
            view,
            fixture.labeled_candidates,
            _OneFold(),
            features=features,
            fitted_fold_sink=sink,
        )
        pipeline = sink[first_valid.fold_index]
        from alpha_lab.agents.data_infra.ifvg.ml.logistic_model import (
            _fitted_parameter_payload,
        )

        return _fitted_parameter_payload(pipeline)

    baseline_stats = _fit_stats(fixture.view)
    perturbed_frame = fixture.view.frame.copy()
    test_candidate = first_valid.test_candidate_ids[0]
    mask = perturbed_frame["candidate_id"].astype(str) == test_candidate
    perturbed_frame.loc[mask, "opposing_size_ticks"] = 1_000_000.0
    perturbed_view = dataclasses.replace(fixture.view, frame=perturbed_frame)
    assert _fit_stats(perturbed_view) == baseline_stats


def test_portable_artifact_relocation_roundtrip(fixture, folds, tmp_path):
    """P1-4: copy the artifact directory, reload by manifest-relative refs +
    checksums, re-transform, np.allclose — nothing depends on the original
    path."""

    sink: dict[int, object] = {}
    run = run_logistic_fold_models(
        fixture.view,
        fixture.labeled_candidates,
        folds,
        features=_features(fixture),
        fitted_fold_sink=sink,
    )
    fold_index, pipeline = next(iter(sorted(sink.items())))
    original = tmp_path / "artifact_original"
    manifest = persist_logistic_fit(
        original,
        pipeline=pipeline,
        protocol=run.protocol,
        fold_index=fold_index,
    )
    assert all(
        "/" not in name and "\\" not in name for name in manifest["references"]
    ), "manifest references must be manifest-relative filenames"

    relocated = tmp_path / "relocated" / "deeper" / "artifact_copy"
    relocated.parent.mkdir(parents=True)
    shutil.copytree(original, relocated)

    fold = next(f for f in folds.folds if f.fold_index == fold_index)
    frame = fixture.view.frame.set_index("candidate_id", verify_integrity=True)
    test_frame = frame.loc[list(fold.test_candidate_ids)]
    from alpha_lab.agents.data_infra.ifvg.ml.logistic_model import _logistic_frame

    test_x = _logistic_frame(
        test_frame,
        features=_features(fixture),
        categoricals=run.protocol.categorical_features,
    )
    expected = pipeline.predict_proba(test_x)[:, 1]

    reloaded, reloaded_manifest = reload_logistic_fit(relocated)
    assert reloaded_manifest["fitted_parameter_payload_hash"] == (
        manifest["fitted_parameter_payload_hash"]
    )
    actual = reloaded.predict_proba(test_x)[:, 1]
    assert np.allclose(actual, expected)


def test_relocated_artifact_with_byte_drift_is_refused(fixture, folds, tmp_path):
    sink: dict[int, object] = {}
    run = run_logistic_fold_models(
        fixture.view,
        fixture.labeled_candidates,
        folds,
        features=_features(fixture),
        fitted_fold_sink=sink,
    )
    fold_index, pipeline = next(iter(sorted(sink.items())))
    directory = tmp_path / "artifact"
    persist_logistic_fit(
        directory, pipeline=pipeline, protocol=run.protocol, fold_index=fold_index
    )
    tampered = directory / LOGISTIC_FIT_PARAMETERS
    tampered.write_text(
        tampered.read_text(encoding="utf-8").replace("1", "2", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum|hash"):
        reload_logistic_fit(directory)
