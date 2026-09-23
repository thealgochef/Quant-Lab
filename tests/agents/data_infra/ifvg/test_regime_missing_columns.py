"""Training-empty regime columns keep a neutral, persisted fold-local schema."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import RegimeProtocolEnvelope
from alpha_lab.agents.data_infra.ifvg.ml.regime_preprocessing import (
    REGIME_MISSINGNESS_POLICY_V1,
    REGIME_MISSINGNESS_POLICY_V2,
    fit_regime_preprocessing,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    load_regime_fit,
    load_regime_protocol,
    persist_regime_fit,
    persist_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_test_namespace
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import known_cluster_fixture


def _inputs():
    fixture = known_cluster_fixture(n=600)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    fold = next(fold for fold in folds.folds if fold.valid)
    folds = replace(
        folds,
        folds=(fold,),
        assignment=folds.assignment.loc[folds.assignment.fold_index == fold.fold_index].copy(),
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=resolve_bundle("B0_CORE").resolved_feature_bundle_id,
        resolved_input_features=fixture.regime_input_features,
    )
    return fixture, folds, protocol


@pytest.mark.parametrize("winsorization", ["none", "clip_p01_p99_train_fitted_v1"])
def test_training_empty_column_is_neutral_with_populated_test_values_and_roundtrips(
    tmp_path, winsorization
):
    fixture, folds, protocol = _inputs()
    protocol = RegimeProtocolEnvelope.from_payload(
        protocol.payload.model_copy(update={"winsorization_policy": winsorization})
    )
    fold = folds.folds[0]
    features = fixture.regime_input_features
    frame = fixture.view.frame.copy()
    train = frame.candidate_id.isin(fold.train_candidate_ids)
    test = frame.candidate_id.isin(fold.test_candidate_ids)
    frame.loc[train, features[0]] = np.nan
    missing_row = fold.train_candidate_ids[0]
    frame.loc[frame.candidate_id == missing_row, list(features)] = np.nan
    frame.loc[test, features[0]] = 1_000_000.0

    def run(values):
        return run_regime_protocol(
            values, folds, protocol, source_artifact_ids=(fixture.view.view_id,),
            bootstrap_refits=2,
        )

    first = run(frame)
    changed = frame.copy()
    changed.loc[test, features[0]] = -2_000_000.0
    second = run(changed)
    fit = first.fold_fits[0]
    preprocessing = fit.preprocessing
    assert protocol.payload.missingness_policy == REGIME_MISSINGNESS_POLICY_V2
    assert preprocessing.parameter_payload["training_all_missing_features"] == [features[0]]
    assert preprocessing.parameter_payload["numeric_output_features"] == list(features)
    assert preprocessing.parameter_payload == second.fold_fits[0].preprocessing.parameter_payload
    assert fit.fit_envelope == second.fold_fits[0].fit_envelope
    np.testing.assert_array_equal(fit.centroids_scaled, second.fold_fits[0].centroids_scaled)
    pd.testing.assert_frame_equal(first.assignments, second.assignments)
    assert preprocessing.training_rows_all_missing == (missing_row,)
    excluded = first.assignments.loc[first.assignments.row_id == missing_row].iloc[0]
    assert not excluded.valid and excluded.missing_reason == "source_feature_missing"
    transformed = preprocessing.transform(frame.loc[test])
    names = preprocessing.output_feature_names
    assert transformed.shape[1] == len(names) == fit.centroids_scaled.shape[1]
    np.testing.assert_array_equal(transformed[:, 0], 0.0)
    indicator = names.index(f"missing_indicator__{features[0]}")
    np.testing.assert_array_equal(transformed[:, indicator], 0.0)

    initialize_test_namespace(tmp_path)
    persist_regime_protocol(tmp_path, protocol)
    persist_regime_fit(tmp_path, fit, first.assignments, observation_frame=frame)
    loaded = load_regime_fit(tmp_path, fit.fit_envelope.regime_fit_id)
    assert loaded.parameters == preprocessing.parameter_payload
    np.testing.assert_array_equal(loaded.transform(frame.loc[test]), transformed)
    np.testing.assert_array_equal(loaded.transform(changed.loc[test]), transformed)
    np.testing.assert_array_equal(
        loaded.predict(frame.loc[test]), fit.estimator.predict(transformed)
    )
    for cluster, descriptors in loaded.artifact.centroid_or_component_descriptors.items():
        assert tuple(name for name, _value in descriptors) == names
        np.testing.assert_allclose(
            [value for _name, value in descriptors], fit.centroids_scaled[cluster], atol=1e-6
        )


def test_no_training_coordinate_available_refuses_before_imputer_fit(monkeypatch):
    from sklearn.impute import SimpleImputer

    fixture, folds, _protocol = _inputs()
    frame = fixture.view.frame.copy()
    frame.loc[frame.candidate_id.isin(folds.folds[0].train_candidate_ids),
              list(fixture.regime_input_features)] = np.nan

    def no_fit(*args, **kwargs):
        raise AssertionError("an all-empty training matrix reached an imputer fit")

    monkeypatch.setattr(SimpleImputer, "fit", no_fit)
    with pytest.raises(ValueError, match="no training row with any regime input present"):
        fit_regime_preprocessing(frame, fixture.regime_input_features, folds.folds[0])


def test_default_five_minute_panel_can_have_valid_rows_with_an_empty_volume_feature():
    from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
        IfvgContextFoldDefinition,
    )
    from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
        CONTEXT_BAR_PANEL_CATEGORICAL_FEATURES,
        CONTEXT_BAR_PANEL_FEATURES,
    )
    from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_materializer import (
        compute_context_bar_panel_features,
    )
    from alpha_lab.agents.data_infra.ifvg.replay_chart_store import resample_label_bars
    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (
        synthetic_label_source_1m,
    )

    days = ("2026-01-05", "2026-01-06")
    bars = synthetic_label_source_1m(days)
    bars.loc[bars.source_date == days[0], "volume"] = 10
    panel, _validity = compute_context_bar_panel_features(
        resample_label_bars(bars, 300), interval_seconds=300
    )
    train = panel.loc[(panel.trading_day == days[0]) & panel.cbp_valid]
    test = panel.loc[(panel.trading_day == days[1]) & panel.cbp_valid]
    empty_feature = "cbp_volume_intensity_zscore_12"
    assert len(train) > 30 and train[empty_feature].isna().all()
    assert test[empty_feature].notna().any()
    features = tuple(
        name for name in CONTEXT_BAR_PANEL_FEATURES
        if name not in CONTEXT_BAR_PANEL_CATEGORICAL_FEATURES
    )
    fold = IfvgContextFoldDefinition(
        fold_index=0, train_days=(days[0],), test_days=(days[1],),
        train_candidate_ids=tuple(train.row_id), test_candidate_ids=tuple(test.row_id), valid=True,
    )
    fitted = fit_regime_preprocessing(panel, features, fold)
    assert fitted.parameter_payload["training_all_missing_features"] == [empty_feature]
    transformed = fitted.transform(test)
    assert transformed.shape[1] == len(fitted.output_feature_names)
    np.testing.assert_array_equal(transformed[:, features.index(empty_feature)], 0.0)


def test_legacy_policy_and_artifact_remain_readable_with_distinct_v2_identity(tmp_path):
    fixture, folds, current = _inputs()
    legacy = RegimeProtocolEnvelope.from_payload(
        current.payload.model_copy(update={"missingness_policy": REGIME_MISSINGNESS_POLICY_V1})
    )
    assert current.resolved_regime_protocol_id != legacy.resolved_regime_protocol_id
    old_run = run_regime_protocol(
        fixture.view.frame, folds, legacy, source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    old_fit = old_run.fold_fits[0]
    assert old_fit.preprocessing.parameter_payload["pipeline"].endswith("standard_scale_v1")
    assert "training_all_missing_features" not in old_fit.preprocessing.parameter_payload
    initialize_test_namespace(tmp_path)
    persist_regime_protocol(tmp_path, legacy)
    persist_regime_fit(tmp_path, old_fit, old_run.assignments, observation_frame=fixture.view.frame)
    loaded = load_regime_fit(tmp_path, old_fit.fit_envelope.regime_fit_id)
    assert "training_all_missing_features" not in loaded._bundle
    assert load_regime_protocol(tmp_path, legacy.resolved_regime_protocol_id) == legacy
    np.testing.assert_array_equal(
        loaded.transform(fixture.view.frame), old_fit.preprocessing.transform(fixture.view.frame)
    )
    new_fit = run_regime_protocol(
        fixture.view.frame, folds, current, source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    ).fold_fits[0]
    assert old_fit.fit_envelope.regime_fit_id != new_fit.fit_envelope.regime_fit_id
