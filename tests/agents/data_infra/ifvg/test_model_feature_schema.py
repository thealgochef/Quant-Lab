"""Actual fits/reloads prove raw, transformed and persisted schema agreement."""

import json
from dataclasses import replace

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.ml.logistic_model import (
    LOGISTIC_FIT_MANIFEST,
    persist_logistic_fit,
    reload_logistic_fit,
    resolve_logistic_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_feature_schema import fitted_feature_schema
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (
    CATBOOST_BUNDLE_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
)
from alpha_lab.agents.data_infra.ifvg.ml.research_evidence import (
    _manifest,
    load_research_run,
    save_research_inputs,
    save_research_run,
)
from alpha_lab.agents.data_infra.ifvg.ml.supervised_ladder import (
    DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    run_supervised_ladder,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_supervised import (
    class_balanced_supervised_fixture,
)

pytestmark = pytest.mark.filterwarnings("ignore:Skipping features without any observed values")
FEATURES = ("risk_ticks", "entry_fvg_size_ticks", "direction", "entry_family")
EMPTY = "entry_fvg_size_ticks"


@pytest.fixture(scope="module")
def fitted():
    fixture = class_balanced_supervised_fixture(n=120)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    fold = next(fold for fold in folds.folds if fold.valid)
    folds = replace(
        folds,
        folds=(fold,),
        assignment=folds.assignment.loc[folds.assignment["fold_index"].eq(fold.fold_index)],
    )
    frame = fixture.view.frame.copy()
    # No training observation in one feature, while held-out continuation
    # rows do have values. This is valid fold-local missingness, not an absent
    # source projection; source mapping itself is tested at the view boundary.
    frame.loc[frame["candidate_id"].isin(fold.train_candidate_ids), EMPTY] = np.nan
    frame.loc[frame["candidate_id"].isin(fold.test_candidate_ids), EMPTY] = 7.0
    view = replace(fixture.view, frame=frame)
    options = {
        "bundle_features": FEATURES,
        "bundle_ref": "a" * 64,
        "protocols": DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    }
    ladder = run_supervised_ladder(view, fixture.labeled_candidates, folds, **options)
    return fixture, view, folds, options, ladder


def test_valid_training_empty_column_has_explicit_fitted_schema_and_reload(fitted, tmp_path):
    fixture, view, folds, options, ladder = fitted
    request_id = save_research_inputs(
        tmp_path, view, fixture.labeled_candidates, folds, run_kwargs=options
    )
    directory = save_research_run(tmp_path, request_id, ladder)
    restored = load_research_run(tmp_path, request_id)
    assert len(list(directory.rglob("feature_schema.json"))) == 2
    fold_index = folds.folds[0].fold_index
    for protocol_id in (LOGISTIC_PROTOCOL_ID, CATBOOST_BUNDLE_PROTOCOL_ID):
        original, loaded = ladder.rung(protocol_id), restored.rung(protocol_id)
        report = loaded.fold_reports[0]
        assert report["training_missingness"]["numeric_training_empty_features"] == [EMPTY]
        schema = report["feature_schema"]
        assert schema["raw_ordered_features"] == list(FEATURES)
        assert list(loaded.prediction_inputs[fold_index]) == list(FEATURES)
        assert schema == fitted_feature_schema(
            loaded.fitted_models[fold_index], loaded.resolved_protocol
        )
        before = original.fitted_models[fold_index].predict_proba(
            original.prediction_inputs[fold_index]
        )
        after = loaded.fitted_models[fold_index].predict_proba(loaded.prediction_inputs[fold_index])
        np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)
        if protocol_id == LOGISTIC_PROTOCOL_ID:
            assert schema["numeric_value_features_dropped_by_training_imputer"] == [EMPTY]
            assert f"numeric__{EMPTY}" not in schema["transformed_feature_names"]
            assert f"numeric__missingindicator_{EMPTY}" in schema["transformed_feature_names"]
            changed = loaded.prediction_inputs[fold_index].copy()
            changed[EMPTY] = 100000.0
            np.testing.assert_array_equal(
                after, loaded.fitted_models[fold_index].predict_proba(changed)
            )
        else:
            assert schema["numeric_value_features_dropped_by_training_imputer"] == []
            assert schema["transformed_feature_names"] == list(FEATURES)


@pytest.mark.parametrize("protocol_id", [LOGISTIC_PROTOCOL_ID, CATBOOST_BUNDLE_PROTOCOL_ID])
def test_model_cannot_be_declared_under_reordered_protocol(fitted, protocol_id):
    _fixture, _view, folds, _options, ladder = fitted
    rung = ladder.rung(protocol_id)
    altered = replace(
        rung.resolved_protocol,
        ordered_features=tuple(reversed(FEATURES)),
        categorical_features=tuple(reversed(rung.resolved_protocol.categorical_features)),
    )
    with pytest.raises(ValueError, match="feature schema disagrees"):
        fitted_feature_schema(rung.fitted_models[folds.folds[0].fold_index], altered)


@pytest.mark.parametrize("mutation", ["prediction_order", "schema_names"])
def test_internally_checksummed_schema_drift_is_rejected(fitted, tmp_path, mutation):
    fixture, view, folds, options, ladder = fitted
    request_id = save_research_inputs(
        tmp_path, view, fixture.labeled_candidates, folds, run_kwargs=options
    )
    directory = save_research_run(tmp_path, request_id, ladder)
    if mutation == "prediction_order":
        import pandas as pd

        path = next(directory.rglob("prediction_inputs.parquet"))
        frame = pd.read_parquet(path)
        frame.loc[:, list(reversed(frame.columns))].to_parquet(path)
        reason = "prediction input schema"
    else:
        path = next(directory.rglob("feature_schema.json"))
        payload = json.loads(path.read_text())
        payload["transformed_feature_names"][0] = "wrong_feature"
        path.write_text(json.dumps(payload), encoding="utf-8")
        reason = "persisted feature schema"
    # A producer bug can write internally checksummed but semantically wrong
    # data; checksum integrity alone is not the schema contract.
    (directory / "manifest.json").unlink()
    _manifest(directory, request_id, kind="run")
    with pytest.raises(ValueError, match=reason):
        load_research_run(tmp_path, request_id)


def test_portable_v2_refuses_schema_drift_and_keeps_v1_readable(fitted, tmp_path):
    _fixture, _view, folds, _options, ladder = fitted
    rung = ladder.rung(LOGISTIC_PROTOCOL_ID)
    persist_logistic_fit(
        tmp_path,
        pipeline=rung.fitted_models[folds.folds[0].fold_index],
        protocol=rung.resolved_protocol,
        fold_index=folds.folds[0].fold_index,
    )
    path = tmp_path / LOGISTIC_FIT_MANIFEST
    manifest = json.loads(path.read_text())
    assert manifest["artifact_schema_version"] == 2
    manifest["feature_schema"]["fitted_dimension"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted feature schema"):
        reload_logistic_fit(tmp_path)
    manifest["artifact_schema_version"] = 1
    manifest.pop("feature_schema")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, _ = reload_logistic_fit(tmp_path)
    assert fitted_feature_schema(loaded, rung.resolved_protocol)["raw_ordered_features"] == list(
        FEATURES
    )


@pytest.mark.parametrize("features", [(), ("risk_ticks", "risk_ticks")])
def test_logistic_raw_schema_refuses_empty_or_duplicate_features(features):
    with pytest.raises(ValueError, match="nonempty and unique"):
        resolve_logistic_protocol(ordered_features=features, feature_registry_hash="b" * 64)
