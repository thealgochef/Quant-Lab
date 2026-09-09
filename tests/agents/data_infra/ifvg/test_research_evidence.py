"""Durable research evidence: synthetic folds only, never a real study."""

from dataclasses import replace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.ml.research_evidence import (
    INPUT_STORE,
    RUN_STORE,
    load_research_inputs,
    load_research_run,
    research_run_request_id,
    save_research_inputs,
    save_research_run,
)
from alpha_lab.agents.data_infra.ifvg.ml.supervised_ladder import run_supervised_ladder
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_supervised import (
    class_balanced_supervised_fixture,
)
from tests.agents.data_infra.ifvg.test_regime_supervised_studies import lane  # noqa: F401


@pytest.fixture(scope="module")
def inputs():
    fixture = class_balanced_supervised_fixture(n=120)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    # One valid fold is sufficient to exercise every model serializer.
    first = next(fold for fold in folds.folds if fold.valid)
    folds = replace(
        folds,
        folds=(first,),
        assignment=folds.assignment.loc[
            folds.assignment["fold_index"].eq(first.fold_index)
        ].reset_index(drop=True),
    )
    return fixture, folds


@pytest.fixture(scope="module")
def run(inputs):
    fixture, folds = inputs
    return run_supervised_ladder(fixture.view, fixture.labeled_candidates, folds, tier=fixture.tier)


def test_reload_reproduces_all_fold_models_without_fit(inputs, run, tmp_path, monkeypatch):
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline

    fixture, folds = inputs
    kwargs = {"tier": fixture.tier}
    request_id = save_research_inputs(
        tmp_path, fixture.view, fixture.labeled_candidates, folds, run_kwargs=kwargs
    )
    assert load_research_run(tmp_path, request_id) is None
    destination = save_research_run(tmp_path, request_id, run)
    assert destination == tmp_path / RUN_STORE / request_id

    def no_fit(*args, **kwargs):
        raise AssertionError("reuse attempted fitting")

    monkeypatch.setattr(CatBoostClassifier, "fit", no_fit)
    monkeypatch.setattr(Pipeline, "fit", no_fit)
    reloaded = load_research_run(tmp_path, request_id)
    assert reloaded.ladder_id == run.ladder_id
    for before, after in zip(run.rungs, reloaded.rungs, strict=True):
        pd.testing.assert_frame_equal(before.predictions, after.predictions)
        assert set(before.fitted_models) == set(after.fitted_models)
        assert len(after.fitted_models) == (0 if "prevalence" in after.protocol_id else 1)
    # Existing completed runs verify and reuse without rewriting model bytes.
    before = {p: p.stat().st_mtime_ns for p in destination.rglob("*") if p.is_file()}
    assert save_research_run(tmp_path, request_id, run) == destination
    assert before == {p: p.stat().st_mtime_ns for p in before}


def test_request_identity_binds_economics_cohort_and_calendar(inputs):
    fixture, folds = inputs
    kwargs = {"tier": fixture.tier, "fold_schedule_id": "a" * 64}
    baseline = research_run_request_id(
        fixture.view, fixture.labeled_candidates, folds, run_kwargs=kwargs
    )
    economics = fixture.labeled_candidates.copy()
    economics.loc[0, "net_r"] -= 0.125
    assert research_run_request_id(fixture.view, economics, folds, run_kwargs=kwargs) != baseline
    changed = fixture.view.frame.copy()
    changed.loc[0, "distance_to_htf_ticks"] += 1
    assert (
        research_run_request_id(
            replace(fixture.view, frame=changed),
            fixture.labeled_candidates,
            folds,
            run_kwargs=kwargs,
        )
        != baseline
    )
    assert (
        research_run_request_id(
            fixture.view,
            fixture.labeled_candidates,
            folds,
            run_kwargs={**kwargs, "fold_schedule_id": "b" * 64},
        )
        != baseline
    )
    assert (
        research_run_request_id(
            replace(fixture.view, frame=fixture.view.frame.iloc[::-1]),
            fixture.labeled_candidates.iloc[::-1],
            folds,
            run_kwargs=kwargs,
        )
        == baseline
    )


def test_inputs_persist_before_zero_fold_run(inputs, tmp_path):
    fixture, folds = inputs
    empty = replace(folds, folds=(), assignment=folds.assignment.iloc[:0])
    request_id = save_research_inputs(tmp_path, fixture.view, fixture.labeled_candidates, empty)
    directory = tmp_path / INPUT_STORE / request_id
    pd.testing.assert_frame_equal(
        pd.read_parquet(directory / "labels.parquet"), fixture.labeled_candidates
    )
    assert (directory / "fold_definitions.json").read_text().strip() == "[]"
    loaded = load_research_inputs(tmp_path, request_id)
    pd.testing.assert_frame_equal(loaded["labels"], fixture.labeled_candidates)
    pd.testing.assert_frame_equal(loaded["candidate_features"], fixture.view.frame)
    assert loaded["fold_assignment"].empty
    assert loaded["fold_definitions"] == []
    assert load_research_run(tmp_path, request_id) is None


def test_frozen_ladder_identity_binds_full_labels_even_with_same_declared_label_id(inputs):
    from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import PREVALENCE_PROTOCOL_ID

    fixture, folds = inputs
    options = {
        "tier": fixture.tier,
        "protocols": (PREVALENCE_PROTOCOL_ID,),
        "label_artifact_id": "a" * 64,
        "fold_schedule_id": "b" * 64,
    }
    before = run_supervised_ladder(fixture.view, fixture.labeled_candidates, folds, **options)
    changed = fixture.labeled_candidates.copy()
    changed.loc[0, "net_r"] -= 0.5
    after = run_supervised_ladder(fixture.view, changed, folds, **options)
    assert before.ladder_id != after.ladder_id
    calendar = run_supervised_ladder(
        fixture.view,
        fixture.labeled_candidates,
        folds,
        **{**options, "fold_schedule_id": "c" * 64},
    )
    assert before.ladder_id != calendar.ladder_id


def test_model_tampering_refuses_before_deserialization(inputs, run, tmp_path, monkeypatch):
    import alpha_lab.agents.data_infra.ifvg.ml.logistic_model as logistic

    fixture, folds = inputs
    request_id = save_research_inputs(tmp_path, fixture.view, fixture.labeled_candidates, folds)
    directory = save_research_run(tmp_path, request_id, run)
    model_path = next(directory.rglob("pipeline.joblib"))
    model_path.write_bytes(model_path.read_bytes() + b"tampered")

    def no_deserialize(*args, **kwargs):
        raise AssertionError("tampered model was deserialized")

    monkeypatch.setattr(logistic.joblib, "load", no_deserialize)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_research_run(tmp_path, request_id)


@pytest.mark.parametrize("mutation", ["arm_membership", "missing_model"])
def test_checksummed_but_incomplete_result_is_refused(inputs, run, tmp_path, mutation):
    import hashlib
    import json

    from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
        canonical_contract_sha256,
    )

    fixture, folds = inputs
    request_id = save_research_inputs(tmp_path, fixture.view, fixture.labeled_candidates, folds)
    directory = save_research_run(tmp_path, request_id, run)
    path = directory / "run.json"
    metadata = json.loads(path.read_text())
    if mutation == "arm_membership":
        metadata["ladders"]["../outside"] = metadata["ladders"].pop("ladder")
        reason = "arm membership"
    else:
        metadata["ladders"]["ladder"]["rungs"][1]["models"] = []
        reason = "model coverage"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    # Simulate an internally checksummed producer bug; checksums alone must
    # not authorize an incomplete run or an out-of-directory model path.
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["run.json"] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_contract_sha256(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        load_research_run(tmp_path, request_id)


def test_different_label_values_cannot_save_under_existing_request(inputs, run, tmp_path):
    fixture, folds = inputs
    changed = fixture.labeled_candidates.copy()
    changed.loc[0, "gross_r"] += 1
    request_id = save_research_inputs(tmp_path, fixture.view, changed, folds)
    with pytest.raises(ValueError, match="labels disagree"):
        save_research_run(tmp_path, request_id, run)


def test_mbp1_both_arms_and_bundle_catboost_roundtrip(inputs, tmp_path):
    import numpy as np

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import mbp1_feature_names
    from alpha_lab.agents.data_infra.ifvg.ml.comparison_rows import label_artifact_content_id
    from alpha_lab.agents.data_infra.ifvg.ml.controlled_feature_study import (
        run_controlled_mbp1_study,
    )
    from tests.agents.ifvg_search.mbp1_fixture import feature_artifact_for_frame

    fixture, folds = inputs
    rng = np.random.default_rng(23)
    features = pd.DataFrame({"candidate_id": fixture.view.frame["candidate_id"]})
    for name in mbp1_feature_names():
        features[name] = rng.normal(size=len(features))
    artifact, features = feature_artifact_for_frame(features, trading_day="2026-01-05")
    kwargs = {
        "challenger_bundle_key": "B2_CORE_ORDER_FLOW",
        "mbp1_features": features,
        "mbp1_feature_artifact": artifact,
        "label_policy_id": "synthetic_fixture_labels_v1",
        "label_artifact_id": label_artifact_content_id(
            "synthetic_fixture_labels_v1", fixture.labeled_candidates
        ),
    }
    request_id = save_research_inputs(
        tmp_path, fixture.view, fixture.labeled_candidates, folds, runner="mbp1", run_kwargs=kwargs
    )
    study = run_controlled_mbp1_study(fixture.view, fixture.labeled_candidates, folds, **kwargs)
    save_research_run(tmp_path, request_id, study)
    restored = load_research_run(tmp_path, request_id)
    assert restored.envelope == study.envelope
    for arm in ("baseline", "challenger"):
        for original, loaded in zip(
            getattr(study, arm).rungs, getattr(restored, arm).rungs, strict=True
        ):
            pd.testing.assert_frame_equal(original.predictions, loaded.predictions)
            assert original.fitted_models.keys() == loaded.fitted_models.keys()


def test_regime_feature_only_roundtrip_preserves_fold_local_inputs(lane, tmp_path):  # noqa: F811
    from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (
        LOGISTIC_PROTOCOL_ID,
        PREVALENCE_PROTOCOL_ID,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_controlled_study import (
        run_controlled_regime_study,
    )

    kwargs = {
        "activation": lane["activation"],
        "challenger_bundle_key": "B7_CORE_REGIME",
        "fold_features": lane["source"],
        "candidate_fold_set": lane["fold_set"],
        "label_artifact_id": lane["label_artifact_id"],
        "label_policy_id": lane["label_policy_id"],
        "protocols": (PREVALENCE_PROTOCOL_ID, LOGISTIC_PROTOCOL_ID),
    }
    request_id = save_research_inputs(
        tmp_path,
        lane["fixture"].view,
        lane["labels"],
        lane["folds"],
        runner="regime_feature",
        run_kwargs=kwargs,
    )
    study = run_controlled_regime_study(
        lane["fixture"].view, lane["labels"], lane["folds"], **kwargs
    )
    save_research_run(tmp_path, request_id, study)
    loaded = load_research_run(tmp_path, request_id)
    assert loaded.envelope == study.envelope
    for fold_index, before in study.challenger.rung(LOGISTIC_PROTOCOL_ID).prediction_inputs.items():
        after = loaded.challenger.rung(LOGISTIC_PROTOCOL_ID).prediction_inputs[fold_index]
        pd.testing.assert_frame_equal(before, after)
