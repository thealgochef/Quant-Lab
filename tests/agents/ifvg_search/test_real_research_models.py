"""Real research stage adapters exercised only on temporary synthetic inputs."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_model import categorical_features_for
from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    build_bundle_feature_view,
    save_bundle_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import mbp1_feature_names
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import CATBOOST_BUNDLE_PROTOCOL_ID
from alpha_lab.agents.data_infra.ifvg.ml.research_evidence import (
    load_research_inputs,
    load_research_run,
)
from alpha_lab.agents.data_infra.ifvg.search import pipeline
from alpha_lab.agents.data_infra.ifvg.search.research_artifacts import (
    load_research_labels,
    run_durable_supervised_stage,
    save_research_labels,
)
from tests.agents.data_infra.ifvg.test_regime_supervised_studies import lane  # noqa: F401
from tests.agents.ifvg_search.mbp1_fixture import feature_artifact_for_frame
from tests.agents.ifvg_search.pipeline_fixture import LABEL_POLICY_ID, build_mini_view


def _no_fit(*args, **kwargs):
    raise AssertionError("completed research attempted another fit")


@pytest.fixture(autouse=True)
def _no_operational_checkpoint(monkeypatch):
    # These are stage integration contexts, not operational worker executions.
    # The model-input checkpoint still records its typed IDs in stage state.
    monkeypatch.setattr(pipeline, "_checkpoint", lambda context: None)


def _context(root, bundle_key="B4_CORE_STRUCTURE_LIQUIDITY", *, day_count=45):
    days = tuple(
        day.strftime("%Y-%m-%d") for day in pd.bdate_range("2026-01-05", periods=day_count)
    )
    # Forty-five authorized days but only forty-three candidate-bearing days:
    # an observed-day calendar would incorrectly produce zero folds.
    empty_days = {days[index] for index in (7, 42) if index < len(days)}
    view, labels = build_mini_view(tuple(day for day in days if day not in empty_days), per_day=4)
    labels["is_warmup"] = False
    resolution = resolve_bundle(bundle_key)
    names = tuple(resolution.payload.resolved_feature_names)
    categorical = set(categorical_features_for(names))
    rng = np.random.default_rng(391)
    frame = view.frame.copy()
    additional = {}
    for name in names:
        if name not in frame and name not in mbp1_feature_names():
            additional[name] = (
                np.where(np.arange(len(frame)) % 2, "synthetic_a", "synthetic_b")
                if name in categorical
                else rng.normal(size=len(frame))
            )
    frame = pd.concat([frame, pd.DataFrame(additional)], axis=1)
    view = replace(view, frame=frame)
    mbp = {}
    if bundle_key == "B2_CORE_ORDER_FLOW":
        metrics = pd.DataFrame({"candidate_id": frame["candidate_id"]})
        for name in mbp1_feature_names():
            metrics[name] = rng.normal(size=len(metrics))
        artifact, metrics = feature_artifact_for_frame(metrics, trading_day=days[0])
        mbp = {"feature_frame": metrics, "feature_envelope": artifact}
        bundle_view, bundle_frame = build_bundle_feature_view(
            view, resolution, mbp1_features=metrics, mbp1_feature_artifact=artifact
        )
    else:
        bundle_view, bundle_frame = build_bundle_feature_view(view, resolution)
    bundle_view = save_bundle_feature_view(root, bundle_view, bundle_frame)
    subject = SimpleNamespace(subject_id="a" * 64, evaluation_dates=days)
    label_envelope = save_research_labels(
        root, subject=subject, view=view, labels=labels, label_policy_id=LABEL_POLICY_ID
    )
    context = SimpleNamespace(
        store_root=root,
        wiring=SimpleNamespace(research_subject=subject),
        semantic=SimpleNamespace(
            payload=SimpleNamespace(
                feature_bundle_ids=(bundle_key,),
                label_policy_id=LABEL_POLICY_ID,
                model_protocol_id=CATBOOST_BUNDLE_PROTOCOL_ID,
                regime_study=None,
            )
        ),
        view=view,
        labeled=labels,
        folds=None,
        regime={},
        bundle_views={bundle_key: bundle_view},
        bundle_frames={bundle_key: bundle_frame},
        mbp1_evidence=mbp,
        label_artifact_id=label_envelope.payload.label_artifact_id,
        research_label_store_id=label_envelope.research_label_id,
        research_model_run_ids=[],
        stage_sidecars={},
        ladder=None,
        state={"stages": {pipeline.QuantLabPipelineStage.S09_TRAIN_MODELS.value: {}}},
    )
    pipeline._stage_s08_folds(context)
    return context, days, empty_days


@pytest.mark.parametrize("bundle_key", ["B4_CORE_STRUCTURE_LIQUIDITY", "B2_CORE_ORDER_FLOW"])
def test_real_model_stage_persists_all_rungs_and_reuses_without_fit(
    tmp_path, monkeypatch, bundle_key
):
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline

    context, days, empty_days = _context(tmp_path, bundle_key)
    assert len(context.folds.folds) == 1
    assert context.folds.folds[0].valid
    assert context.regime["schedule"].payload.authorized_trading_days == days
    assert empty_days.isdisjoint(set(context.labeled["trading_day"]))
    assert context.folds.folds[0].test_days == days[40:45]
    record = json.loads(context.stage_sidecars["research_folds.json"])
    assert record["authorized_trading_days"] == list(days)
    assert record["trading_days_source"] == "frozen_research_logical_calendar_v1"

    outputs, note = run_durable_supervised_stage(context)
    assert "fitted, persisted and reloaded" in note
    request_id = context.research_model_run_ids[-1]
    inputs = load_research_inputs(tmp_path, request_id)
    pd.testing.assert_frame_equal(inputs["labels"], context.labeled)
    assert inputs["request"]["run_kwargs"]["fold_schedule_id"] == record["fold_schedule_id"]
    restored = load_research_run(tmp_path, request_id)
    arms = (
        (restored.baseline, restored.challenger)
        if bundle_key == "B2_CORE_ORDER_FLOW"
        else (restored,)
    )
    for arm in arms:
        assert len(arm.rungs) == 3
        assert all(len(rung.predictions) == 16 for rung in arm.rungs)
        assert sum(len(rung.fitted_models) for rung in arm.rungs) == 2
    if len(arms) == 2:
        for baseline, challenger in zip(arms[0].rungs, arms[1].rungs, strict=True):
            assert (
                baseline.predictions["comparison_row_id"].tolist()
                == challenger.predictions["comparison_row_id"].tolist()
            )
    else:
        assert set(resolve_bundle(bundle_key).payload.resolved_feature_names) == set(
            arms[0].rungs[-1].resolved_protocol.ordered_features
        )

    monkeypatch.setattr(CatBoostClassifier, "fit", _no_fit)
    monkeypatch.setattr(Pipeline, "fit", _no_fit)
    replayed, reused_note = pipeline._stage_s09_supervised_ladder(context)
    assert replayed == outputs
    assert "verified and reloaded" in reused_note


def test_zero_fold_run_retains_inputs_and_typed_insufficiency(tmp_path, monkeypatch):
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline

    context, days, _ = _context(tmp_path, day_count=3)
    assert not context.folds.folds
    summary = json.loads(context.stage_sidecars["fold_summary.json"])
    assert summary == {
        "fold_count": 0,
        "valid_fold_count": 0,
        "invalid_reasons": [],
        "trading_day_count": 3,
    }
    monkeypatch.setattr(CatBoostClassifier, "fit", _no_fit)
    monkeypatch.setattr(Pipeline, "fit", _no_fit)
    outputs, note = pipeline._stage_s09_supervised_ladder(context)
    assert "insufficient evidence" in note
    request_id = context.research_model_run_ids[-1]
    assert request_id in outputs
    loaded = load_research_inputs(tmp_path, request_id)
    assert loaded["fold_assignment"].empty
    assert loaded["fold_definitions"] == []
    assert context.regime["schedule"].payload.authorized_trading_days == days
    restored = load_research_run(tmp_path, request_id)
    assert all(rung.predictions.empty and not rung.fitted_models for rung in restored.rungs)
    _, labels = load_research_labels(tmp_path, context.research_label_store_id)
    pd.testing.assert_frame_equal(labels, context.labeled)


def test_failed_fit_leaves_verified_inputs_discoverable(tmp_path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.ml import supervised_ladder

    context, _, _ = _context(tmp_path)
    monkeypatch.setattr(supervised_ladder, "run_supervised_ladder", _no_fit)
    with pytest.raises(AssertionError, match="another fit"):
        run_durable_supervised_stage(context)
    entry = context.state["stages"][pipeline.QuantLabPipelineStage.S09_TRAIN_MODELS.value]
    (request_id,) = entry["research_model_input_ids"]
    pd.testing.assert_frame_equal(
        load_research_inputs(tmp_path, request_id)["labels"], context.labeled
    )
    assert load_research_run(tmp_path, request_id) is None


def test_feature_only_real_stage_hook_reuses_both_arms_without_fit(lane, monkeypatch):  # noqa: F811
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline

    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import load_regime_fit_assignments
    from alpha_lab.agents.data_infra.ifvg.ml.regime_supervised_stage import run_supervised_substeps

    view = lane["fixture"].view
    run = lane["run"]
    request = SimpleNamespace(
        hard_id_encoding=lane["fold_features"].payload.hard_id_encoding,
        is_panel=False,
        supervised_bundle_key="B0_CORE",
        regime_promotion_decision_id=lane["eligible"].regime_promotion_decision_id,
        owner_decision_artifact_id=lane["owner"].owner_decision_artifact_id,
        comparison_classes_requested=("feature_only",),
    )
    context = SimpleNamespace(
        store_root=lane["root"],
        synthetic=True,
        wiring=SimpleNamespace(research_subject=SimpleNamespace(subject_id="b" * 64)),
        semantic=SimpleNamespace(
            payload=SimpleNamespace(regime_study=request, label_policy_id=lane["label_policy_id"])
        ),
        view=view,
        labeled=lane["labels"],
        folds=lane["folds"],
        label_artifact_id=lane["label_artifact_id"],
        research_model_run_ids=[],
        bundle_views={"B0_CORE": SimpleNamespace(bundle_feature_view_id=view.view_id)},
        bundle_frames={"B0_CORE": view.frame},
        state={"stages": {pipeline.QuantLabPipelineStage.S09_TRAIN_MODELS.value: {}}},
        regime={
            "execution": SimpleNamespace(
                protocol=lane["protocol"],
                run=run,
                verified_fit_assignments={
                    fit.fold_index: load_regime_fit_assignments(
                        lane["root"], fit.fit_envelope.regime_fit_id
                    )
                    for fit in run.fold_fits
                },
            ),
            "candidate_fold_set": lane["fold_set"],
            "regime_fold_set": lane["fold_set"],
            "schedule": lane["schedule"],
        },
    )
    outputs, record, _ = run_supervised_substeps(context)
    request_id = record["S09c"]["research_model_run_id"]
    assert request_id in outputs
    restored = load_research_run(lane["root"], request_id)
    for arm in (restored.baseline, restored.challenger):
        assert len(arm.rungs) == 3
        assert all(not rung.predictions.empty for rung in arm.rungs)
    monkeypatch.setattr(CatBoostClassifier, "fit", _no_fit)
    monkeypatch.setattr(Pipeline, "fit", _no_fit)
    replayed, replayed_record, _ = run_supervised_substeps(context)
    assert replayed == outputs
    assert replayed_record == record
