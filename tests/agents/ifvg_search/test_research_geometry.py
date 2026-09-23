"""Fixed geometry paired worker wiring, with synthetic stage inputs only."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import LOGISTIC_PROTOCOL_ID
from alpha_lab.agents.data_infra.ifvg.ml.research_evidence import load_research_run
from alpha_lab.agents.data_infra.ifvg.search import pipeline
from alpha_lab.agents.data_infra.ifvg.search.research_artifacts import run_durable_supervised_stage
from alpha_lab.agents.data_infra.ifvg.search.research_geometry import (
    GEOMETRY_BUNDLES,
    build_geometry_comparison_report,
)
from alpha_lab.agents.data_infra.ifvg.search.research_runs import ResearchRequest
from tests.agents.ifvg_search.test_real_research_models import _context, _no_fit


def _request(**changes):
    return ResearchRequest.model_validate({
        "display_name": "fixed geometry",
        "evaluation_start": "2026-01-05",
        "evaluation_end": "2026-04-01",
        **changes,
    })


def test_geometry_request_is_fixed_and_historical_serialization_unchanged():
    historical = _request()
    assert "geometry_comparison" not in historical.model_dump(mode="json")
    assert "geometry_comparison" not in json.loads(historical.model_dump_json())
    assert historical.feature_bundle_ids == ("B0_CORE",)
    geometry = _request(geometry_comparison=True)
    assert geometry.model_dump(mode="json")["geometry_comparison"] is True
    assert geometry.feature_bundle_ids == GEOMETRY_BUNDLES
    with pytest.raises(ValueError, match="fixed non-MBP"):
        _request(geometry_comparison=True, mbp1_comparison=True)
    with pytest.raises(ValueError, match="fixed non-MBP"):
        _request(geometry_comparison=True, base_bundle_key="B1_CORE_STRUCTURE")


def test_geometry_s05_uses_child_preparation_and_persists_both_views(tmp_path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.features.geometry_core_atr14 import load_geometry_features
    from tests.agents.data_infra.ifvg.test_geometry_core_atr14 import _inputs

    view, bars, reference = _inputs(monkeypatch)
    context = SimpleNamespace(
        store_root=tmp_path,
        wiring=SimpleNamespace(
            candidate_view_source=lambda: view,
            research_subject=SimpleNamespace(subject_id="a" * 64),
            research_preparation=SimpleNamespace(bars_1m=bars, label_source_reference=reference),
        ),
        semantic=SimpleNamespace(payload=SimpleNamespace(
            feature_bundle_ids=GEOMETRY_BUNDLES, regime_study=None,
        )),
        bundle_views={}, bundle_frames={}, stage_sidecars={}, mbp1_evidence={},
    )
    outputs, _ = pipeline._stage_s05_feature_views(context)
    assert set(context.bundle_views) == set(GEOMETRY_BUNDLES)
    bound = context.bundle_views[GEOMETRY_BUNDLES[0]].payload.geometry_feature_artifact_id
    assert bound in outputs
    envelope, frame = load_geometry_features(tmp_path, bound)
    assert envelope.payload.view_id == view.view_id
    assert len(frame) == len(view.frame)
    assert context.bundle_views["B0_CORE"].payload.geometry_feature_artifact_id is None


def _geometry_context(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "_checkpoint", lambda context: None)
    context, _, _ = _context(tmp_path, "B0_CORE")
    challenger = GEOMETRY_BUNDLES[0]
    names = resolve_bundle(challenger).payload.resolved_feature_names
    frame = context.bundle_frames["B0_CORE"].copy()
    rng = np.random.default_rng(117)
    for name in names:
        if name not in frame:
            frame[name] = rng.normal(size=len(frame))
    # Typed missing values retain rows; the fixed fold-local preprocessing owns them.
    frame.loc[frame.index[::7], names[-1]] = np.nan
    context.bundle_frames[challenger] = frame
    context.bundle_views[challenger] = SimpleNamespace(
        payload=SimpleNamespace(resolved_feature_names=names),
        bundle_feature_view_id="a" * 64,
    )
    context.semantic.payload.feature_bundle_ids = GEOMETRY_BUNDLES
    return context


def test_geometry_worker_persists_pair_reloads_and_refuses_mismatched_rows(tmp_path, monkeypatch):
    from catboost import CatBoostClassifier
    from sklearn.pipeline import Pipeline

    context = _geometry_context(tmp_path, monkeypatch)
    outputs, _ = run_durable_supervised_stage(context)
    report = json.loads(context.stage_sidecars["geometry_comparison.json"])
    assert report["oos_row_count"] == 16
    assert report["parity_status"] == "held"
    ids = report["request_ids_by_bundle"]
    baseline = load_research_run(tmp_path, ids["B0_CORE"])
    challenger = load_research_run(tmp_path, ids[GEOMETRY_BUNDLES[0]])
    assert len(baseline.rungs) == len(challenger.rungs) == 3
    assert sum(len(rung.fitted_models) for rung in baseline.rungs + challenger.rungs) == 4
    assert len(context.research_model_run_ids) == 2
    for arm in (baseline, challenger):
        logistic = arm.rung(LOGISTIC_PROTOCOL_ID)
        assert logistic.fold_reports[0]["feature_schema"]["raw_ordered_features"]
    monkeypatch.setattr(CatBoostClassifier, "fit", _no_fit)
    monkeypatch.setattr(Pipeline, "fit", _no_fit)
    reloaded, note = run_durable_supervised_stage(context)
    assert reloaded == outputs
    assert "verified and reloaded" in note
    assert len(context.research_model_run_ids) == 2
    changed = challenger.rung(LOGISTIC_PROTOCOL_ID).predictions.copy()
    changed.loc[changed.index[0], "net_r"] += 1.0
    altered = replace(challenger.rung(LOGISTIC_PROTOCOL_ID), predictions=changed)
    with pytest.raises(AssertionError, match="cost identity"):
        build_geometry_comparison_report(
            baseline,
            replace(challenger, rungs=tuple(
                altered if rung.protocol_id == LOGISTIC_PROTOCOL_ID else rung
                for rung in challenger.rungs
            )),
        )


def test_geometry_completed_baseline_survives_failed_challenger(tmp_path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.ml import supervised_ladder

    context = _geometry_context(tmp_path, monkeypatch)
    original = supervised_ladder.run_supervised_ladder
    calls = []

    def fail_challenger(*args, **kwargs):
        calls.append(kwargs["bundle_ref"])
        if len(calls) == 2:
            raise RuntimeError("synthetic challenger interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(supervised_ladder, "run_supervised_ladder", fail_challenger)
    with pytest.raises(RuntimeError, match="challenger interruption"):
        run_durable_supervised_stage(context)
    stage = context.state["stages"][pipeline.QuantLabPipelineStage.S09_TRAIN_MODELS.value]
    baseline_id, challenger_id = stage["research_model_input_ids"]
    assert load_research_run(tmp_path, baseline_id) is not None
    assert load_research_run(tmp_path, challenger_id) is None
    calls.clear()

    def challenger_only(*args, **kwargs):
        assert kwargs["bundle_ref"] == resolve_bundle(
            GEOMETRY_BUNDLES[0]
        ).resolved_feature_bundle_id
        calls.append("challenger")
        return original(*args, **kwargs)

    monkeypatch.setattr(supervised_ladder, "run_supervised_ladder", challenger_only)
    run_durable_supervised_stage(context)
    assert calls == ["challenger"]
