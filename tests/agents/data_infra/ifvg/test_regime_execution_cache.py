"""Completed S09a cache tests; all fits use temporary seeded synthetic data."""

from dataclasses import replace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.ml.regime_executor import execute_regime_protocol
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
    persisted_candidate_source,
    persisted_panel_source,
)


def _execute(source, **overrides):
    options = {
        "protocol": source.protocol,
        "observation_source": source.source_ref,
        "fold_set_artifact_id": source.fold_set_envelope.fold_set_artifact_id,
        "bootstrap_refits": 2,
        "reuse_completed": True,
    }
    if getattr(source, "candidate_source_ref", None) is not None:
        options["candidate_as_of_source"] = source.candidate_source_ref
    return execute_regime_protocol(source.root, **{**options, **overrides})


def _no_fit(*args, **kwargs):
    raise AssertionError("completed regime request attempted fitting")


@pytest.mark.parametrize("grain", ["candidate", "panel"])
def test_completed_request_reloads_all_models_and_assignments_without_fit(
    tmp_path, monkeypatch, grain
):
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from alpha_lab.agents.data_infra.ifvg.ml import regime_executor
    from alpha_lab.agents.data_infra.ifvg.ml.regime_execution_cache import verify_cached_execution

    source = (persisted_candidate_source if grain == "candidate" else persisted_panel_source)(
        tmp_path
    )
    first = _execute(source)
    assert first.regime_fit_ids and not any(first.fits_reused)
    before = {path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(KMeans, "fit", _no_fit)
    monkeypatch.setattr(SimpleImputer, "fit", _no_fit)
    monkeypatch.setattr(StandardScaler, "fit", _no_fit)
    monkeypatch.setattr(regime_executor, "run_regime_protocol_from_source", _no_fit)
    # Request identity has no pipeline ID: a descriptive precursor and its
    # later feature-only study call this same numerical execution seam.
    restored = _execute(source)
    verified = verify_cached_execution(tmp_path, restored.regime_execution_request_id)
    assert verified.regime_execution_request_id == restored.regime_execution_request_id
    assert restored.regime_fit_ids == first.regime_fit_ids
    assert all(restored.fits_reused)
    assert restored.run.assessment == first.run.assessment
    assert restored.oos_assignment == first.oos_assignment
    pd.testing.assert_frame_equal(restored.run.assignments, first.run.assignments)
    pd.testing.assert_frame_equal(restored.oos_assignment_frame, first.oos_assignment_frame)
    for before_fit, after_fit in zip(first.run.fold_fits, restored.run.fold_fits, strict=True):
        assert (
            before_fit.preprocessing.parameter_payload == after_fit.preprocessing.parameter_payload
        )
        assert before_fit.preprocessing.training_row_ids == after_fit.preprocessing.training_row_ids
        assert before_fit.fit_envelope == after_fit.fit_envelope
    assert before == {
        path: path.stat().st_mtime_ns for path in tmp_path.rglob("*") if path.is_file()
    }


def test_zero_fit_completed_request_reloads_typed_failure(tmp_path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.ml import regime_executor

    source = persisted_candidate_source(tmp_path, n=80)
    first = _execute(source)
    assert not first.regime_fit_ids and not first.run.assessment.payload.gates_passed
    monkeypatch.setattr(regime_executor, "run_regime_protocol_from_source", _no_fit)
    restored = _execute(source)
    assert restored.run.assessment == first.run.assessment
    assert restored.oos_assignment == first.oos_assignment


def test_changed_bootstrap_or_source_never_reuses_completed_request(tmp_path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.ml import regime_executor

    source = persisted_candidate_source(tmp_path)
    _execute(source)
    monkeypatch.setattr(regime_executor, "run_regime_protocol_from_source", _no_fit)
    with pytest.raises(AssertionError, match="attempted fitting"):
        _execute(source, bootstrap_refits=3)
    # A source alteration is checked before cache lookup, never quietly reused.
    changed = replace(
        source, source_ref=source.source_ref.model_copy(update={"artifact_id": "f" * 64})
    )
    with pytest.raises(ValueError):
        _execute(changed)


def test_corrupt_cached_fit_refuses_before_any_model_deserialization(tmp_path, monkeypatch):
    import joblib

    source = persisted_candidate_source(tmp_path)
    result = _execute(source)
    path = next((tmp_path / "regime_fits" / result.regime_fit_ids[0]).glob("*.joblib"))
    path.write_bytes(path.read_bytes() + b"corrupt")
    monkeypatch.setattr(joblib, "load", _no_fit)
    with pytest.raises(ValueError, match="failed verification"):
        _execute(source)


def test_real_s09a_publishes_cache_id_and_s15_reloads_it_without_fit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.ml import regime_executor
    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import RegimeStudyRequest
    from alpha_lab.agents.data_infra.ifvg.search.pipeline_regime import (
        s09_regime_fit,
        s15_regime_reload_failures,
    )

    source = persisted_candidate_source(tmp_path)
    request = RegimeStudyRequest(
        input_feature_bundle_key="B0_CORE",
        resolved_input_features=source.protocol.payload.resolved_input_features,
        bootstrap_refits=2,
    )
    context = SimpleNamespace(
        store_root=tmp_path,
        wiring=SimpleNamespace(research_subject=object()),
        semantic=SimpleNamespace(payload=SimpleNamespace(regime_study=request)),
        regime={
            "protocol": source.protocol,
            "observation_ref": source.source_ref,
            "regime_fold_set": source.fold_set_envelope,
        },
    )
    outputs, record, _ = s09_regime_fit(context)
    request_id = record["S09a"]["regime_execution_request_id"]
    assert request_id in outputs
    monkeypatch.setattr(regime_executor, "run_regime_protocol_from_source", _no_fit)
    assert s15_regime_reload_failures(context) == {}
    replayed, replayed_record, _ = s09_regime_fit(context)
    assert replayed == outputs and replayed_record == record
