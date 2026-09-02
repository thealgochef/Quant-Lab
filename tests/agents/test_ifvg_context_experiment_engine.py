"""Leakage and immutability tests for the context-v1 experiment engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.artifact_io import (
    VerifiedIfvgArtifact,
    VerifiedIfvgPair,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ArtifactReference,
    ContextFeatureTier,
    IfvgContextExperimentConfig,
    IfvgContextExperimentDatasetConfig,
    IfvgContextExperimentResult,
    IfvgContextLabelConfig,
    PairedIfvgArtifactReference,
    ProfileCapabilityStatus,
    canonical_contract_sha256,
    context_run_identity,
    profile_capability,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_service import (
    run_and_catalog_context_experiment,
)
from alpha_lab.agents.data_infra.ifvg.context_feature_view import (
    apply_observation_filters,
    build_candidate_feature_view,
    features_for_tier,
    m3_cohort_status,
)
from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.context_labels import (
    derive_context_candidate_labels,
)
from alpha_lab.agents.data_infra.ifvg.context_model import categorical_features_for
from alpha_lab.agents.data_infra.ifvg.context_run_store import (
    ImmutableContextStoreError,
    list_context_run_catalog,
    load_candidate_feature_view_frame,
    load_context_experiment_run,
    save_candidate_feature_view,
    save_context_experiment_run,
)
from alpha_lab.agents.data_infra.ifvg.context_statistics import (
    binary_prediction_report,
    block_bootstrap_interval,
)

from .ifvg_v3_fixtures import context_fixture


def _references() -> PairedIfvgArtifactReference:
    return PairedIfvgArtifactReference(
        v2=ArtifactReference(
            artifact_id="1" * 64,
            manifest_payload_sha256="2" * 64,
            artifact_kind="v2",
            dataset_schema_version=2,
            profile_hash="profile",
        ),
        v3=ArtifactReference(
            artifact_id="3" * 64,
            manifest_payload_sha256="4" * 64,
            artifact_kind="v3",
            dataset_schema_version=4,
            profile_hash="profile",
            feature_formula_version="ifvg_context_formula_v2",
        ),
    )


def _pair() -> VerifiedIfvgPair:
    _day, context, core, _emissions = context_fixture()
    refs = _references()
    return VerifiedIfvgPair(
        reference=refs,
        v2=VerifiedIfvgArtifact(refs.v2, Path("."), {}, core, {}),
        v3=VerifiedIfvgArtifact(refs.v3, Path("."), {}, context, {}),
    )


def test_capability_and_tier_contracts_fail_closed() -> None:
    assert profile_capability(
        "ifvg_v2_doc_default_fresh_static_1r"
    ).status is ProfileCapabilityStatus.RUNNABLE
    assert profile_capability(
        "ifvg_v2_ict_clean_fresh_static_1r"
    ).reason == "same_leg_locality_sweep_semantics_unresolved"
    assert profile_capability(
        "any-profile", canonical_short_enabled=True
    ).reason == "canonical_short_profile_not_ratified"
    formula_v1 = _references().v3.model_copy(
        update={"feature_formula_version": "ifvg_context_formula_v1"},
    )
    with pytest.raises(ValueError, match="formula_v2"):
        IfvgContextExperimentConfig(
            dataset=IfvgContextExperimentDatasetConfig(
                artifact_pair=PairedIfvgArtifactReference(
                    v2=_references().v2,
                    v3=formula_v1,
                )
            ),
            feature_tier=ContextFeatureTier.M2,
        )


def test_exact_candidate_view_and_primary_240m_exclusion() -> None:
    view = build_candidate_feature_view(_pair())
    assert view.frame["candidate_id"].is_unique
    assert view.frame["entry_ts_utc"].notna().all()
    assert (
        pd.to_datetime(view.frame["entry_ts_utc"], utc=True)
        == pd.to_datetime(view.frame["feature_as_of_ts"], utc=True)
    ).all()
    assert view.m3_status == "descriptive_only_no_positive_qualification_coverage"
    primary = features_for_tier(ContextFeatureTier.M1_PRIMARY)
    experimental = features_for_tier(ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL)
    assert not any("14400s" in feature or "240m" in feature for feature in primary)
    assert any("14400s" in feature for feature in experimental)
    with pytest.raises(ValueError, match="newly prepared profile"):
        apply_observation_filters(view, {"enable_shorts": ("true",)})

    variation = pd.DataFrame({"ctx_sweep_qualifying_link_count": [0, 1]})
    assert m3_cohort_status(variation) == "model_eligible"
    assert (
        m3_cohort_status(variation.iloc[[1]])
        == "descriptive_only_no_positive_qualification_coverage"
    )


def test_m0_session_labels_are_registered_categorical_features() -> None:
    categoricals = categorical_features_for(features_for_tier(ContextFeatureTier.M0))

    assert "entry_session" in categoricals
    assert "in_engine_session" in categoricals
    assert "in_doc_session" in categoricals


def _label_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a"],
            "setup_id": ["setup-a"],
            "trading_day": ["2026-06-10"],
            "entry_ts_utc": [pd.Timestamp("2026-06-10T20:55:00Z")],
            "entry_ticks": [100],
            "proposed_stop_ticks": [96],
            "direction": ["long"],
        }
    )


def _label_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_id": ["entry", "one", "two", "cutoff", "protected"],
            "close_ts_utc": pd.to_datetime(
                [
                    "2026-06-10T20:55:00Z",
                    "2026-06-10T20:56:00Z",
                    "2026-06-10T20:57:00Z",
                    "2026-06-10T21:00:00Z",
                    "2026-06-11T14:00:00Z",
                ],
                utc=True,
            ),
            "high_ticks": [200, 103, 104, 200, 300],
            "low_ticks": [0, 99, 98, 0, 0],
        }
    )


def test_labels_are_strictly_post_entry_pre_cutoff_and_barrier_specific() -> None:
    r1 = derive_context_candidate_labels(
        _label_candidates(),
        _label_bars(),
        IfvgContextLabelConfig(reward_r=1.0),
    )
    row = r1.labels.iloc[0]
    assert row["resolution"] == "target"
    assert row["resolution_bar_id"] == "two"
    assert row["bars_after_entry_to_resolution"] == 2
    assert row["first_eligible_close_ts_utc"] == pd.Timestamp("2026-06-10T20:56:00Z")
    r2 = derive_context_candidate_labels(
        _label_candidates(),
        _label_bars(),
        IfvgContextLabelConfig(reward_r=2.0),
    )
    assert r2.derivation_id != r1.derivation_id
    assert r2.labels.iloc[0]["label"] == "censored"
    assert r2.labels.iloc[0]["censor_reason"] == "censored_protected_boundary"
    assert r2.labels.iloc[0]["scanned_bar_count"] == 2


def _fold_frame(days: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for index, day in enumerate(days):
        entry = pd.Timestamp(f"{day}T14:00:00Z")
        rows.append(
            {
                "candidate_id": f"c-{index:03d}",
                "setup_id": f"s-{index:03d}",
                "trading_day": day,
                "entry_ts_utc": entry,
                "resolution_ts_utc": entry + timedelta(minutes=5),
                "entry_available": True,
                "resolution_available": True,
                "binary_target": index % 2,
            }
        )
    # This training interval overlaps the first test interval and must purge.
    rows[10]["resolution_ts_utc"] = pd.Timestamp(f"{days[42]}T14:01:00Z")
    # One setup straddles the train/test boundary; both sides are excluded.
    rows[39]["setup_id"] = "boundary"
    rows[40]["setup_id"] = "boundary"
    return pd.DataFrame(rows)


def append_censored_row(frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Append one censored candidate row through a TYPED one-row frame.

    HARDENING-BACKEND section 4.5 (F-18): ``frame.loc[len(frame)] = row``
    with a NaT / ``None`` row went through the deprecated all-NA concat path
    (``FutureWarning``). The explicit form reproduces the old result exactly
    (``binary_target`` widens to ``object`` because an int column receives
    ``None``; ``resolution_ts_utc`` keeps its tz-aware datetime dtype with
    NaT) with every column and dtype stated up front.
    """

    values = row.to_dict()
    typed: dict[str, pd.Series] = {}
    for column in frame.columns:
        value = values[column]
        if value is None:
            typed[column] = pd.Series([None], dtype=object)
        else:
            typed[column] = pd.Series([value], dtype=frame[column].dtype)
    addition = pd.DataFrame(typed, columns=list(frame.columns))
    widened = frame.copy()
    for column in frame.columns:
        if addition[column].dtype == object and widened[column].dtype != object:
            widened[column] = widened[column].astype(object)
    return pd.concat([widened, addition], ignore_index=True, sort=False)


def test_folds_group_setup_then_purge_then_embargo_without_oos_dedup() -> None:
    days = tuple(
        (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)).date().isoformat()
        for index in range(50)
    )
    result = build_context_folds(
        _fold_frame(days),
        authorized_trading_days=days,
        minimum_train_candidates=30,
    )
    assert len(result.folds) == 2
    first = result.folds[0]
    assert first.valid
    assert first.excluded_boundary_setup_ids == ("boundary",)
    assert "c-010" in first.purged_candidate_ids
    assert set(first.embargoed_candidate_ids) == {"c-038"}
    test_rows = result.assignment[result.assignment["partition"] == "test"]
    assert test_rows["candidate_id"].is_unique


def test_fold_setup_boundary_is_detected_before_censoring_exclusion() -> None:
    days = tuple(
        (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)).date().isoformat()
        for index in range(45)
    )
    frame = _fold_frame(days)
    frame.loc[39, "setup_id"] = "censored-boundary"
    frame.loc[40, "setup_id"] = "ordinary-test"
    censored = frame.loc[40].copy()
    censored["candidate_id"] = "censored-test-sibling"
    censored["setup_id"] = "censored-boundary"
    censored["resolution_ts_utc"] = pd.NaT
    censored["resolution_available"] = False
    censored["binary_target"] = None
    frame = append_censored_row(frame, censored)
    result = build_context_folds(
        frame,
        authorized_trading_days=days,
        minimum_train_candidates=30,
    )
    assert "censored-boundary" in result.folds[0].excluded_boundary_setup_ids
    assert "c-039" not in result.folds[0].train_candidate_ids


def test_metrics_include_reference_reliability_and_null_small_cluster_interval() -> None:
    predictions = pd.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "probability": [0.1, 0.8, 0.4, 0.7],
            "training_prevalence": [0.5] * 4,
            "gross_r": [-1.0, 1.0, -1.0, 1.0],
            "net_r": [-1.0, 1.0, -1.0, 1.0],
        }
    )
    report = binary_prediction_report(predictions)
    assert report["auc"] is not None
    assert len(report["reliability_bins"]) == 10
    assert [item["threshold"] for item in report["thresholds"]] == [0.4, 0.5, 0.6, 0.7]
    interval = block_bootstrap_interval(
        pd.DataFrame({"setup_id": ["only"], "net_r": [1.0]}),
        cluster_column="setup_id",
        value_column="net_r",
    )
    assert interval["available"] is False
    assert interval["reason"] == "fewer_than_two_usable_clusters"


def test_run_store_is_full_id_immutable_and_verified(tmp_path: Path) -> None:
    config = IfvgContextExperimentConfig(
        dataset=IfvgContextExperimentDatasetConfig(artifact_pair=_references()),
        feature_tier=ContextFeatureTier.M0,
    )
    label_derivation_id = "c" * 64
    run_id = context_run_identity(
        config_hash=config.identity,
        view_id="b" * 64,
        label_derivation_id=label_derivation_id,
        folds=[],
        model_protocol_hash=None,
        predictions=[],
        status="insufficient_class_coverage",
    )
    result = IfvgContextExperimentResult(
        run_id=run_id,
        config_hash=config.identity,
        view_id="b" * 64,
        label_derivation_id=label_derivation_id,
        status="insufficient_class_coverage",
        folds=(),
        oos_row_ids=(),
        candidate_research_report={"surface": "candidate_research_counterfactual"},
        actual_execution_report={"surface": "actual_v2_execution_only"},
        feature_coverage_report={},
        reconciliation_audit_report={},
    )
    save_context_experiment_run(config, result, base_dir=tmp_path)
    loaded = load_context_experiment_run(result.run_id, base_dir=tmp_path)
    assert loaded.result == result
    with pytest.raises(FileExistsError, match="already exists"):
        save_context_experiment_run(config, result, base_dir=tmp_path)
    with pytest.raises(ImmutableContextStoreError, match="full SHA-256"):
        load_context_experiment_run("../outside", base_dir=tmp_path)


def test_candidate_view_store_recomputes_full_content_id(tmp_path: Path) -> None:
    view = build_candidate_feature_view(_pair())
    save_candidate_feature_view(view, base_dir=tmp_path)
    loaded, manifest = load_candidate_feature_view_frame(view.view_id, base_dir=tmp_path)
    assert loaded["candidate_id"].tolist() == view.frame["candidate_id"].tolist()
    assert manifest["view_id"] == view.view_id
    with pytest.raises(ImmutableContextStoreError, match="full SHA-256"):
        load_candidate_feature_view_frame("../outside", base_dir=tmp_path)


def _empty_context_result(
    config: IfvgContextExperimentConfig,
    *,
    view_id: str,
) -> IfvgContextExperimentResult:
    label_derivation_id = "c" * 64
    run_id = context_run_identity(
        config_hash=config.identity,
        view_id=view_id,
        label_derivation_id=label_derivation_id,
        folds=[],
        model_protocol_hash=None,
        predictions=[],
        status="insufficient_class_coverage",
    )
    return IfvgContextExperimentResult(
        run_id=run_id,
        config_hash=config.identity,
        view_id=view_id,
        label_derivation_id=label_derivation_id,
        status="insufficient_class_coverage",
        folds=(),
        oos_row_ids=(),
        candidate_research_report={
            "surface": "candidate_research_counterfactual",
            "fold_table": ({"fold_index": 0},),
        },
        actual_execution_report={"surface": "actual_v2_execution_only"},
        feature_coverage_report={},
        reconciliation_audit_report={},
    )


def _stub_context_execution(monkeypatch):
    from alpha_lab.agents.data_infra.ifvg import context_experiment_service as service

    pair = _pair()
    view = build_candidate_feature_view(pair)
    config = IfvgContextExperimentConfig(
        dataset=IfvgContextExperimentDatasetConfig(artifact_pair=_references()),
        feature_tier=ContextFeatureTier.M0,
    )
    result = _empty_context_result(config, view_id=view.view_id)
    execution = SimpleNamespace(view=view, result=result, model_run=None)
    monkeypatch.setattr(
        service,
        "load_verified_label_source_bars",
        lambda _artifact: pd.DataFrame(),
    )
    monkeypatch.setattr(
        service,
        "execute_context_experiment",
        lambda observed_pair, observed_config, bars: execution,
    )
    return pair, config, execution


def test_application_service_verifies_and_reuses_duplicate_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pair, config, execution = _stub_context_execution(monkeypatch)
    view_store = tmp_path / "views"
    run_store = tmp_path / "runs"
    catalog = tmp_path / "catalog.json"

    first = run_and_catalog_context_experiment(
        pair,
        config,
        display_name="Reference M0",
        view_store=view_store,
        run_store=run_store,
        catalog_path=catalog,
    )
    second = run_and_catalog_context_experiment(
        pair,
        config,
        view_store=view_store,
        run_store=run_store,
        catalog_path=catalog,
    )

    assert first.reused_view is False
    assert first.reused_run is False
    assert second.reused_view is True
    assert second.reused_run is True
    assert canonical_contract_sha256(second.stored_run.result) == (
        canonical_contract_sha256(execution.result)
    )
    assert len(second.run_manifest_sha256) == 64
    assert list_context_run_catalog(catalog_path=catalog) == [
        {
            "run_id": execution.result.run_id,
            "display_name": "Reference M0",
            "notes": None,
        }
    ]


def test_application_service_refuses_tampered_duplicate_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pair, config, execution = _stub_context_execution(monkeypatch)
    view_store = tmp_path / "views"
    run_store = tmp_path / "runs"
    catalog = tmp_path / "catalog.json"
    run_and_catalog_context_experiment(
        pair,
        config,
        view_store=view_store,
        run_store=run_store,
        catalog_path=catalog,
    )
    result_path = run_store / execution.result.run_id / "result.json"
    result_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ImmutableContextStoreError, match="modified"):
        run_and_catalog_context_experiment(
            pair,
            config,
            view_store=view_store,
            run_store=run_store,
            catalog_path=catalog,
        )
