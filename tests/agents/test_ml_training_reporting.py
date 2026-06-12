"""Regression tests for Streamlit ML training reporting/save safety."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ml.config import (
    DashboardUtilityConfig,
    MLPipelineConfig,
    SessionExperimentConfig,
    session_experiment_from_preset,
)
from alpha_lab.agents.data_infra.ml.model_evaluator import EvaluationResult
from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer


def _load_ml_training_tab():
    module_name = "_ml_training_tab_reporting_under_test"
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "ml_training_tab.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ml_training_tab = _load_ml_training_tab()


def test_utility_ui_bar_type_default_matches_strategy_core_v3_default():
    """The Streamlit dropdown should not drift from the v3 147t config default."""
    assert ml_training_tab._DASHBOARD_UTILITY_BAR_TYPES[0] == DashboardUtilityConfig().bar_type
    assert DashboardUtilityConfig().bar_type == "147t"


def test_session_experiment_defaults_and_presets_are_explicit():
    """Session experiments need separate train/eval/gate scopes, not loose UI flags."""
    default_scope = SessionExperimentConfig()

    assert default_scope.training_sessions == ["asia", "london", "ny"]
    assert default_scope.evaluation_sessions == ["asia", "london", "ny"]
    assert default_scope.production_gate_sessions == ["ny"]
    assert default_scope.report_session_breakdowns is True

    ny_only = session_experiment_from_preset("ny_only")
    assert ny_only.training_sessions == ["ny"]
    assert ny_only.evaluation_sessions == ["ny"]
    assert ny_only.production_gate_sessions == ["ny"]

    asia_london = session_experiment_from_preset("asia_london_only")
    assert asia_london.training_sessions == ["asia", "london"]
    assert asia_london.evaluation_sessions == ["asia", "london"]
    assert asia_london.production_gate_sessions == ["asia", "london"]


def test_session_experiment_config_rejects_unknown_or_empty_sessions():
    """Invalid session labels should fail before a long training run starts."""
    with pytest.raises(ValueError, match="Unsupported session"):
        SessionExperimentConfig(training_sessions=["ny", "europe"])

    with pytest.raises(ValueError, match="At least one session"):
        SessionExperimentConfig(evaluation_sessions=[])


def test_apply_session_experiment_scope_filters_to_train_or_eval_union():
    """Dataset build stays broad; training sees only the configured train/eval union."""
    dataset = pd.DataFrame(
        {
            "row_id": [0, 1, 2, 3, 4],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02 00:00:00Z",
                    "2026-01-02 01:00:00Z",
                    "2026-01-02 02:00:00Z",
                    "2026-01-02 03:00:00Z",
                    "2026-01-02 04:00:00Z",
                ],
            ),
        },
    )
    timestamps = pd.to_datetime(dataset["timestamp"])
    sessions = pd.Series(["asia", "london", "ny", "none", "unknown"])
    scope = SessionExperimentConfig(
        training_sessions=["ny"],
        evaluation_sessions=["london"],
        production_gate_sessions=["ny"],
    )

    scoped, scoped_ts, scoped_sessions, metadata = ml_training_tab.apply_session_experiment_scope(
        dataset,
        timestamps,
        sessions,
        scope,
    )

    assert scoped["row_id"].tolist() == [1, 2]
    expected_ts = timestamps.iloc[[1, 2]].reset_index(drop=True).tolist()
    assert scoped_ts.reset_index(drop=True).tolist() == expected_ts
    assert scoped_sessions.tolist() == ["london", "ny"]
    assert metadata["training_sessions"] == ["ny"]
    assert metadata["evaluation_sessions"] == ["london"]
    assert metadata["rows_before_session_filter"] == 5
    assert metadata["rows_after_session_filter"] == 2
    assert metadata["training_candidate_rows"] == 1
    assert metadata["evaluation_candidate_rows"] == 1


def test_session_series_derives_missing_cached_sessions_from_timestamp():
    """Old utility caches may contain a session column that is null; repair it."""
    dataset = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-07-01 14:00:00+00:00",  # NY session
                    "2025-07-01 07:30:00+00:00",  # London session
                    "2025-07-01 20:00:00+00:00",  # preserve explicit value
                ],
            ),
            "session": [None, np.nan, "ny"],
        },
    )

    sessions = ml_training_tab._session_series_for_dataset(
        dataset,
        pd.to_datetime(dataset["timestamp"]),
    )

    assert sessions.tolist() == ["ny", "london", "ny"]


def test_missing_tradeable_probability_is_zero_when_fold_model_lacks_class_zero():
    """A utility fold trained on classes 1/2 must not treat class-1 prob as class 0."""
    model = SimpleNamespace(classes_=np.array([1, 2]))
    raw_probs = np.array([[0.80, 0.20], [0.10, 0.90]])
    prob_by_class = ml_training_tab._probability_by_class(model, raw_probs)

    assert 0 not in prob_by_class
    prob_tradeable = ml_training_tab._probability_for_class(
        prob_by_class,
        class_label=0,
        n_rows=2,
        fallback=np.array([0.80, 0.10]),
    )

    assert prob_tradeable.tolist() == [0.0, 0.0]


def _failed_eval_result() -> EvaluationResult:
    return EvaluationResult(
        precision=0.40,
        recall=0.50,
        f1=0.44,
        accuracy=0.62,
        roc_auc=0.51,
        pr_auc=0.43,
        confusion_matrix={"tp": 4, "fp": 6, "tn": 90, "fn": 10},
        precision_ci=(0.20, 0.60),
        f1_ci=(0.21, 0.57),
        permutation_p_value=0.40,
        cohens_d=0.05,
        brier_score=0.40,
        n_samples=110,
        n_positive=14,
        n_negative=96,
        fold_metrics=[
            {"fold": 0, "precision": 0.25, "recall": 0.5, "f1": 0.33, "n_test": 55},
            {"fold": 1, "precision": 0.75, "recall": 0.5, "f1": 0.60, "n_test": 55},
        ],
    )


def test_save_trained_model_blocks_failed_quality_gates_by_default(monkeypatch, tmp_path):
    """Programmatic saves should hard-stop unless failed gates are explicitly allowed."""
    save_called = False

    def fake_save_model(*_args, **_kwargs):
        nonlocal save_called
        save_called = True

    monkeypatch.setattr(ExtremaModelTrainer, "save_model", staticmethod(fake_save_model))
    output_dir = tmp_path / "unsafe_model"

    with pytest.raises(ValueError, match="quality gate"):
        ml_training_tab.save_trained_model(
            trained_model=object(),
            eval_result=_failed_eval_result(),
            config=None,
            output_dir=output_dir,
            allow_failed_gates=False,
        )

    assert save_called is False
    assert not output_dir.exists()


def test_save_override_persists_quality_gates_confidence_and_purge_metadata(
    monkeypatch,
    tmp_path,
):
    """An explicit override may save, but evaluation.json must record why it was risky."""

    def fake_save_model(_trained_model, model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        (Path(model_dir) / "model.cbm").write_text("fake model")

    monkeypatch.setattr(ExtremaModelTrainer, "save_model", staticmethod(fake_save_model))
    output_dir = tmp_path / "override_model"

    saved = ml_training_tab.save_trained_model(
        trained_model=object(),
        eval_result=_failed_eval_result(),
        config=MLPipelineConfig(training_mode="dashboard_utility"),
        output_dir=output_dir,
        training_result={
            # W2 P3a/P3b: saves refuse to omit the OOS frame; empty still writes.
            "oos_predictions": ml_training_tab.build_oos_predictions_frame([]),
            "confidence_threshold_stats": {
                "0.70": {
                    "threshold": 0.70,
                    "coverage": 0.25,
                    "precision": 0.80,
                    "trade_count": 5,
                },
            },
            "label_purge": {
                "method": "label_window_end",
                "used_label_window_end": True,
                "n_purged_total": 3,
                "folds": [{"fold": 0, "n_purged": 3}],
            },
        },
        allow_failed_gates=True,
    )

    assert saved == output_dir
    payload = json.loads((output_dir / "evaluation.json").read_text())
    json.dumps(payload["quality_gates"])

    assert payload["quality_gates"]["all_passed"] is False
    assert payload["quality_gates"]["allow_failed_gates"] is True
    precision_gate = payload["quality_gates"]["gates"]["Precision >= 0.55"]
    assert isinstance(precision_gate["passed"], bool)
    assert isinstance(precision_gate["value"], float)
    assert isinstance(precision_gate["threshold"], float)

    threshold_070 = payload["confidence_threshold_stats"]["0.70"]
    assert threshold_070 == {
        "threshold": 0.70,
        "coverage": 0.25,
        "precision": 0.80,
        "trade_count": 5,
    }
    assert payload["label_purge"]["method"] == "label_window_end"
    assert payload["label_purge"]["used_label_window_end"] is True
    assert payload["label_purge"]["n_purged_total"] == 3

    # W2 P3a/P3b: the empty OOS frame still writes with its schema; the bundle
    # carries the strategy contract and the binary-integrity sidecar.
    oos = pd.read_parquet(output_dir / "oos_predictions.parquet")
    assert len(oos) == 0
    assert "prob_tradeable_reversal" in oos.columns
    assert payload["oos_predictions_rows"] == 0
    assert (output_dir / "strategy.json").exists()
    sidecar = (output_dir / "model.cbm.sha256").read_text(encoding="utf-8").split()[0]
    import hashlib

    assert sidecar == hashlib.sha256((output_dir / "model.cbm").read_bytes()).hexdigest()


def test_compute_confidence_threshold_stats_reports_threshold_070_trade_count():
    """The 0.70 confidence gate needs coverage, precision, and actionable trade count."""
    fold_predictions = [
        {
            "fold": 0,
            "y_true": np.array([1, 0, 1, 1, 0]),
            "y_pred": np.array([1, 1, 1, 0, 0]),
            "y_prob": np.array([0.95, 0.80, 0.72, 0.69, 0.20]),
        },
    ]

    stats = ml_training_tab.compute_confidence_threshold_stats(
        fold_predictions,
        thresholds=[0.70],
    )

    assert set(stats) == {"0.70"}
    threshold_070 = stats["0.70"]
    assert threshold_070["threshold"] == 0.70
    assert threshold_070["trade_count"] == 3
    assert threshold_070["coverage"] == pytest.approx(3 / 5)
    assert threshold_070["precision"] == pytest.approx(2 / 3)


def test_exact_label_window_end_purge_beats_forward_window_timestamp_heuristic():
    """When labels expose exact horizon ends, purge rows whose window reaches test."""
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2026-01-02 10:00:00Z",
                "2026-01-02 10:01:00Z",
                "2026-01-02 10:02:00Z",
            ],
        ),
    )
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            # Row 2 would pass the old timestamp-only heuristic but leaks into test.
            "label_window_end": pd.to_datetime(
                [
                    "2026-01-02 10:03:00Z",
                    "2026-01-02 10:04:00Z",
                    "2026-01-02 10:15:00Z",
                ],
            ),
        },
    )
    config = SimpleNamespace(labeling=SimpleNamespace(forward_window=1))

    purged_idx, metadata = ml_training_tab.purge_training_indices_for_label_leakage(
        train_indices=np.array([0, 1, 2]),
        timestamps=timestamps,
        dataset=dataset,
        test_start=pd.Timestamp("2026-01-02 10:10:00Z"),
        config=config,
    )

    np.testing.assert_array_equal(purged_idx, np.array([0, 1]))
    assert metadata["method"] == "label_window_end"
    assert metadata["used_label_window_end"] is True
    assert metadata["n_purged"] == 1
    assert metadata["purge_rule"] == "label_window_end < test_start"


def test_utility_purge_derives_exact_label_window_end_for_cached_rows_without_column():
    """Old utility caches lacking label_window_end should still use the v3 exact cutoff."""
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2026-01-01 15:00:00-05:00",
                "2026-01-02 09:30:00-05:00",
                "2026-01-02 10:00:00-05:00",
            ],
        ),
    )
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            "date": ["2026-01-01", "2026-01-02", "2026-01-02"],
        },
    )
    config = SimpleNamespace(
        training_mode="dashboard_utility",
        labeling=SimpleNamespace(forward_window=1),
    )

    purged_idx, metadata = ml_training_tab.purge_training_indices_for_label_leakage(
        train_indices=np.array([0, 1, 2]),
        timestamps=timestamps,
        dataset=dataset,
        test_start=pd.Timestamp("2026-01-02 10:10:00-05:00"),
        config=config,
    )

    np.testing.assert_array_equal(purged_idx, np.array([0]))
    assert metadata["method"] == "dashboard_utility_label_window_end"
    assert metadata["used_label_window_end"] is True
    assert metadata["n_purged"] == 2
    assert metadata["purge_rule"] == "derived_label_window_end < test_start"


def test_production_gate_report_uses_session_and_confidence_not_argmax():
    """The primary utility report must model the runtime gate, not aggregate argmax."""
    fold_predictions = [
        {
            "fold": 0,
            "y_true": np.array([1, 0, 0, 1, 1]),
            "y_pred": np.array([1, 1, 1, 0, 1]),
            "y_prob": np.array([0.80, 0.95, 0.90, 0.65, 0.72]),
            "session": np.array(["ny", "london", "ny", "ny", "ny"]),
        },
    ]

    report = ml_training_tab.compute_production_gate_report(
        fold_predictions,
        confidence_gate=0.70,
        eligible_session="ny",
        tp_points=15.0,
        sl_points=30.0,
    )

    assert report["confidence_gate"] == 0.70
    assert report["eligible_session"] == "ny"
    assert report["eligible_sessions"] == ["ny"]
    assert report["n_samples"] == 5
    assert report["session_eligible_count"] == 4
    # The london 0.95 is intentionally excluded; selected rows are 0, 2, and 4.
    assert report["trade_count"] == 3
    assert report["coverage"] == pytest.approx(3 / 5)
    assert report["eligible_coverage"] == pytest.approx(3 / 4)
    assert report["tp"] == 2
    assert report["fp"] == 1
    assert report["precision"] == pytest.approx(2 / 3)
    assert report["expectancy_15_30_pts"] == pytest.approx(0.0)


def test_production_gate_report_accepts_multiple_eligible_sessions():
    """Research gates can evaluate Asia/London/NY independently or together."""
    fold_predictions = [
        {
            "fold": 0,
            "y_true": np.array([1, 0, 1, 0]),
            "y_pred": np.array([1, 1, 1, 1]),
            "y_prob": np.array([0.80, 0.95, 0.90, 0.60]),
            "session": np.array(["ny", "london", "asia", "ny"]),
        },
    ]

    report = ml_training_tab.compute_production_gate_report(
        fold_predictions,
        confidence_gate=0.70,
        eligible_sessions=["ny", "london"],
    )

    assert report["eligible_sessions"] == ["ny", "london"]
    assert report["session_eligible_count"] == 3
    # Includes the NY positive and London negative; excludes Asia despite 0.90 prob.
    assert report["trade_count"] == 2
    assert report["tp"] == 1
    assert report["fp"] == 1


def test_session_metrics_report_gate_quality_by_runtime_session():
    """Aggregate OOS is not enough; report must expose the NY/runtime subset."""
    fold_predictions = [
        {
            "fold": 0,
            "y_true": np.array([1, 0, 1, 0, 1, 0]),
            "y_pred": np.array([1, 0, 1, 1, 1, 0]),
            "y_prob": np.array([0.80, 0.20, 0.76, 0.74, 0.68, 0.91]),
            "session": np.array(["ny", "ny", "asia", "ny", "london", "none"]),
        },
    ]

    session_metrics = ml_training_tab.compute_session_metrics(
        fold_predictions,
        confidence_gate=0.70,
    )

    assert set(session_metrics) >= {"all", "ny", "non_ny", "asia", "london", "none"}
    assert session_metrics["all"]["n_samples"] == 6
    assert session_metrics["ny"]["n_samples"] == 3
    assert session_metrics["ny"]["trade_count"] == 2
    assert session_metrics["ny"]["tp"] == 1
    assert session_metrics["ny"]["fp"] == 1
    assert session_metrics["ny"]["precision"] == pytest.approx(0.5)
    assert session_metrics["non_ny"]["n_samples"] == 3


def test_save_persists_gated_session_and_row_level_oos_artifacts(monkeypatch, tmp_path):
    """Saved bundles need audit artifacts for threshold/session analysis after training."""

    def fake_save_model(_trained_model, model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        (Path(model_dir) / "model.cbm").write_text("fake model")

    monkeypatch.setattr(ExtremaModelTrainer, "save_model", staticmethod(fake_save_model))
    output_dir = tmp_path / "auditable_model"
    oos_predictions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-02 10:00:00Z"]),
            "session": ["ny"],
            "label": ["tradeable_reversal"],
            "prob_tradeable_reversal": [0.81],
            "gate_0_70_ny": [True],
        },
    )

    saved = ml_training_tab.save_trained_model(
        trained_model=object(),
        eval_result=_failed_eval_result(),
        config=MLPipelineConfig(training_mode="dashboard_utility"),
        output_dir=output_dir,
        training_result={
            "gated_oos": {"trade_count": 1, "precision": 1.0},
            "session_metrics": {"ny": {"n_samples": 1, "precision": 1.0}},
            "oos_three_class_balance": {
                "tradeable_reversal": 1,
                "trap_reversal": 0,
                "aggressive_blowthrough": 0,
            },
            "oos_predictions": oos_predictions,
        },
        allow_failed_gates=True,
    )

    assert saved == output_dir
    payload = json.loads((output_dir / "evaluation.json").read_text())
    assert payload["gated_oos"]["trade_count"] == 1
    assert payload["session_metrics"]["ny"]["n_samples"] == 1
    assert payload["oos_three_class_balance"]["tradeable_reversal"] == 1
    assert payload["oos_predictions_file"] == "oos_predictions.parquet"
    assert (output_dir / "oos_predictions.parquet").exists()


def test_save_persists_session_experiment_to_evaluation_and_metadata(monkeypatch, tmp_path):
    """Session-scope choices must be auditable after a model bundle is saved."""

    def fake_save_model(_trained_model, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model.cbm").write_text("fake model")
        (model_dir / "metadata.json").write_text(json.dumps({"selected_features": []}))

    monkeypatch.setattr(ExtremaModelTrainer, "save_model", staticmethod(fake_save_model))
    output_dir = tmp_path / "ny_only_model"
    config = MLPipelineConfig(
        training_mode="dashboard_utility",
        session_experiment=SessionExperimentConfig(
            training_sessions=["ny"],
            evaluation_sessions=["ny"],
            production_gate_sessions=["ny"],
        ),
    )

    saved = ml_training_tab.save_trained_model(
        trained_model=object(),
        eval_result=_failed_eval_result(),
        config=config,
        output_dir=output_dir,
        training_result={
            "oos_predictions": ml_training_tab.build_oos_predictions_frame([]),
            "session_experiment": config.session_experiment.model_dump(),
            "session_filter": {
                "training_sessions": ["ny"],
                "evaluation_sessions": ["ny"],
                "production_gate_sessions": ["ny"],
                "training_candidate_rows": 10,
                "evaluation_candidate_rows": 4,
            },
        },
        allow_failed_gates=True,
    )

    assert saved == output_dir
    evaluation = json.loads((output_dir / "evaluation.json").read_text())
    metadata = json.loads((output_dir / "metadata.json").read_text())

    assert evaluation["session_experiment"]["training_sessions"] == ["ny"]
    assert evaluation["session_filter"]["training_candidate_rows"] == 10
    assert evaluation["full_config"]["session_experiment"]["evaluation_sessions"] == ["ny"]
    assert metadata["session_experiment"]["production_gate_sessions"] == ["ny"]
    assert metadata["pipeline_config"]["session_experiment"]["training_sessions"] == ["ny"]


def test_save_refuses_missing_oos_frame_training_result_and_config(monkeypatch, tmp_path):
    """W2 P3a/P3b: a bundle without its OOS evidence, training result, or
    strategy contract is a partial bundle — the save fails loudly."""

    def fake_save_model(_trained_model, model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        (Path(model_dir) / "model.cbm").write_text("fake model")

    monkeypatch.setattr(ExtremaModelTrainer, "save_model", staticmethod(fake_save_model))
    config = MLPipelineConfig(training_mode="dashboard_utility")

    with pytest.raises(ValueError, match="oos_predictions"):
        ml_training_tab.save_trained_model(
            trained_model=object(),
            eval_result=_failed_eval_result(),
            config=config,
            output_dir=tmp_path / "no_oos",
            training_result={},
            allow_failed_gates=True,
        )

    with pytest.raises(ValueError, match="training_result"):
        ml_training_tab.save_trained_model(
            trained_model=object(),
            eval_result=_failed_eval_result(),
            config=config,
            output_dir=tmp_path / "no_result",
            training_result=None,
            allow_failed_gates=True,
        )

    with pytest.raises(ValueError, match="config"):
        ml_training_tab.save_trained_model(
            trained_model=object(),
            eval_result=_failed_eval_result(),
            config=None,
            output_dir=tmp_path / "no_config",
            training_result={"oos_predictions": ml_training_tab.build_oos_predictions_frame([])},
            allow_failed_gates=True,
        )


def test_build_oos_predictions_frame_empty_carries_canonical_schema():
    frame = ml_training_tab.build_oos_predictions_frame([])
    assert frame.empty
    assert list(frame.columns) == [
        "fold",
        "timestamp",
        "session",
        "binary_true_tradeable",
        "label_encoded",
        "label",
        "pred_label_encoded",
        "pred_label",
        "prob_tradeable_reversal",
        "gate_0_70_runtime_sessions",
        "gate_0_70_ny",
    ]


# ── W3a: purged trading-day folds + pinned features ─────────────────────────


def _day_fold_frame(n_dates: int, rows_per_day: int) -> pd.DataFrame:
    dates = [
        str(d.date())
        for d in pd.bdate_range("2026-01-01", periods=n_dates, freq="B")
    ]
    rows = []
    for date_str in dates:
        for i in range(rows_per_day):
            rows.append(
                {
                    "date": date_str,
                    "timestamp": pd.Timestamp(f"{date_str} 14:{i:02d}:00", tz="UTC"),
                    "session": "ny",
                    "label_encoded": i % 3,
                    "int_time_within_2pts": float((i * 7) % 60),
                    "int_absorption_ratio": float((i % 10) / 10.0),
                    "app_max_spread": 0.25 + 0.25 * (i % 4),
                },
            )
    return pd.DataFrame(rows)


def test_purged_trading_day_splits_match_train_dashboard_model_scheme():
    """40/5/5/2 over 52 trading days = 2 folds with a 2-trading-day purge gap."""
    frame = _day_fold_frame(n_dates=52, rows_per_day=2)
    splits = ml_training_tab._purged_trading_day_splits(
        frame,
        train_days=40,
        test_days=5,
        step_days=5,
        purge_days=2,
        min_train_events=30,
    )
    dates = sorted(frame["date"].unique())

    assert [s.fold for s in splits] == [0, 1]

    fold0 = splits[0]
    assert str(fold0.train_start.date()) == dates[0]
    assert str(fold0.train_end.date()) == dates[39]
    assert str(fold0.test_start.date()) == dates[42]
    assert str(fold0.test_end.date()) == dates[46]
    train_dates_0 = set(frame.iloc[fold0.train_indices]["date"])
    test_dates_0 = set(frame.iloc[fold0.test_indices]["date"])
    assert train_dates_0 == set(dates[0:40])
    assert test_dates_0 == set(dates[42:47])
    # The purge gap days belong to NEITHER side.
    assert dates[40] not in train_dates_0 | test_dates_0
    assert dates[41] not in train_dates_0 | test_dates_0

    fold1 = splits[1]
    assert str(fold1.train_start.date()) == dates[5]
    assert str(fold1.test_end.date()) == dates[51]


def test_purged_trading_day_splits_min_train_events_skips_thin_folds():
    frame = _day_fold_frame(n_dates=6, rows_per_day=1)
    splits = ml_training_tab._purged_trading_day_splits(
        frame,
        train_days=2,
        test_days=1,
        step_days=1,
        purge_days=1,
        min_train_events=3,  # 2 train days x 1 row < 3 -> every fold skipped
    )
    assert splits == []


def test_run_walk_forward_training_day_folds_and_pinned_features():
    """The day-fold scheme + pinned list flow through training to the result."""
    frame = _day_fold_frame(n_dates=14, rows_per_day=12)
    config = MLPipelineConfig(
        training_mode="dashboard_utility",
        model={"rfecv_enabled": False, "iterations": 25, "depth": 3,
               "loss_function": "MultiClass"},
        tick_size=0.25,
        instrument="NQ",
    )
    pinned = ["int_time_within_2pts", "app_max_spread"]
    day_folds = {
        "train_days": 4,
        "test_days": 2,
        "step_days": 2,
        "purge_days": 1,
        "min_train_events": 10,
    }

    result = ml_training_tab.run_walk_forward_training(
        frame,
        config,
        "label_encoded",
        day_folds=day_folds,
        pinned_features=pinned,
    )

    assert result["selected_features"] == pinned
    assert result["n_valid_folds"] >= 2
    # Train/test windows in the fold details follow the TRADING-day scheme:
    # 4 train days, a 1-day purge gap, 2 test days.
    dates = sorted(frame["date"].unique())
    first = result["fold_details"][0]
    assert first["train_start"] == dates[0]
    assert first["train_end"] == dates[3]
    assert first["test_start"] == dates[5]
    assert first["test_end"] == dates[6]
    # The pinned list is what the refit model was trained on.
    assert result["trained_model"].selected_features == pinned
    assert not result["oos_predictions"].empty


def test_run_walk_forward_training_rejects_unknown_pinned_feature():
    frame = _day_fold_frame(n_dates=14, rows_per_day=12)
    config = MLPipelineConfig(
        training_mode="dashboard_utility",
        model={"rfecv_enabled": False, "iterations": 25, "depth": 3,
               "loss_function": "MultiClass"},
        tick_size=0.25,
        instrument="NQ",
    )
    with pytest.raises(ValueError, match="pinned_features not in dataset"):
        ml_training_tab.run_walk_forward_training(
            frame,
            config,
            "label_encoded",
            day_folds={
                "train_days": 4,
                "test_days": 2,
                "step_days": 2,
                "purge_days": 1,
                "min_train_events": 10,
            },
            pinned_features=["app_avg_tob_imbalance"],
        )
