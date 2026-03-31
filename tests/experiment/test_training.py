"""Tests for Phase 5 — Walk-Forward CatBoost 3-Class Training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment.training import (
    CLASS_BLOWTHROUGH,
    CLASS_NAMES,
    CLASS_REVERSAL,
    CLASS_TRAP,
    DEFAULT_STEP_DAYS,
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    FoldResult,
    TrainingResult,
    WalkForwardFold,
    _compute_feature_stability,
    _compute_mae_distribution,
    _confusion_matrix_3class,
    _evaluate_verdict,
    create_walk_forward_folds,
    run_experiment,
    save_results,
    train_fold,
)

# ── Synthetic Data Helpers ────────────────────────────────────


def _make_synthetic_feature_matrix(
    n_events: int = 200,
    n_dates: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic feature matrix matching the real schema.

    55 numeric features + 3 string categoricals + 4 metadata columns.
    Label distribution approximately 58/22/20.
    """
    rng = np.random.RandomState(seed)

    # Distribute events across dates
    dates = pd.date_range("2025-11-17", periods=n_dates, freq="B")
    date_strs = [str(d.date()) for d in dates]
    event_dates = rng.choice(date_strs, size=n_events)
    event_dates.sort()

    # Timestamps within each date
    event_ts = []
    for d in event_dates:
        hour = rng.randint(9, 17)
        minute = rng.randint(0, 60)
        event_ts.append(pd.Timestamp(f"{d} {hour:02d}:{minute:02d}:00", tz="US/Eastern"))

    # Labels: ~58% reversal, ~22% trap, ~20% blowthrough
    labels_int = rng.choice([0, 1, 2], size=n_events, p=[0.58, 0.22, 0.20])
    label_names = [CLASS_NAMES[i] for i in labels_int]

    data: dict = {
        "event_ts": event_ts,
        "date": event_dates,
        "label": label_names,
        "label_encoded": labels_int,
    }

    # 55 numeric features (app_ prefix: 27, int_ prefix: 21, ctx_ numeric: 4, extra: 3)
    app_features = [
        "app_buy_volume", "app_sell_volume", "app_buy_sell_ratio",
        "app_large_trade_count", "app_large_trade_vol_pct", "app_aggression_trend",
        "app_total_trade_volume", "app_trade_count", "app_volume_acceleration",
        "app_avg_trade_size", "app_p90_trade_size",
        "app_avg_tob_imbalance", "app_avg_top5_depth", "app_avg_spread",
        "app_max_spread", "app_cancel_rate", "app_book_imbalance_trend",
        "app_avg_bid_depth_ratio", "app_depth_concentration",
        "app_price_change", "app_price_change_pct", "app_tick_direction_bias",
        "app_price_velocity_15m", "app_price_range",
        "app_volatility_full", "app_volatility_recent", "app_volatility_ratio",
    ]
    int_features = [
        "int_total_trade_volume", "int_trade_count", "int_volume_at_level",
        "int_volume_through_level", "int_absorption_ratio",
        "int_buy_sell_ratio", "int_aggression_flip", "int_large_trade_count",
        "int_large_trade_pct",
        "int_avg_tob_imbalance", "int_book_imbalance_shift", "int_avg_spread",
        "int_spread_widening", "int_avg_depth", "int_depth_change",
        "int_max_trade_size", "int_sweep_volume", "int_cancel_burst",
        "int_avg_trade_size", "int_p90_trade_size", "int_size_vs_approach",
        "int_time_within_2pts", "int_time_beyond_level", "int_deceleration_ratio",
    ]
    ctx_numeric = [
        "ctx_approach_from_above", "ctx_day_of_week", "ctx_hour",
        "ctx_time_normalized",
    ]
    for feat in app_features + int_features + ctx_numeric:
        data[feat] = rng.rand(n_events).astype(float)

    # 3 string categorical features
    data["ctx_direction"] = rng.choice(["LONG", "SHORT"], size=n_events)
    data["ctx_level_type"] = rng.choice(
        ["PDH", "PDL", "asia_high", "asia_low", "london_high", "london_low"],
        size=n_events,
    )
    data["ctx_session"] = rng.choice(
        ["Asia", "London", "Pre-market", "NY RTH", "Post-market"],
        size=n_events,
    )

    # max_mae for MAE distribution tests
    data["max_mae"] = rng.uniform(5, 50, size=n_events)

    return pd.DataFrame(data)


def _make_mock_fold_results(n_folds: int = 3, n_features: int = 20) -> list[FoldResult]:
    """Create mock FoldResult objects for testing stability/verdict."""
    rng = np.random.RandomState(123)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    results = []
    for i in range(n_folds):
        n_test = 20
        y_true = rng.choice([0, 1, 2], size=n_test, p=[0.58, 0.22, 0.20])
        y_pred = rng.choice([0, 1, 2], size=n_test, p=[0.58, 0.22, 0.20])
        cm = _confusion_matrix_3class(y_true, y_pred)
        importances = dict(zip(feature_names, rng.rand(n_features)))
        results.append(FoldResult(
            fold=i,
            n_train=100,
            n_test=n_test,
            train_dates_str="2025-11-17 -- 2025-12-31",
            test_dates_str="2026-01-02 -- 2026-01-06",
            accuracy=float(np.mean(y_true == y_pred)),
            reversal_precision=0.6,
            blowthrough_recall=0.5,
            confusion_matrix=cm,
            feature_importances=importances,
            y_true=y_true,
            y_pred=y_pred,
        ))
    return results


# ── TestWalkForwardFolds ──────────────────────────────────────


class TestWalkForwardFolds:
    def test_basic_fold_creation(self):
        """69 dates with 40/5/5 should produce approximately 5 folds."""
        df = _make_synthetic_feature_matrix(n_events=300, n_dates=69)
        folds = create_walk_forward_folds(df, train_days=40, test_days=5, step_days=5)
        # (69 - 40) / 5 = 5.8 → up to 5 full folds possible
        assert len(folds) >= 3
        assert len(folds) <= 7

    def test_no_data_leakage(self):
        """Test dates must be strictly after all train dates in every fold."""
        df = _make_synthetic_feature_matrix(n_events=200, n_dates=60)
        folds = create_walk_forward_folds(df, train_days=30, test_days=5, step_days=5)
        for fold in folds:
            max_train_date = max(fold.train_dates)
            min_test_date = min(fold.test_dates)
            assert max_train_date < min_test_date, (
                f"Fold {fold.fold}: train max {max_train_date} >= test min {min_test_date}"
            )

    def test_train_test_no_overlap(self):
        """Train and test indices must not overlap within any fold."""
        df = _make_synthetic_feature_matrix(n_events=200, n_dates=60)
        folds = create_walk_forward_folds(df, train_days=30, test_days=5, step_days=5)
        for fold in folds:
            overlap = set(fold.train_indices) & set(fold.test_indices)
            assert len(overlap) == 0, f"Fold {fold.fold}: {len(overlap)} overlapping indices"

    def test_minimum_events_expansion(self):
        """All folds should have at least MIN_TRAIN_EVENTS training events."""
        df = _make_synthetic_feature_matrix(n_events=200, n_dates=60)
        folds = create_walk_forward_folds(df, min_train_events=30)
        for fold in folds:
            assert len(fold.train_indices) >= 30, (
                f"Fold {fold.fold}: only {len(fold.train_indices)} train events"
            )

    def test_step_advancement(self):
        """Each fold's test window starts step_days after the previous fold."""
        df = _make_synthetic_feature_matrix(n_events=300, n_dates=70)
        folds = create_walk_forward_folds(df, train_days=40, test_days=5, step_days=5)
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                prev_test_start = folds[i - 1].test_dates[0]
                curr_test_start = folds[i].test_dates[0]
                assert curr_test_start > prev_test_start

    def test_empty_data(self):
        """Empty DataFrame produces zero folds."""
        df = pd.DataFrame(columns=["date", "label_encoded"])
        folds = create_walk_forward_folds(df)
        assert len(folds) == 0

    def test_too_few_dates_for_any_fold(self):
        """If fewer dates than train_days + test_days, no folds created."""
        df = _make_synthetic_feature_matrix(n_events=50, n_dates=10)
        folds = create_walk_forward_folds(df, train_days=40, test_days=5)
        assert len(folds) == 0


# ── TestConfusionMatrix ───────────────────────────────────────


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        """Perfect predictions yield diagonal matrix."""
        y = np.array([0, 0, 1, 1, 2, 2])
        cm = _confusion_matrix_3class(y, y)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 2
        assert cm.sum() == 6
        # Off-diagonals should be zero
        np.fill_diagonal(cm, 0)
        assert cm.sum() == 0

    def test_known_distribution(self):
        """Verify against hand-computed values."""
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = _confusion_matrix_3class(y_true, y_pred)
        # Row 0 (actual=0): predicted as [0, 1, 2] → [1, 1, 1]
        assert list(cm[0]) == [1, 1, 1]
        # Row 1 (actual=1): predicted as [0, 1, 2] → [1, 1, 0]
        assert list(cm[1]) == [1, 1, 0]
        # Row 2 (actual=2): predicted as [0, 1, 2] → [0, 0, 1]
        assert list(cm[2]) == [0, 0, 1]

    def test_sums_to_n(self):
        """Confusion matrix sums to number of predictions."""
        rng = np.random.RandomState(99)
        y_true = rng.choice([0, 1, 2], size=100)
        y_pred = rng.choice([0, 1, 2], size=100)
        cm = _confusion_matrix_3class(y_true, y_pred)
        assert cm.sum() == 100


# ── TestFeatureStability ──────────────────────────────────────


class TestFeatureStability:
    def test_in_all_folds_flag(self):
        """Features appearing in all folds' top 10 should be flagged."""
        results = _make_mock_fold_results(n_folds=3, n_features=20)
        stability = _compute_feature_stability(results, top_n=10)
        # in_all_folds should be True for features in top 10 of all 3 folds
        in_all = stability[stability["in_all_folds"]]
        for _, row in in_all.iterrows():
            assert row["folds_in_top_10"] == 3

    def test_sorted_by_importance(self):
        """Result should be sorted by mean_importance descending."""
        results = _make_mock_fold_results(n_folds=3)
        stability = _compute_feature_stability(results)
        importances = stability["mean_importance"].values
        assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))

    def test_has_required_columns(self):
        """Stability DataFrame should have all required columns."""
        results = _make_mock_fold_results(n_folds=2)
        stability = _compute_feature_stability(results)
        expected_cols = {"feature", "mean_importance", "std_importance",
                         "folds_in_top_10", "in_all_folds"}
        assert expected_cols.issubset(set(stability.columns))


# ── TestVerdict ───────────────────────────────────────────────


class TestVerdict:
    def test_all_pass(self):
        """When all metrics exceed thresholds, all checks pass."""
        verdict = _evaluate_verdict(
            accuracy=0.55, reversal_precision=0.65,
            blowthrough_recall=0.70, accuracy_variance=0.05,
        )
        for check in verdict.values():
            assert check["passed"] is True

    def test_accuracy_fail(self):
        verdict = _evaluate_verdict(
            accuracy=0.30, reversal_precision=0.65,
            blowthrough_recall=0.70, accuracy_variance=0.05,
        )
        assert verdict["Overall Accuracy > 40%"]["passed"] is False

    def test_reversal_precision_fail(self):
        verdict = _evaluate_verdict(
            accuracy=0.55, reversal_precision=0.40,
            blowthrough_recall=0.70, accuracy_variance=0.05,
        )
        assert verdict["Tradeable Reversal Precision > 50%"]["passed"] is False

    def test_blowthrough_recall_fail(self):
        verdict = _evaluate_verdict(
            accuracy=0.55, reversal_precision=0.65,
            blowthrough_recall=0.45, accuracy_variance=0.05,
        )
        assert verdict["Blow-through Recall > 60%"]["passed"] is False

    def test_variance_fail(self):
        verdict = _evaluate_verdict(
            accuracy=0.55, reversal_precision=0.65,
            blowthrough_recall=0.70, accuracy_variance=0.15,
        )
        assert verdict["Cross-fold Accuracy StdDev < 10%"]["passed"] is False


# ── TestMAEDistribution ──────────────────────────────────────


class TestMAEDistribution:
    def test_returns_series_for_true_positives(self):
        """Should return a Series of MAE values for correctly predicted reversals."""
        df = _make_synthetic_feature_matrix(n_events=100, n_dates=30)
        folds = create_walk_forward_folds(df, train_days=15, test_days=5, step_days=5)
        if not folds:
            pytest.skip("No folds created with synthetic data")

        # Create mock fold results with some true positive reversals
        fold_results = []
        for fold in folds:
            n_test = len(fold.test_indices)
            y_true = df.iloc[fold.test_indices]["label_encoded"].values
            # Predict everything as class 0 (reversal) to get some TPs
            y_pred = np.zeros(n_test, dtype=int)
            fold_results.append(FoldResult(
                fold=fold.fold, n_train=len(fold.train_indices), n_test=n_test,
                train_dates_str="", test_dates_str="",
                accuracy=0.0, reversal_precision=0.0, blowthrough_recall=0.0,
                confusion_matrix=np.zeros((3, 3), dtype=int),
                feature_importances={}, y_true=y_true, y_pred=y_pred,
            ))

        mae = _compute_mae_distribution(df, folds, fold_results)
        assert mae is not None
        assert len(mae) > 0
        assert mae.name == "mae_pts"

    def test_returns_none_when_no_max_mae(self):
        """When max_mae column is missing, returns None."""
        df = _make_synthetic_feature_matrix(n_events=50, n_dates=20)
        df = df.drop(columns=["max_mae"])
        folds = [WalkForwardFold(fold=0, train_dates=[], test_dates=[],
                                  train_indices=np.array([0, 1]),
                                  test_indices=np.array([2, 3]))]
        fr = FoldResult(
            fold=0, n_train=2, n_test=2, train_dates_str="", test_dates_str="",
            accuracy=0.0, reversal_precision=0.0, blowthrough_recall=0.0,
            confusion_matrix=np.zeros((3, 3), dtype=int),
            feature_importances={}, y_true=np.array([0, 0]), y_pred=np.array([0, 0]),
        )
        mae = _compute_mae_distribution(df, folds, [fr])
        assert mae is None


# ── TestSaveResults ───────────────────────────────────────────


class TestSaveResults:
    def test_saves_all_files(self, tmp_path):
        """Verify that save_results writes all expected files."""
        fold_results = _make_mock_fold_results(n_folds=2)
        stability = _compute_feature_stability(fold_results)
        verdict = _evaluate_verdict(0.5, 0.6, 0.7, 0.05)
        agg_cm = sum(fr.confusion_matrix for fr in fold_results)

        result = TrainingResult(
            fold_results=fold_results,
            aggregated_confusion=agg_cm,
            overall_accuracy=0.5,
            reversal_precision=0.6,
            blowthrough_recall=0.7,
            accuracy_variance=0.05,
            top_features_stability=stability,
            verdict=verdict,
            mae_distribution=pd.Series([10.0, 15.0, 20.0], name="mae_pts"),
        )

        save_results(result, tmp_path)

        assert (tmp_path / "fold_results.csv").exists()
        assert (tmp_path / "confusion_matrix.csv").exists()
        assert (tmp_path / "feature_stability.csv").exists()
        assert (tmp_path / "mae_distribution.csv").exists()
        assert (tmp_path / "summary.json").exists()

        # Verify JSON content
        import json
        with open(tmp_path / "summary.json") as f:
            summary = json.load(f)
        assert summary["overall_accuracy"] == 0.5
        assert summary["n_folds"] == 2
        assert "verdict" in summary


# ── Integration Tests ─────────────────────────────────────────


class TestTrainFoldIntegration:
    """Integration tests that actually train a CatBoost model."""

    def test_train_fold_synthetic(self):
        """Train on synthetic data, verify FoldResult structure."""
        df = _make_synthetic_feature_matrix(n_events=200, n_dates=60)
        df = df.sort_values(["date", "event_ts"]).reset_index(drop=True)

        folds = create_walk_forward_folds(df, train_days=30, test_days=5, step_days=5)
        assert len(folds) >= 1, "Need at least 1 fold for test"

        fold = folds[0]
        exclude = {"event_ts", "date", "label", "label_encoded", "max_mae"}
        feature_cols = [c for c in df.columns if c not in exclude]
        cat_features = ["ctx_direction", "ctx_level_type", "ctx_session"]

        model, result = train_fold(df, fold, feature_cols, cat_features)

        assert result.fold == 0
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.reversal_precision <= 1.0
        assert 0.0 <= result.blowthrough_recall <= 1.0
        assert result.confusion_matrix.shape == (3, 3)
        assert result.confusion_matrix.sum() == result.n_test
        assert len(result.feature_importances) == len(feature_cols)
        assert len(result.y_true) == result.n_test
        assert len(result.y_pred) == result.n_test

    def test_run_experiment_synthetic(self, tmp_path):
        """Full pipeline on synthetic data saved to tmp_path."""
        df = _make_synthetic_feature_matrix(n_events=200, n_dates=60)
        fm_path = tmp_path / "feature_matrix.parquet"
        df.drop(columns=["max_mae"]).to_parquet(fm_path)

        # Create matching labeled events
        labeled = df[["date", "label"]].copy()
        labeled["max_mae"] = df["max_mae"]
        labeled_path = tmp_path / "labeled_events.parquet"
        labeled.to_parquet(labeled_path)

        output_dir = tmp_path / "results"

        result = run_experiment(
            feature_matrix_path=fm_path,
            labeled_events_path=labeled_path,
            output_dir=output_dir,
            train_days=30,
            test_days=5,
            step_days=5,
        )

        assert isinstance(result, TrainingResult)
        assert len(result.fold_results) >= 1
        assert result.aggregated_confusion.shape == (3, 3)
        assert 0.0 <= result.overall_accuracy <= 1.0
        assert isinstance(result.verdict, dict)
        assert len(result.verdict) == 4

        # Verify files saved
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "fold_results.csv").exists()


_REAL_FM = Path("data/experiment/feature_matrix.parquet")
_REAL_LABELED = Path("data/experiment/labeled_events.parquet")


@pytest.mark.skipif(
    not _REAL_FM.exists() or not _REAL_LABELED.exists(),
    reason="Real data not available",
)
class TestRealDataIntegration:
    def test_real_data_smoke(self, tmp_path):
        """Run on real data, verify output structure."""
        result = run_experiment(
            feature_matrix_path=_REAL_FM,
            labeled_events_path=_REAL_LABELED,
            output_dir=tmp_path / "results",
        )

        assert isinstance(result, TrainingResult)
        assert len(result.fold_results) >= 3
        assert result.aggregated_confusion.shape == (3, 3)
        assert result.aggregated_confusion.sum() > 0
        assert isinstance(result.verdict, dict)
