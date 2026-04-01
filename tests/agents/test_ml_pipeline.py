"""Tests for ML pipeline Phase 3: walk-forward, model training, and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import (
    ModelConfig,
    WalkForwardConfig,
)
from alpha_lab.agents.data_infra.ml.model_evaluator import (
    EvaluationResult,
    ModelEvaluator,
)
from alpha_lab.agents.data_infra.ml.model_trainer import (
    ExtremaModelTrainer,
    TrainedModel,
)
from alpha_lab.agents.data_infra.ml.walk_forward import (
    WalkForwardSplitter,
)

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────


def _make_classification_data(
    n: int = 500,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic classification data with timestamps.

    Returns (X, y, timestamps) where X has `n_features` columns,
    y is binary (0/1), and timestamps span a multi-month range.
    """
    rng = np.random.default_rng(seed)

    # Features: some informative, some noise
    x_data = rng.standard_normal((n, n_features))

    # Labels based on first 3 features (the rest are noise)
    logits = 0.5 * x_data[:, 0] + 0.3 * x_data[:, 1] - 0.4 * x_data[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n) < probs).astype(int)

    features = pd.DataFrame(
        x_data,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y_series = pd.Series(y, name="label")

    # Timestamps spanning 120 days
    timestamps = pd.Series(
        pd.date_range("2026-01-01", periods=n, freq="6h"),
    )

    return features, y_series, timestamps


# ────────────────────────────────────────────────────────────────
# Walk-Forward Splitter
# ────────────────────────────────────────────────────────────────


class TestWalkForwardSplitter:
    """Test walk-forward cross-validation splits."""

    def test_rolling_splits(self):
        """Rolling window should produce non-overlapping test periods."""
        _, _, timestamps = _make_classification_data(n=500)
        config = WalkForwardConfig(train_days=30, test_days=10, gap_days=1)
        splitter = WalkForwardSplitter(config)
        splits = splitter.split(timestamps)

        assert len(splits) > 0

        for split in splits:
            assert len(split.train_indices) > 0
            assert len(split.test_indices) > 0
            # Train should come before test
            assert split.train_end < split.test_start
            # Gap enforcement
            gap = split.test_start - split.train_end
            assert gap.total_seconds() > 0

    def test_no_train_test_overlap(self):
        """Train and test indices should never overlap."""
        _, _, timestamps = _make_classification_data(n=500)
        config = WalkForwardConfig(train_days=30, test_days=10)
        splitter = WalkForwardSplitter(config)
        splits = splitter.split(timestamps)

        for split in splits:
            overlap = set(split.train_indices) & set(split.test_indices)
            assert len(overlap) == 0, f"Fold {split.fold} has overlapping indices"

    def test_expanding_window(self):
        """Expanding window: train_start stays fixed, window grows."""
        _, _, timestamps = _make_classification_data(n=500)
        config = WalkForwardConfig(
            train_days=30, test_days=10, gap_days=1, expanding=True,
        )
        splitter = WalkForwardSplitter(config)
        splits = splitter.split(timestamps)

        assert len(splits) > 0

        # All folds should have same train_start (or very close)
        starts = [s.train_start for s in splits]
        for s in starts:
            assert abs((s - starts[0]).total_seconds()) < 1

        # Training set should grow
        if len(splits) >= 2:
            assert len(splits[1].train_indices) >= len(splits[0].train_indices)

    def test_sklearn_cv_format(self):
        """as_sklearn_cv should return list of (train, test) tuples."""
        _, _, timestamps = _make_classification_data(n=500)
        config = WalkForwardConfig(train_days=30, test_days=10)
        splitter = WalkForwardSplitter(config)
        cv = splitter.as_sklearn_cv(timestamps)

        assert isinstance(cv, list)
        for train_idx, test_idx in cv:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_empty_timestamps(self):
        config = WalkForwardConfig()
        splitter = WalkForwardSplitter(config)
        splits = splitter.split(pd.Series(dtype="datetime64[ns]"))
        assert splits == []

    def test_get_n_splits(self):
        _, _, timestamps = _make_classification_data(n=500)
        config = WalkForwardConfig(train_days=30, test_days=10)
        splitter = WalkForwardSplitter(config)
        n = splitter.get_n_splits(timestamps)
        assert n > 0
        assert n == len(splitter.split(timestamps))

    def test_gap_prevents_leakage(self):
        """With gap_days=5, no test sample should be within 5 days of train."""
        _, _, timestamps = _make_classification_data(n=500)
        config = WalkForwardConfig(train_days=30, test_days=10, gap_days=5)
        splitter = WalkForwardSplitter(config)
        splits = splitter.split(timestamps)

        ts_vals = pd.to_datetime(timestamps)
        for split in splits:
            max_train_ts = ts_vals.iloc[split.train_indices].max()
            min_test_ts = ts_vals.iloc[split.test_indices].min()
            gap = (min_test_ts - max_train_ts).total_seconds() / 86400
            assert gap >= 4.5  # ~5 days gap


# ────────────────────────────────────────────────────────────────
# Model Trainer
# ────────────────────────────────────────────────────────────────


class TestModelTrainer:
    """Test CatBoost model training."""

    def test_basic_training(self):
        """Train a model on synthetic data."""
        feats, y, _ = _make_classification_data(n=200, n_features=5)
        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)

        result = trainer.train(feats, y)
        assert isinstance(result, TrainedModel)
        assert result.model is not None
        assert len(result.selected_features) == 5
        assert len(result.feature_importances) == 5
        assert "train_accuracy" in result.train_metrics
        assert 0.0 <= result.train_metrics["train_accuracy"] <= 1.0

    def test_predictions(self):
        """Trained model should produce valid predictions."""
        feats, y, _ = _make_classification_data(n=200, n_features=5)
        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)

        result = trainer.train(feats, y)
        preds = result.model.predict(feats[result.selected_features])
        preds_flat = preds.flatten()

        # All predictions should be 0 or 1
        assert set(np.unique(preds_flat)).issubset({0, 1})

    def test_probabilities(self):
        """Model should produce calibrated probabilities."""
        feats, y, _ = _make_classification_data(n=200, n_features=5)
        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)

        result = trainer.train(feats, y)
        probs = result.model.predict_proba(feats[result.selected_features])

        assert probs.shape[1] == 2
        assert np.all(probs >= 0) and np.all(probs <= 1)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_training_with_cv_splits(self):
        """Train with walk-forward CV splits for eval set."""
        feats, y, timestamps = _make_classification_data(n=300, n_features=5)
        wf_config = WalkForwardConfig(train_days=30, test_days=10)
        splitter = WalkForwardSplitter(wf_config)
        cv = splitter.as_sklearn_cv(timestamps)

        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)

        result = trainer.train(feats, y, cv_splits=cv)
        assert isinstance(result, TrainedModel)

    def test_training_with_cv_splits_refits_on_full_dataset(self):
        """Final runtime fit should use all labeled rows after CV-based selection."""
        feats, y, timestamps = _make_classification_data(n=300, n_features=5)
        wf_config = WalkForwardConfig(train_days=30, test_days=10)
        splitter = WalkForwardSplitter(wf_config)
        cv = splitter.as_sklearn_cv(timestamps)

        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)

        result = trainer.train(feats, y, cv_splits=cv)
        assert result.train_metrics["n_train_samples"] == float(len(feats))

    def test_save_and_load(self, tmp_path):
        """Model should round-trip through save/load."""
        feats, y, _ = _make_classification_data(n=200, n_features=5)
        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)

        result = trainer.train(feats, y)
        model_dir = tmp_path / "test_model"
        trainer.save_model(result, model_dir)

        loaded = trainer.load_model(model_dir)
        assert isinstance(loaded, TrainedModel)
        assert loaded.selected_features == result.selected_features

        # Predictions should match
        original_preds = result.model.predict(feats[result.selected_features])
        loaded_preds = loaded.model.predict(feats[loaded.selected_features])
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_feature_importances_sum(self):
        """Feature importances should sum to ~100."""
        feats, y, _ = _make_classification_data(n=200, n_features=5)
        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)

        result = trainer.train(feats, y)
        total = sum(result.feature_importances.values())
        assert abs(total - 100.0) < 1.0  # CatBoost importances sum to 100


# ────────────────────────────────────────────────────────────────
# Model Evaluator
# ────────────────────────────────────────────────────────────────


class TestModelEvaluator:
    """Test statistical evaluation."""

    def test_perfect_predictions(self):
        """Perfect classifier should get precision=1, recall=1."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0])

        evaluator = ModelEvaluator(n_bootstrap=100, n_permutations=50)
        result = evaluator.evaluate(y_true, y_pred)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.accuracy == 1.0
        assert result.confusion_matrix["tp"] == 3
        assert result.confusion_matrix["tn"] == 3

    def test_worst_predictions(self):
        """All-wrong classifier."""
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        evaluator = ModelEvaluator(n_bootstrap=100, n_permutations=50)
        result = evaluator.evaluate(y_true, y_pred)

        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_bootstrap_ci(self):
        """CI should bracket the point estimate."""
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], size=200)
        y_pred = rng.choice([0, 1], size=200)

        evaluator = ModelEvaluator(n_bootstrap=200, n_permutations=50)
        result = evaluator.evaluate(y_true, y_pred)

        lo, hi = result.precision_ci
        assert lo <= hi
        # CI should be reasonable width
        assert (hi - lo) < 0.5

    def test_permutation_p_value(self):
        """Perfect classifier should have low permutation p-value."""
        y_true = np.array([1] * 50 + [0] * 50)
        y_pred = np.array([1] * 50 + [0] * 50)

        evaluator = ModelEvaluator(n_bootstrap=50, n_permutations=200)
        result = evaluator.evaluate(y_true, y_pred)

        assert result.permutation_p_value is not None
        assert result.permutation_p_value < 0.1

    def test_cohens_d(self):
        """Cohen's d should be non-negative for a decent classifier."""
        rng = np.random.default_rng(42)
        # Imperfect predictions to get non-zero within-group variance
        y_true = np.array([1] * 50 + [0] * 50)
        y_pred = y_true.copy()
        # Flip a few to create variance
        flip_idx = rng.choice(100, 10, replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]

        evaluator = ModelEvaluator(n_bootstrap=50, n_permutations=50)
        result = evaluator.evaluate(y_true, y_pred)

        assert result.cohens_d is not None
        assert result.cohens_d > 0

    def test_roc_auc_with_probabilities(self):
        """ROC-AUC should be computed when probabilities are provided."""
        rng = np.random.default_rng(42)
        y_true = np.array([1] * 50 + [0] * 50)
        y_prob = np.concatenate([
            rng.uniform(0.6, 1.0, 50),
            rng.uniform(0.0, 0.4, 50),
        ])
        y_pred = (y_prob > 0.5).astype(int)

        evaluator = ModelEvaluator(n_bootstrap=50, n_permutations=50)
        result = evaluator.evaluate(y_true, y_pred, y_prob)

        assert result.roc_auc is not None
        assert result.roc_auc > 0.5  # Better than random

    def test_evaluate_result_fields(self):
        """EvaluationResult should have all expected fields."""
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])

        evaluator = ModelEvaluator(n_bootstrap=50, n_permutations=50)
        result = evaluator.evaluate(y_true, y_pred)

        assert result.n_samples == 4
        assert result.n_positive == 2
        assert result.n_negative == 2

    def test_walk_forward_evaluation(self):
        """Evaluate model across walk-forward splits."""
        feats, y, timestamps = _make_classification_data(n=300, n_features=5)

        # Train model
        config = ModelConfig(iterations=50, depth=4, rfecv_enabled=False)
        trainer = ExtremaModelTrainer(config)
        trained = trainer.train(feats, y)

        # Create splits
        wf_config = WalkForwardConfig(train_days=30, test_days=10)
        splitter = WalkForwardSplitter(wf_config)
        cv = splitter.as_sklearn_cv(timestamps)

        evaluator = ModelEvaluator(n_bootstrap=50, n_permutations=50)
        result = evaluator.evaluate_walk_forward(
            trained.model, feats, y, cv, trained.selected_features,
        )

        assert isinstance(result, EvaluationResult)
        assert len(result.fold_metrics) == len(cv)
        assert result.n_samples > 0

    def test_evaluate_out_of_sample_folds(self):
        """Aggregate metrics should support true per-fold out-of-sample predictions."""
        evaluator = ModelEvaluator(n_bootstrap=50, n_permutations=50)
        fold_predictions = [
            {
                "fold": 0,
                "y_true": np.array([1, 0, 1, 0]),
                "y_pred": np.array([1, 0, 1, 0]),
                "y_prob": np.array([0.9, 0.2, 0.8, 0.1]),
            },
            {
                "fold": 1,
                "y_true": np.array([1, 1, 0, 0]),
                "y_pred": np.array([1, 0, 0, 0]),
                "y_prob": np.array([0.7, 0.4, 0.3, 0.2]),
            },
        ]

        result = evaluator.evaluate_out_of_sample_folds(fold_predictions)

        assert isinstance(result, EvaluationResult)
        assert result.n_samples == 8
        assert len(result.fold_metrics) == 2
        assert result.roc_auc is not None


# ────────────────────────────────────────────────────────────────
# Full Pipeline Integration
# ────────────────────────────────────────────────────────────────


class TestFullPipeline:
    """End-to-end: data → features → train → evaluate."""

    def test_train_evaluate_pipeline(self):
        """Full pipeline: generate data, train, evaluate."""
        feats, y, timestamps = _make_classification_data(
            n=400, n_features=8, seed=42,
        )

        # Walk-forward splits
        wf_config = WalkForwardConfig(train_days=30, test_days=10, gap_days=1)
        splitter = WalkForwardSplitter(wf_config)
        cv = splitter.as_sklearn_cv(timestamps)
        assert len(cv) >= 2, "Need at least 2 CV folds"

        # Train
        model_config = ModelConfig(
            iterations=50, depth=4, rfecv_enabled=False,
        )
        trainer = ExtremaModelTrainer(model_config)
        trained = trainer.train(feats, y, cv_splits=cv)
        assert isinstance(trained, TrainedModel)

        # Evaluate
        evaluator = ModelEvaluator(n_bootstrap=100, n_permutations=100)
        result = evaluator.evaluate_walk_forward(
            trained.model, feats, y, cv, trained.selected_features,
        )

        assert result.precision >= 0.0
        assert result.recall >= 0.0
        assert result.f1 >= 0.0
        assert result.n_samples > 0
        assert len(result.fold_metrics) > 0

        # The model should at least do something non-trivial
        # (not all zeros or all ones)
        assert result.confusion_matrix["tp"] + result.confusion_matrix["fp"] > 0
