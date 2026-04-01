"""
Statistical evaluation for trained ML models.

Computes precision (primary metric), F1, ROC-AUC, confusion matrix,
bootstrap confidence intervals, permutation tests, and Cohen's d
effect size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for model evaluation metrics."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float | None = None
    pr_auc: float | None = None
    confusion_matrix: dict[str, int] = field(default_factory=dict)
    precision_ci: tuple[float, float] = (0.0, 0.0)
    f1_ci: tuple[float, float] = (0.0, 0.0)
    permutation_p_value: float | None = None
    cohens_d: float | None = None
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    fold_metrics: list[dict[str, float | int]] = field(default_factory=list)


class ModelEvaluator:
    """Evaluates trained extrema classification models.

    Provides both point estimates and statistical rigor (bootstrap CI,
    permutation tests) following the paper's evaluation methodology.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        n_permutations: int = 500,
        random_seed: int = 42,
    ) -> None:
        self._n_bootstrap = n_bootstrap
        self._n_permutations = n_permutations
        self._rng = np.random.default_rng(random_seed)

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        y_prob: np.ndarray | pd.Series | None = None,
    ) -> EvaluationResult:
        """Evaluate predictions against ground truth.

        Args:
            y_true: True labels (0/1).
            y_pred: Predicted labels (0/1).
            y_prob: Predicted probabilities for positive class (optional).

        Returns:
            EvaluationResult with all metrics.
        """
        y_t = np.asarray(y_true).flatten()
        y_p = np.asarray(y_pred).flatten()

        # Core metrics
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        accuracy = (tp + tn) / len(y_t) if len(y_t) > 0 else 0.0

        # ROC-AUC and PR-AUC
        roc_auc = None
        pr_auc = None
        if y_prob is not None:
            y_proba = np.asarray(y_prob).flatten()
            roc_auc = self._compute_roc_auc(y_t, y_proba)
            pr_auc = self._compute_pr_auc(y_t, y_proba)

        # Bootstrap CIs
        precision_ci = self._bootstrap_ci(y_t, y_p, "precision")
        f1_ci = self._bootstrap_ci(y_t, y_p, "f1")

        # Permutation test
        perm_p = self._permutation_test(y_t, y_p)

        # Cohen's d effect size
        cohens_d = self._compute_cohens_d(y_t, y_p)

        return EvaluationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            confusion_matrix={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            precision_ci=precision_ci,
            f1_ci=f1_ci,
            permutation_p_value=perm_p,
            cohens_d=cohens_d,
            n_samples=len(y_t),
            n_positive=int(np.sum(y_t == 1)),
            n_negative=int(np.sum(y_t == 0)),
        )

    def evaluate_out_of_sample_folds(
        self,
        fold_predictions: list[dict[str, object]],
    ) -> EvaluationResult:
        """Evaluate true out-of-sample predictions collected per fold.

        Each item must contain ``y_true`` and ``y_pred`` arrays, and may
        optionally include ``y_prob`` and ``fold`` metadata.
        """
        if not fold_predictions:
            msg = "fold_predictions must contain at least one evaluated fold"
            raise ValueError(msg)

        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_prob: list[np.ndarray] = []
        fold_metrics: list[dict[str, float | int]] = []
        has_probabilities = True

        for idx, fold_data in enumerate(fold_predictions):
            y_true = np.asarray(fold_data["y_true"]).flatten()
            y_pred = np.asarray(fold_data["y_pred"]).flatten()
            y_prob_raw = fold_data.get("y_prob")
            y_prob = None if y_prob_raw is None else np.asarray(y_prob_raw).flatten()

            fold_result = self.evaluate(y_true, y_pred, y_prob)
            fold_metrics.append({
                "fold": int(fold_data.get("fold", idx)),
                "precision": fold_result.precision,
                "recall": fold_result.recall,
                "f1": fold_result.f1,
                "n_test": int(len(y_true)),
            })

            all_true.append(y_true)
            all_pred.append(y_pred)
            if y_prob is None:
                has_probabilities = False
            else:
                all_prob.append(y_prob)

        aggregate = self.evaluate(
            np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_prob) if has_probabilities else None,
        )
        aggregate.fold_metrics = fold_metrics
        return aggregate

    def evaluate_walk_forward(
        self,
        model: object,
        features: pd.DataFrame,
        y: pd.Series,
        splits: list[tuple[np.ndarray, np.ndarray]],
        feature_names: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate model across walk-forward splits.

        Args:
            model: Trained classifier with predict/predict_proba methods.
            features: Full feature matrix.
            y: Full label series.
            splits: Walk-forward CV splits.
            feature_names: Features to use (if subset selected by RFECV).

        Returns:
            EvaluationResult with fold-level and aggregate metrics.
        """
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_prob: list[np.ndarray] = []
        fold_metrics: list[dict[str, float | int]] = []

        cols = feature_names or list(features.columns)

        for fold_idx, (_train_idx, test_idx) in enumerate(splits):
            x_test = features.iloc[test_idx][cols]
            y_test = y.iloc[test_idx].values

            predictions = model.predict(x_test).flatten()
            probas = None
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(x_test)[:, 1]

            all_true.append(y_test)
            all_pred.append(predictions)
            if probas is not None:
                all_prob.append(probas)

            # Per-fold metrics
            tp = int(np.sum((y_test == 1) & (predictions == 1)))
            fp = int(np.sum((y_test == 0) & (predictions == 1)))
            fn = int(np.sum((y_test == 1) & (predictions == 0)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            fold_metrics.append({
                "fold": fold_idx,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "n_test": len(y_test),
            })

        # Aggregate
        y_true_all = np.concatenate(all_true)
        y_pred_all = np.concatenate(all_pred)
        y_prob_all = np.concatenate(all_prob) if all_prob else None

        result = self.evaluate(y_true_all, y_pred_all, y_prob_all)
        result.fold_metrics = fold_metrics

        return result

    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: str,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for a metric."""
        n = len(y_true)
        if n < 10:
            return (0.0, 1.0)

        scores: list[float] = []
        for _ in range(self._n_bootstrap):
            idx = self._rng.integers(0, n, size=n)
            yt = y_true[idx]
            yp = y_pred[idx]

            tp = np.sum((yt == 1) & (yp == 1))
            fp = np.sum((yt == 0) & (yp == 1))
            fn = np.sum((yt == 1) & (yp == 0))

            if metric == "precision":
                score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            elif metric == "f1":
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                score = (
                    2 * prec * rec / (prec + rec)
                    if (prec + rec) > 0 else 0.0
                )
            else:
                score = 0.0

            scores.append(float(score))

        lower = float(np.percentile(scores, 100 * alpha / 2))
        upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        return (lower, upper)

    def _permutation_test(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Permutation test: P(random baseline >= observed precision)."""
        n = len(y_true)
        if n < 10:
            return 1.0

        tp_obs = np.sum((y_true == 1) & (y_pred == 1))
        fp_obs = np.sum((y_true == 0) & (y_pred == 1))
        obs_precision = tp_obs / (tp_obs + fp_obs) if (tp_obs + fp_obs) > 0 else 0.0

        count_ge = 0
        for _ in range(self._n_permutations):
            y_perm = self._rng.permutation(y_true)
            tp_perm = np.sum((y_perm == 1) & (y_pred == 1))
            fp_perm = np.sum((y_perm == 0) & (y_pred == 1))
            perm_prec = (
                tp_perm / (tp_perm + fp_perm) if (tp_perm + fp_perm) > 0 else 0.0
            )
            if perm_prec >= obs_precision:
                count_ge += 1

        return count_ge / self._n_permutations

    @staticmethod
    def _compute_cohens_d(
        y_true: np.ndarray, y_pred: np.ndarray,
    ) -> float | None:
        """Cohen's d effect size for the classifier's discrimination."""
        # Compare predicted scores for actual positives vs actual negatives
        pos_correct = y_pred[y_true == 1].astype(float)
        neg_correct = y_pred[y_true == 0].astype(float)

        if len(pos_correct) < 2 or len(neg_correct) < 2:
            return None

        mean_diff = np.mean(pos_correct) - np.mean(neg_correct)
        pooled_std = np.sqrt(
            (np.var(pos_correct, ddof=1) + np.var(neg_correct, ddof=1)) / 2
        )

        if pooled_std == 0:
            return 0.0

        return float(mean_diff / pooled_std)

    @staticmethod
    def _compute_roc_auc(
        y_true: np.ndarray, y_prob: np.ndarray,
    ) -> float | None:
        """Compute ROC-AUC without sklearn dependency at call site."""
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) < 2:
                return None
            return float(roc_auc_score(y_true, y_prob))
        except (ImportError, ValueError):
            return None

    @staticmethod
    def _compute_pr_auc(
        y_true: np.ndarray, y_prob: np.ndarray,
    ) -> float | None:
        """Compute Precision-Recall AUC."""
        try:
            from sklearn.metrics import average_precision_score
            if len(np.unique(y_true)) < 2:
                return None
            return float(average_precision_score(y_true, y_prob))
        except (ImportError, ValueError):
            return None
