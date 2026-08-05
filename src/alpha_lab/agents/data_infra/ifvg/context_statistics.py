"""Deterministic metrics and block-bootstrap uncertainty for IFVG context runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .context_experiment_contracts import (
    IFVG_CONTEXT_CALIBRATION_POLICY_ID,
    THRESHOLD_REPORT_GRID,
)

__all__ = [
    "binary_prediction_report",
    "block_bootstrap_interval",
    "candidate_uncertainty_report",
    "paired_tier_delta_report",
    "executed_trade_uncertainty_report",
    "feature_importance_report",
]


def _finite_probabilities(values: pd.Series) -> np.ndarray:
    probabilities = pd.to_numeric(values, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("predicted probabilities must be finite and in [0, 1]")
    return probabilities


def _calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    if len(set(y.tolist())) < 2:
        return {
            "intercept": None,
            "slope": None,
            "available": False,
            "reason": "single_class_target",
        }
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000)
    model.fit(logit, y)
    return {
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
        "available": True,
        "reason": None,
    }


def _reliability(y: np.ndarray, probability: np.ndarray) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        selected = (probability >= lower) & (
            probability <= upper if index == 9 else probability < upper
        )
        bins.append(
            {
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": int(selected.sum()),
                "mean_probability": (
                    float(probability[selected].mean()) if selected.any() else None
                ),
                "observed_rate": float(y[selected].mean()) if selected.any() else None,
            }
        )
    return bins


def _r_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "count": 0,
            "gross_r_sum": 0.0,
            "gross_r_mean": None,
            "net_r_sum": 0.0,
            "net_r_mean": None,
        }
    gross = pd.to_numeric(frame["gross_r"], errors="raise")
    net = pd.to_numeric(frame["net_r"], errors="raise")
    return {
        "count": len(frame),
        "gross_r_sum": float(gross.sum()),
        "gross_r_mean": float(gross.mean()),
        "net_r_sum": float(net.sum()),
        "net_r_mean": float(net.mean()),
    }


def binary_prediction_report(predictions: pd.DataFrame) -> dict[str, Any]:
    required = {"target", "probability", "training_prevalence", "gross_r", "net_r"}
    missing = sorted(required - set(predictions))
    if missing:
        raise ValueError(f"prediction report is missing columns {missing}")
    if predictions.empty:
        return {
            "status": "insufficient_class_coverage",
            "count": 0,
            "auc": None,
            "auc_reason": "no_oos_predictions",
            "calibration_policy": IFVG_CONTEXT_CALIBRATION_POLICY_ID,
            "reliability_bins": _reliability(np.array([], dtype=int), np.array([])),
            "thresholds": [
                {
                    "threshold": threshold,
                    "coverage_count": 0,
                    "coverage_fraction": None,
                    "r": _r_summary(pd.DataFrame(columns=("gross_r", "net_r"))),
                }
                for threshold in THRESHOLD_REPORT_GRID
            ],
        }
    y = pd.to_numeric(predictions["target"], errors="raise").astype(int).to_numpy()
    if not set(y).issubset({0, 1}):
        raise ValueError("binary prediction targets must be 0/1")
    probability = _finite_probabilities(predictions["probability"])
    reference = _finite_probabilities(predictions["training_prevalence"])
    brier = float(brier_score_loss(y, probability))
    reference_brier = float(np.mean((y - reference) ** 2))
    brier_skill = None if reference_brier == 0 else 1 - brier / reference_brier
    if len(set(y.tolist())) == 2:
        auc = float(roc_auc_score(y, probability))
        auc_reason = None
    else:
        auc = None
        auc_reason = "single_class_oos"
    thresholds = []
    for threshold in THRESHOLD_REPORT_GRID:
        selected = predictions.loc[probability >= threshold]
        thresholds.append(
            {
                "threshold": threshold,
                "coverage_count": len(selected),
                "coverage_fraction": len(selected) / len(predictions),
                "r": _r_summary(selected),
            }
        )
    return {
        "status": "complete",
        "calibration_policy": IFVG_CONTEXT_CALIBRATION_POLICY_ID,
        "count": len(predictions),
        "prevalence": float(y.mean()),
        "mean_probability": float(probability.mean()),
        "brier_score": brier,
        "reference_brier_score": reference_brier,
        "brier_skill_score": brier_skill,
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "auc": auc,
        "auc_reason": auc_reason,
        "calibration": _calibration(y, probability),
        "reliability_bins": _reliability(y, probability),
        "thresholds": thresholds,
        "gross_net_r": _r_summary(predictions),
    }


def block_bootstrap_interval(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    value_column: str,
    repetitions: int = 10_000,
    seed: int = 7,
) -> dict[str, Any]:
    if repetitions != 10_000 or seed != 7:
        raise ValueError("IFVG bootstrap protocol is fixed at 10,000 repetitions, seed 7")
    work = frame[[cluster_column, value_column]].dropna().copy()
    clusters = tuple(sorted(work[cluster_column].astype(str).unique()))
    if len(clusters) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_usable_clusters",
            "cluster_count": len(clusters),
            "repetitions": repetitions,
            "seed": seed,
            "lower": None,
            "upper": None,
        }
    block_values = [
        pd.to_numeric(
            work.loc[work[cluster_column].astype(str) == cluster, value_column],
            errors="raise",
        ).to_numpy(dtype=float)
        for cluster in clusters
    ]
    block_sums = np.array([values.sum() for values in block_values])
    block_counts = np.array([len(values) for values in block_values])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0,
        len(clusters),
        size=(repetitions, len(clusters)),
    )
    estimates = block_sums[sampled].sum(axis=1) / block_counts[sampled].sum(axis=1)
    return {
        "available": True,
        "reason": None,
        "cluster_count": len(clusters),
        "repetitions": repetitions,
        "seed": seed,
        "estimate": float(pd.to_numeric(work[value_column], errors="raise").mean()),
        "lower": float(np.quantile(estimates, 0.025)),
        "upper": float(np.quantile(estimates, 0.975)),
    }


def candidate_uncertainty_report(predictions: pd.DataFrame) -> dict[str, Any]:
    return {
        "setup_cluster_net_r_mean": block_bootstrap_interval(
            predictions,
            cluster_column="setup_id",
            value_column="net_r",
        ),
        "trading_day_block_net_r_mean": block_bootstrap_interval(
            predictions,
            cluster_column="trading_day",
            value_column="net_r",
        ),
    }


def paired_tier_delta_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    value_column: str = "net_r",
) -> dict[str, Any]:
    keys = ("candidate_id", "trading_day", value_column)
    if not set(keys).issubset(left) or not set(keys).issubset(right):
        raise ValueError("tier delta inputs lack paired candidate/day/value columns")
    if set(left["candidate_id"].astype(str)) != set(right["candidate_id"].astype(str)):
        raise ValueError("tier delta requires identical OOS candidate IDs")
    paired = left[list(keys)].merge(
        right[list(keys)],
        on="candidate_id",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if not (
        paired["trading_day_left"].astype(str)
        == paired["trading_day_right"].astype(str)
    ).all():
        raise ValueError("paired tier candidates disagree on trading day")
    paired["trading_day"] = paired["trading_day_left"].astype(str)
    paired["delta"] = pd.to_numeric(
        paired[f"{value_column}_right"], errors="raise"
    ) - pd.to_numeric(paired[f"{value_column}_left"], errors="raise")
    return block_bootstrap_interval(
        paired,
        cluster_column="trading_day",
        value_column="delta",
    )


def executed_trade_uncertainty_report(executed_trades: pd.DataFrame) -> dict[str, Any]:
    return {
        "trading_day_block_realized_r_mean": block_bootstrap_interval(
            executed_trades,
            cluster_column="trading_day",
            value_column="realized_r",
        )
    }


def feature_importance_report(importance: pd.DataFrame) -> list[dict[str, Any]]:
    if importance.empty:
        return []
    required = {
        "fold_index",
        "feature",
        "catboost_importance",
        "permutation_importance_mean",
        "permutation_importance_variance",
        "permutation_repeats",
    }
    missing = sorted(required - set(importance))
    if missing:
        raise ValueError(f"feature importance is missing columns {missing}")
    fold_count = importance["fold_index"].nunique()
    output = []
    for feature, rows in importance.groupby("feature", sort=True):
        output.append(
            {
                "feature": feature,
                "fold_coverage": rows["fold_index"].nunique(),
                "fold_coverage_fraction": rows["fold_index"].nunique() / fold_count,
                "catboost_importance_mean": float(rows["catboost_importance"].mean()),
                "catboost_importance_variance": float(rows["catboost_importance"].var(ddof=0)),
                "permutation_importance_mean": float(
                    rows["permutation_importance_mean"].mean()
                ),
                "permutation_importance_between_fold_variance": float(
                    rows["permutation_importance_mean"].var(ddof=0)
                ),
                "permutation_within_fold_variance_mean": float(
                    rows["permutation_importance_variance"].mean()
                ),
                "permutation_repeats": 20,
                "use": "descriptive_only",
            }
        )
    return output
