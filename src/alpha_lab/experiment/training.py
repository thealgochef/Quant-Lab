"""
Phase 5 - Walk-Forward CatBoost 3-Class Compatibility Training.

Trains a CatBoost MultiClass classifier using walk-forward validation
on the order flow feature matrix (Phase 4 output).  Evaluates against
hypothesis thresholds for tradeable_reversal precision, blow-through
recall, cross-fold stability, and overall accuracy.

This is the retained secondary compatibility/export path for
ML-Trading-Dashboard. It stays self-contained and does not import from
the primary extrema pipeline under agents/data_infra/ml/.

Canonical downstream model export is produced by scripts/train_dashboard_model.py
at data/models/dashboard_3feature_v1.cbm.

Output: printed verdict + saved results to data/experiment/training_results/
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.experiment.features import CAT_FEATURES, LABEL_ENCODING

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────

METADATA_COLS = {"event_ts", "date", "label", "label_encoded"}

# Walk-forward defaults (trading days, not calendar)
DEFAULT_TRAIN_DAYS = 40
DEFAULT_TEST_DAYS = 5
DEFAULT_STEP_DAYS = 5
MIN_TRAIN_EVENTS = 30
EXPAND_INCREMENT = 5

# Hypothesis thresholds
THRESHOLD_ACCURACY = 0.40
THRESHOLD_REVERSAL_PRECISION = 0.50
THRESHOLD_BLOWTHROUGH_RECALL = 0.60
THRESHOLD_ACCURACY_VARIANCE = 0.10

# Label indices (matching LABEL_ENCODING)
CLASS_REVERSAL = 0
CLASS_TRAP = 1
CLASS_BLOWTHROUGH = 2
CLASS_NAMES = {0: "tradeable_reversal", 1: "trap_reversal", 2: "aggressive_blowthrough"}

# Dashboard model: the 3 features computable from real-time MBP-1 + trades
DASHBOARD_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
]


# ── Data Classes ───────────────────────────────────────────────


@dataclass
class WalkForwardFold:
    """A single walk-forward fold definition."""

    fold: int
    train_dates: list[str]
    test_dates: list[str]
    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class FoldResult:
    """Per-fold evaluation results."""

    fold: int
    n_train: int
    n_test: int
    train_dates_str: str  # "first -- last"
    test_dates_str: str   # "first -- last"
    accuracy: float
    reversal_precision: float
    blowthrough_recall: float
    confusion_matrix: np.ndarray  # 3x3
    feature_importances: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class TrainingResult:
    """Aggregated results across all folds."""

    fold_results: list[FoldResult]
    aggregated_confusion: np.ndarray  # 3x3
    overall_accuracy: float
    reversal_precision: float
    blowthrough_recall: float
    accuracy_variance: float
    top_features_stability: pd.DataFrame
    verdict: dict[str, dict]
    mae_distribution: pd.Series | None


# ── Walk-Forward Splitter ──────────────────────────────────────


def create_walk_forward_folds(
    df: pd.DataFrame,
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    min_train_events: int = MIN_TRAIN_EVENTS,
) -> list[WalkForwardFold]:
    """Create walk-forward folds based on trading days in the data.

    Uses actual dates present in the data (trading days), not calendar days.
    Sliding window: train=dates[start:start+train_days], test=dates[start+train_days:start+train_days+test_days].
    Steps forward by step_days trading days each fold.

    If a fold's training set has fewer than min_train_events, the training
    window expands backward by EXPAND_INCREMENT days at a time.  If
    expansion cannot reach min_train_events, the fold is skipped.
    """
    trading_dates = sorted(df["date"].unique())
    n_dates = len(trading_dates)

    folds: list[WalkForwardFold] = []
    fold_idx = 0
    original_start = 0

    while original_start + train_days + test_days <= n_dates:
        test_start_idx = original_start + train_days
        test_end_idx = test_start_idx + test_days

        if test_end_idx > n_dates:
            break

        fold_test_dates = list(trading_dates[test_start_idx:test_end_idx])
        test_mask = df["date"].isin(fold_test_dates)
        test_idx = np.where(test_mask)[0]

        if len(test_idx) == 0:
            original_start += step_days
            continue

        # Start with the default training window; expand backward if needed
        effective_start = original_start
        while True:
            fold_train_dates = list(trading_dates[effective_start:test_start_idx])
            train_mask = df["date"].isin(fold_train_dates)
            train_idx = np.where(train_mask)[0]

            if len(train_idx) >= min_train_events or effective_start == 0:
                break
            effective_start = max(0, effective_start - EXPAND_INCREMENT)

        if len(train_idx) < min_train_events:
            logger.warning(
                "Fold %d: only %d train events after expansion, skipping",
                fold_idx, len(train_idx),
            )
            original_start += step_days
            continue

        folds.append(WalkForwardFold(
            fold=fold_idx,
            train_dates=fold_train_dates,
            test_dates=fold_test_dates,
            train_indices=train_idx,
            test_indices=test_idx,
        ))
        fold_idx += 1
        original_start += step_days

    return folds


# ── Confusion Matrix ──────────────────────────────────────────


def _confusion_matrix_3class(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> np.ndarray:
    """Compute 3x3 confusion matrix. Rows = actual, columns = predicted."""
    cm = np.zeros((3, 3), dtype=int)
    for true_cls in range(3):
        for pred_cls in range(3):
            cm[true_cls, pred_cls] = int(
                np.sum((y_true == true_cls) & (y_pred == pred_cls))
            )
    return cm


# ── Per-Fold Training ─────────────────────────────────────────


def train_fold(
    df: pd.DataFrame,
    fold: WalkForwardFold,
    feature_cols: list[str],
    cat_features: list[str],
) -> tuple[object, FoldResult]:
    """Train CatBoost on one fold with 80/20 internal eval split.

    The training set is split 80/20 chronologically: first 80% for
    fitting, last 20% for early-stopping evaluation.
    """
    from catboost import CatBoostClassifier, Pool

    X_train = df.iloc[fold.train_indices][feature_cols].reset_index(drop=True)
    y_train = df.iloc[fold.train_indices]["label_encoded"].reset_index(drop=True)
    X_test = df.iloc[fold.test_indices][feature_cols].reset_index(drop=True)
    y_test = df.iloc[fold.test_indices]["label_encoded"].reset_index(drop=True)

    # 80/20 chronological split within training set
    n_train = len(X_train)
    split_idx = int(n_train * 0.8)

    X_fit, X_eval = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_eval = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    # Resolve cat_features to column indices
    cat_indices = [
        list(X_train.columns).index(c) for c in cat_features
        if c in X_train.columns
    ]

    model = CatBoostClassifier(
        iterations=1000,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        has_time=True,
        cat_features=cat_indices,
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        auto_class_weights="Balanced",
    )

    train_pool = Pool(X_fit, y_fit, cat_features=cat_indices)
    eval_pool = Pool(X_eval, y_eval, cat_features=cat_indices)

    model.fit(train_pool, eval_set=eval_pool, verbose=100)

    # Predict on test set
    y_pred = model.predict(X_test).flatten().astype(int)
    y_true = y_test.values.astype(int)

    # Per-fold metrics
    cm = _confusion_matrix_3class(y_true, y_pred)
    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0

    # Reversal precision: TP_rev / predicted_rev
    rev_predicted = y_pred == CLASS_REVERSAL
    reversal_precision = (
        float(np.sum(y_true[rev_predicted] == CLASS_REVERSAL) / rev_predicted.sum())
        if rev_predicted.sum() > 0 else 0.0
    )

    # Blowthrough recall: TP_bt / actual_bt
    bt_actual = y_true == CLASS_BLOWTHROUGH
    blowthrough_recall = (
        float(np.sum(y_pred[bt_actual] == CLASS_BLOWTHROUGH) / bt_actual.sum())
        if bt_actual.sum() > 0 else 0.0
    )

    # Feature importances
    importance_values = model.get_feature_importance()
    importance_dict = dict(zip(feature_cols, importance_values, strict=True))

    train_dates_str = f"{fold.train_dates[0]} -- {fold.train_dates[-1]}"
    test_dates_str = f"{fold.test_dates[0]} -- {fold.test_dates[-1]}"

    fold_result = FoldResult(
        fold=fold.fold,
        n_train=n_train,
        n_test=len(y_test),
        train_dates_str=train_dates_str,
        test_dates_str=test_dates_str,
        accuracy=accuracy,
        reversal_precision=reversal_precision,
        blowthrough_recall=blowthrough_recall,
        confusion_matrix=cm,
        feature_importances=importance_dict,
        y_true=y_true.copy(),
        y_pred=y_pred.copy(),
    )

    logger.info(
        "Fold %d: n_train=%d n_test=%d acc=%.3f rev_prec=%.3f bt_recall=%.3f",
        fold.fold, n_train, len(y_test), accuracy,
        reversal_precision, blowthrough_recall,
    )

    return model, fold_result


# ── Feature Importance Stability ──────────────────────────────


def _compute_feature_stability(
    fold_results: list[FoldResult],
    top_n: int = 10,
) -> pd.DataFrame:
    """Compute feature importance stability across folds.

    For each fold, take the top_n features by importance. Report which
    features appear in ALL folds' top_n lists.
    """
    n_folds = len(fold_results)
    per_fold_tops: list[set[str]] = []
    all_importances: dict[str, list[float]] = {}

    for fr in fold_results:
        sorted_feats = sorted(
            fr.feature_importances.items(),
            key=lambda x: x[1], reverse=True,
        )
        top_names = {name for name, _ in sorted_feats[:top_n]}
        per_fold_tops.append(top_names)

        for name, imp in fr.feature_importances.items():
            all_importances.setdefault(name, []).append(imp)

    # Only include features that appeared in at least one fold's top N
    all_top_features = set()
    for tops in per_fold_tops:
        all_top_features |= tops

    rows = []
    for feat in all_top_features:
        imps = all_importances.get(feat, [])
        appearances = sum(1 for tops in per_fold_tops if feat in tops)
        rows.append({
            "feature": feat,
            "mean_importance": float(np.mean(imps)),
            "std_importance": float(np.std(imps)),
            "folds_in_top_10": appearances,
            "in_all_folds": appearances == n_folds,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


# ── MAE Distribution ─────────────────────────────────────────


def _compute_mae_distribution(
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    fold_results: list[FoldResult],
) -> pd.Series | None:
    """Get MAE values for events correctly predicted as tradeable_reversal."""
    if "max_mae" not in df.columns:
        logger.warning("max_mae not in DataFrame, skipping MAE distribution")
        return None

    tp_mae_values: list[float] = []
    for fold, fr in zip(folds, fold_results):
        # True positive reversals: predicted=0 AND actual=0
        tp_mask = (fr.y_true == CLASS_REVERSAL) & (fr.y_pred == CLASS_REVERSAL)
        tp_indices = fold.test_indices[tp_mask]
        tp_mae_values.extend(df.iloc[tp_indices]["max_mae"].tolist())

    if not tp_mae_values:
        return None

    return pd.Series(tp_mae_values, name="mae_pts")


# ── Verdict ───────────────────────────────────────────────────


def _evaluate_verdict(
    accuracy: float,
    reversal_precision: float,
    blowthrough_recall: float,
    accuracy_variance: float,
) -> dict[str, dict]:
    """Evaluate pooled results against hypothesis thresholds."""
    return {
        "Overall Accuracy > 40%": {
            "value": accuracy,
            "threshold": THRESHOLD_ACCURACY,
            "passed": accuracy > THRESHOLD_ACCURACY,
        },
        "Tradeable Reversal Precision > 50%": {
            "value": reversal_precision,
            "threshold": THRESHOLD_REVERSAL_PRECISION,
            "passed": reversal_precision > THRESHOLD_REVERSAL_PRECISION,
        },
        "Blow-through Recall > 60%": {
            "value": blowthrough_recall,
            "threshold": THRESHOLD_BLOWTHROUGH_RECALL,
            "passed": blowthrough_recall > THRESHOLD_BLOWTHROUGH_RECALL,
        },
        "Cross-fold Accuracy StdDev < 10%": {
            "value": accuracy_variance,
            "threshold": THRESHOLD_ACCURACY_VARIANCE,
            "passed": accuracy_variance < THRESHOLD_ACCURACY_VARIANCE,
        },
    }


# ── Print & Save ──────────────────────────────────────────────


def print_results(result: TrainingResult) -> None:
    """Print formatted experiment results to stdout."""
    print(f"\n{'=' * 72}")
    print("  PHASE 5: WALK-FORWARD CATBOOST 3-CLASS EXPERIMENT")
    print(f"{'=' * 72}")

    # Per-fold table
    n = len(result.fold_results)
    print(f"\n  --- Per-Fold Results ({n} folds) ---\n")
    print(f"  {'Fold':>4}  {'N_train':>7}  {'N_test':>6}  "
          f"{'Accuracy':>8}  {'Rev Prec':>8}  {'BT Recall':>9}  {'Test Dates'}")
    print(f"  {'----':>4}  {'-------':>7}  {'------':>6}  "
          f"{'--------':>8}  {'--------':>8}  {'---------':>9}  {'----------'}")
    for fr in result.fold_results:
        print(
            f"  {fr.fold:4d}  {fr.n_train:7d}  {fr.n_test:6d}  "
            f"{fr.accuracy:8.3f}  {fr.reversal_precision:8.3f}  "
            f"{fr.blowthrough_recall:9.3f}  {fr.test_dates_str}"
        )

    # Aggregated confusion matrix
    print(f"\n  --- Aggregated Confusion Matrix (pooled across all test folds) ---\n")
    print(f"  {'Actual \\ Predicted':25s}  "
          f"{'Pred Rev':>8}  {'Pred Trap':>9}  {'Pred BT':>7}")
    print(f"  {'':25s}  {'--------':>8}  {'---------':>9}  {'-------':>7}")
    for i, name in CLASS_NAMES.items():
        row = result.aggregated_confusion[i]
        print(f"  {name:25s}  {row[0]:8d}  {row[1]:9d}  {row[2]:7d}")
    total = result.aggregated_confusion.sum()
    print(f"\n  Total test predictions: {total}")

    # Feature importance stability
    print(f"\n  --- Feature Importance Stability (top 10 per fold) ---\n")
    stable = result.top_features_stability
    in_all = stable[stable["in_all_folds"]]
    print(f"  Features in ALL folds' top 10: {len(in_all)}")
    print()
    for _, row in stable.head(20).iterrows():
        marker = " *" if row["in_all_folds"] else ""
        print(
            f"    {row['feature']:40s}  imp={row['mean_importance']:6.2f} "
            f"(+/-{row['std_importance']:.2f})  "
            f"folds={row['folds_in_top_10']}/{n}{marker}"
        )

    # MAE distribution
    if result.mae_distribution is not None and len(result.mae_distribution) > 0:
        mae = result.mae_distribution
        print(f"\n  --- MAE Distribution (True Positive Reversals, n={len(mae)}) ---\n")
        print(f"    Mean:   {mae.mean():.2f} pts")
        print(f"    Median: {mae.median():.2f} pts")
        print(f"    Std:    {mae.std():.2f} pts")
        print(f"    Min:    {mae.min():.2f} pts")
        print(f"    Max:    {mae.max():.2f} pts")
        for pct in [25, 50, 75, 90]:
            print(f"    P{pct}:    {mae.quantile(pct / 100):.2f} pts")
    else:
        print(f"\n  --- MAE Distribution: N/A (no true-positive reversals) ---")

    # Verdict
    print(f"\n  --- VERDICT (based on pooled test predictions) ---\n")
    all_passed = True
    for name, check in result.verdict.items():
        status = "PASS" if check["passed"] else "FAIL"
        all_passed = all_passed and check["passed"]
        print(
            f"    [{status}] {name}: "
            f"{check['value']:.4f} (threshold: {check['threshold']:.2f})"
        )

    print()
    if all_passed:
        print("  >>> HYPOTHESIS SUPPORTED — all thresholds met.")
    else:
        n_passed = sum(1 for c in result.verdict.values() if c["passed"])
        n_total = len(result.verdict)
        print(f"  >>> HYPOTHESIS NOT SUPPORTED — {n_passed}/{n_total} thresholds met.")
    print(f"\n{'=' * 72}\n")


def save_results(result: TrainingResult, output_dir: Path) -> None:
    """Save results to disk as CSV + JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fold results table
    fold_df = pd.DataFrame([{
        "fold": fr.fold,
        "n_train": fr.n_train,
        "n_test": fr.n_test,
        "train_dates": fr.train_dates_str,
        "test_dates": fr.test_dates_str,
        "accuracy": fr.accuracy,
        "reversal_precision": fr.reversal_precision,
        "blowthrough_recall": fr.blowthrough_recall,
    } for fr in result.fold_results])
    fold_df.to_csv(output_dir / "fold_results.csv", index=False)

    # Aggregated confusion matrix
    cm_df = pd.DataFrame(
        result.aggregated_confusion,
        index=list(CLASS_NAMES.values()),
        columns=[f"pred_{n}" for n in CLASS_NAMES.values()],
    )
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    # Feature stability
    result.top_features_stability.to_csv(
        output_dir / "feature_stability.csv", index=False,
    )

    # MAE distribution
    if result.mae_distribution is not None:
        result.mae_distribution.to_frame().to_csv(
            output_dir / "mae_distribution.csv", index=False,
        )

    # Summary JSON
    verdict_serializable = {}
    for k, v in result.verdict.items():
        verdict_serializable[k] = {
            "value": float(v["value"]),
            "threshold": float(v["threshold"]),
            "passed": bool(v["passed"]),
        }
    summary = {
        "overall_accuracy": float(result.overall_accuracy),
        "reversal_precision": float(result.reversal_precision),
        "blowthrough_recall": float(result.blowthrough_recall),
        "accuracy_variance": float(result.accuracy_variance),
        "n_folds": len(result.fold_results),
        "verdict": verdict_serializable,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", output_dir)


# ── Final Model Training ─────────────────────────────────────


def _train_and_save_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_feature_names: list[str],
    output_dir: Path,
) -> Path:
    """Train a final model on ALL data and save as .cbm file.

    Uses the same hyperparameters as walk-forward folds but trains on the
    entire dataset. The walk-forward validation has already proven
    out-of-sample performance — this final model maximizes training data.
    """
    from catboost import CatBoostClassifier, Pool

    X = df[feature_cols].reset_index(drop=True)
    y = df["label_encoded"].reset_index(drop=True)

    cat_indices = [
        list(X.columns).index(c) for c in cat_feature_names
        if c in X.columns
    ]

    # 80/20 chronological split for early stopping
    split_idx = int(len(X) * 0.8)
    X_fit, X_eval = X.iloc[:split_idx], X.iloc[split_idx:]
    y_fit, y_eval = y.iloc[:split_idx], y.iloc[split_idx:]

    model = CatBoostClassifier(
        iterations=1000,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        has_time=True,
        cat_features=cat_indices,
        early_stopping_rounds=50,
        verbose=100,
        random_seed=42,
        auto_class_weights="Balanced",
    )

    train_pool = Pool(X_fit, y_fit, cat_features=cat_indices)
    eval_pool = Pool(X_eval, y_eval, cat_features=cat_indices)
    model.fit(train_pool, eval_set=eval_pool, verbose=100)

    model_path = output_dir / "model.cbm"
    model.save_model(str(model_path))
    logger.info("Final model saved to %s (%d features)", model_path, len(feature_cols))

    return model_path


# ── Main Orchestrator ─────────────────────────────────────────


def run_experiment(
    feature_matrix_path: Path = Path("data/experiment/feature_matrix.parquet"),
    labeled_events_path: Path = Path("data/experiment/labeled_events.parquet"),
    output_dir: Path = Path("data/experiment/training_results"),
    train_days: int = DEFAULT_TRAIN_DAYS,
    test_days: int = DEFAULT_TEST_DAYS,
    step_days: int = DEFAULT_STEP_DAYS,
    feature_subset: list[str] | None = None,
) -> TrainingResult:
    """Run the complete walk-forward experiment.

    1. Load feature matrix and labeled events.
    2. Attach max_mae from labeled events.
    3. Sort chronologically (required by has_time=True).
    4. Create walk-forward folds.
    5. Train CatBoost per fold, collect results.
    6. Aggregate confusion matrix across folds.
    7. Compute verdict from POOLED test predictions.
    8. Print full report and save to disk.
    9. Train final model on ALL data and save .cbm file.

    Args:
        feature_subset: If provided, use only these feature columns.
            Pass DASHBOARD_FEATURES for the 3-feature dashboard model.
    """
    # 1. Load data
    df = pd.read_parquet(feature_matrix_path)
    labeled = pd.read_parquet(labeled_events_path)
    logger.info("Loaded feature matrix: %s", df.shape)

    # Attach max_mae from labeled events (row-aligned after filtering no_resolution)
    resolved = labeled[labeled["label"] != "no_resolution"].reset_index(drop=True)
    if "max_mae" in resolved.columns and len(resolved) == len(df):
        df["max_mae"] = resolved["max_mae"].values
    else:
        logger.warning(
            "Cannot align max_mae: resolved=%d, feature_matrix=%d",
            len(resolved), len(df),
        )

    # 2. Identify feature columns
    if feature_subset is not None:
        missing = [f for f in feature_subset if f not in df.columns]
        if missing:
            raise ValueError(f"Feature subset columns not in data: {missing}")
        feature_cols = list(feature_subset)
        cat_feature_names = [c for c in CAT_FEATURES if c in feature_cols]
    else:
        exclude = METADATA_COLS | {"max_mae"}
        feature_cols = [c for c in df.columns if c not in exclude]
        cat_feature_names = [c for c in CAT_FEATURES if c in feature_cols]
    logger.info(
        "Features: %d total (%d categorical)", len(feature_cols), len(cat_feature_names),
    )

    # 3. Sort chronologically (required by has_time=True)
    df = df.sort_values(["date", "event_ts"]).reset_index(drop=True)

    # 4. Create walk-forward folds
    folds = create_walk_forward_folds(
        df, train_days, test_days, step_days,
    )
    logger.info("Created %d walk-forward folds", len(folds))

    if not folds:
        raise ValueError(
            f"No valid folds created from {len(df)} events across "
            f"{df['date'].nunique()} trading days with "
            f"train={train_days}, test={test_days}, step={step_days}"
        )

    # 5. Train per fold
    fold_results: list[FoldResult] = []
    for fold in folds:
        _, fold_result = train_fold(df, fold, feature_cols, cat_feature_names)
        fold_results.append(fold_result)

    # 6. Aggregate — POOL all test predictions for verdict metrics
    all_y_true = np.concatenate([fr.y_true for fr in fold_results])
    all_y_pred = np.concatenate([fr.y_pred for fr in fold_results])
    agg_cm = _confusion_matrix_3class(all_y_true, all_y_pred)

    # Overall metrics from pooled predictions
    overall_accuracy = float(np.mean(all_y_true == all_y_pred))

    rev_predicted = all_y_pred == CLASS_REVERSAL
    reversal_precision = (
        float(np.sum(all_y_true[rev_predicted] == CLASS_REVERSAL) / rev_predicted.sum())
        if rev_predicted.sum() > 0 else 0.0
    )

    bt_actual = all_y_true == CLASS_BLOWTHROUGH
    blowthrough_recall = (
        float(np.sum(all_y_pred[bt_actual] == CLASS_BLOWTHROUGH) / bt_actual.sum())
        if bt_actual.sum() > 0 else 0.0
    )

    # Cross-fold accuracy variance (std dev of per-fold accuracies)
    fold_accuracies = [fr.accuracy for fr in fold_results]
    accuracy_variance = float(np.std(fold_accuracies))

    # 7. Feature importance stability
    top_features_stability = _compute_feature_stability(fold_results)

    # 8. MAE distribution for true-positive reversals
    mae_distribution = _compute_mae_distribution(df, folds, fold_results)

    # 9. Verdict (from POOLED metrics)
    verdict = _evaluate_verdict(
        overall_accuracy, reversal_precision,
        blowthrough_recall, accuracy_variance,
    )

    result = TrainingResult(
        fold_results=fold_results,
        aggregated_confusion=agg_cm,
        overall_accuracy=overall_accuracy,
        reversal_precision=reversal_precision,
        blowthrough_recall=blowthrough_recall,
        accuracy_variance=accuracy_variance,
        top_features_stability=top_features_stability,
        verdict=verdict,
        mae_distribution=mae_distribution,
    )

    # Print and save
    print_results(result)
    save_results(result, output_dir)

    # 10. Train final model on ALL data and save .cbm
    _train_and_save_final_model(df, feature_cols, cat_feature_names, output_dir)

    return result


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

    if "--dashboard" in sys.argv:
        # Train a 3-feature model for the live dashboard
        print("Training 3-feature dashboard model...")
        output = Path("data/experiment/dashboard_model")
        run_experiment(
            feature_subset=DASHBOARD_FEATURES,
            output_dir=output,
        )
        print(f"\nDashboard model saved to: {output / 'model.cbm'}")
        print("Upload this file through the dashboard Models tab.")
    else:
        run_experiment()
