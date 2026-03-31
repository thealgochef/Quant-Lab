"""
Phase 6 — Diagnostic Analysis for Walk-Forward Experiment.

Five analyses investigating Phase 5 results:
  1. Fold 2 deep dive (why Jan 27 - Feb 2 collapsed)
  2. Binary classification test (reversal vs. no_reversal)
  3. Top-3 feature only model
  4. Per-feature class distributions for top 5 features
  5. Misclassification pattern analysis

Output: data/experiment/training_results/diagnostics/
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.experiment.features import CAT_FEATURES, LABEL_ENCODING
from alpha_lab.experiment.training import (
    CLASS_BLOWTHROUGH,
    CLASS_NAMES,
    CLASS_REVERSAL,
    CLASS_TRAP,
    METADATA_COLS,
    FoldResult,
    WalkForwardFold,
    _confusion_matrix_3class,
    create_walk_forward_folds,
    train_fold,
)

logger = logging.getLogger(__name__)

TOP_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
    "ctx_time_normalized",
    "ctx_session",
]

TOP_3_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
]


# ── 1. Fold 2 Deep Dive ──────────────────────────────────────


def _fold2_deep_dive(
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    fold_results: list[FoldResult],
    out_dir: Path,
) -> str:
    """Analyze Fold 2 (Jan 27 - Feb 2) in detail."""
    lines: list[str] = []
    w = lines.append

    fold2 = folds[2]
    fr2 = fold_results[2]

    test_df = df.iloc[fold2.test_indices].copy()
    test_df["y_pred"] = fr2.y_pred

    w("=" * 70)
    w("  DIAGNOSTIC 1: FOLD 2 DEEP DIVE (Jan 27 - Feb 2)")
    w("=" * 70)
    w("")
    w(f"  Test dates: {fold2.test_dates[0]} -- {fold2.test_dates[-1]}")
    w(f"  N events: {fr2.n_test}")
    w(f"  Accuracy: {fr2.accuracy:.3f}")
    w(f"  Reversal precision: {fr2.reversal_precision:.3f}")
    w(f"  Blowthrough recall: {fr2.blowthrough_recall:.3f}")

    # Class distribution in test window
    w("")
    w("  --- Class Distribution in Test Window ---")
    for cls_id, cls_name in CLASS_NAMES.items():
        n = int(np.sum(fr2.y_true == cls_id))
        pct = n / fr2.n_test * 100 if fr2.n_test > 0 else 0
        w(f"    {cls_name:25s}: {n:3d} ({pct:.1f}%)")

    # Per-class accuracy
    w("")
    w("  --- Per-Class Accuracy ---")
    for cls_id, cls_name in CLASS_NAMES.items():
        mask = fr2.y_true == cls_id
        if mask.sum() > 0:
            correct = np.sum(fr2.y_pred[mask] == cls_id)
            acc = correct / mask.sum()
            w(f"    {cls_name:25s}: {correct}/{mask.sum()} correct ({acc:.1%})")
        else:
            w(f"    {cls_name:25s}: 0 events in test set")

    # Fold 2 confusion matrix
    w("")
    w("  --- Fold 2 Confusion Matrix ---")
    w(f"  {'Actual \\ Predicted':25s}  {'Pred Rev':>8}  {'Pred Trap':>9}  {'Pred BT':>7}")
    for i, name in CLASS_NAMES.items():
        row = fr2.confusion_matrix[i]
        w(f"  {name:25s}  {row[0]:8d}  {row[1]:9d}  {row[2]:7d}")

    # Misclassifications by level type
    w("")
    w("  --- Misclassifications by Level Type ---")
    test_df["correct"] = fr2.y_true == fr2.y_pred
    if "ctx_level_type" in test_df.columns:
        ct = pd.crosstab(test_df["ctx_level_type"], test_df["correct"])
        ct.columns = ["Wrong", "Correct"]
        for level_type in ct.index:
            wrong = ct.loc[level_type, "Wrong"] if "Wrong" in ct.columns else 0
            correct = ct.loc[level_type, "Correct"] if "Correct" in ct.columns else 0
            total = wrong + correct
            w(f"    {level_type:20s}: {correct}/{total} correct")

    # Misclassifications by session
    w("")
    w("  --- Misclassifications by Session ---")
    if "ctx_session" in test_df.columns:
        ct = pd.crosstab(test_df["ctx_session"], test_df["correct"])
        ct.columns = ["Wrong", "Correct"]
        for session in ct.index:
            wrong = ct.loc[session, "Wrong"] if "Wrong" in ct.columns else 0
            correct = ct.loc[session, "Correct"] if "Correct" in ct.columns else 0
            total = wrong + correct
            w(f"    {session:20s}: {correct}/{total} correct")

    # Feature comparison: Fold 2 test set vs other folds' test sets
    w("")
    w("  --- Top Feature Values: Fold 2 vs Other Folds ---")
    other_test_indices = np.concatenate([
        f.test_indices for i, f in enumerate(folds) if i != 2
    ])
    other_df = df.iloc[other_test_indices]

    numeric_top = [f for f in TOP_FEATURES if f in df.columns and df[f].dtype != object]
    for feat in numeric_top:
        f2_mean = test_df[feat].mean()
        f2_std = test_df[feat].std()
        other_mean = other_df[feat].mean()
        other_std = other_df[feat].std()
        w(f"    {feat:35s}  Fold2: {f2_mean:8.2f} +/-{f2_std:6.2f}  "
          f"Others: {other_mean:8.2f} +/-{other_std:6.2f}")

    w("")
    text = "\n".join(lines)
    (out_dir / "fold2_deep_dive.txt").write_text(text)
    return text


# ── 2. Binary Classification Test ────────────────────────────


def _train_fold_binary(
    df: pd.DataFrame,
    fold: WalkForwardFold,
    feature_cols: list[str],
    cat_features: list[str],
) -> tuple[object, FoldResult]:
    """Train binary CatBoost: reversal (1) vs no_reversal (0)."""
    from catboost import CatBoostClassifier, Pool

    X_train = df.iloc[fold.train_indices][feature_cols].reset_index(drop=True)
    y_train = df.iloc[fold.train_indices]["label_binary"].reset_index(drop=True)
    X_test = df.iloc[fold.test_indices][feature_cols].reset_index(drop=True)
    y_test = df.iloc[fold.test_indices]["label_binary"].reset_index(drop=True)

    n_train = len(X_train)
    split_idx = int(n_train * 0.8)

    X_fit, X_eval = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_fit, y_eval = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    cat_indices = [
        list(X_train.columns).index(c) for c in cat_features
        if c in X_train.columns
    ]

    model = CatBoostClassifier(
        iterations=1000,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=5,
        loss_function="Logloss",
        eval_metric="Accuracy",
        has_time=True,
        cat_features=cat_indices,
        early_stopping_rounds=50,
        verbose=0,
        random_seed=42,
        auto_class_weights="Balanced",
    )

    train_pool = Pool(X_fit, y_fit, cat_features=cat_indices)
    eval_pool = Pool(X_eval, y_eval, cat_features=cat_indices)
    model.fit(train_pool, eval_set=eval_pool, verbose=0)

    y_pred = model.predict(X_test).flatten().astype(int)
    y_true = y_test.values.astype(int)

    accuracy = float(np.mean(y_pred == y_true)) if len(y_true) > 0 else 0.0

    # Reversal = class 1 in binary
    rev_predicted = y_pred == 1
    reversal_precision = (
        float(np.sum(y_true[rev_predicted] == 1) / rev_predicted.sum())
        if rev_predicted.sum() > 0 else 0.0
    )
    rev_actual = y_true == 1
    reversal_recall = (
        float(np.sum(y_pred[rev_actual] == 1) / rev_actual.sum())
        if rev_actual.sum() > 0 else 0.0
    )

    # 2x2 confusion matrix stored in 3x3 for compatibility (only rows/cols 0,1 used)
    cm = np.zeros((2, 2), dtype=int)
    for t in range(2):
        for p in range(2):
            cm[t, p] = int(np.sum((y_true == t) & (y_pred == p)))

    importance_values = model.get_feature_importance()
    importance_dict = dict(zip(feature_cols, importance_values, strict=True))

    fold_result = FoldResult(
        fold=fold.fold,
        n_train=n_train,
        n_test=len(y_test),
        train_dates_str=f"{fold.train_dates[0]} -- {fold.train_dates[-1]}",
        test_dates_str=f"{fold.test_dates[0]} -- {fold.test_dates[-1]}",
        accuracy=accuracy,
        reversal_precision=reversal_precision,
        blowthrough_recall=reversal_recall,  # reuse field for reversal recall
        confusion_matrix=cm,
        feature_importances=importance_dict,
        y_true=y_true.copy(),
        y_pred=y_pred.copy(),
    )

    return model, fold_result


def _binary_classification_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    folds: list[WalkForwardFold],
    out_dir: Path,
) -> str:
    """Re-run with binary labels: reversal=1, no_reversal=0."""
    lines: list[str] = []
    w = lines.append

    # Create binary labels
    df = df.copy()
    df["label_binary"] = (df["label_encoded"] == CLASS_REVERSAL).astype(int)

    w("=" * 70)
    w("  DIAGNOSTIC 2: BINARY CLASSIFICATION (reversal vs no_reversal)")
    w("=" * 70)
    w("")
    w("  Labels: reversal=1 (tradeable_reversal), no_reversal=0 (trap + blowthrough)")
    w("")

    fold_results: list[FoldResult] = []
    for fold in folds:
        _, fr = _train_fold_binary(df, fold, feature_cols, cat_features)
        fold_results.append(fr)

    # Per-fold table
    w(f"  {'Fold':>4}  {'N_test':>6}  {'Accuracy':>8}  "
      f"{'Rev Prec':>8}  {'Rev Recall':>10}")
    w(f"  {'----':>4}  {'------':>6}  {'--------':>8}  "
      f"{'--------':>8}  {'----------':>10}")
    for fr in fold_results:
        w(f"  {fr.fold:4d}  {fr.n_test:6d}  {fr.accuracy:8.3f}  "
          f"{fr.reversal_precision:8.3f}  {fr.blowthrough_recall:10.3f}")

    # Pooled metrics
    all_y_true = np.concatenate([fr.y_true for fr in fold_results])
    all_y_pred = np.concatenate([fr.y_pred for fr in fold_results])

    pooled_accuracy = float(np.mean(all_y_true == all_y_pred))
    rev_pred = all_y_pred == 1
    pooled_rev_precision = (
        float(np.sum(all_y_true[rev_pred] == 1) / rev_pred.sum())
        if rev_pred.sum() > 0 else 0.0
    )
    rev_actual = all_y_true == 1
    pooled_rev_recall = (
        float(np.sum(all_y_pred[rev_actual] == 1) / rev_actual.sum())
        if rev_actual.sum() > 0 else 0.0
    )

    fold_accs = [fr.accuracy for fr in fold_results]
    cross_fold_std = float(np.std(fold_accs))

    w("")
    w("  --- Pooled Binary Results ---")
    w(f"    Pooled accuracy:       {pooled_accuracy:.4f}")
    w(f"    Reversal precision:    {pooled_rev_precision:.4f}")
    w(f"    Reversal recall:       {pooled_rev_recall:.4f}")
    w(f"    Cross-fold acc std:    {cross_fold_std:.4f}")
    w("")

    # Aggregated 2x2 confusion matrix
    agg_cm = np.zeros((2, 2), dtype=int)
    for t in range(2):
        for p in range(2):
            agg_cm[t, p] = int(np.sum((all_y_true == t) & (all_y_pred == p)))

    w("  --- Aggregated Binary Confusion Matrix ---")
    w(f"  {'Actual \\ Predicted':25s}  {'Pred NoRev':>10}  {'Pred Rev':>8}")
    w(f"  {'no_reversal':25s}  {agg_cm[0, 0]:10d}  {agg_cm[0, 1]:8d}")
    w(f"  {'reversal':25s}  {agg_cm[1, 0]:10d}  {agg_cm[1, 1]:8d}")
    w("")

    # Comparison with 3-class
    w("  --- Binary vs 3-Class Comparison ---")
    w(f"    3-class pooled accuracy:   0.4854")
    w(f"    Binary pooled accuracy:    {pooled_accuracy:.4f}")
    w(f"    3-class rev precision:     0.7442")
    w(f"    Binary rev precision:      {pooled_rev_precision:.4f}")
    w(f"    3-class cross-fold std:    0.1350")
    w(f"    Binary cross-fold std:     {cross_fold_std:.4f}")
    w("")

    text = "\n".join(lines)
    (out_dir / "binary_results.txt").write_text(text)

    cm_df = pd.DataFrame(
        agg_cm,
        index=["no_reversal", "reversal"],
        columns=["pred_no_reversal", "pred_reversal"],
    )
    cm_df.to_csv(out_dir / "binary_confusion.csv")

    return text


# ── 3. Top-3 Feature Only Model ──────────────────────────────


def _top3_feature_model(
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    out_dir: Path,
) -> str:
    """Re-run 3-class with only the top 3 features."""
    lines: list[str] = []
    w = lines.append

    w("=" * 70)
    w("  DIAGNOSTIC 3: TOP-3 FEATURE ONLY MODEL")
    w("=" * 70)
    w("")
    w(f"  Features: {', '.join(TOP_3_FEATURES)}")
    w(f"  No categorical features (all 3 are numeric)")
    w("")

    fold_results: list[FoldResult] = []
    for fold in folds:
        _, fr = train_fold(df, fold, TOP_3_FEATURES, cat_features=[])
        fold_results.append(fr)

    # Per-fold table
    w(f"  {'Fold':>4}  {'N_test':>6}  {'Accuracy':>8}  "
      f"{'Rev Prec':>8}  {'BT Recall':>9}")
    w(f"  {'----':>4}  {'------':>6}  {'--------':>8}  "
      f"{'--------':>8}  {'---------':>9}")
    for fr in fold_results:
        w(f"  {fr.fold:4d}  {fr.n_test:6d}  {fr.accuracy:8.3f}  "
          f"{fr.reversal_precision:8.3f}  {fr.blowthrough_recall:9.3f}")

    # Pooled metrics
    all_y_true = np.concatenate([fr.y_true for fr in fold_results])
    all_y_pred = np.concatenate([fr.y_pred for fr in fold_results])

    pooled_accuracy = float(np.mean(all_y_true == all_y_pred))
    rev_pred = all_y_pred == CLASS_REVERSAL
    pooled_rev_precision = (
        float(np.sum(all_y_true[rev_pred] == CLASS_REVERSAL) / rev_pred.sum())
        if rev_pred.sum() > 0 else 0.0
    )
    bt_actual = all_y_true == CLASS_BLOWTHROUGH
    pooled_bt_recall = (
        float(np.sum(all_y_pred[bt_actual] == CLASS_BLOWTHROUGH) / bt_actual.sum())
        if bt_actual.sum() > 0 else 0.0
    )

    fold_accs = [fr.accuracy for fr in fold_results]
    cross_fold_std = float(np.std(fold_accs))

    w("")
    w("  --- Pooled Top-3 Results ---")
    w(f"    Pooled accuracy:       {pooled_accuracy:.4f}")
    w(f"    Reversal precision:    {pooled_rev_precision:.4f}")
    w(f"    Blowthrough recall:    {pooled_bt_recall:.4f}")
    w(f"    Cross-fold acc std:    {cross_fold_std:.4f}")
    w("")

    # Aggregated confusion matrix
    agg_cm = _confusion_matrix_3class(all_y_true, all_y_pred)
    w("  --- Aggregated Confusion Matrix (Top-3 Model) ---")
    w(f"  {'Actual \\ Predicted':25s}  {'Pred Rev':>8}  {'Pred Trap':>9}  {'Pred BT':>7}")
    for i, name in CLASS_NAMES.items():
        row = agg_cm[i]
        w(f"  {name:25s}  {row[0]:8d}  {row[1]:9d}  {row[2]:7d}")
    w("")

    # Comparison with 58-feature model
    w("  --- Top-3 vs Full 58-Feature Comparison ---")
    w(f"    58-feat pooled accuracy:   0.4854")
    w(f"    Top-3 pooled accuracy:     {pooled_accuracy:.4f}")
    w(f"    58-feat rev precision:     0.7442")
    w(f"    Top-3 rev precision:       {pooled_rev_precision:.4f}")
    w(f"    58-feat BT recall:         0.6429")
    w(f"    Top-3 BT recall:           {pooled_bt_recall:.4f}")
    w(f"    58-feat cross-fold std:    0.1350")
    w(f"    Top-3 cross-fold std:      {cross_fold_std:.4f}")
    w("")

    text = "\n".join(lines)
    (out_dir / "top3_results.txt").write_text(text)

    # Save structured CSVs for dashboard consumption
    fold_rows = []
    for fr in fold_results:
        fold_rows.append({
            "fold": fr.fold,
            "n_train": fr.n_train,
            "n_test": fr.n_test,
            "train_dates": fr.train_dates_str,
            "test_dates": fr.test_dates_str,
            "accuracy": fr.accuracy,
            "reversal_precision": fr.reversal_precision,
            "blowthrough_recall": fr.blowthrough_recall,
        })
    pd.DataFrame(fold_rows).to_csv(out_dir / "top3_fold_results.csv", index=False)

    cm_df = pd.DataFrame(
        agg_cm,
        index=["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"],
        columns=["pred_tradeable_reversal", "pred_trap_reversal", "pred_aggressive_blowthrough"],
    )
    cm_df.to_csv(out_dir / "top3_confusion_matrix.csv")

    return text


# ── 4. Per-Feature Class Distributions ───────────────────────


def _per_feature_distributions(
    df: pd.DataFrame,
    out_dir: Path,
) -> str:
    """Show mean/std per class for top 5 most important features."""
    lines: list[str] = []
    w = lines.append

    w("=" * 70)
    w("  DIAGNOSTIC 4: PER-FEATURE CLASS DISTRIBUTIONS")
    w("=" * 70)
    w("")
    w("  Top 5 features by Phase 5 mean importance:")
    w(f"    {', '.join(TOP_FEATURES)}")
    w("")

    rows_for_csv: list[dict] = []

    for feat in TOP_FEATURES:
        if feat not in df.columns:
            continue

        w(f"  --- {feat} ---")

        if df[feat].dtype == object:
            # Categorical: value counts per class
            w(f"  (Categorical)")
            for cls_id, cls_name in CLASS_NAMES.items():
                cls_mask = df["label_encoded"] == cls_id
                vc = df.loc[cls_mask, feat].value_counts()
                total = cls_mask.sum()
                parts = []
                for val, count in vc.items():
                    parts.append(f"{val}={count}({count / total:.0%})")
                w(f"    {cls_name:25s} (n={total:3d}): {', '.join(parts)}")
                for val, count in vc.items():
                    rows_for_csv.append({
                        "feature": feat,
                        "class": cls_name,
                        "metric": f"count_{val}",
                        "value": count,
                    })
        else:
            # Numeric: mean and std per class
            w(f"  (Numeric)")
            w(f"    {'Class':25s}  {'Mean':>10}  {'Std':>10}  {'Median':>10}  {'N':>5}")
            for cls_id, cls_name in CLASS_NAMES.items():
                cls_vals = df.loc[df["label_encoded"] == cls_id, feat]
                mean_val = cls_vals.mean()
                std_val = cls_vals.std()
                med_val = cls_vals.median()
                w(f"    {cls_name:25s}  {mean_val:10.3f}  {std_val:10.3f}  "
                  f"{med_val:10.3f}  {len(cls_vals):5d}")
                rows_for_csv.append({
                    "feature": feat,
                    "class": cls_name,
                    "metric": "mean",
                    "value": float(mean_val),
                })
                rows_for_csv.append({
                    "feature": feat,
                    "class": cls_name,
                    "metric": "std",
                    "value": float(std_val),
                })
                rows_for_csv.append({
                    "feature": feat,
                    "class": cls_name,
                    "metric": "median",
                    "value": float(med_val),
                })
        w("")

    text = "\n".join(lines)
    (out_dir / "feature_distributions.txt").write_text(text)

    dist_df = pd.DataFrame(rows_for_csv)
    dist_df.to_csv(out_dir / "feature_distributions.csv", index=False)

    return text


# ── 5. Misclassification Patterns ────────────────────────────


def _misclassification_patterns(
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    fold_results: list[FoldResult],
    out_dir: Path,
) -> str:
    """Analyze the two most common misclassification patterns."""
    lines: list[str] = []
    w = lines.append

    w("=" * 70)
    w("  DIAGNOSTIC 5: MISCLASSIFICATION PATTERNS")
    w("=" * 70)
    w("")

    # Gather all test predictions with original indices
    all_test_indices: list[int] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for fold, fr in zip(folds, fold_results):
        all_test_indices.extend(fold.test_indices.tolist())
        all_y_true.extend(fr.y_true.tolist())
        all_y_pred.extend(fr.y_pred.tolist())

    all_test_indices = np.array(all_test_indices)
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Top features for comparison (numeric only)
    compare_features = [f for f in TOP_FEATURES if f in df.columns and df[f].dtype != object]

    # Error type 1: Reversals predicted as Traps (26 from confusion matrix)
    w("  --- Error Type 1: Actual REVERSAL predicted as TRAP ---")
    rev_as_trap = (all_y_true == CLASS_REVERSAL) & (all_y_pred == CLASS_TRAP)
    n_rev_as_trap = rev_as_trap.sum()
    w(f"  Count: {n_rev_as_trap}")
    w("")

    rev_correct = (all_y_true == CLASS_REVERSAL) & (all_y_pred == CLASS_REVERSAL)
    n_rev_correct = rev_correct.sum()

    if n_rev_as_trap > 0 and n_rev_correct > 0:
        misclassified_idx = all_test_indices[rev_as_trap]
        correct_idx = all_test_indices[rev_correct]

        w(f"  Feature comparison (misclassified vs correctly classified reversals):")
        w(f"    {'Feature':35s}  {'Misclass Mean':>13}  {'Correct Mean':>12}  {'Delta':>8}")
        w(f"    {'-------':35s}  {'-------------':>13}  {'------------':>12}  {'-----':>8}")
        for feat in compare_features:
            mis_mean = df.iloc[misclassified_idx][feat].mean()
            cor_mean = df.iloc[correct_idx][feat].mean()
            delta = mis_mean - cor_mean
            w(f"    {feat:35s}  {mis_mean:13.3f}  {cor_mean:12.3f}  {delta:+8.3f}")
        w("")

        # Check if misclassified reversals cluster on specific level types
        w("  Misclassified reversals by level type:")
        if "ctx_level_type" in df.columns:
            vc = df.iloc[misclassified_idx]["ctx_level_type"].value_counts()
            for val, count in vc.items():
                w(f"    {val:20s}: {count}")
        w("")

        w("  Misclassified reversals by session:")
        if "ctx_session" in df.columns:
            vc = df.iloc[misclassified_idx]["ctx_session"].value_counts()
            for val, count in vc.items():
                w(f"    {val:20s}: {count}")
        w("")

    # Error type 2: Traps predicted as Reversals (9 from confusion matrix)
    w("  --- Error Type 2: Actual TRAP predicted as REVERSAL ---")
    trap_as_rev = (all_y_true == CLASS_TRAP) & (all_y_pred == CLASS_REVERSAL)
    n_trap_as_rev = trap_as_rev.sum()
    w(f"  Count: {n_trap_as_rev}")
    w("")

    trap_correct = (all_y_true == CLASS_TRAP) & (all_y_pred == CLASS_TRAP)
    n_trap_correct = trap_correct.sum()

    if n_trap_as_rev > 0 and n_trap_correct > 0:
        misclassified_idx = all_test_indices[trap_as_rev]
        correct_idx = all_test_indices[trap_correct]

        w(f"  Feature comparison (traps misclassified as reversals vs correctly classified traps):")
        w(f"    {'Feature':35s}  {'Misclass Mean':>13}  {'Correct Mean':>12}  {'Delta':>8}")
        w(f"    {'-------':35s}  {'-------------':>13}  {'------------':>12}  {'-----':>8}")
        for feat in compare_features:
            mis_mean = df.iloc[misclassified_idx][feat].mean()
            cor_mean = df.iloc[correct_idx][feat].mean()
            delta = mis_mean - cor_mean
            w(f"    {feat:35s}  {mis_mean:13.3f}  {cor_mean:12.3f}  {delta:+8.3f}")
        w("")

        w("  Traps misclassified as reversals by level type:")
        if "ctx_level_type" in df.columns:
            vc = df.iloc[misclassified_idx]["ctx_level_type"].value_counts()
            for val, count in vc.items():
                w(f"    {val:20s}: {count}")
        w("")

        w("  Traps misclassified as reversals by session:")
        if "ctx_session" in df.columns:
            vc = df.iloc[misclassified_idx]["ctx_session"].value_counts()
            for val, count in vc.items():
                w(f"    {val:20s}: {count}")
        w("")
    elif n_trap_as_rev > 0 and n_trap_correct == 0:
        w("  No correctly classified traps to compare against.")
        w("")

    # Summary of all error types
    w("  --- Full Error Type Summary ---")
    agg_cm = _confusion_matrix_3class(all_y_true, all_y_pred)
    for i, actual_name in CLASS_NAMES.items():
        for j, pred_name in CLASS_NAMES.items():
            if i != j and agg_cm[i, j] > 0:
                w(f"    {actual_name} -> {pred_name}: {agg_cm[i, j]} events")
    w("")

    text = "\n".join(lines)
    (out_dir / "misclassification_analysis.txt").write_text(text)
    return text


# ── 6. Slow Reversal Analysis ─────────────────────────────────


def _slow_reversal_analysis(
    df: pd.DataFrame,
    out_dir: Path,
) -> str:
    """Analyze events where int_time_beyond_level > 100s.

    Identifies which order-flow features best separate reversals from
    traps among these 'slow' events where the top-3 tempo features
    cannot distinguish the two classes.
    """
    lines: list[str] = []
    w = lines.append

    threshold = 100.0
    slow = df[df["int_time_beyond_level"] > threshold].copy()

    w("=" * 70)
    w("  DIAGNOSTIC 6: SLOW REVERSAL ANALYSIS (int_time_beyond_level > 100s)")
    w("=" * 70)
    w("")
    w(f"  Filter: int_time_beyond_level > {threshold:.0f} seconds")
    w(f"  Total events matching: {len(slow)} / {len(df)} ({len(slow)/len(df):.1%})")
    w("")

    # Class distribution
    w("  --- Class Distribution ---")
    for cls_id, cls_name in CLASS_NAMES.items():
        n = int((slow["label_encoded"] == cls_id).sum())
        pct = n / len(slow) * 100 if len(slow) > 0 else 0
        w(f"    {cls_name:25s}: {n:3d} ({pct:.1f}%)")
    w("")

    # Per-class mean for ALL 58 features (numeric only)
    exclude = METADATA_COLS | {"max_mae"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype != object]

    rev_mask = slow["label_encoded"] == CLASS_REVERSAL
    trap_mask = slow["label_encoded"] == CLASS_TRAP

    rev_slow = slow[rev_mask]
    trap_slow = slow[trap_mask]

    if len(rev_slow) == 0 or len(trap_slow) == 0:
        w("  Not enough reversals or traps in slow subset for comparison.")
        text = "\n".join(lines)
        (out_dir / "slow_reversal_analysis.txt").write_text(text)
        return text

    # Compute mean per class and absolute difference
    rows: list[dict] = []
    for feat in feature_cols:
        rev_mean = float(rev_slow[feat].mean())
        trap_mean = float(trap_slow[feat].mean())
        bt_mean = float(slow.loc[slow["label_encoded"] == CLASS_BLOWTHROUGH, feat].mean())
        diff = rev_mean - trap_mean
        rows.append({
            "feature": feat,
            "rev_mean": rev_mean,
            "trap_mean": trap_mean,
            "bt_mean": bt_mean,
            "abs_diff": abs(diff),
            "diff": diff,
        })

    rows.sort(key=lambda r: r["abs_diff"], reverse=True)

    # Top 10 by absolute difference
    w("  --- Top 10 Features by |mean(reversal) - mean(trap)| in Slow Subset ---")
    w("")
    w(f"    {'Rank':>4}  {'Feature':35s}  {'Rev Mean':>10}  "
      f"{'Trap Mean':>10}  {'BT Mean':>10}  {'Delta':>8}")
    w(f"    {'----':>4}  {'-------':35s}  {'--------':>10}  "
      f"{'--------':>10}  {'-------':>10}  {'-----':>8}")
    for i, r in enumerate(rows[:10]):
        w(f"    {i+1:4d}  {r['feature']:35s}  {r['rev_mean']:10.3f}  "
          f"{r['trap_mean']:10.3f}  {r['bt_mean']:10.3f}  {r['diff']:+8.3f}")
    w("")

    # Check specifically requested features
    requested = [
        "int_aggression_flip",
        "int_book_imbalance_shift",
        "int_sweep_volume",
        "int_buy_sell_ratio",
        "app_buy_sell_ratio",
        "int_cancel_burst",
    ]
    top10_names = {r["feature"] for r in rows[:10]}

    w("  --- Requested Features (in top 10?) ---")
    w("")
    w(f"    {'Feature':35s}  {'Rev Mean':>10}  {'Trap Mean':>10}  "
      f"{'Delta':>8}  {'Rank':>5}  {'Top 10?'}")
    w(f"    {'-------':35s}  {'--------':>10}  {'--------':>10}  "
      f"{'-----':>8}  {'----':>5}  {'-------'}")
    for feat in requested:
        match = [r for r in rows if r["feature"] == feat]
        if match:
            r = match[0]
            rank = rows.index(r) + 1
            in_top = "YES" if feat in top10_names else "no"
            w(f"    {feat:35s}  {r['rev_mean']:10.3f}  {r['trap_mean']:10.3f}  "
              f"{r['diff']:+8.3f}  {rank:5d}  {in_top}")
        else:
            w(f"    {feat:35s}  (not found in numeric features)")
    w("")

    # Full table for CSV
    full_df = pd.DataFrame(rows)
    full_df.to_csv(out_dir / "slow_reversal_features.csv", index=False)

    text = "\n".join(lines)
    (out_dir / "slow_reversal_analysis.txt").write_text(text)
    return text


# ── 7. Session Accuracy Breakdown (Top-3 Model) ──────────────


def _session_accuracy_top3(
    df: pd.DataFrame,
    folds: list[WalkForwardFold],
    out_dir: Path,
) -> str:
    """Break down top-3 model accuracy by ctx_session."""
    lines: list[str] = []
    w = lines.append

    # Re-run top-3 model to get pooled predictions with indices
    top3_fold_results: list[FoldResult] = []
    for fold in folds:
        _, fr = train_fold(df, fold, TOP_3_FEATURES, cat_features=[])
        top3_fold_results.append(fr)

    # Gather pooled predictions with original indices
    all_test_indices: list[int] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for fold, fr in zip(folds, top3_fold_results):
        all_test_indices.extend(fold.test_indices.tolist())
        all_y_true.extend(fr.y_true.tolist())
        all_y_pred.extend(fr.y_pred.tolist())

    all_test_indices = np.array(all_test_indices)
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    w("=" * 70)
    w("  DIAGNOSTIC 7: SESSION ACCURACY BREAKDOWN (Top-3 Model)")
    w("=" * 70)
    w("")

    # Get session for each test prediction
    sessions = df.iloc[all_test_indices]["ctx_session"].values

    unique_sessions = sorted(set(sessions))
    w(f"  {'Session':15s}  {'N':>5}  {'Accuracy':>8}  "
      f"{'Rev Prec':>8}  {'Rev Recall':>10}  "
      f"{'Rev N':>5}  {'Trap N':>6}  {'BT N':>4}")
    w(f"  {'-------':15s}  {'---':>5}  {'--------':>8}  "
      f"{'--------':>8}  {'----------':>10}  "
      f"{'-----':>5}  {'------':>6}  {'----':>4}")

    session_rows: list[dict] = []
    for session in unique_sessions:
        mask = sessions == session
        y_t = all_y_true[mask]
        y_p = all_y_pred[mask]
        n = len(y_t)

        acc = float(np.mean(y_t == y_p)) if n > 0 else 0.0

        # Reversal precision for this session
        rev_pred = y_p == CLASS_REVERSAL
        rev_prec = (
            float(np.sum(y_t[rev_pred] == CLASS_REVERSAL) / rev_pred.sum())
            if rev_pred.sum() > 0 else float("nan")
        )

        # Reversal recall for this session
        rev_actual = y_t == CLASS_REVERSAL
        rev_recall = (
            float(np.sum(y_p[rev_actual] == CLASS_REVERSAL) / rev_actual.sum())
            if rev_actual.sum() > 0 else float("nan")
        )

        n_rev = int((y_t == CLASS_REVERSAL).sum())
        n_trap = int((y_t == CLASS_TRAP).sum())
        n_bt = int((y_t == CLASS_BLOWTHROUGH).sum())

        rev_prec_str = f"{rev_prec:8.3f}" if not np.isnan(rev_prec) else "     N/A"
        rev_recall_str = f"{rev_recall:10.3f}" if not np.isnan(rev_recall) else "       N/A"

        w(f"  {session:15s}  {n:5d}  {acc:8.3f}  "
          f"{rev_prec_str}  {rev_recall_str}  "
          f"{n_rev:5d}  {n_trap:6d}  {n_bt:4d}")

        session_rows.append({
            "session": session,
            "n": n,
            "accuracy": acc,
            "reversal_precision": rev_prec,
            "reversal_recall": rev_recall,
            "n_reversal": n_rev,
            "n_trap": n_trap,
            "n_blowthrough": n_bt,
        })

    w("")

    # Pooled total for reference
    total_acc = float(np.mean(all_y_true == all_y_pred))
    w(f"  {'TOTAL':15s}  {len(all_y_true):5d}  {total_acc:8.3f}")
    w("")

    # Per-session confusion matrices for the two worst sessions
    w("  --- Per-Session Confusion Matrices ---")
    for session in unique_sessions:
        mask = sessions == session
        y_t = all_y_true[mask]
        y_p = all_y_pred[mask]
        if len(y_t) < 3:
            continue
        cm = _confusion_matrix_3class(y_t, y_p)
        w(f"\n  {session}:")
        w(f"    {'':25s}  {'Pred Rev':>8}  {'Pred Trap':>9}  {'Pred BT':>7}")
        for i, name in CLASS_NAMES.items():
            row = cm[i]
            w(f"    {name:25s}  {row[0]:8d}  {row[1]:9d}  {row[2]:7d}")
    w("")

    # Misclassification error rates by session
    w("  --- Error Analysis: Reversal<->Trap Confusion by Session ---")
    for session in unique_sessions:
        mask = sessions == session
        y_t = all_y_true[mask]
        y_p = all_y_pred[mask]

        rev_as_trap = int(np.sum((y_t == CLASS_REVERSAL) & (y_p == CLASS_TRAP)))
        trap_as_rev = int(np.sum((y_t == CLASS_TRAP) & (y_p == CLASS_REVERSAL)))
        n_rev = int((y_t == CLASS_REVERSAL).sum())
        n_trap = int((y_t == CLASS_TRAP).sum())

        rev_trap_rate = rev_as_trap / n_rev if n_rev > 0 else 0
        trap_rev_rate = trap_as_rev / n_trap if n_trap > 0 else 0

        w(f"    {session:15s}: rev->trap {rev_as_trap}/{n_rev} ({rev_trap_rate:.0%}), "
          f"trap->rev {trap_as_rev}/{n_trap} ({trap_rev_rate:.0%})")
    w("")

    pd.DataFrame(session_rows).to_csv(out_dir / "session_accuracy.csv", index=False)

    text = "\n".join(lines)
    (out_dir / "session_accuracy.txt").write_text(text)
    return text


# ── Main Orchestrator ─────────────────────────────────────────


def run_diagnostics(
    feature_matrix_path: Path = Path("data/experiment/feature_matrix.parquet"),
    labeled_events_path: Path = Path("data/experiment/labeled_events.parquet"),
    output_dir: Path = Path("data/experiment/training_results/diagnostics"),
) -> None:
    """Run all 5 diagnostic analyses.

    First re-runs the 3-class walk-forward training to get fold-level
    predictions, then runs each diagnostic analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_parquet(feature_matrix_path)
    labeled = pd.read_parquet(labeled_events_path)

    # Attach max_mae
    resolved = labeled[labeled["label"] != "no_resolution"].reset_index(drop=True)
    if "max_mae" in resolved.columns and len(resolved) == len(df):
        df["max_mae"] = resolved["max_mae"].values

    # Identify feature columns
    exclude = METADATA_COLS | {"max_mae"}
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_feature_names = [c for c in CAT_FEATURES if c in feature_cols]

    # Sort chronologically
    df = df.sort_values(["date", "event_ts"]).reset_index(drop=True)

    # Create same walk-forward folds as Phase 5
    folds = create_walk_forward_folds(df)
    logger.info("Created %d walk-forward folds for diagnostics", len(folds))

    # Re-run 3-class training to get fold-level predictions
    print("\n  Re-running 3-class walk-forward training for diagnostics...\n")
    fold_results: list[FoldResult] = []
    for fold in folds:
        _, fr = train_fold(df, fold, feature_cols, cat_feature_names)
        fold_results.append(fr)

    # Run all 5 diagnostics
    print("\n")
    sections: list[str] = []

    # 1. Fold 2 deep dive
    text = _fold2_deep_dive(df, folds, fold_results, output_dir)
    sections.append(text)
    print(text)

    # 2. Binary classification
    print("\n  Running binary classification test...\n")
    text = _binary_classification_test(df, feature_cols, cat_feature_names, folds, output_dir)
    sections.append(text)
    print(text)

    # 3. Top-3 feature model
    print("\n  Running top-3 feature model...\n")
    text = _top3_feature_model(df, folds, output_dir)
    sections.append(text)
    print(text)

    # 4. Feature distributions
    text = _per_feature_distributions(df, output_dir)
    sections.append(text)
    print(text)

    # 5. Misclassification patterns
    text = _misclassification_patterns(df, folds, fold_results, output_dir)
    sections.append(text)
    print(text)

    # Final summary
    summary = []
    summary.append("=" * 70)
    summary.append("  PHASE 6 DIAGNOSTICS — SUMMARY OF FINDINGS")
    summary.append("=" * 70)
    summary.append("")
    summary.append("  All diagnostic files saved to:")
    summary.append(f"    {output_dir}")
    summary.append("")
    summary.append("  Files written:")
    for f in sorted(output_dir.iterdir()):
        summary.append(f"    - {f.name}")
    summary.append("")

    summary_text = "\n".join(summary)
    print(summary_text)

    # Save combined report
    full_report = "\n\n".join(sections) + "\n\n" + summary_text
    (output_dir / "full_diagnostic_report.txt").write_text(full_report)

    logger.info("Diagnostics complete. Results in %s", output_dir)


# ── Top-3 Predictions Export ──────────────────────────────────


def _generate_top3_predictions(
    df: pd.DataFrame,
    labeled_events: pd.DataFrame,
    folds: list[WalkForwardFold],
    out_dir: Path,
) -> pd.DataFrame:
    """Re-run top-3 model and save per-event predictions with probabilities.

    Produces a parquet with: event_ts, date, level_name, level_price,
    direction, actual_label, predicted_label, prob_reversal, prob_trap,
    prob_blowthrough, mfe, mae, fold, correct.
    """
    import json as _json

    from catboost import Pool

    rows: list[dict] = []

    for fold in folds:
        model, fr = train_fold(df, fold, TOP_3_FEATURES, cat_features=[])

        # Reconstruct X_test for predict_proba
        X_test = df.iloc[fold.test_indices][TOP_3_FEATURES].reset_index(drop=True)
        test_pool = Pool(X_test)
        proba = model.predict_proba(test_pool)  # (n_test, 3)

        for i, idx in enumerate(fold.test_indices):
            le_row = labeled_events.iloc[idx]

            # Parse level_name from JSON list
            try:
                names = _json.loads(le_row["level_names"])
                level_name = names[0] if names else ""
            except (ValueError, TypeError, KeyError):
                level_name = ""

            rows.append({
                "event_ts": le_row["event_ts"],
                "date": le_row["date"],
                "level_name": level_name,
                "level_price": float(le_row["representative_price"]),
                "direction": le_row["direction"],
                "actual_label": le_row["label"],
                "predicted_label": CLASS_NAMES[int(fr.y_pred[i])],
                "prob_reversal": float(proba[i, CLASS_REVERSAL]),
                "prob_trap": float(proba[i, CLASS_TRAP]),
                "prob_blowthrough": float(proba[i, CLASS_BLOWTHROUGH]),
                "mfe": float(le_row["max_mfe"]),
                "mae": float(le_row["max_mae"]),
                "fold": fold.fold,
                "correct": bool(fr.y_true[i] == fr.y_pred[i]),
            })

    result = pd.DataFrame(rows)
    result.to_parquet(out_dir / "top3_predictions.parquet", index=False)
    print(f"\n  Saved {len(result)} predictions to {out_dir / 'top3_predictions.parquet'}")
    print(f"  Columns: {list(result.columns)}")
    print(f"  Folds: {sorted(result['fold'].unique())}")
    print(f"  Correct: {result['correct'].sum()}/{len(result)} ({result['correct'].mean():.1%})")
    return result


# ── Follow-Up Diagnostics ─────────────────────────────────────


def run_followup_diagnostics(
    feature_matrix_path: Path = Path("data/experiment/feature_matrix.parquet"),
    output_dir: Path = Path("data/experiment/training_results/diagnostics"),
) -> None:
    """Run follow-up analyses: slow reversal + session accuracy."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_matrix_path)
    df = df.sort_values(["date", "event_ts"]).reset_index(drop=True)

    folds = create_walk_forward_folds(df)

    # 6. Slow reversal analysis (no model training needed)
    text = _slow_reversal_analysis(df, output_dir)
    print(text)

    # 7. Session accuracy breakdown (re-runs top-3 model)
    print("\n  Running top-3 model for session breakdown...\n")
    text = _session_accuracy_top3(df, folds, output_dir)
    print(text)


# ── CLI Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "--followup":
        run_followup_diagnostics()
    elif len(sys.argv) > 1 and sys.argv[1] == "--predictions":
        # Generate top-3 predictions with probability scores
        fm_path = Path("data/experiment/feature_matrix.parquet")
        le_path = Path("data/experiment/labeled_events.parquet")
        out = Path("data/experiment/training_results")
        out.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(fm_path)
        df = df.sort_values(["date", "event_ts"]).reset_index(drop=True)

        le = pd.read_parquet(le_path)
        le = le[le["label"] != "no_resolution"].reset_index(drop=True)
        le = le.sort_values(["date", "event_ts"]).reset_index(drop=True)

        folds = create_walk_forward_folds(df)
        _generate_top3_predictions(df, le, folds, out)
    else:
        run_diagnostics()
