"""
Train a 3-feature CatBoost model for the live dashboard.

Features: int_time_beyond_level, int_time_within_2pts, int_absorption_ratio
Classes:  0=tradeable_reversal, 1=trap_reversal, 2=aggressive_blowthrough

Uses purged walk-forward validation (gap between train/test to prevent
label leakage from overlapping forward windows).

Output: data/models/dashboard_3feature_v1.cbm
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

# ── Constants ─────────────────────────────────────────────────

FEATURE_COLS = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
]

CLASS_NAMES = {
    0: "tradeable_reversal",
    1: "trap_reversal",
    2: "aggressive_blowthrough",
}

# Walk-forward settings
TRAIN_DAYS = 40
TEST_DAYS = 5
STEP_DAYS = 5
PURGE_DAYS = 2  # Gap between train/test to prevent label leakage
MIN_TRAIN_EVENTS = 30

# Paths
FEATURE_MATRIX_PATH = Path("data/experiment/feature_matrix.parquet")
OUTPUT_MODEL_PATH = Path("data/models/dashboard_3feature_v1.cbm")


# ── Purged Walk-Forward Splitter ──────────────────────────────


def create_purged_walk_forward_folds(
    df: pd.DataFrame,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    step_days: int = STEP_DAYS,
    purge_days: int = PURGE_DAYS,
    min_train_events: int = MIN_TRAIN_EVENTS,
) -> list[dict]:
    """Create walk-forward folds with a purge gap between train and test.

    The purge gap removes events between the train and test windows to
    prevent label leakage from overlapping forward excursion windows.
    """
    trading_dates = sorted(df["date"].unique())
    n_dates = len(trading_dates)
    folds = []
    fold_idx = 0
    start = 0

    while start + train_days + purge_days + test_days <= n_dates:
        train_end = start + train_days
        test_start = train_end + purge_days  # Skip purge_days
        test_end = test_start + test_days

        if test_end > n_dates:
            break

        train_dates = list(trading_dates[start:train_end])
        test_dates = list(trading_dates[test_start:test_end])

        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"].isin(test_dates)

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) < min_train_events:
            start += step_days
            continue

        if len(test_idx) == 0:
            start += step_days
            continue

        folds.append({
            "fold": fold_idx,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "train_idx": train_idx,
            "test_idx": test_idx,
        })
        fold_idx += 1
        start += step_days

    return folds


# ── Confusion Matrix ─────────────────────────────────────────


def confusion_matrix_3class(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=int)
    for t in range(3):
        for p in range(3):
            cm[t, p] = int(np.sum((y_true == t) & (y_pred == p)))
    return cm


# ── Main ──────────────────────────────────────────────────────


def main() -> None:
    # 1. Load data
    print("Loading feature matrix...")
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  Trading days: {df['date'].nunique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Label distribution:")
    for label, cnt in df["label"].value_counts().sort_index().items():
        print(f"    {label:30s}  {cnt:4d}  ({100*cnt/len(df):.1f}%)")

    # Handle NaN in absorption_ratio (4 events)
    nan_count = df[FEATURE_COLS].isna().any(axis=1).sum()
    if nan_count > 0:
        print(f"\n  Dropping {nan_count} events with NaN features")
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        print(f"  Remaining: {len(df)} events")

    # Sort chronologically
    df = df.sort_values(["date", "event_ts"]).reset_index(drop=True)

    # 2. Create purged walk-forward folds
    folds = create_purged_walk_forward_folds(df)
    print(f"\n  Created {len(folds)} purged walk-forward folds "
          f"(train={TRAIN_DAYS}d, purge={PURGE_DAYS}d, test={TEST_DAYS}d, step={STEP_DAYS}d)")

    if not folds:
        raise ValueError("No valid folds created — not enough data")

    # 3. Train per fold and collect results
    all_y_true = []
    all_y_pred = []
    fold_models = []

    print(f"\n{'=' * 72}")
    print("  WALK-FORWARD TRAINING (purged, 3-feature dashboard model)")
    print(f"{'=' * 72}")
    print(f"\n  {'Fold':>4}  {'N_train':>7}  {'N_test':>6}  "
          f"{'Accuracy':>8}  {'Rev Prec':>8}  {'BT Recall':>9}  {'Test Dates'}")
    print(f"  {'----':>4}  {'-------':>7}  {'------':>6}  "
          f"{'--------':>8}  {'--------':>8}  {'---------':>9}  {'----------'}")

    for fold in folds:
        X_train = df.iloc[fold["train_idx"]][FEATURE_COLS]
        y_train = df.iloc[fold["train_idx"]]["label_encoded"]
        X_test = df.iloc[fold["test_idx"]][FEATURE_COLS]
        y_test = df.iloc[fold["test_idx"]]["label_encoded"]

        model = CatBoostClassifier(
            iterations=200,
            depth=4,
            l2_leaf_reg=5,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            has_time=True,
            verbose=50,
            random_seed=42,
            auto_class_weights="Balanced",
            allow_writing_files=False,
        )

        train_pool = Pool(X_train, y_train)
        model.fit(train_pool, verbose=50)

        y_pred = model.predict(X_test).flatten().astype(int)
        y_true = y_test.values.astype(int)

        accuracy = float(np.mean(y_pred == y_true))

        rev_pred_mask = y_pred == 0
        rev_prec = (
            float(np.sum(y_true[rev_pred_mask] == 0) / rev_pred_mask.sum())
            if rev_pred_mask.sum() > 0 else 0.0
        )

        bt_actual_mask = y_true == 2
        bt_recall = (
            float(np.sum(y_pred[bt_actual_mask] == 2) / bt_actual_mask.sum())
            if bt_actual_mask.sum() > 0 else 0.0
        )

        test_dates_str = f"{fold['test_dates'][0]} -- {fold['test_dates'][-1]}"
        print(
            f"  {fold['fold']:4d}  {len(X_train):7d}  {len(X_test):6d}  "
            f"{accuracy:8.3f}  {rev_prec:8.3f}  {bt_recall:9.3f}  {test_dates_str}"
        )

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        fold_models.append(model)

    # 4. Aggregate results
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    cm = confusion_matrix_3class(all_y_true, all_y_pred)

    overall_accuracy = float(np.mean(all_y_true == all_y_pred))
    total_test = len(all_y_true)

    print(f"\n  --- Aggregated Confusion Matrix (pooled, n={total_test}) ---\n")
    print(f"  {'Actual \\ Predicted':25s}  {'Pred Rev':>8}  {'Pred Trap':>9}  {'Pred BT':>7}")
    print(f"  {'':25s}  {'--------':>8}  {'---------':>9}  {'-------':>7}")
    for i, name in CLASS_NAMES.items():
        row = cm[i]
        total_row = row.sum()
        acc = row[i] / total_row * 100 if total_row > 0 else 0
        print(f"  {name:25s}  {row[0]:8d}  {row[1]:9d}  {row[2]:7d}   ({acc:.1f}% correct)")

    print(f"\n  Overall accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
    print(f"  Total test samples: {total_test}")
    print(f"  Total training samples: {len(df)}")

    # Per-class accuracy
    print(f"\n  --- Per-Class Accuracy ---")
    for i, name in CLASS_NAMES.items():
        actual_mask = all_y_true == i
        if actual_mask.sum() > 0:
            class_acc = float(np.sum(all_y_pred[actual_mask] == i) / actual_mask.sum())
            print(f"    {name:30s}  {class_acc:.3f}  ({actual_mask.sum()} samples)")

    # 5. Train final model on ALL data
    print(f"\n{'=' * 72}")
    print("  TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'=' * 72}\n")

    X_all = df[FEATURE_COLS]
    y_all = df["label_encoded"]

    final_model = CatBoostClassifier(
        iterations=200,
        depth=4,
        l2_leaf_reg=5,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        has_time=True,
        verbose=50,
        random_seed=42,
        auto_class_weights="Balanced",
        allow_writing_files=False,
    )

    final_pool = Pool(X_all, y_all)
    final_model.fit(final_pool, verbose=50)

    # Save model
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(OUTPUT_MODEL_PATH))
    print(f"\n  Final model saved to: {OUTPUT_MODEL_PATH}")
    print(f"  Model file size: {OUTPUT_MODEL_PATH.stat().st_size:,} bytes")

    # Feature importances
    importances = final_model.get_feature_importance()
    print(f"\n  Feature importances:")
    for feat, imp in zip(FEATURE_COLS, importances):
        print(f"    {feat:30s}  {imp:.2f}")

    # 6. Verification
    print(f"\n{'=' * 72}")
    print("  VERIFICATION")
    print(f"{'=' * 72}\n")

    loaded = CatBoostClassifier()
    loaded.load_model(str(OUTPUT_MODEL_PATH))

    test_input = [[30.0, 45.0, 0.5]]
    pred_class = int(loaded.predict(test_input).flatten()[0])
    pred_probs = loaded.predict_proba(test_input)[0]

    print(f"  Test input: {test_input[0]}")
    print(f"  Predicted class: {pred_class} ({CLASS_NAMES[pred_class]})")
    print(f"  Probabilities:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES.values(), pred_probs)):
        print(f"    {name:30s}  {prob:.6f}")
    print(f"  Sum of probabilities: {sum(pred_probs):.10f}")
    print(f"  Valid (sums to 1.0): {abs(sum(pred_probs) - 1.0) < 1e-6}")
    n_features = len(loaded.feature_names_) if loaded.feature_names_ else loaded.tree_count_
    print(f"  Feature count matches: {n_features == 3}")
    print(f"\n  Dashboard FEATURE_COLUMNS: {FEATURE_COLS}")
    print(f"  Model expects {n_features} features")
    print()


if __name__ == "__main__":
    main()
