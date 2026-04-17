"""
Primary extrema-model training workflow for the Streamlit dashboard.

Orchestrates: local tick data scan -> dataset build -> out-of-sample
walk-forward evaluation -> final CatBoost fit -> model save.

All tick data is read from local Parquet files (no Databento API calls).
Data must be pre-downloaded into data/databento/{symbol}/{date}/mbp10.parquet.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
#  DEFAULTS
# ═══════════════════════════════════════════════════════════════

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[0].parent / "data" / "databento"
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[0].parent / "models"

_LABEL_OPTIONS = {"20-tick (5pt) rebound": "label_20t", "40-tick (10pt) rebound": "label_40t",
                  "60-tick (15pt) rebound": "label_60t"}


# ═══════════════════════════════════════════════════════════════
#  PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════


_TICK_FILENAMES = ["mbp10.parquet", "mbp1.parquet", "trades.parquet"]


def get_available_dates(symbol: str, data_dir: Path) -> list[str]:
    """Scan data directory for available cached dates (any tick schema)."""
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return []
    dates = []
    for d in sorted(symbol_dir.iterdir()):
        if d.is_dir() and any((d / f).exists() for f in _TICK_FILENAMES):
            dates.append(d.name)
    return dates


def get_cached_ml_dates(symbol: str, data_dir: Path) -> list[str]:
    """Return dates that have any cached ml_features_*.parquet file."""
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return []
    cached = []
    for d in sorted(symbol_dir.iterdir()):
        if d.is_dir() and any(d.glob("ml_features_*.parquet")):
            cached.append(d.name)
    return cached


def clear_ml_cache(symbol: str, data_dir: Path, dates: list[str] | None = None) -> int:
    """Delete ALL cached ml_features_*.parquet files. Returns count deleted."""
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return 0
    count = 0
    for d in sorted(symbol_dir.iterdir()):
        if not d.is_dir():
            continue
        if dates is not None and d.name not in dates:
            continue
        for cache_file in d.glob("ml_features_*.parquet"):
            cache_file.unlink()
            count += 1
    return count


def build_training_dataset(
    symbol: str,
    dates: list[str],
    data_dir: Path,
    config,
    progress_bar=None,
) -> pd.DataFrame:
    """Build labeled feature matrix from cached tick data.

    Processes one date at a time for memory efficiency and progress reporting.
    Per-date results are cached as ``ml_features.parquet`` so subsequent builds
    skip the expensive extrema detection + feature extraction (~21s/day → ~0.01s).
    """
    import time as _time

    from alpha_lab.agents.data_infra.ml.dataset_builder import ExtremaDatasetBuilder
    from alpha_lab.agents.data_infra.tick_store import TickStore

    frames: list[pd.DataFrame] = []
    cached_count = 0
    cache_tag = config.dataset_config_hash()

    for i, date_str in enumerate(dates):
        cache_path = data_dir / symbol / date_str / f"ml_features_{cache_tag}.parquet"
        t0 = _time.perf_counter()

        if cache_path.exists():
            # Fast path: read pre-built features (~0.01s vs ~21s)
            df = pd.read_parquet(cache_path)
            cached_count += 1
            elapsed = _time.perf_counter() - t0
            if progress_bar is not None:
                progress_bar.progress(
                    (i + 1) / len(dates),
                    text=f"Cached {date_str} ({i + 1}/{len(dates)}, {elapsed:.1f}s)",
                )
        else:
            # Slow path: fresh TickStore per date avoids UNION schema
            # mismatches across dates with different parquet layouts.
            if progress_bar is not None:
                progress_bar.progress(
                    i / len(dates),
                    text=f"Computing {date_str} ({i + 1}/{len(dates)})...",
                )
            store = TickStore(data_dir)
            store.register_symbol_date(symbol, date_str)
            builder = ExtremaDatasetBuilder(store, config, signal_bundle=None)
            df = builder.build_dataset_daily(symbol, [date_str])
            store.close()
            elapsed = _time.perf_counter() - t0
            if not df.empty:
                # Cache for next time
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_path, index=False)
            if progress_bar is not None:
                progress_bar.progress(
                    (i + 1) / len(dates),
                    text=f"Built {date_str} ({i + 1}/{len(dates)}, {elapsed:.1f}s)",
                )

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    logger.info(
        "Dataset built: %d dates (%d cached, %d computed)",
        len(dates), cached_count, len(dates) - cached_count,
    )
    return pd.concat(frames, ignore_index=True)


def run_walk_forward_training(
    dataset: pd.DataFrame,
    config,
    label_column: str = "label_20t",
) -> dict:
    """Walk-forward train + evaluate.

    Returns dict with the refit runtime model, out-of-sample evaluation
    result, and per-fold diagnostics.
    """
    from alpha_lab.agents.data_infra.ml.model_evaluator import ModelEvaluator
    from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer
    from alpha_lab.agents.data_infra.ml.walk_forward import WalkForwardSplitter

    # Separate features from labels/metadata
    if config.training_mode == "dashboard_utility":
        # Dynamic: detect int_* (interaction) and app_* (approach) columns
        feature_cols = [c for c in dataset.columns
                        if c.startswith(("int_", "app_"))]
    else:
        feature_cols = [c for c in dataset.columns
                        if c.startswith(("pl_", "ms_", "sig_"))]
    if label_column not in dataset.columns:
        msg = f"Label column '{label_column}' not found in dataset"
        raise ValueError(msg)

    # Drop rows with missing labels
    valid = dataset[dataset[label_column].notna()].copy()
    if len(valid) == 0:
        msg = "No valid labeled samples after filtering"
        raise ValueError(msg)

    features = valid[feature_cols]
    y = valid[label_column].astype(int)
    timestamps = pd.to_datetime(valid["timestamp"])

    # Walk-forward splits
    splitter = WalkForwardSplitter(config.walk_forward)
    splits = splitter.split(timestamps)

    if len(splits) < 2:
        data_span = (timestamps.max() - timestamps.min()).days
        wf = config.walk_forward
        min_for_1 = wf.train_days + wf.gap_days + wf.test_days
        min_for_2 = wf.train_days + wf.gap_days + 2 * wf.test_days
        msg = (
            f"Only {len(splits)} walk-forward fold(s). Need at least 2.\n"
            f"Your data spans {data_span} calendar days "
            f"({timestamps.min().date()} to {timestamps.max().date()}).\n"
            f"Current windows: train={wf.train_days}d, test={wf.test_days}d, "
            f"gap={wf.gap_days}d → need {min_for_2} days for 2 folds "
            f"({min_for_1} days for 1 fold).\n"
            f"Fix: use a longer date range, or reduce train/test windows."
        )
        raise ValueError(msg)

    # Build preliminary CV index pairs for RFECV (skip single-class folds).
    preliminary_cv: list[tuple] = []
    for split in splits:
        y_train_fold = y.iloc[split.train_indices]
        if y_train_fold.nunique() >= 2:
            preliminary_cv.append((split.train_indices, split.test_indices))

    if not preliminary_cv:
        class_dist = y.value_counts().to_dict()
        msg = (
            f"All {len(splits)} folds skipped — training data in each fold "
            f"had only one class.\n"
            f"Overall class distribution: {class_dist}\n"
            f"Try a larger train window so each fold captures both classes."
        )
        raise ValueError(msg)

    # RFECV: run ONCE before the walk-forward loop so the same feature
    # subset is used for every per-fold model AND the final saved model.
    selected_features = feature_cols
    if (
        config.model.rfecv_enabled
        and len(preliminary_cv) >= 2
    ):
        rfecv_trainer = ExtremaModelTrainer(config.model)
        rfecv_result = rfecv_trainer.train(features, y, cv_splits=preliminary_cv)
        selected_features = rfecv_result.selected_features
        logger.info(
            "RFECV selected %d/%d features (used for all folds and final model)",
            len(selected_features), len(feature_cols),
        )

    # Per-fold config with RFECV disabled (already done above).
    from alpha_lab.agents.data_infra.ml.config import ModelConfig
    fold_model_config = config.model.model_copy(
        update={"rfecv_enabled": False},
    )

    # Per-fold evaluation
    evaluator = ModelEvaluator(n_bootstrap=1000, n_permutations=500)
    fold_details: list[dict] = []
    fold_predictions: list[dict[str, object]] = []

    # Label purge buffer: remove training rows whose forward labeling
    # window (in ticks) could extend into the test period.  Convert
    # tick-based forward_window to a conservative time buffer.
    # NQ averages ~500-1500 ticks/min during RTH; use 5 min as a safe
    # floor per 5000-tick window, then add the walk-forward gap.
    _fw_minutes = max(5, config.labeling.forward_window // 500)
    _purge_buffer = pd.Timedelta(minutes=_fw_minutes)

    valid_cv_splits: list[tuple] = []
    fold_importances: list[dict[str, float]] = []
    skipped = 0
    n_purged_total = 0
    for split in splits:
        # Purge training rows whose labeling horizon may cross into test
        train_ts = timestamps.iloc[split.train_indices]
        safe_cutoff = split.test_start - _purge_buffer
        safe_mask = train_ts <= safe_cutoff
        purged_train_idx = split.train_indices[safe_mask.values]
        n_purged = len(split.train_indices) - len(purged_train_idx)
        n_purged_total += n_purged

        x_train = features.iloc[purged_train_idx][selected_features]
        y_train = y.iloc[purged_train_idx]
        x_test = features.iloc[split.test_indices][selected_features]
        y_test = y.iloc[split.test_indices]

        # CatBoost requires both classes in training data
        if y_train.nunique() < 2:
            logger.warning(
                "Skipping fold %d: train has single class %s",
                split.fold, y_train.value_counts().to_dict(),
            )
            skipped += 1
            continue

        trainer = ExtremaModelTrainer(fold_model_config)
        fold_model = trainer.train(x_train, y_train)
        raw_preds = fold_model.model.predict(x_test).flatten().astype(int)
        raw_probs = fold_model.model.predict_proba(x_test)

        # For evaluation, binarize: "positive" = the signal class.
        # Extrema mode: class 1 (rebound) is positive.
        # Utility mode: class 0 (tradeable_reversal) is positive.
        if config.training_mode == "dashboard_utility":
            eval_y = (y_test.values == 0).astype(int)  # reversal = 1
            eval_preds = (raw_preds == 0).astype(int)
            eval_prob = raw_probs[:, 0]  # P(tradeable_reversal)
        else:
            eval_y = y_test.values
            eval_preds = raw_preds
            eval_prob = raw_probs[:, 1] if raw_probs.shape[1] > 1 else raw_probs[:, 0]

        # Evaluate — handle single-class test gracefully
        fold_eval = evaluator.evaluate(eval_y, eval_preds, eval_prob)
        valid_cv_splits.append((purged_train_idx, split.test_indices))
        fold_importances.append(fold_model.feature_importances)
        fold_predictions.append({
            "fold": split.fold,
            "y_true": eval_y,
            "y_pred": eval_preds,
            "y_prob": eval_prob,
        })
        fold_details.append({
            "fold": split.fold,
            "train_start": str(split.train_start.date()),
            "train_end": str(split.train_end.date()),
            "test_start": str(split.test_start.date()),
            "test_end": str(split.test_end.date()),
            "n_train": len(split.train_indices),
            "n_test": len(split.test_indices),
            "precision": fold_eval.precision,
            "recall": fold_eval.recall,
            "f1": fold_eval.f1,
            "roc_auc": fold_eval.roc_auc,
        })

    if not valid_cv_splits:
        class_dist = y.value_counts().to_dict()
        msg = (
            f"All {len(splits)} folds skipped — training data in each fold "
            f"had only one class.\n"
            f"Overall class distribution: {class_dist}\n"
            f"Try a larger train window so each fold captures both classes."
        )
        raise ValueError(msg)

    if skipped:
        logger.info(
            "Walk-forward: %d/%d folds valid (%d skipped, single-class train)",
            len(valid_cv_splits), len(splits), skipped,
        )

    # Aggregate true out-of-sample fold predictions for quality gates.
    eval_result = evaluator.evaluate_out_of_sample_folds(fold_predictions)
    oos_summary = evaluator.summarize_out_of_sample_predictions(
        fold_predictions,
    )

    # Feature stability: Spearman rank-correlation of feature importances
    # across folds. Low values indicate the model relies on unstable signals.
    feature_stability = None
    if len(fold_importances) >= 2:
        from scipy.stats import spearmanr
        # Build importance matrix (folds x features)
        all_feats = selected_features
        imp_matrix = []
        for imp_dict in fold_importances:
            imp_matrix.append([imp_dict.get(f, 0.0) for f in all_feats])
        imp_arr = np.array(imp_matrix)
        # Mean pairwise Spearman rank-correlation
        correlations = []
        for i in range(len(imp_arr)):
            for j in range(i + 1, len(imp_arr)):
                rho, _ = spearmanr(imp_arr[i], imp_arr[j])
                if np.isfinite(rho):
                    correlations.append(rho)
        feature_stability = float(np.mean(correlations)) if correlations else None

    # RTH coverage: what fraction of OOS samples fall within NY RTH
    # (09:30-16:15 ET). If low, the evaluation may not represent the
    # execution population (dashboard only executes in NY RTH).
    try:
        from zoneinfo import ZoneInfo
        _et = ZoneInfo("America/New_York")
        oos_ts = pd.concat([
            pd.Series(fp["y_true"]).index.to_series()
            for fp in fold_predictions
        ], ignore_index=True) if False else timestamps.iloc[
            np.concatenate([s.test_indices for s in splits if len(s.test_indices) > 0])
        ]
        oos_et = oos_ts.dt.tz_convert(_et) if oos_ts.dt.tz is not None else oos_ts
        rth_mask = (oos_et.dt.hour >= 9) & (
            (oos_et.dt.hour > 9) | (oos_et.dt.minute >= 30)
        ) & (
            (oos_et.dt.hour < 16) | ((oos_et.dt.hour == 16) & (oos_et.dt.minute <= 15))
        )
        rth_fraction = float(rth_mask.mean()) if len(rth_mask) > 0 else 0.0
    except Exception:
        rth_fraction = None

    # Final model on all data using the same feature subset. RFECV
    # is disabled here because feature selection was already done above.
    final_trainer = ExtremaModelTrainer(fold_model_config)
    final_model = final_trainer.train(features[selected_features], y)

    return {
        "trained_model": final_model,
        "eval_result": eval_result,
        "fold_details": fold_details,
        "feature_cols": feature_cols,
        "selected_features": selected_features,
        "n_total": len(valid),
        "n_purged_total": n_purged_total,
        "feature_stability": feature_stability,
        "n_total_folds": len(splits),
        "n_valid_folds": len(valid_cv_splits),
        "n_skipped_folds": skipped,
        "oos_summary": oos_summary,
        "rth_fraction": rth_fraction,
        "full_dataset_class_balance": (
            {
                "tradeable_reversal": int((y == 0).sum()),
                "trap_reversal": int((y == 1).sum()),
                "aggressive_blowthrough": int((y == 2).sum()),
            }
            if config.training_mode == "dashboard_utility"
            else {
                "rebound": int((y == 1).sum()),
                "crossing": int((y == 0).sum()),
            }
        ),
        "class_balance": {
            "rebound": oos_summary["class_balance_true"]["rebound"],
            "crossing": oos_summary["class_balance_true"]["crossing"],
        },
    }


def check_quality_gates(eval_result) -> dict[str, dict]:
    """Apply quality thresholds to evaluation result."""
    fold_precisions = [
        f.get("precision", 0) for f in eval_result.fold_metrics
    ]
    fold_std = float(np.std(fold_precisions)) if fold_precisions else 1.0
    permutation_p = (
        eval_result.permutation_p_value
        if eval_result.permutation_p_value is not None else 1.0
    )

    gates = {
        "Precision >= 0.55": {
            "passed": eval_result.precision >= 0.55,
            "value": f"{eval_result.precision:.3f}",
            "threshold": "0.55",
        },
        "Permutation p < 0.05": {
            "passed": permutation_p < 0.05,
            "value": f"{permutation_p:.4f}",
            "threshold": "0.05",
        },
        "Fold stability (std < 0.15)": {
            "passed": fold_std < 0.15,
            "value": f"{fold_std:.3f}",
            "threshold": "0.15",
        },
        "ROC-AUC > 0.55": {
            "passed": (eval_result.roc_auc or 0) > 0.55,
            "value": f"{eval_result.roc_auc or 0:.3f}",
            "threshold": "0.55",
        },
        "Brier score < 0.25": {
            "passed": (eval_result.brier_score or 1.0) < 0.25,
            "value": f"{eval_result.brier_score or 1.0:.3f}",
            "threshold": "0.25",
        },
        "Test samples >= 200": {
            "passed": eval_result.n_samples >= 200,
            "value": str(eval_result.n_samples),
            "threshold": "200",
        },
    }

    all_passed = all(g["passed"] for g in gates.values())
    return {"gates": gates, "all_passed": all_passed}


def compute_utility_metrics(
    eval_result,
    tp_points: float = 15.0,
    sl_points: float = 30.0,
) -> dict[str, float]:
    """Compute trade-utility metrics from OOS evaluation result.

    Simulates a strategy that takes the predicted rebound side with
    fixed TP/SL and computes expectancy and profit factor.
    """
    cm = eval_result.confusion_matrix
    tp = cm.get("tp", 0)
    fp = cm.get("fp", 0)

    n_trades = tp + fp
    if n_trades == 0:
        return {"expectancy_pts": 0.0, "profit_factor": 0.0, "n_simulated_trades": 0}

    wins = tp
    losses = fp
    gross_gain = wins * tp_points
    gross_loss = losses * sl_points

    expectancy = (gross_gain - gross_loss) / n_trades
    profit_factor = gross_gain / gross_loss if gross_loss > 0 else float("inf")

    return {
        "expectancy_pts": round(expectancy, 2),
        "profit_factor": round(profit_factor, 3),
        "n_simulated_trades": n_trades,
    }


def save_trained_model(
    trained_model,
    eval_result,
    config,
    output_dir: Path,
    *,
    training_result: dict | None = None,
    dates_used: list[str] | None = None,
) -> Path:
    """Save model artifacts and evaluation results.

    Includes ALL metrics shown in the training UI so the saved artifact
    is a complete record of the training run.
    """
    from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model + metadata
    ExtremaModelTrainer.save_model(trained_model, output_dir)

    # Save evaluation — start with the EvaluationResult dataclass
    eval_dict = asdict(eval_result)
    # Convert numpy types for JSON serialization
    for key, val in eval_dict.items():
        if isinstance(val, (np.integer, np.int64)):
            eval_dict[key] = int(val)
        elif isinstance(val, (np.floating, np.float64)):
            eval_dict[key] = float(val)
        elif isinstance(val, tuple):
            eval_dict[key] = [float(v) for v in val]

    # ── Derived rates (shown in UI but not in EvaluationResult) ──
    cm = eval_result.confusion_matrix
    tp = cm.get("tp", 0)
    fp = cm.get("fp", 0)
    tn = cm.get("tn", 0)
    fn = cm.get("fn", 0)
    eval_dict["specificity_tnr"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    eval_dict["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    eval_dict["predicted_positive_rate"] = (tp + fp) / eval_result.n_samples if eval_result.n_samples > 0 else 0.0

    # ── Trade utility metrics ────────────────────────────────────
    utility_15_15 = compute_utility_metrics(eval_result, tp_points=15.0, sl_points=15.0)
    utility_15_30 = compute_utility_metrics(eval_result, tp_points=15.0, sl_points=30.0)
    eval_dict["trade_utility"] = {
        "expectancy_15_15_pts": utility_15_15["expectancy_pts"],
        "expectancy_15_30_pts": utility_15_30["expectancy_pts"],
        "profit_factor_15_30": utility_15_30["profit_factor"],
        "n_simulated_trades": utility_15_30["n_simulated_trades"],
    }

    # ── Training result extras (RTH, stability, OOS summary) ─────
    if training_result is not None:
        eval_dict["rth_fraction"] = training_result.get("rth_fraction")
        eval_dict["feature_stability"] = training_result.get("feature_stability")
        eval_dict["n_purged_total"] = training_result.get("n_purged_total", 0)
        eval_dict["n_total_folds"] = training_result.get("n_total_folds")
        eval_dict["n_valid_folds"] = training_result.get("n_valid_folds")
        eval_dict["n_skipped_folds"] = training_result.get("n_skipped_folds", 0)
        eval_dict["selected_features"] = training_result.get("selected_features")
        eval_dict["class_balance"] = training_result.get("class_balance")
        eval_dict["full_dataset_class_balance"] = training_result.get("full_dataset_class_balance")

        oos = training_result.get("oos_summary")
        if oos:
            eval_dict["oos_summary"] = oos

    # ── Full pipeline config for reproducibility ─────────────────
    if config is not None:
        eval_dict["training_mode"] = getattr(config, "training_mode", "unknown")
        eval_dict["full_config"] = config.model_dump()

    # ── Training dates for reproducibility ────────────────────────
    if dates_used is not None:
        eval_dict["dates_used"] = dates_used
        eval_dict["date_range"] = {
            "start": dates_used[0] if dates_used else None,
            "end": dates_used[-1] if dates_used else None,
            "count": len(dates_used),
        }

    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(eval_dict, f, indent=2, default=str)

    return output_dir


# ═══════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════


def _chart_feature_importance(trained_model, top_n: int = 20) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    imp = trained_model.feature_importances
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in reversed(sorted_imp)]
    values = [x[1] for x in reversed(sorted_imp)]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color="#26a69a",
    ))
    fig.update_layout(
        height=max(300, top_n * 22),
        template="plotly_dark",
        title=f"Top {top_n} Feature Importances",
        margin=dict(l=200, r=20, t=40, b=20),
        xaxis_title="Importance",
    )
    return fig


def _chart_confusion_matrix(eval_result) -> go.Figure:
    """Heatmap confusion matrix."""
    cm = eval_result.confusion_matrix
    tn = cm.get("tn", 0)
    fp = cm.get("fp", 0)
    fn = cm.get("fn", 0)
    tp = cm.get("tp", 0)

    z = [[tn, fp], [fn, tp]]
    text = [[str(tn), str(fp)], [str(fn), str(tp)]]

    fig = go.Figure(go.Heatmap(
        z=z, x=["Pred Crossing", "Pred Rebound"],
        y=["Actual Crossing", "Actual Rebound"],
        text=text, texttemplate="%{text}",
        colorscale="Teal", showscale=False,
    ))
    fig.update_layout(
        height=300, template="plotly_dark",
        title="Confusion Matrix",
        margin=dict(l=120, r=20, t=40, b=40),
    )
    return fig


def _chart_fold_metrics(fold_details: list[dict]) -> go.Figure:
    """Line chart of per-fold precision and F1."""
    folds = [f["fold"] for f in fold_details]
    precisions = [f["precision"] for f in fold_details]
    f1s = [f["f1"] for f in fold_details]
    recalls = [f["recall"] for f in fold_details]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=folds, y=precisions, name="Precision",
        mode="lines+markers", line=dict(color="#26a69a"),
    ))
    fig.add_trace(go.Scatter(
        x=folds, y=f1s, name="F1",
        mode="lines+markers", line=dict(color="#1E88E5"),
    ))
    fig.add_trace(go.Scatter(
        x=folds, y=recalls, name="Recall",
        mode="lines+markers", line=dict(color="#FFA726"),
    ))
    fig.add_hline(y=0.55, line_dash="dash", line_color="red",
                  annotation_text="Min Precision (0.55)")
    fig.update_layout(
        height=300, template="plotly_dark",
        title="Walk-Forward Fold Metrics",
        xaxis_title="Fold", yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════


def render_ml_training_tab() -> None:
    """Render the ML Training tab content."""
    st.subheader("ML Training Workbench")

    # ── Mode Selector ─────────────────────────────────────────
    training_mode = st.radio(
        "Training Mode",
        ["Extrema Rebound/Crossing", "Dashboard Utility (3-class)"],
        key="ml_training_mode",
        horizontal=True,
        help=(
            "**Extrema**: Binary rebound/crossing on tick extrema (research). "
            "**Dashboard Utility**: 3-class level-touch model aligned to "
            "Trading-Dashboard execution semantics."
        ),
    )
    is_utility_mode = training_mode == "Dashboard Utility (3-class)"

    if is_utility_mode:
        st.caption(
            "Dashboard-utility mode: train a 3-class CatBoost model on "
            "level-touch events with utility-aligned TP/SL labeling. "
            "Output is directly compatible with Trading-Dashboard."
        )
    else:
        st.caption(
            "Extrema mode: binary rebound/crossing classifier on tick-level "
            "extrema. This is the research pipeline."
        )

    # ── Configuration ─────────────────────────────────────────
    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

    with col_cfg1:
        st.markdown("**Data**")
        ml_symbol = st.selectbox("Symbol", ["NQ", "ES"], key="ml_symbol")
        ml_data_dir = st.text_input(
            "Data directory", value=str(_DEFAULT_DATA_DIR),
            key="ml_data_dir",
        )
        # Auto-detect date range from available data
        _avail = get_available_dates(ml_symbol, Path(ml_data_dir))
        _first = date.fromisoformat(_avail[0]) if _avail else date.today() - timedelta(days=90)
        _last = date.fromisoformat(_avail[-1]) if _avail else date.today() - timedelta(days=1)
        ml_start = st.date_input(
            "Start date", value=_first, key="ml_start",
        )
        ml_end = st.date_input(
            "End date", value=_last, key="ml_end",
        )

    with col_cfg2:
        st.markdown("**Walk-Forward**")
        # Auto-compute max sensible windows from data span
        _span = (_last - _first).days
        _default_train = min(30, max(5, _span // 4))
        _default_test = min(7, max(2, _span // 12))
        ml_train_days = st.slider(
            "Train window (days)", 5, 90, _default_train, key="ml_train_days",
        )
        ml_test_days = st.slider(
            "Test window (days)", 2, 30, _default_test, key="ml_test_days",
        )
        ml_gap_days = st.slider(
            "Gap (days)", 0, 5, 1, key="ml_gap_days",
        )
        # Show expected fold count
        _min_1 = ml_train_days + ml_gap_days + ml_test_days
        _est_folds = max(0, (_span - ml_train_days - ml_gap_days) // ml_test_days)
        st.caption(f"~{_est_folds} folds from {_span}d span (need {_min_1}d for 1st fold)")
        if is_utility_mode:
            ml_tp = st.slider("TP (points)", 5, 50, 15, key="ml_util_tp")
            ml_sl = st.slider("SL (points)", 5, 50, 30, key="ml_util_sl")
            ml_bar_type = st.selectbox(
                "Bar type", ["987t", "2000t", "147t", "1m"],
                key="ml_bar_type",
                help="Tick bars give finer touch detection than 1m time bars",
            )
            ml_int_window = st.slider(
                "Interaction window (min)", 1, 15, 5, key="ml_int_window",
            )
            ml_approach = st.checkbox(
                "Include approach features",
                key="ml_approach",
                help="Add 27 pre-touch order flow features (90-min window)",
            )
            ml_approach_window = 90
            if ml_approach:
                ml_approach_window = st.slider(
                    "Approach window (min)", 15, 120, 90, key="ml_approach_window",
                )
        else:
            ml_label = st.selectbox(
                "Label threshold", list(_LABEL_OPTIONS.keys()),
                key="ml_label",
            )

    with col_cfg3:
        st.markdown("**CatBoost**")
        ml_iterations = st.slider(
            "Iterations", 100, 2000, 500, 50, key="ml_iterations",
        )
        ml_depth = st.slider("Tree depth", 3, 10, 6, key="ml_depth")
        ml_rfecv = st.checkbox(
            "RFECV feature selection", value=False, key="ml_rfecv",
            help="Recursive feature elimination — slower but may improve.",
        )

    st.divider()

    data_dir = Path(ml_data_dir)
    if is_utility_mode:
        label_col = "label_encoded"
    else:
        label_col = _LABEL_OPTIONS[ml_label]

    # Build a fingerprint of the current UI selections so we can detect
    # when the user changes settings and the stored dataset is stale.
    _util_identity = ""
    if is_utility_mode:
        _util_identity = f"|{ml_bar_type}|{ml_int_window}|{ml_approach}|{ml_approach_window}|{ml_tp}|{ml_sl}"
    _dataset_identity = (
        f"{training_mode}|{ml_symbol}|{ml_start}|{ml_end}|"
        f"{label_col}|{ml_train_days}|{ml_test_days}|{ml_gap_days}"
        f"{_util_identity}"
    )
    if st.session_state.get("_ml_dataset_identity") != _dataset_identity:
        # Settings changed since last build — clear stale results
        st.session_state.pop("ml_dataset", None)
        st.session_state.pop("ml_build_config", None)
        st.session_state.pop("ml_training_result", None)
        st.session_state.pop("ml_train_config", None)
        st.session_state["_ml_dataset_identity"] = _dataset_identity

    # ── Phase 1: Local Data ───────────────────────────────────
    st.markdown("### Step 1: Local Data")

    # Scan for available local data
    available = get_available_dates(ml_symbol, data_dir)
    if available:
        # Filter to selected date range
        dates_in_range = [
            d for d in available
            if ml_start.isoformat() <= d <= ml_end.isoformat()
        ]
        st.success(
            f"{len(available)} dates available for {ml_symbol}: "
            f"{available[0]} to {available[-1]}  |  "
            f"{len(dates_in_range)} in selected range",
        )
    else:
        st.warning(
            f"No local tick data found for {ml_symbol} in `{data_dir}`.\n\n"
            f"Download data from the Databento portal and run "
            f"`python scripts/process_batch_download.py` to process it.",
        )

    st.divider()

    # ── Phase 2: Build Dataset ────────────────────────────────
    st.markdown("### Step 2: Build Dataset")

    if not available:
        st.warning("No local data found. See Step 1 above.")
    else:
        # Show cache status
        cached_dates = get_cached_ml_dates(ml_symbol, data_dir)
        cached_in_range = [d for d in cached_dates
                           if ml_start.isoformat() <= d <= ml_end.isoformat()]
        uncached = len(dates_in_range) - len(cached_in_range)

        if cached_in_range:
            st.caption(
                f"{len(dates_in_range)} dates in range — "
                f"{len(cached_in_range)} cached, {uncached} to compute"
            )
        else:
            st.caption(f"{len(dates_in_range)} dates in range (no cache)")

        col_build, col_clear = st.columns([3, 1])
        with col_build:
            build_btn = st.button(
                "Build Dataset", key="ml_build_btn",
                disabled=len(dates_in_range) == 0,
            )
        with col_clear:
            clear_btn = st.button(
                "Clear Cache", key="ml_clear_cache",
                disabled=len(cached_in_range) == 0,
                help="Delete cached ML features to force recomputation",
            )
            if clear_btn:
                n_cleared = clear_ml_cache(ml_symbol, data_dir, dates_in_range)
                st.toast(f"Cleared {n_cleared} cached feature files")
                st.rerun()

        if build_btn:
            from alpha_lab.agents.data_infra.ml.config import (
                DashboardUtilityConfig,
                FeatureConfig,
                MLPipelineConfig,
            )

            if is_utility_mode:
                build_config = MLPipelineConfig(
                    training_mode="dashboard_utility",
                    dashboard_utility=DashboardUtilityConfig(
                        tp_points=float(ml_tp),
                        sl_points=float(ml_sl),
                        bar_type=ml_bar_type,
                        interaction_window_minutes=ml_int_window,
                        include_approach_features=ml_approach,
                        approach_window_minutes=ml_approach_window,
                    ),
                    tick_size=0.25,
                    instrument=ml_symbol,
                )
                from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
                    build_utility_dataset,
                )
                progress = st.progress(0, text="Building utility dataset...")
                dataset = build_utility_dataset(
                    dates_in_range, data_dir, build_config,
                    progress_fn=lambda frac, text: progress.progress(frac, text=text),
                )
                if dataset.empty:
                    st.error("No touch events detected. Try a longer date range.")
                else:
                    st.session_state["ml_dataset"] = dataset
                    st.session_state["ml_build_config"] = build_config
                    # Clear stale training results from previous dataset
                    st.session_state.pop("ml_training_result", None)
                    st.session_state.pop("ml_train_config", None)
            else:
                build_config = MLPipelineConfig(
                    training_mode="extrema_rebound_crossing",
                    features=FeatureConfig(include_signal_features=False),
                    tick_size=0.25,
                    instrument=ml_symbol,
                )

                progress = st.progress(0, text="Building dataset...")
                dataset = build_training_dataset(
                    ml_symbol, dates_in_range, data_dir, build_config, progress,
                )

                if dataset.empty:
                    st.error("No extrema detected. Try a longer date range.")
                else:
                    st.session_state["ml_dataset"] = dataset
                    st.session_state["ml_build_config"] = build_config
                    # Clear stale training results from previous dataset
                    st.session_state.pop("ml_training_result", None)
                    st.session_state.pop("ml_train_config", None)

        if "ml_dataset" in st.session_state:
            dataset = st.session_state["ml_dataset"]

            if is_utility_mode:
                # 3-class display
                from alpha_lab.agents.data_infra.ml.dashboard_utility_labeling import (
                    CLASS_NAMES as UTIL_CLASS_NAMES,
                )
                n_valid = dataset[label_col].notna().sum() if label_col in dataset.columns else 0
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Touch Events", len(dataset))
                with c2:
                    n_rev = int((dataset[label_col] == 0).sum()) if label_col in dataset.columns else 0
                    st.metric("Reversal (0)", n_rev)
                with c3:
                    n_trap = int((dataset[label_col] == 1).sum()) if label_col in dataset.columns else 0
                    st.metric("Trap (1)", n_trap)
                with c4:
                    n_bt = int((dataset[label_col] == 2).sum()) if label_col in dataset.columns else 0
                    st.metric("Blowthrough (2)", n_bt)

                # Dynamic feature detection
                feature_cols_display = [
                    c for c in dataset.columns if c.startswith(("int_", "app_"))
                ]
                n_int = sum(1 for c in feature_cols_display if c.startswith("int_"))
                n_app = sum(1 for c in feature_cols_display if c.startswith("app_"))
                feat_desc = f"{n_int} interaction"
                if n_app > 0:
                    feat_desc += f" + {n_app} approach"
                st.caption(f"{len(feature_cols_display)} features ({feat_desc})")

                with st.expander("Preview dataset (first 20 rows)"):
                    display_cols = [
                        "timestamp", "direction", "representative_price",
                        "label", label_col,
                    ] + feature_cols_display[:6]
                    show_cols = [c for c in display_cols if c in dataset.columns]
                    st.dataframe(
                        dataset[show_cols].head(20),
                        use_container_width=True, hide_index=True,
                    )
            else:
                # Binary extrema display
                n_valid = dataset[label_col].notna().sum() if label_col in dataset.columns else 0
                n_rebound = int((dataset[label_col] == 1).sum()) if label_col in dataset.columns else 0
                n_crossing = int((dataset[label_col] == 0).sum()) if label_col in dataset.columns else 0

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Extrema", len(dataset))
                with c2:
                    st.metric("Labeled", int(n_valid))
                with c3:
                    st.metric("Rebound (1)", n_rebound)
                with c4:
                    st.metric("Crossing (0)", n_crossing)

                feature_cols = [c for c in dataset.columns
                                if c.startswith(("pl_", "ms_"))]
                st.caption(f"{len(feature_cols)} features: {', '.join(feature_cols[:10])}...")

                with st.expander("Preview dataset (first 20 rows)"):
                    display_cols = ["timestamp", "price", "extremum_type",
                                    label_col] + feature_cols[:5]
                    show_cols = [c for c in display_cols if c in dataset.columns]
                    st.dataframe(
                        dataset[show_cols].head(20),
                        use_container_width=True, hide_index=True,
                    )

    st.divider()

    # ── Phase 3: Train Model ──────────────────────────────────
    st.markdown("### Step 3: Train Model")

    has_dataset = "ml_dataset" in st.session_state
    if not has_dataset:
        st.warning("Build dataset first before training.")
    else:
        train_btn = st.button("Train Model", key="ml_train_btn")

        if train_btn:
            from alpha_lab.agents.data_infra.ml.config import (
                DashboardUtilityConfig,
                FeatureConfig,
                MLPipelineConfig,
                ModelConfig,
                WalkForwardConfig,
            )

            if is_utility_mode:
                train_config = MLPipelineConfig(
                    training_mode="dashboard_utility",
                    dashboard_utility=DashboardUtilityConfig(
                        tp_points=float(ml_tp),
                        sl_points=float(ml_sl),
                        bar_type=ml_bar_type,
                        interaction_window_minutes=ml_int_window,
                        include_approach_features=ml_approach,
                        approach_window_minutes=ml_approach_window,
                    ),
                    walk_forward=WalkForwardConfig(
                        train_days=ml_train_days,
                        test_days=ml_test_days,
                        gap_days=ml_gap_days,
                    ),
                    model=ModelConfig(
                        iterations=ml_iterations,
                        depth=ml_depth,
                        rfecv_enabled=ml_approach,  # RFECV useful when approach features present
                        loss_function="MultiClass",
                    ),
                    tick_size=0.25,
                    instrument=ml_symbol,
                )
            else:
                train_config = MLPipelineConfig(
                    training_mode="extrema_rebound_crossing",
                    features=FeatureConfig(include_signal_features=False),
                    walk_forward=WalkForwardConfig(
                        train_days=ml_train_days,
                        test_days=ml_test_days,
                        gap_days=ml_gap_days,
                    ),
                    model=ModelConfig(
                        iterations=ml_iterations,
                        depth=ml_depth,
                        rfecv_enabled=ml_rfecv,
                    ),
                    tick_size=0.25,
                    instrument=ml_symbol,
                )

            with st.spinner("Training walk-forward model... this may take a few minutes."):
                try:
                    result = run_walk_forward_training(
                        st.session_state["ml_dataset"],
                        train_config,
                        label_col,
                    )
                    st.session_state["ml_training_result"] = result
                    st.session_state["ml_train_config"] = train_config
                except ValueError as e:
                    st.error(str(e))

        if "ml_training_result" in st.session_state:
            result = st.session_state["ml_training_result"]
            ev = result["eval_result"]
            st.info(
                "Metrics and quality gates below are computed from concatenated "
                "out-of-sample fold predictions (evaluation population). "
                "Saving then refits a runtime model on all labeled rows."
            )

            oos = result.get("oos_summary", {})
            oos_rates = oos.get("confusion_rates", {})

            # Aggregate metrics
            st.markdown("#### Out-of-Sample (OOS) Aggregate Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("Precision", f"{ev.precision:.3f}",
                           help=f"95% CI: [{ev.precision_ci[0]:.3f}, {ev.precision_ci[1]:.3f}]")
            with c2:
                st.metric("Recall", f"{ev.recall:.3f}")
            with c3:
                st.metric("F1", f"{ev.f1:.3f}",
                           help=f"95% CI: [{ev.f1_ci[0]:.3f}, {ev.f1_ci[1]:.3f}]")
            with c4:
                st.metric("ROC-AUC", f"{ev.roc_auc:.3f}" if ev.roc_auc else "N/A")
            with c5:
                st.metric("Samples", ev.n_samples)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Walk-forward folds",
                    f"{result['n_valid_folds']} / {result['n_total_folds']}",
                    help=(
                        f"{result['n_skipped_folds']} fold(s) skipped because a "
                        "training window contained only one class."
                    ),
                )
            with c2:
                st.metric("Permutation p-value",
                           f"{ev.permutation_p_value:.4f}"
                           if ev.permutation_p_value is not None else "N/A")
            with c3:
                st.metric("Cohen's d",
                           f"{ev.cohens_d:.3f}"
                           if ev.cohens_d is not None else "N/A")
            with c4:
                balance = result["class_balance"]
                st.metric("OOS Class Balance",
                           f"{balance['rebound']}R / {balance['crossing']}C")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Specificity (TNR)",
                          f"{oos_rates.get('tnr_specificity', 0.0):.3f}")
            with c2:
                st.metric("False Positive Rate (FPR)",
                          f"{oos_rates.get('fpr', 0.0):.3f}")
            with c3:
                st.metric("Predicted Rebound Rate",
                          f"{oos_rates.get('predicted_positive_rate', 0.0):.3f}")

            # RTH coverage, label purging, and feature stability info
            rth_frac = result.get("rth_fraction")
            n_purged = result.get("n_purged_total", 0)
            feat_stability = result.get("feature_stability")
            c1, c2, c3 = st.columns(3)
            with c1:
                if rth_frac is not None:
                    st.metric("RTH Coverage", f"{rth_frac:.1%}",
                              help="Fraction of OOS samples within NY RTH (09:30-16:15 ET)")
                    if rth_frac < 0.5:
                        st.warning(
                            "Low RTH coverage — OOS metrics may not represent "
                            "the dashboard's NY RTH execution population."
                        )
            with c2:
                if n_purged > 0:
                    st.metric("Label-Purged Rows", n_purged,
                              help="Training rows removed to prevent forward-window label leakage")
            with c3:
                if feat_stability is not None:
                    st.metric("Feature Stability", f"{feat_stability:.3f}",
                              help="Mean pairwise Spearman rank-correlation of feature importances across folds (1.0 = perfectly stable)")
                    if feat_stability < 0.5:
                        st.warning("Low feature stability — model may rely on non-stationary signals.")

            full_balance = result.get("full_dataset_class_balance", {})
            if is_utility_mode:
                st.caption(
                    "Reference only (refit population): full labeled dataset balance = "
                    f"{full_balance.get('tradeable_reversal', 0)} Rev / "
                    f"{full_balance.get('trap_reversal', 0)} Trap / "
                    f"{full_balance.get('aggressive_blowthrough', 0)} BT"
                )
            else:
                st.caption(
                    "Reference only (refit population): full labeled dataset balance = "
                    f"{full_balance.get('rebound', 0)}R / {full_balance.get('crossing', 0)}C"
                )

            # Quality gates
            st.markdown("#### Quality Gates")
            gates_result = check_quality_gates(ev)
            gate_cols = st.columns(len(gates_result["gates"]))
            for col, (name, gate) in zip(
                gate_cols, gates_result["gates"].items(), strict=False,
            ):
                with col:
                    icon = "PASS" if gate["passed"] else "FAIL"
                    color = "green" if gate["passed"] else "red"
                    st.markdown(
                        f":{color}[**{icon}**] {name}\n\n"
                        f"Value: {gate['value']} (threshold: {gate['threshold']})",
                    )

            if gates_result["all_passed"]:
                st.success("All quality gates passed.")
            else:
                st.warning("Some quality gates failed. Review before saving.")

            # Utility metrics (simulated trade expectancy)
            st.markdown("#### Trade Utility (Simulated)")
            utility_15_15 = compute_utility_metrics(ev, tp_points=15.0, sl_points=15.0)
            utility_15_30 = compute_utility_metrics(ev, tp_points=15.0, sl_points=30.0)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Expectancy (15/15)",
                          f"{utility_15_15['expectancy_pts']:+.2f} pts",
                          help="Expected points per trade at TP=15 SL=15")
            with c2:
                st.metric("Expectancy (15/30)",
                          f"{utility_15_30['expectancy_pts']:+.2f} pts",
                          help="Expected points per trade at TP=15 SL=30")
            with c3:
                st.metric("Profit Factor (15/30)",
                          f"{utility_15_30['profit_factor']:.2f}",
                          help="Gross gains / gross losses at TP=15 SL=30")
            st.caption(
                f"Based on {utility_15_30['n_simulated_trades']} OOS predicted-rebound trades. "
                "Assumes all predicted rebounds are executed at stated TP/SL."
            )

            # Charts
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.caption(
                    "Feature importance from final refit runtime model "
                    "(trained on all labeled rows, not OOS folds)."
                )
                st.plotly_chart(
                    _chart_feature_importance(result["trained_model"]),
                    use_container_width=True,
                )
            with col_chart2:
                st.plotly_chart(
                    _chart_confusion_matrix(ev),
                    use_container_width=True,
                )

            st.plotly_chart(
                _chart_fold_metrics(result["fold_details"]),
                use_container_width=True,
            )

            if oos.get("threshold_table"):
                st.markdown("#### OOS Threshold vs Coverage")
                threshold_df = pd.DataFrame(oos["threshold_table"])
                for col in ["threshold", "coverage", "precision"]:
                    if col in threshold_df.columns:
                        threshold_df[col] = threshold_df[col].round(3)
                st.dataframe(threshold_df, use_container_width=True, hide_index=True)

            if oos.get("calibration_table"):
                st.markdown("#### OOS Calibration Buckets")
                calib_df = pd.DataFrame(oos["calibration_table"])
                for col in ["bucket_low", "bucket_high",
                            "mean_predicted_prob", "observed_positive_rate"]:
                    if col in calib_df.columns:
                        calib_df[col] = calib_df[col].round(3)
                st.dataframe(calib_df, use_container_width=True, hide_index=True)

            # Fold details table
            with st.expander("Fold Details"):
                fold_df = pd.DataFrame(result["fold_details"])
                for col in ["precision", "recall", "f1"]:
                    if col in fold_df.columns:
                        fold_df[col] = fold_df[col].round(3)
                if "roc_auc" in fold_df.columns:
                    fold_df["roc_auc"] = fold_df["roc_auc"].apply(
                        lambda x: round(x, 3) if x is not None else None,
                    )
                st.dataframe(fold_df, use_container_width=True, hide_index=True)

            # Save model
            st.divider()
            st.markdown("#### Save Model")

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"{ml_symbol}_{timestamp_str}"
            model_name = st.text_input(
                "Model name", value=default_name, key="ml_model_name",
            )
            output_dir = _DEFAULT_MODEL_DIR / model_name

            save_btn = st.button("Save Model", key="ml_save_btn")
            if save_btn:
                saved_path = save_trained_model(
                    result["trained_model"],
                    ev,
                    st.session_state.get("ml_train_config"),
                    output_dir,
                    training_result=result,
                    dates_used=dates_in_range if available else None,
                )
                st.session_state["ml_saved_path"] = str(saved_path)
                st.success(f"Model saved to `{saved_path}`")
                st.info(
                    "Paste this path into the sidebar **ML Extrema Classifier > "
                    "Model directory** field to use this saved runtime bundle for "
                    "signal generation.",
                )

            if "ml_saved_path" in st.session_state:
                st.caption(f"Last saved: {st.session_state['ml_saved_path']}")
