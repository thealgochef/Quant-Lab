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


def get_cached_ml_dates(
    symbol: str, data_dir: Path, cache_filename: str = "ml_features_extrema.parquet",
) -> list[str]:
    """Return dates that have cached feature files for a specific mode."""
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return []
    cached = []
    for d in sorted(symbol_dir.iterdir()):
        if d.is_dir() and (d / cache_filename).exists():
            cached.append(d.name)
    return cached


def clear_ml_cache(
    symbol: str,
    data_dir: Path,
    dates: list[str] | None = None,
    cache_filename: str = "ml_features_extrema.parquet",
) -> int:
    """Delete cached ml_features.parquet files. Returns count deleted."""
    symbol_dir = data_dir / symbol
    if not symbol_dir.exists():
        return 0
    count = 0
    for d in sorted(symbol_dir.iterdir()):
        if not d.is_dir():
            continue
        if dates is not None and d.name not in dates:
            continue
        cache_file = d / cache_filename
        if cache_file.exists():
            cache_file.unlink()
            count += 1
    return count


def build_training_dataset(
    symbol: str,
    dates: list[str],
    data_dir: Path,
    config,
    progress_bar=None,
    cache_filename: str = "ml_features_extrema.parquet",
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

    for i, date_str in enumerate(dates):
        cache_path = data_dir / symbol / date_str / cache_filename
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
    label_column: str = "label_7t",
) -> dict:
    """Walk-forward train + evaluate.

    Returns dict with the refit runtime model, out-of-sample evaluation
    result, and per-fold diagnostics.
    """
    from alpha_lab.agents.data_infra.ml.model_evaluator import ModelEvaluator
    from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer
    from alpha_lab.agents.data_infra.ml.walk_forward import WalkForwardSplitter

    # Separate features from labels/metadata
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

    features = valid[feature_cols].fillna(0.0)
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

    # Per-fold evaluation
    evaluator = ModelEvaluator(n_bootstrap=200, n_permutations=100)
    fold_details: list[dict] = []
    fold_predictions: list[dict[str, object]] = []

    valid_cv_splits: list[tuple] = []
    skipped = 0
    for split in splits:
        x_train = features.iloc[split.train_indices]
        y_train = y.iloc[split.train_indices]
        x_test = features.iloc[split.test_indices]
        y_test = y.iloc[split.test_indices]

        # CatBoost requires both classes in training data
        if y_train.nunique() < 2:
            logger.warning(
                "Skipping fold %d: train has single class %s",
                split.fold, y_train.value_counts().to_dict(),
            )
            skipped += 1
            continue

        trainer = ExtremaModelTrainer(config.model)
        fold_model = trainer.train(x_train, y_train)
        preds = fold_model.model.predict(
            x_test[fold_model.selected_features],
        ).flatten().astype(int)
        probs = fold_model.model.predict_proba(
            x_test[fold_model.selected_features],
        )
        rebound_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

        # Evaluate — handle single-class test gracefully
        fold_eval = evaluator.evaluate(
            y_test.values, preds, rebound_prob,
        )
        valid_cv_splits.append((split.train_indices, split.test_indices))
        fold_predictions.append({
            "fold": split.fold,
            "y_true": y_test.values,
            "y_pred": preds,
            "y_prob": rebound_prob,
            "session": (
                valid.iloc[split.test_indices]["session"].astype(str).values
                if "session" in valid.columns
                else None
            ),
            "zone_id": (
                valid.iloc[split.test_indices]["zone_id"].astype(str).values
                if "zone_id" in valid.columns
                else None
            ),
            "level_names": (
                valid.iloc[split.test_indices]["level_names"].astype(str).values
                if "level_names" in valid.columns
                else None
            ),
            "direction": (
                valid.iloc[split.test_indices]["direction"].astype(str).values
                if "direction" in valid.columns
                else None
            ),
            "approach_direction": (
                valid.iloc[split.test_indices]["approach_direction"].astype(str).values
                if "approach_direction" in valid.columns
                else None
            ),
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
    utility_summary = None
    if label_column == "label_15_30":
        utility_all = evaluator.summarize_utility_15_30_from_oos(
            fold_predictions,
            tp_points=15.0,
            sl_points=30.0,
        )
        utility_ny_rth = evaluator.summarize_utility_15_30_from_oos(
            fold_predictions,
            tp_points=15.0,
            sl_points=30.0,
            session_filter="ny_rth",
        )
        utility_summary = {
            "all": utility_all,
            "ny_rth": utility_ny_rth,
        }

    # Final model on all data (full dataset has both classes).
    trainer = ExtremaModelTrainer(config.model)
    final_model = trainer.train(features, y, cv_splits=valid_cv_splits)

    return {
        "trained_model": final_model,
        "eval_result": eval_result,
        "fold_details": fold_details,
        "feature_cols": feature_cols,
        "n_total": len(valid),
        "n_total_folds": len(splits),
        "n_valid_folds": len(valid_cv_splits),
        "n_skipped_folds": skipped,
        "oos_summary": oos_summary,
        "utility_summary": utility_summary,
        "full_dataset_class_balance": {
            "rebound": int((y == 1).sum()),
            "crossing": int((y == 0).sum()),
        },
        "class_balance": {
            "rebound": oos_summary["class_balance_true"]["rebound"],
            "crossing": oos_summary["class_balance_true"]["crossing"],
        },
    }


def build_dashboard_utility_dataset(
    symbol: str,
    dates: list[str],
    data_dir: Path,
    tp_ticks: int = 15,
    sl_ticks: int = 30,
    progress_bar=None,
    cache_filename: str = "ml_features_dashboard_utility_15_30.parquet",
) -> pd.DataFrame:
    """Build touch-anchored dashboard utility dataset for +15/-30 mode."""
    from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
        DashboardUtilityConfig,
        DashboardUtilityDatasetBuilder,
    )

    builder = DashboardUtilityDatasetBuilder(
        data_dir,
        DashboardUtilityConfig(tp_ticks=tp_ticks, sl_ticks=sl_ticks),
    )
    all_frames: list[pd.DataFrame] = []
    for i, date_str in enumerate(dates):
        cache_path = data_dir / symbol / date_str / cache_filename
        if cache_path.exists():
            try:
                day_df = pd.read_parquet(cache_path)
            except Exception:
                day_df = pd.DataFrame()
        else:
            day_df = builder.build_for_dates(symbol, [date_str])
            if not day_df.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                day_df.to_parquet(cache_path, index=False)

        if not day_df.empty:
            all_frames.append(day_df)

        if progress_bar:
            progress_bar.progress(
                (i + 1) / len(dates), text=f"Processed {i + 1}/{len(dates)} dates...",
            )

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


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
        "Test samples >= 50": {
            "passed": eval_result.n_samples >= 50,
            "value": str(eval_result.n_samples),
            "threshold": "50",
        },
    }

    all_passed = all(g["passed"] for g in gates.values())
    return {"gates": gates, "all_passed": all_passed}


def save_trained_model(
    trained_model,
    eval_result,
    config,
    output_dir: Path,
) -> Path:
    """Save model artifacts and evaluation results."""
    from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer

    output_dir.mkdir(parents=True, exist_ok=True)

    is_dashboard_mode = (
        config is not None
        and getattr(config, "training_mode", "") == "dashboard_utility_15_30"
    )
    extra_metadata = {
        "training_mode": getattr(config, "training_mode", "extrema_rebound_crossing")
        if config is not None else "extrema_rebound_crossing",
        "label_semantics": (
            "label_15_30=1 if +15 hits before -30, else 0"
            if is_dashboard_mode
            else "rebound_vs_crossing"
        ),
        "event_anchor_semantics": (
            "experiment first-touch key-level event seed (1m intersection approximation)"
            if is_dashboard_mode
            else "detected tick extrema"
        ),
        "execution_approximation_semantics": (
            "bullish entry ask / bearish entry bid; long exits vs bid; short exits vs ask"
            if is_dashboard_mode
            else "N/A"
        ),
        "evaluation_population": "concatenated_out_of_sample_fold_predictions",
    }
    # Save model + metadata
    ExtremaModelTrainer.save_model(trained_model, output_dir, extra_metadata=extra_metadata)

    # Save evaluation
    eval_dict = asdict(eval_result)
    # Convert numpy types for JSON serialization
    for key, val in eval_dict.items():
        if isinstance(val, (np.integer, np.int64)):
            eval_dict[key] = int(val)
        elif isinstance(val, (np.floating, np.float64)):
            eval_dict[key] = float(val)
        elif isinstance(val, tuple):
            eval_dict[key] = [float(v) for v in val]

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
    st.subheader("Primary Workflow: ML Extrema Training")
    st.caption(
        "Train either the primary extrema rebound/crossing mode or the "
        "dashboard utility +15/-30 mode with out-of-sample walk-forward evaluation."
    )
    st.info(
        "The Dashboard Compatibility tab is the retained secondary 3-class "
        "export path for ML-Trading-Dashboard."
    )

    mode_options = {
        "Primary: Extrema Rebound/Crossing": "extrema_rebound_crossing",
        "Dashboard Utility 15/30": "dashboard_utility_15_30",
    }
    mode_label = st.selectbox("Training mode", list(mode_options.keys()), key="ml_mode")
    training_mode = mode_options[mode_label]
    if training_mode == "dashboard_utility_15_30":
        st.info(
            "Event anchor: experiment first-touch key-level events (seed only). "
            "Execution approximation: bullish entry ask / bearish entry bid; "
            "long exits vs bid / short exits vs ask. Immediate touch-anchored baseline."
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
        if training_mode == "extrema_rebound_crossing":
            ml_label = st.selectbox(
                "Label threshold", list(_LABEL_OPTIONS.keys()),
                key="ml_label",
            )
        else:
            ml_label = "Utility +15/-30"
            st.caption("Label semantics: +15 before -30 (binary)")

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
    if training_mode == "extrema_rebound_crossing":
        label_col = _LABEL_OPTIONS[ml_label]
    else:
        label_col = "label_15_30"
    cache_filename = (
        "ml_features_extrema.parquet"
        if training_mode == "extrema_rebound_crossing"
        else "ml_features_dashboard_utility_15_30.parquet"
    )

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
        cached_dates = get_cached_ml_dates(ml_symbol, data_dir, cache_filename)
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
                n_cleared = clear_ml_cache(
                    ml_symbol, data_dir, dates_in_range, cache_filename,
                )
                st.toast(f"Cleared {n_cleared} cached feature files")
                st.rerun()

        if build_btn:
            from alpha_lab.agents.data_infra.ml.config import (
                FeatureConfig,
                MLPipelineConfig,
            )

            build_config = MLPipelineConfig(
                features=FeatureConfig(include_signal_features=False),
                tick_size=0.25,
                instrument=ml_symbol,
                training_mode=training_mode,
            )

            progress = st.progress(0, text="Building dataset...")
            if training_mode == "extrema_rebound_crossing":
                dataset = build_training_dataset(
                    ml_symbol, dates_in_range, data_dir, build_config, progress,
                    cache_filename=cache_filename,
                )
            else:
                dataset = build_dashboard_utility_dataset(
                    ml_symbol,
                    dates_in_range,
                    data_dir,
                    tp_ticks=15,
                    sl_ticks=30,
                    progress_bar=progress,
                    cache_filename=cache_filename,
                )

            if dataset.empty:
                st.error("No extrema detected. Try a longer date range.")
            else:
                st.session_state["ml_dataset"] = dataset
                st.session_state["ml_build_config"] = build_config

        if "ml_dataset" in st.session_state:
            dataset = st.session_state["ml_dataset"]
            n_valid = dataset[label_col].notna().sum() if label_col in dataset.columns else 0
            n_positive = int((dataset[label_col] == 1).sum()) if label_col in dataset.columns else 0
            n_negative = int((dataset[label_col] == 0).sum()) if label_col in dataset.columns else 0

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Extrema", len(dataset))
            with c2:
                st.metric("Labeled", int(n_valid))
            with c3:
                st.metric("Positive (1)", n_positive)
            with c4:
                st.metric("Negative (0)", n_negative)

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
                FeatureConfig,
                MLPipelineConfig,
                ModelConfig,
                WalkForwardConfig,
            )

            train_config = MLPipelineConfig(
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
                training_mode=training_mode,
                utility_tp_ticks=15,
                utility_sl_ticks=30,
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

            utility = result.get("utility_summary")
            utility_all = utility.get("all") if utility else None
            utility_ny = utility.get("ny_rth") if utility else None
            if utility_all is not None:
                st.markdown("#### OOS Utility Metrics (15/30)")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Win Rate 15/30", f"{utility_all['win_rate_15_30']:.3f}")
                with c2:
                    st.metric(
                        "Expectancy (pts)",
                        f"{utility_all['expectancy_points_15_30']:.3f}",
                    )
                with c3:
                    pf = utility_all["profit_factor_15_30"]
                    st.metric("Profit Factor", "inf" if pf == float("inf") else f"{pf:.3f}")
                with c4:
                    st.metric("Taken Trades", utility_all["n_taken_trades"])
                if utility_ny is not None:
                    st.caption(
                        "NY RTH filter: "
                        f"n={utility_ny['n_samples']}, "
                        f"taken={utility_ny['n_taken_trades']}, "
                        f"win_rate={utility_ny['win_rate_15_30']:.3f}, "
                        f"expectancy={utility_ny['expectancy_points_15_30']:.3f}",
                    )

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

            full_balance = result.get("full_dataset_class_balance", {})
            st.caption(
                "Reference only (refit population): full labeled dataset balance = "
                f"{full_balance.get('rebound', 0)}R / {full_balance.get('crossing', 0)}C"
            )
            st.caption(
                "Statistical note: bootstrap CI and permutation test currently use "
                "IID row assumptions; market events may be serially correlated."
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

            if utility_all is not None and utility_all.get("threshold_table_15_30"):
                st.markdown("#### OOS Threshold vs Utility (15/30)")
                util_df = pd.DataFrame(utility_all["threshold_table_15_30"])
                for col in ["threshold", "coverage", "win_rate_15_30", "expectancy_points_15_30"]:
                    if col in util_df.columns:
                        util_df[col] = util_df[col].round(3)
                st.dataframe(util_df, use_container_width=True, hide_index=True)

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
            if training_mode == "extrema_rebound_crossing":
                prefix = "extrema"
            else:
                prefix = "dashboard_utility_15_30"
            default_name = f"{prefix}_{ml_symbol}_{timestamp_str}"
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
