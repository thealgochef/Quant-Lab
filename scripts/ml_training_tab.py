"""
Primary extrema-model training workflow for the Streamlit dashboard.

Orchestrates: local tick data scan -> dataset build -> out-of-sample
walk-forward evaluation -> final CatBoost fit -> model save.

All tick data is read from local Parquet files (no Databento API calls).
Data must be pre-downloaded into data/databento/{symbol}/{date}/mbp10.parquet.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

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

_LABEL_OPTIONS = {
    "20-tick (5pt) rebound": "label_20t",
    "40-tick (10pt) rebound": "label_40t",
    "60-tick (15pt) rebound": "label_60t",
}

_DASHBOARD_UTILITY_BAR_TYPES = ["147t", "987t", "2000t", "1m"]


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
        len(dates),
        cached_count,
        len(dates) - cached_count,
    )
    return pd.concat(frames, ignore_index=True)


def _purged_trading_day_splits(
    frame: pd.DataFrame,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
    purge_days: int,
    min_train_events: int,
) -> list:
    """Purged walk-forward folds over TRADING days (the train_dashboard_model
    scheme): contiguous blocks of distinct ``date`` values, a ``purge_days``
    trading-day gap between train and test, stepping ``step_days`` days. Folds
    with fewer than ``min_train_events`` training rows (or zero test rows) are
    skipped, exactly as in scripts/train_dashboard_model.py.
    """
    from alpha_lab.agents.data_infra.ml.walk_forward import WalkForwardSplit

    date_values = frame["date"].astype(str)
    trading_dates = sorted(date_values.unique())
    n_dates = len(trading_dates)
    splits: list[WalkForwardSplit] = []
    fold = 0
    start = 0
    while start + train_days + purge_days + test_days <= n_dates:
        train_dates = trading_dates[start : start + train_days]
        test_start_pos = start + train_days + purge_days
        test_dates = trading_dates[test_start_pos : test_start_pos + test_days]
        train_idx = np.where(date_values.isin(train_dates))[0]
        test_idx = np.where(date_values.isin(test_dates))[0]
        if len(train_idx) >= min_train_events and len(test_idx) > 0:
            splits.append(
                WalkForwardSplit(
                    fold=fold,
                    train_start=pd.Timestamp(train_dates[0]),
                    train_end=pd.Timestamp(train_dates[-1]),
                    test_start=pd.Timestamp(test_dates[0]),
                    test_end=pd.Timestamp(test_dates[-1]),
                    train_indices=train_idx,
                    test_indices=test_idx,
                )
            )
            fold += 1
        start += step_days
    return splits


def resolve_training_kwargs(
    pinned_selection: list[str] | None,
    fold_scheme: str,
    fold_params: dict | None,
    rfecv_checkbox: bool,
) -> dict:
    """Resolve Train-step widget state into ``run_walk_forward_training``
    kwargs (QL-UI-PARITY). Pure — no Streamlit; unit-testable.

    An empty/None pin selection means no pinning; a non-empty selection is
    passed verbatim in the user's selection order and FORCES RFECV off
    (pinning wins over selection — mirrors the core's if/elif at the RFECV
    branch). ``fold_scheme`` "calendar" keeps the calendar
    ``WalkForwardSplitter`` (``day_folds`` None); "purged-days" builds the
    purged trading-day dict from ``fold_params``.
    """
    pinned_features = list(pinned_selection) if pinned_selection else None
    if fold_scheme == "purged-days":
        day_folds = {
            "train_days": int(fold_params["train_days"]),
            "test_days": int(fold_params["test_days"]),
            "step_days": int(fold_params["step_days"]),
            "purge_days": int(fold_params["purge_days"]),
            "min_train_events": int(fold_params["min_train_events"]),
        }
    elif fold_scheme == "calendar":
        day_folds = None
    else:
        msg = f"Unknown fold_scheme {fold_scheme!r} (expected 'calendar' or 'purged-days')"
        raise ValueError(msg)
    return {
        "pinned_features": pinned_features,
        "day_folds": day_folds,
        "rfecv_enabled": bool(rfecv_checkbox) and pinned_features is None,
    }


def run_walk_forward_training(
    dataset: pd.DataFrame,
    config,
    label_column: str = "label_20t",
    *,
    day_folds: dict | None = None,
    pinned_features: list[str] | None = None,
) -> dict:
    """Walk-forward train + evaluate.

    Returns dict with the refit runtime model, out-of-sample evaluation
    result, and per-fold diagnostics.

    ``day_folds`` (W3a): optional purged TRADING-day fold scheme — a dict of
    train_days/test_days/step_days/purge_days/min_train_events passed to
    ``_purged_trading_day_splits`` INSTEAD of the calendar-day
    ``WalkForwardSplitter``. ``pinned_features`` (W3a): exact feature list to
    train/serve (must be a subset of the dataset's feature columns); pinning
    wins over RFECV.
    """
    from alpha_lab.agents.data_infra.ml.model_evaluator import ModelEvaluator
    from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer
    from alpha_lab.agents.data_infra.ml.walk_forward import WalkForwardSplitter

    # Separate features from labels/metadata
    if config.training_mode == "dashboard_utility":
        # Dynamic: detect int_* (interaction) and app_* (approach) columns
        feature_cols = [c for c in dataset.columns if c.startswith(("int_", "app_"))]
    else:
        feature_cols = [c for c in dataset.columns if c.startswith(("pl_", "ms_", "sig_"))]
    if label_column not in dataset.columns:
        msg = f"Label column '{label_column}' not found in dataset"
        raise ValueError(msg)

    # Drop rows with missing labels
    valid = dataset[dataset[label_column].notna()].copy()
    if len(valid) == 0:
        msg = "No valid labeled samples after filtering"
        raise ValueError(msg)

    timestamps = pd.to_datetime(valid["timestamp"])
    sessions = _session_series_for_dataset(valid, timestamps)
    session_scope = _coerce_session_experiment(getattr(config, "session_experiment", None))
    session_filter = None
    if config.training_mode == "dashboard_utility":
        valid, timestamps, sessions, session_filter = apply_session_experiment_scope(
            valid,
            timestamps,
            sessions,
            session_scope,
        )
        if valid.empty:
            msg = (
                "No labeled samples remain after applying session experiment scope. "
                f"training_sessions={session_scope.training_sessions}, "
                f"evaluation_sessions={session_scope.evaluation_sessions}"
            )
            raise ValueError(msg)

    features = valid[feature_cols]
    y = valid[label_column].astype(int)
    timestamps = pd.to_datetime(timestamps).reset_index(drop=True)
    sessions = pd.Series(sessions, dtype="object").reset_index(drop=True)
    if config.training_mode == "dashboard_utility":
        training_session_mask = sessions.isin(
            session_scope.training_sessions,
        ).to_numpy(dtype=bool)
        evaluation_session_mask = sessions.isin(
            session_scope.evaluation_sessions,
        ).to_numpy(dtype=bool)
    else:
        training_session_mask = np.ones(len(valid), dtype=bool)
        evaluation_session_mask = np.ones(len(valid), dtype=bool)
    final_train_indices = np.flatnonzero(training_session_mask)
    if len(final_train_indices) == 0:
        msg = f"No training samples for sessions {session_scope.training_sessions}"
        raise ValueError(msg)
    if len(np.flatnonzero(evaluation_session_mask)) == 0:
        msg = f"No evaluation samples for sessions {session_scope.evaluation_sessions}"
        raise ValueError(msg)

    # Walk-forward splits
    if day_folds is not None:
        splits = _purged_trading_day_splits(valid, **day_folds)
    else:
        splitter = WalkForwardSplitter(config.walk_forward)
        splits = splitter.split(timestamps)

    if len(splits) < 2:
        if day_folds is not None:
            need_for_two = (
                day_folds["train_days"]
                + day_folds["purge_days"]
                + day_folds["test_days"]
                + day_folds["step_days"]
            )
            msg = (
                f"Only {len(splits)} purged trading-day fold(s). Need at least 2.\n"
                f"Your data covers {valid['date'].nunique()} trading days; the "
                f"scheme {day_folds} needs {need_for_two} days for 2 folds.\n"
                f"Fix: use a longer date range, or reduce the fold windows."
            )
            raise ValueError(msg)
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

    # Build preliminary CV index pairs for RFECV (skip empty/single-class folds).
    preliminary_cv: list[tuple] = []
    for split in splits:
        train_idx = np.asarray(split.train_indices)[
            training_session_mask[np.asarray(split.train_indices)]
        ]
        test_idx = np.asarray(split.test_indices)[
            evaluation_session_mask[np.asarray(split.test_indices)]
        ]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        y_train_fold = y.iloc[train_idx]
        if y_train_fold.nunique() >= 2:
            preliminary_cv.append((train_idx, test_idx))

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
    # W3a: an explicit pinned list wins over RFECV (and validates up front).
    selected_features = feature_cols
    if pinned_features is not None:
        missing_pins = [f for f in pinned_features if f not in feature_cols]
        if missing_pins:
            msg = (
                f"pinned_features not in dataset feature columns: {missing_pins} "
                f"(available: {sorted(feature_cols)})"
            )
            raise ValueError(msg)
        selected_features = list(pinned_features)
    elif config.model.rfecv_enabled and len(preliminary_cv) >= 2:
        rfecv_trainer = ExtremaModelTrainer(config.model)
        rfecv_result = rfecv_trainer.train(features, y, cv_splits=preliminary_cv)
        selected_features = rfecv_result.selected_features
        logger.info(
            "RFECV selected %d/%d features (used for all folds and final model)",
            len(selected_features),
            len(feature_cols),
        )

    # Per-fold config with RFECV disabled (already done above).
    fold_model_config = config.model.model_copy(
        update={"rfecv_enabled": False},
    )

    # Per-fold evaluation
    evaluator = ModelEvaluator(n_bootstrap=1000, n_permutations=500)
    fold_details: list[dict] = []
    fold_predictions: list[dict[str, object]] = []

    valid_cv_splits: list[tuple] = []
    fold_importances: list[dict[str, float]] = []
    purge_folds: list[dict[str, object]] = []
    skipped = 0
    skipped_empty_train = 0
    skipped_empty_test = 0
    n_purged_total = 0
    for split in splits:
        raw_train_idx = np.asarray(split.train_indices)
        raw_test_idx = np.asarray(split.test_indices)
        train_idx = raw_train_idx[training_session_mask[raw_train_idx]]
        test_idx = raw_test_idx[evaluation_session_mask[raw_test_idx]]
        if len(train_idx) == 0:
            skipped_empty_train += 1
            continue
        if len(test_idx) == 0:
            skipped_empty_test += 1
            continue
        # Purge training rows whose labeling horizon may cross into test.
        # Utility datasets expose exact label horizon end timestamps; use those
        # when present instead of relying on the extrema forward_window heuristic.
        purged_train_idx, purge_meta = purge_training_indices_for_label_leakage(
            train_indices=train_idx,
            timestamps=timestamps,
            dataset=valid,
            test_start=split.test_start,
            config=config,
        )
        purge_meta["fold"] = int(split.fold)
        purge_folds.append(purge_meta)
        n_purged = int(purge_meta["n_purged"])
        n_purged_total += n_purged

        x_train = features.iloc[purged_train_idx][selected_features]
        y_train = y.iloc[purged_train_idx]
        x_test = features.iloc[test_idx][selected_features]
        y_test = y.iloc[test_idx]

        # CatBoost requires both classes in training data
        if y_train.nunique() < 2:
            logger.warning(
                "Skipping fold %d: train has single class %s",
                split.fold,
                y_train.value_counts().to_dict(),
            )
            skipped += 1
            continue

        trainer = ExtremaModelTrainer(fold_model_config)
        fold_model = trainer.train(x_train, y_train)
        raw_preds = fold_model.model.predict(x_test).flatten().astype(int)
        raw_probs = fold_model.model.predict_proba(x_test)
        prob_by_class = _probability_by_class(fold_model.model, raw_probs)

        # For evaluation, binarize: "positive" = the signal class.
        # Extrema mode: class 1 (rebound) is positive.
        # Utility mode: class 0 (tradeable_reversal) is positive.
        if config.training_mode == "dashboard_utility":
            eval_y = (y_test.values == 0).astype(int)  # reversal = 1
            eval_preds = (raw_preds == 0).astype(int)
            eval_prob = _probability_for_class(
                prob_by_class,
                class_label=0,
                n_rows=len(x_test),
                fallback=raw_probs[:, 0],
            )  # P(tradeable_reversal)
        else:
            eval_y = y_test.values
            eval_preds = raw_preds
            fallback_prob = raw_probs[:, 1] if raw_probs.shape[1] > 1 else raw_probs[:, 0]
            eval_prob = _probability_for_class(
                prob_by_class,
                class_label=1,
                n_rows=len(x_test),
                fallback=fallback_prob,
            )

        # Evaluate — handle single-class test gracefully
        fold_eval = evaluator.evaluate(eval_y, eval_preds, eval_prob)
        valid_cv_splits.append((purged_train_idx, split.test_indices))
        fold_importances.append(fold_model.feature_importances)
        fold_predictions.append(
            {
                "fold": split.fold,
                "y_true": eval_y,
                "y_pred": eval_preds,
                "y_prob": eval_prob,
                "timestamp": timestamps.iloc[test_idx].to_numpy(),
                "session": sessions.iloc[test_idx].to_numpy(),
                "raw_y": y_test.values.astype(int),
                "raw_pred": raw_preds.astype(int),
                "prob_by_class": {
                    str(cls): np.asarray(prob).flatten() for cls, prob in prob_by_class.items()
                },
            }
        )
        fold_details.append(
            {
                "fold": split.fold,
                "train_start": str(split.train_start.date()),
                "train_end": str(split.train_end.date()),
                "test_start": str(split.test_start.date()),
                "test_end": str(split.test_end.date()),
                "n_train": len(train_idx),
                "n_train_after_purge": len(purged_train_idx),
                "n_test": len(test_idx),
                "precision": fold_eval.precision,
                "recall": fold_eval.recall,
                "f1": fold_eval.f1,
                "roc_auc": fold_eval.roc_auc,
            }
        )

    if not valid_cv_splits:
        class_dist = y.value_counts().to_dict()
        msg = (
            f"All {len(splits)} folds skipped — training data in each fold "
            f"had only one class.\n"
            f"Overall class distribution: {class_dist}\n"
            f"Try a larger train window so each fold captures both classes."
        )
        raise ValueError(msg)

    if skipped or skipped_empty_train or skipped_empty_test:
        logger.info(
            "Walk-forward: %d/%d folds valid "
            "(%d single-class, %d empty-train, %d empty-test skipped)",
            len(valid_cv_splits),
            len(splits),
            skipped,
            skipped_empty_train,
            skipped_empty_test,
        )

    # Aggregate true out-of-sample fold predictions for quality gates.
    eval_result = evaluator.evaluate_out_of_sample_folds(fold_predictions)
    oos_summary = evaluator.summarize_out_of_sample_predictions(
        fold_predictions,
    )
    confidence_threshold_stats = compute_confidence_threshold_stats(
        fold_predictions,
        thresholds=[0.70],
    )
    try:
        from strategy_core import constants as sc_constants
    except Exception:  # pragma: no cover - strategy_core is available in Quant-Lab env
        default_confidence_gate = 0.70
        default_tp_points = 15.0
        default_sl_points = 30.0
    else:
        default_confidence_gate = sc_constants.DEFAULT_CONFIDENCE_GATE
        default_tp_points = sc_constants.DEFAULT_TP_POINTS
        default_sl_points = sc_constants.DEFAULT_SL_POINTS
    du_cfg = getattr(config, "dashboard_utility", None)
    tp_points = float(getattr(du_cfg, "tp_points", default_tp_points))
    sl_points = float(getattr(du_cfg, "sl_points", default_sl_points))
    production_gate_sessions = list(session_scope.production_gate_sessions)
    gated_oos = compute_production_gate_report(
        fold_predictions,
        confidence_gate=float(default_confidence_gate),
        eligible_sessions=production_gate_sessions,
        tp_points=tp_points,
        sl_points=sl_points,
    )
    session_metrics = (
        compute_session_metrics(
            fold_predictions,
            confidence_gate=float(default_confidence_gate),
            tp_points=tp_points,
            sl_points=sl_points,
        )
        if bool(session_scope.report_session_breakdowns)
        else {}
    )
    oos_three_class_balance = compute_oos_three_class_balance(fold_predictions)
    oos_predictions = build_oos_predictions_frame(
        fold_predictions,
        confidence_gate=float(default_confidence_gate),
        eligible_sessions=production_gate_sessions,
    )
    if confidence_threshold_stats:
        oos_summary["confidence_threshold_stats"] = confidence_threshold_stats
        threshold_070 = confidence_threshold_stats.get("0.70")
        if threshold_070 is not None:
            for row in oos_summary.get("threshold_table", []):
                if abs(float(row.get("threshold", -1.0)) - 0.70) < 1e-9:
                    row["trade_count"] = threshold_070["trade_count"]
                    break
    oos_summary["gated_oos"] = gated_oos
    oos_summary["session_metrics"] = session_metrics
    oos_summary["session_experiment"] = _json_safe(session_scope.model_dump())
    if session_filter is not None:
        oos_summary["session_filter"] = _json_safe(session_filter)
    if oos_three_class_balance:
        oos_summary["oos_three_class_balance"] = oos_three_class_balance

    purge_methods = {str(f.get("method")) for f in purge_folds}
    label_purge = {
        "method": next(iter(purge_methods)) if len(purge_methods) == 1 else "mixed",
        "used_label_window_end": any(bool(f.get("used_label_window_end")) for f in purge_folds),
        "n_purged_total": int(n_purged_total),
        "folds": purge_folds,
    }

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

    # Legacy RTH-proxy coverage retained for backwards-compatible UI context.
    # The v3 production gate uses Strategy-Core session labels (session == "ny")
    # and is the authoritative execution-population report.
    try:
        from zoneinfo import ZoneInfo

        _et = ZoneInfo("America/New_York")
        flat_oos = _flatten_fold_predictions(fold_predictions)
        if "timestamp" in flat_oos:
            oos_ts = pd.Series(pd.to_datetime(flat_oos["timestamp"], errors="coerce"))
        else:
            oos_ts = pd.Series(pd.to_datetime([], errors="coerce"))
        oos_et = oos_ts.dt.tz_convert(_et) if oos_ts.dt.tz is not None else oos_ts
        rth_mask = (
            (oos_et.dt.hour >= 9)
            & ((oos_et.dt.hour > 9) | (oos_et.dt.minute >= 30))
            & ((oos_et.dt.hour < 16) | ((oos_et.dt.hour == 16) & (oos_et.dt.minute <= 15)))
        )
        rth_fraction = float(rth_mask.mean()) if len(rth_mask) > 0 else 0.0
    except Exception:
        rth_fraction = None

    # Final model on all data using the same feature subset. RFECV
    # is disabled here because feature selection was already done above.
    final_y = y.iloc[final_train_indices]
    if final_y.nunique() < 2:
        class_dist = final_y.value_counts().to_dict()
        msg = (
            "Final refit training population has fewer than two classes after "
            f"session filter {session_scope.training_sessions}: {class_dist}"
        )
        raise ValueError(msg)
    final_trainer = ExtremaModelTrainer(fold_model_config)
    final_model = final_trainer.train(
        features.iloc[final_train_indices][selected_features],
        final_y,
    )

    return {
        "trained_model": final_model,
        "eval_result": eval_result,
        "fold_details": fold_details,
        "feature_cols": feature_cols,
        "selected_features": selected_features,
        "n_total": len(valid),
        "n_training_samples": int(len(final_train_indices)),
        "n_evaluation_candidate_samples": int(np.sum(evaluation_session_mask)),
        "session_experiment": _json_safe(session_scope.model_dump()),
        "session_filter": _json_safe(session_filter),
        "n_purged_total": n_purged_total,
        "label_purge": label_purge,
        "feature_stability": feature_stability,
        "n_total_folds": len(splits),
        "n_valid_folds": len(valid_cv_splits),
        "n_skipped_folds": skipped + skipped_empty_train + skipped_empty_test,
        "n_skipped_single_class_folds": skipped,
        "n_skipped_empty_train_folds": skipped_empty_train,
        "n_skipped_empty_test_folds": skipped_empty_test,
        "oos_summary": oos_summary,
        "confidence_threshold_stats": confidence_threshold_stats,
        "gated_oos": gated_oos,
        "session_metrics": session_metrics,
        "oos_three_class_balance": oos_three_class_balance,
        "oos_predictions": oos_predictions,
        "rth_fraction": rth_fraction,
        "full_dataset_class_balance": (
            {
                "tradeable_reversal": int((final_y == 0).sum()),
                "trap_reversal": int((final_y == 1).sum()),
                "aggressive_blowthrough": int((final_y == 2).sum()),
            }
            if config.training_mode == "dashboard_utility"
            else {
                "rebound": int((final_y == 1).sum()),
                "crossing": int((final_y == 0).sum()),
            }
        ),
        "class_balance": (
            {
                "tradeable_reversal": oos_summary["class_balance_true"]["rebound"],
                "non_tradeable": oos_summary["class_balance_true"]["crossing"],
            }
            if config.training_mode == "dashboard_utility"
            else {
                "rebound": oos_summary["class_balance_true"]["rebound"],
                "crossing": oos_summary["class_balance_true"]["crossing"],
            }
        ),
    }


def _json_safe(value):
    """Recursively convert common numpy/pandas values to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def compute_confidence_threshold_stats(
    fold_predictions: list[dict[str, object]],
    *,
    thresholds: list[float] | None = None,
) -> dict[str, dict[str, float | int]]:
    """Compute OOS confidence-threshold coverage/precision/trade counts.

    ``y_prob`` is expected to be the probability of the positive/actionable class
    already used for evaluation (rebound in extrema mode, tradeable_reversal in
    dashboard-utility mode).
    """
    if not fold_predictions:
        return {}

    all_true: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    for fold_data in fold_predictions:
        y_prob_raw = fold_data.get("y_prob")
        if y_prob_raw is None:
            return {}
        y_true = np.asarray(fold_data["y_true"]).flatten()
        y_prob = np.asarray(y_prob_raw).flatten()
        if len(y_true) != len(y_prob):
            msg = "fold prediction y_true/y_prob lengths must match"
            raise ValueError(msg)
        all_true.append(y_true)
        all_prob.append(y_prob)

    y_true_all = np.concatenate(all_true)
    y_prob_all = np.concatenate(all_prob)
    n_total = len(y_true_all)
    if n_total == 0:
        return {}

    stats: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds or [0.70]:
        threshold_float = float(threshold)
        mask = y_prob_all >= threshold_float
        trade_count = int(np.sum(mask))
        precision = float(np.mean(y_true_all[mask] == 1)) if trade_count > 0 else 0.0
        stats[f"{threshold_float:.2f}"] = {
            "threshold": threshold_float,
            "coverage": float(trade_count / n_total),
            "precision": precision,
            "trade_count": trade_count,
        }
    return stats


def _probability_by_class(model, raw_probs) -> dict[int, np.ndarray]:
    """Map classifier probability columns to integer class labels when available."""
    probs = np.asarray(raw_probs)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = list(range(probs.shape[1]))
    out: dict[int, np.ndarray] = {}
    for idx, cls in enumerate(list(classes)[: probs.shape[1]]):
        try:
            key = int(cls)
        except Exception:
            key = idx
        out[key] = probs[:, idx]
    return out


def _probability_for_class(
    prob_by_class: dict[int, np.ndarray],
    *,
    class_label: int,
    n_rows: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Return a class probability without remapping another known class.

    If a fold model exposes known classes and the requested class was absent
    from that fold's training labels, that class has zero probability. The
    fallback is only for classifiers that do not expose a usable class map.
    """
    if class_label in prob_by_class:
        return np.asarray(prob_by_class[class_label]).flatten().astype(float)
    if prob_by_class:
        return np.zeros(int(n_rows), dtype=float)
    if fallback is None:
        return np.zeros(int(n_rows), dtype=float)
    return np.asarray(fallback).flatten().astype(float)


def _session_series_for_dataset(dataset: pd.DataFrame, timestamps: pd.Series) -> pd.Series:
    """Return per-row Strategy-Core session labels, deriving from timestamps if needed.

    Old utility caches can contain a ``session`` column that is partially null.
    Treat null/blank/unknown values as repairable from timestamp, while preserving
    explicit Strategy-Core labels such as ``none`` for session gaps.
    """
    existing = None
    if "session" in dataset.columns:
        existing = pd.Series(dataset["session"].values, index=dataset.index, dtype="object")

    try:
        from strategy_core import classify_session
    except Exception:
        if existing is not None:
            return existing.fillna("unknown")
        return pd.Series(["unknown"] * len(dataset), index=dataset.index)

    ts_series = pd.Series(timestamps.values, index=dataset.index)
    sessions: list[str] = []
    for ts in pd.to_datetime(ts_series, errors="coerce"):
        if pd.isna(ts):
            sessions.append("unknown")
            continue
        ts = pd.Timestamp(ts)
        ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
        try:
            sessions.append(str(classify_session(ts.to_pydatetime()).session))
        except Exception:
            sessions.append("unknown")
    derived = pd.Series(sessions, index=dataset.index)
    if existing is None:
        return derived

    normalized = existing.astype(str).str.strip().str.lower()
    missing = existing.isna() | normalized.isin({"", "nan", "unknown"})
    repaired = existing.fillna("unknown")
    repaired.loc[missing] = derived.loc[missing]
    return repaired


def _coerce_session_experiment(scope) -> Any:
    """Return a validated SessionExperimentConfig without importing at module load."""
    from alpha_lab.agents.data_infra.ml.config import SessionExperimentConfig

    if isinstance(scope, SessionExperimentConfig):
        return scope
    if scope is None:
        return SessionExperimentConfig()
    if isinstance(scope, dict):
        return SessionExperimentConfig(**scope)
    if hasattr(scope, "model_dump"):
        return SessionExperimentConfig(**scope.model_dump())
    msg = f"Unsupported session experiment config: {type(scope)!r}"
    raise TypeError(msg)


def apply_session_experiment_scope(
    dataset: pd.DataFrame,
    timestamps: pd.Series,
    sessions: pd.Series,
    scope,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    """Filter rows to the union needed by a train/eval session experiment.

    ``training_sessions`` controls rows eligible for per-fold training and final
    refit. ``evaluation_sessions`` controls rows eligible for OOS reported stats.
    The returned frame keeps only their union so unrelated sessions cannot leak
    into splits, metrics, or final refit by accident. Dataset generation itself
    remains broad/cacheable; this is a training/evaluation-stage filter.
    """
    session_scope = _coerce_session_experiment(scope)
    ts = pd.Series(pd.to_datetime(timestamps), index=dataset.index)
    session_series = (
        pd.Series(sessions, index=dataset.index, dtype="object").astype(str).str.strip().str.lower()
    )
    train_set = set(session_scope.training_sessions)
    eval_set = set(session_scope.evaluation_sessions)
    included_sessions = train_set | eval_set
    include_mask = session_series.isin(included_sessions)

    scoped_dataset = dataset.loc[include_mask].reset_index(drop=True)
    scoped_timestamps = ts.loc[include_mask].reset_index(drop=True)
    scoped_sessions = session_series.loc[include_mask].reset_index(drop=True)

    session_counts_before = session_series.value_counts(dropna=False).sort_index()
    session_counts_after = scoped_sessions.value_counts(dropna=False).sort_index()
    train_mask_after = scoped_sessions.isin(train_set)
    eval_mask_after = scoped_sessions.isin(eval_set)
    metadata = {
        "training_sessions": list(session_scope.training_sessions),
        "evaluation_sessions": list(session_scope.evaluation_sessions),
        "production_gate_sessions": list(session_scope.production_gate_sessions),
        "report_session_breakdowns": bool(session_scope.report_session_breakdowns),
        "rows_before_session_filter": int(len(dataset)),
        "rows_after_session_filter": int(len(scoped_dataset)),
        "rows_excluded_by_session_filter": int(len(dataset) - len(scoped_dataset)),
        "training_candidate_rows": int(train_mask_after.sum()),
        "evaluation_candidate_rows": int(eval_mask_after.sum()),
        "session_counts_before": {
            str(k): int(v) for k, v in session_counts_before.to_dict().items()
        },
        "session_counts_after": {str(k): int(v) for k, v in session_counts_after.to_dict().items()},
    }
    return scoped_dataset, scoped_timestamps, scoped_sessions, metadata


def _flatten_fold_predictions(
    fold_predictions: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Concatenate fold prediction arrays with optional timestamp/session metadata."""
    if not fold_predictions:
        return {
            "y_true": np.array([], dtype=int),
            "y_pred": np.array([], dtype=int),
            "y_prob": np.array([], dtype=float),
            "session": np.array([], dtype=object),
        }
    y_true = np.concatenate(
        [np.asarray(fold_data["y_true"]).flatten().astype(int) for fold_data in fold_predictions]
    )
    y_pred = np.concatenate(
        [np.asarray(fold_data["y_pred"]).flatten().astype(int) for fold_data in fold_predictions]
    )
    y_prob = np.concatenate(
        [
            np.asarray(fold_data.get("y_prob", np.zeros(len(fold_data["y_true"]))))
            .flatten()
            .astype(float)
            for fold_data in fold_predictions
        ]
    )
    sessions = []
    for fold_data in fold_predictions:
        n = len(np.asarray(fold_data["y_true"]).flatten())
        raw_sessions = fold_data.get("session")
        if raw_sessions is None:
            sessions.append(np.array(["unknown"] * n, dtype=object))
        else:
            s = np.asarray(raw_sessions, dtype=object).flatten()
            if len(s) != n:
                msg = "fold prediction session/y_true lengths must match"
                raise ValueError(msg)
            sessions.append(s)
    out = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "session": np.concatenate(sessions) if sessions else np.array([], dtype=object),
    }
    optional_keys = ("raw_y", "raw_pred", "timestamp")
    for key in optional_keys:
        values = []
        have_any = False
        for fold_data in fold_predictions:
            n = len(np.asarray(fold_data["y_true"]).flatten())
            raw = fold_data.get(key)
            if raw is None:
                values.append(np.array([None] * n, dtype=object))
            else:
                have_any = True
                arr = np.asarray(raw).flatten()
                if len(arr) != n:
                    msg = f"fold prediction {key}/y_true lengths must match"
                    raise ValueError(msg)
                values.append(arr)
        if have_any:
            out[key] = np.concatenate(values)
    return out


def _binary_trade_metrics(
    y_true: np.ndarray,
    selected_mask: np.ndarray,
    *,
    tp_points: float = 15.0,
    sl_points: float = 30.0,
) -> dict[str, float | int]:
    """Confusion/rate/idealized label-EV metrics for a gated binary action mask."""
    y = np.asarray(y_true).flatten().astype(int)
    selected = np.asarray(selected_mask).flatten().astype(bool)
    if len(y) != len(selected):
        msg = "y_true and selected_mask lengths must match"
        raise ValueError(msg)
    n_total = int(len(y))
    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    tn = int(np.sum(~selected & (y == 0)))
    fn = int(np.sum(~selected & (y == 1)))
    trade_count = tp + fp
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    precision = float(tp / trade_count) if trade_count > 0 else 0.0
    recall = float(tp / positives) if positives > 0 else 0.0
    specificity = float(tn / negatives) if negatives > 0 else 0.0
    fpr = float(fp / negatives) if negatives > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    gross_gain = tp * float(tp_points)
    gross_loss = fp * float(sl_points)
    expectancy = float((gross_gain - gross_loss) / trade_count) if trade_count > 0 else 0.0
    profit_factor = (
        float(gross_gain / gross_loss)
        if gross_loss > 0
        else (float("inf") if gross_gain > 0 else 0.0)
    )
    return {
        "n_samples": n_total,
        "n_positive": positives,
        "n_negative": negatives,
        "trade_count": int(trade_count),
        "coverage": float(trade_count / n_total) if n_total > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "expectancy_15_30_pts": expectancy,
        "profit_factor_15_30": profit_factor,
    }


def _normalize_eligible_sessions(
    eligible_session: str | None = "ny",
    eligible_sessions: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Normalize a production-gate session list while preserving compatibility."""
    from alpha_lab.agents.data_infra.ml.config import SessionExperimentConfig

    raw = [eligible_session or "ny"] if eligible_sessions is None else list(eligible_sessions)
    scope = SessionExperimentConfig(
        training_sessions=["ny"],
        evaluation_sessions=["ny"],
        production_gate_sessions=raw,
    )
    return list(scope.production_gate_sessions)


def compute_production_gate_report(
    fold_predictions: list[dict[str, object]],
    *,
    confidence_gate: float = 0.70,
    eligible_session: str | None = "ny",
    eligible_sessions: list[str] | tuple[str, ...] | None = None,
    tp_points: float = 15.0,
    sl_points: float = 30.0,
) -> dict[str, float | int | str | bool | list[str]]:
    """OOS report for the research gate: session eligibility + confidence."""
    gate_sessions = _normalize_eligible_sessions(eligible_session, eligible_sessions)
    flat = _flatten_fold_predictions(fold_predictions)
    y_true = flat["y_true"]
    y_prob = flat["y_prob"]
    sessions = flat["session"].astype(str)
    eligible_session_label = (
        gate_sessions[0] if len(gate_sessions) == 1 else ",".join(gate_sessions)
    )
    if len(y_true) == 0:
        return {
            "confidence_gate": float(confidence_gate),
            "eligible_session": eligible_session_label,
            "eligible_sessions": gate_sessions,
            "n_samples": 0,
            "session_eligible_count": 0,
            "session_filter_available": False,
            **_binary_trade_metrics(
                y_true,
                np.array([], dtype=bool),
                tp_points=tp_points,
                sl_points=sl_points,
            ),
        }
    session_filter_available = not np.all(sessions == "unknown")
    session_eligible = np.isin(sessions, gate_sessions)
    selected = (y_prob >= float(confidence_gate)) & session_eligible
    metrics = _binary_trade_metrics(
        y_true,
        selected,
        tp_points=tp_points,
        sl_points=sl_points,
    )
    eligible_count = int(np.sum(session_eligible))
    metrics.update(
        {
            "confidence_gate": float(confidence_gate),
            "eligible_session": eligible_session_label,
            "eligible_sessions": gate_sessions,
            "session_eligible_count": eligible_count,
            "eligible_coverage": (
                float(metrics["trade_count"] / eligible_count) if eligible_count > 0 else 0.0
            ),
            "session_filter_available": bool(session_filter_available),
        }
    )
    return metrics


def compute_session_metrics(
    fold_predictions: list[dict[str, object]],
    *,
    confidence_gate: float = 0.70,
    tp_points: float = 15.0,
    sl_points: float = 30.0,
) -> dict[str, dict[str, float | int]]:
    """OOS threshold metrics by Strategy-Core session."""
    flat = _flatten_fold_predictions(fold_predictions)
    y_true = flat["y_true"]
    y_prob = flat["y_prob"]
    sessions = flat["session"].astype(str)

    def group(mask: np.ndarray) -> dict[str, float | int]:
        selected = y_prob[mask] >= float(confidence_gate)
        metrics = _binary_trade_metrics(
            y_true[mask],
            selected,
            tp_points=tp_points,
            sl_points=sl_points,
        )
        metrics["confidence_gate"] = float(confidence_gate)
        return metrics

    if len(y_true) == 0:
        return {"all": group(np.array([], dtype=bool))}

    out: dict[str, dict[str, float | int]] = {
        "all": group(np.ones(len(y_true), dtype=bool)),
        "non_ny": group(sessions != "ny"),
    }
    for session in sorted(set(sessions.tolist())):
        out[str(session)] = group(sessions == session)
    return out


def compute_oos_three_class_balance(
    fold_predictions: list[dict[str, object]],
) -> dict[str, int]:
    """Count OOS dashboard-utility labels by original 3-class encoding."""
    raw_arrays = [fold_data.get("raw_y") for fold_data in fold_predictions]
    if not raw_arrays or all(raw is None for raw in raw_arrays):
        return {}
    raw_y = np.concatenate(
        [
            np.asarray(raw if raw is not None else []).flatten().astype(int)
            for raw in raw_arrays
            if raw is not None
        ]
    )
    names = {
        0: "tradeable_reversal",
        1: "trap_reversal",
        2: "aggressive_blowthrough",
    }
    return {name: int(np.sum(raw_y == idx)) for idx, name in names.items()}


def build_oos_predictions_frame(
    fold_predictions: list[dict[str, object]],
    *,
    confidence_gate: float = 0.70,
    eligible_session: str | None = "ny",
    eligible_sessions: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Materialize row-level OOS predictions for post-training threshold audits."""
    gate_sessions = _normalize_eligible_sessions(eligible_session, eligible_sessions)
    rows: list[dict[str, object]] = []
    class_names = {
        0: "tradeable_reversal",
        1: "trap_reversal",
        2: "aggressive_blowthrough",
    }
    base_columns = [
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
    for fold_data in fold_predictions:
        y_true = np.asarray(fold_data["y_true"]).flatten().astype(int)
        n = len(y_true)
        raw_y = np.asarray(fold_data.get("raw_y", [None] * n)).flatten()
        raw_pred = np.asarray(fold_data.get("raw_pred", [None] * n)).flatten()
        y_prob = np.asarray(fold_data.get("y_prob", np.zeros(n))).flatten().astype(float)
        sessions = np.asarray(fold_data.get("session", ["unknown"] * n), dtype=object).flatten()
        timestamps = np.asarray(fold_data.get("timestamp", [None] * n), dtype=object).flatten()
        prob_by_class_raw = fold_data.get("prob_by_class") or {}
        prob_by_class = {
            int(cls): np.asarray(prob).flatten().astype(float)
            for cls, prob in dict(prob_by_class_raw).items()
        }
        for i in range(n):
            raw_label = raw_y[i]
            try:
                raw_label_int = int(raw_label)
            except Exception:
                raw_label_int = None
            raw_pred_val = raw_pred[i]
            try:
                raw_pred_int = int(raw_pred_val)
            except Exception:
                raw_pred_int = None
            session = str(sessions[i])
            prob_tradeable = float(prob_by_class.get(0, y_prob)[i]) if n else 0.0
            row = {
                "fold": int(fold_data.get("fold", 0)),
                "timestamp": timestamps[i],
                "session": session,
                "binary_true_tradeable": int(y_true[i]),
                "label_encoded": raw_label_int,
                "label": class_names.get(raw_label_int),
                "pred_label_encoded": raw_pred_int,
                "pred_label": class_names.get(raw_pred_int),
                "prob_tradeable_reversal": prob_tradeable,
                "gate_0_70_runtime_sessions": bool(
                    session in gate_sessions and prob_tradeable >= float(confidence_gate)
                ),
                "gate_0_70_ny": bool(
                    session == "ny"
                    and "ny" in gate_sessions
                    and prob_tradeable >= float(confidence_gate)
                ),
            }
            for cls, name in class_names.items():
                probs = prob_by_class.get(cls)
                if probs is not None and len(probs) == n:
                    row[f"prob_{name}"] = float(probs[i])
            rows.append(row)
    if not rows:
        # W2 P3a: an empty result still carries the canonical schema so the
        # unconditional bundle writer can persist a typed (empty) parquet.
        return pd.DataFrame(columns=base_columns)
    return pd.DataFrame(rows)


def _derive_dashboard_utility_label_window_end(
    dataset: pd.DataFrame,
    timestamps: pd.Series,
) -> pd.Series | None:
    """Derive exact utility label horizon end for old caches lacking the column."""
    try:
        from strategy_core import classify_session
        from strategy_core.constants import RTH_END, SESSION_TIMEZONE
    except Exception:
        return None

    dates: list[str | None] = []
    if "date" in dataset.columns:
        for value in dataset["date"]:
            if pd.isna(value):
                dates.append(None)
            else:
                dates.append(str(value))
    else:
        for ts in pd.to_datetime(pd.Series(timestamps), errors="coerce"):
            if pd.isna(ts):
                dates.append(None)
                continue
            ts = pd.Timestamp(ts)
            ts = ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
            try:
                trading_day = classify_session(ts.to_pydatetime()).trading_day
            except Exception:
                trading_day = None
            dates.append(trading_day.isoformat() if trading_day is not None else None)

    values: list[pd.Timestamp | pd.NaT] = []
    for date_str in dates:
        if not date_str:
            values.append(pd.NaT)
            continue
        naive = pd.Timestamp(f"{date_str} {RTH_END.strftime('%H:%M:%S')}")
        values.append(
            naive.tz_localize(
                SESSION_TIMEZONE,
                nonexistent="shift_forward",
                ambiguous=False,
            ),
        )
    return pd.Series(values, index=dataset.index)


def _align_timestamp_series_for_compare(
    values: pd.Series,
    reference,
) -> tuple[pd.Series, pd.Timestamp]:
    """Make a timestamp series and reference timestamp timezone-compatible."""
    series = pd.to_datetime(values, errors="coerce")
    ref = pd.Timestamp(reference)
    series_tz = getattr(series.dt, "tz", None)
    if series_tz is not None and ref.tz is None:
        ref = ref.tz_localize(series_tz)
    elif series_tz is None and ref.tz is not None:
        series = series.dt.tz_localize(ref.tz)
    return series, ref


def purge_training_indices_for_label_leakage(
    *,
    train_indices: np.ndarray,
    timestamps: pd.Series,
    dataset: pd.DataFrame,
    test_start,
    config,
) -> tuple[np.ndarray, dict[str, object]]:
    """Drop train rows whose forward label horizon can touch the test window.

    Dashboard-utility datasets include ``label_window_end`` with the exact label
    horizon end for each row.  When present, that exact horizon is safer than the
    extrema-only ``forward_window`` timestamp heuristic and is used as the purge
    rule.  Rows with missing exact horizons are purged rather than assumed safe.
    """
    train_idx = np.asarray(train_indices)
    n_before = int(len(train_idx))

    if "label_window_end" in dataset.columns:
        label_window_end, test_ts = _align_timestamp_series_for_compare(
            dataset["label_window_end"].iloc[train_idx],
            test_start,
        )
        safe_mask = (label_window_end.notna() & (label_window_end < test_ts)).to_numpy(
            dtype=bool,
        )
        purged_idx = train_idx[safe_mask]
        max_label_window_end = label_window_end.max()
        metadata = {
            "method": "label_window_end",
            "used_label_window_end": True,
            "purge_rule": "label_window_end < test_start",
            "test_start": str(test_ts),
            "n_train_before": n_before,
            "n_train_after": int(len(purged_idx)),
            "n_purged": int(n_before - len(purged_idx)),
            "n_missing_label_window_end": int(label_window_end.isna().sum()),
            "max_label_window_end": (
                None if pd.isna(max_label_window_end) else str(max_label_window_end)
            ),
        }
        return purged_idx, metadata

    if getattr(config, "training_mode", None) == "dashboard_utility":
        derived_label_window_end = _derive_dashboard_utility_label_window_end(
            dataset,
            timestamps,
        )
        if derived_label_window_end is not None:
            label_window_end, test_ts = _align_timestamp_series_for_compare(
                derived_label_window_end.iloc[train_idx],
                test_start,
            )
            safe_mask = (label_window_end.notna() & (label_window_end < test_ts)).to_numpy(
                dtype=bool
            )
            purged_idx = train_idx[safe_mask]
            max_label_window_end = label_window_end.max()
            metadata = {
                "method": "dashboard_utility_label_window_end",
                "used_label_window_end": True,
                "purge_rule": "derived_label_window_end < test_start",
                "test_start": str(test_ts),
                "n_train_before": n_before,
                "n_train_after": int(len(purged_idx)),
                "n_purged": int(n_before - len(purged_idx)),
                "n_missing_label_window_end": int(label_window_end.isna().sum()),
                "max_label_window_end": (
                    None if pd.isna(max_label_window_end) else str(max_label_window_end)
                ),
            }
            return purged_idx, metadata

    train_ts, test_ts = _align_timestamp_series_for_compare(
        timestamps.iloc[train_idx],
        test_start,
    )
    labeling_config = getattr(config, "labeling", None)
    forward_window = int(getattr(labeling_config, "forward_window", 5000))
    fw_minutes = max(5, forward_window // 500)
    purge_buffer = pd.Timedelta(minutes=fw_minutes)
    safe_cutoff = test_ts - purge_buffer
    safe_mask = (train_ts.notna() & (train_ts <= safe_cutoff)).to_numpy(dtype=bool)
    purged_idx = train_idx[safe_mask]
    metadata = {
        "method": "forward_window_heuristic",
        "used_label_window_end": False,
        "purge_rule": "timestamp <= test_start - purge_buffer",
        "test_start": str(test_ts),
        "safe_cutoff": str(safe_cutoff),
        "forward_window_ticks": forward_window,
        "purge_buffer_minutes": int(fw_minutes),
        "n_train_before": n_before,
        "n_train_after": int(len(purged_idx)),
        "n_purged": int(n_before - len(purged_idx)),
    }
    return purged_idx, metadata


def check_quality_gates(eval_result) -> dict:
    """Apply quality thresholds and return a JSON-serializable gate object."""
    fold_precisions = [float(f.get("precision", 0.0)) for f in eval_result.fold_metrics]
    fold_std = float(np.std(fold_precisions)) if fold_precisions else 1.0
    permutation_p = float(
        eval_result.permutation_p_value if eval_result.permutation_p_value is not None else 1.0
    )
    roc_auc = float(eval_result.roc_auc or 0.0)
    brier_score = float(eval_result.brier_score if eval_result.brier_score is not None else 1.0)

    def gate(value, threshold, passed: bool, operator: str) -> dict[str, object]:
        return {
            "passed": bool(passed),
            "value": _json_safe(value),
            "threshold": _json_safe(threshold),
            "operator": operator,
        }

    gates = {
        "Precision >= 0.55": gate(
            float(eval_result.precision),
            0.55,
            eval_result.precision >= 0.55,
            ">=",
        ),
        "Permutation p < 0.05": gate(
            permutation_p,
            0.05,
            permutation_p < 0.05,
            "<",
        ),
        "Fold stability (std < 0.15)": gate(
            fold_std,
            0.15,
            fold_std < 0.15,
            "<",
        ),
        "ROC-AUC > 0.55": gate(
            roc_auc,
            0.55,
            roc_auc > 0.55,
            ">",
        ),
        "Brier score < 0.25": gate(
            brier_score,
            0.25,
            brier_score < 0.25,
            "<",
        ),
        "Test samples >= 200": gate(
            int(eval_result.n_samples),
            200,
            eval_result.n_samples >= 200,
            ">=",
        ),
    }

    all_passed = bool(all(g["passed"] for g in gates.values()))
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
    allow_failed_gates: bool = False,
) -> Path:
    """Save model artifacts and evaluation results.

    Includes ALL metrics shown in the training UI so the saved artifact
    is a complete record of the training run.
    """
    from alpha_lab.agents.data_infra.ml.model_trainer import ExtremaModelTrainer

    quality_gates = check_quality_gates(eval_result)
    quality_gates["allow_failed_gates"] = bool(allow_failed_gates)
    if not quality_gates["all_passed"] and not allow_failed_gates:
        failed = [name for name, gate in quality_gates["gates"].items() if not gate["passed"]]
        msg = (
            "Model quality gates failed; refusing to save. "
            f"Failed gates: {', '.join(failed)}. "
            "Pass allow_failed_gates=True to explicitly override."
        )
        raise ValueError(msg)

    # W2 P3b: artifact honesty — a bundle without its training evidence or its
    # strategy contract is a partial bundle; refuse up front instead of writing
    # one (F27's silent-skip class). Any writer failure below aborts the save.
    if training_result is None:
        msg = (
            "save_trained_model requires training_result; refusing to save a "
            "bundle without its training evidence"
        )
        raise ValueError(msg)
    if config is None:
        msg = (
            "save_trained_model requires config; refusing to save a bundle "
            "without a strategy contract"
        )
        raise ValueError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model + metadata
    ExtremaModelTrainer.save_model(trained_model, output_dir)

    # W2 P3b: binary-integrity sidecar (Trade-Lab verifies it at activation;
    # the store had none). Written unconditionally; failure aborts the save.
    model_file = output_dir / "model.cbm"
    digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    (output_dir / "model.cbm.sha256").write_text(digest + "  model.cbm" + chr(10), encoding="utf-8")

    # Save evaluation — start with the EvaluationResult dataclass
    eval_dict = asdict(eval_result)
    eval_dict["quality_gates"] = quality_gates
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
    eval_dict["specificity_tnr"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    eval_dict["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    eval_dict["predicted_positive_rate"] = (
        (tp + fp) / eval_result.n_samples if eval_result.n_samples > 0 else 0.0
    )

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
    session_experiment_payload = None
    session_filter_payload = None
    if training_result is not None:
        eval_dict["rth_fraction"] = training_result.get("rth_fraction")
        eval_dict["feature_stability"] = training_result.get("feature_stability")
        eval_dict["n_purged_total"] = training_result.get("n_purged_total", 0)
        eval_dict["label_purge"] = _json_safe(training_result.get("label_purge"))
        eval_dict["n_total_folds"] = training_result.get("n_total_folds")
        eval_dict["n_valid_folds"] = training_result.get("n_valid_folds")
        eval_dict["n_skipped_folds"] = training_result.get("n_skipped_folds", 0)
        eval_dict["selected_features"] = training_result.get("selected_features")
        eval_dict["class_balance"] = training_result.get("class_balance")
        eval_dict["full_dataset_class_balance"] = training_result.get("full_dataset_class_balance")
        session_experiment_payload = training_result.get("session_experiment")
        session_filter_payload = training_result.get("session_filter")
        if session_experiment_payload is not None:
            eval_dict["session_experiment"] = _json_safe(session_experiment_payload)
        if session_filter_payload is not None:
            eval_dict["session_filter"] = _json_safe(session_filter_payload)
        for key in ("gated_oos", "session_metrics", "oos_three_class_balance"):
            if training_result.get(key) is not None:
                eval_dict[key] = _json_safe(training_result.get(key))
        # W2 P3a (F27): the OOS writer is UNCONDITIONAL. An empty frame still
        # writes the parquet WITH its schema; a missing/illtyped frame or a
        # failed write fails the save loudly (no silent skip, no partial bundle).
        oos_predictions = training_result.get("oos_predictions")
        if not isinstance(oos_predictions, pd.DataFrame):
            msg = (
                "training_result lacks an oos_predictions DataFrame; refusing to "
                "save a bundle without its row-level OOS evidence"
            )
            raise ValueError(msg)
        oos_predictions.to_parquet(output_dir / "oos_predictions.parquet", index=False)
        eval_dict["oos_predictions_file"] = "oos_predictions.parquet"
        eval_dict["oos_predictions_rows"] = int(len(oos_predictions))

        confidence_stats = training_result.get("confidence_threshold_stats")
        oos_for_conf = training_result.get("oos_summary") or {}
        if not confidence_stats:
            confidence_stats = oos_for_conf.get("confidence_threshold_stats")
        if not confidence_stats:
            for row in oos_for_conf.get("threshold_table", []):
                if abs(float(row.get("threshold", -1.0)) - 0.70) >= 1e-9:
                    continue
                n_oos = int(oos_for_conf.get("n_samples") or eval_result.n_samples or 0)
                trade_count = row.get("trade_count")
                if trade_count is None:
                    trade_count = int(round(float(row.get("coverage", 0.0)) * n_oos))
                confidence_stats = {
                    "0.70": {
                        "threshold": 0.70,
                        "coverage": float(row.get("coverage", 0.0)),
                        "precision": float(row.get("precision", 0.0)),
                        "trade_count": int(trade_count),
                    },
                }
                break
        if confidence_stats:
            eval_dict["confidence_threshold_stats"] = _json_safe(confidence_stats)

        oos = training_result.get("oos_summary")
        if oos:
            eval_dict["oos_summary"] = _json_safe(oos)

    # ── Full pipeline config for reproducibility ─────────────────
    pipeline_config_payload = None
    if config is not None:
        eval_dict["training_mode"] = getattr(config, "training_mode", "unknown")
        pipeline_config_payload = config.model_dump()
        eval_dict["full_config"] = pipeline_config_payload
        if session_experiment_payload is None:
            session_experiment_payload = pipeline_config_payload.get("session_experiment")
        if session_experiment_payload is not None:
            eval_dict["session_experiment"] = _json_safe(session_experiment_payload)

    # ── Training dates for reproducibility ────────────────────────
    if dates_used is not None:
        eval_dict["dates_used"] = dates_used
        eval_dict["date_range"] = {
            "start": dates_used[0] if dates_used else None,
            "end": dates_used[-1] if dates_used else None,
            "count": len(dates_used),
        }

    with open(output_dir / "evaluation.json", "w") as f:
        json.dump(_json_safe(eval_dict), f, indent=2, default=str)

    # Augment model metadata with the full pipeline/session scope. The trainer's
    # own metadata only knows ModelConfig; session experiments live at pipeline
    # scope and must remain auditable from the saved bundle.
    # W2 P3b: metadata augmentation is part of the bundle contract — a failure
    # here aborts the save (it previously logged and shipped a partial bundle).
    metadata_path = output_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    if pipeline_config_payload is not None:
        metadata["pipeline_config"] = _json_safe(pipeline_config_payload)
    if session_experiment_payload is not None:
        metadata["session_experiment"] = _json_safe(session_experiment_payload)
    if session_filter_payload is not None:
        metadata["session_filter"] = _json_safe(session_filter_payload)
    with open(metadata_path, "w") as f:
        json.dump(_json_safe(metadata), f, indent=2, default=str)

    # ── Strategy contract for runtime (Trade-Lab) consumption ─────
    # Emits strategy.json describing the full strategy semantics (sessions,
    # touch rule, feature windows, label policy) so a runtime can be driven by
    # the contract instead of hardcoding semantics. W2 P3b: emission is
    # UNCONDITIONAL and a failure ABORTS the save — the old "never fail the
    # save" swallow was the partial-bundle generator.
    from alpha_lab.agents.data_infra.ml.strategy_contract import (
        build_strategy_contract,
    )

    selected = training_result.get("selected_features")
    if not selected:
        selected = getattr(trained_model, "selected_features", None)

    # E2: strategy_id is the REGISTRY ROUTER id (the plugin this bundle
    # runs), not the bundle name; the bundle keeps its dir-name identity.
    strategy = build_strategy_contract(config, selected, strategy_id="touch_reversal")
    if strategy is None:
        msg = (
            f"build_strategy_contract returned no contract for training_mode "
            f"{getattr(config, 'training_mode', 'unknown')!r}; refusing to save "
            "an unservable bundle"
        )
        raise ValueError(msg)
    with open(output_dir / "strategy.json", "w") as f:
        json.dump(strategy, f, indent=2, default=str)

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

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="#26a69a",
        )
    )
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

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Pred Crossing", "Pred Rebound"],
            y=["Actual Crossing", "Actual Rebound"],
            text=text,
            texttemplate="%{text}",
            colorscale="Teal",
            showscale=False,
        )
    )
    fig.update_layout(
        height=300,
        template="plotly_dark",
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
    fig.add_trace(
        go.Scatter(
            x=folds,
            y=precisions,
            name="Precision",
            mode="lines+markers",
            line=dict(color="#26a69a"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=folds,
            y=f1s,
            name="F1",
            mode="lines+markers",
            line=dict(color="#1E88E5"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=folds,
            y=recalls,
            name="Recall",
            mode="lines+markers",
            line=dict(color="#FFA726"),
        )
    )
    fig.add_hline(
        y=0.55, line_dash="dash", line_color="red", annotation_text="Min Precision (0.55)"
    )
    fig.update_layout(
        height=300,
        template="plotly_dark",
        title="Walk-Forward Fold Metrics",
        xaxis_title="Fold",
        yaxis_title="Score",
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
            "Strategy-Core v3 research semantics."
        ),
    )
    is_utility_mode = training_mode == "Dashboard Utility (3-class)"

    if is_utility_mode:
        st.caption(
            "Dashboard-utility mode: train a 3-class CatBoost model on "
            "level-touch events with utility-aligned TP/SL labeling. "
            "It emits a v3 strategy contract, but Trade-Lab runtime activation "
            "remains blocked until the runtime is repointed/parity-tested."
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
            "Data directory",
            value=str(_DEFAULT_DATA_DIR),
            key="ml_data_dir",
        )
        # Auto-detect date range from available data
        _avail = get_available_dates(ml_symbol, Path(ml_data_dir))
        _first = date.fromisoformat(_avail[0]) if _avail else date.today() - timedelta(days=90)
        _last = date.fromisoformat(_avail[-1]) if _avail else date.today() - timedelta(days=1)
        ml_start = st.date_input(
            "Start date",
            value=_first,
            key="ml_start",
        )
        ml_end = st.date_input(
            "End date",
            value=_last,
            key="ml_end",
        )

    with col_cfg2:
        st.markdown("**Walk-Forward**")
        # Auto-compute max sensible windows from data span
        _span = (_last - _first).days
        _default_train = min(30, max(5, _span // 4))
        _default_test = min(7, max(2, _span // 12))
        ml_train_days = st.slider(
            "Train window (days)",
            5,
            90,
            _default_train,
            key="ml_train_days",
        )
        ml_test_days = st.slider(
            "Test window (days)",
            2,
            30,
            _default_test,
            key="ml_test_days",
        )
        ml_gap_days = st.slider(
            "Gap (days)",
            0,
            5,
            1,
            key="ml_gap_days",
        )
        # Show expected fold count
        _min_1 = ml_train_days + ml_gap_days + ml_test_days
        _est_folds = max(0, (_span - ml_train_days - ml_gap_days) // ml_test_days)
        st.caption(f"~{_est_folds} folds from {_span}d span (need {_min_1}d for 1st fold)")
        if is_utility_mode:
            ml_tp = st.slider("TP (points)", 5, 50, 15, key="ml_util_tp")
            ml_sl = st.slider("SL (points)", 5, 50, 30, key="ml_util_sl")
            ml_bar_type = st.selectbox(
                "Bar type",
                _DASHBOARD_UTILITY_BAR_TYPES,
                key="ml_bar_type",
                help="Tick bars give finer touch detection than 1m time bars",
            )
            ml_int_window = st.slider(
                "Interaction window (min)",
                1,
                15,
                5,
                key="ml_int_window",
            )
            ml_approach = st.checkbox(
                "Include approach features",
                key="ml_approach",
                help="Add 27 pre-touch order flow features (90-min window)",
            )
            ml_approach_window = 90
            if ml_approach:
                ml_approach_window = st.slider(
                    "Approach window (min)",
                    15,
                    120,
                    90,
                    key="ml_approach_window",
                )
        else:
            ml_label = st.selectbox(
                "Label threshold",
                list(_LABEL_OPTIONS.keys()),
                key="ml_label",
            )

    # QL-UI-PARITY training controls (utility mode): resolved into
    # run_walk_forward_training kwargs via resolve_training_kwargs.
    ml_pinned_features: list[str] = []
    ml_fold_scheme = "calendar"
    ml_fold_params: dict[str, int] = {}
    with col_cfg3:
        st.markdown("**CatBoost**")
        ml_iterations = st.slider(
            "Iterations",
            100,
            2000,
            500,
            50,
            key="ml_iterations",
        )
        ml_depth = st.slider("Tree depth", 3, 10, 6, key="ml_depth")
        if is_utility_mode:
            if "ml_dataset" in st.session_state:
                # Same feature-universe derivation as the dataset preview;
                # the core validates the subset and bypasses RFECV when a
                # pin list is present.
                _pin_universe = [
                    c
                    for c in st.session_state["ml_dataset"].columns
                    if c.startswith(("int_", "app_"))
                ]
                ml_pinned_features = st.multiselect(
                    "Pin features (exact model feature_set)",
                    _pin_universe,
                    default=[],
                    key="ml_pin_features",
                    help=(
                        "Exact feature list to train/serve, in selection "
                        "order. Leave empty to train on all features (or "
                        "RFECV selection when enabled)."
                    ),
                )
                if ml_pinned_features:
                    st.caption(
                        "Pinned: RFECV bypassed; these features become the "
                        "model + contract feature_set verbatim."
                    )
            else:
                st.multiselect(
                    "Pin features (exact model feature_set)",
                    [],
                    key="ml_pin_features_placeholder",
                    disabled=True,
                )
                st.caption("Build the dataset first")
        if ml_pinned_features:
            # Forced-off display: pinning bypasses RFECV in the core.
            st.checkbox(
                "RFECV feature selection",
                value=False,
                key="ml_rfecv_pinned_off",
                disabled=True,
                help="Pinned features bypass RFECV; selection is forced off.",
            )
            ml_rfecv = False
        else:
            ml_rfecv = st.checkbox(
                "RFECV feature selection",
                value=False,
                key="ml_rfecv",
                help="Recursive feature elimination — slower but may improve.",
            )
        if is_utility_mode:
            ml_fold_choice = st.selectbox(
                "Fold scheme",
                ["calendar (walk-forward)", "purged trading days"],
                index=0,
                key="ml_fold_scheme",
                help=(
                    "calendar = WalkForwardSplitter over the train/test/gap "
                    "sliders; purged trading days = contiguous TRADING-day "
                    "folds with a purge gap (the D-036 CLI scheme)."
                ),
            )
            if ml_fold_choice == "purged trading days":
                ml_fold_scheme = "purged-days"
                ml_fold_params = {
                    "train_days": int(
                        st.number_input(
                            "Fold train days",
                            min_value=1,
                            value=40,
                            key="ml_fold_train_days",
                        )
                    ),
                    "test_days": int(
                        st.number_input(
                            "Fold test days",
                            min_value=1,
                            value=5,
                            key="ml_fold_test_days",
                        )
                    ),
                    "step_days": int(
                        st.number_input(
                            "Fold step days",
                            min_value=1,
                            value=5,
                            key="ml_fold_step_days",
                        )
                    ),
                    "purge_days": int(
                        st.number_input(
                            "Fold purge days",
                            min_value=0,
                            value=2,
                            key="ml_fold_purge_days",
                        )
                    ),
                    "min_train_events": int(
                        st.number_input(
                            "Min train events",
                            min_value=1,
                            value=30,
                            key="ml_min_train_events",
                        )
                    ),
                }

    ml_session_preset = "all_to_ny"
    ml_train_sessions = ["asia", "london", "ny"]
    ml_eval_sessions = ["asia", "london", "ny"]
    ml_gate_sessions = ["ny"]
    ml_report_session_breakdowns = True
    if is_utility_mode:
        from alpha_lab.agents.data_infra.ml.config import (
            ALLOWED_STRATEGY_SESSIONS,
            SESSION_EXPERIMENT_PRESETS,
            session_experiment_from_preset,
        )

        with st.expander("Session experiment scope", expanded=True):
            ml_session_preset = st.selectbox(
                "Preset",
                list(SESSION_EXPERIMENT_PRESETS.keys()),
                index=list(SESSION_EXPERIMENT_PRESETS.keys()).index("all_to_ny"),
                key="ml_session_preset",
                help=(
                    "Controls which Strategy-Core sessions train, which sessions are "
                    "included in OOS stats, and which sessions are eligible for the "
                    "confidence-gated research report."
                ),
            )
            preset_scope = session_experiment_from_preset(ml_session_preset)
            s1, s2, s3 = st.columns(3)
            with s1:
                ml_train_sessions = st.multiselect(
                    "Training sessions",
                    list(ALLOWED_STRATEGY_SESSIONS),
                    default=preset_scope.training_sessions,
                    key=f"ml_train_sessions_{ml_session_preset}",
                )
            with s2:
                ml_eval_sessions = st.multiselect(
                    "Evaluation sessions",
                    list(ALLOWED_STRATEGY_SESSIONS),
                    default=preset_scope.evaluation_sessions,
                    key=f"ml_eval_sessions_{ml_session_preset}",
                )
            with s3:
                ml_gate_sessions = st.multiselect(
                    "Confidence-gate sessions",
                    list(ALLOWED_STRATEGY_SESSIONS),
                    default=preset_scope.production_gate_sessions,
                    key=f"ml_gate_sessions_{ml_session_preset}",
                )
            ml_report_session_breakdowns = st.checkbox(
                "Report per-session OOS breakdowns",
                value=preset_scope.report_session_breakdowns,
                key=f"ml_report_session_breakdowns_{ml_session_preset}",
            )
            st.caption(
                "Default remains all-session training/evaluation with NY-only gate. "
                "Session scope does not change cached feature generation."
            )

    st.divider()

    data_dir = Path(ml_data_dir)
    label_col = "label_encoded" if is_utility_mode else _LABEL_OPTIONS[ml_label]

    # Build a fingerprint of the current UI selections so we can detect
    # when the user changes settings and the stored dataset is stale.
    _util_identity = ""
    if is_utility_mode:
        _util_identity = (
            f"|{ml_bar_type}|{ml_int_window}|{ml_approach}|"
            f"{ml_approach_window}|{ml_tp}|{ml_sl}|{ml_session_preset}|"
            f"train={','.join(ml_train_sessions)}|eval={','.join(ml_eval_sessions)}|"
            f"gate={','.join(ml_gate_sessions)}|breakdowns={ml_report_session_breakdowns}"
        )
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
        dates_in_range = [d for d in available if ml_start.isoformat() <= d <= ml_end.isoformat()]
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
        cached_in_range = [
            d for d in cached_dates if ml_start.isoformat() <= d <= ml_end.isoformat()
        ]
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
                "Build Dataset",
                key="ml_build_btn",
                disabled=len(dates_in_range) == 0,
            )
        with col_clear:
            clear_btn = st.button(
                "Clear Cache",
                key="ml_clear_cache",
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
                SessionExperimentConfig,
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
                    session_experiment=SessionExperimentConfig(
                        training_sessions=ml_train_sessions,
                        evaluation_sessions=ml_eval_sessions,
                        production_gate_sessions=ml_gate_sessions,
                        report_session_breakdowns=ml_report_session_breakdowns,
                    ),
                    tick_size=0.25,
                    instrument=ml_symbol,
                )
                from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
                    build_utility_dataset,
                )

                progress = st.progress(0, text="Building utility dataset...")
                dataset = build_utility_dataset(
                    dates_in_range,
                    data_dir,
                    build_config,
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
                    # Re-render so dataset-gated Train-step controls (the pin
                    # multiselect) pick up the new dataset on this page view
                    # (mirrors the Clear Cache handler).
                    st.rerun()
            else:
                build_config = MLPipelineConfig(
                    training_mode="extrema_rebound_crossing",
                    features=FeatureConfig(include_signal_features=False),
                    tick_size=0.25,
                    instrument=ml_symbol,
                )

                progress = st.progress(0, text="Building dataset...")
                dataset = build_training_dataset(
                    ml_symbol,
                    dates_in_range,
                    data_dir,
                    build_config,
                    progress,
                )

                if dataset.empty:
                    st.error("No extrema detected. Try a longer date range.")
                else:
                    st.session_state["ml_dataset"] = dataset
                    st.session_state["ml_build_config"] = build_config
                    # Clear stale training results from previous dataset
                    st.session_state.pop("ml_training_result", None)
                    st.session_state.pop("ml_train_config", None)
                    # Re-render so dataset-gated Train-step controls pick up
                    # the new dataset (mirrors the Clear Cache handler).
                    st.rerun()

        if "ml_dataset" in st.session_state:
            dataset = st.session_state["ml_dataset"]

            if is_utility_mode:
                # 3-class display
                has_label_col = label_col in dataset.columns
                n_valid = dataset[label_col].notna().sum() if has_label_col else 0
                n_rev = int((dataset[label_col] == 0).sum()) if has_label_col else 0
                n_trap = int((dataset[label_col] == 1).sum()) if has_label_col else 0
                n_bt = int((dataset[label_col] == 2).sum()) if has_label_col else 0
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Touch Events", len(dataset))
                with c2:
                    st.metric("Reversal (0)", n_rev)
                with c3:
                    st.metric("Trap (1)", n_trap)
                with c4:
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
                        "timestamp",
                        "direction",
                        "representative_price",
                        "label",
                        label_col,
                    ] + feature_cols_display[:6]
                    show_cols = [c for c in display_cols if c in dataset.columns]
                    st.dataframe(
                        dataset[show_cols].head(20),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                # Binary extrema display
                has_label_col = label_col in dataset.columns
                n_valid = dataset[label_col].notna().sum() if has_label_col else 0
                n_rebound = int((dataset[label_col] == 1).sum()) if has_label_col else 0
                n_crossing = int((dataset[label_col] == 0).sum()) if has_label_col else 0

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Total Extrema", len(dataset))
                with c2:
                    st.metric("Labeled", int(n_valid))
                with c3:
                    st.metric("Rebound (1)", n_rebound)
                with c4:
                    st.metric("Crossing (0)", n_crossing)

                feature_cols = [c for c in dataset.columns if c.startswith(("pl_", "ms_"))]
                st.caption(f"{len(feature_cols)} features: {', '.join(feature_cols[:10])}...")

                with st.expander("Preview dataset (first 20 rows)"):
                    display_cols = [
                        "timestamp",
                        "price",
                        "extremum_type",
                        label_col,
                    ] + feature_cols[:5]
                    show_cols = [c for c in display_cols if c in dataset.columns]
                    st.dataframe(
                        dataset[show_cols].head(20),
                        use_container_width=True,
                        hide_index=True,
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
                SessionExperimentConfig,
                WalkForwardConfig,
            )

            train_kwargs = resolve_training_kwargs(
                ml_pinned_features,
                ml_fold_scheme,
                ml_fold_params,
                ml_rfecv,
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
                        rfecv_enabled=train_kwargs["rfecv_enabled"],
                        loss_function="MultiClass",
                    ),
                    session_experiment=SessionExperimentConfig(
                        training_sessions=ml_train_sessions,
                        evaluation_sessions=ml_eval_sessions,
                        production_gate_sessions=ml_gate_sessions,
                        report_session_breakdowns=ml_report_session_breakdowns,
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
                        day_folds=train_kwargs["day_folds"],
                        pinned_features=train_kwargs["pinned_features"],
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
            session_filter = result.get("session_filter") or oos.get("session_filter") or {}
            if is_utility_mode and session_filter:
                st.caption(
                    "Session experiment: "
                    f"train={session_filter.get('training_sessions')}, "
                    f"eval={session_filter.get('evaluation_sessions')}, "
                    f"gate={session_filter.get('production_gate_sessions')} | "
                    f"train rows={session_filter.get('training_candidate_rows')}, "
                    f"eval rows={session_filter.get('evaluation_candidate_rows')}"
                )

            # Aggregate metrics
            st.markdown("#### Out-of-Sample (OOS) Aggregate Metrics")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric(
                    "Precision",
                    f"{ev.precision:.3f}",
                    help=f"95% CI: [{ev.precision_ci[0]:.3f}, {ev.precision_ci[1]:.3f}]",
                )
            with c2:
                st.metric("Recall", f"{ev.recall:.3f}")
            with c3:
                st.metric(
                    "F1", f"{ev.f1:.3f}", help=f"95% CI: [{ev.f1_ci[0]:.3f}, {ev.f1_ci[1]:.3f}]"
                )
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
                st.metric(
                    "Permutation p-value",
                    f"{ev.permutation_p_value:.4f}"
                    if ev.permutation_p_value is not None
                    else "N/A",
                )
            with c3:
                st.metric("Cohen's d", f"{ev.cohens_d:.3f}" if ev.cohens_d is not None else "N/A")
            with c4:
                balance = result["class_balance"]
                if is_utility_mode:
                    st.metric(
                        "OOS Class Balance",
                        f"{balance.get('tradeable_reversal', 0)} Rev / "
                        f"{balance.get('non_tradeable', 0)} Non",
                    )
                else:
                    st.metric(
                        "OOS Class Balance",
                        f"{balance['rebound']}R / {balance['crossing']}C",
                    )

            if is_utility_mode:
                gated = result.get("gated_oos") or oos.get("gated_oos") or {}
                gate_label = gated.get("eligible_session", "ny")
                st.markdown(
                    "#### Confidence-Gated OOS — "
                    f"`session in [{gate_label}]` and "
                    "`P(tradeable_reversal) >= 0.70`"
                )
                g1, g2, g3, g4, g5 = st.columns(5)
                with g1:
                    st.metric("Gated trades", gated.get("trade_count", 0))
                with g2:
                    st.metric("Gate precision", f"{gated.get('precision', 0.0):.3f}")
                with g3:
                    st.metric("Coverage", f"{gated.get('coverage', 0.0):.1%}")
                with g4:
                    st.metric("Eligible coverage", f"{gated.get('eligible_coverage', 0.0):.1%}")
                with g5:
                    st.metric("Idealized EV", f"{gated.get('expectancy_15_30_pts', 0.0):+.2f} pts")
                if gated.get("trade_count", 0) < 30:
                    st.warning(
                        "Production-gate sample is small. Treat aggregate OOS metrics as "
                        "diagnostic only until the gated NY sample is large enough."
                    )
                session_metrics = result.get("session_metrics") or oos.get("session_metrics") or {}
                if session_metrics:
                    with st.expander("Session-filtered OOS metrics"):
                        st.dataframe(
                            pd.DataFrame.from_dict(session_metrics, orient="index"),
                            use_container_width=True,
                        )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Specificity (TNR)", f"{oos_rates.get('tnr_specificity', 0.0):.3f}")
            with c2:
                st.metric("False Positive Rate (FPR)", f"{oos_rates.get('fpr', 0.0):.3f}")
            with c3:
                st.metric(
                    "Predicted Rebound Rate", f"{oos_rates.get('predicted_positive_rate', 0.0):.3f}"
                )

            # RTH coverage, label purging, and feature stability info
            rth_frac = result.get("rth_fraction")
            n_purged = result.get("n_purged_total", 0)
            label_purge = result.get("label_purge") or {}
            feat_stability = result.get("feature_stability")
            c1, c2, c3 = st.columns(3)
            with c1:
                if rth_frac is not None:
                    st.metric(
                        "Legacy RTH Proxy",
                        f"{rth_frac:.1%}",
                        help=("Legacy OOS coverage proxy; v3 production gate uses session == ny"),
                    )
                    if rth_frac < 0.5:
                        st.warning(
                            "Low RTH coverage — OOS metrics may not represent "
                            "the dashboard's NY RTH execution population."
                        )
            with c2:
                if n_purged > 0:
                    purge_method = label_purge.get("method", "unknown")
                    st.metric(
                        "Label-Purged Rows",
                        n_purged,
                        help=(
                            "Training rows removed to prevent forward-window "
                            f"label leakage. Purge method: {purge_method}."
                        ),
                    )
            with c3:
                if feat_stability is not None:
                    st.metric(
                        "Feature Stability",
                        f"{feat_stability:.3f}",
                        help=(
                            "Mean pairwise Spearman rank-correlation of feature "
                            "importances across folds (1.0 = perfectly stable)"
                        ),
                    )
                    if feat_stability < 0.5:
                        st.warning(
                            "Low feature stability — model may rely on non-stationary signals."
                        )

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
                gate_cols,
                gates_result["gates"].items(),
                strict=False,
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
                st.metric(
                    "Expectancy (15/15)",
                    f"{utility_15_15['expectancy_pts']:+.2f} pts",
                    help="Expected points per trade at TP=15 SL=15",
                )
            with c2:
                st.metric(
                    "Expectancy (15/30)",
                    f"{utility_15_30['expectancy_pts']:+.2f} pts",
                    help="Expected points per trade at TP=15 SL=30",
                )
            with c3:
                st.metric(
                    "Profit Factor (15/30)",
                    f"{utility_15_30['profit_factor']:.2f}",
                    help="Gross gains / gross losses at TP=15 SL=30",
                )
            trade_label = "tradeable-reversal" if is_utility_mode else "rebound"
            st.caption(
                f"Based on {utility_15_30['n_simulated_trades']} OOS predicted-{trade_label} "
                "trades. Assumes all predicted positives are executed at stated TP/SL."
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

            confidence_stats = (
                result.get("confidence_threshold_stats")
                or oos.get("confidence_threshold_stats")
                or {}
            )
            threshold_070 = confidence_stats.get("0.70")
            if threshold_070:
                st.caption(
                    "Confidence ≥ 0.70 OOS stats: "
                    f"{threshold_070['trade_count']} trades, "
                    f"coverage {threshold_070['coverage']:.1%}, "
                    f"precision {threshold_070['precision']:.3f}."
                )

            if oos.get("calibration_table"):
                st.markdown("#### OOS Calibration Buckets")
                calib_df = pd.DataFrame(oos["calibration_table"])
                for col in [
                    "bucket_low",
                    "bucket_high",
                    "mean_predicted_prob",
                    "observed_positive_rate",
                ]:
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
                "Model name",
                value=default_name,
                key="ml_model_name",
            )
            output_dir = _DEFAULT_MODEL_DIR / model_name

            allow_failed_gates = False
            if not gates_result["all_passed"]:
                allow_failed_gates = st.checkbox(
                    "Allow save despite failed quality gates",
                    value=False,
                    key="ml_allow_failed_gates",
                    help=(
                        "Explicit override. Failed gates will be written to "
                        "evaluation.json so downstream code can audit the save."
                    ),
                )

            save_btn = st.button(
                "Save Model",
                key="ml_save_btn",
                disabled=not gates_result["all_passed"] and not allow_failed_gates,
            )
            if save_btn:
                try:
                    saved_path = save_trained_model(
                        result["trained_model"],
                        ev,
                        st.session_state.get("ml_train_config"),
                        output_dir,
                        training_result=result,
                        dates_used=dates_in_range if available else None,
                        allow_failed_gates=allow_failed_gates,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["ml_saved_path"] = str(saved_path)
                    st.success(f"Model saved to `{saved_path}`")
                    st.info(
                        "Saved as a research/training bundle. Do not use it for "
                        "Trade-Lab signal generation until the runtime repoint/parity "
                        "gate is explicitly passed.",
                    )

            if "ml_saved_path" in st.session_state:
                st.caption(f"Last saved: {st.session_state['ml_saved_path']}")
