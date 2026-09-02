"""Bundle-aware CatBoost rung — ``ifvg_context_catboost_bundle_v1`` (R6.1 §6.J).

The frozen M0–M3 lane (``context_model.py``) fits CatBoost over a frozen
TIER feature list and is never modified in V1. Bundle-parametrized studies
(the controlled MBP-1 study, the regime studies) need the same nonlinear
challenger over the exact resolved feature names of one bundle plus
optional fold-LOCAL regime features. This module is that rung: it mirrors
``logistic_model.py`` in loop shape (fit on training rows only; predict the
test rows; typed invalid folds; deterministic seed-7 permutation
importance) and pins its own protocol identity — the CatBoost parameters
REUSED BY VALUE from the frozen lane, the preprocessing / missingness
policy (native NaN for numerics; the registered missing token for
categoricals; no imputation, no column dropping), the categorical registry
snapshot (the frozen registry ∪ the block-declared categoricals of the
bundle, in feature order), the package + Python versions, the ordered
features, the resolved bundle id, the feature registry hash, the fold-set
id, the fold-schedule id, the seed, the threshold grid, and the calibration
policy. Prediction rows carry the D13 ``comparison_row_id`` next to the
legacy view-scoped ``oos_row_id``. No search, selection, or promotion
surface exists: the rung is research-only.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import version
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from ..context_experiment_contracts import (
    IFVG_CONTEXT_CALIBRATION_POLICY_ID,
    IFVG_CONTEXT_MODEL_PARAMETERS,
    THRESHOLD_REPORT_GRID,
    canonical_contract_sha256,
)
from ..context_feature_view import CandidateFeatureView
from ..context_folds import ContextFoldSet
from ..context_model import (
    CATEGORICAL_FEATURE_REGISTRY,
    MISSING_CATEGORY,
    categorical_features_for,
)
from .comparison_rows import (
    RegimeFoldFeatureSource,
    assert_fold_local_source_coherent,
    candidate_fold_set_id,
    comparison_row_id,
    default_fold_schedule_id,
    join_fold_local_features,
    label_artifact_content_id,
)
from .model_protocols import CATBOOST_BUNDLE_PROTOCOL_ID

__all__ = [
    "CATBOOST_BUNDLE_PROTOCOL_ID",
    "IFVG_CONTEXT_CATBOOST_BUNDLE_PARAMETERS",
    "CATBOOST_BUNDLE_PREPROCESSING_POLICY",
    "RegimeFoldFeatureSource",
    "ResolvedCatBoostBundleProtocol",
    "CatBoostBundleModelRun",
    "resolve_catboost_bundle_protocol",
    "bundle_rung_categorical_features",
    "run_catboost_bundle_fold_models",
]

#: The frozen lane's CatBoost parameters, reused BY VALUE and pinned into
#: this protocol's identity (a change here is a new protocol, never an
#: override; ``thread_count=1`` keeps the fit single-threaded and reproducible).
IFVG_CONTEXT_CATBOOST_BUNDLE_PARAMETERS: MappingProxyType[str, Any] = MappingProxyType(
    dict(IFVG_CONTEXT_MODEL_PARAMETERS)
)

#: Preprocessing / missingness policy — part of the resolved identity.
CATBOOST_BUNDLE_PREPROCESSING_POLICY: MappingProxyType[str, Any] = MappingProxyType(
    {
        "numeric_missingness": "native_nan_no_imputation_v1",
        "categorical_missing_value": MISSING_CATEGORY,
        "column_dropping": "none",
        "categorical_policy": "frozen_registry_union_block_declared_in_feature_order_v1",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedCatBoostBundleProtocol:
    protocol_id: str
    parameters: dict[str, Any]
    preprocessing_policy: dict[str, Any]
    categorical_registry: dict[str, Any]
    package_versions: dict[str, str]
    python_version: str
    ordered_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    resolved_feature_bundle_id: str
    feature_registry_hash: str
    fold_set_id: str
    fold_schedule_id: str
    random_seed: int
    manual_feature_overrides_hash: str
    resolved_hash: str


@dataclass(frozen=True, slots=True)
class CatBoostBundleModelRun:
    protocol: ResolvedCatBoostBundleProtocol
    predictions: pd.DataFrame
    fold_reports: tuple[dict[str, Any], ...]
    feature_importance: pd.DataFrame
    #: fold_index → the categorical feature indices the FITTED model reports
    #: (``get_cat_feature_indices``) — proof that block-declared categoricals
    #: were fitted as categoricals
    fitted_categorical_indices: dict[int, tuple[int, ...]]
    label_artifact_id: str
    fold_schedule_id: str
    candidate_fold_set_id: str


def bundle_rung_categorical_features(
    features: tuple[str, ...],
    *,
    block_declared: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """The categorical features of a bundle-parametrized rung: the frozen
    M0–M3 registry's categoricals ∪ the block-declared ones, in feature
    order (D8)."""

    declared = set(categorical_features_for(features)) | set(block_declared)
    return tuple(name for name in features if name in declared)


def resolve_catboost_bundle_protocol(
    *,
    ordered_features: tuple[str, ...],
    resolved_feature_bundle_id: str,
    categorical_features: tuple[str, ...],
    feature_registry_hash: str,
    fold_set_id: str,
    fold_schedule_id: str,
    manual_feature_overrides: dict[str, Any] | None = None,
) -> ResolvedCatBoostBundleProtocol:
    """One concrete, hashed bundle-aware CatBoost protocol."""

    overrides = dict(manual_feature_overrides or {})
    forbidden = set(overrides) & {
        "features",
        "categorical_features",
        "model_parameters",
        "threshold",
    }
    if forbidden:
        raise ValueError(f"manual override cannot alter locked protocol: {sorted(forbidden)}")
    features = tuple(str(name) for name in ordered_features)
    if not features:
        raise ValueError("the bundle-aware CatBoost rung requires at least one feature")
    if len(set(features)) != len(features):
        raise ValueError("ordered features must not repeat a feature")
    categorical = tuple(str(name) for name in categorical_features)
    outside = sorted(set(categorical) - set(features))
    if outside:
        raise ValueError(f"categorical features outside the ordered features: {outside}")
    if tuple(name for name in features if name in set(categorical)) != categorical:
        raise ValueError("categorical features must be listed in feature order")
    versions = {
        package: version(package)
        for package in ("catboost", "numpy", "pandas", "scikit-learn")
    }
    override_hash = canonical_contract_sha256(overrides)
    seed = int(IFVG_CONTEXT_CATBOOST_BUNDLE_PARAMETERS["random_seed"])
    payload = {
        "protocol_id": CATBOOST_BUNDLE_PROTOCOL_ID,
        "parameters": dict(IFVG_CONTEXT_CATBOOST_BUNDLE_PARAMETERS),
        "preprocessing_policy": dict(CATBOOST_BUNDLE_PREPROCESSING_POLICY),
        "categorical_registry": CATEGORICAL_FEATURE_REGISTRY,
        "package_versions": versions,
        "python_version": platform.python_version(),
        "ordered_features": list(features),
        "categorical_features": list(categorical),
        "resolved_feature_bundle_id": str(resolved_feature_bundle_id),
        "feature_registry_hash": str(feature_registry_hash),
        "fold_set_id": str(fold_set_id),
        "fold_schedule_id": str(fold_schedule_id),
        "random_seed": seed,
        "threshold_report_grid": list(THRESHOLD_REPORT_GRID),
        "calibration_policy": IFVG_CONTEXT_CALIBRATION_POLICY_ID,
        "manual_feature_overrides_hash": override_hash,
    }
    return ResolvedCatBoostBundleProtocol(
        protocol_id=CATBOOST_BUNDLE_PROTOCOL_ID,
        parameters=dict(IFVG_CONTEXT_CATBOOST_BUNDLE_PARAMETERS),
        preprocessing_policy=dict(CATBOOST_BUNDLE_PREPROCESSING_POLICY),
        categorical_registry=dict(CATEGORICAL_FEATURE_REGISTRY),
        package_versions=versions,
        python_version=platform.python_version(),
        ordered_features=features,
        categorical_features=categorical,
        resolved_feature_bundle_id=str(resolved_feature_bundle_id),
        feature_registry_hash=str(feature_registry_hash),
        fold_set_id=str(fold_set_id),
        fold_schedule_id=str(fold_schedule_id),
        random_seed=seed,
        manual_feature_overrides_hash=override_hash,
        resolved_hash=canonical_contract_sha256(payload),
    )


def _model_frame(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    categoricals: tuple[str, ...],
) -> pd.DataFrame:
    """Categoricals as typed strings (missing → the registered token);
    numerics as float with NaN KEPT — CatBoost owns numeric missingness."""

    missing = sorted(set(features) - set(frame))
    if missing:
        raise ValueError(f"model frame is missing registered features {missing}")
    result = frame.loc[:, list(features)].copy()
    categorical_set = set(categoricals)
    for feature in features:
        if feature in categorical_set:
            column = result[feature]
            result[feature] = column.where(column.notna(), MISSING_CATEGORY).astype(str)
        else:
            result[feature] = pd.to_numeric(result[feature], errors="raise").astype(float)
    return result


def _permutation_importance(
    model: CatBoostClassifier,
    test_x: pd.DataFrame,
    test_y: np.ndarray,
    *,
    repeats: int = 20,
    seed: int = 7,
) -> dict[str, tuple[float, float]]:
    """The frozen lane's draw protocol: one seed-7 RNG per fold, columns
    walked in frame order, raw columns permuted."""

    baseline = log_loss(test_y, model.predict_proba(test_x)[:, 1], labels=[0, 1])
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {column: [] for column in test_x}
    rows = len(test_x)
    for column in test_x:
        source = test_x[column].to_numpy(copy=True)
        # the SAME draws and scores as one predict per repeat — the permuted
        # copies are stacked and scored in one call (rows are independent)
        stacked = pd.concat(
            [
                test_x.assign(**{column: source[rng.permutation(rows)]})
                for _ in range(repeats)
            ],
            ignore_index=True,
        )
        probabilities = model.predict_proba(stacked)[:, 1]
        for repeat in range(repeats):
            block = probabilities[repeat * rows : (repeat + 1) * rows]
            score = log_loss(test_y, block, labels=[0, 1])
            values[column].append(float(score - baseline))
    return {
        column: (float(np.mean(items)), float(np.var(items, ddof=0)))
        for column, items in values.items()
    }


def run_catboost_bundle_fold_models(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    features: tuple[str, ...],
    resolved_feature_bundle_id: str,
    categorical_features: tuple[str, ...] | None = None,
    fold_local_features: RegimeFoldFeatureSource | None = None,
    fold_schedule_id: str | None = None,
    label_artifact_id: str | None = None,
    manual_feature_overrides: dict[str, Any] | None = None,
    fitted_fold_sink: dict[int, CatBoostClassifier] | None = None,
) -> CatBoostBundleModelRun:
    """The same loop shape as ``run_logistic_fold_models``: fit on TRAINING
    rows only, predict the test rows, typed invalid folds.

    ``categorical_features`` defaults to the frozen registry's categoricals
    ∪ the fold-local source's block-declared categoricals (feature order);
    ``fold_schedule_id`` / ``label_artifact_id`` default to the schedule
    derived from the labeled trading days and the label content hash —
    the pipeline passes S08's / S07's exact ids.
    """

    features = tuple(str(name) for name in features)
    static_columns = tuple(str(column) for column in view.frame.columns)
    if fold_local_features is not None:
        assert_fold_local_source_coherent(
            fold_local_features, features=features, static_columns=static_columns
        )
        block_declared = tuple(fold_local_features.categorical_features)
    else:
        block_declared = ()
    categorical = (
        tuple(categorical_features)
        if categorical_features is not None
        else bundle_rung_categorical_features(features, block_declared=block_declared)
    )
    schedule_id = fold_schedule_id or default_fold_schedule_id(folds, labeled_candidates)
    # R6.1-FIX §3.6 (review RA-05): the helper default is the FULL
    # consumed-column content hash, never the narrow pair hash
    label_id = label_artifact_id or label_artifact_content_id(None, labeled_candidates)
    fold_set = candidate_fold_set_id(folds)
    protocol = resolve_catboost_bundle_protocol(
        ordered_features=features,
        resolved_feature_bundle_id=resolved_feature_bundle_id,
        categorical_features=categorical,
        feature_registry_hash=view.feature_registry_hash,
        fold_set_id=fold_set,
        fold_schedule_id=schedule_id,
        manual_feature_overrides=manual_feature_overrides,
    )
    labels = labeled_candidates.set_index("candidate_id", verify_integrity=True)
    feature_frame = view.frame.set_index("candidate_id", verify_integrity=True)
    if not set(labels.index.astype(str)).issubset(set(feature_frame.index.astype(str))):
        raise ValueError("labels contain candidate IDs outside the immutable feature view")
    merged = feature_frame.join(
        labels[["binary_target", "gross_r", "net_r", "trading_day", "setup_id"]],
        how="left",
        rsuffix="_label",
        validate="one_to_one",
    )
    predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    fitted_categorical_indices: dict[int, tuple[int, ...]] = {}
    for fold in folds.folds:
        report: dict[str, Any] = {
            "fold_index": fold.fold_index,
            "valid": fold.valid,
            "invalid_reason": fold.invalid_reason,
            "training_prevalence": fold.training_prevalence,
            "train_candidates": len(fold.train_candidate_ids),
            "test_candidates": len(fold.test_candidate_ids),
        }
        if not fold.valid:
            fold_reports.append(report)
            continue
        train = merged.loc[list(fold.train_candidate_ids)]
        test = merged.loc[list(fold.test_candidate_ids)]
        if fold_local_features is not None:
            # fold k's own rows only: train rows carry fit k's in-sample
            # assignment, test rows fit k's OOS assignment — never another fold
            train = join_fold_local_features(
                train, fold_local_features, fold_index=fold.fold_index
            )
            test = join_fold_local_features(
                test, fold_local_features, fold_index=fold.fold_index
            )
        train_x = _model_frame(train, features=features, categoricals=categorical)
        test_x = _model_frame(test, features=features, categoricals=categorical)
        train_y = pd.to_numeric(train["binary_target"], errors="raise").astype(int).to_numpy()
        test_y = pd.to_numeric(test["binary_target"], errors="raise").astype(int).to_numpy()
        model = CatBoostClassifier(**protocol.parameters)
        model.fit(train_x, train_y, cat_features=list(categorical))
        fitted_categorical_indices[fold.fold_index] = tuple(
            int(index) for index in model.get_cat_feature_indices()
        )
        if fitted_fold_sink is not None:
            fitted_fold_sink[fold.fold_index] = model
        probabilities = model.predict_proba(test_x)[:, 1]
        for candidate_id, probability in zip(test.index.astype(str), probabilities, strict=True):
            source = test.loc[candidate_id]
            predictions.append(
                {
                    "oos_row_id": canonical_contract_sha256(
                        {
                            "view_id": view.view_id,
                            "fold_index": fold.fold_index,
                            "candidate_id": candidate_id,
                        }
                    ),
                    "comparison_row_id": comparison_row_id(
                        fold_schedule_id=schedule_id,
                        candidate_fold_set_id=fold_set,
                        fold_index=fold.fold_index,
                        candidate_id=candidate_id,
                        label_artifact_id=label_id,
                    ),
                    "candidate_id": candidate_id,
                    "setup_id": source.get("setup_id_label", source.get("setup_id")),
                    "trading_day": source.get(
                        "trading_day_label", source.get("trading_day")
                    ),
                    "fold_index": fold.fold_index,
                    "training_prevalence": fold.training_prevalence,
                    "target": int(source["binary_target"]),
                    "probability": float(probability),
                    "gross_r": float(source["gross_r"]),
                    "net_r": float(source["net_r"]),
                }
            )
        native = model.get_feature_importance(type="FeatureImportance")
        permutation = _permutation_importance(model, test_x, test_y)
        for feature, native_value in zip(features, native, strict=True):
            perm_mean, perm_variance = permutation[feature]
            importance_rows.append(
                {
                    "fold_index": fold.fold_index,
                    "feature": feature,
                    "catboost_importance": float(native_value),
                    "permutation_importance_mean": perm_mean,
                    "permutation_importance_variance": perm_variance,
                    "permutation_repeats": 20,
                }
            )
        report["model_fitted"] = True
        fold_reports.append(report)

    prediction_frame = pd.DataFrame(predictions)
    if not prediction_frame.empty:
        if prediction_frame["candidate_id"].duplicated().any():
            raise ValueError("walk-forward protocol emitted duplicate OOS candidates")
        if prediction_frame["oos_row_id"].duplicated().any():
            raise ValueError("walk-forward protocol emitted duplicate OOS row IDs")
        if prediction_frame["comparison_row_id"].duplicated().any():
            raise ValueError("walk-forward protocol emitted duplicate comparison row IDs")
    return CatBoostBundleModelRun(
        protocol=protocol,
        predictions=prediction_frame,
        fold_reports=tuple(fold_reports),
        feature_importance=pd.DataFrame(importance_rows),
        fitted_categorical_indices=fitted_categorical_indices,
        label_artifact_id=label_id,
        fold_schedule_id=schedule_id,
        candidate_fold_set_id=fold_set,
    )
