"""Pinned single-thread CatBoost protocol for IFVG context folds."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from .context_experiment_contracts import (
    IFVG_CONTEXT_CALIBRATION_POLICY_ID,
    IFVG_CONTEXT_MODEL_PARAMETERS,
    IFVG_CONTEXT_MODEL_PROTOCOL_ID,
    ContextFeatureTier,
    IfvgContextFeatureIdentity,
    canonical_contract_sha256,
)
from .context_feature_view import CandidateFeatureView, features_for_tier
from .context_folds import ContextFoldSet

__all__ = [
    "MISSING_CATEGORY",
    "CATEGORICAL_FEATURE_REGISTRY",
    "ResolvedContextModelProtocol",
    "ContextModelRun",
    "categorical_features_for",
    "resolve_context_model_protocol",
    "run_context_fold_models",
]

MISSING_CATEGORY = "__MISSING__"

_CATEGORICAL_EXACT = frozenset(
    {
        "direction",
        "entry_family",
        "entry_session",
        "in_doc_session",
        "in_engine_session",
    }
)
_CATEGORICAL_SUFFIXES = (
    "_high_relationship",
    "_low_relationship",
    "_swing_sequence_state",
    "_structure_direction",
    "_last_break_type",
    "_last_break_direction",
    "_structure_alignment",
    "_break_alignment",
    "_missing_reason",
    "_pool_type",
)
CATEGORICAL_FEATURE_REGISTRY = {
    "exact": tuple(sorted(_CATEGORICAL_EXACT)),
    "suffixes": _CATEGORICAL_SUFFIXES,
    "missing_value": MISSING_CATEGORY,
}


@dataclass(frozen=True, slots=True)
class ResolvedContextModelProtocol:
    protocol_id: str
    parameters: dict[str, Any]
    categorical_registry: dict[str, Any]
    package_versions: dict[str, str]
    python_version: str
    ordered_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    feature_registry_hash: str
    manual_feature_overrides_hash: str
    resolved_hash: str


@dataclass(frozen=True, slots=True)
class ContextModelRun:
    protocol: ResolvedContextModelProtocol
    feature_identity: IfvgContextFeatureIdentity
    predictions: pd.DataFrame
    fold_reports: tuple[dict[str, Any], ...]
    feature_importance: pd.DataFrame


def categorical_features_for(features: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in features
        if feature in _CATEGORICAL_EXACT
        or any(feature.endswith(suffix) for suffix in _CATEGORICAL_SUFFIXES)
    )


def resolve_context_model_protocol(
    *,
    ordered_features: tuple[str, ...],
    feature_registry_hash: str,
    manual_feature_overrides: dict[str, Any] | None = None,
) -> ResolvedContextModelProtocol:
    overrides = dict(manual_feature_overrides or {})
    # Overrides are identity-bearing annotations only; the locked model and
    # feature registries cannot be changed through them.
    forbidden = set(overrides) & {
        "features",
        "categorical_features",
        "model_parameters",
        "threshold",
    }
    if forbidden:
        raise ValueError(f"manual override cannot alter locked protocol: {sorted(forbidden)}")
    categorical = categorical_features_for(ordered_features)
    versions = {
        package: version(package)
        for package in ("catboost", "numpy", "pandas", "scikit-learn")
    }
    override_hash = canonical_contract_sha256(overrides)
    payload = {
        "protocol_id": IFVG_CONTEXT_MODEL_PROTOCOL_ID,
        "parameters": dict(IFVG_CONTEXT_MODEL_PARAMETERS),
        "categorical_registry": CATEGORICAL_FEATURE_REGISTRY,
        "package_versions": versions,
        "python_version": platform.python_version(),
        "ordered_features": list(ordered_features),
        "categorical_features": list(categorical),
        "feature_registry_hash": feature_registry_hash,
        "manual_feature_overrides_hash": override_hash,
        "threshold_report_grid": [0.40, 0.50, 0.60, 0.70],
        "calibration_policy": IFVG_CONTEXT_CALIBRATION_POLICY_ID,
    }
    return ResolvedContextModelProtocol(
        protocol_id=IFVG_CONTEXT_MODEL_PROTOCOL_ID,
        parameters=dict(IFVG_CONTEXT_MODEL_PARAMETERS),
        categorical_registry=CATEGORICAL_FEATURE_REGISTRY,
        package_versions=versions,
        python_version=platform.python_version(),
        ordered_features=ordered_features,
        categorical_features=categorical,
        feature_registry_hash=feature_registry_hash,
        manual_feature_overrides_hash=override_hash,
        resolved_hash=canonical_contract_sha256(payload),
    )


def _model_frame(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    categoricals: tuple[str, ...],
) -> pd.DataFrame:
    missing = sorted(set(features) - set(frame))
    if missing:
        raise ValueError(f"model frame is missing registered features {missing}")
    result = frame.loc[:, features].copy()
    categorical_set = set(categoricals)
    for feature in features:
        if feature in categorical_set:
            result[feature] = result[feature].where(result[feature].notna(), MISSING_CATEGORY)
            result[feature] = result[feature].astype(str)
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
    baseline = log_loss(test_y, model.predict_proba(test_x)[:, 1], labels=[0, 1])
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {column: [] for column in test_x}
    for column in test_x:
        source = test_x[column].to_numpy(copy=True)
        for _ in range(repeats):
            permuted = test_x.copy()
            permuted[column] = source[rng.permutation(len(source))]
            score = log_loss(
                test_y,
                model.predict_proba(permuted)[:, 1],
                labels=[0, 1],
            )
            values[column].append(float(score - baseline))
    return {
        column: (float(np.mean(items)), float(np.var(items, ddof=0)))
        for column, items in values.items()
    }


def run_context_fold_models(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    tier: ContextFeatureTier,
    manual_feature_overrides: dict[str, Any] | None = None,
) -> ContextModelRun:
    features = features_for_tier(tier)
    protocol = resolve_context_model_protocol(
        ordered_features=features,
        feature_registry_hash=view.feature_registry_hash,
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
        train_x = _model_frame(
            train,
            features=features,
            categoricals=protocol.categorical_features,
        )
        test_x = _model_frame(
            test,
            features=features,
            categoricals=protocol.categorical_features,
        )
        train_y = pd.to_numeric(train["binary_target"], errors="raise").astype(int).to_numpy()
        test_y = pd.to_numeric(test["binary_target"], errors="raise").astype(int).to_numpy()
        model = CatBoostClassifier(**protocol.parameters)
        model.fit(
            train_x,
            train_y,
            cat_features=list(protocol.categorical_features),
        )
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
    importance = pd.DataFrame(importance_rows)
    feature_identity = IfvgContextFeatureIdentity(
        view_id=view.view_id,
        artifact_pair_hash=view.artifact_pair_hash,
        feature_registry_hash=view.feature_registry_hash,
        categorical_registry_hash=canonical_contract_sha256(
            protocol.categorical_registry
        ),
        feature_tier=tier,
        ordered_features=features,
        categorical_features=protocol.categorical_features,
        manual_feature_overrides_hash=protocol.manual_feature_overrides_hash,
    )
    return ContextModelRun(
        protocol=protocol,
        feature_identity=feature_identity,
        predictions=prediction_frame,
        fold_reports=tuple(fold_reports),
        feature_importance=importance,
    )
