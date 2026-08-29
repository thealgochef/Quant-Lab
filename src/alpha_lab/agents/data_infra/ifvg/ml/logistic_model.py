"""Model A — `ifvg_context_logistic_l2_v1` (`ML_REGIME_CONTRACT_PLAN.md` §2).

Mirrors the verified CatBoost lane (`context_model.py`) exactly where the
brief demands parity: the fold loop shape, the frozen preprocessing-as-
identity resolution, the prediction-row schema, and — deliberately — the
model-independent ``oos_row_id`` (canonical hash of
``{view_id, fold_index, candidate_id}``) so ladder rows pair one-to-one
across rungs. Every preprocessing step (median imputer with missing
indicator, standard scaler, one-hot encoder) is fitted on TRAINING-FOLD
rows only, inside one sklearn :class:`~sklearn.pipeline.Pipeline`.

Portable fitted artifacts (revision P1-4): :func:`persist_logistic_fit`
writes a manifest whose references are manifest-relative paths with
checksums; :func:`reload_logistic_fit` re-verifies them and never depends
on the original machine's absolute paths.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..context_experiment_contracts import (
    IFVG_CONTEXT_CALIBRATION_POLICY_ID,
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
    label_content_hash,
)
from .model_protocols import LOGISTIC_PROTOCOL_ID

__all__ = [
    "IFVG_CONTEXT_LOGISTIC_PARAMETERS",
    "LOGISTIC_PREPROCESSING_POLICY",
    "ResolvedLogisticProtocol",
    "LogisticModelRun",
    "resolve_logistic_protocol",
    "run_logistic_fold_models",
    "persist_logistic_fit",
    "reload_logistic_fit",
    "LOGISTIC_FIT_MANIFEST",
    "LOGISTIC_FIT_PARAMETERS",
    "LOGISTIC_FIT_PIPELINE",
]

#: Frozen, determinism-pinned logistic parameters (proposal-stamped in the
#: plan; a change is a new protocol, never an override).
IFVG_CONTEXT_LOGISTIC_PARAMETERS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 10_000,
        "tol": 1e-6,
        "fit_intercept": True,
        "class_weight": None,
        "random_state": 7,
        "n_jobs": 1,
    }
)

#: The fold-local preprocessing policy — part of the resolved identity.
LOGISTIC_PREPROCESSING_POLICY: MappingProxyType[str, Any] = MappingProxyType(
    {
        "numeric_imputer": {"strategy": "median", "add_indicator": True},
        "scaler": {"with_mean": True, "with_std": True},
        "categorical_encoder": {
            "kind": "one_hot",
            "handle_unknown": "ignore",
            "missing_value": MISSING_CATEGORY,
        },
        "categorical_registry": CATEGORICAL_FEATURE_REGISTRY,
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedLogisticProtocol:
    protocol_id: str
    parameters: dict[str, Any]
    preprocessing_policy: dict[str, Any]
    package_versions: dict[str, str]
    python_version: str
    ordered_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    feature_registry_hash: str
    manual_feature_overrides_hash: str
    resolved_hash: str


@dataclass(frozen=True, slots=True)
class LogisticModelRun:
    protocol: ResolvedLogisticProtocol
    predictions: pd.DataFrame
    fold_reports: tuple[dict[str, Any], ...]
    feature_importance: pd.DataFrame


def resolve_logistic_protocol(
    *,
    ordered_features: tuple[str, ...],
    feature_registry_hash: str,
    manual_feature_overrides: dict[str, Any] | None = None,
    extra_categorical_features: tuple[str, ...] = (),
) -> ResolvedLogisticProtocol:
    """``extra_categorical_features`` (R6.1 D8): block-declared categoricals
    (e.g. a fold-local hard regime id) one-hot encoded next to the frozen
    registry's — they enter the resolved identity only when present."""

    overrides = dict(manual_feature_overrides or {})
    # Same locked-protocol rule as the CatBoost lane: overrides are
    # identity-bearing annotations only.
    forbidden = set(overrides) & {
        "features",
        "categorical_features",
        "model_parameters",
        "threshold",
    }
    if forbidden:
        raise ValueError(f"manual override cannot alter locked protocol: {sorted(forbidden)}")
    outside = sorted(set(extra_categorical_features) - set(ordered_features))
    if outside:
        raise ValueError(f"categorical features outside the ordered features: {outside}")
    declared = set(categorical_features_for(ordered_features)) | set(extra_categorical_features)
    categorical = tuple(name for name in ordered_features if name in declared)
    versions = {
        package: version(package) for package in ("numpy", "pandas", "scikit-learn")
    }
    override_hash = canonical_contract_sha256(overrides)
    payload = {
        "protocol_id": LOGISTIC_PROTOCOL_ID,
        "parameters": dict(IFVG_CONTEXT_LOGISTIC_PARAMETERS),
        "preprocessing_policy": dict(LOGISTIC_PREPROCESSING_POLICY),
        "package_versions": versions,
        "python_version": platform.python_version(),
        "ordered_features": list(ordered_features),
        "categorical_features": list(categorical),
        "feature_registry_hash": feature_registry_hash,
        "manual_feature_overrides_hash": override_hash,
        "threshold_report_grid": [0.40, 0.50, 0.60, 0.70],
        "calibration_policy": IFVG_CONTEXT_CALIBRATION_POLICY_ID,
    }
    return ResolvedLogisticProtocol(
        protocol_id=LOGISTIC_PROTOCOL_ID,
        parameters=dict(IFVG_CONTEXT_LOGISTIC_PARAMETERS),
        preprocessing_policy=dict(LOGISTIC_PREPROCESSING_POLICY),
        package_versions=versions,
        python_version=platform.python_version(),
        ordered_features=ordered_features,
        categorical_features=categorical,
        feature_registry_hash=feature_registry_hash,
        manual_feature_overrides_hash=override_hash,
        resolved_hash=canonical_contract_sha256(payload),
    )


def _logistic_frame(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    categoricals: tuple[str, ...],
) -> pd.DataFrame:
    """Raw model frame: categoricals as typed strings, numerics as float.

    NaN survives in numeric columns — the fold-fitted imputer owns
    missingness (unlike the CatBoost lane, which delegates to the library).
    """

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


def _build_pipeline(protocol: ResolvedLogisticProtocol) -> Pipeline:
    numeric = [
        feature
        for feature in protocol.ordered_features
        if feature not in set(protocol.categorical_features)
    ]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                numeric,
            )
        )
    if protocol.categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(protocol.categorical_features),
            )
        )
    return Pipeline(
        [
            ("preprocess", ColumnTransformer(transformers, remainder="drop")),
            ("model", LogisticRegression(**protocol.parameters)),
        ]
    )


def _permutation_importance(
    pipeline: Pipeline,
    test_x: pd.DataFrame,
    test_y: np.ndarray,
    *,
    repeats: int = 20,
    seed: int = 7,
) -> dict[str, tuple[float, float]]:
    """Identical draw protocol to the CatBoost lane: one seed-7 RNG per fold,
    columns walked in frame order, raw (pre-transform) columns permuted."""

    baseline = log_loss(test_y, pipeline.predict_proba(test_x)[:, 1], labels=[0, 1])
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {column: [] for column in test_x}
    for column in test_x:
        source = test_x[column].to_numpy(copy=True)
        for _ in range(repeats):
            permuted = test_x.copy()
            permuted[column] = source[rng.permutation(len(source))]
            score = log_loss(
                test_y,
                pipeline.predict_proba(permuted)[:, 1],
                labels=[0, 1],
            )
            values[column].append(float(score - baseline))
    return {
        column: (float(np.mean(items)), float(np.var(items, ddof=0)))
        for column, items in values.items()
    }


def run_logistic_fold_models(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    features: tuple[str, ...],
    manual_feature_overrides: dict[str, Any] | None = None,
    fitted_fold_sink: dict[int, Pipeline] | None = None,
    fold_local_features: RegimeFoldFeatureSource | None = None,
    fold_schedule_id: str | None = None,
    label_artifact_id: str | None = None,
) -> LogisticModelRun:
    """Identical loop shape to ``run_context_fold_models`` (test-enforced).

    ``fitted_fold_sink`` optionally receives each valid fold's fitted
    pipeline (for portable-artifact persistence); it never changes results.
    R6.1: ``fold_local_features`` (fold-LOCAL regime features) are
    left-joined per fold on ``candidate_id`` before fitting — a candidate
    without a fold row keeps NaN values for the fold-fitted imputer +
    indicator; block-declared categoricals are one-hot encoded. Every
    prediction row carries the D13 ``comparison_row_id`` next to the legacy
    ``oos_row_id``; ``fold_schedule_id`` / ``label_artifact_id`` default to
    the schedule derived from the labeled trading days and the label content
    hash (the pipeline passes S08's / S07's exact ids).
    """

    features = tuple(str(name) for name in features)
    if fold_local_features is not None:
        assert_fold_local_source_coherent(
            fold_local_features,
            features=features,
            static_columns=tuple(str(column) for column in view.frame.columns),
        )
        extra_categorical = tuple(fold_local_features.categorical_features)
    else:
        extra_categorical = ()
    protocol = resolve_logistic_protocol(
        ordered_features=features,
        feature_registry_hash=view.feature_registry_hash,
        manual_feature_overrides=manual_feature_overrides,
        extra_categorical_features=extra_categorical,
    )
    schedule_id = fold_schedule_id or default_fold_schedule_id(folds, labeled_candidates)
    label_id = label_artifact_id or label_content_hash(labeled_candidates)
    fold_set = candidate_fold_set_id(folds)
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
        if fold_local_features is not None:
            train = join_fold_local_features(
                train, fold_local_features, fold_index=fold.fold_index
            )
            test = join_fold_local_features(
                test, fold_local_features, fold_index=fold.fold_index
            )
        train_x = _logistic_frame(
            train,
            features=features,
            categoricals=protocol.categorical_features,
        )
        test_x = _logistic_frame(
            test,
            features=features,
            categoricals=protocol.categorical_features,
        )
        train_y = pd.to_numeric(train["binary_target"], errors="raise").astype(int).to_numpy()
        test_y = pd.to_numeric(test["binary_target"], errors="raise").astype(int).to_numpy()
        pipeline = _build_pipeline(protocol)
        pipeline.fit(train_x, train_y)
        if fitted_fold_sink is not None:
            fitted_fold_sink[fold.fold_index] = pipeline
        probabilities = pipeline.predict_proba(test_x)[:, 1]
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
        permutation = _permutation_importance(pipeline, test_x, test_y)
        for feature in features:
            perm_mean, perm_variance = permutation[feature]
            importance_rows.append(
                {
                    "fold_index": fold.fold_index,
                    "feature": feature,
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
    return LogisticModelRun(
        protocol=protocol,
        predictions=prediction_frame,
        fold_reports=tuple(fold_reports),
        feature_importance=pd.DataFrame(importance_rows),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Portable fitted artifacts (revision P1-4)
# ─────────────────────────────────────────────────────────────────────────────

LOGISTIC_FIT_MANIFEST = "manifest.json"
LOGISTIC_FIT_PARAMETERS = "fitted_parameters.json"
LOGISTIC_FIT_PIPELINE = "pipeline.joblib"
_ARTIFACT_SCHEMA_VERSION = 1


def _fitted_parameter_payload(pipeline: Pipeline) -> dict[str, Any]:
    """Canonical JSON view of every fitted statistic in the pipeline."""

    preprocess: ColumnTransformer = pipeline.named_steps["preprocess"]
    model: LogisticRegression = pipeline.named_steps["model"]
    payload: dict[str, Any] = {
        "coefficients": np.asarray(model.coef_).tolist(),
        "intercept": np.asarray(model.intercept_).tolist(),
        "classes": np.asarray(model.classes_).tolist(),
    }
    for name, transformer, columns in preprocess.transformers_:
        if name == "numeric":
            imputer: SimpleImputer = transformer.named_steps["imputer"]
            scaler: StandardScaler = transformer.named_steps["scaler"]
            payload["numeric"] = {
                "columns": list(columns),
                "imputer_statistics": np.asarray(imputer.statistics_).tolist(),
                "imputer_indicator_features": (
                    np.asarray(imputer.indicator_.features_).tolist()
                    if imputer.indicator_ is not None
                    else []
                ),
                "scaler_mean": np.asarray(scaler.mean_).tolist(),
                "scaler_scale": np.asarray(scaler.scale_).tolist(),
            }
        elif name == "categorical":
            payload["categorical"] = {
                "columns": list(columns),
                "categories": [
                    np.asarray(categories).tolist()
                    for categories in transformer.categories_
                ],
            }
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def persist_logistic_fit(
    directory: Path,
    *,
    pipeline: Pipeline,
    protocol: ResolvedLogisticProtocol,
    fold_index: int,
) -> dict[str, Any]:
    """Write the dual-format portable artifact and return its manifest.

    Every reference in the manifest is a **manifest-relative filename** with
    a checksum and byte size; nothing records an absolute path, so the
    artifact directory can be relocated wholesale (P1-4).
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fitted_payload = _fitted_parameter_payload(pipeline)
    fitted_hash = canonical_contract_sha256(fitted_payload)
    parameters_path = directory / LOGISTIC_FIT_PARAMETERS
    parameters_path.write_text(
        json.dumps(fitted_payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    pipeline_path = directory / LOGISTIC_FIT_PIPELINE
    joblib.dump(pipeline, pipeline_path)
    manifest = {
        "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "protocol_id": protocol.protocol_id,
        "protocol_resolved_hash": protocol.resolved_hash,
        "fold_index": fold_index,
        "fitted_parameter_payload_hash": fitted_hash,
        "software_versions": dict(protocol.package_versions),
        "references": {
            LOGISTIC_FIT_PARAMETERS: {
                "file_sha256": _file_sha256(parameters_path),
                "byte_size": parameters_path.stat().st_size,
            },
            LOGISTIC_FIT_PIPELINE: {
                "file_sha256": _file_sha256(pipeline_path),
                "byte_size": pipeline_path.stat().st_size,
            },
        },
    }
    (directory / LOGISTIC_FIT_MANIFEST).write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def reload_logistic_fit(directory: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Checksum-verified reload resolved against the artifact directory root.

    Refuses missing files, byte drift, and fitted-parameter hash mismatches;
    never consults any path recorded outside the directory.
    """

    directory = Path(directory)
    manifest = json.loads((directory / LOGISTIC_FIT_MANIFEST).read_text(encoding="utf-8"))
    for name, reference in manifest["references"].items():
        target = directory / name
        if not target.exists():
            raise FileNotFoundError(f"portable artifact reference missing: {name}")
        actual = _file_sha256(target)
        if actual != reference["file_sha256"]:
            raise ValueError(f"portable artifact reference {name} failed its checksum")
        if target.stat().st_size != reference["byte_size"]:
            raise ValueError(f"portable artifact reference {name} changed size")
    fitted_payload = json.loads(
        (directory / LOGISTIC_FIT_PARAMETERS).read_text(encoding="utf-8")
    )
    if canonical_contract_sha256(fitted_payload) != manifest["fitted_parameter_payload_hash"]:
        raise ValueError("fitted parameter payload does not hash to the manifest value")
    pipeline = joblib.load(directory / LOGISTIC_FIT_PIPELINE)
    return pipeline, manifest
