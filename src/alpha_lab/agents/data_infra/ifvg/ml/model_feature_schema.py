"""Auditable fitted schemas, independent of source-projection completeness.

An all-null training column is a fold-local observation, never proof that its
source mapping is absent or valid. Source/structural-null validation belongs to
the projection contract. These checks preserve the frozen numerical protocols:
logistic may drop training-empty numeric values and keep their indicators;
CatBoost retains every raw numeric column with native NaNs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..context_model import MISSING_CATEGORY

FEATURE_SCHEMA_VERSION = "supervised_model_feature_schema_v1"


def assert_prediction_columns(frame: pd.DataFrame, features: tuple[str, ...]) -> None:
    """Refuse extra, absent, repeated or reordered persisted model inputs."""
    if frame.columns.duplicated().any() or tuple(frame.columns) != tuple(features):
        raise ValueError("prediction input schema disagrees with protocol ordered features")


def training_missingness(frame: pd.DataFrame, categoricals: tuple[str, ...]) -> dict:
    """Record missingness on training rows only, without changing values."""
    categorical = set(categoricals)
    return {
        "numeric_training_empty_features": [
            name for name in frame if name not in categorical and frame[name].isna().all()
        ],
        "categorical_training_empty_features": [
            name for name in frame if name in categorical and frame[name].eq(MISSING_CATEGORY).all()
        ],
    }


def fitted_feature_schema(model, protocol) -> dict[str, Any]:
    """Validate fitted names/types/dimensions against the resolved raw schema."""
    features = tuple(protocol.ordered_features)
    categorical = tuple(protocol.categorical_features)
    if not features or len(set(features)) != len(features):
        raise ValueError("model protocol ordered features must be nonempty and unique")
    if categorical != tuple(name for name in features if name in set(categorical)):
        raise ValueError("model protocol categorical schema disagrees with ordered features")
    proof: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "raw_ordered_features": list(features),
        "categorical_features": list(categorical),
    }
    if hasattr(model, "named_steps"):
        preprocess = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        if tuple(preprocess.feature_names_in_) != features:
            raise ValueError("fitted logistic raw feature schema disagrees with protocol")
        numeric = tuple(name for name in features if name not in set(categorical))
        transformers = {
            name: (transformer, tuple(columns))
            for name, transformer, columns in preprocess.transformers_
            if name != "remainder"
        }
        expected = {
            name
            for name, columns in (("numeric", numeric), ("categorical", categorical))
            if columns
        }
        if set(transformers) != expected:
            raise ValueError("fitted logistic transformer schema disagrees with protocol")
        empty = []
        if numeric:
            transformer, columns = transformers["numeric"]
            if columns != numeric:
                raise ValueError("fitted logistic numeric schema disagrees with protocol")
            imputer = transformer.named_steps["imputer"]
            empty = [
                name
                for name, statistic in zip(numeric, imputer.statistics_, strict=True)
                if np.isnan(statistic)
            ]
        if categorical and transformers["categorical"][1] != categorical:
            raise ValueError("fitted logistic categorical schema disagrees with protocol")
        transformed = list(preprocess.get_feature_names_out())
        if (
            len(transformed) != estimator.n_features_in_
            or len(transformed) != estimator.coef_.shape[1]
            or len(set(transformed)) != len(transformed)
        ):
            raise ValueError("fitted logistic transformed schema disagrees with coefficients")
        proof.update(
            {
                "model_kind": "logistic",
                "transformed_feature_names": transformed,
                "fitted_dimension": len(transformed),
                "numeric_training_empty_features": empty,
                "numeric_value_features_dropped_by_training_imputer": empty,
            }
        )
    else:
        if tuple(model.feature_names_) != features:
            raise ValueError("fitted CatBoost feature schema disagrees with protocol")
        expected_indices = tuple(i for i, name in enumerate(features) if name in categorical)
        if tuple(model.get_cat_feature_indices()) != expected_indices:
            raise ValueError("fitted CatBoost categorical schema disagrees with protocol")
        proof.update(
            {
                "model_kind": "catboost",
                "transformed_feature_names": list(features),
                "fitted_dimension": len(features),
                "categorical_indices": list(expected_indices),
                "numeric_value_features_dropped_by_training_imputer": [],
            }
        )
    return proof
