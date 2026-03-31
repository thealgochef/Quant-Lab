"""
CatBoost model training with optional RFECV feature selection.

Trains a CatBoost classifier for extrema rebound/crossing prediction.
Supports walk-forward CV splits and automatic feature selection via
sklearn's RFECV.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ml.config import ModelConfig

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = ModelConfig()


@dataclass
class TrainedModel:
    """Container for a trained model with metadata."""

    model: object  # CatBoostClassifier
    selected_features: list[str]
    feature_importances: dict[str, float]
    train_metrics: dict[str, float] = field(default_factory=dict)
    config: ModelConfig = field(default_factory=ModelConfig)


class ExtremaModelTrainer:
    """Trains CatBoost models for extrema classification.

    Args:
        config: Model training configuration.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self._config = config or _DEFAULT_CONFIG

    def train(
        self,
        features: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> TrainedModel:
        """Train a CatBoost classifier.

        Args:
            features: Feature matrix.
            y: Binary labels (0=crossing, 1=rebound).
            cv_splits: Optional walk-forward CV splits for RFECV.

        Returns:
            TrainedModel with trained classifier and metadata.
        """
        from catboost import CatBoostClassifier

        cfg = self._config
        feature_names = list(features.columns)

        # Optional RFECV feature selection
        selected_features = feature_names
        if cfg.rfecv_enabled and cv_splits and len(cv_splits) >= 2:
            selected_features = self._run_rfecv(features, y, cv_splits)
            if len(selected_features) < cfg.rfecv_min_features:
                selected_features = feature_names
                logger.warning(
                    "RFECV selected too few features (%d), using all %d",
                    len(selected_features), len(feature_names),
                )

        x_train = features[selected_features]

        model = CatBoostClassifier(
            iterations=cfg.iterations,
            depth=cfg.depth,
            learning_rate=cfg.learning_rate,
            auto_class_weights=cfg.auto_class_weights,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )

        # If we have CV splits, use the last split for eval set
        if cv_splits and len(cv_splits) >= 1:
            last_train_idx, last_test_idx = cv_splits[-1]
            # Use all but last split for training, last for eval
            all_train_idx = np.concatenate(
                [ti for ti, _ in cv_splits[:-1]]
            ) if len(cv_splits) > 1 else last_train_idx

            x_fit = x_train.iloc[all_train_idx]
            y_fit = y.iloc[all_train_idx]
            x_eval = x_train.iloc[last_test_idx]
            y_eval = y.iloc[last_test_idx]

            model.fit(
                x_fit, y_fit,
                eval_set=(x_eval, y_eval),
                verbose=0,
            )
        else:
            model.fit(x_train, y, verbose=0)

        # Feature importances
        importances = model.get_feature_importance()
        importance_dict = dict(
            zip(selected_features, importances, strict=True)
        )

        # Train metrics
        train_preds = model.predict(x_train)
        train_accuracy = float(np.mean(train_preds.flatten() == y.values))

        return TrainedModel(
            model=model,
            selected_features=selected_features,
            feature_importances=importance_dict,
            train_metrics={"train_accuracy": train_accuracy},
            config=cfg,
        )

    def _run_rfecv(
        self,
        features: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[str]:
        """Run RFECV feature selection."""
        from catboost import CatBoostClassifier
        from sklearn.feature_selection import RFECV

        cfg = self._config
        estimator = CatBoostClassifier(
            iterations=min(cfg.iterations, 200),  # Faster for feature selection
            depth=cfg.depth,
            learning_rate=cfg.learning_rate,
            auto_class_weights=cfg.auto_class_weights,
            random_seed=42,
            verbose=0,
        )

        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_splits,
            scoring="precision",
            min_features_to_select=cfg.rfecv_min_features,
            n_jobs=1,
        )

        rfecv.fit(features, y)
        mask = rfecv.support_
        selected = [
            col for col, keep in zip(features.columns, mask, strict=True)
            if keep
        ]

        logger.info(
            "RFECV selected %d/%d features",
            len(selected), len(features.columns),
        )
        return selected

    @staticmethod
    def save_model(trained: TrainedModel, path: str | Path) -> None:
        """Save a trained model to disk.

        Args:
            trained: TrainedModel to save.
            path: Output directory path.
        """
        import json

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        trained.model.save_model(str(out / "model.cbm"))

        metadata = {
            "selected_features": trained.selected_features,
            "feature_importances": trained.feature_importances,
            "train_metrics": trained.train_metrics,
            "config": trained.config.model_dump(),
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Model saved to %s", out)

    @staticmethod
    def load_model(path: str | Path) -> TrainedModel:
        """Load a trained model from disk.

        Args:
            path: Directory containing model.cbm and metadata.json.

        Returns:
            TrainedModel instance.
        """
        import json

        from catboost import CatBoostClassifier

        model_dir = Path(path)
        model = CatBoostClassifier()
        model.load_model(str(model_dir / "model.cbm"))

        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        return TrainedModel(
            model=model,
            selected_features=metadata["selected_features"],
            feature_importances=metadata["feature_importances"],
            train_metrics=metadata.get("train_metrics", {}),
            config=ModelConfig(**metadata.get("config", {})),
        )
