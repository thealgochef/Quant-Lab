"""
CatBoost model training with optional RFECV feature selection.

Trains a CatBoost classifier for extrema rebound/crossing prediction.
Walk-forward CV splits are used only for feature selection; the final
runtime model is fit once on the full labeled dataset.
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
        model = self._build_classifier(enable_early_stopping=False)
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
            train_metrics={
                "train_accuracy": train_accuracy,
                "n_train_samples": float(len(x_train)),
                "n_selected_features": float(len(selected_features)),
            },
            config=cfg,
        )

    def _run_rfecv(
        self,
        features: pd.DataFrame,
        y: pd.Series,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[str]:
        """Run RFECV feature selection."""
        from sklearn.feature_selection import RFECV

        cfg = self._config
        estimator = self._build_classifier(
            iterations=min(cfg.iterations, 200),
            enable_early_stopping=False,
        )

        # Use weighted precision for multiclass compatibility
        scoring = "precision_weighted" if cfg.loss_function == "MultiClass" else "precision"

        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv_splits,
            scoring=scoring,
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

    def _build_classifier(
        self,
        *,
        iterations: int | None = None,
        enable_early_stopping: bool = False,
    ):
        """Construct a CatBoost classifier with repo-safe defaults."""
        from catboost import CatBoostClassifier

        cfg = self._config
        model_kwargs = {
            "iterations": iterations or cfg.iterations,
            "depth": cfg.depth,
            "learning_rate": cfg.learning_rate,
            "loss_function": cfg.loss_function,
            "auto_class_weights": cfg.auto_class_weights,
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
        }
        if enable_early_stopping and cfg.early_stopping_rounds is not None:
            model_kwargs["early_stopping_rounds"] = cfg.early_stopping_rounds
        return CatBoostClassifier(**model_kwargs)

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
