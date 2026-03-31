"""
Prediction Engine — CatBoost inference on observation window features.

Receives completed ObservationWindow objects from the observation manager,
extracts the 3 features, runs model.predict() and model.predict_proba(),
and produces a Prediction object.

Predictions are generated for ALL sessions (Asia, London, Pre-market, NY RTH).
Only reversal predictions during NY RTH are flagged as executable for paper
trading (Phase 4).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from alpha_lab.dashboard.engine.models import ObservationStatus, ObservationWindow
from alpha_lab.dashboard.model import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    Prediction,
)
from alpha_lab.dashboard.model.model_manager import ModelManager


class PredictionEngine:
    """Runs CatBoost inference on observation window features."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._mm = model_manager
        self._callbacks: list[Callable[[Prediction], None]] = []

    def predict(self, observation: ObservationWindow) -> Prediction | None:
        """Run inference on a completed observation.

        Returns None if no model is loaded or observation is not completed.
        """
        model = self._mm.model
        if model is None:
            return None

        if observation.status != ObservationStatus.COMPLETED:
            return None
        if observation.features is None:
            return None

        # Extract feature vector in canonical order
        features = np.array([[
            observation.features[col] for col in FEATURE_COLUMNS
        ]])

        # Run inference
        predicted_idx = int(model.predict(features).flat[0])
        proba = model.predict_proba(features)[0]

        predicted_class = CLASS_NAMES[predicted_idx]
        probabilities = {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)}

        # Execution eligibility: only reversals during NY RTH
        is_executable = (
            predicted_class == "tradeable_reversal"
            and observation.event.session == "ny_rth"
        )

        active_version = self._mm.get_active_version()
        model_version = active_version["version"] if active_version else "unknown"

        prediction = Prediction(
            event_id=observation.event.event_id,
            timestamp=datetime.now(UTC),
            observation=observation,
            predicted_class=predicted_class,
            probabilities=probabilities,
            features=dict(observation.features),
            is_executable=is_executable,
            trade_direction=observation.event.trade_direction,
            level_price=observation.event.level_zone.representative_price,
            model_version=model_version,
        )

        for cb in self._callbacks:
            cb(prediction)

        return prediction

    def on_prediction(self, callback: Callable[[Prediction], None]) -> None:
        """Register callback for new predictions."""
        self._callbacks.append(callback)
