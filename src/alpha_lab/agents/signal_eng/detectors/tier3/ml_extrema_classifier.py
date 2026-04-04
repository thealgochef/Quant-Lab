"""
Category 21: ML Extrema Classifier.

Wraps a trained CatBoost model that predicts rebound/crossing at
detected price extrema. Emits signals when the model predicts
a rebound with sufficient confidence.

Signal composition:
- direction: +1 (bullish rebound at trough), -1 (bearish rebound at peak), 0 (none)
- strength: model predicted probability for rebound class
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.agents.signal_eng.indicators import compute_atr
from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier, Timeframe

logger = logging.getLogger(__name__)

_MIN_BARS = 50


class MLExtremaClassifierDetector(SignalDetector):
    """ML Extrema Classifier: CatBoost rebound/crossing prediction.

    **RESEARCH / EXPERIMENTAL** — This detector approximates tick-level
    training features from bar-level OHLCV data.  Missing features are
    filled with 0.0 and some values (e.g. ``pl_width``) are placeholders.
    The resulting predictions are NOT execution-faithful and should not be
    treated as production-grade signals.  For production-grade level-touch
    predictions, use the retained 3-feature dashboard model path.

    Requires a trained model loaded from disk. If no model is available,
    validate_inputs returns False and the detector is silently skipped.
    """

    _EXPERIMENTAL = True

    detector_id = "ml_extrema_classifier"
    category = "ml_extrema_classifier"
    tier = SignalTier.COMPOSITE
    timeframes = [tf.value for tf in [Timeframe.M5, Timeframe.M15]]

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_confidence: float = 0.6,
    ) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._model = None
        self._selected_features: list[str] = []
        self._loaded = False

        if model_path is not None:
            self._try_load_model()
            if self._loaded:
                import warnings
                warnings.warn(
                    "MLExtremaClassifierDetector uses bar-level feature "
                    "approximations that differ from the tick-level features "
                    "the model was trained on. Predictions are experimental "
                    "and not execution-faithful.",
                    stacklevel=2,
                )

    def _try_load_model(self) -> None:
        """Attempt to load a trained model from disk."""
        try:
            from alpha_lab.agents.data_infra.ml.model_trainer import (
                ExtremaModelTrainer,
            )
            trained = ExtremaModelTrainer.load_model(self.model_path)
            self._model = trained.model
            self._selected_features = trained.selected_features
            self._loaded = True
            logger.info(
                "ML model loaded from %s (%d features)",
                self.model_path, len(self._selected_features),
            )
        except Exception as exc:
            logger.warning("Failed to load ML model from %s: %s", self.model_path, exc)
            self._loaded = False

    def validate_inputs(self, data: DataBundle) -> bool:
        if not self._loaded:
            return False
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if isinstance(df, pd.DataFrame) and len(df) > _MIN_BARS:
                return True
        return False

    def compute(self, data: DataBundle) -> list[SignalVector]:
        signals: list[SignalVector] = []
        if not self._loaded:
            return signals
        for tf in self.timeframes:
            df = data.bars.get(tf)
            if not isinstance(df, pd.DataFrame) or len(df) <= _MIN_BARS:
                continue
            sv = self._compute_timeframe(df, tf, data.instrument)
            if sv is not None:
                signals.append(sv)
        return signals

    def _compute_timeframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        instrument: str,
    ) -> SignalVector | None:
        """Generate signals from bar-level features using the trained model.

        This is a simplified approach that computes features from bars
        rather than ticks (since ticks may not be available in the
        standard DataBundle pipeline). For full tick-level prediction,
        use the ExtremaDatasetBuilder directly.
        """
        high = df["high"]
        low = df["low"]

        atr = compute_atr(df)
        atr_safe = atr.replace(0, np.nan).ffill().fillna(1.0)

        direction = pd.Series(0, index=df.index, dtype=int)
        strength = pd.Series(0.0, index=df.index)
        formation_idx = pd.Series(0, index=df.index, dtype=int)

        # Detect bar-level extrema using simple lookback
        lookback = 10
        for i in range(lookback, len(df) - 1):
            window_high = high.iloc[i - lookback: i + 1]
            window_low = low.iloc[i - lookback: i + 1]

            is_peak = high.iloc[i] == window_high.max()
            is_trough = low.iloc[i] == window_low.min()

            if not (is_peak or is_trough):
                continue

            # Build minimal feature vector from available bar data
            feat_dict = self._build_bar_features(df, i, atr_safe)

            # Check if we have enough features for the model
            missing = set(self._selected_features) - set(feat_dict.keys())
            if missing:
                # Fill missing features with 0.0
                for m in missing:
                    feat_dict[m] = 0.0

            feat_row = pd.DataFrame([feat_dict])[self._selected_features]

            try:
                prob = self._model.predict_proba(feat_row)[0]
                rebound_prob = float(prob[1]) if len(prob) > 1 else float(prob[0])
            except Exception:
                continue

            if rebound_prob < self.min_confidence:
                continue

            if is_trough:
                direction.iloc[i] = 1  # Bullish rebound
            elif is_peak:
                direction.iloc[i] = -1  # Bearish rebound
            strength.iloc[i] = round(rebound_prob, 6)
            formation_idx.iloc[i] = i

        # Forward-fill with decay
        fi = formation_idx.replace(0, np.nan)
        formation_idx = fi.ffill().fillna(0).astype(int)

        dir_filled = direction.replace(0, np.nan).ffill().fillna(0).astype(int)
        str_filled = strength.replace(0.0, np.nan).ffill().fillna(0.0)
        has_signal = formation_idx > 0
        direction = dir_filled.where(has_signal, 0)
        strength = str_filled.where(has_signal, 0.0).clip(0.0, 1.0)

        # Decay: halve every 10 bars
        bars_since = pd.Series(np.arange(len(df)), index=df.index) - formation_idx
        decay = np.power(0.5, bars_since.clip(lower=0) / 10.0)
        strength = (strength * decay).clip(0.0, 1.0)
        strength = strength.where(direction != 0, 0.0)

        return SignalVector(
            signal_id=f"SIG_ML_EXTREMA_CLASSIFIER_{timeframe}_v1",
            category=self.category,
            timeframe=timeframe,
            version=1,
            direction=direction,
            strength=strength,
            formation_idx=formation_idx,
            parameters={
                "min_confidence": self.min_confidence,
                "model_path": str(self.model_path) if self.model_path else None,
            },
            metadata={
                "instrument": instrument,
                "intuition": "ML-predicted rebound at price extrema",
                "bars_processed": len(df),
                "n_features": len(self._selected_features),
            },
        )

    @staticmethod
    def _build_bar_features(
        df: pd.DataFrame, idx: int, atr_safe: pd.Series,
    ) -> dict[str, float]:
        """Build a feature dict from bar-level data at a given index.

        This constructs approximate equivalents of tick-level features
        using OHLCV data, allowing the detector to run in the standard
        pipeline without requiring direct tick access.
        """
        features: dict[str, float] = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1.0, index=df.index)

        # Price-based features
        a = float(atr_safe.iloc[idx])
        if a > 0:
            features["pl_prominence"] = float(
                (high.iloc[idx] - low.iloc[idx]) / a
            )
        features["pl_width"] = 50.0  # Placeholder

        # Volume features
        vol_avg = float(volume.iloc[max(0, idx - 20): idx + 1].mean())
        if vol_avg > 0:
            features["ms_volume_momentum"] = float(volume.iloc[idx]) / vol_avg

        # Momentum features
        if idx >= 20:
            ret_20 = (close.iloc[idx] - close.iloc[idx - 20]) / close.iloc[idx - 20]
            features["ms_price_velocity_20"] = float(ret_20)

        if idx >= 10:
            ret_10 = (close.iloc[idx] - close.iloc[idx - 10]) / close.iloc[idx - 10]
            features["ms_price_velocity_10"] = float(ret_10)

        # Body ratio
        bar_range = high.iloc[idx] - low.iloc[idx]
        if bar_range > 0:
            features["pl_extremum_type"] = (
                1.0 if close.iloc[idx] < df["open"].iloc[idx] else 0.0
            )
            features["ms_volatility"] = float(
                close.iloc[max(0, idx - 20): idx + 1].pct_change().std()
            )

        return features
