"""
Pydantic configuration models for the ML extrema classification pipeline.

All pipeline parameters are centralized here. Each sub-config maps to a
specific pipeline stage (detection → labeling → features → training → eval).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtremaConfig(BaseModel):
    """Configuration for scipy peak-finding on tick prices."""

    window_size: int = Field(
        default=5000,
        ge=50,
        description="Sliding window size in ticks for peak detection",
    )
    min_peak_width: int = Field(
        default=200,
        ge=10,
        description="Minimum peak width in ticks",
    )
    max_peak_width: int = Field(
        default=5000,
        ge=50,
        description="Maximum peak width in ticks",
    )
    min_prominence_ticks: float = Field(
        default=10.0,
        gt=0,
        description="Minimum peak prominence in tick-size units (10 ticks = 2.5pt on NQ)",
    )
    dedup_window: int = Field(
        default=200,
        ge=1,
        description="Window for deduplicating overlapping detections",
    )


class LabelingConfig(BaseModel):
    """Configuration for rebound/crossing label assignment."""

    rebound_thresholds: list[int] = Field(
        default=[20, 40, 60],
        description="Tick thresholds for rebound labeling (20t=5pt, 40t=10pt, 60t=15pt on NQ)",
    )
    crossing_threshold: int = Field(
        default=20,
        ge=1,
        description="Tick threshold for crossing classification",
    )
    forward_window: int = Field(
        default=5000,
        ge=10,
        description="Max ticks to look forward for label assignment",
    )


class FeatureConfig(BaseModel):
    """Configuration for feature extraction."""

    # PL (Price Level) microstructure features
    pl_range_ticks: int = Field(
        default=10,
        ge=1,
        description="Number of tick levels around extremum for PL features",
    )

    # MS (Market Shift) momentum features
    ms_window: int = Field(
        default=237,
        ge=10,
        description="Lookback window in ticks for momentum features",
    )
    rsi_periods: list[int] = Field(
        default=[20, 40, 80, 120, 160, 200],
        description="RSI computation periods on tick prices",
    )

    # Signal detector features (hybrid approach)
    include_signal_features: bool = Field(
        default=True,
        description="Whether to include 20 detector outputs as features",
    )
    signal_bar_timeframe: str = Field(
        default="5m",
        description="Bar timeframe for aligning signal outputs to extrema",
    )


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward cross-validation."""

    train_days: int = Field(
        default=60,
        ge=5,
        description="Training window size in calendar days",
    )
    test_days: int = Field(
        default=20,
        ge=1,
        description="Test window size in calendar days",
    )
    gap_days: int = Field(
        default=1,
        ge=0,
        description="Gap between train and test to prevent leakage",
    )
    expanding: bool = Field(
        default=False,
        description="If True, training window expands; if False, rolls",
    )


class ModelConfig(BaseModel):
    """Configuration for CatBoost model training."""

    model_type: str = Field(
        default="catboost",
        description="Model type (currently only catboost supported)",
    )
    iterations: int = Field(
        default=1000,
        ge=10,
        description="Number of boosting iterations",
    )
    depth: int = Field(
        default=6,
        ge=1,
        le=16,
        description="Tree depth",
    )
    learning_rate: float = Field(
        default=0.03,
        gt=0,
        description="Learning rate",
    )
    auto_class_weights: str = Field(
        default="Balanced",
        description="Class weight balancing strategy",
    )
    rfecv_enabled: bool = Field(
        default=True,
        description="Whether to run RFECV feature selection",
    )
    rfecv_min_features: int = Field(
        default=5,
        ge=1,
        description="Minimum features to keep in RFECV",
    )
    early_stopping_rounds: int = Field(
        default=50,
        ge=5,
        description="Early stopping patience",
    )


class MLPipelineConfig(BaseModel):
    """Top-level configuration combining all pipeline stages."""

    extrema: ExtremaConfig = Field(default_factory=ExtremaConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)

    tick_size: float = Field(
        default=0.25,
        gt=0,
        description="Instrument tick size (NQ=0.25, ES=0.25)",
    )
    instrument: str = Field(
        default="NQ",
        description="Target instrument symbol",
    )
