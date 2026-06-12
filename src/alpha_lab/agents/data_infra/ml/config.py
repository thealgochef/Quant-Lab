"""
Pydantic configuration models for the ML extrema classification pipeline.

All pipeline parameters are centralized here. Each sub-config maps to a
specific pipeline stage (detection → labeling → features → training → eval).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# ── Live-Computable Feature Sets ─────────────────────────────────
# These features can be computed from MBP-1 (top-of-book) + trades,
# which is what Databento live and Rithmic provide. Features requiring
# deeper book levels (MBP-10) or cancel events are excluded.

LIVE_APPROACH_FEATURES = [
    "app_large_trade_vol_pct",  # large trades (size>=10) / total volume
    "app_trade_count",  # number of trades in approach window
    "app_volume_acceleration",  # late sub-window rate / early rate
    "app_avg_trade_size",  # mean trade size
    "app_avg_tob_imbalance",  # avg bid_sz / (bid_sz + ask_sz) at L0
    "app_max_spread",  # max(ask - bid) at L0
    "app_volatility_recent",  # std of 1-min returns in last sub-window
    "app_volatility_ratio",  # recent volatility / full-window volatility
]

LIVE_INTERACTION_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
]

# All features available for live-aligned training and runtime
LIVE_ALL_FEATURES = LIVE_INTERACTION_FEATURES + LIVE_APPROACH_FEATURES

ALLOWED_STRATEGY_SESSIONS = ("asia", "london", "ny")


def _normalize_session_list(value) -> list[str]:
    """Normalize a user-provided session list with validation/deduplication."""
    if value is None:
        return []
    raw_values = [value] if isinstance(value, str) else list(value)
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        session = str(raw).strip().lower()
        if not session:
            continue
        if session not in ALLOWED_STRATEGY_SESSIONS:
            allowed = ", ".join(ALLOWED_STRATEGY_SESSIONS)
            msg = f"Unsupported session {session!r}. Allowed sessions: {allowed}"
            raise ValueError(msg)
        if session not in seen:
            normalized.append(session)
            seen.add(session)
    return normalized


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
    loss_function: str = Field(
        default="Logloss",
        description="CatBoost loss function (Logloss for binary, MultiClass for 3-class)",
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


class SessionExperimentConfig(BaseModel):
    """Configuration for isolating Strategy-Core sessions in experiments.

    These fields intentionally live outside ``DashboardUtilityConfig`` because
    they do not change per-date feature generation. Changing session scope should
    not invalidate expensive ``ml_utility_*`` caches; it only changes training,
    OOS reporting, and research gate calculations.
    """

    training_sessions: list[str] = Field(
        default_factory=lambda: list(ALLOWED_STRATEGY_SESSIONS),
        description="Strategy-Core sessions whose rows may train/refit the model",
    )
    evaluation_sessions: list[str] = Field(
        default_factory=lambda: list(ALLOWED_STRATEGY_SESSIONS),
        description="Strategy-Core sessions whose OOS rows are included in reported stats",
    )
    production_gate_sessions: list[str] = Field(
        default_factory=lambda: ["ny"],
        description="Strategy-Core sessions eligible for the confidence-gated research report",
    )
    report_session_breakdowns: bool = Field(
        default=True,
        description="Report per-session OOS threshold metrics alongside aggregate metrics",
    )

    @field_validator(
        "training_sessions",
        "evaluation_sessions",
        "production_gate_sessions",
        mode="before",
    )
    @classmethod
    def _normalize_sessions(cls, value):
        sessions = _normalize_session_list(value)
        if not sessions:
            msg = "At least one session must be enabled"
            raise ValueError(msg)
        return sessions


SESSION_EXPERIMENT_PRESETS: dict[str, dict] = {
    "all_to_ny": {
        "training_sessions": ["asia", "london", "ny"],
        "evaluation_sessions": ["asia", "london", "ny"],
        "production_gate_sessions": ["ny"],
        "report_session_breakdowns": True,
    },
    "ny_only": {
        "training_sessions": ["ny"],
        "evaluation_sessions": ["ny"],
        "production_gate_sessions": ["ny"],
        "report_session_breakdowns": True,
    },
    "asia_only": {
        "training_sessions": ["asia"],
        "evaluation_sessions": ["asia"],
        "production_gate_sessions": ["asia"],
        "report_session_breakdowns": True,
    },
    "london_only": {
        "training_sessions": ["london"],
        "evaluation_sessions": ["london"],
        "production_gate_sessions": ["london"],
        "report_session_breakdowns": True,
    },
    "asia_london_only": {
        "training_sessions": ["asia", "london"],
        "evaluation_sessions": ["asia", "london"],
        "production_gate_sessions": ["asia", "london"],
        "report_session_breakdowns": True,
    },
    "all_sessions_all_gates": {
        "training_sessions": ["asia", "london", "ny"],
        "evaluation_sessions": ["asia", "london", "ny"],
        "production_gate_sessions": ["asia", "london", "ny"],
        "report_session_breakdowns": True,
    },
}


def session_experiment_from_preset(name: str) -> SessionExperimentConfig:
    """Return a typed session-experiment config for a named preset."""
    try:
        payload = SESSION_EXPERIMENT_PRESETS[str(name)]
    except KeyError as exc:
        known = ", ".join(sorted(SESSION_EXPERIMENT_PRESETS))
        msg = f"Unknown session experiment preset {name!r}. Known presets: {known}"
        raise ValueError(msg) from exc
    return SessionExperimentConfig(**payload)


class DashboardUtilityConfig(BaseModel):
    """Configuration for the dashboard-utility training mode."""

    tp_points: float = Field(
        default=15.0,
        gt=0,
        description="Take-profit threshold in NQ points for tradeable_reversal",
    )
    sl_points: float = Field(
        default=30.0,
        gt=0,
        description="Stop-loss threshold in NQ points (MAE for trap/blowthrough)",
    )
    trap_mfe_min: float = Field(
        default=5.0,
        ge=0,
        description="Minimum MFE to distinguish trap_reversal from aggressive_blowthrough",
    )
    interaction_window_minutes: int = Field(
        default=5,
        ge=1,
        description="Minutes after touch for feature computation window",
    )
    level_proximity_pts: float = Field(
        default=0.50,
        gt=0,
        description="Points proximity for absorption ratio at-level volume",
    )
    bar_type: str = Field(
        default="147t",
        description="Bar type for touch detection and MFE/MAE: '147t', '987t', '2000t', or '1m'",
    )
    include_approach_features: bool = Field(
        default=False,
        description=(
            "Include 27 approach-window order flow features alongside 3 interaction features"
        ),
    )
    approach_window_minutes: int = Field(
        default=90,
        ge=15,
        description="Approach window duration in minutes before the touch event",
    )


class MLPipelineConfig(BaseModel):
    """Top-level configuration combining all pipeline stages."""

    training_mode: str = Field(
        default="extrema_rebound_crossing",
        description="Training mode: 'extrema_rebound_crossing' or 'dashboard_utility'",
    )

    extrema: ExtremaConfig = Field(default_factory=ExtremaConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    dashboard_utility: DashboardUtilityConfig = Field(
        default_factory=DashboardUtilityConfig,
    )
    session_experiment: SessionExperimentConfig = Field(
        default_factory=SessionExperimentConfig,
        description=(
            "Research-only session scope for training/evaluation/gate experiments; "
            "does not affect cached feature generation"
        ),
    )

    tick_size: float = Field(
        default=0.25,
        gt=0,
        description="Instrument tick size (NQ=0.25, ES=0.25)",
    )
    instrument: str = Field(
        default="NQ",
        description="Target instrument symbol",
    )

    def dataset_config_hash(self) -> str:
        """Short hex hash of config fields that affect dataset generation.

        Covers extrema, labeling, features, and tick_size.  Walk-forward
        and model configs do NOT affect the cached feature matrix.

        KEYED ON THE BAR PRICE SOURCE (trade-bar cutover, Part 1): the dashboard
        bars + the 3 interaction features changed from book-mid (0.125 grid) to
        TRADE PRINTS (0.25 grid), AND the canonical outcome re-anchored to the
        decision-time entry. Folding ``strategy_core.constants.BAR_PRICE_SOURCE``
        and the engine version into the payload changes ``cache_tag`` so the stale
        book-mid ``ml_utility_*`` caches are INVALIDATED (a fresh tag is read/written)
        rather than silently reused under the new bar/label definition.
        """
        import hashlib
        import json

        # Single-source the bar/label cutover key off the engine so the cache tag
        # tracks the platform version (the engine axis: v1 book-mid level-entry ->
        # v2 trade-price decision-entry) without restating a literal here. E1: the
        # hash INPUT changes with the axis rename, so dataset cache tags roll.
        from strategy_core import PLATFORM_VERSION
        from strategy_core.constants import BAR_PRICE_SOURCE, LABEL_ENTRY_REFERENCE

        payload = (
            f"mode={self.training_mode}|"
            + json.dumps(self.extrema.model_dump(), sort_keys=True)
            + json.dumps(self.labeling.model_dump(), sort_keys=True)
            + json.dumps(self.features.model_dump(), sort_keys=True)
            + json.dumps(self.dashboard_utility.model_dump(), sort_keys=True)
            + f"|tick_size={self.tick_size}"
            + f"|bar_price_source={BAR_PRICE_SOURCE}"
            + f"|label_entry_reference={LABEL_ENTRY_REFERENCE}"
            + f"|platform_version={PLATFORM_VERSION}"
            # W1 P4a: the labeling pipeline became a batch drive of the SC runtime
            # (per-trade level fold, plugin zones/touches per bar-close, stream
            # features). Folding the pipeline identity rolls the cache tag so
            # pre-W1 ml_utility_* caches are invalidated, never silently reused.
            + "|decision_pipeline=sc_runtime_stream_v1"
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:8]
