"""
Canonical interface contracts for the Alpha Signal Research Lab.

All inter-agent data structures defined here as Pydantic v2 BaseModels.
This is the single source of truth — all agents import from this file.
Matches the architecture spec schemas faithfully.
"""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# ─── Data Infrastructure Contracts (DATA-001 output) ────────────


class QualityReport(BaseModel):
    """Data quality validation results produced by DATA-001."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    passed: bool
    total_bars: int = Field(ge=0)
    gaps_found: int = Field(ge=0)
    gaps_detail: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{timestamp, duration_sec, severity}]",
    )
    volume_zeros: int = Field(ge=0)
    ohlc_violations: int = Field(ge=0)
    cross_tf_mismatches: int = Field(ge=0)
    timestamp_coverage: float = Field(ge=0.0, le=1.0, description="% of expected bars present")
    report_generated_at: str


class SessionMetadata(BaseModel):
    """Session context for a trading day, assembled by DATA-001."""

    session_id: str = Field(description="e.g. 'NQ_2026-02-21_RTH'")
    session_type: str = Field(description="RTH | GLOBEX | PRE_MARKET | POST_MARKET")
    killzone: str = Field(description="LONDON | NEW_YORK | ASIA | OVERLAP | NONE")
    rth_open: str = Field(description="ISO timestamp")
    rth_close: str = Field(description="ISO timestamp")


class PreviousDayLevels(BaseModel):
    """Key reference levels from prior sessions, computed by DATA-001."""

    pd_high: float
    pd_low: float
    pd_mid: float
    pd_close: float
    pw_high: float
    pw_low: float
    overnight_high: float
    overnight_low: float


class DataBundle(BaseModel):
    """
    Primary output of DATA-001. Consumed by SIG-001.

    Contains clean, session-tagged OHLCV bars at all timeframes
    plus quality report and reference levels.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instrument: str = Field(description="NQ | ES")
    bars: dict[str, Any] = Field(
        description="Timeframe -> pd.DataFrame. Keys: '987t','2000t','1m','5m', etc."
    )
    sessions: list[SessionMetadata]
    pd_levels: dict[str, PreviousDayLevels] = Field(
        description="Keyed by date string, e.g. '2026-02-21'"
    )
    quality: QualityReport
    date_range: tuple[str, str] = Field(description="(start_date, end_date)")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def validate_bundle(self) -> bool:
        """Self-validation before handoff to SIG-001."""
        assert self.quality.passed, "Quality check failed"
        for tf, df in self.bars.items():
            if isinstance(df, pd.DataFrame):
                assert len(df) > 0, f"Empty DataFrame for {tf}"
                required = ["open", "high", "low", "close", "volume", "session_id"]
                for col in required:
                    assert col in df.columns, f"Missing {col} in {tf}"
        return True


# ─── Signal Engineering Contracts (SIG-001 output) ──────────────


class SignalVector(BaseModel):
    """A single signal's output across a timeframe, produced by SIG-001."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal_id: str = Field(description="e.g. 'SIG_EMA_CONFLUENCE_5m_v2'")
    category: str = Field(description="e.g. 'ema_confluence'")
    timeframe: str = Field(description="e.g. '5m'")
    version: int = Field(ge=1, description="Incremented on REFINE")
    direction: Any = Field(description="pd.Series of [-1, 0, +1] per bar")
    strength: Any = Field(description="pd.Series of [0, 1] confidence per bar")
    formation_idx: Any = Field(description="pd.Series of bar indices where signal formed")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {'ema_fast': 13, 'ema_mid': 48, ...}",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {'intuition': '...', 'expected_decay': '...'}",
    )


class SignalBundle(BaseModel):
    """
    Primary output of SIG-001. Consumed by VAL-001 (opaque vectors only).

    Contains all signal vectors across all timeframes for a given period.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instrument: str
    signals: list[SignalVector]
    composite_scores: dict[str, Any] = Field(
        default_factory=dict,
        description="Named composite signals -> pd.Series",
    )
    timeframes_covered: list[str]
    total_signals: int = Field(ge=0)
    generation_timestamp: str


# ─── Statistical Validation Contracts (VAL-001 output) ──────────


class SignalVerdict(BaseModel):
    """Verdict for a single signal from VAL-001's test battery."""

    signal_id: str
    verdict: str = Field(description="DEPLOY | REFINE | REJECT")
    ic: float
    ic_tstat: float
    hit_rate: float
    hit_rate_long: float
    hit_rate_short: float
    sharpe: float
    sortino: float
    max_drawdown: float
    profit_factor: float
    decay_half_life: float
    decay_class: str = Field(description="ultra-fast | fast | medium | slow | persistent")
    max_factor_corr: float
    incremental_r2: float
    is_orthogonal: bool
    subsample_stable: bool
    failed_metrics: list[dict[str, Any]] = Field(
        default_factory=list,
        description="[{'metric': 'ic_tstat', 'value': 1.7, 'threshold': 2.0, 'suggestion': '...'}]",
    )
    robustness_detail: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """
    Primary output of VAL-001. Consumed by ORCH-001.

    Contains verdicts for every signal in a bundle plus portfolio-level analysis.
    """

    request_id: str
    signal_bundle_id: str = Field(description="Reference to input SignalBundle request_id")
    verdicts: list[SignalVerdict]
    deploy_count: int = Field(ge=0)
    refine_count: int = Field(ge=0)
    reject_count: int = Field(ge=0)
    bonferroni_adjusted: bool = True
    overall_assessment: str = Field(description="Free-text summary for ORCH-001")
    recommended_composites: list[str] = Field(
        default_factory=list,
        description="Signal IDs that combine well",
    )
    correlation_matrix: dict[str, Any] = Field(
        default_factory=dict,
        description="Pairwise signal correlations",
    )
    timestamp: str


# ─── Execution & Risk Contracts (EXEC-001 output) ───────────────


class CostAnalysis(BaseModel):
    """Transaction cost modeling for a signal."""

    gross_pnl: float
    total_costs: float
    net_pnl: float
    cost_drag_pct: float = Field(description="costs / gross_pnl")
    gross_sharpe: float
    net_sharpe: float
    breakeven_hit_rate: float


class PropFirmFeasibility(BaseModel):
    """Prop firm constraint validation results."""

    worst_day_pnl: float
    max_trailing_dd: float
    passes_daily_limit: bool
    passes_trailing_dd: bool
    consistency_score: float = Field(ge=0.0, le=1.0)
    mc_ruin_probability: float = Field(ge=0.0, le=1.0, description="Monte Carlo P(ruin)")
    passes_mc_check: bool = Field(description="mc_ruin < 0.05")
    recommended_contracts: int = Field(ge=0)
    kelly_fraction: float
    half_kelly_contracts: int = Field(ge=0)


class ExecVerdict(BaseModel):
    """Go/no-go decision for a single signal from EXEC-001."""

    signal_id: str
    verdict: str = Field(description="APPROVED | VETOED")
    turnover: dict[str, Any] = Field(
        description="trades_per_day, avg_holding, flip_rate",
    )
    costs: CostAnalysis
    prop_firm: PropFirmFeasibility
    veto_reason: str | None = Field(
        default=None,
        description="If vetoed, which constraint failed",
    )
    risk_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="stop_loss, take_profit, max_contracts",
    )


class ExecutionReport(BaseModel):
    """
    Primary output of EXEC-001. Consumed by ORCH-001 + MON-001.
    """

    request_id: str
    approved_signals: list[ExecVerdict]
    vetoed_signals: list[ExecVerdict]
    portfolio_risk: dict[str, Any] = Field(
        default_factory=dict,
        description="Combined risk across all approved signals",
    )
    timestamp: str


# ─── Live Monitoring Contracts (MON-001 output) ─────────────────


class Alert(BaseModel):
    """A single alert raised by MON-001."""

    level: str = Field(description="INFO | WARNING | CRITICAL | HALT")
    metric: str = Field(description="Which metric triggered the alert")
    current_value: float
    threshold: float
    backtest_value: float
    message: str
    recommended_action: str
    timestamp: str


class SignalHealthReport(BaseModel):
    """Ongoing health metrics for a deployed signal, tracked by MON-001."""

    signal_id: str
    live_ic: float
    backtest_ic: float
    ic_ratio: float = Field(description="live / backtest")
    live_hit_rate: float
    live_sharpe: float
    gross_pnl_today: float
    net_pnl_today: float
    trades_today: int = Field(ge=0)
    realized_slippage: float
    status: str = Field(description="HEALTHY | DEGRADING | FAILING")


class MonitoringReport(BaseModel):
    """
    Primary output of MON-001. Consumed by ORCH-001.

    Can be REALTIME (per-bar) or DAILY_SUMMARY (end of session).
    """

    request_id: str
    report_type: str = Field(description="REALTIME | DAILY_SUMMARY")
    signals: list[SignalHealthReport]
    alerts: list[Alert]
    regime: dict[str, Any] = Field(
        default_factory=dict,
        description="{'current': 'TRENDING', 'confidence': 0.85, 'transition_detected': False}",
    )
    prop_firm_status: dict[str, Any] = Field(
        default_factory=dict,
        description="{'dd_buffer': 0.72, 'daily_buffer': 0.88, 'total_pnl': 1250.00}",
    )
    recommendations: list[str] = Field(default_factory=list)
    timestamp: str
