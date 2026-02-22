"""
Configuration loader — reads YAML files and produces typed Pydantic models.

Usage:
    settings = load_settings()
    print(settings.instruments["NQ"].tick_value)
    print(settings.prop_firms["apex_50k"].trailing_max_drawdown)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ─── Typed Config Models ────────────────────────────────────────


class InstrumentSpec(BaseModel):
    """Contract specification for a futures instrument."""

    full_name: str
    exchange: str
    tick_size: float
    tick_value: float
    point_value: float
    exchange_nfa_per_side: float
    broker_commission_per_side: float
    avg_slippage_ticks: float
    avg_slippage_per_side: float
    total_round_turn: float
    session_open: str
    session_close: str
    rth_open: str
    rth_close: str


class PropFirmProfile(BaseModel):
    """Prop firm account constraint profile."""

    name: str
    account_size: float
    daily_loss_limit: float | None = None
    trailing_max_drawdown: float
    drawdown_type: str = Field(description="real_time | end_of_day")
    max_contracts: int
    consistency_rule_pct: float
    profit_target: float
    allowed_hours: str = "all"
    news_restriction: bool = False


class KillzoneConfig(BaseModel):
    """Killzone time window definition."""

    start: str
    end: str
    label: str


class SignalThresholds(BaseModel):
    """Validation thresholds for signal approval."""

    ic_tstat_min: float = 2.0
    ic_rolling_window: int = 252
    hit_rate_min: float = 0.51
    hit_rate_horizons: list[int] = Field(default_factory=lambda: [1, 5, 10, 15, 30, 60, 120, 240])
    sharpe_min: float = 1.0
    max_drawdown_max: float = 0.15
    profit_factor_min: float = 1.2
    max_factor_correlation: float = 0.30
    min_incremental_r2: float = 0.005
    subsample_quarters: int = 4
    parameter_sensitivity_pct: float = 20.0


class ExecutionThresholds(BaseModel):
    """Thresholds for execution/risk approval."""

    net_sharpe_min: float = 0.8
    mc_ruin_probability_max: float = 0.05
    turnover_reduction_for_revival: float = 0.50


class PortfolioThresholds(BaseModel):
    """Portfolio-level go/no-go thresholds."""

    min_deploy_signals: int = 8
    composite_ic_min: float = 0.10
    net_of_cost_sharpe_min: float = 1.0


class Settings(BaseModel):
    """Root settings container assembled from all YAML config files."""

    instruments: dict[str, InstrumentSpec]
    prop_firms: dict[str, PropFirmProfile]
    killzones: dict[str, KillzoneConfig]
    signal_thresholds: SignalThresholds
    execution_thresholds: ExecutionThresholds
    portfolio_thresholds: PortfolioThresholds
    data_provider: str = "stub"
    symbols: list[str] = Field(default_factory=lambda: ["NQ", "ES"])
    default_timeframes: list[str] = Field(default_factory=list)
    log_level: str = "INFO"
    run_mode: str = "research"
    max_refine_iterations: int = 3


# ─── Loader ─────────────────────────────────────────────────────


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings(config_dir: Path | None = None) -> Settings:
    """
    Load and merge all YAML configuration files into a Settings object.

    Args:
        config_dir: Path to config directory. Defaults to <project_root>/config/
    """
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[3] / "config"

    settings_data = _read_yaml(config_dir / "settings.yaml")
    instruments_data = _read_yaml(config_dir / "instruments.yaml")
    prop_firms_data = _read_yaml(config_dir / "prop_firms.yaml")
    validation_data = _read_yaml(config_dir / "validation_thresholds.yaml")

    instruments = {
        symbol: InstrumentSpec(**spec) for symbol, spec in instruments_data.items()
    }

    prop_firms = {
        profile_id: PropFirmProfile(**profile)
        for profile_id, profile in prop_firms_data.get("profiles", {}).items()
    }

    killzones = {
        name: KillzoneConfig(**kz)
        for name, kz in settings_data.get("killzones", {}).items()
    }

    signal_thresholds = SignalThresholds(
        **validation_data.get("signal_thresholds", {})
    )
    execution_thresholds = ExecutionThresholds(
        **validation_data.get("execution_thresholds", {})
    )
    portfolio_thresholds = PortfolioThresholds(
        **validation_data.get("portfolio_thresholds", {})
    )

    return Settings(
        instruments=instruments,
        prop_firms=prop_firms,
        killzones=killzones,
        signal_thresholds=signal_thresholds,
        execution_thresholds=execution_thresholds,
        portfolio_thresholds=portfolio_thresholds,
        data_provider=settings_data.get("data", {}).get("provider", "stub"),
        symbols=settings_data.get("data", {}).get("symbols", ["NQ", "ES"]),
        default_timeframes=settings_data.get("data", {}).get("default_timeframes", []),
        log_level=settings_data.get("system", {}).get("log_level", "INFO"),
        run_mode=settings_data.get("pipeline", {}).get("run_mode", "research"),
        max_refine_iterations=settings_data.get("pipeline", {}).get(
            "max_refine_iterations", 3
        ),
    )
