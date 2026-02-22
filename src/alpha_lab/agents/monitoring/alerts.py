"""
Alert generation and routing per architecture spec Section 7.1.

Alert Levels:
- INFO: Metric within normal range but trending toward threshold
- WARNING: Single metric breached threshold, others still healthy
- CRITICAL: Multiple metrics breached OR prop firm buffer < 30%
- HALT: Immediate stop trading required (circuit breaker)

Key Triggers:
- IC drops below 50% of backtest IC for 20+ bars -> WARNING
- Hit rate below 48% for 2+ consecutive windows -> WARNING
- Net Sharpe below 0.5 for 20+ days -> CRITICAL
- Prop firm DD buffer < 30% -> CRITICAL
- Daily loss approaching 80% of limit -> HALT
"""

from __future__ import annotations

from alpha_lab.core.contracts import Alert


def check_ic_degradation(
    live_ic: float, backtest_ic: float, bars_below: int
) -> Alert | None:
    """Check if IC has dropped below 50% of backtest for 20+ bars."""
    raise NotImplementedError


def check_hit_rate_degradation(
    live_hit_rate: float, consecutive_windows_below: int
) -> Alert | None:
    """Check if hit rate below 48% for 2+ consecutive windows."""
    raise NotImplementedError


def check_sharpe_degradation(
    live_sharpe: float, days_below: int
) -> Alert | None:
    """Check if net Sharpe below 0.5 for 20+ days."""
    raise NotImplementedError


def check_prop_firm_buffer(
    dd_buffer_pct: float, daily_buffer_pct: float
) -> Alert | None:
    """Check prop firm drawdown and daily loss buffers."""
    raise NotImplementedError


def evaluate_all_alerts(metrics: dict) -> list[Alert]:
    """Run all alert checks and return triggered alerts."""
    raise NotImplementedError
