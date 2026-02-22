"""
Performance metric collectors for live monitoring.

Continuous metrics (per architecture spec Section 7.1):
1. Rolling IC (20-bar window)
2. Rolling Hit Rate (50-trade window)
3. Cost-Adjusted Sharpe (20-day window)
4. Signal Decay Velocity
5. Turnover Rate
6. Slippage Tracking
7. Fill Rate
"""

from __future__ import annotations

from typing import Any


def compute_rolling_ic(
    signal_values: list[float], returns: list[float], window: int = 20
) -> list[float]:
    """Compute rolling Information Coefficient over a sliding window."""
    raise NotImplementedError


def compute_rolling_hit_rate(
    predictions: list[int], actuals: list[int], window: int = 50
) -> list[float]:
    """Compute rolling directional hit rate over a trade window."""
    raise NotImplementedError


def compute_rolling_sharpe(
    daily_returns: list[float], window: int = 20
) -> list[float]:
    """Compute rolling cost-adjusted Sharpe ratio."""
    raise NotImplementedError


def compute_decay_velocity(
    ic_series: list[float], expected_half_life: float
) -> dict[str, Any]:
    """Compare actual IC decay against expected decay curve."""
    raise NotImplementedError


def compute_slippage_tracking(
    expected_prices: list[float], realized_prices: list[float]
) -> dict[str, float]:
    """Track realized vs assumed slippage."""
    raise NotImplementedError
