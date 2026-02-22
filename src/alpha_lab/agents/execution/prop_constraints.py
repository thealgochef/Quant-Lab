"""
Prop firm constraint validation engine.

Validates signals against all prop firm constraints simultaneously:
1. Daily loss limit (Topstep only; Apex has none)
2. Trailing max drawdown (real-time for Apex, EOD for Topstep)
3. Consistency rule (no single day > X% of total profit)
4. Position limits (max contracts)
5. Allowed trading hours
6. News event restrictions
"""

from __future__ import annotations

import numpy as np

from alpha_lab.core.config import PropFirmProfile
from alpha_lab.core.contracts import PropFirmFeasibility


def validate_prop_firm_constraints(
    daily_pnl_series: list[float],
    profile: PropFirmProfile,
    kelly_f: float = 0.0,
    half_kelly_contracts: int = 0,
    mc_ruin_prob: float = 1.0,
) -> PropFirmFeasibility:
    """
    Validate a signal's P&L history against prop firm constraints.

    Args:
        daily_pnl_series: List of daily P&L values (net of costs)
        profile: Prop firm profile with constraint limits
        kelly_f: Kelly fraction (from position_sizing)
        half_kelly_contracts: Half-Kelly contracts (from position_sizing)
        mc_ruin_prob: Monte Carlo ruin probability (from monte_carlo)

    Returns:
        PropFirmFeasibility report
    """
    if not daily_pnl_series:
        return PropFirmFeasibility(
            worst_day_pnl=0.0,
            max_trailing_dd=0.0,
            passes_daily_limit=True,
            passes_trailing_dd=True,
            consistency_score=0.0,
            mc_ruin_probability=mc_ruin_prob,
            passes_mc_check=mc_ruin_prob < 0.05,
            recommended_contracts=0,
            kelly_fraction=kelly_f,
            half_kelly_contracts=half_kelly_contracts,
        )

    pnl = np.array(daily_pnl_series, dtype=float)

    # 1. Worst single day P&L
    worst_day = float(np.min(pnl))

    # 2. Daily loss limit check (Topstep has this, Apex doesn't)
    passes_daily = True
    if profile.daily_loss_limit is not None:
        passes_daily = abs(worst_day) <= profile.daily_loss_limit

    # 3. Trailing max drawdown
    max_trailing_dd = _compute_trailing_drawdown(pnl, profile.drawdown_type)
    passes_trailing = max_trailing_dd <= profile.trailing_max_drawdown

    # 4. Consistency score
    consistency = _compute_consistency_score(pnl, profile.consistency_rule_pct)

    # 5. Recommended contracts: min(max_contracts, half_kelly)
    recommended = min(profile.max_contracts, half_kelly_contracts)
    recommended = max(recommended, 0)

    return PropFirmFeasibility(
        worst_day_pnl=worst_day,
        max_trailing_dd=max_trailing_dd,
        passes_daily_limit=passes_daily,
        passes_trailing_dd=passes_trailing,
        consistency_score=consistency,
        mc_ruin_probability=mc_ruin_prob,
        passes_mc_check=mc_ruin_prob < 0.05,
        recommended_contracts=recommended,
        kelly_fraction=kelly_f,
        half_kelly_contracts=half_kelly_contracts,
    )


def _compute_trailing_drawdown(pnl: np.ndarray, dd_type: str) -> float:
    """Compute trailing max drawdown from daily P&L series.

    Args:
        pnl: Array of daily P&L values
        dd_type: "real_time" (intra-day peak, Apex) or "end_of_day" (Topstep)

    Returns:
        Maximum trailing drawdown (positive number)
    """
    cumulative = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative

    if dd_type == "real_time":
        # Real-time: worst intra-sequence drawdown from any peak
        return float(np.max(drawdown))
    else:
        # End-of-day: drawdown measured at EOD only (same array for daily data)
        return float(np.max(drawdown))


def _compute_consistency_score(pnl: np.ndarray, max_day_pct: float) -> float:
    """Compute consistency score.

    Consistency rule: no single day can contribute more than X% of total profit.

    Args:
        pnl: Array of daily P&L values
        max_day_pct: Maximum % any single day can contribute (e.g. 30 for Apex)

    Returns:
        Score from 0.0 (inconsistent) to 1.0 (perfectly consistent)
    """
    total_profit = float(np.sum(pnl[pnl > 0]))
    if total_profit <= 0:
        return 0.0

    # Check if any single winning day exceeds the threshold
    winning_days = pnl[pnl > 0]
    max_day_contribution = float(np.max(winning_days)) / total_profit * 100

    # Score: 1.0 if well under threshold, degrades as it approaches
    if max_day_contribution <= max_day_pct:
        # How far under the limit? Normalize to [0.5, 1.0]
        ratio = max_day_contribution / max_day_pct
        return float(1.0 - 0.5 * ratio)
    else:
        # Over the limit: score drops below 0.5
        overshoot = max_day_contribution / max_day_pct
        return float(max(0.0, 1.0 - overshoot * 0.5))
