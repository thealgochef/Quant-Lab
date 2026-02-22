"""
Position sizing models.

Implements:
- Full Kelly fraction: f* = (p * b - q) / b
- Half-Kelly (recommended for safety): f*/2 sized in contracts
- Max contracts given daily loss limit
- Recommended contracts per signal conviction level
"""

from __future__ import annotations

import math

from alpha_lab.core.config import InstrumentSpec


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Compute the Kelly criterion fraction.

    Kelly formula: f* = (p * b - q) / b
    where p = win rate, q = 1 - p, b = avg_win / avg_loss (odds ratio)

    Args:
        win_rate: Historical win rate (0 to 1)
        avg_win: Average winning trade P&L (positive)
        avg_loss: Average losing trade P&L (positive number, not negative)

    Returns:
        Kelly fraction (can be negative if edge is negative).
        Clamped to [-1, 1] range.
    """
    if avg_loss <= 0 or avg_win <= 0 or not (0 < win_rate < 1):
        return 0.0

    b = avg_win / avg_loss  # odds ratio
    p = win_rate
    q = 1.0 - p

    f = (p * b - q) / b

    return float(max(-1.0, min(1.0, f)))


def half_kelly_contracts(
    kelly_f: float,
    account_size: float,
    instrument: InstrumentSpec,
) -> int:
    """Compute half-Kelly position size in contracts.

    Half-Kelly reduces variance by 75% while only giving up 25% of growth.

    Args:
        kelly_f: Full Kelly fraction (from kelly_fraction())
        account_size: Account equity in dollars
        instrument: Instrument spec with point value

    Returns:
        Number of contracts (integer, >= 0)
    """
    if kelly_f <= 0 or account_size <= 0:
        return 0

    half_f = kelly_f / 2.0

    # Dollar amount to risk
    risk_dollars = half_f * account_size

    # Each contract's value exposure = point_value * price
    # Use a typical NQ price (~22000) or ES price (~5500) as reference
    # But actually we size based on margin/risk, not notional
    # For futures: contracts = risk_dollars / (stop_distance * tick_value / tick_size)
    # Simplified: use point_value as proxy for per-contract risk unit
    contracts = risk_dollars / instrument.point_value

    return max(0, int(math.floor(contracts)))


def max_contracts_from_daily_limit(
    daily_loss_limit: float,
    stop_loss_ticks: float,
    instrument: InstrumentSpec,
) -> int:
    """Compute max contracts such that a stop-out stays within daily loss limit.

    Args:
        daily_loss_limit: Maximum allowable daily loss in dollars
        stop_loss_ticks: Stop loss distance in ticks
        instrument: Instrument spec with tick value

    Returns:
        Maximum number of contracts (integer, >= 0)
    """
    if daily_loss_limit <= 0 or stop_loss_ticks <= 0:
        return 0

    # Loss per contract per stop = ticks * tick_value + round_turn_cost
    loss_per_contract = stop_loss_ticks * instrument.tick_value
    # Add round-turn cost per contract
    rt_cost = (
        instrument.exchange_nfa_per_side
        + instrument.broker_commission_per_side
        + instrument.avg_slippage_per_side
    ) * 2
    total_per_contract = loss_per_contract + rt_cost

    if total_per_contract <= 0:
        return 0

    return max(0, int(math.floor(daily_loss_limit / total_per_contract)))


def recommended_contracts(
    signal_strength: float,
    kelly_f: float,
    max_from_limit: int,
    max_from_firm: int,
) -> int:
    """Recommend contracts based on signal conviction and constraints.

    Takes the minimum of Kelly-based sizing and constraint-based limits,
    then scales by signal strength.

    Args:
        signal_strength: Signal strength [0, 1]
        kelly_f: Full Kelly fraction
        max_from_limit: Max contracts from daily loss limit
        max_from_firm: Max contracts allowed by prop firm

    Returns:
        Recommended number of contracts
    """
    if kelly_f <= 0 or signal_strength <= 0:
        return 0

    # Base = min of all constraints
    base = min(max_from_limit, max_from_firm)
    if base <= 0:
        return 0

    # Scale by strength: full contracts at strength=1, reduced at lower
    scaled = base * signal_strength

    return max(1, int(math.floor(scaled)))
