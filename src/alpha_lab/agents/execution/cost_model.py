"""
Transaction cost modeling per architecture spec Section 6.1.

NQ costs: exchange+NFA $2.14/side, broker $0.50/side, slippage 0.5 tick ($2.50/side)
ES costs: exchange+NFA $2.14/side, broker $0.50/side, slippage 0.25 tick ($3.13/side)

Functions:
- compute_round_turn_cost: Total cost for one round-turn trade
- compute_cost_analysis: Full cost breakdown with net P&L and breakeven
- compute_turnover_metrics: Trades/day, holding period, flip rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.core.config import InstrumentSpec
from alpha_lab.core.contracts import CostAnalysis


def compute_round_turn_cost(instrument: InstrumentSpec) -> float:
    """Compute total round-turn cost for an instrument.

    Round-turn = (exchange+NFA + broker + slippage) * 2 sides.
    """
    per_side = (
        instrument.exchange_nfa_per_side
        + instrument.broker_commission_per_side
        + instrument.avg_slippage_per_side
    )
    return per_side * 2


def compute_cost_analysis(
    gross_pnl: float,
    num_trades: int,
    instrument: InstrumentSpec,
    gross_sharpe: float = 0.0,
) -> CostAnalysis:
    """Compute full cost analysis including net P&L and cost drag.

    Args:
        gross_pnl: Total gross P&L in dollars
        num_trades: Number of round-turn trades
        instrument: Instrument spec with cost data
        gross_sharpe: Gross Sharpe ratio (pre-cost)

    Returns:
        CostAnalysis with net figures
    """
    rt_cost = compute_round_turn_cost(instrument)
    total_costs = rt_cost * num_trades
    net_pnl = gross_pnl - total_costs

    cost_drag_pct = total_costs / gross_pnl if gross_pnl > 0 else float("inf")

    # Net Sharpe: approximate reduction proportional to cost drag
    if gross_sharpe > 0 and cost_drag_pct < 1.0:
        net_sharpe = gross_sharpe * (1 - cost_drag_pct)
    else:
        net_sharpe = 0.0

    # Breakeven hit rate: at what win rate does net P&L = 0?
    # Assuming 1:1 RR, breakeven = 0.5 + (cost_per_trade / (2 * avg_trade_size))
    # Simplified: breakeven = costs / (costs + gross_pnl) when gross > 0
    if gross_pnl > 0 and num_trades > 0:
        avg_gross_per_trade = gross_pnl / num_trades
        breakeven_hit_rate = rt_cost / (rt_cost + avg_gross_per_trade)
    else:
        breakeven_hit_rate = 1.0

    return CostAnalysis(
        gross_pnl=gross_pnl,
        total_costs=total_costs,
        net_pnl=net_pnl,
        cost_drag_pct=cost_drag_pct,
        gross_sharpe=gross_sharpe,
        net_sharpe=net_sharpe,
        breakeven_hit_rate=breakeven_hit_rate,
    )


def compute_turnover_metrics(
    signal_direction: pd.Series,
    bars_per_day: int = 78,
) -> dict[str, float]:
    """Compute turnover metrics from a signal direction series.

    Args:
        signal_direction: Series of [-1, 0, +1] values
        bars_per_day: Number of bars per trading day (78 for 5m on NQ RTH)

    Returns:
        Dict with trades_per_day, avg_holding_bars, flip_rate
    """
    direction = signal_direction.dropna()
    if len(direction) < 2:
        return {
            "trades_per_day": 0.0,
            "avg_holding_bars": 0.0,
            "flip_rate": 0.0,
        }

    total_bars = len(direction)
    n_days = max(total_bars / bars_per_day, 1.0)

    # A "trade" is entering a non-zero position
    non_zero = direction[direction != 0]
    entries = (non_zero != non_zero.shift(1)).sum() if len(non_zero) > 0 else 0
    trades_per_day = entries / n_days

    # Average holding: mean consecutive bars in the same non-zero direction
    holding_lengths: list[int] = []
    current_length = 0
    for val in direction.values:
        if val != 0:
            current_length += 1
        else:
            if current_length > 0:
                holding_lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        holding_lengths.append(current_length)
    avg_holding = float(np.mean(holding_lengths)) if holding_lengths else 0.0

    # Flip rate: direct sign changes (-1 -> +1 or +1 -> -1) without going through 0
    flips = 0
    prev = 0
    for val in direction.values:
        if val != 0 and prev != 0 and val != prev:
            flips += 1
        if val != 0:
            prev = val
    flip_rate = flips / max(entries, 1)

    return {
        "trades_per_day": float(trades_per_day),
        "avg_holding_bars": float(avg_holding),
        "flip_rate": float(flip_rate),
    }


def estimate_trade_stats(
    signal_direction: pd.Series,
    signal_strength: pd.Series,
    close: pd.Series,
    horizon: int = 5,
) -> dict[str, float]:
    """Estimate win rate, average win, average loss from signal + price data.

    Used by position sizing and Monte Carlo modules.

    Args:
        signal_direction: Series of [-1, 0, +1]
        signal_strength: Series of [0, 1]
        close: Price series
        horizon: Forward return horizon in bars

    Returns:
        Dict with win_rate, avg_win, avg_loss, num_trades
    """
    from alpha_lab.agents.validation.firewall import compute_forward_returns

    sig = signal_direction * signal_strength
    fwd = compute_forward_returns(close, horizon)

    mask = sig.notna() & fwd.notna() & (sig != 0)
    if mask.sum() < 10:
        return {"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "num_trades": 0}

    # Trade P&L: signal * forward return (positive = correct direction)
    trade_pnl = sig[mask].values * fwd[mask].values

    wins = trade_pnl[trade_pnl > 0]
    losses = trade_pnl[trade_pnl < 0]

    win_rate = len(wins) / len(trade_pnl) if len(trade_pnl) > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0

    return {
        "win_rate": float(win_rate),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "num_trades": int(mask.sum()),
    }
