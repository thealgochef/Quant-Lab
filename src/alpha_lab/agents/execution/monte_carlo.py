"""
Monte Carlo simulation for ruin probability analysis.

Simulates trade sequences to estimate:
- Probability of blowing the account over 100, 500, 1000 trade sequences
- Must be < 5% to pass (architecture spec Section 6.1)
"""

from __future__ import annotations

import numpy as np

from alpha_lab.core.config import PropFirmProfile

_DEFAULT_SEQUENCES = [100, 500, 1000]


def simulate_ruin_probability(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    profile: PropFirmProfile,
    num_simulations: int = 10000,
    trade_sequences: list[int] | None = None,
    rng_seed: int | None = None,
) -> dict[int, float]:
    """
    Run Monte Carlo simulation to estimate ruin probability.

    "Ruin" = account equity drops below the trailing max drawdown limit
    (i.e., the cumulative loss from peak exceeds profile.trailing_max_drawdown).

    Args:
        win_rate: Historical win rate (0 to 1)
        avg_win: Average winning trade P&L (positive dollars)
        avg_loss: Average losing trade P&L (positive number)
        profile: Prop firm profile with drawdown limits
        num_simulations: Number of MC paths per sequence length
        trade_sequences: Sequence lengths to test (default: [100, 500, 1000])
        rng_seed: Optional seed for reproducibility

    Returns:
        Dict of sequence_length -> P(ruin)
    """
    if trade_sequences is None:
        trade_sequences = list(_DEFAULT_SEQUENCES)

    if not (0 < win_rate < 1) or avg_win <= 0 or avg_loss <= 0:
        return {seq: 1.0 for seq in trade_sequences}

    rng = np.random.default_rng(rng_seed)
    max_dd_limit = profile.trailing_max_drawdown

    results: dict[int, float] = {}

    for n_trades in trade_sequences:
        # Generate all simulations at once for efficiency
        # Each row is a simulation, each column is a trade
        outcomes = rng.random((num_simulations, n_trades))
        # Win = avg_win, Loss = -avg_loss
        pnl_matrix = np.where(outcomes < win_rate, avg_win, -avg_loss)

        # Cumulative P&L for each simulation
        cum_pnl = np.cumsum(pnl_matrix, axis=1)

        # Running peak
        running_peak = np.maximum.accumulate(cum_pnl, axis=1)

        # Drawdown at each point
        drawdowns = running_peak - cum_pnl

        # Max drawdown per simulation
        max_drawdowns = np.max(drawdowns, axis=1)

        # Count ruin: max drawdown exceeds limit
        ruin_count = int(np.sum(max_drawdowns >= max_dd_limit))

        results[n_trades] = ruin_count / num_simulations

    return results


def compute_expected_value(
    win_rate: float, avg_win: float, avg_loss: float
) -> float:
    """Compute expected value per trade.

    Args:
        win_rate: Win probability
        avg_win: Average win (positive)
        avg_loss: Average loss (positive)

    Returns:
        Expected P&L per trade
    """
    return win_rate * avg_win - (1 - win_rate) * avg_loss
