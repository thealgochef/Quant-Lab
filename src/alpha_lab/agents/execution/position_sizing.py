"""
Position sizing models.

Implements:
- Full Kelly fraction
- Half-Kelly (recommended for safety)
- Max contracts given daily loss limit
- Recommended contracts per signal conviction level
"""

from __future__ import annotations

from alpha_lab.core.config import InstrumentSpec


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Compute the Kelly criterion fraction."""
    raise NotImplementedError


def half_kelly_contracts(
    kelly_f: float,
    account_size: float,
    instrument: InstrumentSpec,
) -> int:
    """Compute half-Kelly position size in contracts."""
    raise NotImplementedError


def max_contracts_from_daily_limit(
    daily_loss_limit: float,
    stop_loss_ticks: float,
    instrument: InstrumentSpec,
) -> int:
    """Compute max contracts such that a stop-out stays within daily loss limit."""
    raise NotImplementedError
