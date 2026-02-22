"""
Transaction cost modeling per architecture spec Section 6.1.

NQ costs: exchange+NFA $2.14/side, broker $0.50/side, slippage 0.5 tick ($2.50/side)
ES costs: exchange+NFA $2.14/side, broker $0.50/side, slippage 0.25 tick ($3.13/side)
"""

from __future__ import annotations

from alpha_lab.core.config import InstrumentSpec
from alpha_lab.core.contracts import CostAnalysis


def compute_round_turn_cost(instrument: InstrumentSpec) -> float:
    """Compute total round-turn cost for an instrument."""
    raise NotImplementedError


def compute_cost_analysis(
    gross_pnl: float,
    num_trades: int,
    instrument: InstrumentSpec,
) -> CostAnalysis:
    """Compute full cost analysis including net P&L and cost drag."""
    raise NotImplementedError
