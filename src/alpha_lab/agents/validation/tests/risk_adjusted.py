"""
Risk-Adjusted Return testing.

Tests:
- Sharpe ratio (must be > 1.0 to pass)
- Sortino ratio
- Maximum drawdown (must be < 15% to pass)
- Profit factor (must be > 1.2 to pass)
- Calmar ratio
"""

from __future__ import annotations

from typing import Any

from alpha_lab.agents.validation.firewall import ValidationTest
from alpha_lab.core.contracts import SignalVector


class RiskAdjustedTest(ValidationTest):
    """Risk-adjusted return test suite."""

    test_name = "risk_adjusted"

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """
        Compute risk-adjusted return metrics.

        Metrics: Sharpe, Sortino, drawdown, profit factor, Calmar.
        """
        raise NotImplementedError
