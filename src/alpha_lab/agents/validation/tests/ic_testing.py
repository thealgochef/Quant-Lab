"""
Information Coefficient (IC) testing.

Tests:
- Spearman rank correlation: signal vs forward returns
- Rolling IC (252-bar window) for stability
- IC Information Ratio (IC_mean / IC_std)
- IC t-statistic (must be > 2.0 to pass)
- Test at horizons: 1, 5, 10, 15, 30, 60, 120, 240 bars
"""

from __future__ import annotations

from typing import Any

from alpha_lab.agents.validation.firewall import ValidationTest
from alpha_lab.core.contracts import SignalVector


class ICTest(ValidationTest):
    """Information Coefficient test suite."""

    test_name = "information_coefficient"

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute IC metrics across all forward horizons."""
        raise NotImplementedError
