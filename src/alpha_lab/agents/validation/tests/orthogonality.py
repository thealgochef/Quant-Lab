"""
Factor Orthogonality testing.

Tests:
- Correlation with known factors: momentum, mean-reversion, volatility, volume, calendar
- Incremental R² above existing factor exposures
- Maximum single-factor correlation must be < 0.30 to pass
"""

from __future__ import annotations

from typing import Any

from alpha_lab.agents.validation.firewall import ValidationTest
from alpha_lab.core.contracts import SignalVector


class OrthogonalityTest(ValidationTest):
    """Factor orthogonality test suite."""

    test_name = "orthogonality"

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute factor correlations and incremental R² metrics."""
        raise NotImplementedError
