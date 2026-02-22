"""
Hit Rate testing.

Tests:
- Directional accuracy: percentage of correct sign predictions
- Long vs short hit rates (separate)
- Conditional hit rate by signal strength quintile
- Overall hit rate must exceed 51% to pass
"""

from __future__ import annotations

from typing import Any

from alpha_lab.agents.validation.firewall import ValidationTest
from alpha_lab.core.contracts import SignalVector


class HitRateTest(ValidationTest):
    """Directional accuracy test suite."""

    test_name = "hit_rate"

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute hit rate metrics including long/short and by quintile."""
        raise NotImplementedError
