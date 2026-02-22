"""
Robustness testing.

Tests:
- Subsample stability: performance across 4 calendar quarters
- Instrument stability: separate NQ and ES performance
- Regime stability: performance in trending vs mean-reverting vs volatile regimes
- Parameter sensitivity: signal stability under ±20% parameter perturbation
"""

from __future__ import annotations

from typing import Any

from alpha_lab.agents.validation.firewall import ValidationTest
from alpha_lab.core.contracts import SignalVector


class RobustnessTest(ValidationTest):
    """Robustness test suite."""

    test_name = "robustness"

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """
        Compute robustness metrics.

        Across subsamples, instruments, regimes, and parameter
        sensitivity.
        """
        raise NotImplementedError
