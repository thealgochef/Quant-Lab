"""
Validation Firewall — the critical boundary between SIG-001 and VAL-001.

Ensures that only opaque signal vectors (direction + strength arrays)
cross the boundary. No signal parameters, category names, or indicator
types are transmitted. See architecture spec Section 10.

This module orchestrates running all validation tests and assembling verdicts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from alpha_lab.core.contracts import SignalVector, SignalVerdict


class ValidationTest(ABC):
    """Abstract base for a single validation test in the test battery."""

    test_name: str

    @abstractmethod
    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """
        Run this test on a signal.

        Args:
            signal: Opaque signal vector (direction + strength only)
            price_data: Corresponding price series for forward return computation

        Returns:
            Dict of metric_name -> value
        """


def strip_signal_metadata(signal: SignalVector) -> dict[str, Any]:
    """
    Strip implementation details before passing to validation.

    Per firewall rules (Section 10.1):
    - ALLOWED: direction array, strength array, timeframe
    - BLOCKED: parameters, category, indicator type
    """
    raise NotImplementedError


def assemble_verdict(
    signal_id: str, test_results: dict[str, dict[str, Any]], thresholds: dict[str, float]
) -> SignalVerdict:
    """
    Combine results from all validation tests into a single SignalVerdict.

    Applies threshold checks and determines DEPLOY/REFINE/REJECT.
    """
    raise NotImplementedError
