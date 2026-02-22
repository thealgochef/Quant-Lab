"""
Prop firm constraint validation engine.

Validates signals against all prop firm constraints simultaneously:
1. Daily loss limit
2. Trailing max drawdown
3. Consistency rule (no single day > X% of total profit)
4. Position limits
5. Allowed trading hours
6. News event restrictions
"""

from __future__ import annotations

from alpha_lab.core.config import PropFirmProfile
from alpha_lab.core.contracts import PropFirmFeasibility


def validate_prop_firm_constraints(
    daily_pnl_series: list[float],
    profile: PropFirmProfile,
) -> PropFirmFeasibility:
    """
    Validate a signal's P&L history against prop firm constraints.

    Checks all constraints simultaneously and returns feasibility report.
    """
    raise NotImplementedError
