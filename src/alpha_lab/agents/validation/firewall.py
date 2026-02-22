"""
Validation Firewall — the critical boundary between SIG-001 and VAL-001.

Ensures that only opaque signal vectors (direction + strength arrays)
cross the boundary. No signal parameters, category names, or indicator
types are transmitted. See architecture spec Section 10.

This module orchestrates running all validation tests and assembling verdicts.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from alpha_lab.core.contracts import SignalVector, SignalVerdict

logger = logging.getLogger(__name__)

# Default thresholds (can be overridden from config)
DEFAULT_THRESHOLDS: dict[str, float] = {
    "ic_tstat_min": 2.0,
    "hit_rate_min": 0.51,
    "sharpe_min": 1.0,
    "max_drawdown_max": 0.15,
    "profit_factor_min": 1.2,
    "max_factor_correlation": 0.30,
    "min_incremental_r2": 0.005,
}


class ValidationTest(ABC):
    """Abstract base for a single validation test in the test battery."""

    test_name: str

    @abstractmethod
    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """
        Run this test on a signal.

        Args:
            signal: Opaque signal vector (direction + strength only)
            price_data: Corresponding price DataFrame for forward return computation

        Returns:
            Dict of metric_name -> value
        """


def strip_signal_metadata(signal: SignalVector) -> dict[str, Any]:
    """
    Strip implementation details before passing to validation.

    Per firewall rules (Section 10.1):
    - ALLOWED: signal_id, direction array, strength array, timeframe
    - BLOCKED: parameters, category, indicator type, metadata
    """
    return {
        "signal_id": signal.signal_id,
        "direction": signal.direction,
        "strength": signal.strength,
        "timeframe": signal.timeframe,
    }


def compute_forward_returns(
    close: pd.Series, horizon: int = 1
) -> pd.Series:
    """Compute forward returns at a given horizon.

    forward_return[i] = close[i + horizon] / close[i] - 1

    Uses shift(-horizon) so the return is point-in-time aligned
    with the signal bar.
    """
    return close.shift(-horizon) / close - 1


def run_test_battery(
    signal: SignalVector,
    price_data: pd.DataFrame,
    tests: list[ValidationTest],
) -> dict[str, dict[str, Any]]:
    """Run all validation tests on a signal.

    Args:
        signal: The signal vector to validate
        price_data: OHLCV DataFrame for the signal's timeframe
        tests: List of ValidationTest instances

    Returns:
        Dict of test_name -> metric results
    """
    results: dict[str, dict[str, Any]] = {}

    for test in tests:
        try:
            metrics = test.evaluate(signal, price_data)
            results[test.test_name] = metrics
        except Exception as exc:
            logger.warning("Test %s failed for %s: %s", test.test_name, signal.signal_id, exc)
            results[test.test_name] = {"error": str(exc)}

    return results


def assemble_verdict(
    signal_id: str,
    test_results: dict[str, dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> SignalVerdict:
    """
    Combine results from all validation tests into a single SignalVerdict.

    Applies threshold checks and determines DEPLOY/REFINE/REJECT.
    """
    th = thresholds or DEFAULT_THRESHOLDS

    # Extract metrics from test results (with safe defaults)
    ic_results = test_results.get("information_coefficient", {})
    hr_results = test_results.get("hit_rate", {})
    ra_results = test_results.get("risk_adjusted", {})
    decay_results = test_results.get("decay_analysis", {})
    ortho_results = test_results.get("orthogonality", {})
    robust_results = test_results.get("robustness", {})

    ic = ic_results.get("ic_mean", 0.0)
    ic_tstat = ic_results.get("ic_tstat", 0.0)
    hit_rate = hr_results.get("hit_rate_overall", 0.5)
    hit_rate_long = hr_results.get("hit_rate_long", 0.5)
    hit_rate_short = hr_results.get("hit_rate_short", 0.5)
    sharpe = ra_results.get("sharpe", 0.0)
    sortino = ra_results.get("sortino", 0.0)
    max_drawdown = ra_results.get("max_drawdown", 1.0)
    profit_factor = ra_results.get("profit_factor", 0.0)
    decay_half_life = decay_results.get("half_life", 0.0)
    decay_class = decay_results.get("decay_class", "ultra-fast")
    max_factor_corr = ortho_results.get("max_factor_corr", 1.0)
    incremental_r2 = ortho_results.get("incremental_r2", 0.0)
    is_orthogonal = ortho_results.get("is_orthogonal", False)
    subsample_stable = robust_results.get("subsample_stable", False)

    # Check each threshold
    failed_metrics: list[dict[str, Any]] = []

    if ic_tstat < th.get("ic_tstat_min", 2.0):
        failed_metrics.append({
            "metric": "ic_tstat",
            "value": ic_tstat,
            "threshold": th["ic_tstat_min"],
            "suggestion": "Increase signal-to-noise ratio",
        })

    if hit_rate < th.get("hit_rate_min", 0.51):
        failed_metrics.append({
            "metric": "hit_rate",
            "value": hit_rate,
            "threshold": th["hit_rate_min"],
            "suggestion": "Improve directional accuracy",
        })

    if sharpe < th.get("sharpe_min", 1.0):
        failed_metrics.append({
            "metric": "sharpe",
            "value": sharpe,
            "threshold": th["sharpe_min"],
            "suggestion": "Reduce variance or increase mean return",
        })

    if max_drawdown > th.get("max_drawdown_max", 0.15):
        failed_metrics.append({
            "metric": "max_drawdown",
            "value": max_drawdown,
            "threshold": th["max_drawdown_max"],
            "suggestion": "Add risk controls to limit drawdown",
        })

    if profit_factor < th.get("profit_factor_min", 1.2):
        failed_metrics.append({
            "metric": "profit_factor",
            "value": profit_factor,
            "threshold": th["profit_factor_min"],
            "suggestion": "Improve win/loss ratio",
        })

    if max_factor_corr > th.get("max_factor_correlation", 0.30):
        failed_metrics.append({
            "metric": "max_factor_corr",
            "value": max_factor_corr,
            "threshold": th["max_factor_correlation"],
            "suggestion": "Reduce exposure to known factors",
        })

    # Determine verdict
    # REJECT: fundamental failure (very low IC, or suspected look-ahead)
    if ic < 0.01 or max_factor_corr > 0.50 or ic_tstat < 0.5:
        verdict = "REJECT"
    elif len(failed_metrics) == 0:
        verdict = "DEPLOY"
    else:
        verdict = "REFINE"

    return SignalVerdict(
        signal_id=signal_id,
        verdict=verdict,
        ic=ic,
        ic_tstat=ic_tstat,
        hit_rate=hit_rate,
        hit_rate_long=hit_rate_long,
        hit_rate_short=hit_rate_short,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        decay_half_life=decay_half_life,
        decay_class=decay_class,
        max_factor_corr=max_factor_corr,
        incremental_r2=incremental_r2,
        is_orthogonal=is_orthogonal,
        subsample_stable=subsample_stable,
        failed_metrics=failed_metrics,
        robustness_detail=robust_results,
    )
