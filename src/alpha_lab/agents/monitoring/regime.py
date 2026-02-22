"""
Regime detection and classification.

Regime Types (per architecture spec Section 7.1):
- TRENDING: EMA alignment confirmed, KAMA slope strong
- RANGING: EMAs flat/interleaved, KAMA flat, low ADX
- VOLATILE: ATR spike > 2x normal, wide EMA spread
- TRANSITIONAL: CHOCH detected, EMA crossover in progress

On regime transition: notify ORCH-001 with new classification
and recommended signal weight adjustments.
"""

from __future__ import annotations

from typing import Any

from alpha_lab.core.enums import Regime


def classify_regime(market_data: dict[str, Any]) -> tuple[Regime, float]:
    """
    Classify current market regime.

    Args:
        market_data: Dict containing EMA values, KAMA slope, ATR, ADX, etc.

    Returns:
        Tuple of (regime, confidence_score)
    """
    raise NotImplementedError


def detect_regime_transition(
    previous_regime: Regime, current_regime: Regime, confidence: float
) -> dict[str, Any] | None:
    """
    Detect if a regime transition has occurred.

    Returns transition details if detected, None otherwise.
    """
    raise NotImplementedError


def compute_regime_signal_weights(regime: Regime) -> dict[str, float]:
    """
    Compute recommended signal weight adjustments for the current regime.

    Returns dict of signal_category -> weight_multiplier
    """
    raise NotImplementedError
