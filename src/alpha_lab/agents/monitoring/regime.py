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

from datetime import UTC, datetime
from typing import Any

import numpy as np

from alpha_lab.core.enums import Regime

# ─── Regime Signal Weight Tables ──────────────────────────────

_REGIME_WEIGHTS: dict[Regime, dict[str, float]] = {
    Regime.TRENDING: {
        "trend_following": 1.2,
        "mean_reversion": 0.5,
        "breakout": 1.0,
        "momentum": 1.3,
        "volume": 1.0,
        "default": 1.0,
    },
    Regime.RANGING: {
        "trend_following": 0.6,
        "mean_reversion": 1.3,
        "breakout": 0.8,
        "momentum": 0.5,
        "volume": 1.1,
        "default": 1.0,
    },
    Regime.VOLATILE: {
        "trend_following": 0.7,
        "mean_reversion": 0.7,
        "breakout": 0.7,
        "momentum": 0.7,
        "volume": 0.7,
        "default": 0.7,
    },
    Regime.TRANSITIONAL: {
        "trend_following": 0.5,
        "mean_reversion": 0.5,
        "breakout": 0.5,
        "momentum": 0.5,
        "volume": 0.5,
        "default": 0.5,
    },
}


def classify_regime(market_data: dict[str, Any]) -> tuple[Regime, float]:
    """
    Classify current market regime using heuristic scoring.

    Args:
        market_data: Dict containing:
            - ema_values: list[float] — [fast, mid, slow] EMA values
            - kama_slope: float — KAMA slope indicator
            - atr_current: float — current ATR value
            - atr_avg: float — average ATR for reference
            - adx: float — ADX trend strength indicator

    Returns:
        Tuple of (regime, confidence_score) where confidence is 0-1.
    """
    ema_values = market_data.get("ema_values", [])
    kama_slope = market_data.get("kama_slope", 0.0)
    atr_current = market_data.get("atr_current", 0.0)
    atr_avg = market_data.get("atr_avg", 1.0)
    adx = market_data.get("adx", 15.0)

    scores: dict[Regime, float] = {
        Regime.TRENDING: 0.0,
        Regime.RANGING: 0.0,
        Regime.VOLATILE: 0.0,
        Regime.TRANSITIONAL: 0.0,
    }

    # ── TRENDING indicators ──
    if len(ema_values) >= 3:
        aligned_up = ema_values[0] > ema_values[1] > ema_values[2]
        aligned_down = ema_values[0] < ema_values[1] < ema_values[2]
        if aligned_up or aligned_down:
            scores[Regime.TRENDING] += 3.0

    if adx > 25:
        scores[Regime.TRENDING] += 2.0

    if abs(kama_slope) > 0.3:
        scores[Regime.TRENDING] += 2.0

    # ── RANGING indicators ──
    if len(ema_values) >= 3:
        ema_arr = np.array(ema_values, dtype=float)
        ema_mean = float(np.mean(ema_arr))
        if ema_mean > 0:
            ema_cv = float(np.std(ema_arr) / ema_mean)
            if ema_cv < 0.005:  # EMAs tightly clustered
                scores[Regime.RANGING] += 3.0

    if adx < 20:
        scores[Regime.RANGING] += 2.0

    if abs(kama_slope) < 0.1:
        scores[Regime.RANGING] += 2.0

    # ── VOLATILE indicators ──
    atr_ratio = atr_current / atr_avg if atr_avg > 0 else 1.0
    if atr_ratio > 2.0:
        scores[Regime.VOLATILE] += 5.0
    elif atr_ratio > 1.5:
        scores[Regime.VOLATILE] += 2.0

    # ── TRANSITIONAL indicators ──
    # Moderate ADX + moderate KAMA slope = potential transition
    if 15 < adx < 25 and 0.1 < abs(kama_slope) < 0.3:
        scores[Regime.TRANSITIONAL] += 2.0

    # Not cleanly aligned EMAs but not tightly clustered either
    if len(ema_values) >= 3:
        fast, mid, slow = ema_values[0], ema_values[1], ema_values[2]
        aligned = (fast > mid > slow) or (fast < mid < slow)
        if not aligned:
            ema_arr = np.array(ema_values, dtype=float)
            ema_mean = float(np.mean(ema_arr))
            if ema_mean > 0:
                ema_cv = float(np.std(ema_arr) / ema_mean)
                if ema_cv >= 0.005:
                    scores[Regime.TRANSITIONAL] += 1.0

    # ── Select best regime ──
    total_score = sum(scores.values())
    if total_score == 0:
        return Regime.RANGING, 0.5  # default

    best_regime = max(scores, key=lambda r: scores[r])
    confidence = scores[best_regime] / total_score

    return best_regime, round(confidence, 3)


def detect_regime_transition(
    previous_regime: Regime, current_regime: Regime, confidence: float
) -> dict[str, Any] | None:
    """
    Detect if a regime transition has occurred.

    Only triggers on high-confidence transitions (confidence >= 0.6).

    Returns:
        Transition details dict if detected, None otherwise.
    """
    if previous_regime == current_regime:
        return None

    if confidence < 0.6:
        return None

    return {
        "from": previous_regime.value,
        "to": current_regime.value,
        "confidence": confidence,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def compute_regime_signal_weights(regime: Regime) -> dict[str, float]:
    """
    Compute recommended signal weight adjustments for the current regime.

    Returns dict of signal_category -> weight_multiplier.
    Categories not in the table get the 'default' weight.
    """
    return dict(_REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS[Regime.RANGING]))
