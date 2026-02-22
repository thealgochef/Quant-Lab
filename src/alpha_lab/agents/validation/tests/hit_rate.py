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

import numpy as np
import pandas as pd

from alpha_lab.agents.validation.firewall import ValidationTest, compute_forward_returns
from alpha_lab.core.contracts import SignalVector

_DEFAULT_HORIZON = 5


class HitRateTest(ValidationTest):
    """Directional accuracy test suite."""

    test_name = "hit_rate"

    def __init__(self, horizon: int = _DEFAULT_HORIZON) -> None:
        self.horizon = horizon

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute hit rate metrics including long/short and by quintile."""
        close = price_data["close"]
        direction = signal.direction
        strength = signal.strength

        if isinstance(direction, pd.Series):
            direction = direction.values
        if isinstance(strength, pd.Series):
            strength = strength.values

        direction = pd.Series(direction, index=close.index, dtype=float)
        strength = pd.Series(strength, index=close.index, dtype=float)

        fwd = compute_forward_returns(close, self.horizon)

        # Only evaluate bars where signal has a direction
        active = direction != 0
        mask = active & fwd.notna()

        if mask.sum() < 20:
            return {
                "hit_rate_overall": 0.5,
                "hit_rate_long": 0.5,
                "hit_rate_short": 0.5,
                "hit_rate_by_quintile": {},
                "n_signals": 0,
                "n_long": 0,
                "n_short": 0,
            }

        dir_masked = direction[mask]
        fwd_masked = fwd[mask]
        str_masked = strength[mask]

        # Correct prediction: sign of direction matches sign of forward return
        correct = (np.sign(dir_masked) == np.sign(fwd_masked))
        hit_rate_overall = float(correct.mean())

        # Long signals
        long_mask = dir_masked > 0
        n_long = int(long_mask.sum())
        hit_rate_long = float(correct[long_mask].mean()) if n_long > 5 else 0.5

        # Short signals
        short_mask = dir_masked < 0
        n_short = int(short_mask.sum())
        hit_rate_short = float(correct[short_mask].mean()) if n_short > 5 else 0.5

        # Hit rate by strength quintile
        hit_by_quintile: dict[str, float] = {}
        try:
            quintiles = pd.qcut(str_masked, 5, labels=False, duplicates="drop")
            for q in sorted(quintiles.dropna().unique()):
                q_mask = quintiles == q
                if q_mask.sum() >= 5:
                    hit_by_quintile[f"Q{int(q) + 1}"] = float(correct[q_mask].mean())
        except ValueError:
            pass

        return {
            "hit_rate_overall": hit_rate_overall,
            "hit_rate_long": hit_rate_long,
            "hit_rate_short": hit_rate_short,
            "hit_rate_by_quintile": hit_by_quintile,
            "n_signals": int(mask.sum()),
            "n_long": n_long,
            "n_short": n_short,
        }
