"""
Risk-Adjusted Return testing.

Tests:
- Sharpe ratio (must be > 1.0 to pass)
- Sortino ratio
- Maximum drawdown (must be < 15% to pass)
- Profit factor (must be > 1.2 to pass)
- Calmar ratio
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.agents.validation.firewall import ValidationTest, compute_forward_returns
from alpha_lab.core.contracts import SignalVector

_DEFAULT_HORIZON = 5
# Annualization factor: assume ~252 trading days, bars per day varies by TF
# Use 252 as a reasonable default for daily-equivalent Sharpe
_ANNUALIZATION = np.sqrt(252)


class RiskAdjustedTest(ValidationTest):
    """Risk-adjusted return test suite."""

    test_name = "risk_adjusted"

    def __init__(self, horizon: int = _DEFAULT_HORIZON) -> None:
        self.horizon = horizon

    def evaluate(self, signal: SignalVector, price_data: Any) -> dict[str, Any]:
        """Compute risk-adjusted return metrics."""
        close = price_data["close"]
        direction = signal.direction

        if isinstance(direction, pd.Series):
            direction = direction.values
        direction = pd.Series(direction, index=close.index, dtype=float)

        fwd = compute_forward_returns(close, self.horizon)

        # Strategy returns: direction * forward return
        strat_returns = direction * fwd
        valid = strat_returns.dropna()

        if len(valid) < 30:
            return {
                "sharpe": 0.0,
                "sortino": 0.0,
                "max_drawdown": 1.0,
                "profit_factor": 0.0,
                "calmar": 0.0,
                "total_return": 0.0,
                "n_bars": 0,
            }

        # Sharpe ratio (annualized)
        mean_ret = valid.mean()
        std_ret = valid.std()
        sharpe = (mean_ret / std_ret * _ANNUALIZATION) if std_ret > 0 else 0.0

        # Sortino ratio (downside deviation)
        downside = valid[valid < 0]
        downside_std = downside.std() if len(downside) > 1 else std_ret
        sortino = (mean_ret / downside_std * _ANNUALIZATION) if downside_std > 0 else 0.0

        # Maximum drawdown
        cum_returns = (1 + valid).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = float(abs(drawdown.min()))

        # Profit factor = gross profits / gross losses
        gross_profit = valid[valid > 0].sum()
        gross_loss = abs(valid[valid < 0].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Calmar ratio = annualized return / max drawdown
        total_return = float(cum_returns.iloc[-1] - 1) if len(cum_returns) > 0 else 0.0
        calmar = (total_return / max_drawdown) if max_drawdown > 0 else 0.0

        return {
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(min(profit_factor, 100.0)),
            "calmar": float(calmar),
            "total_return": total_return,
            "n_bars": len(valid),
        }
