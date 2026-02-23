"""
Execution & Risk Agent (EXEC-001) — Risk Manager.

Determines whether validated signals can be profitably and safely
executed in prop firm accounts (Apex Trader Funding, Topstep).

See architecture spec Section 6 for full system prompt.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.agents.execution.cost_model import (
    compute_cost_analysis,
    compute_turnover_metrics,
    estimate_trade_stats,
)
from alpha_lab.agents.execution.monte_carlo import simulate_ruin_probability
from alpha_lab.agents.execution.position_sizing import (
    half_kelly_contracts,
    kelly_fraction,
    max_contracts_from_daily_limit,
)
from alpha_lab.agents.execution.prop_constraints import validate_prop_firm_constraints
from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.config import InstrumentSpec, PropFirmProfile
from alpha_lab.core.contracts import (
    ExecutionReport,
    ExecVerdict,
    SignalVerdict,
    ValidationReport,
)
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope

logger = logging.getLogger(__name__)

# Default stop loss in ticks for position sizing
_DEFAULT_STOP_TICKS = 8.0
# Net Sharpe threshold for approval
_NET_SHARPE_MIN = 0.8
# MC ruin probability threshold
_MC_RUIN_MAX = 0.05


class ExecutionAgent(BaseAgent):
    """
    EXEC-001: Risk Manager.

    Responsibilities:
    - Transaction cost modeling (slippage, commissions)
    - Turnover analysis (trades/day, holding period, flip rate)
    - Net-of-cost alpha computation
    - Prop firm feasibility (daily loss, trailing DD, consistency, MC ruin)
    - Position sizing (Kelly, half-Kelly)
    - APPROVED/VETOED verdicts
    """

    def __init__(
        self,
        bus: MessageBus,
        instrument: InstrumentSpec | None = None,
        prop_firm: PropFirmProfile | None = None,
    ) -> None:
        super().__init__(AgentID.EXECUTION, "Execution & Risk", bus)
        self._instrument = instrument
        self._prop_firm = prop_firm

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle EXECUTION_REQUEST messages from Orchestrator."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.EXECUTION_REQUEST:
            self.send_ack(envelope)
            try:
                validation_report = ValidationReport.model_validate(
                    envelope.payload["validation_report"]
                )
                price_data = envelope.payload.get("price_data", {})
                signal_data = envelope.payload.get("signal_data", {})
                instrument = self._instrument
                prop_firm = self._prop_firm

                # Allow per-request override of instrument/firm
                if "instrument_spec" in envelope.payload:
                    instrument = InstrumentSpec.model_validate(
                        envelope.payload["instrument_spec"]
                    )
                if "prop_firm_profile" in envelope.payload:
                    prop_firm = PropFirmProfile.model_validate(
                        envelope.payload["prop_firm_profile"]
                    )

                report = self.analyze_signals(
                    validation_report, price_data, signal_data,
                    instrument, prop_firm,
                )
                self.send_message(
                    receiver=AgentID.ORCHESTRATOR,
                    message_type=MessageType.EXECUTION_REPORT,
                    payload={"report": report.model_dump()},
                    request_id=envelope.request_id,
                )
            except Exception:
                self.logger.exception("Failed to analyze signals")
                self.send_nack(envelope, "Execution analysis failed")
        else:
            self.send_nack(
                envelope,
                f"Unexpected message type: {envelope.message_type.value}",
            )

        self.transition_state(AgentState.IDLE)

    def analyze_signals(
        self,
        validation_report: ValidationReport,
        price_data: dict[str, Any] | None = None,
        signal_data: dict[str, Any] | None = None,
        instrument: InstrumentSpec | None = None,
        prop_firm: PropFirmProfile | None = None,
    ) -> ExecutionReport:
        """Run full cost and risk analysis on DEPLOY-grade signals.

        Args:
            validation_report: Report from VAL-001 with verdicts
            price_data: Dict of timeframe -> pd.DataFrame with OHLCV data
            signal_data: Dict of signal_id -> {direction, strength, timeframe}
            instrument: Instrument specification
            prop_firm: Prop firm profile

        Returns:
            ExecutionReport with approved/vetoed verdicts
        """
        price_data = price_data or {}
        signal_data = signal_data or {}
        instrument = instrument or self._instrument
        prop_firm = prop_firm or self._prop_firm

        approved: list[ExecVerdict] = []
        vetoed: list[ExecVerdict] = []

        # Only evaluate DEPLOY verdicts
        deploy_verdicts = [
            v for v in validation_report.verdicts if v.verdict == "DEPLOY"
        ]

        for sv in deploy_verdicts:
            verdict = self._evaluate_single_signal(
                sv, price_data, signal_data, instrument, prop_firm
            )
            if verdict.verdict == "APPROVED":
                approved.append(verdict)
            else:
                vetoed.append(verdict)

        return ExecutionReport(
            request_id=validation_report.request_id,
            approved_signals=approved,
            vetoed_signals=vetoed,
            portfolio_risk=_build_portfolio_risk(approved),
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _evaluate_single_signal(
        self,
        sv: SignalVerdict,
        price_data: dict[str, Any],
        signal_data: dict[str, Any],
        instrument: InstrumentSpec | None,
        prop_firm: PropFirmProfile | None,
    ) -> ExecVerdict:
        """Evaluate a single DEPLOY signal for execution feasibility."""
        sig_info = signal_data.get(sv.signal_id, {})
        direction = sig_info.get("direction")
        strength = sig_info.get("strength")
        timeframe = sig_info.get("timeframe", "5m")

        # Get price data for this timeframe
        bars = price_data.get(timeframe)
        has_signal_data = (
            direction is not None
            and strength is not None
            and isinstance(bars, pd.DataFrame)
            and not bars.empty
        )

        # --- Turnover metrics ---
        if has_signal_data and isinstance(direction, pd.Series):
            turnover = compute_turnover_metrics(direction)
        else:
            turnover = {
                "trades_per_day": 0.0,
                "avg_holding_bars": 0.0,
                "flip_rate": 0.0,
            }

        # --- Trade stats ---
        if has_signal_data and isinstance(bars, pd.DataFrame) and "close" in bars:
            trade_stats = estimate_trade_stats(
                direction, strength, bars["close"]
            )
        else:
            trade_stats = {
                "win_rate": sv.hit_rate,
                "avg_win": max(sv.sharpe * 100, 50),
                "avg_loss": 50.0,
                "num_trades": 100,
            }

        win_rate = trade_stats["win_rate"]
        avg_win = trade_stats["avg_win"]
        avg_loss = trade_stats["avg_loss"]
        num_trades = trade_stats["num_trades"]

        # --- Cost analysis ---
        if instrument is not None:
            gross_pnl = win_rate * avg_win * num_trades - (1 - win_rate) * avg_loss * num_trades
            costs = compute_cost_analysis(
                gross_pnl=max(gross_pnl, 0.01),
                num_trades=num_trades,
                instrument=instrument,
                gross_sharpe=sv.sharpe,
            )
        else:
            costs = _default_cost_analysis(sv)

        # --- Position sizing ---
        kelly_f = kelly_fraction(win_rate, avg_win, avg_loss)
        hk_contracts = 0
        max_from_limit = 0
        if instrument is not None and prop_firm is not None:
            hk_contracts = half_kelly_contracts(
                kelly_f, prop_firm.account_size, instrument
            )
            daily_limit = prop_firm.daily_loss_limit or prop_firm.trailing_max_drawdown
            max_from_limit = max_contracts_from_daily_limit(
                daily_limit, _DEFAULT_STOP_TICKS, instrument
            )

        # --- Monte Carlo ruin probability ---
        mc_ruin = 1.0
        if prop_firm is not None and win_rate > 0 and avg_win > 0 and avg_loss > 0:
            # Derive seed from signal_id for unique-per-signal but reproducible paths
            sig_hash = int(hashlib.sha256(sv.signal_id.encode()).hexdigest(), 16)
            rng_seed = sig_hash % (2**31)
            ruin_probs = simulate_ruin_probability(
                win_rate, avg_win, avg_loss, prop_firm,
                num_simulations=1000,
                trade_sequences=[100, 500],
                rng_seed=rng_seed,
            )
            mc_ruin = max(ruin_probs.values()) if ruin_probs else 1.0

        # --- Prop firm feasibility ---
        if prop_firm is not None:
            # Build synthetic daily P&L from trade stats
            pnl_hash = int(hashlib.sha256(sv.signal_id.encode()).hexdigest(), 16)
            daily_pnl = _synthetic_daily_pnl(
                win_rate, avg_win, avg_loss, num_trades, n_days=60,
                seed=pnl_hash % (2**31),
            )
            feasibility = validate_prop_firm_constraints(
                daily_pnl, prop_firm,
                kelly_f=kelly_f,
                half_kelly_contracts=hk_contracts,
                mc_ruin_prob=mc_ruin,
            )
        else:
            feasibility = _default_feasibility(kelly_f, hk_contracts, mc_ruin)

        # --- Verdict ---
        veto_reason = _check_veto(costs, feasibility, prop_firm)

        exec_verdict = "APPROVED" if veto_reason is None else "VETOED"

        risk_params: dict[str, Any] = {
            "stop_loss_ticks": _DEFAULT_STOP_TICKS,
            "max_contracts": min(
                hk_contracts,
                max_from_limit,
                prop_firm.max_contracts if prop_firm else 999,
            ),
            "kelly_fraction": kelly_f,
        }

        return ExecVerdict(
            signal_id=sv.signal_id,
            verdict=exec_verdict,
            turnover=turnover,
            costs=costs,
            prop_firm=feasibility,
            veto_reason=veto_reason,
            risk_parameters=risk_params,
        )


def _check_veto(
    costs, feasibility, prop_firm: PropFirmProfile | None
) -> str | None:
    """Check if any constraint warrants a veto."""
    if costs.net_sharpe < _NET_SHARPE_MIN:
        return f"Net Sharpe {costs.net_sharpe:.2f} < {_NET_SHARPE_MIN}"

    if not feasibility.passes_trailing_dd:
        return "Trailing drawdown exceeds prop firm limit"

    if not feasibility.passes_daily_limit:
        return "Daily loss exceeds prop firm limit"

    if not feasibility.passes_mc_check:
        return f"MC ruin probability {feasibility.mc_ruin_probability:.2%} > 5%"

    if feasibility.consistency_score < 0.3:
        return f"Consistency score {feasibility.consistency_score:.2f} too low"

    return None


def _default_cost_analysis(sv: SignalVerdict):
    """Build a default CostAnalysis when no instrument spec is available."""
    from alpha_lab.core.contracts import CostAnalysis

    return CostAnalysis(
        gross_pnl=0.0,
        total_costs=0.0,
        net_pnl=0.0,
        cost_drag_pct=0.0,
        gross_sharpe=sv.sharpe,
        net_sharpe=sv.sharpe * 0.8,
        breakeven_hit_rate=0.5,
    )


def _default_feasibility(kelly_f, hk_contracts, mc_ruin):
    """Build default PropFirmFeasibility when no profile is available."""
    from alpha_lab.core.contracts import PropFirmFeasibility

    return PropFirmFeasibility(
        worst_day_pnl=0.0,
        max_trailing_dd=0.0,
        passes_daily_limit=True,
        passes_trailing_dd=True,
        consistency_score=1.0,
        mc_ruin_probability=mc_ruin,
        passes_mc_check=mc_ruin < 0.05,
        recommended_contracts=0,
        kelly_fraction=kelly_f,
        half_kelly_contracts=hk_contracts,
    )


def _synthetic_daily_pnl(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    num_trades: int,
    n_days: int = 60,
    seed: int | None = None,
) -> list[float]:
    """Generate synthetic daily P&L for prop firm constraint testing.

    Uses expected value per trade * trades per day with some variance.
    """
    trades_per_day = max(num_trades / n_days, 1)
    ev_per_trade = win_rate * avg_win - (1 - win_rate) * avg_loss
    daily_ev = ev_per_trade * trades_per_day

    # Add realistic variance: std proportional to sqrt(trades_per_day)
    rng = np.random.default_rng(seed)
    trade_std = np.sqrt(win_rate * avg_win**2 + (1 - win_rate) * avg_loss**2)
    daily_std = trade_std * np.sqrt(trades_per_day)

    daily_pnl = rng.normal(daily_ev, daily_std, n_days)
    return [float(x) for x in daily_pnl]


def _build_portfolio_risk(approved: list[ExecVerdict]) -> dict[str, Any]:
    """Build portfolio-level risk summary from approved signals."""
    if not approved:
        return {
            "total_signals": 0,
            "total_contracts": 0,
            "combined_net_sharpe": 0.0,
        }

    total_contracts = sum(
        v.risk_parameters.get("max_contracts", 0) for v in approved
    )
    net_sharpes = [v.costs.net_sharpe for v in approved]
    avg_net_sharpe = float(np.mean(net_sharpes)) if net_sharpes else 0.0

    return {
        "total_signals": len(approved),
        "total_contracts": total_contracts,
        "combined_net_sharpe": avg_net_sharpe,
    }
