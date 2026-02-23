"""
Live Monitoring Agent (MON-001) — Production Ops.

Tracks every deployed signal in real-time and detects degradation
before it impacts P&L. Early warning system for the entire lab.

See architecture spec Section 7 for full system prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from alpha_lab.agents.monitoring.alerts import evaluate_all_alerts
from alpha_lab.agents.monitoring.dashboard import (
    assemble_daily_report,
)
from alpha_lab.agents.monitoring.regime import (
    classify_regime,
    compute_regime_signal_weights,
    detect_regime_transition,
)
from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import MonitoringReport, SignalHealthReport
from alpha_lab.core.enums import (
    AgentID,
    AgentState,
    MessageType,
    Regime,
    SignalHealth,
)
from alpha_lab.core.message import MessageBus, MessageEnvelope

logger = logging.getLogger(__name__)


@dataclass
class SignalHealthState:
    """Internal state for tracking a deployed signal."""

    signal_id: str
    backtest_ic: float = 0.05
    backtest_hit_rate: float = 0.55
    backtest_sharpe: float = 1.5
    risk_params: dict[str, Any] = field(default_factory=dict)

    # Current live metrics
    live_ic: float = 0.0
    live_hit_rate: float = 0.5
    live_sharpe: float = 0.0

    # Daily P&L
    trades_today: int = 0
    gross_pnl_today: float = 0.0
    net_pnl_today: float = 0.0
    realized_slippage: float = 0.0

    # Alert tracking counters
    bars_below_ic_threshold: int = 0
    consecutive_windows_below_hr: int = 0
    days_below_sharpe_threshold: int = 0


class MonitoringAgent(BaseAgent):
    """
    MON-001: Production Ops.

    Responsibilities:
    - Continuous monitoring of deployed signal performance
    - Rolling IC, hit rate, Sharpe tracking
    - Alert generation (INFO/WARNING/CRITICAL/HALT)
    - Regime classification and transition detection
    - Prop firm buffer tracking (DD and daily limit headroom)
    - Daily report generation (mandatory, even on no-trade days)
    """

    def __init__(self, bus: MessageBus) -> None:
        super().__init__(AgentID.MONITORING, "Live Monitoring", bus)

        # Active signal tracking
        self._active_signals: dict[str, SignalHealthState] = {}

        # Regime state
        self._current_regime: Regime = Regime.RANGING
        self._regime_confidence: float = 0.5

        # Prop firm status
        self._prop_firm_status: dict[str, Any] = {
            "account_size": 50000,
            "current_dd": 0.0,
            "daily_loss": 0.0,
            "trailing_dd_limit": 2500,
            "daily_loss_limit": None,
            "dd_buffer_pct": 1.0,
            "daily_buffer_pct": 1.0,
            "total_pnl": 0.0,
        }

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle DEPLOY_COMMAND and RESUME_COMMAND messages."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.DEPLOY_COMMAND:
            self.send_ack(envelope)
            try:
                self._handle_deploy_command(envelope)
            except Exception:
                self.logger.exception("Failed to handle DEPLOY_COMMAND")
                self.send_nack(envelope, "Deploy command processing failed")

        elif envelope.message_type == MessageType.RESUME_COMMAND:
            self.send_ack(envelope)
            self.logger.info("Monitoring resumed")

        else:
            self.send_nack(
                envelope,
                f"Unexpected message type: {envelope.message_type.value}",
            )

        self.transition_state(AgentState.IDLE)

    # ─── Deploy Command ───────────────────────────────────────

    def _handle_deploy_command(self, envelope: MessageEnvelope) -> None:
        """Initialize monitoring for newly deployed signals."""
        approved_signals = envelope.payload.get("approved_signals", [])

        for sig in approved_signals:
            signal_id = sig.get("signal_id", "unknown")
            self._active_signals[signal_id] = SignalHealthState(
                signal_id=signal_id,
                backtest_ic=sig.get("ic", 0.05),
                backtest_hit_rate=sig.get("hit_rate", 0.55),
                backtest_sharpe=sig.get("sharpe", 1.5),
                risk_params=sig.get("risk_parameters", {}),
            )

        self.logger.info(
            "Monitoring initialized for %d signals: %s",
            len(approved_signals),
            [s.get("signal_id", "?") for s in approved_signals],
        )

    # ─── Metric Updates ───────────────────────────────────────

    def update_metrics(
        self,
        signal_id: str,
        *,
        live_ic: float | None = None,
        live_hit_rate: float | None = None,
        live_sharpe: float | None = None,
        trades_today: int | None = None,
        gross_pnl_today: float | None = None,
        net_pnl_today: float | None = None,
        realized_slippage: float | None = None,
    ) -> None:
        """Update live metrics for a tracked signal.

        Also updates alert counter state based on threshold comparisons.
        """
        state = self._active_signals.get(signal_id)
        if state is None:
            self.logger.warning("Signal %s not tracked, ignoring update", signal_id)
            return

        if live_ic is not None:
            state.live_ic = live_ic
            # Track IC degradation counter
            if state.backtest_ic > 0 and live_ic < 0.5 * state.backtest_ic:
                state.bars_below_ic_threshold += 1
            else:
                state.bars_below_ic_threshold = 0

        if live_hit_rate is not None:
            state.live_hit_rate = live_hit_rate
            if live_hit_rate < 0.48:
                state.consecutive_windows_below_hr += 1
            else:
                state.consecutive_windows_below_hr = 0

        if live_sharpe is not None:
            state.live_sharpe = live_sharpe
            if live_sharpe < 0.5:
                state.days_below_sharpe_threshold += 1
            else:
                state.days_below_sharpe_threshold = 0

        if trades_today is not None:
            state.trades_today = trades_today
        if gross_pnl_today is not None:
            state.gross_pnl_today = gross_pnl_today
        if net_pnl_today is not None:
            state.net_pnl_today = net_pnl_today
        if realized_slippage is not None:
            state.realized_slippage = realized_slippage

    # ─── Regime ───────────────────────────────────────────────

    def update_regime(self, market_data: dict[str, Any]) -> dict[str, Any] | None:
        """Classify regime and detect transitions.

        Sends REGIME_SHIFT message to ORCH-001 if a transition is detected.

        Returns:
            Transition dict if detected, None otherwise.
        """
        new_regime, confidence = classify_regime(market_data)
        transition = detect_regime_transition(
            self._current_regime, new_regime, confidence
        )

        if transition is not None:
            self.logger.info(
                "Regime transition: %s -> %s (confidence=%.2f)",
                self._current_regime.value,
                new_regime.value,
                confidence,
            )
            self.send_message(
                receiver=AgentID.ORCHESTRATOR,
                message_type=MessageType.REGIME_SHIFT,
                payload={
                    "transition": transition,
                    "weights": compute_regime_signal_weights(new_regime),
                },
            )

        self._current_regime = new_regime
        self._regime_confidence = confidence

        return transition

    # ─── Alert Evaluation ─────────────────────────────────────

    def evaluate_alerts(self) -> list:
        """Run all alert checks across active signals and prop firm status.

        Escalates CRITICAL and HALT alerts to ORCH-001.

        Returns:
            List of triggered Alert objects.
        """
        from alpha_lab.core.contracts import Alert

        all_alerts: list[Alert] = []

        for state in self._active_signals.values():
            metrics = {
                "live_ic": state.live_ic,
                "backtest_ic": state.backtest_ic,
                "bars_below_ic": state.bars_below_ic_threshold,
                "live_hit_rate": state.live_hit_rate,
                "consecutive_hr_windows_below": state.consecutive_windows_below_hr,
                "live_sharpe": state.live_sharpe,
                "days_below_sharpe": state.days_below_sharpe_threshold,
                "dd_buffer_pct": self._prop_firm_status.get("dd_buffer_pct", 1.0),
                "daily_buffer_pct": self._prop_firm_status.get(
                    "daily_buffer_pct", 1.0
                ),
            }
            alerts = evaluate_all_alerts(metrics)
            all_alerts.extend(alerts)

        # Escalate severe alerts to orchestrator
        for alert in all_alerts:
            if alert.level in ("CRITICAL", "HALT"):
                self.escalate(
                    issue=alert.message,
                    request_id=None,
                )

        return all_alerts

    # ─── Signal Health ────────────────────────────────────────

    def check_signal_health(self, signal_id: str) -> SignalHealthReport:
        """Check current health metrics for a deployed signal.

        Args:
            signal_id: The signal to check

        Returns:
            SignalHealthReport with current metrics

        Raises:
            KeyError: If signal_id is not being tracked
        """
        state = self._active_signals.get(signal_id)
        if state is None:
            raise KeyError(f"Signal {signal_id} not tracked")

        ic_ratio = (
            state.live_ic / state.backtest_ic if state.backtest_ic > 0 else 0.0
        )

        # Determine health status
        if (
            state.bars_below_ic_threshold >= 20
            or state.days_below_sharpe_threshold >= 20
        ):
            status = SignalHealth.FAILING.value
        elif (
            state.bars_below_ic_threshold >= 5
            or state.consecutive_windows_below_hr >= 1
            or state.days_below_sharpe_threshold >= 5
        ):
            status = SignalHealth.DEGRADING.value
        else:
            status = SignalHealth.HEALTHY.value

        return SignalHealthReport(
            signal_id=signal_id,
            live_ic=state.live_ic,
            backtest_ic=state.backtest_ic,
            ic_ratio=ic_ratio,
            live_hit_rate=state.live_hit_rate,
            live_sharpe=state.live_sharpe,
            gross_pnl_today=state.gross_pnl_today,
            net_pnl_today=state.net_pnl_today,
            trades_today=state.trades_today,
            realized_slippage=state.realized_slippage,
            status=status,
        )

    # ─── Daily Report ─────────────────────────────────────────

    def generate_daily_report(self) -> MonitoringReport:
        """Generate end-of-session daily report (mandatory).

        Assembles signal health, alerts, regime, and prop firm status.
        """
        signal_health = [
            self.check_signal_health(sid)
            for sid in self._active_signals
        ]
        alerts = self.evaluate_alerts()

        regime_info = {
            "current": self._current_regime.value,
            "confidence": self._regime_confidence,
            "transition_detected": False,
        }

        return assemble_daily_report(
            signal_health=signal_health,
            alerts=alerts,
            regime=regime_info,
            prop_firm_status=self._prop_firm_status,
        )

    @property
    def active_signals(self) -> dict[str, SignalHealthState]:
        """Access tracked signals (read-only view)."""
        return dict(self._active_signals)

    @property
    def current_regime(self) -> Regime:
        """Current market regime classification."""
        return self._current_regime

    @property
    def prop_firm_status(self) -> dict[str, Any]:
        """Current prop firm risk status."""
        return dict(self._prop_firm_status)
