"""
Statistical Validation Agent (VAL-001) — Quant Reviewer.

The firewall against overfitting. Receives signal vectors as OPAQUE
NUMERICAL ARRAYS. Does not know how they were constructed.

See docs/agent_prompts/VAL-001.md for full system prompt.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from alpha_lab.agents.validation.firewall import (
    DEFAULT_THRESHOLDS,
    assemble_verdict,
    run_test_battery,
    strip_signal_metadata,
)
from alpha_lab.agents.validation.tests.decay_analysis import DecayAnalysisTest
from alpha_lab.agents.validation.tests.hit_rate import HitRateTest
from alpha_lab.agents.validation.tests.ic_testing import ICTest
from alpha_lab.agents.validation.tests.orthogonality import OrthogonalityTest
from alpha_lab.agents.validation.tests.risk_adjusted import RiskAdjustedTest
from alpha_lab.agents.validation.tests.robustness import RobustnessTest
from alpha_lab.core.agent_base import BaseAgent
from alpha_lab.core.contracts import (
    SignalBundle,
    SignalVerdict,
    ValidationReport,
)
from alpha_lab.core.enums import AgentID, AgentState, MessageType
from alpha_lab.core.message import MessageBus, MessageEnvelope

logger = logging.getLogger(__name__)

# Bonferroni alpha adjustment
_BONFERRONI_ALPHA = 0.05


class ValidationAgent(BaseAgent):
    """
    VAL-001: Quant Reviewer.

    Responsibilities:
    - Run full test battery on every signal (IC, hit rate, Sharpe, decay, orthogonality, robustness)
    - Issue DEPLOY/REFINE/REJECT verdicts per signal
    - Apply Bonferroni correction for multiple testing
    - Flag suspected look-ahead bias
    - Never share test methodology with SIG-001 (firewall)
    """

    def __init__(
        self,
        bus: MessageBus,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        super().__init__(AgentID.VALIDATION, "Statistical Validation", bus)
        self._thresholds = thresholds or dict(DEFAULT_THRESHOLDS)
        self._test_battery = [
            ICTest(),
            HitRateTest(),
            RiskAdjustedTest(),
            DecayAnalysisTest(),
            OrthogonalityTest(),
            RobustnessTest(),
        ]

    def handle_message(self, envelope: MessageEnvelope) -> None:
        """Handle SIGNAL_BUNDLE messages (opaque vectors only)."""
        self.transition_state(AgentState.PROCESSING)
        self.logger.info(
            "Received %s from %s (request_id=%s)",
            envelope.message_type.value,
            envelope.sender.value,
            envelope.request_id,
        )

        if envelope.message_type == MessageType.SIGNAL_BUNDLE:
            self.send_ack(envelope)
            try:
                bundle = SignalBundle.model_validate(envelope.payload["bundle"])
                price_data = envelope.payload.get("price_data", {})
                report = self.validate_signal_bundle(bundle, price_data)
                self.send_message(
                    receiver=AgentID.ORCHESTRATOR,
                    message_type=MessageType.VALIDATION_REPORT,
                    payload={"report": report.model_dump()},
                    request_id=envelope.request_id,
                )
            except Exception:
                self.logger.exception("Failed to validate signal bundle")
                self.send_nack(envelope, "Validation failed")
        else:
            self.send_nack(
                envelope,
                f"Unexpected message type: {envelope.message_type.value}",
            )

        self.transition_state(AgentState.IDLE)

    def validate_signal_bundle(
        self,
        bundle: SignalBundle,
        price_data: dict[str, Any] | None = None,
    ) -> ValidationReport:
        """
        Run the full test battery on a SignalBundle.

        Args:
            bundle: SignalBundle from SIG-001
            price_data: Dict of timeframe -> pd.DataFrame with OHLCV data.
                        Needed for forward return computation.

        Returns:
            ValidationReport with verdicts for every signal
        """
        price_data = price_data or {}
        verdicts: list[SignalVerdict] = []

        # Bonferroni: adjust thresholds for multiple signals
        n_signals = len(bundle.signals)
        thresholds = self._apply_bonferroni(n_signals)

        for signal in bundle.signals:
            # Firewall: strip implementation details
            _ = strip_signal_metadata(signal)

            # Get price data for this signal's timeframe
            bars = price_data.get(signal.timeframe)
            if not isinstance(bars, pd.DataFrame) or bars.empty:
                logger.warning(
                    "No price data for timeframe %s, skipping %s",
                    signal.timeframe,
                    signal.signal_id,
                )
                continue

            # Run test battery
            test_results = run_test_battery(signal, bars, self._test_battery)

            # Assemble verdict
            verdict = assemble_verdict(signal.signal_id, test_results, thresholds)

            # Look-ahead bias check: IC > 0.20 is suspicious
            ic = test_results.get("information_coefficient", {}).get("ic_mean", 0)
            if abs(ic) > 0.20:
                logger.warning(
                    "Suspected look-ahead bias for %s: IC=%.4f",
                    signal.signal_id,
                    ic,
                )
                verdict = SignalVerdict(
                    **{**verdict.model_dump(), "verdict": "REJECT"},
                )

            verdicts.append(verdict)

        deploy_count = sum(1 for v in verdicts if v.verdict == "DEPLOY")
        refine_count = sum(1 for v in verdicts if v.verdict == "REFINE")
        reject_count = sum(1 for v in verdicts if v.verdict == "REJECT")

        return ValidationReport(
            request_id=bundle.request_id,
            signal_bundle_id=bundle.request_id,
            verdicts=verdicts,
            deploy_count=deploy_count,
            refine_count=refine_count,
            reject_count=reject_count,
            bonferroni_adjusted=n_signals > 1,
            overall_assessment=_build_assessment(deploy_count, refine_count, reject_count),
            recommended_composites=[
                v.signal_id for v in verdicts if v.verdict == "DEPLOY"
            ],
            correlation_matrix={},
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _apply_bonferroni(self, n_signals: int) -> dict[str, float]:
        """Adjust IC t-stat threshold for multiple testing."""
        thresholds = dict(self._thresholds)
        if n_signals > 1:
            # Bonferroni: raise the t-stat bar
            from scipy.stats import norm

            adjusted_alpha = _BONFERRONI_ALPHA / n_signals
            adjusted_tstat = float(norm.ppf(1 - adjusted_alpha / 2))
            thresholds["ic_tstat_min"] = max(
                thresholds.get("ic_tstat_min", 2.0), adjusted_tstat
            )
        return thresholds


def _build_assessment(deploy: int, refine: int, reject: int) -> str:
    """Build a summary assessment string."""
    total = deploy + refine + reject
    if total == 0:
        return "No signals validated (missing price data)"
    parts = []
    if deploy > 0:
        parts.append(f"{deploy} DEPLOY")
    if refine > 0:
        parts.append(f"{refine} REFINE")
    if reject > 0:
        parts.append(f"{reject} REJECT")
    return f"{total} signals evaluated: {', '.join(parts)}"
