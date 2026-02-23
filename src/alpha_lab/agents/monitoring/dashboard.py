"""
Dashboard data assembly for the monitoring agent.

Assembles all monitoring data into structured reports
for the Orchestrator's daily summary consumption.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from alpha_lab.core.contracts import Alert, MonitoringReport, SignalHealthReport
from alpha_lab.core.enums import AlertLevel


def assemble_daily_report(
    signal_health: list[SignalHealthReport],
    alerts: list[Alert],
    regime: dict[str, Any],
    prop_firm_status: dict[str, Any],
) -> MonitoringReport:
    """Assemble all monitoring data into a MonitoringReport.

    Auto-generates recommendations based on alert levels.
    """
    recommendations = _generate_recommendations(alerts, signal_health)

    return MonitoringReport(
        request_id=str(uuid.uuid4()),
        report_type="DAILY_SUMMARY",
        signals=signal_health,
        alerts=alerts,
        regime=regime,
        prop_firm_status=prop_firm_status,
        recommendations=recommendations,
        timestamp=datetime.now(UTC).isoformat(),
    )


def assemble_realtime_update(
    signal_health: list[SignalHealthReport],
    active_alerts: list[Alert],
) -> MonitoringReport:
    """Assemble a real-time monitoring update.

    Lightweight version without prop firm status or regime detail.
    """
    recommendations = _generate_recommendations(active_alerts, signal_health)

    return MonitoringReport(
        request_id=str(uuid.uuid4()),
        report_type="REALTIME",
        signals=signal_health,
        alerts=active_alerts,
        regime={},
        prop_firm_status={},
        recommendations=recommendations,
        timestamp=datetime.now(UTC).isoformat(),
    )


def _generate_recommendations(
    alerts: list[Alert],
    signal_health: list[SignalHealthReport],
) -> list[str]:
    """Generate actionable recommendations from alerts and signal health."""
    recommendations: list[str] = []

    # Check alert severity levels
    levels = {a.level for a in alerts}

    if AlertLevel.HALT.value in levels:
        recommendations.append("HALT all trading immediately")
    if AlertLevel.CRITICAL.value in levels:
        recommendations.append("Reduce exposure by 50%")
    if AlertLevel.WARNING.value in levels:
        recommendations.append("Monitor degrading metrics closely")

    # Check signal health statuses
    failing = [s for s in signal_health if s.status == "FAILING"]
    degrading = [s for s in signal_health if s.status == "DEGRADING"]

    if failing:
        ids = ", ".join(s.signal_id for s in failing)
        recommendations.append(f"Consider removing failing signals: {ids}")
    if degrading:
        ids = ", ".join(s.signal_id for s in degrading)
        recommendations.append(f"Review degrading signals: {ids}")

    if not recommendations:
        recommendations.append("Continue normal operations")

    return recommendations
