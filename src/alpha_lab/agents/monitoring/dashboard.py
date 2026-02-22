"""
Dashboard data assembly for the monitoring agent.

Assembles all monitoring data into structured reports
for the Orchestrator's daily summary consumption.
"""

from __future__ import annotations

from typing import Any

from alpha_lab.core.contracts import Alert, MonitoringReport, SignalHealthReport


def assemble_daily_report(
    signal_health: list[SignalHealthReport],
    alerts: list[Alert],
    regime: dict[str, Any],
    prop_firm_status: dict[str, Any],
) -> MonitoringReport:
    """Assemble all monitoring data into a MonitoringReport."""
    raise NotImplementedError


def assemble_realtime_update(
    signal_health: list[SignalHealthReport],
    active_alerts: list[Alert],
) -> MonitoringReport:
    """Assemble a real-time monitoring update."""
    raise NotImplementedError
