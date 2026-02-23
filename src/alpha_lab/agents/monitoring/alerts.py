"""
Alert generation and routing per architecture spec Section 7.1.

Alert Levels:
- INFO: Metric within normal range but trending toward threshold
- WARNING: Single metric breached threshold, others still healthy
- CRITICAL: Multiple metrics breached OR prop firm buffer < 30%
- HALT: Immediate stop trading required (circuit breaker)

Key Triggers:
- IC drops below 50% of backtest IC for 20+ bars -> WARNING
- Hit rate below 48% for 2+ consecutive windows -> WARNING
- Net Sharpe below 0.5 for 20+ days -> CRITICAL
- Prop firm DD buffer < 30% -> CRITICAL
- Daily loss approaching 80% of limit -> HALT
"""

from __future__ import annotations

from datetime import UTC, datetime

from alpha_lab.core.contracts import Alert
from alpha_lab.core.enums import AlertLevel

# ─── Thresholds ───────────────────────────────────────────────

IC_RATIO_THRESHOLD = 0.50  # Live IC must be >= 50% of backtest IC
IC_BARS_THRESHOLD = 20  # Must be below for 20+ bars

HIT_RATE_THRESHOLD = 0.48  # Minimum acceptable hit rate
HIT_RATE_WINDOWS_THRESHOLD = 2  # Must be below for 2+ consecutive windows

SHARPE_THRESHOLD = 0.5  # Minimum acceptable net Sharpe
SHARPE_DAYS_THRESHOLD = 20  # Must be below for 20+ days

DD_BUFFER_CRITICAL = 0.30  # DD buffer < 30% → CRITICAL
DAILY_BUFFER_HALT = 0.20  # Daily loss buffer < 20% → HALT
DAILY_BUFFER_WARNING = 0.80  # Daily loss buffer < 80% → WARNING


def check_ic_degradation(
    live_ic: float, backtest_ic: float, bars_below: int
) -> Alert | None:
    """Check if IC has dropped below 50% of backtest for 20+ bars.

    Returns WARNING alert if triggered, None otherwise.
    """
    if backtest_ic <= 0:
        return None

    ic_ratio = live_ic / backtest_ic
    if ic_ratio < IC_RATIO_THRESHOLD and bars_below >= IC_BARS_THRESHOLD:
        return Alert(
            level=AlertLevel.WARNING.value,
            metric="ic_degradation",
            current_value=live_ic,
            threshold=backtest_ic * IC_RATIO_THRESHOLD,
            backtest_value=backtest_ic,
            message=(
                f"IC dropped to {live_ic:.4f} "
                f"({ic_ratio:.0%} of backtest {backtest_ic:.4f}) "
                f"for {bars_below} bars"
            ),
            recommended_action=(
                "Review signal parameters or reduce position size"
            ),
            timestamp=datetime.now(UTC).isoformat(),
        )
    return None


def check_hit_rate_degradation(
    live_hit_rate: float, consecutive_windows_below: int
) -> Alert | None:
    """Check if hit rate below 48% for 2+ consecutive windows.

    Returns WARNING alert if triggered, None otherwise.
    """
    if (
        live_hit_rate < HIT_RATE_THRESHOLD
        and consecutive_windows_below >= HIT_RATE_WINDOWS_THRESHOLD
    ):
        return Alert(
            level=AlertLevel.WARNING.value,
            metric="hit_rate_degradation",
            current_value=live_hit_rate,
            threshold=HIT_RATE_THRESHOLD,
            backtest_value=0.0,
            message=(
                f"Hit rate {live_hit_rate:.1%} below {HIT_RATE_THRESHOLD:.0%} "
                f"for {consecutive_windows_below} consecutive windows"
            ),
            recommended_action=(
                "Signal may be degrading, consider pausing new entries"
            ),
            timestamp=datetime.now(UTC).isoformat(),
        )
    return None


def check_sharpe_degradation(
    live_sharpe: float, days_below: int
) -> Alert | None:
    """Check if net Sharpe below 0.5 for 20+ days.

    Returns CRITICAL alert if triggered, None otherwise.
    """
    if live_sharpe < SHARPE_THRESHOLD and days_below >= SHARPE_DAYS_THRESHOLD:
        return Alert(
            level=AlertLevel.CRITICAL.value,
            metric="sharpe_degradation",
            current_value=live_sharpe,
            threshold=SHARPE_THRESHOLD,
            backtest_value=0.0,
            message=(
                f"Net Sharpe {live_sharpe:.2f} below {SHARPE_THRESHOLD} "
                f"for {days_below} days"
            ),
            recommended_action=(
                "Signal failing profitability threshold, halt deployment"
            ),
            timestamp=datetime.now(UTC).isoformat(),
        )
    return None


def check_prop_firm_buffer(
    dd_buffer_pct: float, daily_buffer_pct: float
) -> Alert | None:
    """Check prop firm drawdown and daily loss buffers.

    Priority: HALT (daily buffer < 20%) > CRITICAL (DD buffer < 30%)
              > WARNING (daily buffer < 80%).
    Returns the highest-severity alert triggered, or None.
    """
    now = datetime.now(UTC).isoformat()

    # Most severe first: daily loss approaching limit
    if daily_buffer_pct < DAILY_BUFFER_HALT:
        return Alert(
            level=AlertLevel.HALT.value,
            metric="daily_loss_buffer",
            current_value=daily_buffer_pct,
            threshold=DAILY_BUFFER_HALT,
            backtest_value=0.0,
            message=(
                f"Daily loss buffer at {daily_buffer_pct:.0%} "
                f"(limit: {DAILY_BUFFER_HALT:.0%}). "
                f"Immediate stop required."
            ),
            recommended_action="HALT all trading immediately",
            timestamp=now,
        )

    # DD buffer critically low
    if dd_buffer_pct < DD_BUFFER_CRITICAL:
        return Alert(
            level=AlertLevel.CRITICAL.value,
            metric="dd_buffer",
            current_value=dd_buffer_pct,
            threshold=DD_BUFFER_CRITICAL,
            backtest_value=0.0,
            message=(
                f"Drawdown buffer at {dd_buffer_pct:.0%} "
                f"(critical threshold: {DD_BUFFER_CRITICAL:.0%})"
            ),
            recommended_action="Reduce exposure by 50% immediately",
            timestamp=now,
        )

    # Daily buffer approaching limit (but not yet critical)
    if daily_buffer_pct < DAILY_BUFFER_WARNING:
        return Alert(
            level=AlertLevel.WARNING.value,
            metric="daily_loss_buffer",
            current_value=daily_buffer_pct,
            threshold=DAILY_BUFFER_WARNING,
            backtest_value=0.0,
            message=(
                f"Daily loss buffer at {daily_buffer_pct:.0%} "
                f"(warning threshold: {DAILY_BUFFER_WARNING:.0%})"
            ),
            recommended_action="Monitor closely, consider reducing position size",
            timestamp=now,
        )

    return None


def evaluate_all_alerts(metrics: dict) -> list[Alert]:
    """Run all alert checks and return triggered alerts, sorted by severity.

    Args:
        metrics: Dict with keys:
            - live_ic, backtest_ic, bars_below_ic
            - live_hit_rate, consecutive_hr_windows_below
            - live_sharpe, days_below_sharpe
            - dd_buffer_pct, daily_buffer_pct

    Returns:
        List of Alert objects sorted by severity (HALT > CRITICAL > WARNING > INFO)
    """
    alerts: list[Alert] = []

    alert = check_ic_degradation(
        live_ic=metrics.get("live_ic", 0.0),
        backtest_ic=metrics.get("backtest_ic", 0.05),
        bars_below=metrics.get("bars_below_ic", 0),
    )
    if alert is not None:
        alerts.append(alert)

    alert = check_hit_rate_degradation(
        live_hit_rate=metrics.get("live_hit_rate", 0.5),
        consecutive_windows_below=metrics.get("consecutive_hr_windows_below", 0),
    )
    if alert is not None:
        alerts.append(alert)

    alert = check_sharpe_degradation(
        live_sharpe=metrics.get("live_sharpe", 0.0),
        days_below=metrics.get("days_below_sharpe", 0),
    )
    if alert is not None:
        alerts.append(alert)

    alert = check_prop_firm_buffer(
        dd_buffer_pct=metrics.get("dd_buffer_pct", 1.0),
        daily_buffer_pct=metrics.get("daily_buffer_pct", 1.0),
    )
    if alert is not None:
        alerts.append(alert)

    # Sort by severity: HALT=0, CRITICAL=1, WARNING=2, INFO=3
    level_order = {
        AlertLevel.HALT.value: 0,
        AlertLevel.CRITICAL.value: 1,
        AlertLevel.WARNING.value: 2,
        AlertLevel.INFO.value: 3,
    }
    alerts.sort(key=lambda a: level_order.get(a.level, 99))

    return alerts
