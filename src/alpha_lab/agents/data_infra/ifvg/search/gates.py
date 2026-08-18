"""Strategy-gate evaluation with human-explainable failures (§3.4; §13 UI).

Research gates evaluate ONLY in research scope — verification fixtures use
``verification_control_flow_gates_v1`` (search/verification.py) and calling
this module on a verification run is a programming error the caller guards.
Every threshold value remains a proposal until owner-ratified (decision 7);
the resolved values live in the frozen charter identity.
"""

from __future__ import annotations

from .charter import ResolvedStrategyGateThresholds
from .failure import FailureReason
from .identities import FrozenContract
from .strategy_metrics import StrategyMetrics

__all__ = ["GateCheck", "StrategyGateReport", "evaluate_strategy_gates"]


class GateCheck(FrozenContract):
    gate_id: str
    threshold: float | int | bool
    observed: float | int | None
    passed: bool
    explanation: str


class StrategyGateReport(FrozenContract):
    passed: bool
    checks: tuple[GateCheck, ...]
    failure_reason: FailureReason | None
    human_explanation: str


def _check(
    gate_id: str,
    threshold,
    observed,
    passed: bool,
    explanation: str,
) -> GateCheck:
    return GateCheck(
        gate_id=gate_id,
        threshold=threshold,
        observed=observed,
        passed=passed,
        explanation=explanation,
    )


def evaluate_strategy_gates(
    metrics: StrategyMetrics,
    thresholds: ResolvedStrategyGateThresholds,
) -> StrategyGateReport:
    checks: list[GateCheck] = []

    def _minimum(gate_id: str, observed, threshold, unit: str, reason_word: str):
        missing = observed is None
        passed = (not missing) and observed >= threshold
        checks.append(
            _check(
                gate_id,
                threshold,
                observed,
                passed,
                (
                    f"{reason_word} unavailable (no resolved trades)"
                    if missing
                    else f"{reason_word} {observed:g}{unit} vs required ≥ {threshold:g}{unit}"
                ),
            )
        )

    def _maximum(gate_id: str, observed, threshold, unit: str, reason_word: str):
        missing = observed is None
        # a cap over NOTHING holds (zero trades already fails the min gates),
        # but a cap that could not be measured while trades exist fails closed
        passed = (
            (missing and metrics.executed_trades == 0)
            or (not missing and observed <= threshold)
        )
        checks.append(
            _check(
                gate_id,
                threshold,
                observed,
                passed,
                (
                    (
                        f"{reason_word} unmeasurable with "
                        f"{metrics.executed_trades} executed trades (fail-closed)"
                        if metrics.executed_trades
                        else f"{reason_word} unavailable (no resolved trades)"
                    )
                    if missing
                    else f"{reason_word} {observed:g}{unit} vs allowed ≤ {threshold:g}{unit}"
                ),
            )
        )

    _minimum(
        "min_executed_trades",
        metrics.executed_trades,
        thresholds.min_executed_trades,
        " trades",
        "executed trades",
    )
    _minimum(
        "min_independent_days",
        metrics.independent_days,
        thresholds.min_independent_days,
        " days",
        "independent trading days",
    )
    _minimum(
        "min_net_expectancy_r",
        metrics.net_expectancy_r,
        thresholds.min_net_expectancy_r,
        "R",
        "net expectancy",
    )
    _minimum(
        "min_profit_factor",
        metrics.profit_factor,
        thresholds.min_profit_factor,
        "",
        "profit factor",
    )
    _maximum(
        "max_drawdown_r",
        metrics.max_drawdown_r,
        thresholds.max_drawdown_r,
        "R",
        "max drawdown",
    )
    _maximum(
        "max_time_under_water_days",
        metrics.time_under_water_days,
        thresholds.max_time_under_water_days,
        " days",
        "time under water",
    )
    _minimum(
        "min_session_stability_score",
        metrics.session_stability_score,
        thresholds.min_session_stability_score,
        "",
        "session stability",
    )
    _minimum(
        "min_time_block_sign_consistency",
        metrics.time_block_sign_consistency,
        thresholds.min_time_block_sign_consistency,
        "",
        "time-block sign consistency",
    )
    _maximum(
        "max_top_day_pnl_share",
        metrics.top_day_pnl_share,
        thresholds.max_top_day_pnl_share,
        "",
        "top-day PnL share",
    )
    _maximum(
        "max_top_setup_pnl_share",
        metrics.top_setup_pnl_share,
        thresholds.max_top_setup_pnl_share,
        "",
        "top-setup PnL share",
    )

    ci = metrics.net_expectancy_bootstrap_ci95
    if not thresholds.require_bootstrap_ci_excludes_zero:
        checks.append(
            _check(
                "require_bootstrap_ci_excludes_zero",
                False,
                None,
                True,
                "bootstrap-CI exclusion of zero is not required by this charter",
            )
        )
    elif ci is None:
        checks.append(
            _check(
                "require_bootstrap_ci_excludes_zero",
                True,
                None,
                False,
                "bootstrap CI unavailable (fewer than two trading days)",
            )
        )
    else:
        low, high = ci
        excludes = low > 0.0 or high < 0.0
        checks.append(
            _check(
                "require_bootstrap_ci_excludes_zero",
                True,
                low,
                excludes,
                f"net-expectancy 95% CI [{low:+.3f}R, {high:+.3f}R] "
                + ("excludes zero" if excludes else "straddles zero"),
            )
        )

    failures = [check for check in checks if not check.passed]
    if not failures:
        return StrategyGateReport(
            passed=True,
            checks=tuple(checks),
            failure_reason=None,
            human_explanation="all strategy feasibility gates passed",
        )
    first = failures[0]
    reason_by_gate = {
        "min_executed_trades": FailureReason.INSUFFICIENT_TRADES,
        "min_independent_days": FailureReason.INSUFFICIENT_DAYS,
        "min_net_expectancy_r": FailureReason.NEGATIVE_EXPECTANCY,
        "min_profit_factor": FailureReason.NEGATIVE_EXPECTANCY,
        "max_drawdown_r": FailureReason.DRAWDOWN,
        "max_time_under_water_days": FailureReason.DRAWDOWN,
        "min_session_stability_score": FailureReason.KNIFE_EDGE,
        "min_time_block_sign_consistency": FailureReason.KNIFE_EDGE,
        "max_top_day_pnl_share": FailureReason.KNIFE_EDGE,
        "max_top_setup_pnl_share": FailureReason.KNIFE_EDGE,
        "require_bootstrap_ci_excludes_zero": FailureReason.NEGATIVE_EXPECTANCY,
    }
    return StrategyGateReport(
        passed=False,
        checks=tuple(checks),
        failure_reason=reason_by_gate.get(first.gate_id, FailureReason.INVARIANT),
        human_explanation=(
            f"failed {len(failures)} of {len(checks)} strategy gates; first: "
            f"{first.gate_id} — {first.explanation}"
        ),
    )
