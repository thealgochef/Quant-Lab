"""Results presentation (UI-3; plan §7 "Results") — pure.

The selected configuration's strategy metrics read against the charter's
RESOLVED strategy gates, its prop vector (the worst value across the
simulated firms, D-#18) read against the resolved prop gates, and the
explorer column guide built from the metric registry. The thresholds are
the frozen charter's — every reading carries the ``proposed`` caveat unless
the caller states the gates are owner-ratified — and no threshold is
invented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..search.charter import OBJECTIVE_DIRECTIONS
from .metric_registry import (
    MetricReading,
    describe,
    evaluate_interval,
    evaluate_metric,
)

__all__ = [
    "column_guide",
    "prop_readings",
    "strategy_readings",
    "worst_vector_value",
]


def _field(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


_STRATEGY_GATED: tuple[tuple[str, str], ...] = (
    ("executed_trades", "min_executed_trades"),
    ("independent_days", "min_independent_days"),
    ("net_expectancy_r", "min_net_expectancy_r"),
    ("profit_factor", "min_profit_factor"),
    ("max_drawdown_r", "max_drawdown_r"),
    ("time_under_water_days", "max_time_under_water_days"),
    ("session_stability_score", "min_session_stability_score"),
    ("time_block_sign_consistency", "min_time_block_sign_consistency"),
    ("top_day_pnl_share", "max_top_day_pnl_share"),
    ("top_setup_pnl_share", "max_top_setup_pnl_share"),
)


def strategy_readings(
    metrics: Any, gates: Any, *, proposed: bool = True
) -> tuple[MetricReading, ...]:
    """The strategy metrics of one configuration against the resolved gates.

    ``metrics`` is a ``StrategyMetrics`` (or mapping) — ``None`` yields
    UNAVAILABLE readings; ``gates`` is the charter's
    ``ResolvedStrategyGateThresholds`` (or mapping). The bootstrap interval
    is read only when the charter requires it to exclude zero.
    """

    readings: list[MetricReading] = []
    for key, gate_field in _STRATEGY_GATED:
        gate = _field(gates, gate_field)
        readings.append(
            evaluate_metric(
                key,
                _field(metrics, key),
                gate=None if gate is None else float(gate),
                proposed=proposed,
            )
        )
    if bool(_field(gates, "require_bootstrap_ci_excludes_zero")):
        interval = _field(metrics, "net_expectancy_bootstrap_ci95")
        lower, upper = (interval if interval else (None, None))
        readings.append(
            evaluate_interval(
                "net_expectancy_bootstrap_ci95",
                lower=lower,
                upper=upper,
                available=interval is not None,
                reason=None if interval is not None else "no cluster-bootstrap interval persisted",
            )
        )
    return tuple(readings)


def worst_vector_value(
    summaries: Mapping[str, Mapping[str, Any]], attribute: str, *, worst: bool = True
) -> float | None:
    """The worst-across-simulations value of one vector attribute (D-#18):
    the minimum for a maximise metric, the maximum for a minimise metric."""

    values = [
        value
        for value in (
            dict(summary.get("payout_reliability_vector") or {}).get(attribute)
            for summary in summaries.values()
        )
        if value is not None
    ]
    if not values:
        return None
    if not worst:
        return float(values[0])
    direction = OBJECTIVE_DIRECTIONS.get(attribute, "maximize")
    return float(min(values) if direction == "maximize" else max(values))


_PROP_GATED: tuple[tuple[str, str], ...] = (
    ("first_payout_probability_60d", "minimum_first_payout_probability_60d"),
    ("three_payout_probability", "minimum_three_payout_probability"),
    ("expected_net_payout_90d", "minimum_expected_net_payout_90d"),
    ("p10_net_payout_90d", "minimum_p10_net_payout_90d"),
    ("p90_payout_drought_days", "maximum_p90_payout_drought_days"),
    ("breach_probability_90d", "maximum_breach_probability_90d"),
)
_PROP_UNGATED: tuple[str, ...] = (
    "payout_probability_per_rolling_30d",
    "expected_replacement_cost",
    "median_account_lifetime_days",
)


def prop_readings(
    summaries: Mapping[str, Mapping[str, Any]], gates: Any, *, proposed: bool = True
) -> tuple[MetricReading, ...]:
    """The prop vector (worst across firms) against the resolved prop gates;
    empty summaries yield UNAVAILABLE readings, never a pass."""

    readings: list[MetricReading] = []
    for key, gate_field in _PROP_GATED:
        gate = _field(gates, gate_field)
        readings.append(
            evaluate_metric(
                key,
                worst_vector_value(summaries, key),
                gate=None if gate is None else float(gate),
                proposed=proposed,
            )
        )
    for key in _PROP_UNGATED:
        readings.append(evaluate_metric(key, worst_vector_value(summaries, key)))
    breach = worst_vector_value(summaries, "breach_probability_90d")
    readings.append(
        evaluate_metric("survival_probability_90d", None if breach is None else 1.0 - breach)
    )
    return tuple(readings)


_DIRECTION_WORDS = {
    "higher_better": "higher is better",
    "lower_better": "lower is better",
    "target": "closer to the target is better",
    "descriptive": "descriptive — no direction",
}


def column_guide(columns: Mapping[str, str | None]) -> list[dict[str, str]]:
    """One guide row per explorer column: the registry's human name,
    definition, direction and unit (or the honest 'not persisted' note)."""

    rows: list[dict[str, str]] = []
    for column, key in columns.items():
        if key is None:
            rows.append(
                {
                    "column": column,
                    "metric": "—",
                    "definition": (
                        "schema-reserved or contextual column — not persisted as a registry "
                        "metric by the current evaluation"
                    ),
                    "direction": "—",
                    "unit": "—",
                }
            )
            continue
        spec = describe(key)
        rows.append(
            {
                "column": column,
                "metric": spec.human_name,
                "definition": spec.definition,
                "direction": _DIRECTION_WORDS[spec.directionality],
                "unit": spec.unit,
            }
        )
    return rows
