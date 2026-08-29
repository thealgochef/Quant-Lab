"""``stratified_frontier`` — child × regime × metric over the pooled frontier (R6.1 §6.G).

Per child, the ``cohort_descriptive`` strata already built for it are laid
out against the frontier's objective metrics. ``on_frontier`` is READ from
the persisted pooled frontier only — it is never recomputed per stratum, so
a stratified view can never move a child onto or off the frontier
(``frontier_role="descriptive_view_never_selection_input"``).
"""

from __future__ import annotations

from collections.abc import Mapping

from ..search.orchestrator import SearchFrontierEnvelope
from .regime_stratified_contracts import (
    CohortDescriptiveBody,
    FrontierStratumCell,
    StratifiedFrontierBody,
)

__all__ = ["build_stratified_frontier_body"]

#: StrategyMetrics scalar fields that a frontier objective may name.
_METRIC_FIELDS = (
    "executed_trades",
    "independent_days",
    "gross_expectancy_r",
    "net_expectancy_r",
    "profit_factor",
    "max_drawdown_r",
    "time_under_water_days",
    "time_block_sign_consistency",
    "session_stability_score",
    "top_day_pnl_share",
    "top_setup_pnl_share",
)


def build_stratified_frontier_body(
    *,
    frontier: SearchFrontierEnvelope,
    strategy_bodies: Mapping[str, CohortDescriptiveBody],
    objective_metrics: tuple[str, ...],
) -> StratifiedFrontierBody:
    """Cells for every (feasible child with strata) × stratum × objective."""

    metrics = tuple(objective_metrics)
    if not metrics:
        raise ValueError("stratified_frontier requires at least one objective metric")
    result = frontier.payload.frontier
    feasible = tuple(sorted(result.feasible_ids))
    frontier_ids = set(result.frontier_ids)
    children = tuple(child for child in feasible if child in strategy_bodies)
    without = tuple(child for child in feasible if child not in strategy_bodies)
    cells: list[FrontierStratumCell] = []
    works: dict[str, tuple[int, ...]] = {}
    for child in children:
        body = strategy_bodies[child]
        if body.core_replay_id != child:
            raise ValueError("a strategy body names another core replay id than its key")
        works[child] = tuple(body.works_only_in_regime)
        for row in body.strata:
            values: dict[str, float | None] = {}
            for metric in metrics:
                if row.metrics is None:
                    values[metric] = None
                elif metric in _METRIC_FIELDS:
                    raw = getattr(row.metrics, metric)
                    values[metric] = None if raw is None else float(raw)
                else:
                    values[metric] = None
            cells.append(
                FrontierStratumCell(
                    core_replay_id=child,
                    key=row.key,
                    executed_trades=row.executed_trades,
                    typed_state=row.typed_state,
                    metric_values=values,
                )
            )
    return StratifiedFrontierBody(
        frontier_id=frontier.frontier_id,
        objective_metrics=metrics,
        children=children,
        on_frontier={child: child in frontier_ids for child in children},
        cells=tuple(cells),
        works_only_in_regime_by_child=works,
        children_without_strata=without,
    )
