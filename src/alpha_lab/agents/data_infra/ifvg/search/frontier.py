"""Deterministic non-dominated frontier + lexicographic tie-breaks (CS §12).

The selected interior configuration is a **Development Exploratory
Representative** — a publishable representative requires a separately frozen,
owner-approved outer evaluation protocol (owner decision 15). The resolved
objective policy lives in the charter identity; this module is pure.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .charter import OBJECTIVE_DIRECTIONS
from .identities import FrozenContract, ImmutableMap

__all__ = ["ObjectiveSpec", "FrontierResult", "build_frontier"]


class ObjectiveSpec(FrozenContract):
    metric: str
    direction: Literal["maximize", "minimize"]


class FrontierResult(FrozenContract):
    feasible_ids: tuple[str, ...]
    frontier_ids: tuple[str, ...]
    dominance_edges: tuple[tuple[str, str], ...]  # (dominator, dominated)
    development_exploratory_representative_id: str | None
    per_objective_champions: ImmutableMap[str, str]
    tie_break_trace: tuple[str, ...]


def _dominates(
    left: Mapping[str, float], right: Mapping[str, float], objectives: tuple[ObjectiveSpec, ...]
) -> bool:
    at_least_as_good = True
    strictly_better = False
    for objective in objectives:
        l_value = left[objective.metric]
        r_value = right[objective.metric]
        better = l_value > r_value if objective.direction == "maximize" else l_value < r_value
        worse = l_value < r_value if objective.direction == "maximize" else l_value > r_value
        if worse:
            at_least_as_good = False
            break
        if better:
            strictly_better = True
    return at_least_as_good and strictly_better


def build_frontier(
    feasible_metrics: Mapping[str, Mapping[str, float]],
    *,
    objectives: tuple[ObjectiveSpec, ...],
    lexicographic_tie_breaks: tuple[str, ...],
) -> FrontierResult:
    """O(n²) dominance in sorted-id order with a persisted tie-break trace.

    ``feasible_metrics`` maps child id → objective values (feasibility gates
    already applied by the caller — infeasible children never enter).
    """

    ids = tuple(sorted(feasible_metrics))
    for child_id in ids:
        missing = [o.metric for o in objectives if o.metric not in feasible_metrics[child_id]]
        if missing:
            raise ValueError(f"child {child_id[:12]}… lacks objective metrics {missing}")
    edges: list[tuple[str, str]] = []
    dominated: set[str] = set()
    for left in ids:
        for right in ids:
            if left == right:
                continue
            if _dominates(feasible_metrics[left], feasible_metrics[right], objectives):
                edges.append((left, right))
                dominated.add(right)
    frontier = tuple(child_id for child_id in ids if child_id not in dominated)

    champions: dict[str, str] = {}
    for objective in objectives:
        best = None
        for child_id in ids:
            value = feasible_metrics[child_id][objective.metric]
            if best is None:
                best = (value, child_id)
                continue
            better = (
                value > best[0] if objective.direction == "maximize" else value < best[0]
            )
            if better:
                best = (value, child_id)
        if best is not None:
            champions[objective.metric] = best[1]

    trace: list[str] = []
    representative = None
    if frontier:
        pool = list(frontier)
        trace.append(f"frontier candidates: {len(pool)}")
        for tie_break in lexicographic_tie_breaks:
            if len(pool) == 1:
                break
            if tie_break == "core_replay_id":
                pool.sort()
                trace.append("tie-break core_replay_id: lexicographic minimum")
                pool = pool[:1]
                continue
            values = {
                child_id: feasible_metrics[child_id].get(tie_break)
                for child_id in pool
            }
            if any(value is None for value in values.values()):
                trace.append(f"tie-break {tie_break}: unavailable, skipped")
                continue
            direction = OBJECTIVE_DIRECTIONS[tie_break]
            select_best = min if direction == "minimize" else max
            best_value = select_best(values.values())
            pool = sorted(cid for cid, value in values.items() if value == best_value)
            trace.append(
                f"tie-break {tie_break}: {direction}={best_value:g}, remaining={len(pool)}"
            )
        if len(pool) > 1:
            pool.sort()
            trace.append("final tie-break: lexicographic minimum id")
            pool = pool[:1]
        representative = pool[0]
        trace.append(
            "selected Development Exploratory Representative "
            f"{representative[:12]}… (development lane only; an owner-approved "
            "outer evaluation protocol is required before any publication "
            "claim)"
        )
    return FrontierResult(
        feasible_ids=ids,
        frontier_ids=frontier,
        dominance_edges=tuple(edges),
        development_exploratory_representative_id=representative,
        per_objective_champions=champions,
        tie_break_trace=tuple(trace),
    )
