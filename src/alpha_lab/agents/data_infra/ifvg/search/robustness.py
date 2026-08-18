"""Neighbor stability, plateau width, knife-edge warnings (CS §12; brief §9.18).

``outer_fold_recurrence`` remains schema-reserved (exploratory lane: null).
"""

from __future__ import annotations

from collections.abc import Mapping

from .identities import FrozenContract, ImmutableMap

__all__ = ["NeighborCheck", "RobustnessReport", "evaluate_robustness"]


class NeighborCheck(FrozenContract):
    child_id: str
    neighbor_id: str
    axis: str
    metric: str
    child_value: float
    neighbor_value: float
    degradation: float


class RobustnessReport(FrozenContract):
    child_id: str
    neighbor_checks: tuple[NeighborCheck, ...]
    worst_neighbor_degradation_r: float | None
    plateau_width: int | None
    knife_edge: bool
    knife_edge_warnings: tuple[str, ...]
    per_axis_plateau: ImmutableMap[str, int]
    outer_fold_recurrence: None = None  # schema-reserved (owner decision 15)


def evaluate_robustness(
    child_id: str,
    *,
    axis_grids: Mapping[str, tuple[str, ...]],
    child_positions: Mapping[str, Mapping[str, str]],
    metric_by_child: Mapping[str, float],
    metric_name: str = "net_expectancy_r",
    degradation_warning_r: float | None = None,
) -> RobustnessReport:
    """±1-step neighbor stability on the declared axis grids.

    ``axis_grids`` maps axis → ORDERED registered value ids; ``child_positions``
    maps child id → {axis: value id}. A neighbor is the child at exactly one
    axis step away with every other axis identical; missing neighbors are not
    fabricated. Plateau width per axis counts the contiguous run of
    same-or-better neighbors around the child (including itself).
    """

    position = child_positions.get(child_id)
    if position is None:
        raise ValueError(f"unknown child {child_id[:12]}…")
    child_value = metric_by_child.get(child_id)
    if child_value is None:
        raise ValueError(f"child {child_id[:12]}… lacks metric {metric_name}")

    index_of: dict[tuple[tuple[str, str], ...], str] = {}
    for cid, pos in child_positions.items():
        index_of[tuple(sorted(pos.items()))] = cid

    checks: list[NeighborCheck] = []
    per_axis_plateau: dict[str, int] = {}
    warnings: list[str] = []
    for axis, grid in sorted(axis_grids.items()):
        value = position.get(axis)
        if value is None or value not in grid:
            continue
        index = grid.index(value)
        # ±1-step neighbors
        for step in (-1, 1):
            neighbor_index = index + step
            if not 0 <= neighbor_index < len(grid):
                continue
            neighbor_position = dict(position)
            neighbor_position[axis] = grid[neighbor_index]
            neighbor_id = index_of.get(tuple(sorted(neighbor_position.items())))
            if neighbor_id is None:
                continue
            neighbor_value = metric_by_child.get(neighbor_id)
            if neighbor_value is None:
                continue
            checks.append(
                NeighborCheck(
                    child_id=child_id,
                    neighbor_id=neighbor_id,
                    axis=axis,
                    metric=metric_name,
                    child_value=float(child_value),
                    neighbor_value=float(neighbor_value),
                    degradation=float(child_value - neighbor_value),
                )
            )
        # plateau width along this axis: contiguous run around the child whose
        # metric stays within the warning band (or is better)
        band = degradation_warning_r if degradation_warning_r is not None else 0.0
        width = 1
        for direction in (-1, 1):
            cursor = index + direction
            while 0 <= cursor < len(grid):
                neighbor_position = dict(position)
                neighbor_position[axis] = grid[cursor]
                neighbor_id = index_of.get(tuple(sorted(neighbor_position.items())))
                neighbor_value = (
                    metric_by_child.get(neighbor_id) if neighbor_id else None
                )
                if neighbor_value is None or child_value - neighbor_value > band:
                    break
                width += 1
                cursor += direction
        per_axis_plateau[axis] = width

    worst = max((check.degradation for check in checks), default=None)
    knife_edge = False
    if (
        degradation_warning_r is not None
        and worst is not None
        and worst > degradation_warning_r
    ):
        knife_edge = True
        warnings.append(
            f"worst ±1-step neighbor degrades {metric_name} by {worst:g}R "
            f"(> {degradation_warning_r:g}R warning band)"
        )
    if per_axis_plateau and min(per_axis_plateau.values()) <= 1 and checks:
        knife_edge = True
        warnings.append("plateau width 1 on at least one searched axis")
    return RobustnessReport(
        child_id=child_id,
        neighbor_checks=tuple(checks),
        worst_neighbor_degradation_r=worst,
        plateau_width=min(per_axis_plateau.values()) if per_axis_plateau else None,
        knife_edge=knife_edge,
        knife_edge_warnings=tuple(warnings),
        per_axis_plateau=per_axis_plateau,
    )
